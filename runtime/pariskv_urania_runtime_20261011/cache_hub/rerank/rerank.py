
"""
4-bit RaBitQ Rerank 参考实现

用于 CUDA 加速的精排模块：
1. Coarse filter (topk 筛选碰撞计数最高的候选)
2. Gather 候选的 4-bit codes 和 weights
3. Unpack 4-bit → sign + magnitude
4. 计算近似内积得分
5. Rerank topk

longbench v2, 128k输入, 单层解码测试的时间 (kv_len≈128K, candidates≈13K, final_topk=100):
┌────────────────────────────────────────────────────────────────────────┐
│ get_candidate_cache_with_rabitq_rerank [1.147 ms] 总时间                │
├────────────────────────────────────────────────────────────────────────┤
│ Step 0: rerank_0_coarse_filter     │ 241.6 µs │ 21.1% │ topk 粗筛       │
│ Step 1: rerank_1_gather            │  89.5 µs │  7.8% │ gather codes   │
│ Step 2: rerank_2_unpack_reconstruct│ 383.5 µs │ 33.5% │ 解包+查表重构    │
│ Step 3: rerank_3_score             │ 338.2 µs │ 29.5% │ 内积+加权求和    │
│ Step 4: rerank_4_topk_gather       │  90.1 µs │  7.9% │ topk 精排       │
└────────────────────────────────────────────────────────────────────────┘

优化目标：将 Step 1-3 融合成一个 CUDA kernel,减少 memory bandwidth
"""

import math
import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch
import numpy as np
import triton


@lru_cache()
def _prepare_for_load() -> str:
    """获取当前模块的目录路径，用于加载 CUDA kernel"""
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


# 测试数据生成
def generate_test_input(
    bs: int = 1,
    kv_heads: int = 8,
    B: int = 16,           # 子空间数量 (head_dim / m)
    m: int = 8,           # 每个子空间的维度
    kv_len: int = 128*1024,   # retrieval zone 长度
    candidate_ratio: float = 0.1,  # 候选集比例
    final_topk: int = 100,         # 最终返回的 topk
    seed: int = 42
):
    """
    生成测试输入数据
    
    Returns:
        cache_cnt: [bs, kv_heads, kv_len] 碰撞计数 (int32)
        key_4bit: [bs, kv_heads, B, kv_len, m//2] 4-bit packed codes (uint8)
        key_block_weight: [bs, kv_heads, B, kv_len] 融合权重 (bf16)
        q_blocks: [bs, kv_heads, B, m] query 的 SRHT 分块结果 (bf16)
        q_norm: [bs, kv_heads, 1] query 模长 (bf16)
        mag_centers: [8] 幅值重构中心 (bf16)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda:0')
    
    # 碰撞计数：模拟碰撞结果，范围 [0, B]
    cache_cnt = torch.randint(0, B + 1, (bs, kv_heads, kv_len), 
                              dtype=torch.int32, device=device)
    
    # 4-bit packed codes: 每个 byte 存 2 个 4-bit 值
    # 4-bit = 1-bit sign + 3-bit magnitude
    # 格式: [sign(1bit) | magnitude(3bit)]
    # 
    # 生成真实分布的 4-bit 数据：
    #   sign ∈ {0, 1}  (1-bit)
    #   magnitude ∈ {0, 1, ..., 7}  (3-bit, 对应 _mag_centers 的 8 个 level)
    sign = torch.randint(0, 2, (bs, kv_heads, B, kv_len, m), device=device)  # [bs, kv, B, L, m]
    mag = torch.randint(0, 8, (bs, kv_heads, B, kv_len, m), device=device)   # [bs, kv, B, L, m]
    v4bit = (sign << 3) | mag  # [bs, kv, B, L, m], 每个元素是 4-bit 值 (0-15)
    
    # 打包成 bytes: 每 2 个 4-bit 值打包成 1 个 uint8
    # v_even = v4bit[..., 0::2], v_odd = v4bit[..., 1::2]
    # packed = (v_odd << 4) | v_even
    v_even = v4bit[..., 0::2]  # [bs, kv, B, L, m//2]
    v_odd = v4bit[..., 1::2]   # [bs, kv, B, L, m//2]
    key_4bit = ((v_odd << 4) | v_even).to(torch.uint8)  # [bs, kv, B, L, m//2]
    
    # Block weights: (r/α) * ||k||，已融合 key norm
    key_block_weight = torch.randn(bs, kv_heads, B, kv_len, 
                                   dtype=torch.bfloat16, device=device).abs()
    
    # Query blocks: SRHT 变换后的分块结果
    q_blocks = torch.randn(bs, kv_heads, B, m, 
                           dtype=torch.bfloat16, device=device)
    
    # Query norm: ||q||
    q_norm = torch.randn(bs, kv_heads, 1, 
                         dtype=torch.bfloat16, device=device).abs() + 0.1
    
    # 幅值重构中心 (Lloyd-Max 量化)
    mag_centers = torch.tensor(
        [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
        dtype=torch.bfloat16, device=device
    )
    
    return {
        'cache_cnt': cache_cnt,
        'key_4bit': key_4bit,
        'key_block_weight': key_block_weight,
        'q_blocks': q_blocks,
        'q_norm': q_norm,
        'mag_centers': mag_centers,
        'candidate_ratio': candidate_ratio,
        'final_topk': final_topk,
        'B': B,
        'm': m,
    }


# ============================================================================
# 参考实现 (PyTorch)
# ============================================================================


def _coarse_filter(cache_cnt, candidate_ratio, final_topk):
    """
    碰撞粗筛：基于碰撞阈值选出所有 >= 阈值的 key（公共逻辑）
    Args:
        cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
        candidate_ratio: 候选集比例
        final_topk: 最终返回的 topk 数量
    Returns:
        coarse_indices: [bs, kv_heads, max_candidates] 粗筛后的候选索引
        max_candidates: int - 实际候选数量
    """
    kv_len = cache_cnt.shape[-1]
    candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
    candidate_len = min(candidate_len, kv_len)
    candidate_len = max(candidate_len, final_topk)
    
    vals, idx = torch.topk(cache_cnt, k=candidate_len, dim=-1)
    thr = vals[..., -1]
    
    thr_count = (cache_cnt == thr.unsqueeze(-1)).sum(dim=-1)
    gt_count = (cache_cnt > thr.unsqueeze(-1)).sum(dim=-1)
    ge_count = gt_count + thr_count
    max_candidates = min(int(ge_count.max().item()), kv_len)
    
    if max_candidates > candidate_len:
        _, coarse_indices = torch.topk(cache_cnt, k=max_candidates, dim=-1)
    else:
        coarse_indices = idx
        max_candidates = candidate_len
    
    return coarse_indices, max_candidates


def _unpack_4bit_codes(packed: torch.Tensor, m: int = 8) -> tuple:
    """
    从 4-bit packed 解出符号和幅值。
    
    Args:
        packed: [..., m//2] uint8 (m=8 时为 4 bytes)
    Returns:
        s_sign: [..., m] int8, ±1
        t_codes: [..., m] int64, 0-7 (用于查表 mag_centers)
    """
    # 解包 2 个 4-bit 值
    v_even = packed & 0x0F           # [..., m//2], 低 4 位
    v_odd = (packed >> 4) & 0x0F     # [..., m//2], 高 4 位
    # 交错合并: [v0, v1, v2, v3, ...] from [even0, even1, ...] and [odd0, odd1, ...]
    v4bit = torch.stack([v_even, v_odd], dim=-1).view(*packed.shape[:-1], m)  # [..., m]
    # 拆分符号和幅值
    sign_bits = (v4bit >> 3) & 1     # [..., m], 0/1
    s_sign = (2 * sign_bits - 1).to(torch.int8)  # 0->-1, 1->+1
    t_codes = (v4bit & 0x7).to(torch.int64)      # [..., m], 0-7
    return s_sign, t_codes


def _coarse_filter_truncate(cache_cnt: torch.Tensor, candidate_ratio: float) -> tuple:
    """
    简化版碰撞粗筛：直接按碰撞计数从高到低排序，取前 candidate_len 个
    
    Args:
        cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
        candidate_ratio: 候选集比例
    Returns:
        coarse_indices: [bs, kv_heads, candidate_len] 粗筛后的候选索引
        candidate_len: int - 候选数量
    """
    kv_len = cache_cnt.shape[-1]
    candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
    
    _, coarse_indices = torch.topk(cache_cnt, k=candidate_len, dim=-1)
    return coarse_indices, candidate_len


def rerank_rabitq_ref(
    cache_cnt: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    candidate_ratio: float,
    final_topk: int,
    B: int,
    m: int,
) -> torch.Tensor:
    """
    4-bit RaBitQ 近似内积精排（参考实现）
    
    Args:
        cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
        key_4bit: [bs, kv_heads, B, kv_len, m//2] 4-bit packed codes
        key_block_weight: [bs, kv_heads, B, kv_len] bf16, 已融合 (r/α) * ||k||
        q_blocks: [bs, kv_heads, B, m] query 的 SRHT 分块结果
        q_norm: [bs, kv_heads, 1] query 的模长 ||q||
        mag_centers: [8] 幅值重构中心
        candidate_ratio: 候选集比例
        final_topk: 最终返回的 topk
        B: 子空间数量
        m: 每个子空间的维度
    
    Returns:
        final_indices: [bs, kv_heads, final_topk] 精排后的索引
    """
    # ===== Step 0: Coarse filter =====
    coarse_indices, max_candidates = _coarse_filter(cache_cnt, candidate_ratio, final_topk)
    
    if max_candidates <= final_topk:
        return coarse_indices[:, :, :final_topk]
    
    # ===== Step 1: Gather candidates' 4-bit codes and weights =====
    # idx_code: [bs, kv, B, C] for gathering weights
    idx_code = coarse_indices.unsqueeze(2).expand(-1, -1, B, -1)
    cand_weight = torch.gather(key_block_weight, dim=3, index=idx_code)  # [bs,kv,B,C] bf16
    # Gather 4-bit packed: [bs, kv, B, C, m//2]
    idx_4bit = coarse_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, B, -1, key_4bit.shape[-1])
    cand_4bit = torch.gather(key_4bit, dim=3, index=idx_4bit)  # [bs,kv,B,C,m//2]
    
    
    # ===== Step 2: Unpack 4-bit codes → sign + magnitude =====
    orig_shape = cand_4bit.shape[:-1]  # [bs, kv, B, C]
    s_sign, t_codes = _unpack_4bit_codes(cand_4bit.view(-1, cand_4bit.shape[-1]), m=m)
    s_sign = s_sign.view(*orig_shape, m)    # [bs, kv, B, C, m] int8
    t_codes = t_codes.view(*orig_shape, m)  # [bs, kv, B, C, m] int64
    # Lookup magnitude centers
    a_vals = mag_centers[t_codes]  # [bs, kv, B, C, m] bf16
    

    # ===== Step 3: Compute approximate scores =====
    # v^(4b) = sign * magnitude
    v4b = a_vals * s_sign.to(a_vals.dtype)  # [bs, kv, B, C, m] bf16
    # Per-block inner product: <q_b, v^(4b)>
    q_exp = q_blocks.unsqueeze(3)  # [bs, kv, B, 1, m]
    per_block_scores = (q_exp * v4b).sum(dim=-1)  # [bs, kv, B, C] bf16
    # Weighted sum over blocks
    cosine_proxy = (per_block_scores * cand_weight).sum(dim=2)  # [bs, kv, C] bf16
    # Scale by query norm
    approx_scores = cosine_proxy * q_norm  # [bs, kv, C] bf16
    
    
    # ===== Step 4: Rerank (topk + gather) =====
    _, rerank_indices = torch.topk(approx_scores, k=final_topk, dim=-1)
    final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
    return final_indices



@lru_cache()
def load_kernel_module(
    path: str | Iterable[str],
    name: str,
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    if isinstance(path, str):
        path = (path,)

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    module_path = os.path.join(build_dir, f"{name}.so")
    source_paths = [f"{abs_path}/{p}" for p in path]

    # Fast path: reuse an already-built extension when it is newer than sources.
    # This avoids PyTorch JIT's file lock and ninja checks on every process start.
    if os.path.exists(module_path):
        module_mtime = os.path.getmtime(module_path)
        latest_source_mtime = max(os.path.getmtime(src) for src in source_paths)
        if module_mtime >= latest_source_mtime:
            spec = importlib.util.spec_from_file_location(name, module_path)
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                return module

    return load(
        name=name,
        sources=source_paths,
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++20"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++20"],
        extra_ldflags=list(ldflags or []) or None,
        build_directory=build_dir,
        verbose=True,
    )



def _validate_partial_rerank_inputs(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
) -> tuple[int, int, int]:
    bs, kv_heads, B_dim, kv_len, _ = key_4bit.shape
    assert B_dim == B, f"B mismatch: key_4bit has {B_dim}, got B={B}"
    assert m == 8, f"m must be 8, got {m}"
    assert coarse_indices.dtype == torch.int64, f"coarse_indices must be int64, got {coarse_indices.dtype}"
    assert coarse_indices.is_contiguous()
    assert key_4bit.stride(-1) == 1
    assert key_block_weight.stride(-1) == 1
    assert q_blocks.is_contiguous()
    assert q_norm.is_contiguous()
    assert mag_centers.is_contiguous()
    assert coarse_indices.is_cuda
    assert key_4bit.is_cuda
    assert key_block_weight.is_cuda
    assert q_blocks.is_cuda
    assert q_norm.is_cuda
    assert mag_centers.is_cuda
    return bs, kv_heads, kv_len


def partial_rerank_cuda(     
    coarse_indices: torch.Tensor,     
    key_4bit: torch.Tensor,     
    key_block_weight: torch.Tensor,     
    q_blocks: torch.Tensor,     
    q_norm: torch.Tensor,     
    mag_centers: torch.Tensor,     
    B: int,     
    m: int,
    kernel=None,  # 可选：预加载的 CUDA kernel，避免每次查找 
) -> tuple:
    """
    Returns:
        approx_scores: [bs, kv_heads, candidate_len] - 近似分数
        v4b: [bs, kv_heads, B, candidate_len, m] - unpack 后的 4bit key 向量
    """
    if q_norm.dtype != torch.bfloat16:
        q_norm = q_norm.to(torch.bfloat16).contiguous()
    bs, kv_heads, _ = _validate_partial_rerank_inputs(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B,
        m,
    )
    candidate_len = coarse_indices.shape[-1]
    approx_scores = torch.zeros(bs, kv_heads, candidate_len, dtype=torch.bfloat16, device=coarse_indices.device)
    # 分配 v4b 输出 tensor: [bs, kv_heads, B, candidate_len, m]
    v4b = torch.zeros(bs, kv_heads, B, candidate_len, m, dtype=torch.bfloat16, device=coarse_indices.device)
     
    # 使用预加载的 kernel 或懒加载
    if kernel is None:
        kernel = load_kernel_module("rerank.cu", "rerank")
    kernel.partial_rerank(         
        coarse_indices,         
        key_4bit,         
        key_block_weight,         
        q_blocks,         
        q_norm,         
        mag_centers,         
        B, m,         
        approx_scores,
        v4b
    )     
    return approx_scores, v4b  


def partial_rerank_scores_cuda(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
    kernel=None,
) -> torch.Tensor:
    """Scores-only RaBitQ rerank path.

    This keeps the same score math as partial_rerank_cuda but does not allocate
    or write the unpacked v4b debug tensor.
    """
    if q_norm.dtype != torch.bfloat16:
        q_norm = q_norm.to(torch.bfloat16).contiguous()
    bs, kv_heads, _ = _validate_partial_rerank_inputs(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B,
        m,
    )
    candidate_len = coarse_indices.shape[-1]
    approx_scores = torch.empty(
        bs,
        kv_heads,
        candidate_len,
        dtype=torch.bfloat16,
        device=coarse_indices.device,
    )

    if kernel is None:
        kernel = load_kernel_module("rerank.cu", "rerank")
    kernel.partial_rerank_scores(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B,
        m,
        approx_scores,
    )
    return approx_scores




def partial_rerank_topk_scores_cuda(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
    topk_scores: torch.Tensor,
    topk_indices: torch.Tensor,
    topk: int,
    kernel=None,
) -> None:
    """Fused per-chunk top-k RaBitQ rerank (chunk_size=32, bitonic sort in warp).

    Writes results into pre-allocated topk_scores and topk_indices tensors.
    """
    if q_norm.dtype != torch.bfloat16:
        q_norm = q_norm.to(torch.bfloat16).contiguous()
    _validate_partial_rerank_inputs(
        coarse_indices, key_4bit, key_block_weight,
        q_blocks, q_norm, mag_centers, B, m,
    )
    if kernel is None:
        kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    kernel.partial_rerank_topk_scores(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B, m,
        topk_scores,
        topk_indices,
        topk,
    )


def partial_rerank_dense_sparse_cuda(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
    n_dense: int,
    topk: int,
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
    kernel=None,
) -> None:
    """Fused dense/sparse RaBitQ rerank.

    Input coarse_indices must be sorted: dense chunks first (n_dense),
    then sparse chunks. Output layout:
      [dense: n_dense*16 scores/indices | sparse: n_sparse*topk scores/indices]
    """
    if q_norm.dtype != torch.bfloat16:
        q_norm = q_norm.to(torch.bfloat16).contiguous()
    _validate_partial_rerank_inputs(
        coarse_indices, key_4bit, key_block_weight,
        q_blocks, q_norm, mag_centers, B, m,
    )
    if kernel is None:
        kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    kernel.partial_rerank_dense_sparse(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B, m,
        n_dense,
        topk,
        out_scores,
        out_indices,
    )


def partial_rerank_chunk_dense_sparse_cuda(
    chunk_ids: torch.Tensor,
    dense_mask: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
    dense_topk: int,
    sparse_topk: int,
    out_scores: torch.Tensor,
    out_indices: torch.Tensor,
    kernel=None,
) -> None:
    """Chunk-ID based dense/sparse RaBitQ rerank.

    Accepts chunk IDs directly (no token-index expansion needed).
    The kernel computes token_idx = chunk_id * 16 + lane internally.
    Dense chunks (dense_mask=True) keep top dense_topk after bitonic sort;
    sparse chunks keep top sparse_topk.
    Output: uniform [bs, kv_heads, nk * 16], unused slots = -inf.

    NOTE: both rerank_topk.cu and rerank_topk_c8.cu now accept fp32 q_norm.
    """
    if kernel is None:
        kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    if q_norm.dtype != torch.float32:
        q_norm = q_norm.float().contiguous()
    kernel.partial_rerank_chunk_dense_sparse(
        chunk_ids,
        dense_mask,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B, m,
        dense_topk,
        sparse_topk,
        out_scores,
        out_indices,
    )


def dense_mask_from_topk_cuda(
    positions: torch.Tensor,
    mask: torch.Tensor,
    kernel=None,
) -> None:
    """Fused replacement for ``mask.fill_(False); mask.scatter_(2, positions, True)``.

    Single CUDA kernel that:
      1. zeros the whole ``mask`` row for each (b, h),
      2. sets ``mask[b, h, positions[b, h, j]] = True`` for j ∈ [0, n_dense).

    Output is bit-identical to the two-step PyTorch version. Saves one kernel
    launch overhead per call (~5 μs) and the implicit inter-kernel barrier,
    totalling ~20 ms across a full decode pass.

    Args:
        positions: [bs, kv_heads, n_dense] int64 — top-k indices into the nk
            dimension. Values outside [0, nk) are silently ignored.
        mask:      [bs, kv_heads, nk] bool — overwritten in place.
    """
    if kernel is None:
        kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    kernel.mask_from_topk(positions.contiguous(), mask)


def partial_chunk_density_scores_cuda(
    chunk_ids: torch.Tensor,
    chunk_centroids: torch.Tensor,
    raw_q: torch.Tensor,
    out_scores: torch.Tensor,
    kernel=None,
) -> None:
    """Compute density scores: centroid[chunk_id] @ raw_q (fused CUDA kernel)."""
    if chunk_centroids.dtype != torch.bfloat16:
        chunk_centroids = chunk_centroids.to(torch.bfloat16)
    if raw_q.dtype != torch.bfloat16:
        raw_q = raw_q.to(torch.bfloat16)
    if out_scores.dtype != torch.float32:
        raise TypeError(f"out_scores must be float32, got {out_scores.dtype}")
    if kernel is None:
        kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    kernel.partial_chunk_density_scores(
        chunk_ids,
        chunk_centroids,
        raw_q,
        out_scores,
    )


def coarse_2bit_score_cuda(
    key_4bit: torch.Tensor,
    q_signs: torch.Tensor,
    q_mags: torch.Tensor,
    B: int,
    m: int,
    kernel=None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Fused 2-bit coarse score：在一个 CUDA kernel 内同时完成
      1. 从 4-bit packed key 抽 sign-bit / magnitude-high-bit 两个 bit-plane
      2. 与 query 端 packed sign / mag-high 做 6 次 popcount + 求和
      3. 输出 score = (D + N(m_q)+N(m_k)+N(m_q·m_k)) - 2·(N(xs)+...)

    替代 PyTorch 版的 `_extract_signs_and_mag_high_from_4bit` + `2stage_2bit_score`
    两段，整体 ~150x 加速（Stage-1 主要瓶颈）。

    Args:
        key_4bit:   [bs, kv_heads, B, kv_len, m//2] uint8 packed (sign+magnitude)
        q_signs:    [bs, kv_heads, B] uint8，bit_i = sign(dim_i)（_encode_query_2bit_packed 输出）
        q_mags:     [bs, kv_heads, B] uint8，bit_i = (|q_dim_i| >= τ)
        B, m:       sub-space 数量与每 sub-space 维度，目前仅支持 (B=16, m=8)
        kernel:     可选预加载的 CUDA module；不传则懒加载并缓存
        out:        可选的 [bs, kv, kv_len] int32 输出 buffer，用于 in-place 复用

    Returns:
        score: [bs, kv_heads, kv_len] int32，越大越相似（值域 [-4D, +4D]，D = m·B）
    """
    bs, kv_heads, B_dim, kv_len, half_m = key_4bit.shape
    assert B_dim == B and 2 * half_m == m, \
        f"shape mismatch: B_dim={B_dim} expected {B}, half_m={half_m} expected {m//2}"
    assert key_4bit.dtype == torch.uint8
    assert q_signs.dtype == torch.uint8
    assert q_mags.dtype == torch.uint8
    assert key_4bit.stride(-1) == 1, "key_4bit must be contiguous in last dim"

    # q_signs / q_mags 必须 contiguous（kernel 按 [bs*kv_heads, B] 线性读）
    q_signs_c = q_signs.contiguous()
    q_mags_c  = q_mags.contiguous()

    if out is None:
        out = torch.empty(bs, kv_heads, kv_len, dtype=torch.int32, device=key_4bit.device)
    else:
        assert out.shape == (bs, kv_heads, kv_len)
        assert out.dtype == torch.int32
        assert out.is_contiguous()

    if kernel is None:
        kernel = load_kernel_module("coarse_2bit_score.cu", "coarse_2bit_score")

    kernel.coarse_2bit_score(
        key_4bit, q_signs_c, q_mags_c, B, m, out
    )
    return out


def rerank_rabitq_fused(     
    cache_cnt: torch.Tensor,     
    key_4bit: torch.Tensor,     
    key_block_weight: torch.Tensor,     
    q_blocks: torch.Tensor,     
    q_norm: torch.Tensor,     
    mag_centers: torch.Tensor,     
    candidate_ratio: float,     
    final_topk: int,     
    B: int,     
    m: int, 
) -> tuple:     
    # ===== Step 0: Coarse filter =====     
    coarse_indices, max_candidates = _coarse_filter_truncate(cache_cnt, candidate_ratio)          
    if max_candidates <= final_topk:         
        return coarse_indices[:, :, :final_topk], None, None
    
    approx_scores, v4b = partial_rerank_cuda(         
        coarse_indices,         
        key_4bit,         
        key_block_weight,         
        q_blocks,         
        q_norm,         
        mag_centers,         
        B,         
        m     
    )     
    # ===== Step 4: Rerank (topk + gather) =====    
    _, rerank_indices = torch.topk(approx_scores, k=final_topk, dim=-1)     
    final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)     
    return final_indices, approx_scores, v4b   
