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
import torch
import numpy as np
import triton
import os
from functools import lru_cache
from typing import Any, Iterable


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
    device = torch.device('cuda:1')
    
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
        'cache_cnt': cache_cnt.cuda(),
        'key_4bit': key_4bit.cuda(),
        'key_block_weight': key_block_weight.cuda(),
        'q_blocks': q_blocks.cuda(),
        'q_norm': q_norm.cuda(),
        'mag_centers': mag_centers.cuda(),
        'candidate_ratio': candidate_ratio,
        'final_topk': final_topk,
        'B': B,
        'm': m,
    }


# ============================================================================
# 参考实现 (PyTorch)
# ============================================================================
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

def _coarse_filter(cache_cnt, candidate_ratio):
    """
    碰撞粗筛：基于碰撞阈值选出所有 >= 阈值的 key（公共逻辑）
    Args:
        cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
        candidate_ratio: 候选集比例
    Returns:
        coarse_indices: [bs, kv_heads, max_candidates] 粗筛后的候选索引
        max_candidates: int - 实际候选数量
    """
    kv_len = cache_cnt.shape[-1]
    candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
    candidate_len = min(candidate_len, kv_len)
    candidate_len = max(candidate_len, 100)
    
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
    coarse_indices, max_candidates = _coarse_filter_truncate(cache_cnt, candidate_ratio)
    
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
    v4b = a_vals * s_sign.to(a_vals.dtype)  # [bs, kv, B, C, m] bf16

    # 划分块的时候：需要考虑shape 为[bs, kv, B] 每个thread 来处理一个B


    # ===== Step 3: Compute approximate scores =====
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
    return final_indices, approx_scores



@lru_cache()
def _prepare_for_load() -> str:
    import os
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


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
    return load(
        name=name,
        sources=[f"{abs_path}/{p}" for p in path],
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++20"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++20"],
        extra_ldflags=list(ldflags or []) or None,
        build_directory=build_dir,
        verbose=True,
    )

def partial_rerank_cuda(
    coarse_indices: torch.Tensor,
    key_4bit: torch.Tensor,
    key_block_weight: torch.Tensor,
    q_blocks: torch.Tensor,
    q_norm: torch.Tensor,
    mag_centers: torch.Tensor,
    B: int,
    m: int,
) -> torch.Tensor:
    bs, kv_heads, B, kv_len, _ = key_4bit.shape
    candidate_len = coarse_indices.shape[-1]

    approx_scores = torch.zeros(bs, kv_heads, candidate_len, dtype=torch.bfloat16, device=coarse_indices.device)

    # contiguous
    assert coarse_indices.is_contiguous()
    assert key_4bit.is_contiguous()
    assert key_block_weight.is_contiguous()
    assert q_blocks.is_contiguous()
    assert q_norm.is_contiguous()
    assert mag_centers.is_contiguous()
    
    # device 
    assert coarse_indices.is_cuda
    assert key_4bit.is_cuda
    assert key_block_weight.is_cuda
    assert q_blocks.is_cuda
    assert q_norm.is_cuda
    assert mag_centers.is_cuda

    load_kernel_module("rerank.cu", "rerank").partial_rerank(
        coarse_indices,
        key_4bit,
        key_block_weight,
        q_blocks,
        q_norm,
        mag_centers,
        B, m,
        approx_scores
    )

    return approx_scores

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
    coarse_indices, max_candidates = _coarse_filter_truncate(cache_cnt, candidate_ratio)
    
    if max_candidates <= final_topk:
        return coarse_indices[:, :, :final_topk]

    approx_scores = partial_rerank_cuda(
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
    return final_indices, approx_scores


# 
# 测试入口
# ============================================================================
if __name__ == "__main__":
    print("4-bit RaBitQ Rerank Benchmark (Reference Implementation)")
    
    # 测试配置
    # test_configs = [
    #     # (kv_len, candidate_ratio, final_topk)
    #     (4096, 0.2, 100),
    #     (8192, 0.2, 100),
    #     (16384, 0.2, 100),
    #     (32768, 0.2, 100),
    #     (65536, 0.2, 100),
    #     (131072, 0.2, 100),  # 128K
    # ]
    # test_configs = [
    # (4096, 0.05, 100),    # 改成 0.05 = 5%
    # (8192, 0.05, 100),
    # (16384, 0.05, 100),
    # (32768, 0.05, 100),
    # (65536, 0.05, 100),
    # (131072, 0.05, 100),
    # ]
    
    test_configs = [
        # (kv_len, candidate_ratio, final_topk)
        (4096, 0.1, 100),
        (8192, 0.1, 100),
        (16384, 0.1, 100),
        (32768, 0.1, 100),
        (65536, 0.1, 100),
        (131072, 0.1, 100),  # 128K
    ]
    
    bs = 1
    kv_heads = 8
    B = 16
    m = 8
    
    print(f"\nConfig: bs={bs}, kv_heads={kv_heads}, B={B}, m={m}")
    print(f"{'kv_len':>10} {'candidates(10%)':>12} {'step0_ms':>10} {'step1-3_ms':>12} {'step4_ms':>10} {'total_ms':>10}")
    
    for kv_len, candidate_ratio, final_topk in test_configs:
        # 生成测试数据
        inputs = generate_test_input(
            bs=bs, kv_heads=kv_heads, B=B, m=m,
            kv_len=kv_len, candidate_ratio=candidate_ratio, final_topk=final_topk
        )
        
        candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
        
        # 预先计算 coarse_indices 用于后续步骤的 benchmark
        coarse_indices, max_candidates = _coarse_filter_truncate(inputs['cache_cnt'], candidate_ratio)
        
        # Step 0: Coarse filter benchmark
        def bench_step0():
            return _coarse_filter_truncate(inputs['cache_cnt'], candidate_ratio)
        
        # Step 1-3: partial_rerank_cuda benchmark
        def bench_step1_3():
            return partial_rerank_cuda(
                coarse_indices,
                inputs['key_4bit'],
                inputs['key_block_weight'],
                inputs['q_blocks'],
                inputs['q_norm'],
                inputs['mag_centers'],
                B, m
            )
        
        # 预先计算 approx_scores 用于 step4 benchmark
        approx_scores = bench_step1_3()
        
        # Step 4: topk + gather benchmark
        def bench_step4():
            _, rerank_indices = torch.topk(approx_scores, k=final_topk, dim=-1)
            final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
            return final_indices
        
        # 总体 benchmark
        def bench_total():
            return rerank_rabitq_fused(
                inputs['cache_cnt'], inputs['key_4bit'], inputs['key_block_weight'],
                inputs['q_blocks'], inputs['q_norm'], inputs['mag_centers'],
                candidate_ratio, final_topk, B, m
            )
        
        step0_ms = triton.testing.do_bench(bench_step0)
        step1_3_ms = triton.testing.do_bench(bench_step1_3)
        step4_ms = triton.testing.do_bench(bench_step4)
        total_ms = triton.testing.do_bench(bench_total)
        
        print(f"{kv_len:>10} {candidate_len:>12} {step0_ms:>10.3f} {step1_3_ms:>12.3f} {step4_ms:>10.3f} {total_ms:>10.3f}")
    
    print("-" * 90)
    print("\nStep 说明:")
    print("  Step 0:   Coarse filter (PyTorch topk 按碰撞计数筛选候选)")
    print("  Step 1-3: partial_rerank_cuda (4-bit RaBitQ 近似内积计算)")
    print("  Step 4:   Final topk + gather (PyTorch)")
    print("\nDone!")