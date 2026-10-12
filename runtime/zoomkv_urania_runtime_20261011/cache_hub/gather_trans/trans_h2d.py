from typing import Any

import torch
import numpy as np
import triton

import os
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

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
    module_path = os.path.join(build_dir, f"{name}.so")
    source_paths = [f"{abs_path}/{p}" for p in path]

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


def h2d_gather_kv(
    src_k_caches, # bs, heads, kv_len, dim
    src_v_caches, # bs, heads, kv_len, dim
    selected_indices, # bs, heads, max_topk
    dst_k_caches, # bs, heads, topk, dim
    dst_v_caches, # bs, heads, topk, dim
) -> torch.Tensor:
    '''
    '''
    # check shape
    bs, heads, kv_len, dim = src_k_caches.shape
    assert src_k_caches.shape == src_v_caches.shape
    _, _, topk = selected_indices.shape
    assert dst_k_caches.shape == dst_v_caches.shape
    assert dst_k_caches.shape[-2] >= topk
    assert dim == 128
    # check device
    assert src_k_caches.is_cpu
    assert src_v_caches.is_cpu
    assert src_k_caches.is_pinned()
    assert src_v_caches.is_pinned()
    assert dst_k_caches.is_cuda
    assert dst_v_caches.is_cuda
    # check type 
    assert src_k_caches.dtype == src_v_caches.dtype
    assert dst_k_caches.dtype == dst_v_caches.dtype
    assert src_k_caches.dtype == dst_k_caches.dtype
    assert src_k_caches.dtype == torch.bfloat16
    # check continguous
    assert src_k_caches.is_contiguous()
    assert dst_k_caches.is_contiguous()
    assert src_v_caches.is_contiguous()
    assert dst_v_caches.is_contiguous()
    # run
    load_kernel_module("trans_h2d.cu", "h2d_gather_kv").h2d_gather_kv(
        src_k_caches,
        src_v_caches,
        selected_indices,
        dst_k_caches,
        dst_v_caches,
    )


def h2d_gather_kv_packed(
    src_k_caches,        # CPU pinned [bs, kv_heads, src_len, dim]   bf16
    src_v_caches,        # CPU pinned [bs, kv_heads, src_len, dim]   bf16
    dst_k_packed,        # GPU         [total_k, dim]                bf16
    dst_v_packed,        # GPU         [total_k, dim]                bf16
    indices_packed,      # GPU         [total_topk]                  int64
    cu_seqlens_topk,     # GPU         [V + 1]                       int32
    cu_seqlens_k,        # GPU         [V + 1]                       int32
    sl_end: int,         # sink_size + local_size
    global_k: int,       # max(adaptive_k); grid Y dim
):
    """Variable-K H2D gather that writes directly into the *final*
    flash_attn_varlen packed buffer.

    For each virtual batch ``vh = b * kv_heads + h`` the kernel writes rows
    ``[cu_seqlens_k[vh] + sl_end, cu_seqlens_k[vh + 1])`` of ``dst_*_packed``
    using KV indices from ``indices_packed[cu_seqlens_topk[vh]:cu_seqlens_topk[vh+1]]``.
    The first ``sl_end`` rows of every segment are reserved for sink+local
    KV that the caller fills separately on GPU; this kernel does not touch them.
    """
    bs, kv_heads, _src_len, dim = src_k_caches.shape
    V = bs * kv_heads
    assert src_k_caches.shape == src_v_caches.shape
    assert dim == 128
    # device
    assert src_k_caches.is_cpu and src_v_caches.is_cpu
    assert src_k_caches.is_pinned() and src_v_caches.is_pinned()
    assert dst_k_packed.is_cuda and dst_v_packed.is_cuda
    assert indices_packed.is_cuda
    assert cu_seqlens_topk.is_cuda and cu_seqlens_k.is_cuda
    # dtype
    assert src_k_caches.dtype == torch.bfloat16
    assert src_v_caches.dtype == torch.bfloat16
    assert dst_k_packed.dtype == torch.bfloat16
    assert dst_v_packed.dtype == torch.bfloat16
    assert indices_packed.dtype == torch.int64
    assert cu_seqlens_topk.dtype == torch.int32
    assert cu_seqlens_k.dtype == torch.int32
    # contiguity / shape
    assert src_k_caches.is_contiguous() and src_v_caches.is_contiguous()
    assert dst_k_packed.is_contiguous() and dst_v_packed.is_contiguous()
    assert dst_k_packed.dim() == 2 and dst_k_packed.shape[1] == dim
    assert dst_v_packed.shape == dst_k_packed.shape
    assert cu_seqlens_topk.shape[0] == V + 1
    assert cu_seqlens_k.shape[0] == V + 1
    assert indices_packed.dim() == 1

    load_kernel_module("trans_h2d.cu", "h2d_gather_kv").h2d_gather_kv_packed(
        src_k_caches,
        src_v_caches,
        dst_k_packed,
        dst_v_packed,
        indices_packed,
        cu_seqlens_topk,
        cu_seqlens_k,
        int(sl_end),
        int(global_k),
    )


if __name__ == "__main__":
    bs = 2
    heads = 8
    kv_len = 32 * 1024
    dim = 128
    topk = 100
    device = "cuda:0"
    max_topk = int(kv_len * 0.2)

    src_k_caches = torch.randn(bs, heads, kv_len, dim, dtype = torch.bfloat16, device = "cpu", pin_memory=True)
    src_v_caches = torch.randn(bs, heads, kv_len, dim, dtype = torch.bfloat16, device = "cpu", pin_memory=True)

    selected_indices = torch.randint(low=0, high=kv_len, size=(bs, heads, topk), dtype=torch.long).to(device)

    # accuracy
    def cuda():
        dst_k_caches = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        dst_v_caches = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        h2d_gather_kv(src_k_caches, src_v_caches, selected_indices, dst_k_caches, dst_v_caches)
        return dst_k_caches, dst_v_caches
    
    def ref():
        # Use different variable names to avoid shadowing
        src_k_cuda = src_k_caches.cuda()
        src_v_cuda = src_v_caches.cuda()
        dst_k_ref = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        dst_v_ref = torch.zeros(bs, heads, max_topk, dim, dtype = torch.bfloat16, device = device)
        for i in range(bs):
            for j in range(heads):
                dst_k_ref[i][j][:topk].copy_(src_k_cuda[i][j][selected_indices[i][j]])
                dst_v_ref[i][j][:topk].copy_(src_v_cuda[i][j][selected_indices[i][j]])
        return dst_k_ref, dst_v_ref

    dst_k_ref, dst_v_ref = ref()
    dst_k_cuda, dst_v_cuda = cuda()
    assert (dst_k_ref - dst_k_cuda).sum() == 0, "dst_k_ref and dst_k_cuda are not equal"
    assert (dst_v_ref - dst_v_cuda).sum() == 0, "dst_v_ref and dst_v_cuda are not equal"

    # ----------------------------------------------------------------------
    # h2d_gather_kv_packed correctness check (variable per-(b,h) K)
    # ----------------------------------------------------------------------
    sl_end = 5
    V = bs * heads
    rng = torch.Generator().manual_seed(0)
    adaptive_k = torch.randint(low=1, high=topk + 1, size=(V,), generator=rng, dtype=torch.int32)
    seqlens_k = adaptive_k + sl_end
    cu_seqlens_topk_cpu = torch.zeros(V + 1, dtype=torch.int32)
    torch.cumsum(adaptive_k, dim=0, out=cu_seqlens_topk_cpu[1:])
    cu_seqlens_k_cpu = torch.zeros(V + 1, dtype=torch.int32)
    torch.cumsum(seqlens_k, dim=0, out=cu_seqlens_k_cpu[1:])

    total_topk = int(cu_seqlens_topk_cpu[-1].item())
    total_k = int(cu_seqlens_k_cpu[-1].item())
    global_k = int(adaptive_k.max().item())

    indices_packed_cpu = torch.empty(total_topk, dtype=torch.int64)
    selected_indices_cpu = selected_indices.cpu()
    for vh in range(V):
        b_, h_ = vh // heads, vh % heads
        ak = int(adaptive_k[vh].item())
        st = int(cu_seqlens_topk_cpu[vh].item())
        indices_packed_cpu[st:st + ak] = selected_indices_cpu[b_, h_, :ak]

    indices_packed = indices_packed_cpu.to(device)
    cu_seqlens_topk = cu_seqlens_topk_cpu.to(device)
    cu_seqlens_k = cu_seqlens_k_cpu.to(device)
    dst_k_packed = torch.zeros(total_k, dim, dtype=torch.bfloat16, device=device)
    dst_v_packed = torch.zeros(total_k, dim, dtype=torch.bfloat16, device=device)

    h2d_gather_kv_packed(
        src_k_caches, src_v_caches,
        dst_k_packed, dst_v_packed,
        indices_packed, cu_seqlens_topk, cu_seqlens_k,
        sl_end, global_k,
    )

    src_k_cuda_ref = src_k_caches.cuda()
    src_v_cuda_ref = src_v_caches.cuda()
    ref_k = torch.zeros(total_k, dim, dtype=torch.bfloat16, device=device)
    ref_v = torch.zeros(total_k, dim, dtype=torch.bfloat16, device=device)
    for vh in range(V):
        b_, h_ = vh // heads, vh % heads
        ak = int(adaptive_k[vh].item())
        seg_dst_start = int(cu_seqlens_k_cpu[vh].item()) + sl_end
        seg_topk_start = int(cu_seqlens_topk_cpu[vh].item())
        idx = indices_packed[seg_topk_start:seg_topk_start + ak]
        ref_k[seg_dst_start:seg_dst_start + ak] = src_k_cuda_ref[b_, h_, idx]
        ref_v[seg_dst_start:seg_dst_start + ak] = src_v_cuda_ref[b_, h_, idx]

    assert (ref_k - dst_k_packed).abs().sum().item() == 0, "packed K mismatch"
    assert (ref_v - dst_v_packed).abs().sum().item() == 0, "packed V mismatch"
    print("[h2d_gather_kv_packed] OK", "total_k=", total_k, "total_topk=", total_topk)
