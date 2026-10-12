"""Python wrapper for the KIVI partial-chunk dense/sparse rerank CUDA kernel.

This mirrors ``cache_hub.rerank.rerank.partial_rerank_chunk_dense_sparse_cuda``
in spirit (loads the .cu via ``torch.utils.cpp_extension.load``, caches the
handle), but the underlying kernel does on-the-fly KIVI dequant instead of
RaBitQ 1+3 reconstruction.

Output buffers are written in place — same contract as the production
``partial_rerank_chunk_dense_sparse_cuda``: shape ``[bs, kv_heads,
nk * 16]``, sparse-side unused slots get score ``-1e30``.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

# Re-use the same loader as cache_hub.rerank.rerank to share the build dir
# and respect the `_prepare_for_load` jiggery (FlashInfer header paths etc.)
from cache_hub.rerank.rerank import load_kernel_module


@lru_cache()
def _kivi_kernel() -> Any:
    return load_kernel_module("../kivi/kivi_qk_dot.cu", "kivi_qk_dot")


def partial_chunk_kivi_qk_dense_sparse_cuda(
    chunk_ids: torch.Tensor,                   # [bs, kv, nk] int64
    dense_mask: torch.Tensor,                  # [bs, kv, nk] bool
    packed_K: torch.Tensor,                    # [bs, kv, n_chunks, D//8, group_size] int32  (chunk-major)
    chunk_min: torch.Tensor,                   # [bs, kv, n_chunks_max, D] bf16
    chunk_max: torch.Tensor,                   # [bs, kv, n_chunks_max, D] bf16
    raw_q: torch.Tensor,                       # [bs, kv, D] bf16
    dense_topk: int,
    sparse_topk: int,
    group_size: int,                           # 8 or 16
    out_scores: torch.Tensor,                  # [bs, kv, nk*group_size] fp32 (pre-allocated)
    out_indices: torch.Tensor,                 # [bs, kv, nk*group_size] int64
    kernel=None,
) -> None:
    """Run the KIVI partial-chunk dense/sparse rerank CUDA kernel in place."""
    if kernel is None:
        kernel = _kivi_kernel()
    if chunk_ids.dtype != torch.int64:
        chunk_ids = chunk_ids.to(torch.int64)
    if dense_mask.dtype != torch.bool:
        dense_mask = dense_mask.bool()
    if packed_K.dtype != torch.int32:
        raise TypeError(f"packed_K must be int32; got {packed_K.dtype}")
    for name, t in (("chunk_min", chunk_min), ("chunk_max", chunk_max),
                    ("raw_q", raw_q)):
        if t.dtype != torch.bfloat16:
            raise TypeError(f"{name} must be bfloat16; got {t.dtype}")
    if out_scores.dtype != torch.float32:
        raise TypeError(f"out_scores must be float32; got {out_scores.dtype}")
    if out_indices.dtype != torch.int64:
        raise TypeError(f"out_indices must be int64; got {out_indices.dtype}")
    if group_size not in (8, 16):
        raise ValueError(f"group_size must be 8 or 16; got {group_size}")

    kernel.partial_chunk_kivi_qk_dense_sparse(
        chunk_ids, dense_mask, packed_K, chunk_min, chunk_max, raw_q,
        int(dense_topk), int(sparse_topk), int(group_size),
        out_scores, out_indices,
    )
