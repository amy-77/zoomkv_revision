"""PyTorch reference for the KIVI stage-2 q·k estimator.

This module mirrors the call surface of
``cache_hub.rerank.partial_rerank_chunk_dense_sparse_cuda`` so that
``polar_cache.compute()`` can swap the production RaBitQ kernel for a
KIVI-style one with no changes outside the kernel call itself.

End-to-end semantics (matches the CUDA kernel we will write next):

  Inputs::

      chunk_ids       : [bs, kv, nk] int64
      dense_mask      : [bs, kv, nk] bool
      packed_K        : [bs, kv, n_chunks, D // (32 // bits), group_size] int32
                        (chunk-major; matches kivi_qk_dot.cu A-1 layout)
      chunk_min       : [bs, kv, n_chunks, D] bf16   (== KIVI ``mn``)
      chunk_max       : [bs, kv, n_chunks, D] bf16   (used to derive scale)
      raw_q           : [bs, kv, D] bf16             (GQA-averaged query)
      group_size      : == quest_chunk_size
      dense_topk      : keep top-K tokens for chunks marked dense
                        (production default = group_size, i.e. all tokens)
      sparse_topk     : keep top-K tokens for chunks marked sparse
                        (production default = 8)

  Outputs (written in-place when ``out_scores``/``out_indices`` are passed)::

      out_scores  : [bs, kv, nk * group_size] bf16
      out_indices : [bs, kv, nk * group_size] int64
                    Global token IDs in K's retrieval-zone coordinates.
                    Slots beyond the per-chunk ``k_eff`` are filled with
                    ``-inf`` / ``0`` (consistent with the existing CUDA
                    contract).

This file is the *reference* implementation: correctness > speed.  The
CUDA kernel will fuse `gather → unpack → dequant → dot → per-chunk
top-K` into a single launch per (bs, kv, chunk).
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from cache_hub.kivi.quant_pack import (
    _unpack_along_last_dim,
    chunk_minmax_to_scale_zero,
)


# ---------------------------------------------------------------------------
# Per-token dequant of the chunks of interest only.
# ---------------------------------------------------------------------------

def dequant_kcache_4bit(
    packed: torch.Tensor,                                  # [bs, kv, n_chunks, n_pack, g] int32 (chunk-major)
    chunk_min: torch.Tensor,                               # [bs, kv, n_chunks, D]
    chunk_max: torch.Tensor,                               # [bs, kv, n_chunks, D]
    chunk_ids: torch.Tensor,                               # [bs, kv, nk] int64
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize only the tokens of the selected chunks.

    ``packed`` is in **chunk-major** layout
    ``[bs, kv, n_chunks, n_pack, group_size]`` int32 (matches the storage
    used by ``kivi_qk_dot.cu`` after the A-1 layout switch).

    Returns ``K_chunks`` with shape ``[bs, kv, nk, group_size, D]`` in
    ``chunk_min.dtype``.
    """
    assert packed.dim() == 5, (
        f"packed_K must be chunk-major [bs, kv, n_chunks, n_pack, g]; "
        f"got shape={tuple(packed.shape)}"
    )
    bs, kv, n_chunks_pkd, feat_per_int, pkd_group = packed.shape
    assert pkd_group == group_size, (
        f"packed_K inner dim ({pkd_group}) must equal group_size ({group_size})"
    )
    D = chunk_min.shape[-1]
    nk = chunk_ids.shape[-1]
    scale, mn = chunk_minmax_to_scale_zero(chunk_min, chunk_max, bits=bits)

    # Gather full chunks of packed codes by chunk_id, then unpack.  The
    # chunk-major layout makes this trivial: we just index axis-2 with the
    # selected chunk_ids.
    chunk_ids_c = chunk_ids.clamp(min=0, max=n_chunks_pkd - 1)                 # safety
    chunk_gather_idx = (
        chunk_ids_c
        .unsqueeze(-1)
        .unsqueeze(-1)
        .expand(bs, kv, nk, feat_per_int, group_size)
    )                                                                          # [bs, kv, nk, n_pack, g]
    packed_chunks = torch.gather(
        packed,
        2,
        chunk_gather_idx,
    )                                                                          # [bs, kv, nk, n_pack, g] int32
    # Convert to per-token packed view: [bs, kv, nk, g, n_pack].
    packed_gather = packed_chunks.transpose(-1, -2).contiguous()
    codes = _unpack_along_last_dim(packed_gather, bits=bits)                   # [bs, kv, nk, g, D] int32

    chunk_gather_min = chunk_ids.unsqueeze(-1).expand(bs, kv, nk, D)           # [bs, kv, nk, D]
    scale_c = torch.gather(scale, 2, chunk_gather_min).unsqueeze(3)            # [bs, kv, nk, 1, D]
    mn_c    = torch.gather(mn,    2, chunk_gather_min).unsqueeze(3)            # [bs, kv, nk, 1, D]
    K_chunks = codes.to(scale_c.dtype) * scale_c + mn_c                        # [bs, kv, nk, g, D]
    return K_chunks


# ---------------------------------------------------------------------------
# Partial chunk dense/sparse rerank (drop-in for partial_rerank_chunk_dense_sparse_cuda).
# ---------------------------------------------------------------------------

def partial_chunk_kivi_qk_ref(
    chunk_ids: torch.Tensor,                               # [bs, kv, nk]   int64
    dense_mask: torch.Tensor,                              # [bs, kv, nk]   bool
    packed_K: torch.Tensor,                                # [bs, kv, n_chunks, D//feat, g] int32
    chunk_min: torch.Tensor,                               # [bs, kv, n_chunks, D] bf16
    chunk_max: torch.Tensor,                               # [bs, kv, n_chunks, D] bf16
    raw_q: torch.Tensor,                                   # [bs, kv, D]    bf16
    *,
    group_size: int,
    dense_topk: int,
    sparse_topk: int,
    bits: int = 4,
    out_scores: Optional[torch.Tensor] = None,             # [bs, kv, nk*g] fp32
    out_indices: Optional[torch.Tensor] = None,            # [bs, kv, nk*g] int64
    neg_inf: float = -3.0e38,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """KIVI per-chunk dense/sparse rerank.

    Equivalent (up to bf16 rounding) to the production sequence
    ``partial_rerank_chunk_dense_sparse_cuda(... key_4bit, key_block_weight,
    q_blocks, q_norm, mag_centers, polar_B, polar_m, dense_topk, sparse_topk,
    out_scores, out_indices)`` but operating on KIVI per-chunk per-channel
    packed K plus the cache's already-maintained ``chunk_min``/``chunk_max``.
    """
    bs, kv, nk = chunk_ids.shape
    D = chunk_min.shape[-1]
    g = group_size

    K_chunks = dequant_kcache_4bit(
        packed_K, chunk_min, chunk_max, chunk_ids, group_size=g, bits=bits,
    )                                                                          # [bs, kv, nk, g, D]
    tok_scores = (K_chunks * raw_q.view(bs, kv, 1, 1, D)).sum(dim=-1)          # [bs, kv, nk, g]

    sorted_scores, sorted_pos = tok_scores.topk(g, dim=-1, largest=True)
    offs = torch.arange(g, device=chunk_ids.device, dtype=torch.int64)
    token_ids = chunk_ids.unsqueeze(-1) * g + offs                             # [bs, kv, nk, g]
    sorted_tok = torch.gather(token_ids, -1, sorted_pos)                       # [bs, kv, nk, g]

    k_eff = torch.where(
        dense_mask,
        torch.full((), int(dense_topk), device=chunk_ids.device, dtype=torch.int64),
        torch.full((), int(sparse_topk), device=chunk_ids.device, dtype=torch.int64),
    )                                                                          # [bs, kv, nk]
    rank = torch.arange(g, device=chunk_ids.device, dtype=torch.int64).view(1, 1, 1, g)
    keep = rank < k_eff.unsqueeze(-1)                                          # [bs, kv, nk, g]
    masked_scores = sorted_scores.masked_fill(~keep, neg_inf).to(
        out_scores.dtype if out_scores is not None else torch.float32,
    )
    masked_idx = sorted_tok.masked_fill(~keep, 0)

    flat_scores = masked_scores.reshape(bs, kv, nk * g)
    flat_idx = masked_idx.reshape(bs, kv, nk * g)

    if out_scores is None:
        out_scores = flat_scores.contiguous()
    else:
        out_scores.copy_(flat_scores)
    if out_indices is None:
        out_indices = flat_idx.contiguous()
    else:
        out_indices.copy_(flat_idx)
    return out_scores, out_indices
