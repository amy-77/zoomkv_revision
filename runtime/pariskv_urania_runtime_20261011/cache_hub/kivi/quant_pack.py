"""KIVI 4-bit (per-quest-chunk per-channel) key cache packing.

Layout & invariants (production-aligned):

  K              : [bs, kv_heads, T, D]                  bf16
                   raw post-RoPE keys (same tensor that today is fed to
                   ``batch_encode`` in ``polar_cache_pure_rabitq_*``).
  chunk_min      : [bs, kv_heads, n_chunks, D]           bf16
  chunk_max      : [bs, kv_heads, n_chunks, D]           bf16
                   per-(quest-)chunk per-channel raw K min/max — maintained
                   by ``_compute_and_store_chunk_minmax`` independently of
                   any stage-2 estimator.  We re-use them as KIVI's
                   per-group ``mn`` / ``mx`` (one chunk == one KIVI group).

  packed         : [bs, kv_heads, T, D // FEAT_PER_INT]  int32
                   per-token, per-channel 4-bit codes packed along the
                   head-dim axis (``FEAT_PER_INT = 32 // 4 = 8``).
  scale          : [bs, kv_heads, n_chunks, D]           bf16
                   = (chunk_max - chunk_min) / 15  (4-bit levels = 15).

Storage per token (D = 128, chunk = 16):
  packed         : 4 bit × 128 dims / 8         = 64 B
  per-chunk min  : 2 B × 128 / 16               = 16 B
  per-chunk scale: 2 B × 128 / 16               = 16 B
                                              ≈ 96 B/token — identical to the
                                              current 4-bit RaBitQ pipeline
                                              (packed 64 B + block_weight 32 B).

For ``bits ∈ {2, 4, 8}`` the helpers below stay generic; production uses
``bits = 4``.
"""

from __future__ import annotations

from typing import Tuple

import torch


def _bits_constants(bits: int) -> Tuple[int, int]:
    """Return ``(levels, feat_per_int)`` for ``bits``-wide codes packed in int32."""
    if bits not in (2, 4, 8):
        raise ValueError(f"bits must be one of (2, 4, 8); got {bits}")
    if 32 % bits != 0:
        raise ValueError(f"bits ({bits}) must divide 32")
    levels = (1 << bits) - 1
    feat_per_int = 32 // bits
    return levels, feat_per_int


def chunk_minmax_to_scale_zero(
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Derive ``(scale, zero)`` from the cache's existing ``chunk_min``/``chunk_max``.

    Returns raw ``scale = (max - min) / levels`` (NOT clamped) so the
    reconstruction ``code * scale + min`` reproduces ``min`` exactly on flat
    chunks (where ``code`` rounds to 0).  Pass through ``_safe_div_scale``
    if you need a non-zero divisor for the quant direction.
    """
    levels, _ = _bits_constants(bits)
    scale = (chunk_max - chunk_min) / float(levels)
    return scale, chunk_min


def _expand_chunk_meta_to_tokens(
    meta: torch.Tensor,         # [bs, kv, n_chunks, D]
    group_size: int,
    T: int,
) -> torch.Tensor:
    """Broadcast a per-chunk tensor to per-token by repeat-interleave along T.

    Only used in pure-Python fallback paths; the production kernel will index
    by ``token // group_size`` directly to avoid this materialization.
    """
    bs, kv, n_chunks, D = meta.shape
    assert n_chunks * group_size >= T, (
        f"chunk meta covers {n_chunks * group_size} tokens, < T={T}"
    )
    expanded = meta.unsqueeze(3).expand(bs, kv, n_chunks, group_size, D)
    return expanded.reshape(bs, kv, n_chunks * group_size, D)[:, :, :T, :].contiguous()


def quantize_kcache_4bit(
    K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    """Quantize raw K into integer codes (NOT packed yet).

    All arithmetic runs in K's input dtype (typically bf16) — matching
    ``cache_hub.kivi_offline.eval_kivi_quant.kivi_simulate_dequant`` and
    what a fused bf16-only CUDA kernel will do.  Internally clamps the
    scale divisor to avoid /0 on flat chunks (KIVI uses ``1e-8`` clamp).

    K          : [bs, kv, T, D]                 K.dtype (bf16 in production)
    chunk_min  : [bs, kv, n_chunks, D]          K.dtype
    chunk_max  : [bs, kv, n_chunks, D]          K.dtype
    group_size : == quest_chunk_size (one chunk = one KIVI group)
    returns    : [bs, kv, T_trunc, D] int32     values in [0, 2^bits - 1]
    """
    bs, kv, T, D = K.shape
    n_chunks = T // group_size
    T_trunc = n_chunks * group_size
    levels, _ = _bits_constants(bits)
    # Match sim semantics exactly: scale = (max - min)/levels, then clamp(min=1e-8)
    # on the *divisor only* so the dequant reconstruction can still use the raw
    # scale (yielding mn when max==min).
    mn = chunk_min[..., :n_chunks, :]
    mx = chunk_max[..., :n_chunks, :]
    scale = (mx - mn) / float(levels)
    scale_safe = scale.clamp(min=1e-8)
    scale_t = _expand_chunk_meta_to_tokens(scale_safe, group_size, T_trunc)
    mn_t = _expand_chunk_meta_to_tokens(mn, group_size, T_trunc)
    K_use = K[..., :T_trunc, :]
    q = ((K_use - mn_t) / scale_t).clamp_(0.0, float(levels)).round_()
    return q.to(torch.int32)


def _pack_along_last_dim(codes_int32: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack ``bits``-wide non-negative integer codes along the last dim into int32.

    codes_int32: [..., D]  with values in [0, 2^bits - 1]
    returns    : [..., D // feat_per_int]  int32
    """
    _, feat_per_int = _bits_constants(bits)
    shape = codes_int32.shape
    D = shape[-1]
    assert D % feat_per_int == 0, (
        f"D={D} not divisible by feat_per_int={feat_per_int} (bits={bits})"
    )
    n_pack = D // feat_per_int
    view = codes_int32.view(*shape[:-1], n_pack, feat_per_int)
    shifts = torch.arange(feat_per_int, device=codes_int32.device,
                          dtype=torch.int32) * int(bits)
    packed = (view << shifts).sum(dim=-1, dtype=torch.int32)
    return packed.contiguous()


def pack_kcache_4bit(
    K: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode + pack K with KIVI semantics, re-using the cache's ``chunk_min``/``chunk_max``.

    Returns
    -------
    packed : [bs, kv, T_trunc, D // (32 // bits)] int32
    scale  : [bs, kv, n_chunks, D] same dtype as ``chunk_min``
              (kept as a separate tensor so callers can persist it independently
              of ``chunk_min`` if they wish; in production we keep both).
    """
    codes = quantize_kcache_4bit(K, chunk_min, chunk_max, group_size, bits=bits)
    packed = _pack_along_last_dim(codes, bits=bits)
    scale, _ = chunk_minmax_to_scale_zero(chunk_min, chunk_max, bits=bits)
    return packed, scale


def _unpack_along_last_dim(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Inverse of ``_pack_along_last_dim``; returns int32 codes."""
    _, feat_per_int = _bits_constants(bits)
    levels = (1 << bits) - 1
    shifts = torch.arange(feat_per_int, device=packed.device,
                          dtype=torch.int32) * int(bits)
    codes = (packed.unsqueeze(-1) >> shifts) & levels
    out_shape = packed.shape[:-1] + (packed.shape[-1] * feat_per_int,)
    return codes.view(*out_shape).contiguous()


def unpack_kcache_4bit(
    packed: torch.Tensor,
    chunk_min: torch.Tensor,
    chunk_max: torch.Tensor,
    group_size: int,
    bits: int = 4,
) -> torch.Tensor:
    """Dequantize packed → ``K_approx`` (bf16, same layout as the original K).

    Only used for offline sanity / Python qk_dot reference; the production
    CUDA kernel will dequant inline per dot-product micro-batch.
    """
    codes = _unpack_along_last_dim(packed, bits=bits)                          # int32 [bs, kv, T, D]
    bs, kv, T, D = codes.shape
    n_chunks = (T + group_size - 1) // group_size
    scale, mn = chunk_minmax_to_scale_zero(
        chunk_min[..., :n_chunks, :], chunk_max[..., :n_chunks, :], bits=bits,
    )
    scale_t = _expand_chunk_meta_to_tokens(scale, group_size, T)
    mn_t = _expand_chunk_meta_to_tokens(mn, group_size, T)
    K_approx = codes.to(scale_t.dtype) * scale_t + mn_t
    return K_approx.to(chunk_min.dtype)
