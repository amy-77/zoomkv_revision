"""KIVI-style stage-2 estimator for polar_cache.

This package replaces the production 4-bit RaBitQ stage-2 pipeline
(SRHT + 1-bit sign + 3-bit magnitude per block + `partial_rerank_*`
kernels) with KIVI's per-(quest-)chunk per-channel min/max quantization
of raw K.

Modules
-------
- ``quant_pack`` -- encode raw post-RoPE K into int32-packed codes using
  the same per-(chunk, channel) ``min``/``max`` that
  ``polar_cache._compute_and_store_chunk_minmax`` already maintains.
- ``qk_dot``     -- PyTorch reference for the per-token stage-2 q·k
  estimator (drop-in replacement for
  ``rerank.partial_rerank_chunk_dense_sparse_cuda``).  Operates on
  packed K + chunk_min + chunk_max + raw_q (no SRHT, no codebook).

The CUDA kernel that mirrors ``qk_dot`` lives next to this package once
it's implemented; for now we ship the PyTorch reference so the whole
pipeline can be validated end-to-end before kernel work.
"""

from cache_hub.kivi.quant_pack import (  # noqa: F401
    pack_kcache_4bit,
    unpack_kcache_4bit,
    chunk_minmax_to_scale_zero,
)
from cache_hub.kivi.qk_dot import (  # noqa: F401
    partial_chunk_kivi_qk_ref,
    dequant_kcache_4bit,
)
