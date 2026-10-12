"""Python wrapper for the fused float-topk + gather + skipped-concat kernel.

Companion to ``float_topk_ops.float_topk_3d``: replaces the 3-launch chain at
the tail of ``_cds_dense_sparse_select`` with a single CUDA launch that writes
into a caller-provided buffer (CUDA-Graph friendly).

Caller contract:
    out[:, :, :k]            ← lut[:, :, argtop_k(scores)]
    out[:, :, k:k+sk_count]  ← skipped[:, :, :sk_count]

``out`` is pre-allocated (size (bs, kv, k + max_sk)).  Caller slices to the
useful region ``out[:, :, :k + sk_count]`` after the kernel returns.

Shapes / dtypes:
    scores  : [bs, kv, L]    fp32
    lut     : [bs, kv, L]    int64
    skipped : [bs, kv, sk]   int64   (sk may be 0; pass empty slice)
    out     : [bs, kv, ≥k+sk] int64  (contiguous)

Returns the input ``out`` tensor (for chaining); no allocation.
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable

import torch


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_fused_topk_module(
    path: str = "float_topk_gather_concat.cu",
    name: str = "float_topk_gather_concat_ext",
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    source_paths = [f"{abs_path}/{path}"]
    module_path = os.path.join(build_dir, f"{name}.so")

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
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++20", "--use_fast_math"],
        build_directory=build_dir,
        verbose=True,
    )


def float_topk_gather_concat_3d(
    scores: torch.Tensor,
    lut: torch.Tensor,
    skipped: torch.Tensor,
    out: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """Fused top-k + indirect gather + skipped-tail concat.

    Args:
        scores  : [bs, kv, L] fp32     – ranking scores
        lut     : [bs, kv, L] int64    – token-id lookup table; value at
                                         ``lut[b, h, i]`` is the global token
                                         id corresponding to score
                                         ``scores[b, h, i]``
        skipped : [bs, kv, sk] int64   – pre-computed skipped token ids to
                                         append; ``sk`` may be 0
        out     : [bs, kv, ≥k+sk] int64 – contiguous output buffer (caller
                                          pre-allocated; caller slices to
                                          ``out[:, :, :k + sk]`` after the
                                          call)
        k       : Python int           – number of top-k elements

    Returns:
        ``out`` (same buffer; returned for convenience).
    """
    assert scores.is_cuda and lut.is_cuda and skipped.is_cuda and out.is_cuda, \
        "all tensors must be on CUDA"
    assert scores.dim() == 3 and lut.dim() == 3 and skipped.dim() == 3 and out.dim() == 3, \
        "all tensors must be 3D"
    assert scores.dtype == torch.float32, f"scores must be fp32; got {scores.dtype}"
    assert lut.dtype == torch.int64, f"lut must be int64; got {lut.dtype}"
    assert skipped.dtype == torch.int64, f"skipped must be int64; got {skipped.dtype}"
    assert out.dtype == torch.int64, f"out must be int64; got {out.dtype}"
    assert scores.shape == lut.shape, f"scores/lut shape mismatch: {scores.shape} vs {lut.shape}"
    assert skipped.shape[:2] == scores.shape[:2], \
        f"skipped (bs, kv) mismatch: {skipped.shape[:2]} vs {scores.shape[:2]}"
    assert out.shape[:2] == scores.shape[:2], \
        f"out (bs, kv) mismatch: {out.shape[:2]} vs {scores.shape[:2]}"

    sk = skipped.shape[2]
    assert k > 0 and k <= scores.shape[2], f"k={k} out of range [1, {scores.shape[2]}]"
    assert out.shape[2] >= k + sk, \
        f"out last dim={out.shape[2]} must be ≥ k+sk={k + sk}"
    assert out.is_contiguous(), "out must be contiguous"

    kernel = load_fused_topk_module()
    kernel.float_topk_gather_concat_3d(scores, lut, skipped, out, int(k))
    return out
