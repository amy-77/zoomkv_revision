"""
Float Top-K v2: radix select with global-memory candidate buffers.

Fixes shared-memory overflow in float_topk when scores are highly clustered
(e.g. Quest bbox upper-bound scores).

Usage:
    from cache_hub.topk.float_topk_v2_ops import float_topk_v2, float_topk_v2_3d
    indices = float_topk_v2(scores_2d, k)
    indices = float_topk_v2_3d(scores_3d, k)
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
def load_float_topk_v2_module(
    path: str = "float_topk_v2.cu",
    name: str = "float_topk_v2_ext",
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


def float_topk_v2(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K v2 using radix select with global-memory buffers (2D input).

    Args:
        scores: [B, L] float tensor on CUDA
        k: number of top elements to select

    Returns:
        indices: [B, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 2, "scores must be 2D [B, L]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(1), f"k={k} cannot exceed L={scores.size(1)}"

    kernel = load_float_topk_v2_module()
    return kernel.float_topk_v2(scores, k)


def float_topk_v2_3d(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    Float Top-K v2 using radix select with global-memory buffers (3D input).

    Args:
        scores: [bs, kv_heads, n_chunks] float tensor on CUDA
        k: number of top elements to select

    Returns:
        indices: [bs, kv_heads, k] int64 tensor
    """
    assert scores.is_cuda, "scores must be on CUDA"
    assert scores.dim() == 3, "scores must be 3D [bs, kv_heads, n_chunks]"
    assert k > 0, f"k must be positive, got {k}"
    assert k <= scores.size(2), f"k={k} cannot exceed n_chunks={scores.size(2)}"

    kernel = load_float_topk_v2_module()
    return kernel.float_topk_v2_3d(scores, k)
