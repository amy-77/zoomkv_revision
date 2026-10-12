"""
Fused per-chunk top-k with index gather (CUDA).

Usage:
    from cache_hub.topk.chunk_topk_ops import chunk_topk_gather
    top_scores, top_indices = chunk_topk_gather(scores_4d, indices_4d, k=2)
"""

import os
import warnings
import importlib.util
from functools import lru_cache
from typing import Any, Iterable, Tuple

import torch


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_chunk_topk_module(
    path: str = "chunk_topk.cu",
    name: str = "chunk_topk_ext",
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


def chunk_topk_gather(
    scores: torch.Tensor,
    indices: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused per-chunk top-k selection with index gather.

    Args:
        scores:  [bs, kv_heads, nk, chunk_size]  float tensor (chunk_size ≤ 32)
        indices: [bs, kv_heads, nk, chunk_size]  int64 tensor
        k:       number of top elements per chunk

    Returns:
        (top_scores, top_indices): each [bs, kv_heads, nk, k]
    """
    kernel = load_chunk_topk_module()
    return kernel.chunk_topk_gather(
        scores.contiguous(),
        indices.contiguous(),
        k,
    )
