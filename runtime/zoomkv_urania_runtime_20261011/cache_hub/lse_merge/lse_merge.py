from functools import lru_cache
from typing import Any, Iterable

import importlib.util
import os
import warnings

import torch


@lru_cache()
def _prepare_for_load() -> str:
    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_lse_merge_module(
    path: str = "lse_merge.cu",
    name: str = "lse_merge_ext",
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


def _canonical_lse(lse: torch.Tensor, attn_out: torch.Tensor) -> torch.Tensor:
    """Normalize FlashAttention LSE to [batch, heads, 1]."""
    if attn_out.dim() != 4 or attn_out.size(1) != 1:
        raise ValueError(f"attn_out must be [bs, 1, heads, dim], got {tuple(attn_out.shape)}")

    bs = attn_out.size(0)
    heads = attn_out.size(2)

    if lse.dim() == 2 and lse.shape == (bs, heads):
        return lse.unsqueeze(-1)

    if lse.dim() == 3:
        if lse.shape == (bs, heads, 1):
            return lse
        if lse.shape == (bs, 1, heads):
            return lse.transpose(1, 2).contiguous()

    raise ValueError(
        "lse must have shape [bs, heads, 1], [bs, 1, heads], or [bs, heads]; "
        f"got {tuple(lse.shape)} for attn_out={tuple(attn_out.shape)}"
    )


def lse_merge_torch(
    attn_out_a: torch.Tensor,
    lse_a: torch.Tensor,
    attn_out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Merge two independently normalized attention outputs with their LSEs.

    This is mathematically equivalent to running attention once over the
    concatenation of both KV segments:

        softmax([logits_a, logits_b]) @ [values_a, values_b]

    Args:
        attn_out_a/b: [bs, 1, heads, dim]
        lse_a/b:      [bs, heads, 1] (FlashAttention return_softmax_lse layout)

    Returns:
        [bs, 1, heads, dim], same dtype as attn_out_a.
    """
    if attn_out_a.shape != attn_out_b.shape:
        raise ValueError(
            f"attn outputs must have identical shape, got {attn_out_a.shape} and {attn_out_b.shape}"
        )
    if attn_out_a.device != attn_out_b.device:
        raise ValueError("attn outputs must be on the same device")

    out_dtype = attn_out_a.dtype
    lse_a = _canonical_lse(lse_a, attn_out_a).to(torch.float32)
    lse_b = _canonical_lse(lse_b, attn_out_b).to(torch.float32)

    max_lse = torch.maximum(lse_a, lse_b)
    weight_a = torch.exp(lse_a - max_lse)
    weight_b = torch.exp(lse_b - max_lse)
    denom = weight_a + weight_b

    merged = (
        weight_a * attn_out_a.squeeze(1).to(torch.float32)
        + weight_b * attn_out_b.squeeze(1).to(torch.float32)
    ) / denom
    return merged.unsqueeze(1).to(out_dtype)


def lse_merge_cuda(
    attn_out_a: torch.Tensor,
    lse_a: torch.Tensor,
    attn_out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Fused CUDA implementation of lse_merge_torch."""
    if not attn_out_a.is_cuda:
        raise ValueError("attn_out_a must be a CUDA tensor")
    if attn_out_a.shape != attn_out_b.shape:
        raise ValueError(
            f"attn outputs must have identical shape, got {attn_out_a.shape} and {attn_out_b.shape}"
        )
    lse_a = _canonical_lse(lse_a, attn_out_a).to(torch.float32).contiguous()
    lse_b = _canonical_lse(lse_b, attn_out_b).to(torch.float32).contiguous()

    kernel = load_lse_merge_module()
    return kernel.lse_merge(
        attn_out_a.contiguous(),
        lse_a,
        attn_out_b.contiguous(),
        lse_b,
    )
