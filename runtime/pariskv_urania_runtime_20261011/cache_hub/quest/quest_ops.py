"""JIT loader for the hierarchical Quest CUDA kernels.

If ``quest_chunk_score.cu`` is not shipped with this checkout, scoring falls
back to the Quest upper bound in PyTorch:

    score(chunk) = sum_d max(q_d * max_d, q_d * min_d)
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


class QuestPytorch:
    """In-place Quest scoring with the same call signature as the CUDA module."""

    def quest_chunk_score(self, raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid):
        n = int(n_chunks)
        q = raw_q.unsqueeze(2)
        score = torch.maximum(
            q * chunk_max[:, :, :n, :],
            q * chunk_min[:, :, :n, :],
        ).sum(dim=-1)
        if chunk_valid is not None:
            score = score.masked_fill(~chunk_valid[:, :, :n], torch.finfo(score.dtype).min)
        scores_out.copy_(score.to(dtype=scores_out.dtype))

    def quest_sub_chunk_score(
        self, raw_q, chunk_min, chunk_max, large_idx, sub_scores, nk_large, factor,
    ):
        nk = int(nk_large)
        factor = int(factor)
        dim = raw_q.shape[-1]
        base = large_idx[:, :, :nk].to(dtype=torch.int64) * factor
        offs = torch.arange(factor, device=raw_q.device, dtype=torch.int64)
        small_idx = base.unsqueeze(-1) + offs
        n_small = chunk_min.shape[2]
        in_range = (small_idx >= 0) & (small_idx < n_small)
        safe = small_idx.clamp(0, max(n_small - 1, 0))
        gather_idx = safe.reshape(raw_q.shape[0], raw_q.shape[1], nk * factor, 1).expand(
            -1, -1, -1, dim
        )
        mn = torch.gather(chunk_min, 2, gather_idx)
        mx = torch.gather(chunk_max, 2, gather_idx)
        q = raw_q.unsqueeze(2)
        score = torch.maximum(q * mx, q * mn).sum(dim=-1)
        score = score.masked_fill(~in_range.reshape_as(score), torch.finfo(score.dtype).min)
        sub_scores.copy_(score.to(dtype=sub_scores.dtype))

    def quest_map_back(self, large_idx, sub_topk_pos, chunk_idx, factor, n_chunks):
        factor = int(factor)
        pos = sub_topk_pos.to(dtype=torch.int64)
        slot = torch.div(pos, factor, rounding_mode="floor").clamp(0, large_idx.shape[-1] - 1)
        sub = torch.remainder(pos, factor)
        parent = torch.gather(large_idx, 2, slot)
        global_idx = parent * factor + sub
        n = int(n_chunks)
        if n > 0:
            global_idx = global_idx.clamp(0, n - 1)
        chunk_idx.copy_(global_idx)


def _source_path() -> Path:
    configured = os.environ.get("ZOOMKV_QUEST_CUDA_SOURCE")
    candidates = [
        Path(configured) if configured else None,
        Path(__file__).with_name("quest_chunk_score.cu"),
        Path(__file__).resolve().parents[3]
        / "vllm20_github"
        / "vllm"
        / "v1"
        / "attention"
        / "ops"
        / "zoomkv"
        / "cuda"
        / "quest_chunk_score.cu",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate
    raise FileNotFoundError("quest_chunk_score.cu was not found")


@lru_cache(maxsize=1)
def load_kernel_module():
    try:
        source = _source_path()
    except FileNotFoundError:
        print(
            "[Init]   quest_chunk_score.cu not found, using PyTorch Quest scoring",
            flush=True,
        )
        return QuestPytorch()
    build_directory = Path(__file__).with_name("build")
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="zoomkv_quest_ops",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-std=c++20"],
        verbose=True,
    )
