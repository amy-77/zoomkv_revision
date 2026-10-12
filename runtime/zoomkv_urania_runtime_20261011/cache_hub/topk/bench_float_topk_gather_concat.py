"""Bench fused topk_gather_concat against the 3-step PyTorch chain.

CDS-realistic shape: bs=8, kv=8, L=6400 (nk=400 * group_size=16),
k=100 (final_topk), sk=12 (typical mid-stream skipped count).
"""

from __future__ import annotations

import time
import torch

from cache_hub.topk.float_topk_ops import float_topk_3d
from cache_hub.topk.float_topk_gather_concat_ops import float_topk_gather_concat_3d


def _bench_3step(scores, lut, skipped, k, iters=200, warmup=50):
    sk = skipped.shape[2]
    torch.cuda.synchronize()
    for _ in range(warmup):
        topk_pos = float_topk_3d(scores, k)
        final = torch.gather(lut, dim=2, index=topk_pos)
        if sk > 0:
            final = torch.cat([final, skipped], dim=-1)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        topk_pos = float_topk_3d(scores, k)
        final = torch.gather(lut, dim=2, index=topk_pos)
        if sk > 0:
            final = torch.cat([final, skipped], dim=-1)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def _bench_fused(scores, lut, skipped, out, k, iters=200, warmup=50):
    torch.cuda.synchronize()
    for _ in range(warmup):
        float_topk_gather_concat_3d(scores, lut, skipped, out, k)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        float_topk_gather_concat_3d(scores, lut, skipped, out, k)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main():
    device = torch.device("cuda")
    torch.manual_seed(0)

    cases = [
        # bs, kv, L, k, sk, label
        (8, 8, 6400, 100, 0,  "bs=8 sk=0  (steady state)"),
        (8, 8, 6400, 100, 12, "bs=8 sk=12 (mid-stream)"),
        (6, 8, 6400, 100, 12, "bs=6 sk=12 (matches nsys run)"),
        (1, 8, 6400, 100, 12, "bs=1 sk=12"),
        (8, 8, 3200, 100, 0,  "bs=8 L=3200 sk=0 (smaller cand)"),
    ]

    print(f"{'shape':<32} {'3-step (ms)':>12} {'fused (ms)':>12} {'speedup':>10}")
    print("-" * 72)
    for bs, kv, L, k, sk, label in cases:
        scores = torch.randn(bs, kv, L, device=device, dtype=torch.float32)
        lut = torch.arange(L, device=device, dtype=torch.int64).expand(bs, kv, L).contiguous()
        if sk > 0:
            skipped = torch.randint(L, L + 1024, (bs, kv, sk), device=device, dtype=torch.int64)
        else:
            skipped = torch.empty(bs, kv, 0, device=device, dtype=torch.int64)
        out = torch.empty(bs, kv, k + sk, device=device, dtype=torch.int64).contiguous()

        t_3step = _bench_3step(scores, lut, skipped, k)
        t_fused = _bench_fused(scores, lut, skipped, out, k)
        speedup = t_3step / t_fused
        print(f"{label:<32} {t_3step*1000:>11.2f}us {t_fused*1000:>11.2f}us {speedup:>9.2f}x")


if __name__ == "__main__":
    main()
