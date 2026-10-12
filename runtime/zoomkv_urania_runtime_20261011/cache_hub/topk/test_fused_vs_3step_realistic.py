"""Direct comparison of fused vs 3-step on identical inputs (no generation drift).

For each test case we:
  1. Build random `out_scores` (fp32) and `out_indices` (int64 LUT)
  2. Optionally build skipped_indices
  3. Run both paths on the SAME tensors, collect outputs
  4. Compare the SET of selected token IDs (order-invariant; downstream
     attention is set-invariant under softmax)

A non-tie input is also tested to verify exact set equality (no tie-breaking
ambiguity).
"""

from __future__ import annotations

import torch

from cache_hub.topk.float_topk_ops import float_topk_3d
from cache_hub.topk.float_topk_gather_concat_ops import float_topk_gather_concat_3d


def _three_step(scores, lut, skipped, k):
    sk = skipped.shape[2]
    topk_pos = float_topk_3d(scores, k)
    final = torch.gather(lut, dim=2, index=topk_pos)
    if sk > 0:
        final = torch.cat([final, skipped], dim=-1)
    return final


def _fused(scores, lut, skipped, k):
    sk = skipped.shape[2]
    bs, kv = scores.shape[:2]
    out = torch.empty(bs, kv, k + sk, dtype=torch.int64, device=scores.device).contiguous()
    float_topk_gather_concat_3d(scores, lut, skipped, out, k)
    return out


def _compare_topk_set(out_a, out_b, k, label):
    """Compare the SET (not order) of the first k entries per row."""
    bs, kv = out_a.shape[:2]
    diffs = []
    for b in range(bs):
        for h in range(kv):
            sa = set(out_a[b, h, :k].tolist())
            sb = set(out_b[b, h, :k].tolist())
            if sa != sb:
                diffs.append((b, h, sa - sb, sb - sa))
    if diffs:
        b, h, only_a, only_b = diffs[0]
        n = len(diffs)
        return f"FAIL {label}: {n} (bs,kv) rows differ; first @(b={b},h={h}) only_a={list(only_a)[:5]} only_b={list(only_b)[:5]}"
    return f"PASS {label}: all (bs,kv) rows have identical top-k token sets"


def _compare_skipped_tail(out_a, out_b, k, sk, label):
    if sk == 0:
        return f"PASS {label}: sk=0, no tail to check"
    tail_a = out_a[:, :, k:k + sk]
    tail_b = out_b[:, :, k:k + sk]
    if torch.equal(tail_a, tail_b):
        return f"PASS {label}: skipped tail bit-identical"
    return f"FAIL {label}: skipped tail differs"


def main():
    device = torch.device("cuda")

    cases = [
        # (bs, kv, L, k, sk, force_unique_scores, label)
        (8, 8, 6400, 100, 0,  True,  "bs=8 sk=0 unique"),
        (8, 8, 6400, 100, 12, True,  "bs=8 sk=12 unique"),
        (6, 8, 6400, 100, 12, True,  "bs=6 sk=12 unique"),
        (1, 8, 6400, 100, 12, True,  "bs=1 sk=12 unique"),
        # With ties (default float distribution): set equality should still hold
        # in expectation but may have tie-induced differences.  We expect this
        # to either pass exactly, or differ ONLY at tie boundaries.
        (8, 8, 6400, 100, 12, False, "bs=8 sk=12 (ties allowed)"),
    ]

    print(f"{'case':<32} {'topk-set':<60} {'tail':<40}")
    print("-" * 132)

    for bs, kv, L, k, sk, force_unique, label in cases:
        torch.manual_seed(42)
        if force_unique:
            # Generate strictly distinct scores (no ties at all): use
            # arange permuted, scaled to fp32 distinct values.
            scores = torch.empty(bs, kv, L, dtype=torch.float32, device=device)
            for b in range(bs):
                for h in range(kv):
                    perm = torch.randperm(L, device=device).to(torch.float32)
                    scores[b, h] = perm * 1e-3  # scaled to distinct fp32 values
        else:
            scores = torch.randn(bs, kv, L, device=device, dtype=torch.float32)

        lut = (torch.arange(L, device=device, dtype=torch.int64) + 1000).expand(bs, kv, L).contiguous()
        if sk > 0:
            skipped = torch.randint(0, 1000, (bs, kv, sk), device=device, dtype=torch.int64)
        else:
            skipped = torch.empty(bs, kv, 0, device=device, dtype=torch.int64)

        out_3 = _three_step(scores, lut, skipped, k)
        out_f = _fused(scores, lut, skipped, k)

        msg_topk = _compare_topk_set(out_3, out_f, k, label)
        msg_tail = _compare_skipped_tail(out_3, out_f, k, sk, label)
        print(f"{label:<32} {msg_topk:<60} {msg_tail:<40}")


if __name__ == "__main__":
    main()
