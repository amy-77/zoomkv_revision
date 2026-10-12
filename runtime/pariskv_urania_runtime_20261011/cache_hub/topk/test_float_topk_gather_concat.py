"""Unit test for the fused topk_gather_concat kernel.

Validates against the 3-step PyTorch reference:
    topk_pos = torch.topk(scores, k).indices
    final    = gather(lut, dim=-1, index=topk_pos)
    if sk > 0: final = cat([final, skipped], dim=-1)

Tests:
    1. topk score sets agree (radix vs torch.topk; tie ordering may differ
       so we compare score *values* not raw indices)
    2. skipped tail copy is exact
    3. sk_count == 0 path (no skipped) works
    4. random shapes / k / sk
"""

from __future__ import annotations

import torch

from cache_hub.topk.float_topk_gather_concat_ops import float_topk_gather_concat_3d


def _ref_topk_gather_concat(scores: torch.Tensor, lut: torch.Tensor,
                            skipped: torch.Tensor, k: int) -> torch.Tensor:
    """3-step reference (torch.topk + gather + cat) returning [bs, kv, k + sk]."""
    _, topk_pos = torch.topk(scores, k=k, dim=-1)
    final = torch.gather(lut, dim=2, index=topk_pos)
    sk = skipped.shape[2]
    if sk > 0:
        final = torch.cat([final, skipped], dim=-1)
    return final


def _check_topk_score_set(out: torch.Tensor, ref: torch.Tensor,
                          scores: torch.Tensor, lut: torch.Tensor,
                          k: int, sk: int) -> None:
    """Verify out[:, :, :k] selects positions whose scores match torch.topk's
    score set.  Tie-breaking may differ between radix and torch.topk so we
    compare *score values* of the selected entries, not raw token-ids.
    """
    bs, kv = scores.shape[:2]
    bad = []
    for b in range(bs):
        for h in range(kv):
            # Map token-id back to position in scores via lut (lut is bijective
            # in our test setup since lut[i] = i + offset).
            our_ids = out[b, h, :k].tolist()
            ref_ids = ref[b, h, :k].tolist()
            our_scores = sorted(
                [scores[b, h, lut[b, h] == tid].item() for tid in our_ids],
                reverse=True,
            )
            ref_scores = sorted(
                [scores[b, h, lut[b, h] == tid].item() for tid in ref_ids],
                reverse=True,
            )
            if our_scores != ref_scores:
                bad.append((b, h, our_scores, ref_scores))
    if bad:
        b, h, ours, refs = bad[0]
        raise AssertionError(
            f"top-k score set mismatch at (bs={b}, kv={h}); "
            f"ours[:5]={ours[:5]}, ref[:5]={refs[:5]}"
        )


def _check_skipped_tail(out: torch.Tensor, skipped: torch.Tensor,
                        k: int, sk: int) -> None:
    if sk == 0:
        return
    actual = out[:, :, k:k + sk]
    if not torch.equal(actual, skipped):
        diff = (actual != skipped).nonzero()[:5]
        raise AssertionError(
            f"skipped tail mismatch; first diff indices: {diff.tolist()}"
        )


def _run_case(bs: int, kv: int, L: int, k: int, sk: int,
              max_sk: int | None = None, seed: int = 0) -> None:
    if max_sk is None:
        max_sk = sk
    torch.manual_seed(seed)
    device = torch.device("cuda")

    scores = torch.randn(bs, kv, L, device=device, dtype=torch.float32)
    # LUT: sequential ids per (bs, kv) row (values won't collide across rows
    # but we only check per-row gather correctness anyway).
    lut = torch.arange(L, device=device, dtype=torch.int64).expand(bs, kv, L).contiguous()
    if sk > 0:
        skipped = torch.randint(L, L + 1024, (bs, kv, sk), device=device, dtype=torch.int64)
    else:
        skipped = torch.empty(bs, kv, 0, device=device, dtype=torch.int64)

    out = torch.empty(bs, kv, k + max_sk, device=device, dtype=torch.int64).contiguous()
    out.fill_(-999)  # detect uninitialized slots

    float_topk_gather_concat_3d(scores, lut, skipped, out, k)
    torch.cuda.synchronize()

    # No -999 in the active region.
    active = out[:, :, :k + sk]
    assert (active != -999).all(), \
        f"uninitialized slot in active region (bs={bs}, kv={kv}, k={k}, sk={sk})"

    ref = _ref_topk_gather_concat(scores, lut, skipped, k)
    _check_topk_score_set(out, ref, scores, lut, k, sk)
    _check_skipped_tail(out, skipped, k, sk)
    print(f"  ✓ bs={bs} kv={kv} L={L} k={k} sk={sk}")


def main():
    print("test_float_topk_gather_concat ...")

    # CDS-realistic: nk * group_size = 400 * 16 = 6400, final_topk = 100
    _run_case(bs=1, kv=8, L=6400, k=100, sk=0,  seed=0)
    _run_case(bs=1, kv=8, L=6400, k=100, sk=12, seed=1)
    _run_case(bs=8, kv=8, L=6400, k=100, sk=12, seed=2)
    _run_case(bs=8, kv=8, L=6400, k=100, sk=0,  seed=3)

    # Edge: small L
    _run_case(bs=2, kv=4, L=512,  k=64,  sk=8,  seed=4)

    # Edge: k = 1
    _run_case(bs=1, kv=2, L=1024, k=1,   sk=0,  seed=5)

    # Edge: out has extra slack space (max_sk > sk)
    _run_case(bs=2, kv=4, L=2048, k=50,  sk=8,  max_sk=64, seed=6)

    print("All cases passed.")


if __name__ == "__main__":
    main()
