"""
Test: new chunk-ID based dense/sparse kernel vs old sorted-token kernel.

Verifies:
1. New kernel produces same final tokens as old kernel (recall equivalence)
2. Both kernels' recall@500 against exact ground-truth scores
"""

import sys, os, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import (
    load_kernel_module,
    partial_rerank_scores_cuda,
    partial_rerank_dense_sparse_cuda,
    partial_rerank_chunk_dense_sparse_cuda,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topk.float_topk_ops import float_topk_3d

CHUNK_SIZE = 16


def make_inputs(bs, kv_heads, B, m, kv_len, nk, device, seed=42):
    """Generate random inputs and select nk random chunks."""
    torch.manual_seed(seed)

    sign = torch.randint(0, 2, (bs, kv_heads, B, kv_len, m), device=device)
    mag = torch.randint(0, 8, (bs, kv_heads, B, kv_len, m), device=device)
    v4bit = (sign << 3) | mag
    v_even = v4bit[..., 0::2]
    v_odd = v4bit[..., 1::2]
    key_4bit = ((v_odd << 4) | v_even).to(torch.uint8)

    key_block_weight = (
        torch.randn(bs, kv_heads, B, kv_len, dtype=torch.bfloat16, device=device).abs()
        + 0.01
    )

    q_blocks = torch.randn(bs, kv_heads, B, m, dtype=torch.bfloat16, device=device)
    q_norm = torch.randn(bs, kv_heads, dtype=torch.float32, device=device).abs() + 0.1
    mag_centers = torch.tensor(
        [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
        dtype=torch.bfloat16,
        device=device,
    )

    max_chunk_id = kv_len // CHUNK_SIZE
    chunk_idx = torch.stack([
        torch.randperm(max_chunk_id, device=device)[:nk] for _ in range(bs * kv_heads)
    ]).reshape(bs, kv_heads, nk).to(torch.int64)

    return chunk_idx, key_4bit, key_block_weight, q_blocks, q_norm, mag_centers


def run_old_kernel(chunk_idx, nk, n_dense, sparse_topk, kv_len,
                   key_4bit, key_block_weight, q_blocks, q_norm, mag_centers,
                   B, m, final_topk, kernel):
    """Old path: sort + gather + expand token indices + partial_rerank_dense_sparse."""
    bs, kv_heads, _ = chunk_idx.shape
    chunk = CHUNK_SIZE

    # Simulate density scores (random for testing)
    torch.manual_seed(99)
    density_scores = torch.randn(bs, kv_heads, nk, device=chunk_idx.device)

    _, sorted_order = density_scores.sort(dim=-1, descending=True)
    sorted_chunk_idx = torch.gather(chunk_idx, 2, sorted_order)

    chunk_offs = torch.arange(chunk, device=chunk_idx.device)
    all_token_indices = (sorted_chunk_idx * chunk).unsqueeze(-1) + chunk_offs
    all_token_indices = all_token_indices.reshape(bs, kv_heads, nk * chunk)
    all_token_indices = all_token_indices.clamp(min=0, max=kv_len - 1)

    n_sparse = nk - n_dense
    total_out_len = n_dense * chunk + n_sparse * sparse_topk
    out_scores = torch.empty(bs, kv_heads, total_out_len, dtype=torch.bfloat16, device=key_4bit.device)
    out_indices = torch.empty(bs, kv_heads, total_out_len, dtype=torch.int64, device=key_4bit.device)

    partial_rerank_dense_sparse_cuda(
        all_token_indices.contiguous(), key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        n_dense, sparse_topk,
        out_scores, out_indices,
        kernel=kernel,
    )
    torch.cuda.synchronize()

    actual_topk = min(final_topk, total_out_len)
    topk_pos = float_topk_3d(out_scores.float(), actual_topk)
    final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
    return final_indices, out_scores, out_indices


def run_new_kernel(chunk_idx, nk, n_dense, sparse_topk, kv_len,
                   key_4bit, key_block_weight, q_blocks, q_norm, mag_centers,
                   B, m, final_topk, kernel):
    """New path: float_topk_3d + dense_mask + partial_rerank_chunk_dense_sparse."""
    bs, kv_heads, _ = chunk_idx.shape
    chunk = CHUNK_SIZE

    torch.manual_seed(99)
    density_scores = torch.randn(bs, kv_heads, nk, device=chunk_idx.device)

    dense_positions = float_topk_3d(density_scores.float(), n_dense)
    dense_mask = torch.zeros(bs, kv_heads, nk, dtype=torch.bool, device=chunk_idx.device)
    dense_mask.scatter_(2, dense_positions, True)

    total_out_len = nk * chunk
    out_scores = torch.empty(bs, kv_heads, total_out_len, dtype=torch.bfloat16, device=key_4bit.device)
    out_indices = torch.empty(bs, kv_heads, total_out_len, dtype=torch.int64, device=key_4bit.device)

    partial_rerank_chunk_dense_sparse_cuda(
        chunk_idx.contiguous(), dense_mask.contiguous(),
        key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        sparse_topk,
        out_scores, out_indices,
        kernel=kernel,
    )
    torch.cuda.synchronize()

    actual_topk = min(final_topk, total_out_len)
    topk_pos = float_topk_3d(out_scores.float(), actual_topk)
    final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
    return final_indices, out_scores, out_indices


def ground_truth_topk(chunk_idx, kv_len, key_4bit, key_block_weight,
                      q_blocks, q_norm, mag_centers, B, m, final_topk, rerank_kernel):
    """Get exact top-k from all candidate tokens (no dense/sparse, score all)."""
    bs, kv_heads, nk = chunk_idx.shape
    chunk = CHUNK_SIZE
    chunk_offs = torch.arange(chunk, device=chunk_idx.device)
    all_token_indices = (chunk_idx * chunk).unsqueeze(-1) + chunk_offs
    all_token_indices = all_token_indices.reshape(bs, kv_heads, nk * chunk)
    all_token_indices = all_token_indices.clamp(min=0, max=kv_len - 1)

    all_scores = partial_rerank_scores_cuda(
        all_token_indices.contiguous(), key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        kernel=rerank_kernel,
    )

    actual_topk = min(final_topk, all_scores.shape[-1])
    _, topk_pos = torch.topk(all_scores.float(), k=actual_topk, dim=-1)
    gt_indices = torch.gather(all_token_indices, dim=2, index=topk_pos)
    return gt_indices


def compute_recall(selected, ground_truth, K):
    """Per-head recall@K."""
    bs, kv_heads, _ = selected.shape
    recalls = []
    for b in range(bs):
        for h in range(kv_heads):
            sel_set = set(selected[b, h].cpu().tolist())
            gt_set = set(ground_truth[b, h, :K].cpu().tolist())
            recalls.append(len(sel_set & gt_set) / len(gt_set))
    return sum(recalls) / len(recalls)


def test_equivalence(device, rerank_kernel, topk_kernel):
    """New kernel should select same tokens as old kernel (same density scores)."""
    bs, kv_heads, B, m = 1, 8, 16, 8
    kv_len, nk = 4096, 100
    n_dense, sparse_topk, final_topk = 10, 4, 500

    chunk_idx, k4, kw, qb, qn, mc = make_inputs(bs, kv_heads, B, m, kv_len, nk, device)

    old_final, _, _ = run_old_kernel(
        chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
        B, m, final_topk, topk_kernel,
    )
    new_final, _, _ = run_new_kernel(
        chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
        B, m, final_topk, topk_kernel,
    )

    mismatches = 0
    for h in range(kv_heads):
        old_set = set(old_final[0, h].cpu().tolist())
        new_set = set(new_final[0, h].cpu().tolist())
        overlap = len(old_set & new_set) / max(len(old_set), 1)
        if overlap < 1.0:
            mismatches += 1
            print(f"  H{h}: overlap={overlap*100:.1f}% old={len(old_set)} new={len(new_set)}")

    ok = mismatches == 0
    print(f"[test_equivalence] mismatches={mismatches}/{kv_heads} {'PASSED' if ok else 'FAILED'}")
    return ok


def test_recall_at_500(device, rerank_kernel, topk_kernel):
    """Recall@500 of new kernel vs ground truth (exact scoring of all candidate tokens)."""
    bs, kv_heads, B, m = 1, 8, 16, 8
    kv_len, nk = 8192, 200
    n_dense, sparse_topk, final_topk = 20, 4, 500

    chunk_idx, k4, kw, qb, qn, mc = make_inputs(bs, kv_heads, B, m, kv_len, nk, device, seed=123)

    gt = ground_truth_topk(chunk_idx, kv_len, k4, kw, qb, qn, mc, B, m, final_topk, rerank_kernel)

    new_final, _, _ = run_new_kernel(
        chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
        B, m, final_topk, topk_kernel,
    )

    for K in [100, 500]:
        if K > final_topk:
            continue
        recall = compute_recall(new_final, gt, K)
        print(f"  New kernel Recall@{K} = {recall*100:.2f}%")

    old_final, _, _ = run_old_kernel(
        chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
        B, m, final_topk, topk_kernel,
    )

    for K in [100, 500]:
        if K > final_topk:
            continue
        recall = compute_recall(old_final, gt, K)
        print(f"  Old kernel Recall@{K} = {recall*100:.2f}%")

    print("[test_recall_at_500] done")


def test_various_configs(device, rerank_kernel, topk_kernel):
    """Test recall across different nk, n_dense, sparse_topk configs."""
    bs, kv_heads, B, m = 1, 8, 16, 8
    kv_len = 8192
    final_topk = 500

    configs = [
        (50, 5, 4),    # small nk
        (100, 10, 4),  # medium
        (200, 20, 4),  # large
        (200, 20, 8),  # larger sparse topk
        (200, 40, 4),  # more dense chunks
    ]

    for nk, n_dense, sparse_topk in configs:
        chunk_idx, k4, kw, qb, qn, mc = make_inputs(
            bs, kv_heads, B, m, kv_len, nk, device, seed=nk * 7,
        )

        gt = ground_truth_topk(
            chunk_idx, kv_len, k4, kw, qb, qn, mc, B, m, final_topk, rerank_kernel,
        )

        new_final, _, _ = run_new_kernel(
            chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
            B, m, min(final_topk, n_dense * CHUNK_SIZE + (nk - n_dense) * sparse_topk),
            topk_kernel,
        )
        r500 = compute_recall(new_final, gt, min(500, new_final.shape[-1]))

        old_final, _, _ = run_old_kernel(
            chunk_idx, nk, n_dense, sparse_topk, kv_len, k4, kw, qb, qn, mc,
            B, m, min(final_topk, n_dense * CHUNK_SIZE + (nk - n_dense) * sparse_topk),
            topk_kernel,
        )
        r500_old = compute_recall(old_final, gt, min(500, old_final.shape[-1]))

        print(
            f"  nk={nk:>3d} n_dense={n_dense:>2d} sparse_topk={sparse_topk} | "
            f"new R@500={r500*100:6.2f}%  old R@500={r500_old*100:6.2f}%"
        )

    print("[test_various_configs] done")


if __name__ == "__main__":
    device = "cuda:1"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Device: {device}")
    print("Compiling kernels ...")
    rerank_kernel = load_kernel_module("rerank.cu", "rerank")
    topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    print("Kernels ready.\n")

    test_equivalence(device, rerank_kernel, topk_kernel)
    print()
    test_recall_at_500(device, rerank_kernel, topk_kernel)
    print()
    test_various_configs(device, rerank_kernel, topk_kernel)
