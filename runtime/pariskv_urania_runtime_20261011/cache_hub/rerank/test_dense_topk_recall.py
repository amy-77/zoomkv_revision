"""
Compare recall@500 between two CDS configs:
  A) dense_topk=8,  sparse_topk=4  (smaller output → faster)
  B) dense_topk=16, sparse_topk=4  (current default — dense全保留)

Reports recall@K against ground-truth top-K from exact RaBitQ scores
on the candidate token pool, averaged over many heads & random samples.
"""

import sys, os, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import (
    load_kernel_module,
    partial_rerank_scores_cuda,
    partial_rerank_chunk_dense_sparse_cuda,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topk.float_topk_ops import float_topk_3d

CHUNK_SIZE = 16


def make_inputs(bs, kv_heads, B, m, kv_len, nk, device, seed):
    torch.manual_seed(seed)
    sign = torch.randint(0, 2, (bs, kv_heads, B, kv_len, m), device=device)
    mag = torch.randint(0, 8, (bs, kv_heads, B, kv_len, m), device=device)
    v4bit = (sign << 3) | mag
    v_even = v4bit[..., 0::2]
    v_odd = v4bit[..., 1::2]
    key_4bit = ((v_odd << 4) | v_even).to(torch.uint8)

    key_block_weight = (
        torch.randn(bs, kv_heads, B, kv_len, dtype=torch.bfloat16, device=device).abs() + 0.01
    )
    q_blocks = torch.randn(bs, kv_heads, B, m, dtype=torch.bfloat16, device=device)
    q_norm = torch.randn(bs, kv_heads, dtype=torch.float32, device=device).abs() + 0.1
    mag_centers = torch.tensor(
        [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
        dtype=torch.bfloat16, device=device,
    )

    max_chunk_id = kv_len // CHUNK_SIZE
    chunk_idx = torch.stack([
        torch.randperm(max_chunk_id, device=device)[:nk] for _ in range(bs * kv_heads)
    ]).reshape(bs, kv_heads, nk).to(torch.int64)

    return chunk_idx, key_4bit, key_block_weight, q_blocks, q_norm, mag_centers


def expand_tokens(chunk_idx, kv_len):
    bs, kv_heads, nk = chunk_idx.shape
    offs = torch.arange(CHUNK_SIZE, device=chunk_idx.device)
    tokens = (chunk_idx * CHUNK_SIZE).unsqueeze(-1) + offs
    return tokens.reshape(bs, kv_heads, nk * CHUNK_SIZE).clamp(min=0, max=kv_len - 1)


def ground_truth(chunk_idx, kv_len, key_4bit, key_block_weight,
                 q_blocks, q_norm, mag_centers, B, m, final_topk, rerank_kernel):
    """Exact RaBitQ scoring of all nk*16 candidate tokens, then global top-K."""
    bs, kv_heads, _ = chunk_idx.shape
    all_tokens = expand_tokens(chunk_idx, kv_len)
    scores = partial_rerank_scores_cuda(
        all_tokens.contiguous(), key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        kernel=rerank_kernel,
    )
    k = min(final_topk, scores.shape[-1])
    _, topk_pos = torch.topk(scores.float(), k=k, dim=-1)
    gt_indices = torch.gather(all_tokens, dim=2, index=topk_pos)
    return gt_indices


def run_cds(chunk_idx, kv_len, key_4bit, key_block_weight, q_blocks, q_norm, mag_centers,
            B, m, dense_topk, sparse_topk, dense_ratio, final_topk, kernel,
            centroids=None):
    """Run new CDS kernel + global topk."""
    bs, kv_heads, nk = chunk_idx.shape
    chunk = CHUNK_SIZE

    if centroids is None:
        centroids = torch.randn(bs, kv_heads, nk, kv_heads, device=chunk_idx.device)
    raw_q = torch.randn(bs, kv_heads, B * m // 16, device=chunk_idx.device)

    density_scores = torch.randn(bs, kv_heads, nk, device=chunk_idx.device)
    n_dense = max(1, int(nk * dense_ratio))
    dense_pos = float_topk_3d(density_scores.float(), n_dense)
    dense_mask = torch.zeros(bs, kv_heads, nk, dtype=torch.bool, device=chunk_idx.device)
    dense_mask.scatter_(2, dense_pos, True)

    out_scores = torch.empty(bs, kv_heads, nk * chunk, dtype=torch.bfloat16, device=key_4bit.device)
    out_indices = torch.empty(bs, kv_heads, nk * chunk, dtype=torch.int64, device=key_4bit.device)

    partial_rerank_chunk_dense_sparse_cuda(
        chunk_idx, dense_mask,
        key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        dense_topk, sparse_topk,
        out_scores, out_indices,
        kernel=kernel,
    )

    k = min(final_topk, out_scores.shape[-1])
    topk_pos = float_topk_3d(out_scores.float(), k)
    final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
    return final_indices, dense_mask


def recall_at_k(selected, gt, K):
    """Per-head recall@K vs ground truth top-K."""
    bs, kv_heads, _ = selected.shape
    rs = []
    for b in range(bs):
        for h in range(kv_heads):
            sel = set(selected[b, h, :K].cpu().tolist())
            gtset = set(gt[b, h, :K].cpu().tolist())
            rs.append(len(sel & gtset) / K)
    return sum(rs) / len(rs)


def run_compare(device, rerank_kernel, topk_kernel, kv_len, nk, dense_ratio, sparse_topk,
                final_topk, bs=1, kv_heads=8, B=16, m=8, n_trials=5):
    """Run n_trials random samples, average recall@500 for dense_topk in {8, 16}."""
    results = {8: [], 16: []}
    for trial in range(n_trials):
        seed = 1000 + trial
        inputs = make_inputs(bs, kv_heads, B, m, kv_len, nk, device, seed=seed)
        chunk_idx, k4, kw, qb, qn, mc = inputs

        # Use SAME density scores across both configs by fixing seed inside run_cds
        torch.manual_seed(seed + 7777)
        density_scores = torch.randn(bs, kv_heads, nk, device=chunk_idx.device)
        n_dense = max(1, int(nk * dense_ratio))
        dense_pos = float_topk_3d(density_scores.float(), n_dense)
        dense_mask = torch.zeros(bs, kv_heads, nk, dtype=torch.bool, device=chunk_idx.device)
        dense_mask.scatter_(2, dense_pos, True)

        # Ground truth — exact scoring of all nk*16 candidate tokens
        gt = ground_truth(chunk_idx, kv_len, k4, kw, qb, qn, mc, B, m, final_topk, rerank_kernel)

        for dense_topk in [8, 16]:
            out_scores = torch.empty(bs, kv_heads, nk * CHUNK_SIZE, dtype=torch.bfloat16, device=k4.device)
            out_indices = torch.empty(bs, kv_heads, nk * CHUNK_SIZE, dtype=torch.int64, device=k4.device)
            partial_rerank_chunk_dense_sparse_cuda(
                chunk_idx, dense_mask, k4, kw,
                qb.reshape(-1, B, m).contiguous(),
                qn.reshape(-1).contiguous(),
                mc.contiguous(), B, m,
                dense_topk, sparse_topk,
                out_scores, out_indices,
                kernel=topk_kernel,
            )
            k_final = min(final_topk, out_scores.shape[-1])
            topk_pos = float_topk_3d(out_scores.float(), k_final)
            final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
            r = recall_at_k(final_indices, gt, min(500, k_final))
            results[dense_topk].append(r * 100)

    return results


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Device: {device}")
    print("Compiling kernels ...")
    rerank_kernel = load_kernel_module("rerank.cu", "rerank")
    topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    print("Ready.\n")

    print("="*78)
    print(f"{'kv_len':>8} {'nk':>5} {'dratio':>7} {'spk':>4} {'fk':>5} | "
          f"{'recall@500 (dense=8)':>22} | {'recall@500 (dense=16)':>23} | {'Δ':>6}")
    print("="*78)

    configs = [
        # (kv_len, nk, dense_ratio, sparse_topk, final_topk)
        (4096, 50, 0.1, 4, 500),
        (8192, 100, 0.1, 4, 500),
        (16384, 200, 0.1, 4, 500),
        (32768, 400, 0.1, 4, 500),
        (65536, 800, 0.1, 4, 500),
        (131072, 1600, 0.1, 4, 500),
        (16384, 200, 0.2, 4, 500),   # higher dense_ratio
        (16384, 200, 0.1, 8, 500),   # higher sparse_topk
    ]

    for cfg in configs:
        kv_len, nk, dratio, sparse_topk, final_topk = cfg
        res = run_compare(device, rerank_kernel, topk_kernel,
                          kv_len, nk, dratio, sparse_topk, final_topk, n_trials=5)
        m8 = sum(res[8]) / len(res[8])
        m16 = sum(res[16]) / len(res[16])
        delta = m16 - m8
        print(f"{kv_len:>8} {nk:>5} {dratio:>7.2f} {sparse_topk:>4} {final_topk:>5} | "
              f"{m8:>20.2f}%  | {m16:>21.2f}%  | {delta:>+5.2f}")
