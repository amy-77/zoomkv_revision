"""
Benchmark: old sorted-token kernel vs new chunk-ID kernel.
Measures Python-side prep + kernel + global topk, separately and combined.
"""

import sys, os, time, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import (
    load_kernel_module,
    partial_rerank_dense_sparse_cuda,
    partial_rerank_chunk_dense_sparse_cuda,
)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topk.float_topk_ops import float_topk_3d

CHUNK_SIZE = 16


def make_inputs(bs, kv_heads, B, m, kv_len, nk, device, seed=42):
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

    density_scores = torch.randn(bs, kv_heads, nk, device=device)

    return chunk_idx, key_4bit, key_block_weight, q_blocks, q_norm, mag_centers, density_scores


def bench_old(chunk_idx, nk, n_dense, sparse_topk, kv_len,
              key_4bit, key_block_weight, q_blocks, q_norm, mag_centers,
              density_scores, B, m, final_topk, kernel, warmup=10, iters=100):
    bs, kv_heads, _ = chunk_idx.shape
    chunk = CHUNK_SIZE
    chunk_offs = torch.arange(chunk, device=chunk_idx.device)
    n_sparse = nk - n_dense
    total_out_len = n_dense * chunk + n_sparse * sparse_topk

    qb = q_blocks.reshape(-1, B, m).contiguous()
    qn = q_norm.reshape(-1).contiguous()
    mc = mag_centers.contiguous()

    out_scores = torch.empty(bs, kv_heads, total_out_len, dtype=torch.bfloat16, device=key_4bit.device)
    out_indices = torch.empty(bs, kv_heads, total_out_len, dtype=torch.int64, device=key_4bit.device)

    def run():
        _, sorted_order = density_scores.sort(dim=-1, descending=True)
        sorted_chunk_idx = torch.gather(chunk_idx, 2, sorted_order)
        all_token_indices = (sorted_chunk_idx * chunk).unsqueeze(-1) + chunk_offs
        all_token_indices = all_token_indices.reshape(bs, kv_heads, nk * chunk)
        all_token_indices = all_token_indices.clamp(min=0, max=kv_len - 1)

        partial_rerank_dense_sparse_cuda(
            all_token_indices.contiguous(), key_4bit, key_block_weight,
            qb, qn, mc, B, m, n_dense, sparse_topk,
            out_scores, out_indices, kernel=kernel,
        )

        actual_topk = min(final_topk, total_out_len)
        topk_pos = float_topk_3d(out_scores.float(), actual_topk)
        final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
        return final_indices

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    return elapsed


def bench_new(chunk_idx, nk, n_dense, sparse_topk, kv_len,
              key_4bit, key_block_weight, q_blocks, q_norm, mag_centers,
              density_scores, B, m, final_topk, kernel, warmup=10, iters=100):
    bs, kv_heads, _ = chunk_idx.shape
    chunk = CHUNK_SIZE
    total_out_len = nk * chunk

    qb = q_blocks.reshape(-1, B, m).contiguous()
    qn = q_norm.reshape(-1).contiguous()
    mc = mag_centers.contiguous()

    out_scores = torch.empty(bs, kv_heads, total_out_len, dtype=torch.bfloat16, device=key_4bit.device)
    out_indices = torch.empty(bs, kv_heads, total_out_len, dtype=torch.int64, device=key_4bit.device)

    def run():
        dense_positions = float_topk_3d(density_scores.float(), n_dense)
        dense_mask = torch.zeros(bs, kv_heads, nk, dtype=torch.bool, device=chunk_idx.device)
        dense_mask.scatter_(2, dense_positions, True)

        partial_rerank_chunk_dense_sparse_cuda(
            chunk_idx.contiguous(), dense_mask.contiguous(),
            key_4bit, key_block_weight,
            qb, qn, mc, B, m, sparse_topk,
            out_scores, out_indices, kernel=kernel,
        )

        actual_topk = min(final_topk, total_out_len)
        topk_pos = float_topk_3d(out_scores.float(), actual_topk)
        final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
        return final_indices

    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        run()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters * 1000
    return elapsed


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Device: {device}")
    print("Compiling kernels ...")
    topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    _ = float_topk_3d(torch.randn(1, 1, 64, device=device), k=4)
    print("Ready.\n")

    bs, B, m = 1, 16, 8
    final_topk = 500

    configs = [
        # (kv_heads, kv_len, nk, n_dense, sparse_topk)
        (8, 4096, 50, 5, 4),
        (8, 8192, 100, 10, 4),
        (8, 16384, 200, 20, 4),
        (8, 32768, 400, 40, 4),
        (8, 65536, 800, 80, 4),
        (8, 131072, 1600, 160, 4),
        (8, 8192, 200, 20, 8),
    ]

    print(f"{'kv_heads':>8} {'kv_len':>8} {'nk':>5} {'n_dense':>7} {'sp_topk':>7} | "
          f"{'old (ms)':>10} {'new (ms)':>10} {'speedup':>8}")
    print("-" * 80)

    for kv_heads, kv_len, nk, n_dense, sparse_topk in configs:
        inputs = make_inputs(bs, kv_heads, B, m, kv_len, nk, device)
        chunk_idx, k4, kw, qb, qn, mc, ds = inputs

        t_old = bench_old(chunk_idx, nk, n_dense, sparse_topk, kv_len,
                          k4, kw, qb, qn, mc, ds, B, m, final_topk, topk_kernel)
        t_new = bench_new(chunk_idx, nk, n_dense, sparse_topk, kv_len,
                          k4, kw, qb, qn, mc, ds, B, m, final_topk, topk_kernel)

        speedup = t_old / t_new if t_new > 0 else float('inf')
        print(f"{kv_heads:>8} {kv_len:>8} {nk:>5} {n_dense:>7} {sparse_topk:>7} | "
              f"{t_old:>10.3f} {t_new:>10.3f} {speedup:>7.2f}x")
