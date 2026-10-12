"""
Unit test for rerank_topk.cu — CHUNK_SIZE=32, warp-level bitonic sort, top-k.

Validates the kernel against two independent references:
  1. Pure-Python RaBitQ scoring (sequential bf16 accumulation) + torch.topk
  2. rerank.cu (scores-only kernel) + torch.topk  (cross-kernel exact match)

Primary target: chunk_size=32, topk=8 — each warp handles one chunk of 32 keys
and selects top-8 via bitonic sort.

NOTE: the kernel has a flat_bh = bidx * blockDim.y + bidy convention (should be
gridDim.y), so bs > 1 produces wrong results. All tests use bs=1.
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank import load_kernel_module, partial_rerank_scores_cuda

CHUNK_SIZE = 32


# ── Input generation ──────────────────────────────────────────────────────

def make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=42):
    torch.manual_seed(seed)
    assert candidate_len % CHUNK_SIZE == 0

    coarse_indices = torch.stack([
        torch.randperm(kv_len, device=device)[:candidate_len]
        for _ in range(bs * kv_heads)
    ]).reshape(bs, kv_heads, candidate_len).to(torch.int64)

    sign = torch.randint(0, 2, (bs, kv_heads, B, kv_len, m), device=device)
    mag  = torch.randint(0, 8, (bs, kv_heads, B, kv_len, m), device=device)
    v4bit = (sign << 3) | mag
    v_even = v4bit[..., 0::2]
    v_odd  = v4bit[..., 1::2]
    key_4bit = ((v_odd << 4) | v_even).to(torch.uint8)

    key_block_weight = torch.randn(
        bs, kv_heads, B, kv_len, dtype=torch.bfloat16, device=device
    ).abs() + 0.01

    q_blocks = torch.randn(
        bs, kv_heads, B, m, dtype=torch.bfloat16, device=device
    )
    q_norm = (torch.randn(bs, kv_heads, dtype=torch.bfloat16, device=device).abs() + 0.1)
    mag_centers = torch.tensor(
        [0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
        dtype=torch.bfloat16, device=device,
    )
    return coarse_indices, key_4bit, key_block_weight, q_blocks, q_norm, mag_centers


# ── Pure-Python reference ─────────────────────────────────────────────────

def python_rabitq_scores(coarse_indices, key_4bit, key_block_weight,
                         q_blocks, q_norm, mag_centers, B, m):
    """Pure-Python RaBitQ scoring with sequential bf16 accumulation.

    Matches the CUDA kernel's computation order exactly:
      for each block i:
          decode 4-bit key → m bf16 values
          dot = sequential sum of key[j] * q[j]
          res += dot * weight[i]
      res *= q_norm
    """
    bs, kv_heads, cand_len = coarse_indices.shape
    device = coarse_indices.device

    # Gather keys for candidate tokens: [bs, kv_heads, B, cand_len, m//2]
    idx_expand = coarse_indices[:, :, None, :, None].expand(-1, -1, B, -1, m // 2)
    gathered = torch.gather(key_4bit, 3, idx_expand)

    # Decode 4-bit → bf16 values
    even = (gathered & 0x0F).long()
    odd = ((gathered >> 4) & 0x0F).long()
    even_val = torch.where(
        (even & 0x8).bool(), mag_centers[even & 0x7], -mag_centers[even & 0x7]
    ).to(torch.bfloat16)
    odd_val = torch.where(
        (odd & 0x8).bool(), mag_centers[odd & 0x7], -mag_centers[odd & 0x7]
    ).to(torch.bfloat16)
    # Interleave even/odd → [bs, kv_heads, B, cand_len, m]
    decoded = torch.stack([even_val, odd_val], dim=-1).reshape(
        bs, kv_heads, B, cand_len, m
    )

    # q_blocks: [bs, kv_heads, B, m]
    q_exp = q_blocks.unsqueeze(3)  # [bs, kv_heads, B, 1, m]
    products = decoded * q_exp     # [bs, kv_heads, B, cand_len, m]

    # Sequential dot product (matching CUDA bf16 accumulation order)
    dot = torch.zeros(bs, kv_heads, B, cand_len, dtype=torch.bfloat16, device=device)
    for j in range(m):
        dot = dot + products[..., j]

    # Gather block weights: [bs, kv_heads, B, cand_len]
    widx = coarse_indices[:, :, None, :].expand(-1, -1, B, -1)
    weights = torch.gather(key_block_weight, 3, widx)

    # Sequential accumulation across blocks (matching CUDA order)
    weighted = dot * weights
    res = torch.zeros(bs, kv_heads, cand_len, dtype=torch.bfloat16, device=device)
    for blk in range(B):
        res = res + weighted[:, :, blk]

    return res * q_norm.unsqueeze(-1)


# ── Reference top-k ───────────────────────────────────────────────────────

def reference_topk_from_scores(scores, coarse_indices, topk):
    """Group scores into chunks of CHUNK_SIZE, return per-chunk top-k."""
    bs, kv_heads, cand_len = scores.shape
    num_chunks = cand_len // CHUNK_SIZE
    chunked = scores.reshape(bs, kv_heads, num_chunks, CHUNK_SIZE)
    tk_scores, tk_local = torch.topk(chunked, k=topk, dim=-1, sorted=True)

    chunk_offsets = (
        torch.arange(num_chunks, device=scores.device).view(1, 1, -1, 1) * CHUNK_SIZE
    )
    global_pos = tk_local + chunk_offsets
    tk_indices = torch.gather(
        coarse_indices.unsqueeze(2).expand(-1, -1, num_chunks, cand_len),
        3, global_pos,
    )
    return tk_scores, tk_indices


def get_rerank_cu_scores(coarse_indices, key_4bit, key_block_weight,
                         q_blocks, q_norm, mag_centers, B, m, rerank_kernel):
    """Get full scores from rerank.cu (scores-only path)."""
    bs, kv_heads, _ = coarse_indices.shape
    return partial_rerank_scores_cuda(
        coarse_indices.contiguous(), key_4bit, key_block_weight,
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(), B, m,
        kernel=rerank_kernel,
    )


# ── CUDA topk kernel runner ──────────────────────────────────────────────

def run_cuda_topk(coarse_indices, key_4bit, key_block_weight, q_blocks, q_norm,
                  mag_centers, B, m, topk, topk_kernel):
    bs, kv_heads, candidate_len = coarse_indices.shape
    num_chunks = candidate_len // CHUNK_SIZE

    topk_scores = torch.empty(
        bs, kv_heads, num_chunks, topk, dtype=torch.bfloat16,
        device=coarse_indices.device,
    )
    topk_indices = torch.empty(
        bs, kv_heads, num_chunks, topk, dtype=torch.int64,
        device=coarse_indices.device,
    )

    topk_kernel.partial_rerank_topk_scores(
        coarse_indices.contiguous(),
        key_4bit,
        key_block_weight.contiguous(),
        q_blocks.reshape(-1, B, m).contiguous(),
        q_norm.reshape(-1).contiguous(),
        mag_centers.contiguous(),
        B, m,
        topk_scores,
        topk_indices,
        topk,
    )
    torch.cuda.synchronize()
    return topk_scores, topk_indices


# ── Tests ─────────────────────────────────────────────────────────────────

def test_python_ref_vs_rerank_cu(device, rerank_kernel, topk_kernel):
    """Sanity: Python reference scores should match rerank.cu exactly."""
    bs, kv_heads, B, m = 1, 2, 16, 8
    kv_len, candidate_len = 256, 64

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device)
    ci, k4, kw, qb, qn, mc = inputs

    py_scores = python_rabitq_scores(ci, k4, kw, qb, qn, mc, B, m)
    cu_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)

    match = torch.equal(py_scores, cu_scores)
    if not match:
        diff = (py_scores.float() - cu_scores.float()).abs()
        n = (py_scores != cu_scores).sum().item()
        print(f"  {n}/{py_scores.numel()} differ, max_diff={diff.max().item():.6f}")

    print(f"[test_python_ref_vs_rerank_cu] {'PASSED' if match else 'FAILED'}")
    assert match


def test_topk_8_scores(device, rerank_kernel, topk_kernel):
    """chunk=32 find top-8: kernel scores match reference."""
    bs, kv_heads, B, m = 1, 4, 16, 8
    kv_len, candidate_len, topk = 512, 128, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, _ = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, _ = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    match = torch.equal(ref_sorted, cuda_sorted)

    if not match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        n = (ref_sorted != cuda_sorted).sum().item()
        print(f"  {n} score values differ, max_diff={diff.max().item()}")

    print(f"[test_topk_8_scores] {'PASSED' if match else 'FAILED'}")
    assert match


def test_topk_8_indices(device, rerank_kernel, topk_kernel):
    """chunk=32 find top-8: kernel selects same candidates as reference."""
    bs, kv_heads, B, m = 1, 4, 16, 8
    kv_len, candidate_len, topk = 512, 128, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    _, ref_indices = reference_topk_from_scores(cu_all_scores, ci, topk)
    _, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    num_chunks = candidate_len // CHUNK_SIZE
    mismatches = 0
    for h in range(kv_heads):
        for c in range(num_chunks):
            ref_set = set(ref_indices[0, h, c].tolist())
            cuda_set = set(cuda_indices[0, h, c].tolist())
            if ref_set != cuda_set:
                mismatches += 1
                if mismatches <= 3:
                    print(f"    H{h} C{c}: ref={sorted(ref_set)} cuda={sorted(cuda_set)}")

    total = kv_heads * num_chunks
    ok = mismatches == 0
    print(f"[test_topk_8_indices] mismatches={mismatches}/{total} {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_topk_1(device, rerank_kernel, topk_kernel):
    """Top-1 per chunk of 32: should return argmax."""
    bs, kv_heads, B, m = 1, 2, 16, 8
    kv_len, candidate_len, topk = 256, 64, 1

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=77)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, ref_indices = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    score_match = torch.equal(ref_scores, cuda_scores)
    idx_match = torch.equal(ref_indices, cuda_indices)

    if not score_match:
        diff = (ref_scores.float() - cuda_scores.float()).abs()
        print(f"  max score diff = {diff.max().item()}")
    if not idx_match:
        n = (ref_indices != cuda_indices).sum().item()
        print(f"  {n} indices differ")

    ok = score_match and idx_match
    print(f"[test_topk_1] {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_topk_4(device, rerank_kernel, topk_kernel):
    """Top-4 per chunk of 32."""
    bs, kv_heads, B, m = 1, 2, 16, 8
    kv_len, candidate_len, topk = 256, 64, 4

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=55)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, _ = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    match = torch.equal(ref_sorted, cuda_sorted)

    if not match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        n = (ref_sorted != cuda_sorted).sum().item()
        print(f"  {n} score values differ, max_diff={diff.max().item()}")

    print(f"[test_topk_4] {'PASSED' if match else 'FAILED'}")
    assert match


def test_topk_8_large(device, rerank_kernel, topk_kernel):
    """Large candidate set (1024 tokens = 32 chunks), multi-block."""
    bs, kv_heads, B, m = 1, 4, 16, 8
    kv_len, candidate_len, topk = 4096, 1024, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=999)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, ref_indices = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    score_match = torch.equal(ref_sorted, cuda_sorted)

    num_chunks = candidate_len // CHUNK_SIZE
    idx_mismatches = 0
    for h in range(kv_heads):
        for c in range(num_chunks):
            if set(ref_indices[0, h, c].tolist()) != set(cuda_indices[0, h, c].tolist()):
                idx_mismatches += 1

    ok = score_match and idx_mismatches == 0
    if not score_match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        n = (ref_sorted != cuda_sorted).sum().item()
        print(f"  {n} score values differ, max_diff={diff.max().item()}")
    if idx_mismatches:
        total = kv_heads * num_chunks
        print(f"  {idx_mismatches}/{total} chunks have different indices")

    print(f"[test_topk_8_large] {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_many_heads(device, rerank_kernel, topk_kernel):
    """8 kv_heads to stress-test flat_bh indexing (bs=1)."""
    bs, kv_heads, B, m = 1, 8, 16, 8
    kv_len, candidate_len, topk = 1024, 256, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=2024)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, ref_indices = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    score_match = torch.equal(ref_sorted, cuda_sorted)

    num_chunks = candidate_len // CHUNK_SIZE
    idx_mismatches = 0
    for h in range(kv_heads):
        for c in range(num_chunks):
            if set(ref_indices[0, h, c].tolist()) != set(cuda_indices[0, h, c].tolist()):
                idx_mismatches += 1

    ok = score_match and idx_mismatches == 0
    if not score_match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        n = (ref_sorted != cuda_sorted).sum().item()
        print(f"  {n} values differ, max_diff={diff.max().item()}")
    if idx_mismatches:
        total = kv_heads * num_chunks
        print(f"  {idx_mismatches}/{total} chunks have different indices")

    print(f"[test_many_heads] {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_single_chunk(device, rerank_kernel, topk_kernel):
    """Edge case: exactly 1 chunk (32 tokens), top-8."""
    bs, kv_heads, B, m = 1, 2, 16, 8
    kv_len, candidate_len, topk = 128, 32, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=314)
    ci, k4, kw, qb, qn, mc = inputs

    cu_all_scores = get_rerank_cu_scores(ci, k4, kw, qb, qn, mc, B, m, rerank_kernel)
    ref_scores, ref_indices = reference_topk_from_scores(cu_all_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    score_match = torch.equal(ref_sorted, cuda_sorted)

    idx_match = True
    for h in range(kv_heads):
        if set(ref_indices[0, h, 0].tolist()) != set(cuda_indices[0, h, 0].tolist()):
            idx_match = False

    ok = score_match and idx_match
    if not score_match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        print(f"  score diff: max={diff.max().item()}")
    if not idx_match:
        print(f"  index mismatch in single chunk")

    print(f"[test_single_chunk] {'PASSED' if ok else 'FAILED'}")
    assert ok


def test_python_ref_topk(device, rerank_kernel, topk_kernel):
    """End-to-end: kernel top-8 matches Python reference top-8 (no rerank.cu)."""
    bs, kv_heads, B, m = 1, 2, 16, 8
    kv_len, candidate_len, topk = 256, 64, 8

    inputs = make_inputs(bs, kv_heads, B, m, kv_len, candidate_len, device, seed=100)
    ci, k4, kw, qb, qn, mc = inputs

    py_scores = python_rabitq_scores(ci, k4, kw, qb, qn, mc, B, m)
    ref_scores, ref_indices = reference_topk_from_scores(py_scores, ci, topk)
    cuda_scores, cuda_indices = run_cuda_topk(ci, k4, kw, qb, qn, mc, B, m, topk, topk_kernel)

    ref_sorted = ref_scores.sort(dim=-1, descending=True).values
    cuda_sorted = cuda_scores.sort(dim=-1, descending=True).values
    score_match = torch.equal(ref_sorted, cuda_sorted)

    num_chunks = candidate_len // CHUNK_SIZE
    idx_mismatches = 0
    for h in range(kv_heads):
        for c in range(num_chunks):
            if set(ref_indices[0, h, c].tolist()) != set(cuda_indices[0, h, c].tolist()):
                idx_mismatches += 1

    ok = score_match and idx_mismatches == 0
    if not score_match:
        diff = (ref_sorted.float() - cuda_sorted.float()).abs()
        n = (ref_sorted != cuda_sorted).sum().item()
        print(f"  {n} score values differ, max_diff={diff.max().item()}")
    if idx_mismatches:
        print(f"  {idx_mismatches}/{kv_heads * num_chunks} chunks have different indices")

    print(f"[test_python_ref_topk] {'PASSED' if ok else 'FAILED'}")
    assert ok


# ── main ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Using device: {device}")
    print(f"CHUNK_SIZE={CHUNK_SIZE}")
    print("Compiling kernels …")
    rerank_kernel = load_kernel_module("rerank.cu", "rerank")
    topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
    print("Kernels compiled.\n")

    tests = [
        test_python_ref_vs_rerank_cu,
        test_topk_8_scores,
        test_topk_8_indices,
        test_topk_1,
        test_topk_4,
        test_topk_8_large,
        test_many_heads,
        test_single_chunk,
        test_python_ref_topk,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t(device, rerank_kernel, topk_kernel)
            passed += 1
        except AssertionError as e:
            failed += 1
            print(f"  !! {t.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"  !! {t.__name__} ERROR: {e}")

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
