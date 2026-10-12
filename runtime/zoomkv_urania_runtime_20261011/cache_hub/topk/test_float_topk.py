"""
Unit test for float_topk.cu — verifies radix-select top-k against torch.topk.

Tests both 2D and 3D interfaces. All tests require 100% set-overlap with
torch.topk (the selected indices must be the same set, order may differ).
"""

import sys
import os
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from topk.float_topk_ops import load_float_topk_module


def check_overlap(input_tensor, cuda_indices, k, label=""):
    """Return (overlap_count, total) comparing cuda_indices vs torch.topk."""
    _, ref_indices = torch.topk(input_tensor, k=k, dim=-1)

    total = 0
    overlap = 0
    if input_tensor.dim() == 2:
        for i in range(input_tensor.size(0)):
            ref_set = set(ref_indices[i].cpu().tolist())
            cuda_set = set(cuda_indices[i].cpu().tolist())
            overlap += len(ref_set & cuda_set)
            total += k
    elif input_tensor.dim() == 3:
        for i in range(input_tensor.size(0)):
            for j in range(input_tensor.size(1)):
                ref_set = set(ref_indices[i, j].cpu().tolist())
                cuda_set = set(cuda_indices[i, j].cpu().tolist())
                overlap += len(ref_set & cuda_set)
                total += k
    return overlap, total


def test_basic(device, kernel):
    """Basic correctness: random floats, various (B, L, k) combos."""
    configs = [
        (1, 1000, 100),
        (4, 2000, 50),
        (1, 512, 10),
        (2, 10000, 500),
    ]
    for B, L, k in configs:
        x = torch.randn(B, L, dtype=torch.float32, device=device)
        indices = kernel.float_topk(x, k)
        overlap, total = check_overlap(x, indices, k)
        ok = overlap == total
        if not ok:
            print(f"  B={B} L={L} k={k}: {overlap}/{total}")
        assert ok, f"basic B={B} L={L} k={k}: overlap {overlap}/{total}"
    print("[test_basic] PASSED")


def test_topk_1(device, kernel):
    """k=1: should return argmax."""
    x = torch.randn(8, 500, dtype=torch.float32, device=device)
    indices = kernel.float_topk(x, 1)
    ref_argmax = x.argmax(dim=-1)
    match = torch.equal(indices.squeeze(-1), ref_argmax)
    if not match:
        n = (indices.squeeze(-1) != ref_argmax).sum().item()
        print(f"  {n}/{x.size(0)} argmax mismatches")
    assert match
    print("[test_topk_1] PASSED")


def test_k_near_length(device, kernel):
    """k close to L."""
    L = 300
    k = 290
    x = torch.randn(2, L, dtype=torch.float32, device=device)
    indices = kernel.float_topk(x, k)
    overlap, total = check_overlap(x, indices, k)
    assert overlap == total, f"k_near_length: {overlap}/{total}"
    print("[test_k_near_length] PASSED")


def test_k_equals_length(device, kernel):
    """k == L: should return all indices."""
    L = 128
    x = torch.randn(2, L, dtype=torch.float32, device=device)
    indices = kernel.float_topk(x, L)
    for i in range(2):
        assert set(indices[i].cpu().tolist()) == set(range(L))
    print("[test_k_equals_length] PASSED")


def test_negative_values(device, kernel):
    """All negative values."""
    x = -torch.rand(4, 1000, dtype=torch.float32, device=device) - 1.0
    indices = kernel.float_topk(x, 50)
    overlap, total = check_overlap(x, indices, 50)
    assert overlap == total, f"negative: {overlap}/{total}"
    print("[test_negative_values] PASSED")


def test_mixed_sign(device, kernel):
    """Mix of large positive and negative values."""
    x = torch.randn(4, 2000, dtype=torch.float32, device=device) * 100
    indices = kernel.float_topk(x, 100)
    overlap, total = check_overlap(x, indices, 100)
    assert overlap == total, f"mixed_sign: {overlap}/{total}"
    print("[test_mixed_sign] PASSED")


def test_large_length(device, kernel):
    """L = 128K — realistic KV cache length."""
    L = 128 * 1024
    k = 100
    x = torch.randn(1, L, dtype=torch.float32, device=device)
    indices = kernel.float_topk(x, k)
    overlap, total = check_overlap(x, indices, k)
    assert overlap == total, f"large_length: {overlap}/{total}"
    print("[test_large_length] PASSED")


def test_3d_interface(device, kernel):
    """3D interface: [bs, kv_heads, L]."""
    bs, kv_heads, L, k = 1, 8, 4096, 64
    x = torch.randn(bs, kv_heads, L, dtype=torch.float32, device=device)
    indices = kernel.float_topk_3d(x, k)
    assert indices.shape == (bs, kv_heads, k)
    overlap, total = check_overlap(x, indices, k)
    assert overlap == total, f"3d: {overlap}/{total}"
    print("[test_3d_interface] PASSED")


def test_bf16_input(device, kernel):
    """bf16 input — kernel should auto-convert to float32."""
    x = torch.randn(2, 1000, dtype=torch.bfloat16, device=device)
    indices = kernel.float_topk(x, 50)
    x_f32 = x.float()
    overlap, total = check_overlap(x_f32, indices, 50)
    assert overlap == total, f"bf16: {overlap}/{total}"
    print("[test_bf16_input] PASSED")


def test_repeated_values(device, kernel):
    """Many repeated values — stresses the radix boundary logic."""
    x = torch.zeros(1, 1000, dtype=torch.float32, device=device)
    x[0, :50] = 1.0
    x[0, 50:100] = 0.5
    indices = kernel.float_topk(x, 50)
    vals = x[0][indices[0]]
    assert (vals == 1.0).all(), "repeated: should select all 1.0 elements"
    print("[test_repeated_values] PASSED")


if __name__ == "__main__":
    device = "cuda:0"
    if len(sys.argv) > 1:
        device = sys.argv[1]

    print(f"Using device: {device}")
    print("Compiling float_topk kernel ...")
    kernel = load_float_topk_module()
    print("Kernel compiled.\n")

    tests = [
        test_basic,
        test_topk_1,
        test_k_near_length,
        test_k_equals_length,
        test_negative_values,
        test_mixed_sign,
        test_large_length,
        test_3d_interface,
        test_bf16_input,
        test_repeated_values,
    ]

    passed = 0
    failed = 0
    for t in tests:
        try:
            t(device, kernel)
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
