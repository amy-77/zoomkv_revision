"""
Quick integration test: verify float_topk_3d can be imported and used
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
from cache_hub.topk.float_topk_ops import float_topk_3d

print("Testing float_topk_3d integration...")

# Simulate Quest chunk scores shape
bs, kv_heads, n_chunks = 1, 8, 4743
k = 474

torch.manual_seed(42)
chunk_scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')

# Test the function
chunk_idx = float_topk_3d(chunk_scores, k=k)

print(f"Input shape: {chunk_scores.shape}")
print(f"Output shape: {chunk_idx.shape}")
print(f"Output dtype: {chunk_idx.dtype}")

# Verify correctness
_, ref_idx = torch.topk(chunk_scores, k=k, dim=-1, largest=True)

overlap = 0
for b in range(bs):
    for h in range(kv_heads):
        ref_set = set(ref_idx[b, h].cpu().tolist())
        test_set = set(chunk_idx[b, h].cpu().tolist())
        overlap += len(ref_set & test_set)

total = bs * kv_heads * k
print(f"Overlap with torch.topk: {overlap}/{total} ({overlap/total*100:.1f}%)")

if overlap / total >= 0.99:
    print("✓ Integration test passed!")
else:
    print("✗ Integration test failed!")
