"""test_trans_h2d_packed_pathological.py — h2d_gather_kv_packed 病态对照

针对生产 varlen 路径，构造 4 类专项测试：

1) 生产规模 K：bs=1, kv_heads=8, kv_len=16k, 每 head 的 K ∈ [50, 800]，
   多个随机 seed 全部对齐 ref。
2) 极小 K=1 的退化情形（即我们怀疑的 fk=1 case）。
3) 大 sl_end (320=64+256)，验证 kernel 不写入 sink+local 段。
4) bf16 比特一致性（不只是数值近似，src/dst 必须逐字节相同）。

跑法:
    cd /data/qwang/q/thalia/kvcache/ANNCache
    python cache_hub/gather_trans/test_trans_h2d_packed_pathological.py
"""
from __future__ import annotations

import sys
import os

import torch

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from trans_h2d import h2d_gather_kv_packed  # noqa: E402


def _build(bs, kv_heads, kv_len, dim, sl_end, total_k_per_head, seed, device="cuda:0"):
    """构造一个 packed buffer 的输入；total_k_per_head 是每个 (b,h) 的 adaptive_k 列表 [V]."""
    g = torch.Generator(device='cpu').manual_seed(seed)

    src_k = torch.randn(bs, kv_heads, kv_len, dim, dtype=torch.bfloat16,
                        device='cpu', pin_memory=True)
    src_v = torch.randn(bs, kv_heads, kv_len, dim, dtype=torch.bfloat16,
                        device='cpu', pin_memory=True)

    V = bs * kv_heads
    adaptive_k = torch.tensor(total_k_per_head, dtype=torch.int32)
    assert adaptive_k.shape == (V,)
    seqlens_k = adaptive_k + sl_end

    cu_seqlens_topk_cpu = torch.zeros(V + 1, dtype=torch.int32)
    torch.cumsum(adaptive_k, dim=0, out=cu_seqlens_topk_cpu[1:])
    cu_seqlens_k_cpu = torch.zeros(V + 1, dtype=torch.int32)
    torch.cumsum(seqlens_k, dim=0, out=cu_seqlens_k_cpu[1:])

    total_topk = int(cu_seqlens_topk_cpu[-1].item())
    total_k = int(cu_seqlens_k_cpu[-1].item())
    global_k = int(adaptive_k.max().item())

    # 模拟生产：每个 (b, h) 的 indices 从 [0, kv_len) 随机 sample，可能有重复
    indices_packed_cpu = torch.empty(total_topk, dtype=torch.int64)
    for vh in range(V):
        ak = int(adaptive_k[vh].item())
        st = int(cu_seqlens_topk_cpu[vh].item())
        indices_packed_cpu[st:st + ak] = torch.randint(
            low=0, high=kv_len, size=(ak,), generator=g, dtype=torch.int64,
        )

    # dst buffer：先填入「哨兵值」，验证 kernel 不应触碰的位置保持哨兵
    SENTINEL = 7.5
    dst_k = torch.full((total_k, dim), SENTINEL,
                       dtype=torch.bfloat16, device=device)
    dst_v = torch.full((total_k, dim), SENTINEL,
                       dtype=torch.bfloat16, device=device)

    return dict(
        src_k=src_k, src_v=src_v, dst_k=dst_k, dst_v=dst_v,
        indices_packed=indices_packed_cpu.to(device),
        cu_seqlens_topk=cu_seqlens_topk_cpu.to(device),
        cu_seqlens_k=cu_seqlens_k_cpu.to(device),
        sl_end=sl_end, global_k=global_k,
        adaptive_k=adaptive_k, total_k=total_k, total_topk=total_topk,
        bs=bs, kv_heads=kv_heads, V=V, sentinel=SENTINEL,
    )


def _verify(b, label, *, check_sentinel=True):
    """运行 kernel + 跟纯 PyTorch ref 对照. b 是 _build 的输出."""
    h2d_gather_kv_packed(
        b['src_k'], b['src_v'],
        b['dst_k'], b['dst_v'],
        b['indices_packed'],
        b['cu_seqlens_topk'], b['cu_seqlens_k'],
        int(b['sl_end']), int(b['global_k']),
    )
    torch.cuda.synchronize()

    # 构造 ref：把整个 src 搬到 GPU，按 indices_packed 在每段 sl_end 后写
    src_k_cuda = b['src_k'].cuda()
    src_v_cuda = b['src_v'].cuda()
    SENT = b['sentinel']
    ref_k = torch.full_like(b['dst_k'], SENT)
    ref_v = torch.full_like(b['dst_v'], SENT)
    cu_topk = b['cu_seqlens_topk'].cpu()
    cu_k = b['cu_seqlens_k'].cpu()
    ak = b['adaptive_k']
    bs, kv_heads = b['bs'], b['kv_heads']
    sl_end = b['sl_end']
    indices = b['indices_packed']
    for vh in range(b['V']):
        b_idx = vh // kv_heads
        h_idx = vh % kv_heads
        n = int(ak[vh].item())
        if n == 0:
            continue
        st_topk = int(cu_topk[vh].item())
        st_k    = int(cu_k[vh].item()) + sl_end
        idx = indices[st_topk:st_topk + n]
        ref_k[st_k:st_k + n] = src_k_cuda[b_idx, h_idx, idx]
        ref_v[st_k:st_k + n] = src_v_cuda[b_idx, h_idx, idx]

    # 1) 写入区域必须按位等于 ref
    diff_k = (ref_k - b['dst_k']).abs().sum().item()
    diff_v = (ref_v - b['dst_v']).abs().sum().item()
    write_ok = (diff_k == 0 and diff_v == 0)
    print(f"  [{label}] 写入区域逐位相等         : {'PASS' if write_ok else 'FAIL'}"
          f"  (diff_k={diff_k}, diff_v={diff_v})")

    # 2) sl_end 段 (sink+local) 应保持哨兵值（kernel 不应触碰）
    sentinel_ok = True
    if check_sentinel:
        bad = 0
        for vh in range(b['V']):
            st_k = int(cu_k[vh].item())
            sub_k = b['dst_k'][st_k:st_k + sl_end]
            sub_v = b['dst_v'][st_k:st_k + sl_end]
            # bf16 比较 sentinel：先转 fp32 做精确比较
            ok_k = ((sub_k.float() == SENT).all()).item()
            ok_v = ((sub_v.float() == SENT).all()).item()
            if not (ok_k and ok_v):
                bad += 1
        sentinel_ok = (bad == 0)
        print(f"  [{label}] sink+local 段未被触碰     : "
              f"{'PASS' if sentinel_ok else f'FAIL ({bad}/{b[chr(34)+chr(86)+chr(34)] if False else b[chr(86)]} segs touched)'}")
        # ↑ 这行的格式化太挤了，重写一下
        if not sentinel_ok:
            print(f"      ({bad}/{b['V']} 段被错误覆盖)")

    return write_ok and sentinel_ok


def case_production_scale(seeds=(0, 1, 2, 7, 42)):
    """生产规模：bs=1, kv_heads=8, kv_len=16k，每 head K ∈ [50,800]."""
    print("\n" + "=" * 72)
    print(" Case 1: production_scale  (kv_len=16k, K ∈ [50, 800])")
    print("=" * 72)
    bs, kv_heads, kv_len, dim, sl_end = 1, 8, 16384, 128, 320
    V = bs * kv_heads
    g = torch.Generator(device='cpu').manual_seed(0)
    all_ok = True
    for seed in seeds:
        gg = torch.Generator(device='cpu').manual_seed(seed)
        ks = torch.randint(50, 801, (V,), generator=gg, dtype=torch.int32).tolist()
        b = _build(bs, kv_heads, kv_len, dim, sl_end, ks, seed=seed)
        ok = _verify(b, f"prod_seed{seed}")
        all_ok = all_ok and ok
    return all_ok


def case_extreme_k1():
    """所有 head K=1 的极端退化（fk=1 模式）."""
    print("\n" + "=" * 72)
    print(" Case 2: extreme_k1  (every head adaptive_k=1)")
    print("=" * 72)
    bs, kv_heads, kv_len, dim, sl_end = 1, 8, 8192, 128, 320
    V = bs * kv_heads
    ks = [1] * V
    b = _build(bs, kv_heads, kv_len, dim, sl_end, ks, seed=11)
    return _verify(b, "k1_all")


def case_mixed_k():
    """K 分布跨度极大：有的 head=1，有的 head=800."""
    print("\n" + "=" * 72)
    print(" Case 3: mixed_k  (K mix of 1, 50, 200, 500, 800)")
    print("=" * 72)
    bs, kv_heads, kv_len, dim, sl_end = 1, 8, 16384, 128, 320
    ks = [1, 50, 100, 200, 400, 600, 800, 50]
    b = _build(bs, kv_heads, kv_len, dim, sl_end, ks, seed=99)
    return _verify(b, "mixed")


def case_bs2_long():
    """bs=2，kv_heads=8，K ∈ [100, 600]，确认 batch 维寻址 OK."""
    print("\n" + "=" * 72)
    print(" Case 4: bs2_long  (bs=2, kv_heads=8, K ∈ [100, 600])")
    print("=" * 72)
    bs, kv_heads, kv_len, dim, sl_end = 2, 8, 16384, 128, 320
    V = bs * kv_heads
    g = torch.Generator(device='cpu').manual_seed(123)
    ks = torch.randint(100, 601, (V,), generator=g, dtype=torch.int32).tolist()
    b = _build(bs, kv_heads, kv_len, dim, sl_end, ks, seed=123)
    return _verify(b, "bs2")


def case_zero_k():
    """部分 head adaptive_k=0（写入区域为空）."""
    print("\n" + "=" * 72)
    print(" Case 5: zero_k  (部分 head adaptive_k=0)")
    print("=" * 72)
    bs, kv_heads, kv_len, dim, sl_end = 1, 8, 8192, 128, 320
    ks = [0, 50, 0, 200, 0, 100, 0, 50]
    b = _build(bs, kv_heads, kv_len, dim, sl_end, ks, seed=314)
    return _verify(b, "zero_k")


def main():
    if not torch.cuda.is_available():
        print("CUDA not available")
        return 1

    print("=" * 72)
    print(" h2d_gather_kv_packed pathological correctness")
    print("=" * 72)
    results = [
        case_production_scale(),
        case_extreme_k1(),
        case_mixed_k(),
        case_bs2_long(),
        case_zero_k(),
    ]
    print()
    print("=" * 72)
    if all(results):
        print("  >>> ALL h2d_gather_kv_packed checks PASSED <<<")
        return 0
    else:
        print("  >>> SOME h2d_gather_kv_packed checks FAILED <<<")
        return 1


if __name__ == "__main__":
    sys.exit(main())
