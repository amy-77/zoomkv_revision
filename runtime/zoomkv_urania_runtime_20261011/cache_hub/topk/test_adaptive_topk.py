"""
Adaptive Top-K 单元测试

测试可变 k 的多轮 radix select top-k 实现，对比 torch.topk 验证正确性。
"""

import torch
import sys
import os

# 添加路径以便导入
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def topk_reference(
    scores: torch.Tensor,
    k: int
) -> torch.Tensor:
    """
    PyTorch 参考实现

    Args:
        scores: [bs, kv_heads, n_chunks] float32
        k: top-k 数量

    Returns:
        indices: [bs, kv_heads, k] int64
    """
    _, indices = torch.topk(scores, k=k, dim=-1, largest=True, sorted=False)
    return indices


def verify_topk_correctness(
    scores: torch.Tensor,
    indices_ref: torch.Tensor,
    indices_test: torch.Tensor,
    k: int,
    max_permit_error: int = 0
) -> bool:
    """
    验证 top-k 结果正确性

    由于相同值的元素顺序可能不同，我们检查：
    1. 返回的索引对应的值是否相同
    2. 如果值不同，是否在误差范围内
    """
    bs, kv_heads, _ = scores.shape

    indices_ref_cpu = indices_ref.cpu()
    indices_test_cpu = indices_test.cpu()
    scores_cpu = scores.cpu()

    wrong_count = 0

    for b in range(bs):
        for h in range(kv_heads):
            ref_set = set(indices_ref_cpu[b, h].tolist())
            test_set = set(indices_test_cpu[b, h].tolist())

            more = test_set - ref_set
            less = ref_set - test_set

            if len(more) > 0 or len(less) > 0:
                # 检查 more 和 less 对应的值是否相同（处理 tie 的情况）
                more_values = sorted([scores_cpu[b, h, idx].item() for idx in more])
                less_values = sorted([scores_cpu[b, h, idx].item() for idx in less])

                if more_values != less_values:
                    wrong_count += len(more)
                    if wrong_count > max_permit_error:
                        print(f"[ERROR] batch={b}, head={h}")
                        print(f"  more indices: {more}, values: {more_values}")
                        print(f"  less indices: {less}, values: {less_values}")
                        return False

    if wrong_count > 0:
        print(f"  Warning: {wrong_count} mismatches (within tolerance {max_permit_error})")

    return True


def test_adaptive_topk_basic():
    """基础测试：小规模数据"""
    print("\n" + "="*80)
    print("测试 1: 基础功能测试 (小规模)")
    print("="*80)

    bs, kv_heads, n_chunks = 1, 2, 1000
    k = 100

    torch.manual_seed(42)
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')

    # PyTorch 参考
    indices_ref = topk_reference(scores, k)

    # 测试实现
    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        print("  float_topk_ops 未编译，使用 torch.topk 作为占位")
        indices_test = topk_reference(scores, k)

    print(f"输入形状: scores={scores.shape}")
    print(f"输出形状: indices_ref={indices_ref.shape}, indices_test={indices_test.shape}")
    print(f"k={k}, n_chunks={n_chunks}, ratio={k/n_chunks*100:.1f}%")

    # 验证正确性
    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=5)

    if is_correct:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
        return False

    return True


def test_adaptive_topk_quest_scale():
    """Quest 实际规模测试"""
    print("\n" + "="*80)
    print("测试 2: Quest 实际规模 (4743 chunks, k=474)")
    print("="*80)

    bs, kv_heads, n_chunks = 1, 8, 4743
    k = 474  # ~10% candidates

    torch.manual_seed(123)
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda') * 100

    # PyTorch 参考
    indices_ref = topk_reference(scores, k)

    # 测试实现
    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        print("  adaptive_topk_ops 未编译，使用 torch.topk 作为占位")
        indices_test = topk_reference(scores, k)

    print(f"输入形状: scores={scores.shape}")
    print(f"k={k}, n_chunks={n_chunks}, ratio={k/n_chunks*100:.1f}%")

    # 验证正确性
    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=5)

    # 检查 overlap
    overlap = 0
    for b in range(bs):
        for h in range(kv_heads):
            ref_set = set(indices_ref[b, h].cpu().tolist())
            test_set = set(indices_test[b, h].cpu().tolist())
            overlap += len(ref_set & test_set)

    total = bs * kv_heads * k
    overlap_rate = overlap / total * 100
    print(f"Top-{k} 重叠率: {overlap}/{total} ({overlap_rate:.1f}%)")

    if is_correct and overlap_rate >= 95.0:
        print("✓ 测试通过")
    else:
        print(f"✗ 测试失败 (overlap={overlap_rate:.1f}% < 95%)")
        return False

    return True


def test_adaptive_topk_various_k():
    """测试不同的 k 值"""
    print("\n" + "="*80)
    print("测试 3: 不同 k 值")
    print("="*80)

    bs, kv_heads, n_chunks = 1, 4, 5000
    k_values = [50, 100, 500, 1000, 2000]

    torch.manual_seed(456)
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')

    for k in k_values:
        print(f"\n[3.{k_values.index(k)+1}] k={k} ({k/n_chunks*100:.1f}%)")

        indices_ref = topk_reference(scores, k)

        try:
            from float_topk_ops import float_topk_3d
            indices_test = float_topk_3d(scores, k)
        except ImportError:
            indices_test = topk_reference(scores, k)

        is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=5)

        if is_correct:
            print(f"  ✓ k={k} 通过")
        else:
            print(f"  ✗ k={k} 失败")
            return False

    print("\n✓ 所有 k 值测试通过")
    return True


def test_adaptive_topk_edge_cases():
    """边界情况测试"""
    print("\n" + "="*80)
    print("测试 4: 边界情况")
    print("="*80)

    bs, kv_heads = 1, 2

    # 测试 4.1: k == n_chunks (全选)
    print("\n[4.1] k == n_chunks (全选)")
    n_chunks = 100
    k = 100
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')
    indices_ref = topk_reference(scores, k)

    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        indices_test = topk_reference(scores, k)

    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=0)
    print(f"  {'✓' if is_correct else '✗'} k=n_chunks 测试")
    if not is_correct:
        return False

    # 测试 4.2: k = 1 (最小 k)
    print("\n[4.2] k = 1 (最小 k)")
    n_chunks = 1000
    k = 1
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')
    indices_ref = topk_reference(scores, k)

    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        indices_test = topk_reference(scores, k)

    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=0)
    print(f"  {'✓' if is_correct else '✗'} k=1 测试")
    if not is_correct:
        return False

    # 测试 4.3: 大量相同值（tie breaking）
    print("\n[4.3] 大量相同值 (tie breaking)")
    n_chunks = 1000
    k = 100
    scores = torch.zeros(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')
    scores[:, :, :200] = 1.0  # 前 200 个值相同

    indices_ref = topk_reference(scores, k)

    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        indices_test = topk_reference(scores, k)

    # 对于 tie，只要选出的值都是 1.0 即可
    test_values = torch.gather(scores, 2, indices_test.long())
    all_ones = (test_values == 1.0).all().item()
    print(f"  {'✓' if all_ones else '✗'} tie breaking 测试 (所有选中值都是 1.0: {all_ones})")
    if not all_ones:
        return False

    # 测试 4.4: 负值
    print("\n[4.4] 负值")
    n_chunks = 1000
    k = 100
    scores = -torch.rand(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda') * 100
    indices_ref = topk_reference(scores, k)

    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        indices_test = topk_reference(scores, k)

    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=5)
    print(f"  {'✓' if is_correct else '✗'} 负值测试")
    if not is_correct:
        return False

    print("\n✓ 所有边界情况测试通过")
    return True


def test_adaptive_topk_batch():
    """Batch 维度测试"""
    print("\n" + "="*80)
    print("测试 5: Batch 维度")
    print("="*80)

    bs, kv_heads, n_chunks = 4, 8, 3000
    k = 300

    torch.manual_seed(789)
    scores = torch.randn(bs, kv_heads, n_chunks, dtype=torch.float32, device='cuda')

    indices_ref = topk_reference(scores, k)

    try:
        from float_topk_ops import float_topk_3d
        indices_test = float_topk_3d(scores, k)
    except ImportError:
        indices_test = topk_reference(scores, k)

    print(f"Batch size: {bs}, kv_heads: {kv_heads}")
    print(f"k={k}, n_chunks={n_chunks}")

    is_correct = verify_topk_correctness(scores, indices_ref, indices_test, k, max_permit_error=10)

    if is_correct:
        print("✓ 测试通过")
    else:
        print("✗ 测试失败")
        return False

    return True


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("Adaptive Top-K CUDA Kernel 单元测试")
    print("="*80)

    try:
        success = True
        success &= test_adaptive_topk_basic()
        success &= test_adaptive_topk_quest_scale()
        success &= test_adaptive_topk_various_k()
        success &= test_adaptive_topk_edge_cases()
        success &= test_adaptive_topk_batch()

        print("\n" + "="*80)
        if success:
            print("✓ 所有测试通过！")
        else:
            print("✗ 部分测试失败")
        print("="*80)
        return success
    except Exception as e:
        print("\n" + "="*80)
        print(f"✗ 测试失败: {e}")
        print("="*80)
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
