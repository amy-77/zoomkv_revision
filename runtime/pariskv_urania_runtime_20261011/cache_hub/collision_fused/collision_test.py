import argparse
import sys
import os
import torch
import triton
import numpy as np
import math

# 支持直接运行: python collision_test.py 或 python -m cache_hub.collision_fused.collision_test
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
try:
    from .collison_interface import update_cache_cnt_cuda_interface
except ImportError:
    from cache_hub.collision_fused.collison_interface import update_cache_cnt_cuda_interface

def generate_test_input(
    bs=1,
    kv_heads=8,
    B=16,
    num_clusters=256,
    kv_len=2048,
    collision_ratio=0.2,  # 新增：碰撞比例参数
    tier_ratios=None      # 新增：层级比例参数
):
    """
    生成update_cache_cnt_overall_ref函数的完整测试输入数据
    
    Args:
        bs: batch size
        kv_heads: kv头数
        B: 子空间数量
        num_clusters: 聚类数量
        kv_len: key/value序列长度
        collision_ratio: 碰撞比例（用于计算目标key数量）
        tier_ratios: 层级比例列表，默认使用[0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
    
    Returns:
        sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
        cluster_key_counts: [bs, kv_heads, B, num_clusters] 每个cluster包含的key数量
        codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
        tier_ratios: 层级比例列表
        kv_len: key/value序列长度（方便外部使用）
        collision_ratio: 碰撞比例（方便外部使用）
    """
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. 生成 sorted_cluster_ids: [bs, kv_heads, B, num_clusters]
    sorted_cluster_ids = torch.stack([
        torch.stack([
            torch.stack([
                torch.randperm(num_clusters) for _ in range(B)
            ]) for _ in range(kv_heads)
        ]) for _ in range(bs)
    ]).to(torch.int32)

    # 2. 生成 cluster_key_counts: [bs, kv_heads, B, num_clusters]
    # 每个cluster包含的key数量，范围在 [1, kv_len//2]（合理的数值范围）
    cluster_key_counts = torch.randint(1, max(2, kv_len//2), 
                                     (bs, kv_heads, B, num_clusters), 
                                     dtype=torch.int32)
    
    # 3. 生成 codebook: [bs, kv_heads, B, kv_len]
    codebook = torch.randint(0, num_clusters, (bs, kv_heads, B, kv_len), dtype=torch.int32)
    
    # 4. 设置 tier_ratios（默认值与原函数一致）
    if tier_ratios is None:
        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        # tier_ratios = [1.0, ]

    # 5. 移动到CUDA（如果可用）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sorted_cluster_ids = sorted_cluster_ids.to(device)
    cluster_key_counts = cluster_key_counts.to(device)
    codebook = codebook.to(device)

    return (
        sorted_cluster_ids,
        cluster_key_counts,
        codebook,
        tier_ratios,
        kv_len,          # 返回供外部计算target_keys使用
        collision_ratio  # 返回供外部计算target_keys使用
    )

# 修复原update_cache_cnt_overall_ref函数中的错误
def update_cache_cnt_overall_ref(
    sorted_cluster_ids, 
    cluster_key_counts, 
    codebook,
    tier_ratios,
    kv_len,          # 新增参数
    collision_ratio  # 新增参数
):
    """
    修复后的完整函数：更新每个cache的计数器（多层级版本）
    """
    # 根据sorted_cluster_ids重排cluster_key_counts 
    # 根据排序后进行gather
    sorted_key_counts = torch.gather(cluster_key_counts, dim=-1, index=sorted_cluster_ids.long())
    # 前缀和
    cumsum_counts = torch.cumsum(sorted_key_counts, dim=-1)  # [bs, kv_heads, B, num_clusters]

    before_cumsum = cumsum_counts - sorted_key_counts  # [bs, kv_heads, B, num_clusters]
    
    # 移除原函数中重复定义的tier_ratios，使用传入的参数
    # tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
    cache_cnt = None
    for ratio in tier_ratios:
        # 计算目标key数量（修复变量未定义问题）
        target_keys = math.ceil(kv_len * collision_ratio * ratio)
        valid_mask = before_cumsum < target_keys
        topk_cluster_cnt = valid_mask.sum(dim=-1)
        if cache_cnt is None:
            cache_cnt = update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook)
        else:
            # 修复语法错误：cache_cnt+ = → cache_cnt +=
            cache_cnt += update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook)

    return cache_cnt

def update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook):
    """
    Step 3: 更新每个cache的计数器（碰撞次数）- 优化版本
    Args:
        sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
        topk_cluster_cnt: [bs, kv_heads, B] 每个子空间选择的cluster数量
        codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
        没有[cluster-->key id: 每个cluster对应的key的id]
    Returns:
        cache_cnt: [bs, kv_heads, kv_len] 每个key在所有子空间的累积碰撞次数（累积所有子空间）
    """
    bs, kv_heads, B, kv_len = codebook.shape
    num_clusters = sorted_cluster_ids.shape[-1]
    # 初始化碰撞计数器
    cache_cnt = torch.zeros(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    # cluster_range: [num_clusters] = [0, 1, 2, ..., num_clusters-1]
    cluster_range = torch.arange(num_clusters, device=codebook.device)  # [num_clusters]=[64]
    cluster_range_expanded = cluster_range.view(1, 1, 1, -1).expand(bs, kv_heads, B, -1)
    # 扩展维度用于广播: cluster_range_expanded: [bs, kv_heads, B, num_clusters]=[1, 8, 8, 64] 
    topk_cnt_expanded = topk_cluster_cnt.unsqueeze(-1)  # topk_cnt_expanded: [bs, kv_heads, B, 1]
    # valid_cluster_mask: [bs, kv_heads, B, num_clusters]
    # True表示该cluster ID在选中范围内
    valid_cluster_mask = cluster_range_expanded < topk_cnt_expanded
    # 获取选中的cluster IDs，并将无效的设为-1（这样不会匹配任何codebook）
    # 使用clone+mask赋值（可能比torch.where更快，因为避免了每次创建tensor的开销）
    selected_clusters = sorted_cluster_ids.clone()  # [bs, kv_heads, B, num_clusters]
    selected_clusters[~valid_cluster_mask] = -1
    # 向量化比较：对于每个子空间B，检查codebook中的每个key是否在selected_clusters中
    # codebook: [bs, kv_heads, B, kv_len]
    # selected_clusters: [bs, kv_heads, B, num_clusters]
    # 扩展维度用于广播 codebook_expanded: [bs, kv_heads, B, kv_len, 1]
    
    # 扩展维度用于广播 selected_clusters_expanded: [bs, kv_heads, B, 1, num_clusters]
    codebook_expanded = codebook.unsqueeze(-1)  # [bs, kv_heads, B, kv_len, 1]=[1, 8, 8, 165, 1]  
    # 每个cluster存哪些key, [bs,kv_heads=8, B=8, num_clusters=64, kv_len=165, 1]
    selected_clusters_expanded = selected_clusters.unsqueeze(-2)  # [bs, kv_heads, B, 1, num_clusters]=[1, 8, 8, 1, 64]
    # matches: [bs, kv_heads, B, kv_len, num_clusters]=[1, 8, 8, 165, 64]
    # matches[b, h, sub, key, cluster_id] = True if codebook[bs,h,b,key] == selected_clusters[bs,h,b,cluster]
    matches = (codebook_expanded == selected_clusters_expanded) 
    # 对num_clusters维度求和，得到每个key在每个子空间的碰撞次数
    # collisions_per_subspace: [bs, kv_heads, B, kv_len]=[1, 8, 8, 165]
    # kv_len维度存储子空间b的key的碰撞成功值: 每个key所属的cluster与selected_clusters是否碰撞成功，如果成功=1，否则=0
    collisions_per_subspace = matches.sum(dim=-1).int()  
    # 对所有子空间B求和，得到每个key的总碰撞次数
    cache_cnt = collisions_per_subspace.sum(dim=2)  # [bs, kv_heads, kv_len]=[1, 8, 165]        
    return cache_cnt





def update_cache_cnt_cuda(
    sorted_cluster_ids, 
    cluster_key_counts, 
    codebook,
    tier_ratios,
    kv_len,          # 新增参数
    collision_ratio  # 新增参数
):
    tier_nums = [ math.ceil(kv_len * i * collision_ratio) for i in tier_ratios]

    return update_cache_cnt_cuda_interface(
        sorted_cluster_ids, 
        cluster_key_counts, 
        codebook,
        tier_nums
    )

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", type=int, default=1)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--B", type=int, default=16)
    parser.add_argument("--num-clusters", type=int, default=256)
    parser.add_argument("--kv-len", type=int, default=2048)
    parser.add_argument("--collision-ratio", type=float, default=0.2)
    parser.add_argument("--skip-ref", action="store_true")
    return parser.parse_args()

# 测试用例
if __name__ == "__main__":
    args = _parse_args()
    # torch.set_printoptions(threshold=torch.inf)
    # 生成测试输入
    (sorted_cluster_ids, cluster_key_counts, codebook,
     tier_ratios, kv_len, collision_ratio) = generate_test_input(
        bs=args.bs,
        kv_heads=args.kv_heads,
        B=args.B,
        num_clusters=args.num_clusters,
        kv_len=args.kv_len,
        collision_ratio=args.collision_ratio,
    )

    # 调用函数并输出结果形状
    def ref():
        result_ref = update_cache_cnt_overall_ref(
            sorted_cluster_ids, cluster_key_counts, codebook,
            tier_ratios, kv_len, collision_ratio
        )
        return result_ref

    codebook = codebook.to(torch.uint8)
    sorted_cluster_ids = sorted_cluster_ids.to(torch.int64)

    def cuda():
        result_cuda = update_cache_cnt_cuda(
            sorted_cluster_ids, cluster_key_counts, codebook,
            tier_ratios, kv_len, collision_ratio
        )
        return result_cuda
    if not args.skip_ref:
        print("[DEBUG] running ref()", flush=True)
        result_ref = ref()
        print("[DEBUG] ref() done", flush=True)
    print("[DEBUG] running cuda()", flush=True)
    result_cuda = cuda()
    print("[DEBUG] cuda() done", flush=True)

    if not args.skip_ref:
        assert((result_ref - result_cuda).sum() == 0)

    runing_time = triton.testing.do_bench(cuda)
    print(runing_time)

    # print("OKKKKKKKKKKKKKKKKKKKKKKKKKK", runing_time)
    # torch.cuda.synchronize()
    # print(f"输入参数形状：")
    # print(f"  sorted_cluster_ids: {sorted_cluster_ids.shape}")
    # print(f"  cluster_key_counts: {cluster_key_counts.shape}")
    # print(f"  codebook: {codebook.shape}")
    # print(f"输出结果形状：cache_cnt = {result_ref.shape}")
    # print(result_ref.sum(-1), result_ref.size())