from typing import Any

import torch
import numpy as np
import triton

import os
from functools import lru_cache
from typing import Any, Iterable

def generate_test_input(
    bs, 
    kv_heads, 
    B, 
    num_clusters, 
    kv_len
):
    """
    生成测试输入数据
    """
    # 设置随机种子以保证可重复性
    torch.manual_seed(42)
    np.random.seed(42)

    # 生成 sorted_cluster_ids: [bs, kv_heads, B, num_clusters]
    # 使用随机排列生成排序后的cluster IDs
    sorted_cluster_ids = torch.stack([
        torch.stack([
            torch.stack([
                torch.randperm(num_clusters) for _ in range(B)
            ]) for _ in range(kv_heads)
        ]) for _ in range(bs)
    ]).to(torch.int32)
    
    # 生成 topk_cluster_cnt: [bs, kv_heads, B]
    # 每个子空间选择的cluster数量，范围在 [1, num_clusters-1]
    topk_cluster_cnt = torch.randint(1, num_clusters, (bs, kv_heads, B), dtype=torch.int32)
    # 生成 codebook: [bs, kv_heads, B, kv_len]
    # 每个key在每个子空间的cluster ID，范围在 [0, num_clusters-1]
    codebook = torch.randint(0, num_clusters, (bs, kv_heads, B, kv_len), dtype=torch.int32)
    return sorted_cluster_ids.cuda(), topk_cluster_cnt.cuda(), codebook.cuda()



def update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook):
    """
    Step 3: 更新每个cache的计数器（碰撞次数）- 优化版本（使用Triton kernel）
    Args:
        sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
        topk_cluster_cnt: [bs, kv_heads, B] 每个子空间选择的cluster数量
        codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
        没有[cluster-->key id: 每个cluster对应的key的id]
    Returns:
        cache_cnt: [bs, kv_heads, kv_len] 每个key的碰撞次数（累积所有子空间）
    """
    # Use original PyTorch implementation (CUDA kernels disabled)
    # Get data dimensions
    bs, kv_heads, B, kv_len = codebook.shape
    num_clusters = sorted_cluster_ids.shape[-1]
    # 初始化碰撞计数器
    cache_cnt = torch.zeros(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    # cluster_range: [num_clusters] = [0, 1, 2, ..., num_clusters-1]
    cluster_range = torch.arange(num_clusters, device=codebook.device)  # [num_clusters]=[64]
    # 扩展维度用于广播: [bs, kv_heads, B, num_clusters]=[1, 8, 8, 64], 
    cluster_range_expanded = cluster_range.view(1, 1, 1, -1).expand(bs, kv_heads, B, -1)
    topk_cnt_expanded = topk_cluster_cnt.unsqueeze(-1)  # [bs, kv_heads, B, 1]
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






@lru_cache()
def _prepare_for_load() -> str:
    import os
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module="torch.utils.cpp_extension"
    )
    return os.path.dirname(os.path.abspath(__file__))


@lru_cache()
def load_kernel_module(
    path: str | Iterable[str],
    name: str,
    *,
    build: str = "build",
    cflags: Iterable[str] | None = None,
    cuda_flags: Iterable[str] | None = None,
    ldflags: Iterable[str] | None = None,
) -> Any:
    from torch.utils.cpp_extension import load

    if isinstance(path, str):
        path = (path,)

    abs_path = _prepare_for_load()
    build_dir = f"{abs_path}/{build}"
    os.makedirs(build_dir, exist_ok=True)
    return load(
        name=name,
        sources=[f"{abs_path}/{p}" for p in path],
        extra_cflags=list(cflags or []) or ["-O3", "-std=c++20"],
        extra_cuda_cflags=list(cuda_flags or []) or ["-O3", "-std=c++20"],
        extra_ldflags=list(ldflags or []) or None,
        build_directory=build_dir,
        verbose=True,
    )


def update_cache_cnt_cuda(
    sorted_cluster_ids, 
    topk_cluster_cnt, 
    codebook
) -> torch.Tensor:
    '''
    Args:
        sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
        topk_cluster_cnt: [bs, kv_heads, B] 每个子空间选择的cluster数量
        codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
        [bs, kv_heads, B, num_cluster, kv_len] -> bs * heads * B * num_cluster = 1 * 8 * 16 * 256
        [bs, kv_heads, B]->[num_cluster] -> [bs, kv_heads, B] 1 * 8 * 16;
        [bs, heads, kvlend, dim] -> 1 * 8 * 128
    Returns:
        cache_cnt: [bs, kv_heads, kv_len] 每个key的碰撞次数（累积所有子空间）
    '''
    # 转置codebook，使得最后两个维度对应于子空间和cluster ID
    codebook = codebook.transpose(-1, -2).contiguous()
    # bs, kv_heads, B, kv_len = codebook.shape
    # codebook = codebook.view(bs, kv_heads, kv_len, B)
    bs, kv_heads, kv_len, B = codebook.shape
    # 初始化碰撞计数器
    cache_cnt = torch.empty(bs, kv_heads, kv_len, dtype=torch.int32, device=codebook.device)
    load_kernel_module("collision.cu", "update_cache_cnt").update_cache_cnt(
        sorted_cluster_ids,
        topk_cluster_cnt,
        codebook,
        cache_cnt,
    )
    return cache_cnt


if __name__ == "__main__":

    batchsize = 1
    kv_heads = 8
    B = 16
    num_clusters = 256
    kv_len = 32*1024 + 3
    test_accuracy = False


    print("Testing...")
    print("Performance comparison:\n")
    print("batchsize kv_len subspace cluster_num ref_perf cuda_perf")
    for B, num_clusters in [(16, 256)]:
    # for B, num_clusters in [(32, 64)]:
        # 正确性测试
        sorted_cluster_ids, topk_cluster_cnt, codebook = generate_test_input(
            batchsize, kv_heads, B, num_clusters, kv_len
        )

        if test_accuracy:
            # print(sorted_cluster_ids.shape, topk_cluster_cnt.shape, codebook.shape)
            # cnt_ref = update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook)
            cnt_cuda = update_cache_cnt_cuda(sorted_cluster_ids, topk_cluster_cnt, codebook)
            # res = (cnt_ref - cnt_cuda).sum().item()

            # if res == 0:
            #     print(f"{B} {num_clusters}: Pass!")
            # else: 
            #     # print(f"error value: {cnt_ref}, {cnt_cuda}, {cnt_cuda.shape}, {sorted_cluster_ids}")
            #     raise ValueError("Wrong result")

        # 性能测试
        for bs in [1]:
            for kv_len in [i for i in [128*1024]]:
                print(f"{bs} {kv_len} {B} {num_clusters} ", end="")
                sorted_cluster_ids, topk_cluster_cnt, codebook = generate_test_input(
                    batchsize, kv_heads, B, num_clusters, kv_len
                )

                # def ref():
                    # update_cache_cnt_ref(sorted_cluster_ids, topk_cluster_cnt, codebook)

                def cuda():
                    update_cache_cnt_cuda(sorted_cluster_ids, topk_cluster_cnt, codebook)

                # ref_perf = triton.testing.do_bench(ref)

                with torch.cuda.nvtx.range("collision"):
                    cauda_perf = triton.testing.do_bench(cuda)

                # print(f'{ref_perf:.2f} {cauda_perf:.2f} {ref_perf/cauda_perf:.2f}')

                print(f'{cauda_perf:.2f}')