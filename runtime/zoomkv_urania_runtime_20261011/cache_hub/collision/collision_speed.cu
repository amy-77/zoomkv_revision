#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <cuda_runtime.h>
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

struct CounterParams {
    const int *__restrict__ sorted_cluster_ids_ptr; // [bs, kv_heads, B, num_clusters]
    const int *__restrict__ topk_cluster_cnt_ptr;   // [bs, kv_heads, B]
    const int *__restrict__ code_book_ptr;   // [bs, kv_heads, kv_len, B]
    int *__restrict__ cache_cnt_ptr; // [bs, kv_heads, kv_len]
    int batchsize;
    int kv_heads;
    int sub_spaces;
    int num_clusters;
    int kv_lens;
};


template<int SUB_SPACE, int NUM_CLUSTERS, int BLOCK_SIZE>
__global__
void update_cache_cnt_kernel(const CounterParams params){
    const int warp_idx = threadIdx.x / 32;
    const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int kv_lens = params.kv_lens;
    const int batchsize = params.batchsize;
    const int kv_heads = params.kv_heads;
    const int* cluster_ids_ptr = params.sorted_cluster_ids_ptr + bidx * NUM_CLUSTERS * SUB_SPACE;
    const int* cluster_end_ptr = params.topk_cluster_cnt_ptr + bidx * SUB_SPACE;

    const int* code_book_ptr = params.code_book_ptr + bidx * SUB_SPACE * kv_lens;
    int tidx = threadIdx.x;

    // clear shared memory
    __shared__ int cluster_ids_smem[SUB_SPACE * NUM_CLUSTERS];
    __shared__ int cache_cnt_smem[BLOCK_SIZE];
    __shared__ int target_idx_smem[BLOCK_SIZE];
    __shared__ int cluster_end_smem[SUB_SPACE];

    #pragma unroll
    for (int i = tidx; i < SUB_SPACE * NUM_CLUSTERS / 4; i += blockDim.x) {

        const int4* gptr = reinterpret_cast<const int4*>(cluster_ids_ptr) + i;
        int4 val = __ldg(gptr);

        int4* sptr = reinterpret_cast<int4*>(cluster_ids_smem) + i;
        *sptr = val;
    }
    #pragma unroll
    for (int i = tidx; i < SUB_SPACE; i += blockDim.x) {
        cluster_end_smem[i] = __ldg(cluster_end_ptr + i);
    }

    __syncthreads();
    const int start_k = bidy * BLOCK_SIZE;
    const int end_k = min(kv_lens, start_k + BLOCK_SIZE);
    // 一个block处理BLOCK_SIZE个kv_len
    for (int ks = start_k + warp_idx; ks < end_k; ks += warp_num) {
        int cnt = 0;
        // 每个warp 计算一个kv_len，遍历所有子空间
        #pragma unroll
        for(int i = 0; i < SUB_SPACE; i += 1) {
            bool any_hit_in_warp = false;
            // 读取第i个子空间的 target 
            const int target_cluster_id = __ldg(code_book_ptr + ks * SUB_SPACE + i);
            // 读取第i个子空间的 end
            int cluster_end = *(cluster_end_smem + i);
            int cluster_end_warp = (cluster_end + 31) / 32 * 32;
            #pragma unroll
            for (int j = tid_in_warp; j < cluster_end_warp; j += 32){
                const int cluster_ids = *(cluster_ids_smem + i * NUM_CLUSTERS + j);
                bool hit = (cluster_ids == target_cluster_id) && (j < cluster_end);
                any_hit_in_warp = any_hit_in_warp | hit;
            }
            // 如果warp内有hit，则为1，否则为0
            bool hit_res = __any_sync(0xFFFFFFFF, any_hit_in_warp);
            if ( hit_res ) {
                cnt += 1;
            }
        }

        if (tid_in_warp == 0){
            cache_cnt_smem[ks - start_k] = cnt;
        }
    }
    __syncthreads();

    const int total = end_k - start_k;
    int* cache_cnt_ptr = params.cache_cnt_ptr + bidx * kv_lens;
    
    #pragma unroll
    for (int i = tidx; i < total; i += blockDim.x) {
        cache_cnt_ptr[start_k + i] = cache_cnt_smem[i];
    }
}

template<int SUB_SPACES, int NUM_CLUSTERS, int BLOCK_SIZE>
inline void launch_update_cache_cnt(dim3 grid, dim3 block, int smem_size,
                             const CounterParams &params) 
{

    // 设置动态 shared memory 上限
    cudaFuncSetAttribute(
        update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        smem_size);

    // launch
    update_cache_cnt_kernel<SUB_SPACES, NUM_CLUSTERS, BLOCK_SIZE><<<grid, block, smem_size>>>(params);
}


void update_cache_cnt_interface(
    at::Tensor sorted_cluster_ids,
    at::Tensor topk_cluster_cnt,
    at::Tensor code_book,
    at::Tensor cache_cnt
){
    /*
        Args:
            sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
            topk_cluster_cnt: [bs, kv_heads, B] 每个子空间选择的cluster数量
            codebook: [bs, kv_heads, kv_len, B] 每个key在每个子空间的cluster ID
        Returns:
            cache_cnt: [bs, kv_heads, kv_len] 每个key的碰撞次数（累积所有子空间）
    */

    // check cuda
    CHECK_DEVICE(sorted_cluster_ids);
    CHECK_DEVICE(topk_cluster_cnt);
    CHECK_DEVICE(code_book);

    // check continguous
    CHECK_CONTIGUOUS(sorted_cluster_ids);
    CHECK_CONTIGUOUS(topk_cluster_cnt);
    CHECK_CONTIGUOUS(code_book);

    // check shape
    TORCH_CHECK(sorted_cluster_ids.dim() == 4);
    TORCH_CHECK(topk_cluster_cnt.dim() == 3);
    TORCH_CHECK(code_book.dim() == 4);
    const int batchsize = sorted_cluster_ids.size(0);
    const int kv_heads = sorted_cluster_ids.size(1);
    const int sub_spaces = sorted_cluster_ids.size(2);
    const int num_clusters = sorted_cluster_ids.size(3);
    const int kv_lens = code_book.size(2);
    TORCH_CHECK(topk_cluster_cnt.size(0) == batchsize);
    TORCH_CHECK(code_book.size(0) == batchsize);
    TORCH_CHECK(topk_cluster_cnt.size(1) == kv_heads);
    TORCH_CHECK(code_book.size(1) == kv_heads);
    TORCH_CHECK(topk_cluster_cnt.size(2) == sub_spaces);
    TORCH_CHECK(code_book.size(3) == sub_spaces);
    // 每个 warp 处理32个cluster, 同时处理8个sub_space
    TORCH_CHECK(num_clusters % 32 == 0, "num_clusters must be a multiple of 32");
    TORCH_CHECK(sub_spaces % 8 == 0, "sub_spaces must be a multiple of 8");

    // Per Token
    // 防止kBlockNum 过大，kv_heads 提出来放到一个block 中执行
    constexpr int BLOCK_SIZE = 1024;
    const unsigned int kBlockNumX = batchsize * kv_heads;
    const unsigned int kBlockNumY = (kv_lens + BLOCK_SIZE - 1) / BLOCK_SIZE;
    // block 内部的线程数
    // const unsigned int kThreadsPerBlock = min(1024, sub_spaces * 32);

    // launch kernel
    const c10::Device dev = cache_cnt.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid = dim3{kBlockNumX, kBlockNumY};
    // const auto block = dim3{min(kThreadsPerBlock, sub_spaces * 32)};
    const auto block = dim3{256};

    CounterParams params{
        sorted_cluster_ids.data_ptr<int>(),
        topk_cluster_cnt.data_ptr<int>(),
        code_book.data_ptr<int>(),
        cache_cnt.data_ptr<int>(),
        batchsize,
        kv_heads,
        sub_spaces,
        num_clusters,
        kv_lens
    };

    int smem_size = (sub_spaces * 2 + sub_spaces * num_clusters + BLOCK_SIZE) * 4;

    if (sub_spaces == 16) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<16, 32, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<16, 64, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<16, 128, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<16, 256, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else if (sub_spaces == 32) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<32, 32, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<32, 64, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<32, 128, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<32, 256, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128 or 256");
        }
    } else{
        TORCH_CHECK(false, "sub_spaces must be 8, 16 or 32.");
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_cache_cnt", &update_cache_cnt_interface);
}