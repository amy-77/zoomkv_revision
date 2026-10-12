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
#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
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


template<int SUB_SPACE, int NUM_CLUSTERS>
__global__
void update_cache_cnt_kernel(const CounterParams params){
    const int warp_idx = threadIdx.x / 32;
    const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int bid = blockIdx.x;
    const int kv_lens = params.kv_lens;
    const int batchsize = params.batchsize;
    const int kv_heads = params.kv_heads;
    const int* cluster_ids_ptr = params.sorted_cluster_ids_ptr + bid / kv_lens * NUM_CLUSTERS * SUB_SPACE;
    const int* code_book_ptr = params.code_book_ptr + bid * SUB_SPACE;
    const int* cluster_end_ptr = params.topk_cluster_cnt_ptr + bid / kv_lens * SUB_SPACE;
    int* cache_cnt_ptr = params.cache_cnt_ptr;
    int tidx = threadIdx.x;
    int cnt = 0;
    // clear shared memory
    __shared__ int cnt_shared[SUB_SPACE];
    #pragma unroll
    for(int i = tidx; i < SUB_SPACE; i += blockDim.x) {
        cnt_shared[i] = 0;
    }

    // 一个warp 处理一个子空间
    #pragma unroll
    for(int i = warp_idx; i < SUB_SPACE; i += warp_num) {
        // 读取target ID
        const int target_cluster_id = __ldg(code_book_ptr + i);
        const int cluster_end = __ldg(cluster_end_ptr + i);
        bool any_hit_in_warp = false;
        if constexpr (NUM_CLUSTERS % 128 == 0) {
            // 4个一组
            constexpr int NUM_CLUSTERS_INT4 = NUM_CLUSTERS / 4;
            const auto cluster_ids_ptr_int4 = reinterpret_cast<const int4*>(cluster_ids_ptr);
            #pragma unroll
            for (int j = tid_in_warp; j < NUM_CLUSTERS_INT4; j += 32){
                // 读取sorted cluster id to int4
                const int4 cluster_ids_int4 = __ldg(cluster_ids_ptr_int4 + i * NUM_CLUSTERS_INT4 + j);
                bool hit =
                    ((j * 4 < cluster_end) & (target_cluster_id == cluster_ids_int4.x)) |
                    ((j * 4 + 1 < cluster_end) & (target_cluster_id == cluster_ids_int4.y)) |
                    ((j * 4 + 2 < cluster_end) & (target_cluster_id == cluster_ids_int4.z)) |
                    ((j * 4 + 3 < cluster_end) & (target_cluster_id == cluster_ids_int4.w));

                bool hit_res = __any_sync(0xFFFFFFFF, hit);
                any_hit_in_warp = any_hit_in_warp | hit_res;
            }
        } else {
            #pragma unroll
            for (int j = tid_in_warp; j < NUM_CLUSTERS; j += 32){
                // 读取sorted cluster id
                const int cluster_ids = __ldg(cluster_ids_ptr + i * NUM_CLUSTERS + j);
                bool hit = (cluster_ids == target_cluster_id) && (j < cluster_end);
                bool hit_res = __any_sync(0xFFFFFFFF, hit);
                any_hit_in_warp = any_hit_in_warp || hit_res;
            }
        }
        if (any_hit_in_warp && tid_in_warp == 0) {
            cnt_shared[i] = 1;
        }
    }

    // reduce all
    __syncthreads();

    int res_cnt = 0;
    if (tidx == 0){
        #pragma unroll
        for (int i = 0; i < SUB_SPACE; i++){
            res_cnt += cnt_shared[i];
        }
        cache_cnt_ptr[bid] = res_cnt;
    }
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
    CHECK_CUDA(sorted_cluster_ids);
    CHECK_CUDA(topk_cluster_cnt);
    CHECK_CUDA(code_book);

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
    TORCH_CHECK(
        sub_spaces == 4 || sub_spaces == 8 || sub_spaces == 16 || sub_spaces == 32,
        "sub_spaces must be 4, 8, 16 or 32"
    );

    // Per Token
    // 防止kBlockNum 过大，kv_heads 提出来放到一个block 中执行
    const unsigned int kBlockNum = batchsize * kv_heads * kv_lens;
    // block 内部的线程数
    const unsigned int kThreadsPerBlock = min(1024, sub_spaces * 32);

    // launch kernel
    const c10::Device dev = cache_cnt.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid = dim3{kBlockNum};
    const auto block = dim3{min(kThreadsPerBlock, sub_spaces * 32)};

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

    if (sub_spaces == 4) {
        if (num_clusters == 32) {
            update_cache_cnt_kernel<4, 32><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 64) {
            update_cache_cnt_kernel<4, 64><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 128) {
            update_cache_cnt_kernel<4, 128><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 256) {
            update_cache_cnt_kernel<4, 256><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 512) {
            update_cache_cnt_kernel<4, 512><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 1024) {
            update_cache_cnt_kernel<4, 1024><<<grid, block, sub_spaces * 4>>>(params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
        }
    } else if (sub_spaces == 8) {
        if (num_clusters == 32) {
            update_cache_cnt_kernel<8, 32><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 64) {
            update_cache_cnt_kernel<8, 64><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 128) {
            update_cache_cnt_kernel<8, 128><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 256) {
            update_cache_cnt_kernel<8, 256><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 512) {
            update_cache_cnt_kernel<8, 512><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 1024) {
            update_cache_cnt_kernel<8, 1024><<<grid, block, sub_spaces * 4>>>(params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
        }
    } else if (sub_spaces == 16) {
        if (num_clusters == 32) {
            update_cache_cnt_kernel<16, 32><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 64) {
            update_cache_cnt_kernel<16, 64><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 128) {
            update_cache_cnt_kernel<16, 128><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 256) {
            update_cache_cnt_kernel<16, 256><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 512) {
            update_cache_cnt_kernel<16, 512><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 1024) {
            update_cache_cnt_kernel<16, 1024><<<grid, block, sub_spaces * 4>>>(params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
        }
    } else if (sub_spaces == 32) {
        if (num_clusters == 32) {
            update_cache_cnt_kernel<32, 32><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 64) {
            update_cache_cnt_kernel<32, 64><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 128) {
            update_cache_cnt_kernel<32, 128><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 256) {
            update_cache_cnt_kernel<32, 256><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 512) {
            update_cache_cnt_kernel<32, 512><<<grid, block, sub_spaces * 4>>>(params);
        } else if (num_clusters == 1024) {
            update_cache_cnt_kernel<32, 1024><<<grid, block, sub_spaces * 4>>>(params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
        }
    } else{
        TORCH_CHECK(false, "sub_spaces must be 8, 16 or 32");
    }
    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("update_cache_cnt", &update_cache_cnt_interface);
}