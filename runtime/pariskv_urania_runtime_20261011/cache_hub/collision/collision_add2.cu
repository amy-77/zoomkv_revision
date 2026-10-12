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
    const int64_t *__restrict__ sorted_cluster_ids_ptr; // [bs, kv_heads, B, num_clusters] - int64 避免转换
    const int64_t *__restrict__ topk_cluster_cnt_ptr;   // [bs, kv_heads, B] - int64 避免转换
    const int64_t *__restrict__ topk_cluster_cnt_top5_ptr;   // [bs, kv_heads, B] - int64 避免转换
    const uint8_t *__restrict__ code_book_ptr;   // [bs, kv_heads, kv_len, B] - uint8
    int *__restrict__ cache_cnt_ptr; // [bs, kv_heads, kv_len]
    uint32_t *__restrict__ cluster_ids_bits; // [bs, kv_heads, B, num_clusters/8]
    uint32_t *__restrict__ cluster_ids_top5_bits; // [bs, kv_heads, B, num_clusters/8]
    int batchsize;
    int kv_heads;
    int sub_spaces;
    int num_clusters;
    int kv_lens;
};


template<int Bits>
struct __align__(4) uint_bits{
    static_assert(Bits > 0, "Bit width must be positive.");
    static constexpr int StorageWords = (Bits + 31) / 32;
    uint32_t data[StorageWords];
};

template<int NUM_CLUSTER>
__global__ void convert_to_bits(const CounterParams params){

    const int warp_idx = threadIdx.x / 32;
    const int warp_num = blockDim.x / 32;
    const int tid_in_warp = threadIdx.x % 32;
    const int bidx = blockIdx.x;
    // input - 使用 int64_t
    const int64_t* cluster_ids_ptr = params.sorted_cluster_ids_ptr + bidx * NUM_CLUSTER;
    const int64_t* cluster_end_ptr = params.topk_cluster_cnt_ptr + bidx;
    const int64_t* cluster_end_top5_ptr = params.topk_cluster_cnt_top5_ptr + bidx;
    // output
    uint32_t* cluster_ids_bits_ptr = params.cluster_ids_bits + bidx * NUM_CLUSTER / 32;
    uint32_t* cluster_ids_bits_top5_ptr = params.cluster_ids_top5_bits + bidx * NUM_CLUSTER / 32;
    int tidx = threadIdx.x;

    // 读取end - 转为 int
    const int end = static_cast<int>(__ldg(cluster_end_ptr));
    const int end_top5 = static_cast<int>(__ldg(cluster_end_top5_ptr));

    __shared__ uint_bits<NUM_CLUSTER> bits;
    __shared__ uint_bits<NUM_CLUSTER> bits_top5;
    
    // 初始化 确保为0
    if (tidx < NUM_CLUSTER / 32){
        bits.data[tidx] = 0u;
        bits_top5.data[tidx] = 0u;  // 同时初始化 bits_top5
    }
    __syncthreads();

    // 读取一个cluster 并转化 每个warp 使用一个uint32 存储数据
    int index = static_cast<int>(__ldg(cluster_ids_ptr + tidx));
    
    // 每个线程都来操作下
    if (tidx < end){
        // bits.set_pos(index);
        atomicOr( &(bits.data[index / 32]), 1u << (index % 32));
    }

    if (tidx < end_top5){
        atomicOr( &(bits_top5.data[index / 32]), 1u << (index % 32));
    }

    __syncthreads();

    // global写入
    #pragma unroll
    for(int i = tidx; i < NUM_CLUSTER / 32; i += blockDim.x){
        cluster_ids_bits_ptr[i] = bits.data[i];
        cluster_ids_bits_top5_ptr[i] = bits_top5.data[i];
    }
}

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
    const void* cluster_ids_ptr_bits = params.cluster_ids_bits + bidx * SUB_SPACE * NUM_CLUSTERS / 32;
    const void* cluster_ids_ptr_top5_bits = params.cluster_ids_top5_bits + bidx * SUB_SPACE * NUM_CLUSTERS / 32;
    int* cache_cnt_ptr = params.cache_cnt_ptr + bidx * kv_lens;

    // codebook 现在是 uint8 类型
    const uint8_t* code_book_ptr = params.code_book_ptr + bidx * SUB_SPACE * kv_lens;
    int tidx = threadIdx.x;

    // clear shared memory
    __shared__ uint_bits<NUM_CLUSTERS> cluster_ids_bits_smem[SUB_SPACE];
    __shared__ uint_bits<NUM_CLUSTERS> cluster_ids_bits_top5_smem[SUB_SPACE];

    // b128 读取
    #pragma unroll
    for (int i = tidx; i < SUB_SPACE * NUM_CLUSTERS / 32 / 4; i += blockDim.x) {

        // 读取最开始的
        const int4* gptr = reinterpret_cast<const int4*>(cluster_ids_ptr_bits) + i;
        int4 val = __ldg(gptr);

        int4* sptr = reinterpret_cast<int4*>(cluster_ids_bits_smem) + i;
        *sptr = val;

        // 读取 top5 的
        const int4* gptr_top5 = reinterpret_cast<const int4*>(cluster_ids_ptr_top5_bits) + i;
        int4 val_top5 = __ldg(gptr_top5);

        int4* sptr_top5 = reinterpret_cast<int4*>(cluster_ids_bits_top5_smem) + i;
        *sptr_top5 = val_top5;
    }
    __syncthreads();

    const int start_k = bidy * BLOCK_SIZE;
    const int end_k = min(kv_lens, start_k + BLOCK_SIZE);

    #pragma unroll
    for (int ks = tidx + start_k; ks < end_k; ks += blockDim.x) {
        int cnt = 0;
        // 优化：一次读取多个 uint8（16个 = 128 bits），然后处理
        // 当 SUB_SPACE=16 时，正好可以一次性读取所有 16 个 cluster IDs
        #pragma unroll
        for(int i = 0; i < SUB_SPACE; i += 1) {
            // 读取 uint8 并转换为 int
            const int target_cluster_id = static_cast<int>(code_book_ptr[ks * SUB_SPACE + i]);
            const int idx_0 = target_cluster_id / 32;
            const int idx_1 = target_cluster_id % 32;
            bool hit = cluster_ids_bits_smem[i].data[idx_0] & (1u << idx_1);
            if (hit) cnt += 1;

            bool hit_top5 = cluster_ids_bits_top5_smem[i].data[idx_0] & (1u << idx_1);
            if (hit_top5) cnt += 1;
        }
        __syncwarp();
        cache_cnt_ptr[ks] = cnt;
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
    at::Tensor topk_cluster_cnt_top5,
    at::Tensor code_book,
    at::Tensor cache_cnt
){
    /*
        Args:
            sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs (int32)
            topk_cluster_cnt: [bs, kv_heads, B] 每个子空间选择的cluster数量 (int32)
            codebook: [bs, kv_heads, kv_len, B] 每个key在每个子空间的cluster ID (uint8)
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

    // check dtype
    TORCH_CHECK(code_book.scalar_type() == at::ScalarType::Byte, 
                "code_book must be uint8 tensor");
    TORCH_CHECK(sorted_cluster_ids.scalar_type() == at::ScalarType::Long,
                "sorted_cluster_ids must be int64 tensor");
    TORCH_CHECK(topk_cluster_cnt.scalar_type() == at::ScalarType::Long,
                "topk_cluster_cnt must be int64 tensor");

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

    // launch kernel
    const c10::Device dev = cache_cnt.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid = dim3{kBlockNumX, kBlockNumY};
    const auto block = dim3{256};
    
    torch::Tensor cluster_ids_bits = torch::zeros({batchsize, kv_heads, sub_spaces, num_clusters / 32}, sorted_cluster_ids.options().dtype(torch::kUInt32));
    torch::Tensor cluster_ids_top5_bits = torch::zeros({batchsize, kv_heads, sub_spaces, num_clusters / 32}, sorted_cluster_ids.options().dtype(torch::kUInt32));

    CounterParams params{
        sorted_cluster_ids.data_ptr<int64_t>(),  // int64
        topk_cluster_cnt.data_ptr<int64_t>(),    // int64
        topk_cluster_cnt_top5.data_ptr<int64_t>(), // int64
        code_book.data_ptr<uint8_t>(),           // uint8
        cache_cnt.data_ptr<int>(),
        cluster_ids_bits.data_ptr<uint32_t>(),
        cluster_ids_top5_bits.data_ptr<uint32_t>(),
        batchsize,
        kv_heads,
        sub_spaces,
        num_clusters,
        kv_lens
    };

    if (num_clusters == 32) {
        convert_to_bits<32><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<32>) * 2 >>>(params);
    } else if (num_clusters == 64) {    
        convert_to_bits<64><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<64>) * 2 >>>(params);
    } else if (num_clusters == 128) {
        convert_to_bits<128><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<128>) * 2 >>>(params);
    } else if (num_clusters == 256) {
        convert_to_bits<256><<<batchsize * kv_heads * sub_spaces, num_clusters, sizeof(uint_bits<256>) * 2 >>>(params);
    } else if (num_clusters == 512) {
        convert_to_bits<512><<<batchsize * kv_heads * sub_spaces, 512, sizeof(uint_bits<512>) * 2 >>>(params);
    } else if (num_clusters == 1024) {
        convert_to_bits<1024><<<batchsize * kv_heads * sub_spaces, 1024, sizeof(uint_bits<1024>) * 2 >>>(params);
    } else {
        TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
    }

    int smem_size = (sub_spaces * 2 * num_clusters);
    if (sub_spaces == 16) {
        if (num_clusters == 32) {
            launch_update_cache_cnt<16, 32, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 64) {
            launch_update_cache_cnt<16, 64, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 128) {
            launch_update_cache_cnt<16, 128, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 256) {
            launch_update_cache_cnt<16, 256, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 512) {
            launch_update_cache_cnt<16, 512, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 1024) {
            launch_update_cache_cnt<16, 1024, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
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
        } else if (num_clusters == 512) {
            launch_update_cache_cnt<32, 512, BLOCK_SIZE>(grid, block, smem_size, params);
        } else if (num_clusters == 1024) {
            launch_update_cache_cnt<32, 1024, BLOCK_SIZE>(grid, block, smem_size, params);
        } else {
            TORCH_CHECK(false, "num_clusters must be 32, 64, 128, 256, 512 or 1024");
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