#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <cuda_runtime.h>

#define WARP_SIZE 32
#define CHUNK_SIZE 8
#define CHUNKS_PER_WARP (WARP_SIZE / CHUNK_SIZE)  // 4

// This c8 module only exposes the kernels used by the CDS chunk-id path:
// 1) partial_chunk_density_scores
// 2) partial_rerank_chunk_dense_sparse

struct RerankChunkDenseSparseParams {
    const void *__restrict__ chunk_ids_ptr;      // [bs, kv_heads, nk] int64
    const void *__restrict__ dense_mask_ptr;      // [bs, kv_heads, nk] bool
    const void *__restrict__ key_4bit_ptr;
    const void *__restrict__ key_block_weight_ptr;
    const void *__restrict__ q_blocks_ptr;
    const void *__restrict__ q_norm_ptr;
    const void *__restrict__ mag_centers_ptr;
    void *__restrict__ out_scores_ptr;            // [bs, kv_heads, nk * CHUNK_SIZE] bf16
    void *__restrict__ out_indices_ptr;           // [bs, kv_heads, nk * CHUNK_SIZE] int64
    int m;
    int kv_len;
    int num_chunks;      // = nk
    int dense_topk;
    int sparse_topk;

    int key4_bs_stride;
    int key4_head_stride;
    int key4_B_stride;
    int key4_len_stride;

    int keyw_bs_stride;
    int keyw_head_stride;
    int keyw_B_stride;
};

struct ChunkDensityScoresParams {
    const void *__restrict__ chunk_ids_ptr;       // [bs, kv_heads, nk] int64
    const void *__restrict__ chunk_centroids_ptr; // [bs, kv_heads, n_chunks, head_dim] bf16
    const void *__restrict__ raw_q_ptr;           // [bs, kv_heads, head_dim] bf16
    void *__restrict__ out_scores_ptr;            // [bs, kv_heads, nk] float32
    int nk;
    int n_chunks;
    int head_dim;

    int64_t chunk_ids_bs_stride;
    int64_t chunk_ids_head_stride;
    int64_t chunk_ids_nk_stride;

    int64_t cent_bs_stride;
    int64_t cent_head_stride;
    int64_t cent_chunk_stride;
    int64_t cent_dim_stride;

    int64_t rawq_bs_stride;
    int64_t rawq_head_stride;
    int64_t rawq_dim_stride;

    int64_t out_bs_stride;
    int64_t out_head_stride;
    int64_t out_nk_stride;
};

__global__
void partial_chunk_density_scores_kernel(
    __constant__ ChunkDensityScoresParams params
) {
    constexpr int BLOCK_SIZE = 128;
    __shared__ float reduce_buf[BLOCK_SIZE];

    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int slot = blockIdx.z;
    const int tidx = threadIdx.x;

    if (slot >= params.nk) {
        return;
    }

    const int64_t* chunk_ids_ptr = reinterpret_cast<const int64_t*>(params.chunk_ids_ptr);
    const __nv_bfloat16* centroids_ptr =
        reinterpret_cast<const __nv_bfloat16*>(params.chunk_centroids_ptr);
    const __nv_bfloat16* raw_q_ptr = reinterpret_cast<const __nv_bfloat16*>(params.raw_q_ptr);
    float* out_ptr = reinterpret_cast<float*>(params.out_scores_ptr);

    const int64_t chunk_slot_offset =
        static_cast<int64_t>(bidx) * params.chunk_ids_bs_stride +
        static_cast<int64_t>(bidy) * params.chunk_ids_head_stride +
        static_cast<int64_t>(slot) * params.chunk_ids_nk_stride;
    const int64_t chunk_id64 = __ldg(chunk_ids_ptr + chunk_slot_offset);

    float acc = 0.0f;
    const bool valid_chunk_id = (chunk_id64 >= 0 && chunk_id64 < params.n_chunks);
    if (valid_chunk_id) {
        const int64_t base_cent_offset =
            static_cast<int64_t>(bidx) * params.cent_bs_stride +
            static_cast<int64_t>(bidy) * params.cent_head_stride +
            chunk_id64 * params.cent_chunk_stride;
        const int64_t base_q_offset =
            static_cast<int64_t>(bidx) * params.rawq_bs_stride +
            static_cast<int64_t>(bidy) * params.rawq_head_stride;

        for (int d = tidx; d < params.head_dim; d += BLOCK_SIZE) {
            const int64_t cent_off = base_cent_offset + static_cast<int64_t>(d) * params.cent_dim_stride;
            const int64_t q_off = base_q_offset + static_cast<int64_t>(d) * params.rawq_dim_stride;
            const float cent_v = __bfloat162float(centroids_ptr[cent_off]);
            const float q_v = __bfloat162float(raw_q_ptr[q_off]);
            acc += cent_v * q_v;
        }
    }

    reduce_buf[tidx] = acc;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tidx < stride) {
            reduce_buf[tidx] += reduce_buf[tidx + stride];
        }
        __syncthreads();
    }

    if (tidx == 0) {
        const int64_t out_offset =
            static_cast<int64_t>(bidx) * params.out_bs_stride +
            static_cast<int64_t>(bidy) * params.out_head_stride +
            static_cast<int64_t>(slot) * params.out_nk_stride;
        out_ptr[out_offset] = valid_chunk_id ? reduce_buf[0] : -1e30f;
    }
}

void partial_chunk_density_scores_interface(
    at::Tensor chunk_ids,        // [bs, kv_heads, nk] int64
    at::Tensor chunk_centroids,  // [bs, kv_heads, n_chunks, head_dim] bf16
    at::Tensor raw_q,            // [bs, kv_heads, head_dim] bf16
    at::Tensor out_scores        // [bs, kv_heads, nk] float32 (pre-allocated)
) {
    TORCH_CHECK(chunk_ids.scalar_type() == at::ScalarType::Long,
                "chunk_ids must be int64");
    TORCH_CHECK(chunk_centroids.scalar_type() == at::ScalarType::BFloat16,
                "chunk_centroids must be bfloat16");
    TORCH_CHECK(raw_q.scalar_type() == at::ScalarType::BFloat16,
                "raw_q must be bfloat16");
    TORCH_CHECK(out_scores.scalar_type() == at::ScalarType::Float,
                "out_scores must be float32");

    TORCH_CHECK(chunk_ids.dim() == 3, "chunk_ids must be [bs, kv_heads, nk]");
    TORCH_CHECK(chunk_centroids.dim() == 4,
                "chunk_centroids must be [bs, kv_heads, n_chunks, head_dim]");
    TORCH_CHECK(raw_q.dim() == 3, "raw_q must be [bs, kv_heads, head_dim]");
    TORCH_CHECK(out_scores.dim() == 3, "out_scores must be [bs, kv_heads, nk]");

    const int bs = chunk_ids.size(0);
    const int kv_heads = chunk_ids.size(1);
    const int nk = chunk_ids.size(2);
    const int n_chunks = chunk_centroids.size(2);
    const int head_dim = chunk_centroids.size(3);

    TORCH_CHECK(chunk_centroids.size(0) == bs && chunk_centroids.size(1) == kv_heads,
                "chunk_centroids batch/head mismatch");
    TORCH_CHECK(raw_q.size(0) == bs && raw_q.size(1) == kv_heads,
                "raw_q batch/head mismatch");
    TORCH_CHECK(raw_q.size(2) == head_dim,
                "raw_q head_dim mismatch: expected ", head_dim, " got ", raw_q.size(2));
    TORCH_CHECK(out_scores.size(0) == bs && out_scores.size(1) == kv_heads,
                "out_scores batch/head mismatch");
    TORCH_CHECK(out_scores.size(2) >= nk,
                "out_scores last dim too small: need ", nk, " got ", out_scores.size(2));

    const c10::Device dev = chunk_ids.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr int BLOCK_SIZE = 128;
    const auto grid = dim3{(unsigned)bs, (unsigned)kv_heads, (unsigned)nk};
    const auto block = dim3{BLOCK_SIZE};

    ChunkDensityScoresParams params{
        chunk_ids.data_ptr(),
        chunk_centroids.data_ptr(),
        raw_q.data_ptr(),
        out_scores.data_ptr(),
        nk, n_chunks, head_dim,
        chunk_ids.stride(0), chunk_ids.stride(1), chunk_ids.stride(2),
        chunk_centroids.stride(0), chunk_centroids.stride(1),
        chunk_centroids.stride(2), chunk_centroids.stride(3),
        raw_q.stride(0), raw_q.stride(1), raw_q.stride(2),
        out_scores.stride(0), out_scores.stride(1), out_scores.stride(2),
    };

    partial_chunk_density_scores_kernel<<<grid, block, 0, stream>>>(params);
    auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "partial_chunk_density_scores(c8) kernel failed: ", cudaGetErrorString(result));
}

template<int B>
__global__
void partial_rerank_chunk_dense_sparse_kernel(
    __constant__ RerankChunkDenseSparseParams params
) {
    const int bidx = blockIdx.x;   // bs
    const int bidy = blockIdx.y;   // kv_head
    const int bidz = blockIdx.z;   // chunk-block
    const int tidx = threadIdx.x;

    const int num_chunks     = params.num_chunks;
    const int dense_topk     = params.dense_topk;
    const int sparse_topk    = params.sparse_topk;
    const int key4_B_stride  = params.key4_B_stride;
    const int key4_len_stride = params.key4_len_stride;
    const int keyw_B_stride  = params.keyw_B_stride;
    const int kv_len         = params.kv_len;
    const int m = params.m;

    const int warp_id       = tidx / WARP_SIZE;
    const int lane_id       = tidx % WARP_SIZE;
    const int sub_id        = lane_id / CHUNK_SIZE;
    const int lane_in_chunk = lane_id % CHUNK_SIZE;

    const int chunks_per_block = (blockDim.x / WARP_SIZE) * CHUNKS_PER_WARP;
    const int chunk_slot = bidz * chunks_per_block + warp_id * CHUNKS_PER_WARP + sub_id;

    const bool valid_chunk = (chunk_slot < num_chunks);
    const int flat_bh = bidx * gridDim.y + bidy;

    // ---- read chunk ID and dense flag ----
    const int64_t* chunk_ids_ptr =
        reinterpret_cast<const int64_t*>(params.chunk_ids_ptr) + flat_bh * num_chunks;
    const char* dense_mask_ptr =
        reinterpret_cast<const char*>(params.dense_mask_ptr) + flat_bh * num_chunks;

    int64_t real_chunk_id = 0;
    bool is_dense = false;
    if (valid_chunk) {
        real_chunk_id = __ldg(chunk_ids_ptr + chunk_slot);
        is_dense = (dense_mask_ptr[chunk_slot] != 0);
    }

    // token_idx from chunk_id
    int token_idx = static_cast<int>(real_chunk_id) * CHUNK_SIZE + lane_in_chunk;
    if (token_idx >= kv_len) token_idx = kv_len - 1;
    const bool valid_token = valid_chunk && (token_idx < kv_len);

    // ---- pointer setup ----
    const uint32_t* cand_4bit_ptr =
        reinterpret_cast<const uint32_t*>(params.key_4bit_ptr) +
        bidx * params.key4_bs_stride + bidy * params.key4_head_stride;
    const __nv_bfloat16* key_block_weight_ptr =
        reinterpret_cast<const __nv_bfloat16*>(params.key_block_weight_ptr) +
        bidx * params.keyw_bs_stride + bidy * params.keyw_head_stride;
    const __nv_bfloat16* q_blocks_ptr =
        reinterpret_cast<const __nv_bfloat16*>(params.q_blocks_ptr) + flat_bh * B * m;
    float q_norm_f =
        *(reinterpret_cast<const float*>(params.q_norm_ptr) + flat_bh);

    __nv_bfloat16 mag_centers[8];
    *reinterpret_cast<uint4*>(&mag_centers[0]) =
        __ldg(reinterpret_cast<const uint4*>(params.mag_centers_ptr));

    // ---- compute RaBitQ score ----
    float score_f = -1e30f;

    if (valid_token) {
        __nv_bfloat16 res = __float2bfloat16(0.0f);
        #pragma unroll(4)
        for (int i = 0; i < B; i++) {
            __nv_bfloat16 bit4_keys[8];
            __nv_bfloat16 q_blk[8];
            *reinterpret_cast<uint4*>(&q_blk[0]) =
                __ldg(reinterpret_cast<const uint4*>(q_blocks_ptr) + i);
            uint32_t bit4_key =
                __ldg(cand_4bit_ptr + i * key4_B_stride + token_idx * key4_len_stride);
            __nv_bfloat16 bit4_weight =
                __ldg(key_block_weight_ptr + i * keyw_B_stride + token_idx);
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                const uint8_t even = static_cast<uint8_t>((bit4_key >> (j * 8))     & 0x0F);
                const uint8_t odd  = static_cast<uint8_t>((bit4_key >> (j * 8 + 4)) & 0x0F);
                __nv_bfloat16 val = mag_centers[even & 0x7];
                bit4_keys[j * 2]     = (even & 0x8) ? val : -val;
                val = mag_centers[odd & 0x7];
                bit4_keys[j * 2 + 1] = (odd  & 0x8) ? val : -val;
            }
            __nv_bfloat16 temp_res = __float2bfloat16(0.0f);
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                temp_res += bit4_keys[j] * q_blk[j];
            }
            res += temp_res * bit4_weight;
        }
        score_f = __bfloat162float(res) * q_norm_f;
    }

    // ---- bitonic sort within CHUNK_SIZE-lane chunk (descending) ----
    #pragma unroll
    for (int k = 2; k <= CHUNK_SIZE; k <<= 1) {
        #pragma unroll
        for (int j = k >> 1; j >= 1; j >>= 1) {
            float other_score = __shfl_xor_sync(0xFFFFFFFF, score_f, j);
            int   other_idx   = __shfl_xor_sync(0xFFFFFFFF, token_idx, j);
            bool desc        = ((lane_in_chunk & k) == 0);
            bool is_lower    = ((lane_in_chunk & j) == 0);
            bool want_larger = (desc == is_lower);
            bool do_swap = want_larger ? (other_score > score_f)
                                       : (other_score < score_f);
            if (do_swap) {
                score_f   = other_score;
                token_idx = other_idx;
            }
        }
    }

    // ---- write output: uniform nk * CHUNK_SIZE layout ----
    if (valid_chunk) {
        const int out_base = flat_bh * num_chunks * CHUNK_SIZE
                           + chunk_slot * CHUNK_SIZE + lane_in_chunk;
        const int effective_topk = is_dense ? dense_topk : sparse_topk;
        if (lane_in_chunk < effective_topk) {
            reinterpret_cast<__nv_bfloat16*>(params.out_scores_ptr)[out_base] =
                __float2bfloat16(score_f);
            reinterpret_cast<int64_t*>(params.out_indices_ptr)[out_base] =
                static_cast<int64_t>(token_idx);
        } else {
            reinterpret_cast<__nv_bfloat16*>(params.out_scores_ptr)[out_base] =
                __float2bfloat16(-1e30f);
            reinterpret_cast<int64_t*>(params.out_indices_ptr)[out_base] =
                0;
        }
    }
}

void partial_rerank_chunk_dense_sparse_interface(
    at::Tensor chunk_ids,          // [bs, kv_heads, nk] int64
    at::Tensor dense_mask,         // [bs, kv_heads, nk] bool
    at::Tensor key_4bit,
    at::Tensor key_block_weight,
    at::Tensor q_blocks,
    at::Tensor q_norm,
    at::Tensor mag_centers,
    int B, int m,
    int dense_topk,
    int sparse_topk,
    at::Tensor out_scores,         // [bs, kv_heads, nk * CHUNK_SIZE] bf16 (pre-allocated)
    at::Tensor out_indices         // [bs, kv_heads, nk * CHUNK_SIZE] int64 (pre-allocated)
) {
    TORCH_CHECK(key_4bit.scalar_type() == at::ScalarType::Byte);
    TORCH_CHECK(key_block_weight.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(q_blocks.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(q_norm.scalar_type() == at::ScalarType::Float);
    TORCH_CHECK(mag_centers.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out_scores.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(out_indices.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(chunk_ids.scalar_type() == at::ScalarType::Long);
    TORCH_CHECK(dense_mask.scalar_type() == at::ScalarType::Bool);
    TORCH_CHECK(m == 8, "m must be 8");
    TORCH_CHECK(dense_topk >= 1 && dense_topk <= CHUNK_SIZE);
    TORCH_CHECK(sparse_topk >= 1 && sparse_topk <= CHUNK_SIZE);

    const int bs       = key_4bit.size(0);
    const int kv_heads = key_4bit.size(1);
    const int kv_len   = key_4bit.size(3);
    const int nk       = chunk_ids.size(2);

    TORCH_CHECK(out_scores.size(2) >= nk * CHUNK_SIZE,
                "out_scores too small: need ", nk * CHUNK_SIZE, " got ", out_scores.size(2));

    constexpr int BLOCK_SIZE = 128;
    const int chunks_per_block = (BLOCK_SIZE / WARP_SIZE) * CHUNKS_PER_WARP;
    const int grid_z = (nk + chunks_per_block - 1) / chunks_per_block;

    const c10::Device dev = key_4bit.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const auto grid  = dim3{(unsigned)bs, (unsigned)kv_heads, (unsigned)grid_z};
    const auto block = dim3{BLOCK_SIZE};

    RerankChunkDenseSparseParams params{
        chunk_ids.data_ptr(),
        dense_mask.data_ptr(),
        key_4bit.data_ptr(),
        key_block_weight.data_ptr(),
        q_blocks.data_ptr(),
        q_norm.data_ptr(),
        mag_centers.data_ptr(),
        out_scores.data_ptr(),
        out_indices.data_ptr(),
        m, kv_len, nk, dense_topk, sparse_topk,
        int(key_4bit.stride(0) / 4), int(key_4bit.stride(1) / 4),
        int(key_4bit.stride(2) / 4), int(key_4bit.stride(3) / 4),
        int(key_block_weight.stride(0)), int(key_block_weight.stride(1)),
        int(key_block_weight.stride(2)),
    };

    if (B == 16) {
        partial_rerank_chunk_dense_sparse_kernel<16><<<grid, block, 0, stream>>>(params);
    } else {
        TORCH_CHECK(false, "only support B = 16");
    }

    auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "partial_rerank_chunk_dense_sparse(c8) kernel failed: ", cudaGetErrorString(result));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("partial_rerank_chunk_dense_sparse", &partial_rerank_chunk_dense_sparse_interface);
    m.def("partial_chunk_density_scores", &partial_chunk_density_scores_interface);
}

