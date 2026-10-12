/**
 * Fused Hierarchical Quest kernel
 *
 * Combines 4 separate kernel launches into a single kernel:
 *   Phase 1: Score large chunks         (cooperative multi-warp, scores → smem)
 *   Phase 2: Radix top-K on large scores (in-smem, selected indices → smem)
 *   Phase 3: Score sub-chunks            (using smem indices, scores → smem)
 *   Phase 4: Radix top-K + map-back      (in-smem, final indices → HBM)
 *
 * All intermediate results stay in shared memory (~16KB).
 * Grid: (bs * kv_heads,), Block: (256,) = 8 warps.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>

namespace {

constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int N_WARPS = BLOCK_THREADS / WARP_SIZE;
constexpr int HEAD_DIM = 128;
constexpr int RADIX = 256;

// Max supported problem sizes
constexpr int MAX_LARGE_CHUNKS = 2048;
constexpr int MAX_SUB_TOTAL = 4096;
constexpr int MAX_NK_LARGE = 512;
constexpr int MAX_CAND = 4096;

// Convert float to order-preserving uint32
__device__ __forceinline__ uint32_t float_to_ordered(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint8_t extract_byte(uint32_t x, int byte_offset) {
    return static_cast<uint8_t>((x >> byte_offset) & 0xFF);
}

// Shared memory layout (offsets computed at runtime via pointers)
struct SmemLayout {
    __nv_bfloat16 s_q[HEAD_DIM];             // 256 B
    float s_scores[MAX_SUB_TOTAL];           // up to 16 KB (reused for both phases)
    int16_t s_large_idx[MAX_NK_LARGE];       // up to 1 KB
    int s_hist[2][RADIX + 1];                // 2056 B
    int s_counter;
    int s_threshold_bin;
    int s_num_input[2];
    int s_last_remain;
    int16_t s_cand[2][MAX_CAND];             // up to 8 KB
};

// ============================================================================
// Phase 1 & 3: Cooperative multi-warp bf16 d128 chunk scoring
// ============================================================================

__device__ __forceinline__ float score_one_chunk_from_smem_q(
    const __nv_bfloat16* __restrict__ s_q,
    const __nv_bfloat16* __restrict__ min_ptr,
    const __nv_bfloat16* __restrict__ max_ptr,
    int lane)
{
    const int d_base = lane * 4;

    const uint2 q_u2   = *reinterpret_cast<const uint2*>(s_q + d_base);
    const uint2 min_u2 = *reinterpret_cast<const uint2*>(min_ptr + d_base);
    const uint2 max_u2 = *reinterpret_cast<const uint2*>(max_ptr + d_base);

    const __nv_bfloat162 q_lo  = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.x);
    const __nv_bfloat162 q_hi  = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.y);
    const __nv_bfloat162 mn_lo = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.x);
    const __nv_bfloat162 mn_hi = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.y);
    const __nv_bfloat162 mx_lo = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.x);
    const __nv_bfloat162 mx_hi = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.y);

    float q0 = __bfloat162float(__low2bfloat16(q_lo));
    float q1 = __bfloat162float(__high2bfloat16(q_lo));
    float q2 = __bfloat162float(__low2bfloat16(q_hi));
    float q3 = __bfloat162float(__high2bfloat16(q_hi));
    float mn0 = __bfloat162float(__low2bfloat16(mn_lo));
    float mn1 = __bfloat162float(__high2bfloat16(mn_lo));
    float mn2 = __bfloat162float(__low2bfloat16(mn_hi));
    float mn3 = __bfloat162float(__high2bfloat16(mn_hi));
    float mx0 = __bfloat162float(__low2bfloat16(mx_lo));
    float mx1 = __bfloat162float(__high2bfloat16(mx_lo));
    float mx2 = __bfloat162float(__low2bfloat16(mx_hi));
    float mx3 = __bfloat162float(__high2bfloat16(mx_hi));

    float sum = fmaxf(q0 * mn0, q0 * mx0);
    sum += fmaxf(q1 * mn1, q1 * mx1);
    sum += fmaxf(q2 * mn2, q2 * mx2);
    sum += fmaxf(q3 * mn3, q3 * mx3);

    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    return sum;
}

// ============================================================================
// Phase 2 & 4: In-shared-memory radix top-K
// ============================================================================

// Suffix sum (Hillis-Steele) on s_hist[2][RADIX+1]
__device__ void suffix_sum_smem(int s_hist[][RADIX + 1], int tid) {
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (tid < RADIX) {
            int j = 1 << i;
            int src = i & 1;
            int dst = src ^ 1;
            int value = s_hist[src][tid];
            if (tid < RADIX - j) {
                value += s_hist[src][tid + j];
            }
            s_hist[dst][tid] = value;
        }
        __syncthreads();
    }
}

// ============================================================================
// Main fused kernel
// ============================================================================

__global__ void __launch_bounds__(BLOCK_THREADS)
fused_hierarchical_quest_kernel(
    const __nv_bfloat16* __restrict__ q,           // [bs*kv_heads, 128]
    const __nv_bfloat16* __restrict__ large_min,   // [bs*kv_heads, n_large_max, 128]
    const __nv_bfloat16* __restrict__ large_max,   // [bs*kv_heads, n_large_max, 128]
    const __nv_bfloat16* __restrict__ small_min,   // [bs*kv_heads, n_small_max, 128]
    const __nv_bfloat16* __restrict__ small_max,   // [bs*kv_heads, n_small_max, 128]
    int32_t* __restrict__ out_indices,             // [bs*kv_heads, nk_small]
    int n_large,
    int n_large_max,
    int n_small_max,
    int nk_large,
    int nk_small,
    int factor)
{
    const int bh = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane = tid % WARP_SIZE;

    // Shared memory
    extern __shared__ char smem_raw[];
    SmemLayout& sm = *reinterpret_cast<SmemLayout*>(smem_raw);

    const int nk_sub_total = nk_large * factor;

    // ========================================================================
    // Phase 0: Load q to shared memory
    // ========================================================================
    if (tid < HEAD_DIM) {
        sm.s_q[tid] = q[bh * HEAD_DIM + tid];
    }
    __syncthreads();

    // ========================================================================
    // Phase 1: Score large chunks (8 warps cooperatively)
    // ========================================================================
    const __nv_bfloat16* lmin_base = large_min + (int64_t)bh * n_large_max * HEAD_DIM;
    const __nv_bfloat16* lmax_base = large_max + (int64_t)bh * n_large_max * HEAD_DIM;

    for (int base = 0; base < n_large; base += N_WARPS) {
        int chunk_id = base + warp_id;
        if (chunk_id < n_large) {
            const __nv_bfloat16* min_ptr = lmin_base + (int64_t)chunk_id * HEAD_DIM;
            const __nv_bfloat16* max_ptr = lmax_base + (int64_t)chunk_id * HEAD_DIM;

            float score = score_one_chunk_from_smem_q(sm.s_q, min_ptr, max_ptr, lane);

            if (lane == 0) {
                sm.s_scores[chunk_id] = score;
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 2: Radix top-K on s_scores[n_large] → s_large_idx[nk_large]
    // ========================================================================
    {
        int topk_remain = nk_large;
        const int cand_buf_size = min(n_large, MAX_CAND);

        // Round 0: histogram on byte 24
        if (tid < RADIX + 1) {
            sm.s_hist[0][tid] = 0;
            sm.s_hist[1][tid] = 0;
        }
        if (tid == 0) {
            sm.s_counter = 0;
            sm.s_num_input[0] = 0;
            sm.s_threshold_bin = 0;
            sm.s_last_remain = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < n_large; idx += BLOCK_THREADS) {
            uint32_t ord = float_to_ordered(sm.s_scores[idx]);
            uint8_t bin = extract_byte(ord, 24);
            atomicAdd(&sm.s_hist[0][bin], 1);
        }
        __syncthreads();

        suffix_sum_smem(sm.s_hist, tid);

        if (tid < RADIX) {
            int cur = sm.s_hist[0][tid];
            int nxt = sm.s_hist[0][tid + 1];
            if (cur > topk_remain && nxt <= topk_remain) {
                sm.s_threshold_bin = tid;
            }
        }
        __syncthreads();

        int thr0 = sm.s_threshold_bin;
        topk_remain -= sm.s_hist[0][thr0 + 1];

        if (topk_remain == 0) {
            for (int idx = tid; idx < n_large; idx += BLOCK_THREADS) {
                uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                if (extract_byte(ord, 24) > thr0) {
                    int pos = atomicAdd(&sm.s_counter, 1);
                    if (pos < nk_large) sm.s_large_idx[pos] = (int16_t)idx;
                }
            }
            __syncthreads();
            goto phase2_done;
        }

        // Collect > threshold, buffer == threshold candidates
        if (tid < RADIX + 1) {
            sm.s_hist[0][tid] = 0;
            sm.s_hist[1][tid] = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < n_large; idx += BLOCK_THREADS) {
            uint32_t ord = float_to_ordered(sm.s_scores[idx]);
            uint8_t bin = extract_byte(ord, 24);
            if (bin > thr0) {
                int pos = atomicAdd(&sm.s_counter, 1);
                if (pos < nk_large) sm.s_large_idx[pos] = (int16_t)idx;
            } else if (bin == thr0) {
                int cpos = atomicAdd(&sm.s_num_input[0], 1);
                if (cpos < cand_buf_size) {
                    sm.s_cand[0][cpos] = (int16_t)idx;
                    uint8_t sub_bin = extract_byte(ord, 16);
                    atomicAdd(&sm.s_hist[0][sub_bin], 1);
                }
            }
        }
        __syncthreads();

        // Rounds 1-3: refine
        for (int round = 1; round < 4; ++round) {
            int r_idx = (round - 1) & 1;
            int num_input = min((int)sm.s_num_input[r_idx], cand_buf_size);
            int byte_offset = 24 - round * 8;

            suffix_sum_smem(sm.s_hist, tid);

            if (tid < RADIX) {
                int cur = sm.s_hist[0][tid];
                int nxt = sm.s_hist[0][tid + 1];
                if (cur > topk_remain && nxt <= topk_remain) {
                    sm.s_threshold_bin = tid;
                    sm.s_last_remain = topk_remain - nxt;
                }
            }
            if (tid == 0) {
                sm.s_num_input[r_idx ^ 1] = 0;
            }
            __syncthreads();

            int thr = sm.s_threshold_bin;
            topk_remain -= sm.s_hist[0][thr + 1];

            if (topk_remain == 0) {
                for (int i = tid; i < num_input; i += BLOCK_THREADS) {
                    int idx = (int)sm.s_cand[r_idx][i];
                    uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                    if (extract_byte(ord, byte_offset) > thr) {
                        int pos = atomicAdd(&sm.s_counter, 1);
                        if (pos < nk_large) sm.s_large_idx[pos] = (int16_t)idx;
                    }
                }
                __syncthreads();
                goto phase2_done;
            }

            // Reset histogram
            if (tid < RADIX + 1) {
                sm.s_hist[0][tid] = 0;
                sm.s_hist[1][tid] = 0;
            }
            __syncthreads();

            for (int i = tid; i < num_input; i += BLOCK_THREADS) {
                int idx = (int)sm.s_cand[r_idx][i];
                uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                uint8_t bin = extract_byte(ord, byte_offset);
                if (bin > thr) {
                    int pos = atomicAdd(&sm.s_counter, 1);
                    if (pos < nk_large) sm.s_large_idx[pos] = (int16_t)idx;
                } else if (bin == thr) {
                    if (round == 3) {
                        int pos = atomicAdd(&sm.s_last_remain, -1);
                        if (pos > 0) {
                            sm.s_large_idx[nk_large - pos] = (int16_t)idx;
                        }
                    } else {
                        int cpos = atomicAdd(&sm.s_num_input[r_idx ^ 1], 1);
                        if (cpos < cand_buf_size) {
                            sm.s_cand[r_idx ^ 1][cpos] = (int16_t)idx;
                            uint8_t sub_bin = extract_byte(ord, byte_offset - 8);
                            atomicAdd(&sm.s_hist[0][sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }
    phase2_done:
    __syncthreads();

    // ========================================================================
    // Phase 3: Score sub-chunks of selected large chunks
    // ========================================================================
    const __nv_bfloat16* smin_base = small_min + (int64_t)bh * n_small_max * HEAD_DIM;
    const __nv_bfloat16* smax_base = small_max + (int64_t)bh * n_small_max * HEAD_DIM;

    for (int base = 0; base < nk_sub_total; base += N_WARPS) {
        int sub_id = base + warp_id;
        if (sub_id < nk_sub_total) {
            int sel = sub_id / factor;
            int sub_off = sub_id % factor;
            int large_chunk_id = (int)sm.s_large_idx[sel];
            int small_chunk_id = large_chunk_id * factor + sub_off;

            float score;
            if (small_chunk_id >= 0 && small_chunk_id < n_small_max) {
                const __nv_bfloat16* min_ptr = smin_base + (int64_t)small_chunk_id * HEAD_DIM;
                const __nv_bfloat16* max_ptr = smax_base + (int64_t)small_chunk_id * HEAD_DIM;
                score = score_one_chunk_from_smem_q(sm.s_q, min_ptr, max_ptr, lane);
            } else {
                score = -INFINITY;
            }

            if (lane == 0) {
                sm.s_scores[sub_id] = score;
            }
        }
    }
    __syncthreads();

    // ========================================================================
    // Phase 4: Radix top-K on s_scores[nk_sub_total] with fused map-back
    // ========================================================================
    {
        int topk_remain = nk_small;
        const int cand_buf_size = min(nk_sub_total, MAX_CAND);
        int32_t* out_ptr = out_indices + bh * nk_small;

        // Round 0
        if (tid < RADIX + 1) {
            sm.s_hist[0][tid] = 0;
            sm.s_hist[1][tid] = 0;
        }
        if (tid == 0) {
            sm.s_counter = 0;
            sm.s_num_input[0] = 0;
            sm.s_threshold_bin = 0;
            sm.s_last_remain = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < nk_sub_total; idx += BLOCK_THREADS) {
            uint32_t ord = float_to_ordered(sm.s_scores[idx]);
            uint8_t bin = extract_byte(ord, 24);
            atomicAdd(&sm.s_hist[0][bin], 1);
        }
        __syncthreads();

        suffix_sum_smem(sm.s_hist, tid);

        if (tid < RADIX) {
            int cur = sm.s_hist[0][tid];
            int nxt = sm.s_hist[0][tid + 1];
            if (cur > topk_remain && nxt <= topk_remain) {
                sm.s_threshold_bin = tid;
            }
        }
        __syncthreads();

        int thr0 = sm.s_threshold_bin;
        topk_remain -= sm.s_hist[0][thr0 + 1];

        // Macro for map-back: sub_idx → global small chunk index
        #define EMIT_FINAL(pos, sub_idx) do {                       \
            int _sel = (sub_idx) / factor;                          \
            int _sub_off = (sub_idx) % factor;                      \
            int _large_id = (int)sm.s_large_idx[_sel];              \
            out_ptr[(pos)] = _large_id * factor + _sub_off;         \
        } while(0)

        if (topk_remain == 0) {
            for (int idx = tid; idx < nk_sub_total; idx += BLOCK_THREADS) {
                uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                if (extract_byte(ord, 24) > thr0) {
                    int pos = atomicAdd(&sm.s_counter, 1);
                    if (pos < nk_small) EMIT_FINAL(pos, idx);
                }
            }
            __syncthreads();
            goto phase4_done;
        }

        // Collect > threshold, buffer == threshold
        if (tid < RADIX + 1) {
            sm.s_hist[0][tid] = 0;
            sm.s_hist[1][tid] = 0;
        }
        __syncthreads();

        for (int idx = tid; idx < nk_sub_total; idx += BLOCK_THREADS) {
            uint32_t ord = float_to_ordered(sm.s_scores[idx]);
            uint8_t bin = extract_byte(ord, 24);
            if (bin > thr0) {
                int pos = atomicAdd(&sm.s_counter, 1);
                if (pos < nk_small) EMIT_FINAL(pos, idx);
            } else if (bin == thr0) {
                int cpos = atomicAdd(&sm.s_num_input[0], 1);
                if (cpos < cand_buf_size) {
                    sm.s_cand[0][cpos] = (int16_t)idx;
                    uint8_t sub_bin = extract_byte(ord, 16);
                    atomicAdd(&sm.s_hist[0][sub_bin], 1);
                }
            }
        }
        __syncthreads();

        // Rounds 1-3
        for (int round = 1; round < 4; ++round) {
            int r_idx = (round - 1) & 1;
            int num_input = min((int)sm.s_num_input[r_idx], cand_buf_size);
            int byte_offset = 24 - round * 8;

            suffix_sum_smem(sm.s_hist, tid);

            if (tid < RADIX) {
                int cur = sm.s_hist[0][tid];
                int nxt = sm.s_hist[0][tid + 1];
                if (cur > topk_remain && nxt <= topk_remain) {
                    sm.s_threshold_bin = tid;
                    sm.s_last_remain = topk_remain - nxt;
                }
            }
            if (tid == 0) {
                sm.s_num_input[r_idx ^ 1] = 0;
            }
            __syncthreads();

            int thr = sm.s_threshold_bin;
            topk_remain -= sm.s_hist[0][thr + 1];

            if (topk_remain == 0) {
                for (int i = tid; i < num_input; i += BLOCK_THREADS) {
                    int idx = (int)sm.s_cand[r_idx][i];
                    uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                    if (extract_byte(ord, byte_offset) > thr) {
                        int pos = atomicAdd(&sm.s_counter, 1);
                        if (pos < nk_small) EMIT_FINAL(pos, idx);
                    }
                }
                __syncthreads();
                goto phase4_done;
            }

            // Reset histogram
            if (tid < RADIX + 1) {
                sm.s_hist[0][tid] = 0;
                sm.s_hist[1][tid] = 0;
            }
            __syncthreads();

            for (int i = tid; i < num_input; i += BLOCK_THREADS) {
                int idx = (int)sm.s_cand[r_idx][i];
                uint32_t ord = float_to_ordered(sm.s_scores[idx]);
                uint8_t bin = extract_byte(ord, byte_offset);
                if (bin > thr) {
                    int pos = atomicAdd(&sm.s_counter, 1);
                    if (pos < nk_small) EMIT_FINAL(pos, idx);
                } else if (bin == thr) {
                    if (round == 3) {
                        int pos = atomicAdd(&sm.s_last_remain, -1);
                        if (pos > 0) {
                            EMIT_FINAL(nk_small - pos, idx);
                        }
                    } else {
                        int cpos = atomicAdd(&sm.s_num_input[r_idx ^ 1], 1);
                        if (cpos < cand_buf_size) {
                            sm.s_cand[r_idx ^ 1][cpos] = (int16_t)idx;
                            uint8_t sub_bin = extract_byte(ord, byte_offset - 8);
                            atomicAdd(&sm.s_hist[0][sub_bin], 1);
                        }
                    }
                }
            }
            __syncthreads();
        }
        #undef EMIT_FINAL
    }
    phase4_done:
    ;
}

}  // namespace

// ============================================================================
// C++ wrapper
// ============================================================================

void fused_hierarchical_quest_cuda(
    torch::Tensor q,           // [bs, kv_heads, 128] bf16
    torch::Tensor large_min,   // [bs, kv_heads, n_large_max, 128] bf16
    torch::Tensor large_max,
    torch::Tensor small_min,   // [bs, kv_heads, n_small_max, 128] bf16
    torch::Tensor small_max,
    torch::Tensor out,         // [bs, kv_heads, nk_small] int32
    int64_t n_large,
    int64_t n_large_max,
    int64_t n_small_max,
    int64_t nk_large,
    int64_t nk_small,
    int64_t factor)
{
    TORCH_CHECK(q.is_cuda() && q.is_contiguous(), "q must be contiguous CUDA bf16");
    TORCH_CHECK(large_min.is_cuda() && large_min.is_contiguous(), "large_min must be contiguous CUDA");
    TORCH_CHECK(large_max.is_cuda() && large_max.is_contiguous(), "large_max must be contiguous CUDA");
    TORCH_CHECK(small_min.is_cuda() && small_min.is_contiguous(), "small_min must be contiguous CUDA");
    TORCH_CHECK(small_max.is_cuda() && small_max.is_contiguous(), "small_max must be contiguous CUDA");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous(), "out must be contiguous CUDA int32");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16, "q must be bf16");
    TORCH_CHECK(large_min.scalar_type() == at::ScalarType::BFloat16, "large_min must be bf16");
    TORCH_CHECK(large_max.scalar_type() == at::ScalarType::BFloat16, "large_max must be bf16");
    TORCH_CHECK(small_min.scalar_type() == at::ScalarType::BFloat16, "small_min must be bf16");
    TORCH_CHECK(small_max.scalar_type() == at::ScalarType::BFloat16, "small_max must be bf16");
    TORCH_CHECK(out.scalar_type() == at::ScalarType::Int, "out must be int32");
    TORCH_CHECK(n_large <= MAX_LARGE_CHUNKS, "n_large exceeds maximum");
    TORCH_CHECK(n_large <= n_large_max, "n_large must <= n_large_max");
    TORCH_CHECK(nk_large * factor <= MAX_SUB_TOTAL, "nk_sub_total exceeds maximum");
    TORCH_CHECK(nk_large <= MAX_NK_LARGE, "nk_large exceeds maximum");
    TORCH_CHECK(nk_large <= n_large, "nk_large must <= n_large");
    TORCH_CHECK(nk_small <= nk_large * factor, "nk_small must <= nk_large * factor");

    const int bs = q.size(0);
    const int kv_heads = q.size(1);
    const int B = bs * kv_heads;

    size_t smem_bytes = sizeof(SmemLayout);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (smem_bytes > 48 * 1024) {
        cudaFuncSetAttribute(
            fused_hierarchical_quest_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_bytes);
    }

    fused_hierarchical_quest_kernel<<<B, BLOCK_THREADS, smem_bytes, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(large_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(large_max.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(small_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(small_max.data_ptr()),
        out.data_ptr<int32_t>(),
        (int)n_large, (int)n_large_max, (int)n_small_max,
        (int)nk_large, (int)nk_small, (int)factor);

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_hierarchical_quest", &fused_hierarchical_quest_cuda,
          "Fused hierarchical Quest: score + topk + sub-score + topk (CUDA)",
          py::arg("q"), py::arg("large_min"), py::arg("large_max"),
          py::arg("small_min"), py::arg("small_max"), py::arg("out"),
          py::arg("n_large"), py::arg("n_large_max"), py::arg("n_small_max"),
          py::arg("nk_large"), py::arg("nk_small"), py::arg("factor"));
}
