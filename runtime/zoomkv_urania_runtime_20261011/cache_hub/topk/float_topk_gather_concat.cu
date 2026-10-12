/**
 * Fused float top-k + indirect gather + skipped-tail concat.
 *
 * Replaces the 3-launch chain in stage-2 cds_dense_sparse_select tail:
 *     topk_pos = float_topk_3d(out_scores, k)              # [B, k]
 *     final    = gather(out_indices, dim=-1, index=topk_pos)  # [B, k]
 *     final    = cat([final, skipped], dim=-1)             # [B, k+sk]
 *
 * with a single CUDA launch (caller pre-allocates `out`, no allocations on
 * the decode hot path → CUDA-Graph friendly):
 *     out[b, 0:k]      = lut[b, top_k_argmax(scores[b])]
 *     out[b, k:k+sk]   = skipped[b, 0:sk]
 *
 * Algorithm: same 4-round 8-bit radix select as float_topk.cu (sglang-style).
 * Two modifications relative to the baseline kernel:
 *   1. every position-write becomes ``out[pos] = lut[idx]`` (indirect load
 *      from the int64 token-id LUT instead of writing the in-row index).
 *   2. the skipped tail is copied at kernel entry by all threads, before any
 *      __syncthreads(); the writes target out[k..k+sk] which is disjoint from
 *      the topk write range out[0..k], so no fence is needed and the copy
 *      overlaps with histogram construction.
 *
 * sk_count == 0 (steady-state when no chunk is mid-build) is handled by the
 * loop bound — the kernel issues no extra branches in that case.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_SIZE = 1024;
constexpr int RADIX = 256;

__device__ __forceinline__ uint32_t float_to_ordered_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint8_t extract_byte(uint32_t x, int byte_offset) {
    return static_cast<uint8_t>((x >> byte_offset) & 0xFF);
}

// Naive path for length <= k.  Pads unused slots with the row-0 sentinel
// (NOT -1; see the sentinel-init note in the kernel — ``out`` here holds final
// token ids that h2d_gather_kv consumes directly as a gather row offset, so the
// pad must stay in range).  Caller is expected to ensure length > k in
// production (n_pos = nk * group size ≫ final_topk in stage-2 cds; this branch
// never fires there).
__device__ void naive_topk_gather(
    int64_t* __restrict__ out,
    const int64_t* __restrict__ lut,
    int32_t length,
    int32_t k)
{
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        out[i] = (i < length) ? lut[i] : 0;
    }
}

__global__ void float_topk_gather_concat_kernel(
    const float*   __restrict__ scores,    // [B, L]
    const int64_t* __restrict__ lut,       // [B, L]
    const int64_t* __restrict__ skipped,   // [B, sk_count]; may be empty (sk_count == 0)
    int64_t*       __restrict__ out,       // [B, out_stride]; rows store [k tops | sk_count skipped]
    int64_t scores_stride,
    int64_t lut_stride,
    int64_t skipped_stride,
    int64_t out_stride,
    int32_t length,
    int32_t k,
    int32_t sk_count)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const float*   in      = scores + bid * scores_stride;
    const int64_t* lut_row = lut + bid * lut_stride;
    int64_t*       out_row = out + bid * out_stride;

    // 0) Sentinel init for the top-k region [0, k).  Synced from float_topk.cu:
    //    the radix-select rounds below can leave some of the k slots unwritten
    //    when fewer than k candidates survive (duplicate-heavy scores / ties at
    //    the final-round threshold).  ``out`` is a reused pre-allocated buffer,
    //    so an unwritten slot would otherwise carry a stale token id from a
    //    previous decode step and silently corrupt the gather.
    //
    //    NB: the sentinel value here is 0, NOT the -1 used in float_topk.cu.
    //    float_topk.cu emits *positions* that downstream consumers (quest_map_
    //    back / torch.gather) explicitly skip on -1.  This fused kernel instead
    //    emits *final token ids* that h2d_gather_kv reads directly as a source
    //    row offset (index * uint4_per_dim); a -1 would index one row before the
    //    buffer for (b=0, h=0) → illegal memory access.  Row 0 (the sink token)
    //    is always valid and benign — an under-filled slot just re-attends an
    //    already-attended sink row, far safer than garbage or a crash.
    //    The skipped tail [k, k+sk) is written explicitly below, so it is left
    //    untouched here.  Single coalesced pass of k writes — negligible.
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        out_row[i] = 0;
    }
    __syncthreads();

    // 1) Skipped-tail copy.  Disjoint write range vs topk; no sync needed.
    //    Loop is no-op when sk_count == 0 (steady state for most decode steps).
    if (sk_count > 0) {
        const int64_t* sk_row = skipped + bid * skipped_stride;
        for (int i = tid; i < sk_count; i += BLOCK_SIZE) {
            out_row[k + i] = sk_row[i];
        }
    }

    // 2) Top-k radix select.  Identical 4-round structure to float_topk.cu.
    if (length <= k) {
        naive_topk_gather(out_row, lut_row, length, k);
        return;
    }

    __shared__ int s_histogram_buf[2][RADIX + 1];
    extern __shared__ int s_input_idx[];

    __shared__ int s_counter;
    __shared__ int s_threshold_bin;
    __shared__ int s_num_input[2];
    __shared__ int s_last_remain;

    int* s_histogram = s_histogram_buf[0];
    const int smem_idx_size = (64 * 1024 / sizeof(int)) / 2;

    int topk_remain = k;

    // ===== Round 0: 8-bit coarse histogram (bits 24-31) =====
    if (tid < RADIX + 1) {
        s_histogram_buf[0][tid] = 0;
        s_histogram_buf[1][tid] = 0;
    }
    if (tid == 0) {
        s_counter = 0;
        s_num_input[0] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        uint32_t ordered = float_to_ordered_uint32(in[idx]);
        uint8_t bin = extract_byte(ordered, 24);
        atomicAdd(&s_histogram_buf[0][bin], 1);
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (tid < RADIX) {
            int j = 1 << i;
            int k_idx = i & 1;
            int value = s_histogram_buf[k_idx][tid];
            if (tid < RADIX - j) {
                value += s_histogram_buf[k_idx][tid + j];
            }
            s_histogram_buf[k_idx ^ 1][tid] = value;
        }
        __syncthreads();
    }

    s_histogram = s_histogram_buf[0];

    if (tid < RADIX && s_histogram[tid] > topk_remain && s_histogram[tid + 1] <= topk_remain) {
        s_threshold_bin = tid;
    }
    __syncthreads();

    const int threshold_bin_r0 = s_threshold_bin;
    topk_remain -= s_histogram[threshold_bin_r0 + 1];

    if (topk_remain == 0) {
        for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
            uint32_t ordered = float_to_ordered_uint32(in[idx]);
            uint8_t bin = extract_byte(ordered, 24);
            if (bin > threshold_bin_r0) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out_row[pos] = lut_row[idx];
            }
        }
        return;
    }

    if (tid < RADIX + 1) {
        s_histogram_buf[0][tid] = 0;
        s_histogram_buf[1][tid] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        float val = in[idx];
        uint32_t ordered = float_to_ordered_uint32(val);
        uint8_t bin = extract_byte(ordered, 24);

        if (bin > threshold_bin_r0) {
            int pos = atomicAdd(&s_counter, 1);
            if (pos < k) out_row[pos] = lut_row[idx];
        } else if (bin == threshold_bin_r0) {
            int pos = atomicAdd(&s_num_input[0], 1);
            if (pos < smem_idx_size) {
                s_input_idx[pos] = idx;
                uint8_t sub_bin = extract_byte(ordered, 16);
                atomicAdd(&s_histogram_buf[0][sub_bin], 1);
            }
        }
    }
    __syncthreads();

    // ===== Rounds 1-3 =====
    #pragma unroll
    for (int round = 1; round < 4; ++round) {
        const int r_idx = (round - 1) % 2;
        const int num_input = min(s_num_input[r_idx], smem_idx_size);
        const int byte_offset = 24 - round * 8;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (tid < RADIX) {
                int j = 1 << i;
                int k_idx = i & 1;
                int value = s_histogram_buf[k_idx][tid];
                if (tid < RADIX - j) {
                    value += s_histogram_buf[k_idx][tid + j];
                }
                s_histogram_buf[k_idx ^ 1][tid] = value;
            }
            __syncthreads();
        }

        s_histogram = s_histogram_buf[0];

        if (tid < RADIX && s_histogram[tid] > topk_remain && s_histogram[tid + 1] <= topk_remain) {
            s_threshold_bin = tid;
            s_last_remain = topk_remain - s_histogram[tid + 1];
        }
        if (tid == 0) {
            s_num_input[r_idx ^ 1] = 0;
        }
        __syncthreads();

        const int threshold_bin = s_threshold_bin;
        topk_remain -= s_histogram[threshold_bin + 1];

        if (topk_remain == 0) {
            for (int i = tid; i < num_input; i += BLOCK_SIZE) {
                int idx = s_input_idx[r_idx * smem_idx_size + i];
                uint32_t ordered = float_to_ordered_uint32(in[idx]);
                uint8_t bin = extract_byte(ordered, byte_offset);
                if (bin > threshold_bin) {
                    int pos = atomicAdd(&s_counter, 1);
                    if (pos < k) out_row[pos] = lut_row[idx];
                }
            }
            return;
        }

        if (tid < RADIX + 1) {
            s_histogram_buf[0][tid] = 0;
            s_histogram_buf[1][tid] = 0;
        }
        __syncthreads();

        for (int i = tid; i < num_input; i += BLOCK_SIZE) {
            int idx = s_input_idx[r_idx * smem_idx_size + i];
            float val = in[idx];
            uint32_t ordered = float_to_ordered_uint32(val);
            uint8_t bin = extract_byte(ordered, byte_offset);

            if (bin > threshold_bin) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out_row[pos] = lut_row[idx];
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    // Last round: fill remaining slots from the back.
                    int pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        out_row[k - pos] = lut_row[idx];
                    }
                } else {
                    int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (pos < smem_idx_size) {
                        s_input_idx[(r_idx ^ 1) * smem_idx_size + pos] = idx;
                        uint8_t sub_bin = extract_byte(ordered, byte_offset - 8);
                        atomicAdd(&s_histogram_buf[0][sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace


// =====================================================================
// Python Interface
// =====================================================================

void float_topk_gather_concat_3d_cuda(
    torch::Tensor scores,    // [bs, kv, L] fp32
    torch::Tensor lut,       // [bs, kv, L] int64
    torch::Tensor skipped,   // [bs, kv, sk] int64 (sk may be 0)
    torch::Tensor out,       // [bs, kv, >= k + sk] int64 (caller pre-allocated)
    int64_t k)
{
    TORCH_CHECK(scores.is_cuda() && lut.is_cuda() && skipped.is_cuda() && out.is_cuda(),
                "all tensors must be on CUDA");
    TORCH_CHECK(scores.dim() == 3, "scores must be 3D [bs, kv, L]");
    TORCH_CHECK(lut.dim() == 3, "lut must be 3D [bs, kv, L]");
    TORCH_CHECK(skipped.dim() == 3, "skipped must be 3D [bs, kv, sk]");
    TORCH_CHECK(out.dim() == 3, "out must be 3D [bs, kv, k+sk]");
    TORCH_CHECK(scores.scalar_type() == torch::kFloat32, "scores must be fp32");
    TORCH_CHECK(lut.scalar_type() == torch::kInt64, "lut must be int64");
    TORCH_CHECK(skipped.scalar_type() == torch::kInt64, "skipped must be int64");
    TORCH_CHECK(out.scalar_type() == torch::kInt64, "out must be int64");
    TORCH_CHECK(scores.sizes() == lut.sizes(),
                "scores and lut must have matching shape");
    TORCH_CHECK(skipped.size(0) == scores.size(0) && skipped.size(1) == scores.size(1),
                "skipped (bs, kv) must match scores");
    TORCH_CHECK(out.size(0) == scores.size(0) && out.size(1) == scores.size(1),
                "out (bs, kv) must match scores");

    const int bs = scores.size(0);
    const int kv = scores.size(1);
    const int L = scores.size(2);
    const int sk = skipped.size(2);
    TORCH_CHECK(k > 0 && k <= L, "k out of range");
    TORCH_CHECK(out.size(2) >= k + sk, "out last dim must be >= k + sk_count");

    auto scores_c  = scores.contiguous();
    auto lut_c     = lut.contiguous();
    auto skipped_c = skipped.contiguous();
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous (alias-target buffer)");

    const int B = bs * kv;

    c10::cuda::CUDAGuard device_guard(scores.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    constexpr size_t smem = 64 * 1024;
    auto err_attr = cudaFuncSetAttribute(
        float_topk_gather_concat_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(smem));
    TORCH_CHECK(err_attr == cudaSuccess,
                "cudaFuncSetAttribute failed: ", cudaGetErrorString(err_attr));

    const int64_t scores_row_stride  = L;
    const int64_t lut_row_stride     = L;
    const int64_t skipped_row_stride = (sk > 0) ? static_cast<int64_t>(sk) : 0;
    const int64_t out_row_stride     = out.size(2);

    float_topk_gather_concat_kernel<<<B, BLOCK_SIZE, smem, stream>>>(
        scores_c.data_ptr<float>(),
        lut_c.data_ptr<int64_t>(),
        skipped_c.data_ptr<int64_t>(),
        out.data_ptr<int64_t>(),
        scores_row_stride,
        lut_row_stride,
        skipped_row_stride,
        out_row_stride,
        L,
        static_cast<int>(k),
        sk
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "kernel failed: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float_topk_gather_concat_3d", &float_topk_gather_concat_3d_cuda,
          "Fused float top-k + indirect gather + skipped concat (3D, in-place out)",
          py::arg("scores"), py::arg("lut"), py::arg("skipped"), py::arg("out"), py::arg("k"));
}
