/**
 * Float Top-K v2: radix select with global-memory candidate buffers.
 *
 * Fixes the shared-memory overflow in float_topk.cu when scores are highly
 * clustered (e.g. Quest bbox upper-bound scores where thousands of values
 * share the same high byte).
 *
 * Change from v1: candidate index buffers moved from shared memory (3840-slot
 * limit) to a pre-allocated global-memory workspace (length slots per buffer).
 * Histogram and prefix-sum still run in shared memory.
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

__device__ void naive_topk(int64_t* __restrict__ indices, int32_t length, int32_t k) {
    const int tid = threadIdx.x;
    for (int i = tid; i < k; i += BLOCK_SIZE) {
        indices[i] = (i < length) ? i : -1;
    }
}

/**
 * Find the threshold bin in a suffix-sum histogram.
 *
 * s_histogram[tid] = count of elements with byte >= tid (non-increasing).
 * Returns the bin where the suffix sum first exceeds topk_remain, i.e.
 *   s_histogram[bin] > topk_remain && s_histogram[bin+1] <= topk_remain.
 *
 * Handles tid == RADIX-1 correctly (s_histogram[RADIX] is defined as 0).
 */
__device__ void find_threshold(
    const int* __restrict__ s_histogram,
    int topk_remain,
    int* __restrict__ s_threshold_bin)
{
    const int tid = threadIdx.x;
    if (tid < RADIX) {
        int cur = s_histogram[tid];
        int nxt = (tid < RADIX - 1) ? s_histogram[tid + 1] : 0;
        if (cur > topk_remain && nxt <= topk_remain) {
            *s_threshold_bin = tid;
        }
    }
}

__global__ void float_topk_v2_kernel(
    const float* __restrict__ input,   // [B, L]
    int64_t* __restrict__ indices,     // [B, k]
    int32_t* __restrict__ workspace,   // [B, 2*L]  candidate index double-buffer
    int64_t input_stride,
    int32_t length,
    int32_t k)
{
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    const float* in = input + bid * input_stride;
    int64_t* out = indices + bid * k;

    if (length <= k) {
        naive_topk(out, length, k);
        return;
    }

    // Global-memory double buffer for candidate indices (no size limit)
    int32_t* g_buf = workspace + static_cast<int64_t>(bid) * 2 * length;

    // Shared memory: histogram only (~2 KB)
    __shared__ int s_histogram_buf[2 * RADIX];
    __shared__ int s_counter;
    __shared__ int s_threshold_bin;
    __shared__ int s_num_input[2];
    __shared__ int s_last_remain;

    int* s_histogram = s_histogram_buf;
    int topk_remain = k;

    // ===== Round 0: byte 24 =====
    if (tid < RADIX) {
        s_histogram[tid] = 0;
        s_histogram[tid + RADIX] = 0;
    }
    if (tid == 0) {
        s_counter = 0;
        s_num_input[0] = 0;
    }
    __syncthreads();

    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        uint32_t ordered = float_to_ordered_uint32(in[idx]);
        uint8_t bin = extract_byte(ordered, 24);
        atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    // Suffix sum
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (tid < RADIX) {
            int j = 1 << i;
            int k_idx = i & 1;
            int value = s_histogram_buf[k_idx * RADIX + tid];
            if (tid < RADIX - j) {
                value += s_histogram_buf[k_idx * RADIX + tid + j];
            }
            s_histogram_buf[(k_idx ^ 1) * RADIX + tid] = value;
        }
        __syncthreads();
    }

    find_threshold(s_histogram, topk_remain, &s_threshold_bin);
    __syncthreads();

    int threshold_bin = s_threshold_bin;
    int above_count = (threshold_bin < RADIX - 1) ? s_histogram[threshold_bin + 1] : 0;
    topk_remain -= above_count;

    // Early exit: all top-k elements are strictly above threshold
    if (topk_remain == 0) {
        for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
            uint32_t ordered = float_to_ordered_uint32(in[idx]);
            uint8_t bin = extract_byte(ordered, 24);
            if (bin > threshold_bin) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out[pos] = idx;
            }
        }
        return;
    }

    // Reset histogram for round 1
    if (tid < RADIX) {
        s_histogram[tid] = 0;
        s_histogram[tid + RADIX] = 0;
    }
    __syncthreads();

    // Write above-threshold to output; collect at-threshold into g_buf[0]
    for (int idx = tid; idx < length; idx += BLOCK_SIZE) {
        float val = in[idx];
        uint32_t ordered = float_to_ordered_uint32(val);
        uint8_t bin = extract_byte(ordered, 24);

        if (bin > threshold_bin) {
            int pos = atomicAdd(&s_counter, 1);
            if (pos < k) out[pos] = idx;
        } else if (bin == threshold_bin) {
            int pos = atomicAdd(&s_num_input[0], 1);
            g_buf[pos] = idx;
            uint8_t sub_bin = extract_byte(ordered, 16);
            atomicAdd(&s_histogram[sub_bin], 1);
        }
    }
    __syncthreads();

    // ===== Rounds 1-3 =====
    #pragma unroll
    for (int round = 1; round < 4; ++round) {
        const int r_idx = (round - 1) % 2;
        const int num_input = s_num_input[r_idx];
        const int byte_offset = 24 - round * 8;

        int32_t* cur_buf = g_buf + r_idx * length;
        int32_t* nxt_buf = g_buf + (r_idx ^ 1) * length;

        // Suffix sum
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (tid < RADIX) {
                int j = 1 << i;
                int k_idx = i & 1;
                int value = s_histogram_buf[k_idx * RADIX + tid];
                if (tid < RADIX - j) {
                    value += s_histogram_buf[k_idx * RADIX + tid + j];
                }
                s_histogram_buf[(k_idx ^ 1) * RADIX + tid] = value;
            }
            __syncthreads();
        }

        find_threshold(s_histogram, topk_remain, &s_threshold_bin);
        if (tid == 0) {
            s_num_input[r_idx ^ 1] = 0;
        }
        __syncthreads();

        threshold_bin = s_threshold_bin;
        above_count = (threshold_bin < RADIX - 1) ? s_histogram[threshold_bin + 1] : 0;
        topk_remain -= above_count;

        if (topk_remain == 0) {
            for (int i = tid; i < num_input; i += BLOCK_SIZE) {
                int idx = cur_buf[i];
                uint32_t ordered = float_to_ordered_uint32(in[idx]);
                uint8_t bin = extract_byte(ordered, byte_offset);
                if (bin > threshold_bin) {
                    int pos = atomicAdd(&s_counter, 1);
                    if (pos < k) out[pos] = idx;
                }
            }
            return;
        }

        // Reset histogram
        if (tid < RADIX) {
            s_histogram[tid] = 0;
            s_histogram[tid + RADIX] = 0;
        }
        __syncthreads();

        for (int i = tid; i < num_input; i += BLOCK_SIZE) {
            int idx = cur_buf[i];
            float val = in[idx];
            uint32_t ordered = float_to_ordered_uint32(val);
            uint8_t bin = extract_byte(ordered, byte_offset);

            if (bin > threshold_bin) {
                int pos = atomicAdd(&s_counter, 1);
                if (pos < k) out[pos] = idx;
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    int pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        out[k - pos] = idx;
                    }
                } else {
                    int pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    nxt_buf[pos] = idx;
                    uint8_t sub_bin = extract_byte(ordered, byte_offset - 8);
                    atomicAdd(&s_histogram[sub_bin], 1);
                }
            }
        }
        __syncthreads();
    }
}

}  // namespace


// =====================================================================
// Python interface
// =====================================================================

torch::Tensor float_topk_v2_cuda(
    torch::Tensor input,
    int64_t k)
{
    TORCH_CHECK(input.is_cuda(), "input must be on CUDA");
    TORCH_CHECK(input.dim() == 2, "input must be 2D [B, L]");
    TORCH_CHECK(k > 0, "k must be positive");

    const int B = input.size(0);
    const int L = input.size(1);

    TORCH_CHECK(k <= L, "k cannot exceed sequence length");

    torch::Tensor input_float;
    if (input.scalar_type() != torch::kFloat32) {
        input_float = input.to(torch::kFloat32).contiguous();
    } else {
        input_float = input.contiguous();
    }

    auto indices = torch::empty({B, k},
        torch::TensorOptions().dtype(torch::kInt64).device(input.device()));

    auto workspace = torch::empty({B, 2 * L},
        torch::TensorOptions().dtype(torch::kInt32).device(input.device()));

    c10::cuda::CUDAGuard device_guard(input.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    float_topk_v2_kernel<<<B, BLOCK_SIZE, 0, stream>>>(
        input_float.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        workspace.data_ptr<int32_t>(),
        input_float.stride(0),
        L,
        k
    );

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "float_topk_v2 kernel failed: ", cudaGetErrorString(err));

    return indices;
}


torch::Tensor float_topk_v2_3d_cuda(
    torch::Tensor input,
    int64_t k)
{
    TORCH_CHECK(input.dim() == 3, "input must be 3D [bs, kv_heads, kv_len]");

    const int bs = input.size(0);
    const int kv_heads = input.size(1);
    const int kv_len = input.size(2);

    auto input_2d = input.reshape({bs * kv_heads, kv_len});
    auto indices_2d = float_topk_v2_cuda(input_2d, k);
    return indices_2d.reshape({bs, kv_heads, k});
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("float_topk_v2", &float_topk_v2_cuda,
          "Float Top-K v2 with global-memory candidate buffers (2D input)",
          py::arg("input"), py::arg("k"));
    m.def("float_topk_v2_3d", &float_topk_v2_3d_cuda,
          "Float Top-K v2 with global-memory candidate buffers (3D input)",
          py::arg("input"), py::arg("k"));
}
