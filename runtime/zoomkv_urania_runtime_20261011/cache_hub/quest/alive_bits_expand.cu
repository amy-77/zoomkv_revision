#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdint>

constexpr int BASE_CHUNK = 8;
constexpr int BLOCK_DIM = 256;

/* One block per query, 256 threads collaboratively expand bitmap.
   Phase 1: each thread popcounts its word range
   Phase 2: exclusive prefix-sum → per-thread write offset
   Phase 3: each thread scatters its token indices */
__global__ void alive_bits_expand_kernel(
    const int32_t* __restrict__ alive_bits,   // [Q, W]
    int64_t* __restrict__ indices,            // [Q, buf_width]
    int32_t* __restrict__ counts,             // [Q]
    int W, int n_base, int n_tokens, int buf_width)
{
    const int q = blockIdx.x;
    const int tid = threadIdx.x;

    extern __shared__ int smem[];             // BLOCK_DIM + 1

    const auto* bits = reinterpret_cast<const uint32_t*>(alive_bits + q * W);
    int64_t* out = indices + static_cast<int64_t>(q) * buf_width;

    const int wpt = (W + BLOCK_DIM - 1) / BLOCK_DIM;
    const int w_begin = tid * wpt;
    const int w_end   = min(w_begin + wpt, W);

    /* --- Phase 1: popcount --- */
    int my_count = 0;
    for (int w = w_begin; w < w_end; ++w) {
        int first_bit = w * 32;
        if (first_bit >= n_base) break;
        uint32_t word = bits[w];
        if (first_bit + 32 > n_base) {
            int valid = n_base - first_bit;
            word &= (1u << valid) - 1;
        }
        my_count += __popc(word) * BASE_CHUNK;
    }
    smem[tid] = my_count;
    __syncthreads();

    /* --- Phase 2: exclusive prefix-sum (sequential, 256 iters) --- */
    if (tid == 0) {
        int acc = 0;
        for (int i = 0; i < BLOCK_DIM; ++i) {
            int c = smem[i];
            smem[i] = acc;
            acc += c;
        }
        int trailing = n_tokens - n_base * BASE_CHUNK;
        if (trailing < 0) trailing = 0;
        counts[q] = acc + trailing;
        smem[BLOCK_DIM] = acc;
    }
    __syncthreads();

    /* --- Phase 3: scatter indices --- */
    int pos = smem[tid];
    for (int w = w_begin; w < w_end; ++w) {
        int first_bit = w * 32;
        if (first_bit >= n_base) break;
        uint32_t word = bits[w];
        if (first_bit + 32 > n_base) {
            int valid = n_base - first_bit;
            word &= (1u << valid) - 1;
        }
        while (word != 0) {
            int bit = __ffs(word) - 1;
            int base_token = (first_bit + bit) * BASE_CHUNK;
            #pragma unroll
            for (int t = 0; t < BASE_CHUNK; ++t)
                out[pos++] = base_token + t;
            word &= word - 1;
        }
    }

    /* trailing tokens (at most BASE_CHUNK-1, thread 0 only) */
    if (tid == 0) {
        int tail_start = smem[BLOCK_DIM];
        int usable = n_base * BASE_CHUNK;
        for (int t = usable; t < n_tokens; ++t)
            out[tail_start++] = t;
    }
}

/* Host entry — caller supplies pre-allocated buffers for zero-alloc fast path */
std::tuple<torch::Tensor, torch::Tensor> alive_bits_to_indices(
    torch::Tensor alive_bits,   // [Q, W] int32
    int n_base,
    int n_tokens,
    torch::Tensor indices_buf,  // [Q, buf_width] int64  (pre-allocated)
    torch::Tensor counts_buf)   // [Q] int32             (pre-allocated)
{
    TORCH_CHECK(alive_bits.is_cuda(), "alive_bits must be on CUDA");
    TORCH_CHECK(alive_bits.dtype() == torch::kInt32, "alive_bits must be int32");

    const int Q = alive_bits.size(0);
    const int W = alive_bits.size(1);
    const int buf_width = indices_buf.size(1);
    const int smem_bytes = (BLOCK_DIM + 1) * sizeof(int);

    alive_bits_expand_kernel<<<Q, BLOCK_DIM, smem_bytes>>>(
        alive_bits.data_ptr<int32_t>(),
        indices_buf.data_ptr<int64_t>(),
        counts_buf.data_ptr<int32_t>(),
        W, n_base, n_tokens, buf_width);

    return {indices_buf, counts_buf};
}

/* Convenience wrapper that allocates output (for standalone testing) */
std::tuple<torch::Tensor, torch::Tensor> alive_bits_to_indices_alloc(
    torch::Tensor alive_bits,
    int n_base,
    int n_tokens)
{
    const int Q = alive_bits.size(0);
    int buf_width = n_tokens;  // worst-case: all tokens alive

    auto opts_i64 = torch::TensorOptions().device(alive_bits.device()).dtype(torch::kInt64);
    auto opts_i32 = torch::TensorOptions().device(alive_bits.device()).dtype(torch::kInt32);
    auto indices_buf = torch::empty({Q, static_cast<int64_t>(buf_width)}, opts_i64);
    auto counts_buf = torch::empty({Q}, opts_i32);

    return alive_bits_to_indices(alive_bits, n_base, n_tokens, indices_buf, counts_buf);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("alive_bits_to_indices", &alive_bits_to_indices,
          "Expand alive_bits bitmap to token indices (pre-allocated buffers)");
    m.def("alive_bits_to_indices_alloc", &alive_bits_to_indices_alloc,
          "Expand alive_bits bitmap to token indices (auto-allocate)");
}
