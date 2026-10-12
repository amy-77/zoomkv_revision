// fused_normalize_sign.cu
// Fused kernel: norm + normalize + sign_multiply
// Replaces 3 separate PyTorch kernels (reduce_kernel, vectorized_elementwise, elementwise)
// Input:  query [bs * kv_heads, head_dim] bf16
//         signs [head_dim] bf16 (+1/-1)
// Output: out   [bs * kv_heads, head_dim] bf16  = (q / ||q||) * signs
//         norms [bs * kv_heads] float32          = ||q||

#include <torch/extension.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// One block per row (one query vector). head_dim <= 128, so one warp suffices.
// We use 128 threads for head_dim=128 (1 element per thread).

template <int HEAD_DIM>
__global__ void fused_normalize_sign_kernel(
    const __nv_bfloat16* __restrict__ query,   // [N, HEAD_DIM]
    const __nv_bfloat16* __restrict__ signs,   // [HEAD_DIM]
    __nv_bfloat16* __restrict__ out,           // [N, HEAD_DIM]
    float* __restrict__ norms,                 // [N]
    int N
) {
    const int row = blockIdx.x;
    if (row >= N) return;
    const int tid = threadIdx.x;

    const __nv_bfloat16* q_row = query + row * HEAD_DIM;

    // Each thread loads one element (HEAD_DIM threads per block)
    float val = 0.0f;
    if (tid < HEAD_DIM) {
        val = __bfloat162float(q_row[tid]);
    }

    // Compute sum of squares via warp reduction + shared memory
    __shared__ float s_partial[4]; // up to 4 warps
    float sq = val * val;

    // Warp-level reduction
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_down_sync(0xffffffff, sq, offset);
    }

    const int warp_id = tid / 32;
    const int lane_id = tid % 32;
    if (lane_id == 0) {
        s_partial[warp_id] = sq;
    }
    __syncthreads();

    // Final reduction across warps (thread 0)
    float norm_sq;
    if (tid == 0) {
        norm_sq = s_partial[0];
        for (int i = 1; i < (HEAD_DIM + 31) / 32; i++) {
            norm_sq += s_partial[i];
        }
        float norm = sqrtf(norm_sq);
        if (norm < 1e-8f) norm = 1e-8f;
        // Store norm and 1/norm
        norms[row] = norm;
        s_partial[0] = 1.0f / norm;
    }
    __syncthreads();

    float inv_norm = s_partial[0];

    // Normalize and multiply by sign
    if (tid < HEAD_DIM) {
        float normalized = val * inv_norm;
        float sign_val = __bfloat162float(signs[tid]);
        float result = normalized * sign_val;
        out[row * HEAD_DIM + tid] = __float2bfloat16(result);
    }
}

std::vector<torch::Tensor> fused_normalize_sign(
    torch::Tensor query,   // [bs, kv_heads, head_dim] bf16
    torch::Tensor signs    // [head_dim] bf16
) {
    TORCH_CHECK(query.is_cuda() && signs.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == torch::kBFloat16, "query must be bf16");
    TORCH_CHECK(signs.scalar_type() == torch::kBFloat16, "signs must be bf16");

    const int bs = query.size(0);
    const int kv_heads = query.size(1);
    const int head_dim = query.size(2);
    const int N = bs * kv_heads;

    TORCH_CHECK(head_dim == 128, "Only head_dim=128 supported");
    TORCH_CHECK(signs.size(0) == head_dim, "signs must have size head_dim");

    auto query_flat = query.contiguous().view({N, head_dim});
    auto out = torch::empty_like(query_flat);
    auto norms = torch::empty({N}, query.options().dtype(torch::kFloat32));

    const int threads = 128; // head_dim=128, one thread per element
    const int blocks = N;

    fused_normalize_sign_kernel<128><<<blocks, threads>>>(
        reinterpret_cast<const __nv_bfloat16*>(query_flat.data_ptr<at::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(signs.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>()),
        norms.data_ptr<float>(),
        N
    );

    // Reshape outputs
    out = out.view({bs, kv_heads, head_dim});
    norms = norms.view({bs, kv_heads, 1});

    return {out, norms};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_normalize_sign", &fused_normalize_sign,
          "Fused norm + normalize + sign multiply (bf16, head_dim=128)");
}
