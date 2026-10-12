#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

template <typename T>
__device__ __forceinline__ float to_float(T v);

template <>
__device__ __forceinline__ float to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float to_float<double>(double v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float to_float<at::Half>(at::Half v) {
    return __half2float(reinterpret_cast<__half&>(v));
}

template <>
__device__ __forceinline__ float to_float<at::BFloat16>(at::BFloat16 v) {
    return __bfloat162float(reinterpret_cast<__nv_bfloat16&>(v));
}

template <typename scalar_t>
__global__ void adaptive_k_from_top_scores_kernel(
    const scalar_t* __restrict__ top_scores,  // [rows, K]
    const scalar_t* __restrict__ log_total,   // [rows, 1]
    int32_t* __restrict__ out_k,              // [rows]
    int rows,
    int K,
    float threshold
) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    // Each row is short (K <= a few hundred). Let one thread scan to avoid
    // shared-memory prefix complexity; launch overhead dominates otherwise.
    if (threadIdx.x == 0) {
        const scalar_t* row_scores = top_scores + static_cast<int64_t>(row) * K;
        const float ltot = to_float<scalar_t>(log_total[row]);

        float cumsum = 0.0f;
        int k = K;
        #pragma unroll 4
        for (int j = 0; j < K; ++j) {
            cumsum += __expf(to_float<scalar_t>(row_scores[j]) - ltot);
            if (cumsum >= threshold) {
                k = j + 1;
                break;
            }
        }
        out_k[row] = k;
    }
}

}  // namespace

torch::Tensor adaptive_k_from_top_scores_cuda(
    torch::Tensor top_scores,  // [rows, K]
    torch::Tensor log_total,   // [rows, 1]
    double threshold
) {
    TORCH_CHECK(top_scores.is_cuda(), "top_scores must be CUDA tensor");
    TORCH_CHECK(log_total.is_cuda(), "log_total must be CUDA tensor");
    TORCH_CHECK(top_scores.dim() == 2, "top_scores must be 2D [rows, K]");
    TORCH_CHECK(log_total.dim() == 2, "log_total must be 2D [rows, 1]");
    TORCH_CHECK(log_total.size(0) == top_scores.size(0), "row mismatch");
    TORCH_CHECK(log_total.size(1) == 1, "log_total second dim must be 1");
    TORCH_CHECK(top_scores.scalar_type() == log_total.scalar_type(),
                "top_scores/log_total dtype must match");

    const int rows = static_cast<int>(top_scores.size(0));
    const int K = static_cast<int>(top_scores.size(1));
    TORCH_CHECK(rows > 0, "rows must be > 0");
    TORCH_CHECK(K > 0, "K must be > 0");

    auto top_scores_c = top_scores.contiguous();
    auto log_total_c = log_total.contiguous();
    auto out_k = torch::empty({rows},
        torch::TensorOptions().dtype(torch::kInt32).device(top_scores.device()));

    c10::cuda::CUDAGuard device_guard(top_scores.device());
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    const dim3 grid(rows);
    const dim3 block(32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        top_scores_c.scalar_type(),
        "adaptive_k_from_top_scores_cuda",
        [&] {
            adaptive_k_from_top_scores_kernel<scalar_t><<<grid, block, 0, stream>>>(
                top_scores_c.data_ptr<scalar_t>(),
                log_total_c.data_ptr<scalar_t>(),
                out_k.data_ptr<int32_t>(),
                rows,
                K,
                static_cast<float>(threshold)
            );
        });

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "adaptive_k_from_top_scores kernel failed: ", cudaGetErrorString(err));
    return out_k;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adaptive_k_from_top_scores", &adaptive_k_from_top_scores_cuda,
          "Adaptive-k from top scores (CUDA)",
          py::arg("top_scores"), py::arg("log_total"), py::arg("threshold"));
}

