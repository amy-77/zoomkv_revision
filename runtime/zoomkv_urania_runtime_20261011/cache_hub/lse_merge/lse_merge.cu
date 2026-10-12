#include <ATen/Dispatch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cmath>
#include <cstdint>
#include <torch/extension.h>

namespace {

template <typename scalar_t>
__global__ void lse_merge_kernel(
    const scalar_t* __restrict__ attn_a,
    const float* __restrict__ lse_a,
    const scalar_t* __restrict__ attn_b,
    const float* __restrict__ lse_b,
    scalar_t* __restrict__ out,
    int total,
    int heads,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const int d = idx % dim;
    const int h = (idx / dim) % heads;
    const int b = idx / (heads * dim);
    const int lse_idx = b * heads + h;
    const int out_idx = (b * heads + h) * dim + d;  // [bs, 1, heads, dim]

    const float la = lse_a[lse_idx];
    const float lb = lse_b[lse_idx];
    const float m = fmaxf(la, lb);
    const float wa = expf(la - m);
    const float wb = expf(lb - m);
    const float denom = wa + wb;

    const float va = static_cast<float>(attn_a[out_idx]);
    const float vb = static_cast<float>(attn_b[out_idx]);
    out[out_idx] = static_cast<scalar_t>((wa * va + wb * vb) / denom);
}

}  // namespace

torch::Tensor lse_merge_cuda(
    torch::Tensor attn_a,
    torch::Tensor lse_a,
    torch::Tensor attn_b,
    torch::Tensor lse_b) {
    TORCH_CHECK(attn_a.is_cuda(), "attn_a must be CUDA");
    TORCH_CHECK(attn_b.is_cuda(), "attn_b must be CUDA");
    TORCH_CHECK(lse_a.is_cuda(), "lse_a must be CUDA");
    TORCH_CHECK(lse_b.is_cuda(), "lse_b must be CUDA");
    TORCH_CHECK(attn_a.is_contiguous(), "attn_a must be contiguous");
    TORCH_CHECK(attn_b.is_contiguous(), "attn_b must be contiguous");
    TORCH_CHECK(lse_a.is_contiguous(), "lse_a must be contiguous");
    TORCH_CHECK(lse_b.is_contiguous(), "lse_b must be contiguous");
    TORCH_CHECK(attn_a.sizes() == attn_b.sizes(), "attn_a and attn_b shape mismatch");
    TORCH_CHECK(attn_a.dim() == 4 && attn_a.size(1) == 1,
                "attn_a must have shape [bs, 1, heads, dim]");
    TORCH_CHECK(lse_a.scalar_type() == at::ScalarType::Float, "lse_a must be float32");
    TORCH_CHECK(lse_b.scalar_type() == at::ScalarType::Float, "lse_b must be float32");

    const int bs = static_cast<int>(attn_a.size(0));
    const int heads = static_cast<int>(attn_a.size(2));
    const int dim = static_cast<int>(attn_a.size(3));
    TORCH_CHECK(lse_a.dim() == 3 && lse_a.size(0) == bs && lse_a.size(1) == heads && lse_a.size(2) == 1,
                "lse_a must have shape [bs, heads, 1]");
    TORCH_CHECK(lse_b.dim() == 3 && lse_b.size(0) == bs && lse_b.size(1) == heads && lse_b.size(2) == 1,
                "lse_b must have shape [bs, heads, 1]");

    auto out = torch::empty_like(attn_a);
    const int total = bs * heads * dim;
    if (total == 0) return out;

    const c10::Device dev = attn_a.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    constexpr int threads = 256;
    const int blocks = (total + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        attn_a.scalar_type(),
        "lse_merge_cuda",
        [&] {
            lse_merge_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
                attn_a.data_ptr<scalar_t>(),
                lse_a.data_ptr<float>(),
                attn_b.data_ptr<scalar_t>(),
                lse_b.data_ptr<float>(),
                out.data_ptr<scalar_t>(),
                total,
                heads,
                dim);
        });

    const auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "lse_merge kernel failed: ", cudaGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("lse_merge", &lse_merge_cuda);
}
