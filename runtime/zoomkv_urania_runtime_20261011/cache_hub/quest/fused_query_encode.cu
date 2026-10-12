// fused_query_encode.cu
//
// Fuses 3 kernel launches into 1 for the per-step query encoding path:
//     fused_normalize_sign(query, signs)  →  Y_normalized
//     hadamard_transform(Y, scale)         →  Y_hadamard       (FWHT, stride 1..64)
//     view(bs, kv_heads, B, m)             →  q_blocks         (free)
//
// Ported from jingze_debug/.../fused_query_encode.cu with two adjustments
// for cache_hub:
//   1. q_norm output is **float32** (not bf16) to match downstream
//      partial_rerank_chunk_dense_sparse_kernel's fp32 q_norm read path
//      (see rerank_topk.cu Step A). This avoids the bf16-conversion
//      elementwise kernel the Python wrapper used to launch.
//   2. extension name follows our cache_hub.quest convention.
//
// Layout:
//   Grid:   (bs, kv_heads)
//   Block:  (head_dim=128,) — one thread per dim element
//   Shmem:  128 floats = 512 B, used for cross-warp norm reduction AND
//           cross-warp FWHT butterfly (strides 32 and 64).
//
// Numerical: the FWHT here uses the same Walsh-Hadamard butterfly order as
// the `fast_hadamard_transform` PyPI package (stride 1, 2, 4, ..., 64), so
// the output q_blocks is bit-identical to the previous
//     hadamard_transform(fused_normalize_sign(q), scale=1/sqrt(128))
// path up to a single float→bf16 rounding at the final store.

#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

#include <cstdint>
#include <cmath>

constexpr int HEAD_DIM = 128;
constexpr int NUM_WARPS = HEAD_DIM / 32;  // 4

__global__
void fused_query_encode_kernel(
    const __nv_bfloat16* __restrict__ query,   // [bs, kv_heads, head_dim]
    const __nv_bfloat16* __restrict__ signs,   // [head_dim]
    __nv_bfloat16* __restrict__ q_blocks,      // [bs, kv_heads, head_dim] (= flat B*m)
    float* __restrict__ q_norm_out,            // [bs, kv_heads]  ← fp32
    float hadamard_scale                        // 1/sqrt(head_dim)
) {
    const int bid = blockIdx.x * gridDim.y + blockIdx.y;  // flat (bs, kv_head)
    const int tid = threadIdx.x;  // 0..127

    extern __shared__ float smem[];  // [128] for both norm reduction and butterfly

    // 1. Load query element
    float val = __bfloat162float(query[bid * HEAD_DIM + tid]);

    // 2. Compute L2 norm (warp-reduce → shmem → cross-warp reduce)
    float sq = val * val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sq += __shfl_xor_sync(0xFFFFFFFF, sq, offset);
    }
    if ((tid & 31) == 0) {
        smem[tid >> 5] = sq;
    }
    __syncthreads();

    float norm_sq;
    if (tid < NUM_WARPS) {
        norm_sq = smem[tid];
    } else {
        norm_sq = 0.0f;
    }
    if (tid < 32) {
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            norm_sq += __shfl_xor_sync(0xFFFFFFFF, norm_sq, offset);
        }
    }
    if (tid == 0) smem[0] = norm_sq;
    __syncthreads();
    norm_sq = smem[0];

    const float norm = sqrtf(norm_sq);
    const float inv_norm = (norm > 1e-8f) ? (1.0f / norm) : 0.0f;

    // 3. Normalize + sign multiply
    val *= inv_norm;
    const float sign_val = __bfloat162float(signs[tid]);
    val *= sign_val;

    // 4. Walsh-Hadamard butterfly: strides 1..16 within warp via shfl_xor.
    #pragma unroll
    for (int s = 1; s <= 16; s <<= 1) {
        const float partner = __shfl_xor_sync(0xFFFFFFFF, val, s);
        if ((tid & s) == 0) {
            val = val + partner;
        } else {
            val = partner - val;
        }
    }

    // 5. Strides 32 and 64: cross-warp via shmem.
    smem[tid] = val;
    __syncthreads();
    {
        const float partner = smem[tid ^ 32];
        if ((tid & 32) == 0) val = val + partner;
        else                  val = partner - val;
    }
    smem[tid] = val;
    __syncthreads();
    {
        const float partner = smem[tid ^ 64];
        if ((tid & 64) == 0) val = val + partner;
        else                  val = partner - val;
    }

    // 6. Scale + write q_blocks (bf16).
    val *= hadamard_scale;
    q_blocks[bid * HEAD_DIM + tid] = __float2bfloat16(val);

    // 7. Write q_norm as fp32 (one thread per block).
    if (tid == 0) {
        q_norm_out[bid] = norm;
    }
}


void fused_query_encode_interface(
    at::Tensor query,        // [bs, kv_heads, head_dim] bf16
    at::Tensor signs,        // [head_dim] bf16
    at::Tensor q_blocks,     // [bs, kv_heads, head_dim] bf16 (output)
    at::Tensor q_norm,       // [bs, kv_heads] fp32 (output)
    double hadamard_scale
) {
    TORCH_CHECK(query.is_cuda() && signs.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(query.scalar_type() == at::ScalarType::BFloat16, "query must be bf16");
    TORCH_CHECK(signs.scalar_type() == at::ScalarType::BFloat16, "signs must be bf16");
    TORCH_CHECK(q_blocks.scalar_type() == at::ScalarType::BFloat16, "q_blocks must be bf16");
    TORCH_CHECK(q_norm.scalar_type() == at::ScalarType::Float,
                "q_norm must be float32 (downstream rerank kernel expects fp32)");
    TORCH_CHECK(query.size(-1) == HEAD_DIM, "head_dim must be 128");
    TORCH_CHECK(signs.size(0) == HEAD_DIM, "signs must have size head_dim");
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(q_blocks.is_contiguous(), "q_blocks must be contiguous");
    TORCH_CHECK(q_norm.is_contiguous(), "q_norm must be contiguous");

    const int bs = query.size(0);
    const int kv_heads = query.size(1);

    const c10::Device dev = query.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    const int smem_size = HEAD_DIM * sizeof(float);  // 512 B

    fused_query_encode_kernel<<<dim3{(unsigned)bs, (unsigned)kv_heads}, HEAD_DIM, smem_size, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(signs.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(q_blocks.data_ptr()),
        q_norm.data_ptr<float>(),
        static_cast<float>(hadamard_scale)
    );

    const cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "fused_query_encode failed: ", cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_query_encode", &fused_query_encode_interface,
          "Fused normalize + sign + FWHT for query encoding (bf16 q, fp32 q_norm)");
}
