/**
 * Fused per-chunk top-k with index gather.
 *
 * For each chunk of chunk_size elements (≤ 32), selects the top-k highest
 * scoring entries and returns both their scores and original indices.
 * Replaces torch.topk + 2x torch.gather with a single warp-shuffle kernel.
 *
 * Grid:  (nk, bs * kv_heads)   — one warp per chunk
 * Block: (32)                   — one warp
 *
 * Supports chunk_size ∈ {8, 16, 32}, k ∈ {1..chunk_size}.
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

__global__ void chunk_topk_gather_kernel(
    const float* __restrict__ scores,       // [B*H, nk, chunk_size]
    const int64_t* __restrict__ indices,    // [B*H, nk, chunk_size]
    float* __restrict__ out_scores,         // [B*H, nk, k]
    int64_t* __restrict__ out_indices,      // [B*H, nk, k]
    int chunk_size,
    int k)
{
    const int ci = blockIdx.x;    // chunk index within nk
    const int bh = blockIdx.y;    // flattened (batch, head)
    const int tx = threadIdx.x;   // lane [0, 31]

    const int in_base = (bh * gridDim.x + ci) * chunk_size;
    float my_score = (tx < chunk_size) ? scores[in_base + tx] : -INFINITY;
    int64_t my_idx = (tx < chunk_size) ? indices[in_base + tx] : -1;

    const int out_base = (bh * gridDim.x + ci) * k;

    for (int r = 0; r < k; r++) {
        // Warp-level argmax via butterfly reduction
        float best_s = my_score;
        int best_lane = tx;

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            float other_s = __shfl_xor_sync(0xffffffff, best_s, off);
            int other_l = __shfl_xor_sync(0xffffffff, best_lane, off);
            if (other_s > best_s || (other_s == best_s && other_l < best_lane)) {
                best_s = other_s;
                best_lane = other_l;
            }
        }

        // Broadcast winner's data from the winning lane
        int64_t winner_idx = __shfl_sync(0xffffffff, my_idx, best_lane);
        float winner_score = best_s;

        if (tx == 0) {
            out_scores[out_base + r] = winner_score;
            out_indices[out_base + r] = winner_idx;
        }

        // Eliminate winner so it's not picked again
        if (tx == best_lane) {
            my_score = -INFINITY;
        }
    }
}

}  // namespace


std::vector<torch::Tensor> chunk_topk_gather_cuda(
    torch::Tensor scores,
    torch::Tensor indices,
    int64_t k)
{
    TORCH_CHECK(scores.dim() == 4, "scores must be 4D [bs, kv, nk, chunk_size]");
    TORCH_CHECK(indices.dim() == 4, "indices must be 4D [bs, kv, nk, chunk_size]");
    TORCH_CHECK(scores.is_cuda() && indices.is_cuda(), "inputs must be on CUDA");
    TORCH_CHECK(scores.is_contiguous() && indices.is_contiguous(), "inputs must be contiguous");

    const int bs = scores.size(0);
    const int kv_heads = scores.size(1);
    const int nk = scores.size(2);
    const int chunk_size = scores.size(3);

    TORCH_CHECK(chunk_size <= 32, "chunk_size must be <= 32");
    TORCH_CHECK(k > 0 && k <= chunk_size, "k must be in [1, chunk_size]");

    // Ensure float32 scores
    torch::Tensor scores_f32;
    if (scores.scalar_type() != torch::kFloat32) {
        scores_f32 = scores.to(torch::kFloat32).contiguous();
    } else {
        scores_f32 = scores;
    }

    auto out_scores = torch::empty({bs, kv_heads, nk, k},
        torch::TensorOptions().dtype(torch::kFloat32).device(scores.device()));
    auto out_indices = torch::empty({bs, kv_heads, nk, k},
        torch::TensorOptions().dtype(torch::kInt64).device(scores.device()));

    const int BH = bs * kv_heads;
    if (BH == 0 || nk == 0) {
        return {out_scores, out_indices};
    }

    dim3 grid(nk, BH);
    dim3 block(32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();

    chunk_topk_gather_kernel<<<grid, block, 0, stream>>>(
        scores_f32.data_ptr<float>(),
        indices.data_ptr<int64_t>(),
        out_scores.data_ptr<float>(),
        out_indices.data_ptr<int64_t>(),
        chunk_size,
        static_cast<int>(k));

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "chunk_topk_gather kernel failed: ", cudaGetErrorString(err));

    return {out_scores, out_indices};
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunk_topk_gather", &chunk_topk_gather_cuda,
          "Fused per-chunk top-k with index gather (CUDA)",
          py::arg("scores"), py::arg("indices"), py::arg("k"));
}
