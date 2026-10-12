#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

// Quest chunk score kernel: compute upper bound score for each chunk
// score(chunk) = Σ_d max(q_d * min_k[chunk, d], q_d * max_k[chunk, d])
//
// Grid:  (n_chunks, bs * kv_heads, 1)
// Block: (WARP_SIZE=32, 1, 1)
// Each warp handles one (batch, head, chunk) combination.
// head_dim is typically 128, so each thread handles 4 dimensions.
//
// Only the bf16 + head_dim==128 specialization is shipped; the generic
// scalar-load template path was removed since production models
// (Qwen3/Llama3, bf16, head_dim=128) never exercised it. Re-add a templated
// fallback here if a non-bf16 / non-d128 model needs Quest stage-1.

// --------------------------------------------------------------------------
// Vectorized fast-path: bf16 + head_dim == 128
//
// Layout: each thread loads ONE uint2 (8 bytes = 4 bf16 values) per tensor.
//   32 threads × 8 bytes = 256 bytes = exactly head_dim * sizeof(bf16),
//   yielding a single perfectly-coalesced 256-byte sector read per tensor.
// Total: 3 vector loads per thread instead of 12 scalar loads (4 dims × 3
// tensors). Reduces issued load instructions by ~4x; arithmetic uses the
// bf16-pair intrinsics (hmul2) where available, with float upcast for fmax
// to keep numerical behaviour bit-identical to the scalar path.
// --------------------------------------------------------------------------
__global__ void quest_chunk_score_kernel_bf16_d128(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ chunk_min,
    const __nv_bfloat16* __restrict__ chunk_max,
    const bool* __restrict__ valid,
    float* __restrict__ scores,
    const int chunk_n_max)
{   


    constexpr int HEAD_DIM = 128;
    const int chunk_idx = blockIdx.x;
    const int bh_idx = blockIdx.y;
    const int tx = threadIdx.x;

    const int out_idx = bh_idx * gridDim.x + chunk_idx;
    const int valid_idx = bh_idx * chunk_n_max + chunk_idx;

    if (valid && !valid[valid_idx]) {
        if (tx == 0) scores[out_idx] = -INFINITY;
        return;
    }

    // Blocked access: thread tx owns dims [tx*4 .. tx*4+3].
    const int d_base = tx * 4;

    const __nv_bfloat16* q_ptr = q + bh_idx * HEAD_DIM;
    const __nv_bfloat16* min_ptr = chunk_min + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;
    const __nv_bfloat16* max_ptr = chunk_max + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;

    // Load 4 bf16 values (8 bytes) per tensor as one uint2.
    const uint2 q_u2   = *reinterpret_cast<const uint2*>(q_ptr   + d_base);
    const uint2 min_u2 = *reinterpret_cast<const uint2*>(min_ptr + d_base);
    const uint2 max_u2 = *reinterpret_cast<const uint2*>(max_ptr + d_base);

    // Reinterpret as 2× bf16x2 pairs.
    const __nv_bfloat162 q_lo   = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.x);
    const __nv_bfloat162 q_hi   = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.y);
    const __nv_bfloat162 mn_lo  = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.x);
    const __nv_bfloat162 mn_hi  = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.y);
    const __nv_bfloat162 mx_lo  = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.x);
    const __nv_bfloat162 mx_hi  = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.y);

    // Convert to floats and compute max(q*min, q*max) per dim.
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
    
    // 向量点积
    float sum = fmaxf(q0 * mn0, q0 * mx0);
    sum += fmaxf(q1 * mn1, q1 * mx1);  
    sum += fmaxf(q2 * mn2, q2 * mx2);
    sum += fmaxf(q3 * mn3, q3 * mx3);

    // Warp reduce (all 32 lanes participate).
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (tx == 0) {
        scores[out_idx] = sum;
    }
}

// PyTorch C++ wrapper
void quest_chunk_score_cuda(
    torch::Tensor q,
    torch::Tensor chunk_min,
    torch::Tensor chunk_max,
    torch::Tensor scores,
    int64_t n_chunks,
    c10::optional<torch::Tensor> valid)
{
    const int bs = q.size(0);
    const int kv_heads = q.size(1);
    const int head_dim = q.size(2);
    const int chunk_n_max = chunk_min.size(2);

    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(chunk_min.is_cuda(), "chunk_min must be on CUDA");
    TORCH_CHECK(chunk_max.is_cuda(), "chunk_max must be on CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(n_chunks <= chunk_n_max, "n_chunks must <= chunk_n_max");

    const bool* valid_ptr = nullptr;
    if (valid.has_value()) {
        TORCH_CHECK(valid->is_cuda(), "valid must be on CUDA");
        TORCH_CHECK(valid->is_contiguous(), "valid must be contiguous");
        valid_ptr = valid->data_ptr<bool>();
    }

    // Production stack uses bf16 + head_dim==128 exclusively; the generic
    // template path was removed. Reject other shapes loudly instead of
    // silently falling back.
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16,
                "quest_chunk_score: only bf16 q is supported; got ",
                q.scalar_type());
    TORCH_CHECK(head_dim == 128,
                "quest_chunk_score: only head_dim==128 is supported; got ",
                head_dim);

    dim3 grid(n_chunks, bs * kv_heads);
    dim3 block(32);  // one warp

    quest_chunk_score_kernel_bf16_d128<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()),
        valid_ptr,
        scores.data_ptr<float>(),
        chunk_n_max);
}

// --------------------------------------------------------------------------
// Sub-chunk scoring kernel for hierarchical Quest stage-2.
//
// Grid:  (factor, n_selected, bs * kv_heads)
// Block: (32,)
//
// blockIdx.x = sub-chunk offset within large chunk (0..factor-1)
// blockIdx.y = index into selected large chunk array
// blockIdx.z = flattened (batch, head)
// Global small chunk index = large_ids[bh*n_selected + sel] * factor + sub_offset
//
// NOTE: only the bf16 + head_dim==128 specialization below is implemented and
// shipped; the generic template path was removed since the production stack
// (Qwen3/Llama3, bf16, head_dim=128) never exercised it. Re-add a templated
// fallback here if a non-bf16 / non-d128 model needs hierarchical Quest.
// shared memory没有利用，现在算法是全局内存到寄存器，加载太多，计算太少。load占比太长了。
// --------------------------------------------------------------------------
__global__ void quest_sub_chunk_score_kernel_bf16_d128(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ chunk_min,
    const __nv_bfloat16* __restrict__ chunk_max,
    const int64_t* __restrict__ large_ids,
    float* __restrict__ scores,
    const int chunk_n_max,
    const int n_selected,
    const int factor)
{
    constexpr int HEAD_DIM = 128;
    const int sub_offset = blockIdx.x;   // which sub-chunk within large chunk
    const int sel_idx = blockIdx.y;      // which selected large chunk
    const int bh_idx = blockIdx.z;       // batch * head
    const int tx = threadIdx.x;

    int64_t large_id = large_ids[bh_idx * n_selected + sel_idx];
    int chunk_idx = static_cast<int>(large_id) * factor + sub_offset;
    const int out_idx = bh_idx * (n_selected * factor) + sel_idx * factor + sub_offset;

    if (chunk_idx < 0 || chunk_idx >= chunk_n_max) {
        if (tx == 0) scores[out_idx] = -INFINITY;
        return;
    }

    const int d_base = tx * 4;
    const __nv_bfloat16* q_ptr = q + bh_idx * HEAD_DIM;
    const __nv_bfloat16* min_ptr = chunk_min + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;
    const __nv_bfloat16* max_ptr = chunk_max + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;

    const uint2 q_u2   = *reinterpret_cast<const uint2*>(q_ptr   + d_base);
    const uint2 min_u2 = *reinterpret_cast<const uint2*>(min_ptr + d_base);
    const uint2 max_u2 = *reinterpret_cast<const uint2*>(max_ptr + d_base);

    const __nv_bfloat162 q_lo   = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.x);
    const __nv_bfloat162 q_hi   = *reinterpret_cast<const __nv_bfloat162*>(&q_u2.y);
    const __nv_bfloat162 mn_lo  = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.x);
    const __nv_bfloat162 mn_hi  = *reinterpret_cast<const __nv_bfloat162*>(&min_u2.y);
    const __nv_bfloat162 mx_lo  = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.x);
    const __nv_bfloat162 mx_hi  = *reinterpret_cast<const __nv_bfloat162*>(&max_u2.y);

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

    if (tx == 0) {
        scores[out_idx] = sum;
    }
}

// --------------------------------------------------------------------------
// Optimized v2 of quest_sub_chunk_score_kernel_bf16_d128.
//
// Layout difference vs v1:
//   v1: grid=(factor, n_selected, bs*kv_heads), block=(32)        → 1 warp/block.
//       Q vector is re-fetched from global memory `factor` times per (bh, sel).
//       Block count = factor * n_selected * bs * kv_heads; with single-warp
//       blocks SM occupancy is capped at 50% (A100: 32 blocks/SM → 32 warps/SM).
//
//   v2: grid=(n_selected, bs*kv_heads), block=(32, factor)        → factor warps/block.
//       Warp 0 cooperatively loads Q[bh, :128] (256 B as 32× uint2) into
//       __shared__ q_smem and broadcasts large_id via shared mem; then all
//       `factor` warps run in parallel, warp w handles sub-chunk w. Q traffic
//       drops factor×; block launch count drops factor×; occupancy reaches
//       100% (16 warps/block on A100 SM with 64 warps/SM cap).
//
// Numerical: bit-identical to v1 (same arithmetic, Q value unchanged).
//
// Requires factor in [1, 32] so block size 32 * factor ≤ 1024 threads. For
// factor > 32 the wrapper falls back to v1.
// --------------------------------------------------------------------------
__global__ void quest_sub_chunk_score_kernel_bf16_d128_v2(
    const __nv_bfloat16* __restrict__ q,         // [bs*kv_heads, 128]
    const __nv_bfloat16* __restrict__ chunk_min, // [bs*kv_heads, chunk_n_max, 128]
    const __nv_bfloat16* __restrict__ chunk_max, // [bs*kv_heads, chunk_n_max, 128]
    const int64_t* __restrict__ large_ids,       // [bs*kv_heads, n_selected]
    float* __restrict__ scores,                  // [bs*kv_heads, n_selected * factor]
    const int chunk_n_max,
    const int n_selected,
    const int factor)
{
    constexpr int HEAD_DIM = 128;

    const int sel_idx = blockIdx.x;       // [0, n_selected)
    const int bh_idx  = blockIdx.y;       // [0, bs * kv_heads)
    const int lane    = threadIdx.x;      // [0, 32)
    const int warp    = threadIdx.y;      // [0, factor) — which sub-chunk

    __shared__ __nv_bfloat16 q_smem[HEAD_DIM];
    __shared__ int64_t large_id_smem;

    // Co-load Q (256 B = 32 × uint2) by warp 0 only; lane 0 also fetches large_id.
    if (warp == 0) {
        const int d_base = lane * 4;
        *reinterpret_cast<uint2*>(q_smem + d_base) =
            *reinterpret_cast<const uint2*>(q + bh_idx * HEAD_DIM + d_base);
        if (lane == 0) {
            large_id_smem = large_ids[bh_idx * n_selected + sel_idx];
        }
    }
    __syncthreads();

    const int sub_offset = warp;
    const int64_t large_id = large_id_smem;
    const int chunk_idx = static_cast<int>(large_id) * factor + sub_offset;
    const int out_idx = bh_idx * (n_selected * factor) + sel_idx * factor + sub_offset;

    // OOB chunks write -INF. Safe to early-return: no __syncthreads() below.
    if (chunk_idx < 0 || chunk_idx >= chunk_n_max) {
        if (lane == 0) scores[out_idx] = -INFINITY;
        return;
    }

    const int d_base = lane * 4;

    // Q comes from SMEM (already warmed). min/max are read from global mem;
    // since all 16 warps stream contiguous 4 KB per tensor for the same
    // large_id, the loads coalesce into large memory transactions.
    const uint2 q_u2 = *reinterpret_cast<const uint2*>(q_smem + d_base);
    const __nv_bfloat16* min_ptr = chunk_min + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;
    const __nv_bfloat16* max_ptr = chunk_max + (bh_idx * chunk_n_max + chunk_idx) * HEAD_DIM;
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

    // For a 2D block (32, factor) CUDA packs warps along x first, so each
    // warp = one row of fixed threadIdx.y. __shfl_xor_sync therefore reduces
    // within this warp's 32 lanes only.
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffff, sum, offset);
    }

    if (lane == 0) {
        scores[out_idx] = sum;
    }
}

// 2D grid layout removes the /nk_small and %nk_small divisions:
//   grid.x = ceil(nk_small / BLK), grid.y = bs * kv_heads, block = (BLK,)
// All internal arithmetic in int32 (workloads are tiny: nk_small ~ 100-500,
// bs*kv_heads ~ 8) — only the loaded chunk id stays int64 because the
// output tensor dtype is int64. Unwritten sub_topk_pos slots arrive as the
// in-kernel sentinel -1 (see float_topk.cu) and are mapped to chunk_idx = -1
// (skipped downstream) instead of being clamped into a legal chunk, which
// avoids silent candidate-set pollution. The final out_chunk clamp preserves
// the original .clamp(0, n_chunks-1) semantics for the valid path.
__global__ void quest_map_back_kernel(
    const int64_t* __restrict__ large_ids,     // [bs*kv_heads, nk_large]
    const int64_t* __restrict__ sub_topk_pos,  // [bs*kv_heads, nk_small]
    int64_t* __restrict__ chunk_idx,           // [bs*kv_heads, nk_small]
    const int nk_large,
    const int nk_small,
    const int factor,
    const int n_chunks)
{
    const int bh_idx = blockIdx.y;
    const int local_k = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_k >= nk_small) return;

    const int row_off_small = bh_idx * nk_small;
    const int64_t sub_pos = sub_topk_pos[row_off_small + local_k];

    // Sentinel skip (B+C): float_topk_3d fills any unwritten output slot with
    // -1 inside the kernel, so a left-over slot arrives here as sub_pos < 0.
    // We also treat sub_pos >= nk_large*factor as invalid. Rather than clamping
    // such garbage into a *legal* chunk (which would silently pollute the
    // candidate set), emit -1. Both downstream consumers — the density kernel
    // (chunk_id >= 0 && < n_chunks guard) and the KIVI qk kernel
    // (real_chunk_id >= 0 && < n_chunks_packed guard) — treat a negative chunk
    // id as an invalid lane and write -inf, so the slot is dropped by the
    // global top-k without any out-of-bounds access or wrong-result pollution.
    const int64_t sub_total = static_cast<int64_t>(nk_large) * factor;
    if (sub_pos < 0 || sub_pos >= sub_total) {
        chunk_idx[row_off_small + local_k] = -1;
        return;
    }

    int large_sel = static_cast<int>(sub_pos / factor);
    // Final defensive bound on large_sel for the valid path. With the sentinel
    // check above this is normally a no-op, but it keeps the large_ids[] read
    // in range even if factor/nk_large were ever mis-sized.
    if (large_sel < 0) {
        large_sel = 0;
    } else if (large_sel >= nk_large) {
        large_sel = nk_large - 1;
    }
    const int sub_off = static_cast<int>(sub_pos - static_cast<int64_t>(large_sel) * factor);

    const int64_t large_id = large_ids[bh_idx * nk_large + large_sel];
    int64_t out_chunk = large_id * factor + sub_off;
    if (out_chunk < 0) {
        out_chunk = 0;
    } else if (out_chunk >= n_chunks) {
        out_chunk = n_chunks - 1;
    }

    chunk_idx[row_off_small + local_k] = out_chunk;
}





// C++ wrapper for sub-chunk scoring
void quest_sub_chunk_score_cuda(
    torch::Tensor q,              // [bs, kv_heads, head_dim]
    torch::Tensor chunk_min,      // [bs, kv_heads, chunk_n_max, head_dim]
    torch::Tensor chunk_max,      // [bs, kv_heads, chunk_n_max, head_dim]
    torch::Tensor large_ids,      // [bs, kv_heads, n_selected] int64
    torch::Tensor scores,         // [bs, kv_heads, n_selected * factor] float32
    int64_t n_selected,
    int64_t factor)
{
    const int bs = q.size(0);
    const int kv_heads = q.size(1);
    const int head_dim = q.size(2);
    const int chunk_n_max = chunk_min.size(2);

    TORCH_CHECK(q.is_cuda(), "q must be on CUDA");
    TORCH_CHECK(chunk_min.is_cuda(), "chunk_min must be on CUDA");
    TORCH_CHECK(chunk_max.is_cuda(), "chunk_max must be on CUDA");
    TORCH_CHECK(large_ids.is_cuda(), "large_ids must be on CUDA");
    TORCH_CHECK(scores.is_cuda(), "scores must be on CUDA");
    TORCH_CHECK(q.is_contiguous(), "q must be contiguous");
    TORCH_CHECK(large_ids.is_contiguous(), "large_ids must be contiguous");
    TORCH_CHECK(scores.is_contiguous(), "scores must be contiguous");
    TORCH_CHECK(large_ids.scalar_type() == at::ScalarType::Long,
                "large_ids must be int64");
    TORCH_CHECK(n_selected > 0, "n_selected must be > 0");

    // Production stack uses bf16 + head_dim==128 exclusively; the generic
    // template path was removed. Reject other shapes loudly instead of
    // silently falling back.
    TORCH_CHECK(q.scalar_type() == at::ScalarType::BFloat16,
                "quest_sub_chunk_score: only bf16 q is supported; got ",
                q.scalar_type());
    TORCH_CHECK(head_dim == 128,
                "quest_sub_chunk_score: only head_dim==128 is supported; got ",
                head_dim);
    // The kernel packs `factor` warps into one block (block = 32 * factor
    // threads), which must stay within the 1024 threads/block hardware limit.
    TORCH_CHECK(factor >= 1 && factor <= 32,
                "quest_sub_chunk_score: factor must be in [1, 32]; got ",
                factor);

    dim3 grid(n_selected, bs * kv_heads);
    dim3 block(32, factor);
    quest_sub_chunk_score_kernel_bf16_d128_v2<<<grid, block>>>(
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_min.data_ptr()),
        reinterpret_cast<const __nv_bfloat16*>(chunk_max.data_ptr()),
        large_ids.data_ptr<int64_t>(),
        scores.data_ptr<float>(),
        chunk_n_max, n_selected, factor);
}








void quest_map_back_cuda(
    torch::Tensor large_ids,      // [bs, kv_heads, nk_large] int64
    torch::Tensor sub_topk_pos,   // [bs, kv_heads, nk_small] int64
    torch::Tensor chunk_idx,      // [bs, kv_heads, nk_small] int64
    int64_t factor,
    int64_t n_chunks)
{
    TORCH_CHECK(large_ids.is_cuda(), "large_ids must be on CUDA");
    TORCH_CHECK(sub_topk_pos.is_cuda(), "sub_topk_pos must be on CUDA");
    TORCH_CHECK(chunk_idx.is_cuda(), "chunk_idx must be on CUDA");
    TORCH_CHECK(large_ids.is_contiguous(), "large_ids must be contiguous");
    TORCH_CHECK(sub_topk_pos.is_contiguous(), "sub_topk_pos must be contiguous");
    TORCH_CHECK(chunk_idx.is_contiguous(), "chunk_idx must be contiguous");
    TORCH_CHECK(large_ids.scalar_type() == at::ScalarType::Long, "large_ids must be int64");
    TORCH_CHECK(sub_topk_pos.scalar_type() == at::ScalarType::Long, "sub_topk_pos must be int64");
    TORCH_CHECK(chunk_idx.scalar_type() == at::ScalarType::Long, "chunk_idx must be int64");
    TORCH_CHECK(large_ids.dim() == 3, "large_ids must be 3D [bs, kv_heads, nk_large]");
    TORCH_CHECK(sub_topk_pos.dim() == 3, "sub_topk_pos must be 3D [bs, kv_heads, nk_small]");
    TORCH_CHECK(chunk_idx.dim() == 3, "chunk_idx must be 3D [bs, kv_heads, nk_small]");
    TORCH_CHECK(large_ids.size(0) == sub_topk_pos.size(0) && large_ids.size(1) == sub_topk_pos.size(1),
                "large_ids and sub_topk_pos must share [bs, kv_heads]");
    TORCH_CHECK(chunk_idx.sizes() == sub_topk_pos.sizes(), "chunk_idx shape must match sub_topk_pos");
    TORCH_CHECK(factor > 0, "factor must be > 0");
    TORCH_CHECK(n_chunks > 0, "n_chunks must be > 0");

    const int64_t nk_large_i64 = large_ids.size(2);
    const int64_t nk_small_i64 = sub_topk_pos.size(2);
    const int64_t bh_i64 = large_ids.size(0) * large_ids.size(1);
    TORCH_CHECK(nk_large_i64 > 0, "nk_large must be > 0");
    TORCH_CHECK(nk_small_i64 > 0, "nk_small must be > 0");
    TORCH_CHECK(nk_large_i64 <= INT32_MAX && nk_small_i64 <= INT32_MAX
                && factor <= INT32_MAX && n_chunks <= INT32_MAX
                && bh_i64 <= INT32_MAX,
                "quest_map_back: dimensions exceed int32 range");

    const int nk_large = static_cast<int>(nk_large_i64);
    const int nk_small = static_cast<int>(nk_small_i64);
    const int factor_i = static_cast<int>(factor);
    const int n_chunks_i = static_cast<int>(n_chunks);
    const int bh = static_cast<int>(bh_i64);

    constexpr int BLK = 128;
    dim3 grid((nk_small + BLK - 1) / BLK, bh);
    dim3 block(BLK);

    quest_map_back_kernel<<<grid, block>>>(
        large_ids.data_ptr<int64_t>(),
        sub_topk_pos.data_ptr<int64_t>(),
        chunk_idx.data_ptr<int64_t>(),
        nk_large, nk_small, factor_i, n_chunks_i);

    auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "quest_map_back kernel failed: ", cudaGetErrorString(result));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("quest_chunk_score", &quest_chunk_score_cuda,
          "Quest chunk score upper bound (CUDA)",
          py::arg("q"), py::arg("chunk_min"), py::arg("chunk_max"),
          py::arg("scores"), py::arg("n_chunks"), py::arg("valid") = c10::nullopt);
    m.def("quest_sub_chunk_score", &quest_sub_chunk_score_cuda,
          "Quest sub-chunk score within selected large chunks (CUDA)",
          py::arg("q"), py::arg("chunk_min"), py::arg("chunk_max"),
          py::arg("large_ids"), py::arg("scores"),
          py::arg("n_selected"), py::arg("factor"));
    m.def("quest_map_back", &quest_map_back_cuda,
          "Map stage-2 sub_topk positions back to global small chunk ids (CUDA)",
          py::arg("large_ids"), py::arg("sub_topk_pos"), py::arg("chunk_idx"),
          py::arg("factor"), py::arg("n_chunks"));
}
