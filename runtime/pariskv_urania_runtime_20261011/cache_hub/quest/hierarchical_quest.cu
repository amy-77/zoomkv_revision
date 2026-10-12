/**
 * Hierarchical Quest — multi-kernel design.
 *
 * Splits the pipeline into per-stage kernels launched from a C++ host loop:
 *   hq_init_alive          — set alive bit array
 *   hq_group_reduce        — OR-reduce alive bits → group bits  (2D grid)
 *   hq_stage_select        — radix-select top-k per query       (1D grid)
 *   hq_update_alive        — AND selected groups back into alive (2D grid)
 *
 * Improvements over v1 fused kernel:
 *   1. 2D grids for reduce/update → better SM utilisation at low n_queries
 *   2. Compact alive_bits output (uint32 bit array), no dense bool mask
 *   3. factor=1 fast paths — skip reduce, update is bitwise AND
 *   4. Metadata as kernel args, workspace pre-allocatable from Python
 */

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace {

constexpr int RADIX = 256;
constexpr int BASE_CHUNK = 8;
constexpr int MAX_STAGES = 4;

// Block sizes: small for reduce/update (more blocks), large for radix select
constexpr int BLK_RU  = 128;   // reduce / update
constexpr int BLK_SEL = 512;   // radix select

__device__ __forceinline__ uint32_t f2u(float x) {
    uint32_t b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ uint8_t byte_of(uint32_t x, int off) {
    return static_cast<uint8_t>((x >> off) & 0xFF);
}

// ─── K1: init alive bits ────────────────────────────────────────────────────
// Grid: (ceil(alive_words / BLK_RU), n_queries)

__global__ void hq_init_alive(
    uint32_t* __restrict__ alive,    // [n_queries, alive_words]
    int alive_words,
    int n_base)
{
    const int qi = blockIdx.y;
    const int wi = blockIdx.x * BLK_RU + threadIdx.x;
    if (wi >= alive_words) return;

    uint32_t* row = alive + qi * alive_words;
    if (wi == alive_words - 1 && (n_base % 32) != 0)
        row[wi] = (1u << (n_base % 32)) - 1u;
    else
        row[wi] = 0xFFFFFFFFu;
}

// ─── K2: group reduce ───────────────────────────────────────────────────────
// alive_bits → group_bits.  Skipped when factor==1 (host aliases instead).
// Grid: (ceil(alive_words / BLK_RU), n_queries)

__global__ void hq_group_reduce(
    const uint32_t* __restrict__ alive,   // [n_queries, alive_words]
    uint32_t* __restrict__ group_out,     // [n_queries, group_stride]
    int alive_words,
    int group_words,
    int group_stride,
    int n_base,
    int factor,
    int n_groups)
{
    const int qi = blockIdx.y;
    const int wi = blockIdx.x * BLK_RU + threadIdx.x;
    if (wi >= alive_words) return;

    const uint32_t* a_row = alive + qi * alive_words;
    uint32_t* g_row = group_out + qi * group_stride;

    uint32_t w = a_row[wi];

    if (factor <= 32) {
        int groups_per_word = 32 / factor;
        int base_group = wi * groups_per_word;

        for (int g = 0; g < groups_per_word; g++) {
            int gid = base_group + g;
            if (gid >= n_groups) break;
            uint32_t mask = ((1u << factor) - 1u) << (g * factor);
            if (w & mask)
                atomicOr(&g_row[gid / 32], 1u << (gid % 32));
        }
    } else {
        // factor > 32: multiple alive words per group
        int words_per_group = factor / 32;
        int gid = wi / words_per_group;
        if (gid < n_groups && w != 0)
            atomicOr(&g_row[gid / 32], 1u << (gid % 32));
    }
}

// ─── K3: stage select (radix top-k per query) ──────────────────────────────
// Grid: (n_queries,)   Block: BLK_SEL
//
// Reads group_alive bits + scores → writes selected bits into sel_out.
// When n_select >= n_alive, copies group_alive directly.

__global__ void hq_stage_select(
    const float* __restrict__ scores,        // [n_queries, score_stride]
    int score_stride,
    const uint32_t* __restrict__ group_alive, // [n_queries, ga_stride]
    uint32_t* __restrict__ sel_out,           // [n_queries, sel_stride]
    int group_words,
    int ga_stride,
    int sel_stride,
    int n_available,
    float ratio,
    int32_t* __restrict__ workspace,          // [n_queries, ws_stride]
    int ws_stride)
{
    const int qi = blockIdx.x;
    const int tid = threadIdx.x;

    const uint32_t* ga = group_alive + qi * ga_stride;
    uint32_t* so = sel_out + qi * sel_stride;
    const float* sc = scores + static_cast<int64_t>(qi) * score_stride;
    int32_t* ws = workspace + static_cast<int64_t>(qi) * ws_stride;

    // ── shared memory ──
    __shared__ int s_hist[2 * RADIX];
    __shared__ int s_scalars[8];
    // scalars: [counter, threshold, numinput0, numinput1, lastremain, nalive, ...]
    constexpr int SC = 0, ST = 1, SN0 = 2, SN1 = 3, SLR = 4;

    // ── count alive ──
    __shared__ int s_pc[BLK_SEL];
    int local_pc = 0;
    for (int i = tid; i < group_words; i += BLK_SEL)
        local_pc += __popc(ga[i]);
    s_pc[tid] = local_pc;
    __syncthreads();
    for (int s = BLK_SEL / 2; s > 0; s >>= 1) {
        if (tid < s) s_pc[tid] += s_pc[tid + s];
        __syncthreads();
    }
    int n_alive = s_pc[0];

    if (n_alive == 0) {
        // All dead → clear sel_out
        for (int i = tid; i < group_words; i += BLK_SEL) so[i] = 0;
        return;
    }

    int n_select = static_cast<int>(ceilf(static_cast<float>(n_alive) * ratio));
    if (n_select < 1) n_select = 1;
    if (n_select > n_alive) n_select = n_alive;

    // If keeping all alive → just copy
    if (n_select >= n_alive) {
        for (int i = tid; i < group_words; i += BLK_SEL) so[i] = ga[i];
        return;
    }

    // ── radix select ──
    // Clear sel_out
    for (int i = tid; i < group_words; i += BLK_SEL) so[i] = 0;

    int topk_remain = n_select;

    // == Round 0: byte 24 ==
    if (tid < RADIX) { s_hist[tid] = 0; s_hist[tid + RADIX] = 0; }
    if (tid == 0) { s_scalars[SC] = 0; s_scalars[SN0] = 0; }
    __syncthreads();

    for (int idx = tid; idx < n_available; idx += BLK_SEL) {
        if (!((ga[idx / 32] >> (idx % 32)) & 1u)) continue;
        atomicAdd(&s_hist[byte_of(f2u(sc[idx]), 24)], 1);
    }
    __syncthreads();

    // Suffix sum
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        if (tid < RADIX) {
            int j = 1 << i, src = (i & 1) * RADIX, dst = (src ^ RADIX);
            int v = s_hist[src + tid];
            if (tid < RADIX - j) v += s_hist[src + tid + j];
            s_hist[dst + tid] = v;
        }
        __syncthreads();
    }

    // Find threshold
    if (tid < RADIX) {
        int cur = s_hist[tid], nxt = (tid < RADIX - 1) ? s_hist[tid + 1] : 0;
        if (cur > topk_remain && nxt <= topk_remain) s_scalars[ST] = tid;
    }
    __syncthreads();

    int thr = s_scalars[ST];
    int above = (thr < RADIX - 1) ? s_hist[thr + 1] : 0;
    topk_remain -= above;

    if (topk_remain == 0) {
        for (int idx = tid; idx < n_available; idx += BLK_SEL) {
            if (!((ga[idx / 32] >> (idx % 32)) & 1u)) continue;
            if (byte_of(f2u(sc[idx]), 24) > thr)
                atomicOr(&so[idx / 32], 1u << (idx % 32));
        }
        return;
    }

    // Reset histogram
    if (tid < RADIX) { s_hist[tid] = 0; s_hist[tid + RADIX] = 0; }
    __syncthreads();

    int32_t* g_buf = ws;  // double buffer [2 * n_available]

    for (int idx = tid; idx < n_available; idx += BLK_SEL) {
        if (!((ga[idx / 32] >> (idx % 32)) & 1u)) continue;
        uint32_t ord = f2u(sc[idx]);
        uint8_t b = byte_of(ord, 24);
        if (b > thr) {
            atomicOr(&so[idx / 32], 1u << (idx % 32));
        } else if (b == thr) {
            int pos = atomicAdd(&s_scalars[SN0], 1);
            g_buf[pos] = idx;
            atomicAdd(&s_hist[byte_of(ord, 16)], 1);
        }
    }
    __syncthreads();

    // == Rounds 1-3 ==
    #pragma unroll
    for (int round = 1; round < 4; ++round) {
        int r = (round - 1) % 2;
        int num_in = (r == 0) ? s_scalars[SN0] : s_scalars[SN1];
        int boff = 24 - round * 8;
        int32_t* cur = g_buf + r * n_available;
        int32_t* nxt = g_buf + (r ^ 1) * n_available;

        // Suffix sum
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (tid < RADIX) {
                int j = 1 << i, src = (i & 1) * RADIX, dst = (src ^ RADIX);
                int v = s_hist[src + tid];
                if (tid < RADIX - j) v += s_hist[src + tid + j];
                s_hist[dst + tid] = v;
            }
            __syncthreads();
        }

        if (tid < RADIX) {
            int c2 = s_hist[tid], n2 = (tid < RADIX - 1) ? s_hist[tid + 1] : 0;
            if (c2 > topk_remain && n2 <= topk_remain) s_scalars[ST] = tid;
        }
        if (tid == 0) { if (r == 0) s_scalars[SN1] = 0; else s_scalars[SN0] = 0; }
        __syncthreads();

        thr = s_scalars[ST];
        above = (thr < RADIX - 1) ? s_hist[thr + 1] : 0;
        topk_remain -= above;

        if (topk_remain == 0) {
            for (int i = tid; i < num_in; i += BLK_SEL) {
                int idx = cur[i];
                if (byte_of(f2u(sc[idx]), boff) > thr)
                    atomicOr(&so[idx / 32], 1u << (idx % 32));
            }
            return;
        }

        if (tid < RADIX) { s_hist[tid] = 0; s_hist[tid + RADIX] = 0; }
        if (round == 3 && tid == 0) s_scalars[SLR] = topk_remain;
        __syncthreads();

        for (int i = tid; i < num_in; i += BLK_SEL) {
            int idx = cur[i];
            uint32_t ord = f2u(sc[idx]);
            uint8_t b = byte_of(ord, boff);
            if (b > thr) {
                atomicOr(&so[idx / 32], 1u << (idx % 32));
            } else if (b == thr) {
                if (round == 3) {
                    if (atomicAdd(&s_scalars[SLR], -1) > 0)
                        atomicOr(&so[idx / 32], 1u << (idx % 32));
                } else {
                    int slot;
                    if (r == 0) slot = atomicAdd(&s_scalars[SN1], 1);
                    else        slot = atomicAdd(&s_scalars[SN0], 1);
                    nxt[slot] = idx;
                    atomicAdd(&s_hist[byte_of(ord, boff - 8)], 1);
                }
            }
        }
        __syncthreads();
    }
}

// ─── K4: update alive ───────────────────────────────────────────────────────
// Grid: (ceil(alive_words / BLK_RU), n_queries)
//
// factor==1 fast path: alive &= selected  (no expand needed)
// factor>1 : expand selected groups back to base-chunk granularity, AND

__global__ void hq_update_alive_f1(
    uint32_t* __restrict__ alive,
    const uint32_t* __restrict__ selected,
    int alive_words,
    int sel_stride)
{
    const int qi = blockIdx.y;
    const int wi = blockIdx.x * BLK_RU + threadIdx.x;
    if (wi >= alive_words) return;
    alive[qi * alive_words + wi] &= selected[qi * sel_stride + wi];
}

__global__ void hq_update_alive(
    uint32_t* __restrict__ alive,
    const uint32_t* __restrict__ selected,
    int alive_words,
    int group_words,
    int group_stride,
    int n_base,
    int factor,
    int n_groups)
{
    const int qi = blockIdx.y;
    const int wi = blockIdx.x * BLK_RU + threadIdx.x;
    if (wi >= alive_words) return;

    uint32_t* a_row = alive + qi * alive_words;
    const uint32_t* s_row = selected + qi * group_stride;

    int covered_words = (n_groups * factor) / 32;
    if (wi >= covered_words) return;  // beyond covered → keep alive

    uint32_t old = a_row[wi];
    if (old == 0) return;

    if (factor <= 32) {
        int groups_per_word = 32 / factor;
        int base_group = wi * groups_per_word;
        uint32_t expand = 0;
        for (int g = 0; g < groups_per_word; g++) {
            int gid = base_group + g;
            if (gid >= n_groups) break;
            if ((s_row[gid / 32] >> (gid % 32)) & 1u)
                expand |= ((1u << factor) - 1u) << (g * factor);
        }
        a_row[wi] = old & expand;
    } else {
        int gid = wi / (factor / 32);
        if (!((s_row[gid / 32] >> (gid % 32)) & 1u))
            a_row[wi] = 0;
    }
}

// ─── K5: clear group bits ───────────────────────────────────────────────────
// Needed before group_reduce (which uses atomicOr).
// Grid: (ceil(group_words / BLK_RU), n_queries)

__global__ void hq_clear(
    uint32_t* __restrict__ buf,
    int words_per_query)
{
    const int qi = blockIdx.y;
    const int wi = blockIdx.x * BLK_RU + threadIdx.x;
    if (wi >= words_per_query) return;
    buf[qi * words_per_query + wi] = 0;
}

}  // namespace


// ─── Host orchestrator ──────────────────────────────────────────────────────

static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

torch::Tensor hierarchical_quest_cuda(
    std::vector<torch::Tensor> level_scores,   // [n_stages] × [n_queries, n_chunks_s]
    std::vector<int64_t> chunk_sizes,
    std::vector<double> ratios,
    int64_t n_base,
    int64_t n_tokens,
    c10::optional<torch::Tensor> alive_buf,
    c10::optional<torch::Tensor> group_buf,
    c10::optional<torch::Tensor> ws_buf)
{
    const int S = static_cast<int>(level_scores.size());
    TORCH_CHECK(S > 0 && S <= MAX_STAGES, "n_stages must be 1-4");

    auto device = level_scores[0].device();
    const int Q = level_scores[0].size(0);
    const int nb = static_cast<int>(n_base);
    const int aw = cdiv(nb, 32);

    // Contiguous float32 scores
    std::vector<torch::Tensor> sc(S);
    int max_groups = 0;
    for (int s = 0; s < S; s++) {
        sc[s] = level_scores[s].contiguous().to(torch::kFloat32);
        int ng = nb / (static_cast<int>(chunk_sizes[s]) / BASE_CHUNK);
        if (ng > max_groups) max_groups = ng;
    }
    int max_gw = cdiv(max_groups, 32);

    // Allocate / reuse buffers
    auto opts_u = torch::TensorOptions().dtype(torch::kInt32).device(device);

    torch::Tensor alive = alive_buf.has_value() ? alive_buf.value()
        : torch::empty({Q, aw}, opts_u);
    torch::Tensor group = group_buf.has_value() ? group_buf.value()
        : torch::empty({Q, max_gw}, opts_u);
    torch::Tensor sel = torch::empty({Q, max_gw}, opts_u);

    int ws_per_q = 2 * max_groups;
    torch::Tensor workspace = ws_buf.has_value() ? ws_buf.value()
        : torch::empty({Q, ws_per_q}, opts_u);

    c10::cuda::CUDAGuard guard(device);
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // ── Init alive ──
    {
        dim3 grid(cdiv(aw, BLK_RU), Q);
        hq_init_alive<<<grid, BLK_RU, 0, stream>>>(
            reinterpret_cast<uint32_t*>(alive.data_ptr<int32_t>()), aw, nb);
    }

    // ── Per-stage loop ──
    for (int s = 0; s < S; s++) {
        int factor = static_cast<int>(chunk_sizes[s]) / BASE_CHUNK;
        int n_groups = nb / factor;
        int n_available = std::min(n_groups, static_cast<int>(sc[s].size(1)));
        int gw = cdiv(n_available, 32);
        float ratio = static_cast<float>(ratios[s]);

        uint32_t* alive_ptr = reinterpret_cast<uint32_t*>(alive.data_ptr<int32_t>());
        uint32_t* group_ptr = reinterpret_cast<uint32_t*>(group.data_ptr<int32_t>());
        uint32_t* sel_ptr_base = reinterpret_cast<uint32_t*>(sel.data_ptr<int32_t>());

        bool is_f1 = (factor == 1);

        // For factor==1, group_bits = alive_bits (alias), select writes to sel buf
        const uint32_t* ga_ptr;
        uint32_t* sel_ptr;
        int sel_gw;

        int ga_stride;   // row stride for group_alive pointer
        int sel_stride;  // row stride for sel_out pointer

        if (is_f1) {
            ga_ptr = alive_ptr;       // group_alive = alive_bits directly
            sel_ptr = sel_ptr_base;   // select output goes to sel buf
            sel_gw = aw;              // same word count as alive
            ga_stride = aw;           // alive buffer stride
            sel_stride = max_gw;      // sel buffer stride
        } else {
            // Clear group buf, then reduce
            {
                dim3 grid(cdiv(gw, BLK_RU), Q);
                hq_clear<<<grid, BLK_RU, 0, stream>>>(group_ptr, max_gw);
            }
            {
                dim3 grid(cdiv(aw, BLK_RU), Q);
                hq_group_reduce<<<grid, BLK_RU, 0, stream>>>(
                    alive_ptr, group_ptr, aw, gw, max_gw, nb, factor, n_available);
            }
            ga_ptr = group_ptr;         // read group_alive from group buf
            sel_ptr = sel_ptr_base;     // write selected to SEPARATE sel buf
            sel_gw = gw;
            ga_stride = max_gw;         // group buffer stride
            sel_stride = max_gw;        // sel buffer stride
        }

        // Stage select (radix top-k per query)
        hq_stage_select<<<Q, BLK_SEL, 0, stream>>>(
            sc[s].data_ptr<float>(),
            static_cast<int>(sc[s].stride(0)),
            ga_ptr,
            sel_ptr,
            sel_gw,
            ga_stride,
            sel_stride,
            n_available,
            ratio,
            workspace.data_ptr<int32_t>(),
            ws_per_q);

        // Update alive
        if (is_f1) {
            dim3 grid(cdiv(aw, BLK_RU), Q);
            hq_update_alive_f1<<<grid, BLK_RU, 0, stream>>>(
                alive_ptr, sel_ptr, aw, sel_stride);
        } else {
            dim3 grid(cdiv(aw, BLK_RU), Q);
            hq_update_alive<<<grid, BLK_RU, 0, stream>>>(
                alive_ptr, sel_ptr, aw, gw, max_gw, nb, factor, n_groups);
        }
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "hierarchical_quest failed: ", cudaGetErrorString(err));

    return alive;  // [Q, aw] uint32 bit array (stored as int32)
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hierarchical_quest", &hierarchical_quest_cuda,
          "Hierarchical Quest filtering — multi-kernel (compact alive_bits output)",
          py::arg("level_scores"),
          py::arg("chunk_sizes"),
          py::arg("ratios"),
          py::arg("n_base"),
          py::arg("n_tokens"),
          py::arg("alive_buf") = c10::nullopt,
          py::arg("group_buf") = c10::nullopt,
          py::arg("ws_buf") = c10::nullopt);
}
