// Fused 2-bit coarse score kernel for the 2-stage RaBitQ pipeline.
//
// 替代 polar_cache_pure_rabitq_2stage_quantize_2bit_coarse_kernel.py 里
// `_2stage_2bit_coarse_then_rabitq_topk` 的 Stage-1 PyTorch 实现：
//
//   1. _extract_signs_and_mag_high_from_4bit (key_4bit -> 2 个 bit-plane)
//   2. 6 次 popcount LUT + sum (拼出每 (bs, kv, l) 的 score)
//
// 一次扫 key_4bit、寄存器内位 shuffle、6 个 __popc 累加，O(B) 寄存器循环，没有
// 任何中间 tensor。语义和原 PyTorch 实现一致：
//
//   v_q · v_k = (1 - 2·xs) · (1 + m_q + m_k + m_q·m_k),  xs := q_s ^ k_s
//   score(l) = Σ_d v_q · v_k
//            = (D + N(m_q) + N(m_k) + N(m_q·m_k))
//              - 2·(N(xs) + N(m_q·xs) + N(m_k·xs) + N(m_q·m_k·xs))
//
// 其中 N(·) 是对所有 D=B·m 维 popcount 的总和。query 一侧 (m_q) 在
// 整个 kv_len 维度共享，每个 thread 自己 loop B 次 __popc 即可，几乎 0 开销。
//
// 性能预期 (kv_len=130k, B=16, m=8, bs=1, kv_heads=8)：
//   * key_4bit 读：1 × 8 × 16 × 130k × 4 byte = 67 MB
//   * 输出 score：1 × 8 × 130k × 4 byte = 4 MB
//   * 总 HBM 流量 ~71 MB，A100 BW 1.5 TB/s → 理论 ~50 µs，加 launch ~30 µs
//     → ~80 µs / layer / decode step，对比 PyTorch 实现 ~9 ms 提升 ~100×。

#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <torch/python.h>

namespace {

// 每个 packed byte = (sign_odd<<7) | (mag_odd_lo3<<4) | (sign_even<<3) | mag_even_lo3
// 我们只关心 sign-bit (bit 3 / 7) 和 magnitude-high bit (bit 2 / 6)。
// extract_sign_pair / extract_mag_pair 把一个 byte 内 (even, odd) 两个维度的
// 单 bit 抽出来塞进低 2 位（bit0=even, bit1=odd）。
__device__ __forceinline__ uint32_t extract_sign_pair(uint32_t b) {
    return ((b >> 3) & 1u) | (((b >> 7) & 1u) << 1);
}
__device__ __forceinline__ uint32_t extract_mag_pair(uint32_t b) {
    return ((b >> 2) & 1u) | (((b >> 6) & 1u) << 1);
}

// 一个 block (m=8) 对应 m/2 = 4 个 packed byte，正好 1 个 uint32。
// 把 4 byte 内的 8 个 sign-bit / mag-high-bit 重排成两个 8-bit byte，
// bit_i (0..7) 对应 dim i 的 sign / mag-high。
__device__ __forceinline__ void unpack_sign_mag_8dim(uint32_t key32,
                                                    uint32_t &sign_byte,
                                                    uint32_t &mag_byte) {
    const uint32_t b0 = key32 & 0xFFu;
    const uint32_t b1 = (key32 >> 8) & 0xFFu;
    const uint32_t b2 = (key32 >> 16) & 0xFFu;
    const uint32_t b3 = (key32 >> 24) & 0xFFu;
    sign_byte = extract_sign_pair(b0)
              | (extract_sign_pair(b1) << 2)
              | (extract_sign_pair(b2) << 4)
              | (extract_sign_pair(b3) << 6);
    mag_byte  = extract_mag_pair(b0)
              | (extract_mag_pair(b1) << 2)
              | (extract_mag_pair(b2) << 4)
              | (extract_mag_pair(b3) << 6);
}

struct Coarse2bitParams {
    const void *__restrict__ key_4bit_ptr;     // uint8 [bs, kv, B, kv_len, m/2]
    const void *__restrict__ q_signs_ptr;      // uint8 [bs, kv, B]
    const void *__restrict__ q_mags_ptr;       // uint8 [bs, kv, B]
    void       *__restrict__ score_ptr;        // int32 [bs, kv, kv_len]
    int kv_len;
    int m;
    // 都按 uint32 元素数（即 byte/4）算
    int key4_bs_stride;
    int key4_head_stride;
    int key4_B_stride;
    int key4_len_stride;
};

// Grid: (bs, kv_heads, ceil(kv_len / BLOCK_SIZE))
// Block: (BLOCK_SIZE)，每 thread 输出一个 (bs, kv, l) 的 score。
// Shared mem: 2*B byte (B=16 → 32 byte) 缓存当前 (bs, kv) 的 q_signs / q_mags。
template <int B>
__global__ void coarse_2bit_score_kernel(Coarse2bitParams p) {
    const int bidx = blockIdx.x;     // bs
    const int bidy = blockIdx.y;     // kv_heads
    const int bidz = blockIdx.z;     // kv tile
    const int tidx = threadIdx.x;
    const int kv_len = p.kv_len;
    const int m = p.m;
    const int D = m * B;
    const int head_id = bidx * gridDim.y + bidy;

    __shared__ uint8_t q_signs_smem[B];
    __shared__ uint8_t q_mags_smem[B];

    // load q-side bit-planes (very small, B bytes each)
    if (tidx < B) {
        const uint8_t* qs_base = reinterpret_cast<const uint8_t*>(p.q_signs_ptr) + head_id * B;
        const uint8_t* qm_base = reinterpret_cast<const uint8_t*>(p.q_mags_ptr)  + head_id * B;
        q_signs_smem[tidx] = qs_base[tidx];
        q_mags_smem[tidx]  = qm_base[tidx];
    }
    __syncthreads();

    const int global_id = bidz * blockDim.x + tidx;
    if (global_id >= kv_len) return;

    // |q| 端 popcount 总数（与 kv_len 无关，每 thread 算一次也只是 B 次 __popc）
    int pc_qm_total = 0;
    #pragma unroll
    for (int b = 0; b < B; b++) {
        pc_qm_total += __popc((uint32_t)q_mags_smem[b]);
    }

    const uint32_t* key32_base = reinterpret_cast<const uint32_t*>(p.key_4bit_ptr)
                              + bidx * p.key4_bs_stride
                              + bidy * p.key4_head_stride;

    int pc_xor = 0, pc_qm_xor = 0, pc_km_xor = 0, pc_qm_km_xor = 0;
    int pc_km = 0, pc_qm_km = 0;

    #pragma unroll
    for (int b = 0; b < B; b++) {
        const uint32_t key32 = __ldg(key32_base
                                    + b * p.key4_B_stride
                                    + global_id * p.key4_len_stride);
        uint32_t k_sign, k_mag;
        unpack_sign_mag_8dim(key32, k_sign, k_mag);

        const uint32_t qs = (uint32_t)q_signs_smem[b];
        const uint32_t qm = (uint32_t)q_mags_smem[b];

        const uint32_t xs        = qs ^ k_sign;
        const uint32_t qm_xor    = qm & xs;
        const uint32_t km_xor    = k_mag & xs;
        const uint32_t qm_km     = qm & k_mag;
        const uint32_t qm_km_xor = qm_km & xs;

        pc_xor       += __popc(xs);
        pc_qm_xor    += __popc(qm_xor);
        pc_km_xor    += __popc(km_xor);
        pc_qm_km_xor += __popc(qm_km_xor);
        pc_km        += __popc(k_mag);
        pc_qm_km     += __popc(qm_km);
    }

    const int score = (D + pc_qm_total + pc_km + pc_qm_km)
                    - 2 * (pc_xor + pc_qm_xor + pc_km_xor + pc_qm_km_xor);

    int* score_ptr = reinterpret_cast<int*>(p.score_ptr) + head_id * kv_len;
    score_ptr[global_id] = score;
}

} // anonymous namespace


void coarse_2bit_score_interface(
    at::Tensor key_4bit,
    at::Tensor q_signs,
    at::Tensor q_mags,
    int B, int m,
    at::Tensor score
) {
    TORCH_CHECK(key_4bit.scalar_type() == at::ScalarType::Byte,
                "key_4bit must be uint8");
    TORCH_CHECK(q_signs.scalar_type() == at::ScalarType::Byte,
                "q_signs must be uint8");
    TORCH_CHECK(q_mags.scalar_type() == at::ScalarType::Byte,
                "q_mags must be uint8");
    TORCH_CHECK(score.scalar_type() == at::ScalarType::Int,
                "score must be int32");
    TORCH_CHECK(m == 8, "m must be 8 (otherwise 4-byte uint32 unpack invalid)");
    TORCH_CHECK(key_4bit.size(-1) == m / 2,
                "key_4bit last dim must equal m/2");
    TORCH_CHECK(key_4bit.stride(-1) == 1,
                "key_4bit must be contiguous in last dim");
    TORCH_CHECK(score.is_contiguous(), "score must be contiguous");

    TORCH_CHECK(key_4bit.is_cuda() && q_signs.is_cuda()
                && q_mags.is_cuda() && score.is_cuda(),
                "all tensors must be on CUDA");

    const int bs = key_4bit.size(0);
    const int kv_heads = key_4bit.size(1);
    const int kv_len = key_4bit.size(3);

    // q_signs / q_mags 必须 contiguous，因为 kernel 直接按 (bs*kv_heads, B) 线性
    // 读，没传 stride。Python wrapper 那边已经做了 .contiguous()。
    TORCH_CHECK(q_signs.is_contiguous(), "q_signs must be contiguous");
    TORCH_CHECK(q_mags.is_contiguous(),  "q_mags must be contiguous");

    constexpr int BLOCK_SIZE = 128;
    const int block_num = (kv_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    const c10::Device dev = key_4bit.device();
    c10::cuda::CUDAGuard device_guard(dev);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    Coarse2bitParams params{
        key_4bit.data_ptr(),
        q_signs.data_ptr(),
        q_mags.data_ptr(),
        score.data_ptr(),
        kv_len, m,
        // key_4bit byte-stride → uint32-stride (÷4)，与 rerank.cu 一致
        int(key_4bit.stride(0) / 4),
        int(key_4bit.stride(1) / 4),
        int(key_4bit.stride(2) / 4),
        int(key_4bit.stride(3) / 4),
    };

    dim3 grid((unsigned)bs, (unsigned)kv_heads, (unsigned)block_num);
    dim3 block((unsigned)BLOCK_SIZE);

    if (B == 16) {
        coarse_2bit_score_kernel<16><<<grid, block, 0, stream>>>(params);
    } else {
        TORCH_CHECK(false, "only B=16 currently supported");
    }

    auto err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess,
                "coarse_2bit_score kernel failed: ",
                cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("coarse_2bit_score", &coarse_2bit_score_interface,
          "Fused 2-bit coarse score (extract sign+mag-high planes from 4-bit packed key, "
          "popcount-based per-(bs,kv,l) similarity score in one pass).");
}
