#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstddef>
#include <cstdint>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mutex>
#include <optional>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>
#include <unordered_map>
#include <cuda_runtime.h>
#define CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

// ---------------------------------------------------------------------------
// CPU pinned host pointer  ->  UVA device pointer cache.
//
// Why we need this:
//   This translation unit launches kernels that dereference *host-pinned*
//   memory directly via UVA (so the GPU PCIe-reads from CPU RAM).  When the
//   process is profiled by nsys 2023.2.3, CUPTI's cudaLaunchKernel intercept
//   validates each kernel-argument pointer; if it sees the *host-side* virtual
//   address of a mapped pinned buffer it can SIGSEGV inside cudaLaunchKernel
//   on first launch.
//
//   The portable fix is to pass a bona fide *device-side* alias of the same
//   pinned buffer obtained via cudaHostGetDevicePointer().  On UVA systems
//   the returned pointer maps to the exact same physical host pages, so the
//   kernel's actual PCIe traffic is unchanged — there is no copy and no
//   measurable runtime cost.  We cache the translation per host pointer so
//   the per-call overhead is a single hashmap lookup (sub-microsecond).
//
//   Lifetime: callers (KIVI / RaBitQ pure-rabitq cache modules) allocate
//   their CPU pinned KV buffers once at __init__ and reuse them for the
//   whole session, so caching by host pointer is safe.
// ---------------------------------------------------------------------------
namespace {
inline void* host_pinned_to_uva_dptr(void* host_ptr) {
    static std::mutex mu;
    static std::unordered_map<void*, void*> cache;
    {
        std::lock_guard<std::mutex> lk(mu);
        auto it = cache.find(host_ptr);
        if (it != cache.end()) {
            return it->second;
        }
    }
    void* dev_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer(&dev_ptr, host_ptr, 0);
    TORCH_CHECK(err == cudaSuccess,
                "cudaHostGetDevicePointer failed (is the buffer host-pinned and mapped?): ",
                cudaGetErrorString(err));
    {
        std::lock_guard<std::mutex> lk(mu);
        cache.emplace(host_ptr, dev_ptr);
    }
    return dev_ptr;
}

// Resolve a tensor's data pointer to something nsys/CUPTI is happy to see
// at kernel-launch time.  CUDA tensors pass through unchanged; CPU pinned
// tensors get a UVA device-pointer alias.
inline void* resolve_kernel_arg_ptr(const at::Tensor& t) {
    void* p = t.data_ptr();
    if (t.is_cuda()) {
        return p;
    }
    // Tensor must be a CPU tensor here; require it to be host-pinned so that
    // cudaHostGetDevicePointer is well-defined.
    TORCH_CHECK(t.is_pinned(),
                "h2d_gather_kv: CPU source tensor must be host-pinned (allocated with pin_memory=True)");
    return host_pinned_to_uva_dptr(p);
}
}  // namespace


template<int DIM>
__global__ 
void h2d_gather_kv_kernel(
    void* __restrict__ src_k_caches_ptr, // 默认连续
    void* __restrict__ src_v_caches_ptr, // 默认连续
    void* __restrict__ dst_k_caches_ptr, // 默认连续
    void* __restrict__ dst_v_caches_ptr, // 默认连续
    void* __restrict__ selected_indices_ptr,
    int src_len,
    int dst_len
) {
    const int bs = gridDim.x;
    const int heads = gridDim.y;
    const int topk = gridDim.z;
    const int tidx = threadIdx.x;

    constexpr int uint4_per_dim = DIM * 2 / 16; // 256 / 16 = 16
    if (DIM == 128) {
        // Calculate the index for this thread - use int64_t for indices
        int64_t index = __ldg(reinterpret_cast<int64_t*>(selected_indices_ptr)
                             + blockIdx.x * heads * topk + blockIdx.y * topk + blockIdx.z);
        
        // Process k-cache
        if (tidx < 16) {
            uint4* src_ptr_k = reinterpret_cast<uint4*>(src_k_caches_ptr);
            uint4* dst_ptr_k = reinterpret_cast<uint4*>(dst_k_caches_ptr);
            
            // Source: read from the index position in source tensor
            src_ptr_k += (blockIdx.x * heads * src_len * uint4_per_dim) 
                         + (blockIdx.y * src_len * uint4_per_dim)
                         + (index * uint4_per_dim);
            
            // Destination: write to the blockIdx.z position in dest tensor (topk index)
            dst_ptr_k += (blockIdx.x * heads * dst_len * uint4_per_dim) 
                         + (blockIdx.y * dst_len * uint4_per_dim)
                         + (blockIdx.z * uint4_per_dim);

            dst_ptr_k[tidx] = src_ptr_k[tidx];
        }
        // Process v-cache  
        else if (tidx < 32) {
            uint4* src_ptr_v = reinterpret_cast<uint4*>(src_v_caches_ptr);
            uint4* dst_ptr_v = reinterpret_cast<uint4*>(dst_v_caches_ptr);
            int v_tidx = tidx - 16;
            
            // Source: read from the index position in source tensor  
            src_ptr_v += (blockIdx.x * heads * src_len * uint4_per_dim) 
                         + (blockIdx.y * src_len * uint4_per_dim)
                         + (index * uint4_per_dim);
            
            // Destination: write to the blockIdx.z position in dest tensor (topk index)
            dst_ptr_v += (blockIdx.x * heads * dst_len * uint4_per_dim) 
                         + (blockIdx.y * dst_len * uint4_per_dim)
                         + (blockIdx.z * uint4_per_dim);

            dst_ptr_v[v_tidx] = src_ptr_v[v_tidx];
        }
    }
}

void h2d_gather_kv(
    at::Tensor& src_k_caches, // bs, heads, kv_len, dim
    at::Tensor& src_v_caches, // bs, heads, kv_len, dim
    at::Tensor& selected_indices, // bs, heads, topk
    at::Tensor& dst_k_caches, // bs, heads, max_topk, dim
    at::Tensor& dst_v_caches // bs, heads, max_topk, dim
){
    // src_*: CPU host-pinned buffers accessed by the kernel via UVA.  We must
    // hand the kernel a *device-side* alias of these pointers so that nsys/
    // CUPTI doesn't reject them at cudaLaunchKernel intercept time.  See the
    // comment block above host_pinned_to_uva_dptr() for the full rationale.
    void * src_k_caches_ptr = resolve_kernel_arg_ptr(src_k_caches);
    void * src_v_caches_ptr = resolve_kernel_arg_ptr(src_v_caches);
    void * dst_k_caches_ptr = dst_k_caches.data_ptr();
    void * dst_v_caches_ptr = dst_v_caches.data_ptr();
    void * selected_indices_ptr = selected_indices.data_ptr();

    unsigned int bs = src_k_caches.size(0);
    unsigned int heads = src_k_caches.size(1);
    unsigned int src_len = src_k_caches.size(2);
    unsigned int dim = src_k_caches.size(3);
    unsigned int topk = selected_indices.size(2);
    unsigned int dst_len = dst_k_caches.size(2);

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    const c10::Device dev = dst_v_caches.device();
    c10::cuda::CUDAGuard device_guard(dev);

    dim3 grid = {bs, heads, topk};
    if (dim == 128) {
        h2d_gather_kv_kernel<128><<<grid, 32, 0, stream>>>(
            src_k_caches_ptr, 
            src_v_caches_ptr,
            dst_k_caches_ptr,
            dst_v_caches_ptr,
            selected_indices_ptr,
            src_len, dst_len
        );
    } else {
        TORCH_CHECK(false, "The dim must be 128.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "update_cache_cnt kernel failed:", ::cudaGetErrorString(result));

}


// ---------------------------------------------------------------------------
// Packed variant: directly writes the gathered topk rows into the *final*
// flash_attn_varlen packed buffer, using per-(batch, head) variable K.
//
// Layout / semantics:
//   src_k/src_v       : CPU pinned [bs, kv_heads, src_len, DIM]   (bf16)
//   dst_k_packed      : GPU        [total_k, DIM]                  (bf16)
//   dst_v_packed      : GPU        [total_k, DIM]                  (bf16)
//
//   indices_packed    : GPU        [total_topk]   int64
//                       Concatenated absolute KV indices, one segment per
//                       (b, h) pair, in row-major (b, h) order. The length
//                       of the (b, h) segment equals adaptive_k[b, h].
//
//   cu_seqlens_topk   : GPU        [V + 1]        int32   (V = bs * kv_heads)
//                       Prefix sums of adaptive_k. cu_seqlens_topk[vh] is the
//                       starting offset of (b, h)'s segment inside
//                       indices_packed.
//
//   cu_seqlens_k      : GPU        [V + 1]        int32
//                       Prefix sums of (sl_end + adaptive_k). Determines the
//                       starting row of (b, h)'s segment inside the *final*
//                       packed attention buffer (the same buffer used by
//                       flash_attn_varlen_func). The first sl_end rows of
//                       each segment are reserved for sink+local KV that the
//                       caller fills separately on GPU; this kernel only
//                       writes rows [cu_seqlens_k[vh] + sl_end ..
//                                    cu_seqlens_k[vh + 1]).
//
//   sl_end            : int   (= sink_size + local_size)
//   global_k          : int   (= max(adaptive_k) over all heads; grid Y dim)
//
// Grid layout: (V, global_k, 1). Each block handles a single output row of
// the packed buffer for one (b, h) pair. Threads 0..15 copy the K row, 16..31
// copy the V row (uint4-vectorized, DIM=128 only).
// ---------------------------------------------------------------------------
template<int DIM>
__global__
void h2d_gather_kv_kernel_packed(
    void* __restrict__ src_k_ptr,
    void* __restrict__ src_v_ptr,
    void* __restrict__ dst_k_packed_ptr,
    void* __restrict__ dst_v_packed_ptr,
    void* __restrict__ indices_packed_ptr,
    void* __restrict__ cu_seqlens_topk_ptr,
    void* __restrict__ cu_seqlens_k_ptr,
    int src_len,
    int kv_heads,
    int sl_end
) {
    const int vh = blockIdx.x;
    const int slot = blockIdx.y;
    const int tidx = threadIdx.x;

    const int* cu_topk = reinterpret_cast<int*>(cu_seqlens_topk_ptr);
    const int seg_start = cu_topk[vh];
    const int seg_len = cu_topk[vh + 1] - seg_start;
    if (slot >= seg_len) return;

    const int b = vh / kv_heads;
    const int h = vh - b * kv_heads;

    const int packed_topk_pos = seg_start + slot;
    const int64_t index = __ldg(
        reinterpret_cast<int64_t*>(indices_packed_ptr) + packed_topk_pos);

    const int* cu_k = reinterpret_cast<int*>(cu_seqlens_k_ptr);
    const int dst_row = cu_k[vh] + sl_end + slot;

    constexpr int uint4_per_dim = DIM * 2 / 16; // bf16 -> 2B/elem; 16B/uint4

    if (DIM == 128) {
        const int64_t src_off =
            (int64_t)b * kv_heads * src_len * uint4_per_dim
          + (int64_t)h * src_len * uint4_per_dim
          + (int64_t)index * uint4_per_dim;
        const int64_t dst_off = (int64_t)dst_row * uint4_per_dim;

        if (tidx < 16) {
            uint4* src = reinterpret_cast<uint4*>(src_k_ptr) + src_off;
            uint4* dst = reinterpret_cast<uint4*>(dst_k_packed_ptr) + dst_off;
            dst[tidx] = src[tidx];
        } else if (tidx < 32) {
            uint4* src = reinterpret_cast<uint4*>(src_v_ptr) + src_off;
            uint4* dst = reinterpret_cast<uint4*>(dst_v_packed_ptr) + dst_off;
            const int v_tidx = tidx - 16;
            dst[v_tidx] = src[v_tidx];
        }
    }
}




void h2d_gather_kv_packed(
    at::Tensor& src_k_caches,    // CPU pinned [bs, kv_heads, src_len, dim]
    at::Tensor& src_v_caches,    // CPU pinned [bs, kv_heads, src_len, dim]
    at::Tensor& dst_k_packed,    // GPU         [total_k, dim]
    at::Tensor& dst_v_packed,    // GPU         [total_k, dim]
    at::Tensor& indices_packed,  // GPU         [total_topk] int64
    at::Tensor& cu_seqlens_topk, // GPU         [V + 1]      int32
    at::Tensor& cu_seqlens_k,    // GPU         [V + 1]      int32
    int64_t sl_end,
    int64_t global_k             // grid Y dim (= max adaptive_k)
) {
    // src_*: CPU host-pinned. See host_pinned_to_uva_dptr() comment for why
    // we route through a UVA device-pointer alias before kernel launch.
    void* src_k_ptr = resolve_kernel_arg_ptr(src_k_caches);
    void* src_v_ptr = resolve_kernel_arg_ptr(src_v_caches);
    void* dst_k_ptr = dst_k_packed.data_ptr();
    void* dst_v_ptr = dst_v_packed.data_ptr();
    void* indices_ptr = indices_packed.data_ptr();
    void* cu_topk_ptr = cu_seqlens_topk.data_ptr();
    void* cu_k_ptr = cu_seqlens_k.data_ptr();

    const int bs = src_k_caches.size(0);
    const int kv_heads = src_k_caches.size(1);
    const int src_len = src_k_caches.size(2);
    const int dim = src_k_caches.size(3);
    const int V = bs * kv_heads;

    if (global_k <= 0 || V <= 0) return;

    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    c10::cuda::CUDAGuard device_guard(dst_k_packed.device());

    dim3 grid = {(unsigned)V, (unsigned)global_k, 1u};
    if (dim == 128) {
        h2d_gather_kv_kernel_packed<128><<<grid, 32, 0, stream>>>(
            src_k_ptr, src_v_ptr,
            dst_k_ptr, dst_v_ptr,
            indices_ptr, cu_topk_ptr, cu_k_ptr,
            src_len, kv_heads, (int)sl_end
        );
    } else {
        TORCH_CHECK(false, "The dim must be 128.");
    }

    const auto result = cudaGetLastError();
    TORCH_CHECK(result == cudaSuccess,
                "h2d_gather_kv_packed kernel failed:",
                ::cudaGetErrorString(result));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("h2d_gather_kv", &h2d_gather_kv);
  m.def("h2d_gather_kv_packed", &h2d_gather_kv_packed);
}