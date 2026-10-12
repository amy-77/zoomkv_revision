"""
Flash Attention Compatibility Layer

提供跨 GPU 架构的兼容性支持：
- Ampere+ (compute capability >= 8.0): 使用 Flash Attention 2.x
- Turing (compute capability 7.5): 使用 Flash Attention 1.x
"""

import os
import torch

# 设置 Triton 缓存目录到有足够空间的位置；保留用户已设置的值，否则落到项目根 .triton 下
os.environ.setdefault(
    'TRITON_CACHE_DIR',
    os.path.join(
        os.environ.get('PROJECT_ROOT', os.getcwd()),
        '.triton', 'cache',
    ),
)


def get_compute_capability():
    """获取当前 GPU 的计算能力"""
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        return major + minor / 10
    return 0.0


def check_flash_attn_support():
    """检查是否支持 Flash Attention 2.x"""
    compute_cap = get_compute_capability()
    # Flash Attention 2.x 需要 compute capability >= 8.0 (Ampere+)
    return compute_cap >= 8.0


# 全局标志：是否支持 Flash Attention 2.x
_FLASH_ATTN_SUPPORTED = check_flash_attn_support()

# 检查是否禁用 Triton 回退（通过环境变量控制）
_DISABLE_TRITON_FALLBACK = os.environ.get('DISABLE_TRITON_FALLBACK', 'false').lower() in ('true', '1', 'yes')


def flash_attn_with_kvcache_compat(
    q,
    k_cache,
    v_cache,
    causal=False,
    **kwargs
):
    """
    Flash Attention 兼容函数
    
    自动选择最优实现：
    - GPU >= Ampere: Flash Attention 2.x (最快)
    - GPU < Ampere: Flash Attention 1.x (兼容 Turing)
    
    Args:
        q: Query tensor [batch, seqlen_q, num_heads, head_dim]
        k_cache: Key cache [batch, seqlen_k, num_heads, head_dim]
        v_cache: Value cache [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask
        **kwargs: Additional arguments
    
    Returns:
        out: Attention output [batch, seqlen_q, num_heads, head_dim]
    """
    if _FLASH_ATTN_SUPPORTED:
        # 使用 Flash Attention 2.x (最快)
        try:
            from flash_attn import flash_attn_with_kvcache
            return flash_attn_with_kvcache(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                causal=causal,
                **kwargs
            )
        except Exception as e:
            print(f"[Flash Attention] Warning: Flash Attention 2.x failed: {e}")
            if _DISABLE_TRITON_FALLBACK:
                print("[Flash Attention] Triton fallback is disabled. Raising error.")
                raise RuntimeError(
                    f"Flash Attention 2.x failed and Triton fallback is disabled. "
                    f"Error: {e}. "
                    f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
                )
            print("[Flash Attention] Falling back to Triton implementation")
            return _triton_attention_fallback(q, k_cache, v_cache, causal)
    else:
        # GPU 不支持 Flash Attention 2.x
        if _DISABLE_TRITON_FALLBACK:
            raise RuntimeError(
                f"GPU compute capability {get_compute_capability()} < 8.0 does not support Flash Attention 2.x, "
                f"and Triton fallback is disabled. "
                f"To enable Triton fallback, unset DISABLE_TRITON_FALLBACK environment variable."
            )
        # 使用 Triton 实现
        return _triton_attention_fallback(q, k_cache, v_cache, causal)


def _triton_attention_fallback(q, k, v, causal=False):
    """
    Triton Flash Attention 实现 (用于不支持 Flash Attention 2.x 的 GPU)
    
    使用 Triton kernel 实现 Flash Attention，支持 Turing GPU (compute capability 7.5+)
    
    注意：Turing GPU 不支持 BFloat16，会自动转换为 Float16
    
    Args:
        q: [batch, seqlen_q, num_heads, head_dim]
        k: [batch, seqlen_k, num_heads, head_dim]
        v: [batch, seqlen_k, num_heads, head_dim]
        causal: Whether to apply causal mask
    
    Returns:
        out: [batch, seqlen_q, num_heads, head_dim]
    """
    try:
        # Turing GPU 不支持 BFloat16，需要转换为 Float16
        original_dtype = q.dtype
        need_conversion = (original_dtype == torch.bfloat16)
        
        if need_conversion:
            # print(f"[Flash Attention] Converting BF16 → FP16 for Turing GPU")
            q = q.to(torch.float16)
            k = k.to(torch.float16)
            v = v.to(torch.float16)
        
        from attn_hub.triton_flash_attn import triton_flash_attention
        out = triton_flash_attention(q, k, v, causal=causal)
        
        # 转换回原始数据类型
        if need_conversion:
            out = out.to(original_dtype)
        
        return out
    except Exception as e:
        print(f"[Flash Attention] Triton implementation failed: {e}")
        print("[Flash Attention] Error details:")
        import traceback
        traceback.print_exc()
        raise RuntimeError(
            f"Triton Flash Attention failed. GPU compute capability: {get_compute_capability()}. "
            f"Please check Triton installation or GPU compatibility."
        )


# 在模块导入时打印 GPU 兼容性信息
compute_cap = get_compute_capability()
if compute_cap > 0:
    print(f"[Flash Attention Compat] GPU Compute Capability: {compute_cap}")
    if _FLASH_ATTN_SUPPORTED:
        print("[Flash Attention Compat] Using Flash Attention 2.x (native)")
        if _DISABLE_TRITON_FALLBACK:
            print("[Flash Attention Compat] Triton fallback is DISABLED (will raise error if Flash Attention 2.x fails)")
    else:
        print(f"[Flash Attention Compat] GPU compute capability {compute_cap} < 8.0")
        if _DISABLE_TRITON_FALLBACK:
            print("[Flash Attention Compat] Triton fallback is DISABLED (will raise error)")
        else:
            print("[Flash Attention Compat] Using Triton Flash Attention (fallback)")
else:
    print("[Flash Attention Compat] No CUDA device found")
