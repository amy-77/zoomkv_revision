"""
PolarANN Attention Functions

专门为 PolarANN 方法设计的 Attention 函数
- Prefill阶段：使用标准 Flash Attention
- Decode阶段：使用 Polar ANN collision 检索 + GPU 端到端计算
"""

from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat as flash_attn_with_kvcache


def polar_prefill_attn(query_states, key_states, value_states, causal):
    """
    PolarANN Prefill阶段的Attention计算
    
    Prefill阶段使用标准的Flash Attention计算全部tokens的attention
    后台：prefill_update_kv_cache 会进行 Polar编码 + 建立倒排索引（全GPU）
    
    Args:
        query_states: query向量 [batch_size, seq_len, num_heads, head_dim]
        key_states: key向量 [batch_size, seq_len, num_kv_heads, head_dim]
        value_states: value向量 [batch_size, seq_len, num_kv_heads, head_dim]
        causal: 是否使用因果mask
    
    Returns:
        attn_out: attention输出 [batch_size, seq_len, num_heads, head_dim]
    """
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states,
        causal=causal
    )
    
    return attn_out


def polar_decode_attn(query_states, key_states, value_states, layer_idx, kv_cache):
    """
    使用 Polar ANN collision-based 检索方法：
    1. 子空间碰撞计数：在多个子空间中统计 query 与 key 的碰撞次数
    2. 候选集筛选：选择碰撞次数最高的 keys 作为候选集
    3. 精确计算 Top-K：对候选集计算精确相似度，选出最终的 top-k tokens
    4. Flash Attention：只对 Steady Zone + Top-K Retrieved Tokens 计算 attention
    Args:
        query_states: query向量 [batch_size, 1, num_heads, head_dim]
        key_states: 未使用（保留接口兼容性）
        value_states: 未使用（保留接口兼容性）
        layer_idx: 层索引
        kv_cache: polar_cache 实例
    Returns:
        attn_out: attention输出 [batch_size, 1, num_heads, head_dim]
    
    Raises:
        TypeError: 如果 kv_cache 未实现 PolarANN 所需的 compute 接口
    """
    # 接口检查（duck typing）：
    # 避免因动态模块加载路径不同导致的 isinstance/类名误判。
    if kv_cache is None or not hasattr(kv_cache, "compute") or not callable(getattr(kv_cache, "compute")):
        got_type = type(kv_cache).__name__
        got_module = type(kv_cache).__module__
        raise TypeError(
            "polar_decode_attn expects kv_cache to provide a callable compute(query_states, layer_idx), "
            f"but got {got_module}.{got_type}. "
            "Please ensure attention_type='PolarANN' and cache initialization are correctly configured."
        )
    
    # 调用 polar_cache 的 compute 方法
    # 内部会执行：GPU Polar ANN 检索 → 组装 KV → Flash Attention
    attn_out = kv_cache.compute(
        query_states.contiguous(), layer_idx
    )
    
    return attn_out

