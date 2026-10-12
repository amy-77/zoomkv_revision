import torch
from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat as flash_attn_with_kvcache


def prefill_full_flash_attn(query_states, key_states, value_states, causal):
    """
    Prefill forward using flash attention with KV cache.
    注意：对于 GQA (Grouped Query Attention)，需要先 repeat K/V 到与 Q 相同的 head 数量。
    flash_attn_with_kvcache 期望 K/V 的 head 数与 Q 相同。
    """
    # 检查是否是 GQA (num_kv_heads != num_q_heads)
    bsz, seq_len, num_q_heads, head_dim = query_states.shape
    _, _, num_kv_heads, _ = key_states.shape
    
    if num_kv_heads != num_q_heads:
        # GQA：需要 repeat K/V
        num_key_value_groups = num_q_heads // num_kv_heads
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=2)
    
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        causal=causal
    )
    
    return attn_out



def decode_full_flash_attn(query_states, key_states, value_states, layer_idx, full_attn_cache):
    """
    Decode forward using flash attention with KV cache.
    注意：对于 GQA，K/V cache 中存储的是原始的 num_kv_heads，
    flash_attn_with_kvcache 可以自动处理 GQA（不需要手动 repeat）。
    """
    # 🔧 修复：直接使用 valid_length，无需手动补偿
    # 因为 flash_attn_cache.decode_update_kv_cache 已经在第一层更新了 valid_length
    # 所有层看到的 valid_length 都是正确的（包含了刚写入的新 token）
    valid_len = full_attn_cache.valid_length
    
    # decode 阶段：flash_attn_with_kvcache 会自动处理 GQA（不需要手动 repeat K/V）
    attn_out = flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        cache_seqlens=valid_len,
    )
    
    return attn_out
