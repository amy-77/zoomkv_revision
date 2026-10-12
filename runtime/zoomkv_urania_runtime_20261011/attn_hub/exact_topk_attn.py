from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat as flash_attn_with_kvcache

def exact_topk_prefill_attn(query_states, key_states, value_states, causal):
    """
    Prefill forward for Exact Top-K.
    Uses standard Flash Attention as we want to store full history first.
    """
    # Ensure GQA handling if needed, similar to other implementations
    bsz, seq_len, num_q_heads, head_dim = query_states.shape
    _, _, num_kv_heads, _ = key_states.shape
    
    if num_kv_heads != num_q_heads:
        num_key_value_groups = num_q_heads // num_kv_heads
        key_states = key_states.repeat_interleave(num_key_value_groups, dim=2)
        value_states = value_states.repeat_interleave(num_key_value_groups, dim=2)

    return flash_attn_with_kvcache(
        q=query_states, 
        k_cache=key_states, 
        v_cache=value_states, 
        causal=causal
    )

def exact_topk_decode_attn(query_states, key_states, value_states, layer_idx, kv_cache):
    """
    Decode forward for Exact Top-K.
    Delegates to ExactTopKCache.compute() which performs the selection and attention.
    """
    # Note: kv_cache.decode_update_kv_cache() should have been called before this 
    # (usually in LlamaModel) to append the new token.
    
    return kv_cache.compute(query_states, layer_idx)

