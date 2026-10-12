import torch
import math
from .flash_attn_cache import flash_attn_cache
try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

class ExactTopKCache(flash_attn_cache):
    """
    A class representing the KV Cache with Exact Top-K selection logic.
    Inherits from flash_attn_cache for memory management but implements
    Top-K selection during decoding.
    """
    def __init__(
        self,
        valid_start,
        layer_num: int,
        batch_size: int,
        max_length: int,
        num_key_value_heads: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        layer_mapping: dict,
        num_gpus: int,
        model_size: int,
        candidate_ratio: float,
        recent_size: int = 64,
        sink_size: int = 4,
        enable_decode_timing: bool = True
    ) -> None:
        super().__init__(
            valid_start, layer_num, batch_size, max_length, 
            num_key_value_heads, num_heads, head_dim, dtype, 
            layer_mapping, num_gpus, model_size, enable_decode_timing
        )
        self.candidate_ratio = candidate_ratio
        self.recent_size = recent_size  # 固定的 recent tokens 数量（local window）
        self.sink_size = sink_size
        self.num_key_value_groups = num_heads // num_key_value_heads
        # Top-K specific stats
        self.decode_token_count = 0
        # Check for flash_attn_func
        if flash_attn_func is None:
            print("Warning: flash_attn_func not found. Top-K computation will fail.")



    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        Update the cache with new token.
        Identical to flash_attn_cache but we don't strictly need the return value
        for external attention if we use compute().
        """
        # Call parent to update cache
        super().decode_update_kv_cache(key_states, value_states, layer_idx)
        # Return None because we compute attention internally in compute()
        # taking the query. The caller should call compute() after this.
        return None, None

    def compute(self, query_states, layer_idx):
        """
        Perform Top-K selection and compute attention.
        Args:
            query_states: [batch, 1, num_heads, head_dim]
            layer_idx: current layer index 
        Returns:
            attn_output: [batch, 1, num_heads, head_dim]
        """
        # query_states from model is usually [B, 1, H, D]
        # We need [B, H, 1, D] for scoring logic
        q = query_states.transpose(1, 2) # [B, H, 1, D]
        
        # Get full valid cache
        # self.valid_length is a tensor, assuming batch size 1 for now or uniform length
        # logic similar to flash_attn_cache usage
        # Note: valid_length has been incremented in decode_update_kv_cache
        valid_len = self.valid_length[0].item()
        
        # Retrieve from cache: [B, MaxLen, KvH, D] -> slice -> [B, Len, KvH, D]
        k_full = self.key_cache[layer_idx][:, :valid_len, :, :]
        v_full = self.value_cache[layer_idx][:, :valid_len, :, :]
        
        # Transpose to [B, KvH, Len, D] for processing
        k_full = k_full.permute(0, 2, 1, 3)
        v_full = v_full.permute(0, 2, 1, 3)
        
        # Expand GQA if needed: [B, KvH, Len, D] -> [B, H, Len, D]
        if self.num_key_value_groups > 1:
            k_full = k_full.repeat_interleave(self.num_key_value_groups, dim=1)
            v_full = v_full.repeat_interleave(self.num_key_value_groups, dim=1)
            
        # Perform Exact Top-K Selection
        # We use the logic from ExactTopKQuantizer._exact_topk_selection
        
        batch_size, num_heads, _, head_dim = q.shape
        total_kv_len = valid_len
        
        # Calculate budgets
        # Sink: 固定数量的开头 tokens
        sink_keep = min(self.sink_size, total_kv_len)
        # Recent (Local): 固定数量的最近 tokens
        recent_keep = min(self.recent_size, total_kv_len - sink_keep)
        # Important: 根据 candidate_ratio 从中间区域选择
        total_budget = max(1, int(total_kv_len * self.candidate_ratio))
        important_keep = max(0, total_budget - sink_keep - recent_keep)
        
        # Compute scores
        # q: [B, H, 1, D], k_full: [B, H, L, D]
        # score: [B, H, L]
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum('bhqd,bhkd->bhqk', q, k_full).squeeze(2) * scale
        
        # Select Top-K indices in the middle region (between sink and recent)
        # Region: [sink_keep : -recent_keep]
        # Note: if sink_keep + recent_keep >= total_kv_len, middle region is empty.
        
        middle_start = sink_keep
        middle_end = total_kv_len - recent_keep
        
        if middle_end > middle_start and important_keep > 0:
            # Select from middle region
            middle_scores = scores[..., middle_start:middle_end]
            # topk
            topk_val, topk_indices = middle_scores.topk(min(important_keep, middle_scores.shape[-1]), dim=-1)
            
            # Adjust indices to be absolute
            topk_indices = topk_indices + middle_start
            
            # Sort indices to preserve temporal order (optional but good for attention)
            topk_indices, _ = topk_indices.sort(dim=-1)
            
            # Gather keys/values
            # We need to gather from k_full: [B, H, L, D]
            # indices: [B, H, K] -> expand to [B, H, K, D]
            gather_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            
            k_topk = k_full.gather(dim=2, index=gather_indices)
            v_topk = v_full.gather(dim=2, index=gather_indices)
            
            # Get sink and recent parts
            k_sink = k_full[:, :, :sink_keep, :]
            v_sink = v_full[:, :, :sink_keep, :]
            
            k_recent = k_full[:, :, middle_end:, :]
            v_recent = v_full[:, :, middle_end:, :]
            
            # Concatenate: Sink + TopK + Recent
            k_selected = torch.cat([k_sink, k_topk, k_recent], dim=2)
            v_selected = torch.cat([v_sink, v_topk, v_recent], dim=2)
            
        else:
            # No middle region or no budget for it, just take sink + recent (or full if small)
            if total_kv_len <= sink_keep + recent_keep:
                k_selected = k_full
                v_selected = v_full
            else:
                # Should generally fall into above case, but handle overlap carefully
                # If middle_end <= middle_start, it means recent + sink overlap or cover everything
                # Just taking full is safest if overlap, but let's follow logic:
                # if total > sink+recent, we have middle.
                # if total <= sink+recent, we just take full.
                k_selected = k_full
                v_selected = v_full

        # Compute Attention using Flash Attention
        # flash_attn_func expects [batch, seq_len, heads, dim] usually?
        # Let's check usage in ExactTopKQuantizer:
        # attn_output = flash_attn_func(
        #           query_states.transpose(1, 2),
        #           selected_key_states.transpose(1, 2), 
        #           selected_value_states.transpose(1, 2),
        #           causal=True).transpose(1, 2)
        #
        # In Quantizer, query_states was [B, H, 1, D]. Transposed -> [B, 1, H, D].
        # selected_key_states was [B, H, L_new, D]. Transposed -> [B, L_new, H, D].
        # So flash_attn_func expects [B, Seq, H, D].
        
        # Prepare inputs for flash_attn_func
        q_in = q.transpose(1, 2) # [B, 1, H, D]
        k_in = k_selected.transpose(1, 2) # [B, L_sel, H, D]
        v_in = v_selected.transpose(1, 2) # [B, L_sel, H, D]
        
        attn_output = flash_attn_func(
            q_in, k_in, v_in, 
            causal=True # Causal masking? usually yes for decoder.
            # Note: since we selected tokens, relative positions are lost/changed. 
            # But standard flash attn with causal=True masks based on position in the sequence provided.
            # Since q is length 1 (last), and k/v are past + current, 
            # q should attend to all k/v provided.
            # "causal=True" with q_len=1 might just mean "attend to all valid k".
        )
        
        # attn_output: [B, 1, H, D]
        
        return attn_output

