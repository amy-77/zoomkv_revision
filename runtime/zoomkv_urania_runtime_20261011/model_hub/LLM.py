import os
import time
import torch
import torch.nn.functional as F
import torch.cuda.nvtx as nvtx
from termcolor import colored

# 设置 Triton 缓存目录到有足够空间的位置
os.environ.setdefault('TRITON_CACHE_DIR', '/data/qyl/zoomkv_revision/results/pariskv/.triton_cache')


class LLM:
    """
    A class representing the LLM (currently support Llama and Qwen).
    """

    def __init__(
        self, 
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str
    ) -> None:
        """ Initializes the LLM.
        Args:
            model_name (str): The name of the model.
            max_length (int): The maximum length (prefill+decode) of sequences.
            dtype (torch.dtype): The data type for model computations.
            device_map (str): The device for model, suppor 'cuda:x' or 'auto (automatically use all visible GPUs)'.
        """

        self.model_name = model_name
        self.max_length = max_length
        self.dtype = dtype
        self.device_map = device_map
        self.timing_stats = {
            "decode_update_kv_cache": 0.0,
            "decode_attention": 0.0,
        }
        # 默认单样本逐条 Prefill，可通过配置提升为多样本并行
        self.prefill_chunk_size = 1


    def _update_prefill_chunk_size(self, cfg):
        """
        根据配置更新 prefill_chunk_size，缺省情况下 fallback 到 batch_size。
        cfg 可以是整体配置或某个 attention_type 的子配置。
        """
        chunk = None
        if isinstance(cfg, dict):
            chunk = cfg.get("prefill_chunk_size")
            if chunk is None:
                attn_cfg = cfg.get(getattr(self, "attention_type", None), {})
                if isinstance(attn_cfg, dict):
                    chunk = attn_cfg.get("prefill_chunk_size")
        if isinstance(chunk, int) and chunk > 0:
            self.prefill_chunk_size = min(chunk, self.batch_size)
        else:
            # 确保 chunk size 不超过 batch_size
            self.prefill_chunk_size = min(self.prefill_chunk_size or 1, self.batch_size)


    def sample_next_token(self, logits, temperature=1.0, top_p=0.0, top_k=0):
        """
        采样下一个token
        Args:
            logits: [batch_size, vocab_size] 或 [batch_size, 1, vocab_size] 模型输出的logits
            temperature: 温度参数，控制随机性 (0.0-2.0)
                - temperature=0: 等同于greedy (argmax)
            top_p: nucleus sampling，保留累积概率>=top_p的tokens (0.0-1.0)
            top_k: top-k sampling，只保留概率最高的k个tokens
        Returns:
            next_token: [batch_size] 采样得到的下一个token
        """
        # 确保 logits 是 2D: [batch_size, vocab_size]
        if logits.dim() == 3:
            logits = logits.squeeze(1)  # [batch_size, 1, vocab_size] -> [batch_size, vocab_size]
        elif logits.dim() == 1:
            logits = logits.unsqueeze(0)  # [vocab_size] -> [1, vocab_size]
        
        # Greedy decoding (temperature=0 或接近0)
        if temperature < 1e-5:
            return logits.argmax(dim=-1)
        
        # 应用temperature
        logits = logits / temperature
        
        # Top-k filtering
        if top_k > 0:
            # 保留top-k，其余设为-inf
            top_k = min(top_k, logits.size(-1))  # 确保k不超过vocab_size
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        # Top-p (nucleus) filtering
        if top_p > 0.0 and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # 移除累积概率超过top_p的tokens
            # 例如：probs=[0.5, 0.3, 0.15, 0.05], top_p=0.9
            #      cumulative=[0.5, 0.8, 0.95, 1.0]
            #      > 0.9 得到 [F, F, T, T]
            sorted_indices_to_remove = cumulative_probs > top_p
            
            # 右移一位，保留第一个使累积概率超过阈值的token
            # 右移后：[F, F, F, T]，保留前3个token（累积概率0.95刚好>=0.9）
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False  # 确保至少保留概率最高的token
            
            # 将排序后的mask映射回原始logits顺序
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # 计算概率并采样
        probs = F.softmax(logits, dim=-1)
        
        # 检查是否有有效的概率分布
        if torch.isnan(probs).any() or torch.isinf(probs).any():
            print(colored(" [Warning] NaN or Inf in probs, falling back to argmax", 'red'))
            return logits.argmax(dim=-1)
        
        # 确保概率和为正数（防止所有概率为0）
        prob_sum = probs.sum(dim=-1)
        if (prob_sum <= 0).any():
            print(colored(" [Warning] Invalid probability distribution, falling back to argmax", 'red'))
            return logits.argmax(dim=-1)
        
        # multinomial 需要 2D 输入: [batch_size, vocab_size]
        # 输出: [batch_size, 1]，我们 squeeze 成 [batch_size]
        try:
            next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        except RuntimeError as e:
            print(colored(f" [Warning] Multinomial sampling failed: {e}, falling back to argmax", 'red'))
            next_token = logits.argmax(dim=-1)
        
        return next_token
    
    
    
    
    

    def layer_prefill(self, layer_idx, start_bdx, hidden_states):
        # print(f'Layer = {layer_idx}, start_bdx = {start_bdx}')

        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]
        
        # original hidden_states used as residual, clone a new one to process
        temp_hidden_states = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            temp_hidden_states[:, start_idx:end_idx, :] = self.layernorm(temp_hidden_states[:, start_idx:end_idx, :], 
                                                                         layer.input_layernorm_variance_epsilon, 
                                                                         layer.input_layernorm_weight)
        
        # 1. 计算QKV
        query_states, key_states, value_states = self.wqkv(temp_hidden_states, layer)
        del temp_hidden_states
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)       # reshape [bs, seq_len, dim] => [bs, seq_len, head, head_dim]
        key_states = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
        
        # 2. 更新KV cache（异步offload + 编码 + 建索引）
        key_states, value_states = self.kv_cache.prefill_update_kv_cache(query_states, key_states, value_states, layer_idx, start_bdx)
        # 3. 计算attention
        temp_attn_out = self.prefill_attention(query_states, key_states, value_states)

 
        # 同步：等待异步操作完成（RetroInfer 需要，Polar ANN 是空操作）
        self.kv_cache.sync(layer_idx, start_bdx)

        del query_states, key_states, value_states

        # Reshape attention output from [bsz, seq_len, num_heads, head_dim] to [bsz, seq_len, num_heads * head_dim]
        # For Qwen3 MoE: num_heads * head_dim may not equal hidden_size
        attn_out_dim = temp_attn_out.shape[-2] * temp_attn_out.shape[-1]  # num_heads * head_dim
        temp_attn_out = temp_attn_out.reshape(bsz, seq_len, attn_out_dim)
        hidden_states += self.wo(temp_attn_out, layer, bsz, seq_len, attn_out_dim)
        del temp_attn_out

        # post attention
        residual = hidden_states.clone()

        # chunk for lower memory comsumption
        for start_idx in range(0, seq_len, 8192//bsz):
            end_idx = min(seq_len, start_idx + 8192//bsz)
            hidden_states[:, start_idx:end_idx, :] = self.layernorm(hidden_states[:, start_idx:end_idx, :], 
                                                                    layer.post_attention_layernorm_variance_epsilon, 
                                                                    layer.post_attention_layernorm_weight)
            hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :], layer)   
        
        hidden_states += residual

        del residual
                                                                                                   
        return hidden_states

     
    # 逐层decode 
    def layer_decode(self, layer_idx, hidden_states):
        residual = hidden_states
        bsz, seq_len, dim = hidden_states.shape
        layer = self.layers[layer_idx]

        hidden_states = self.layernorm(hidden_states, layer.input_layernorm_variance_epsilon, layer.input_layernorm_weight)
        
        query_states, key_states, value_states = self.wqkv(hidden_states, layer)
        query_states, key_states = self.position_embedd(query_states, key_states)

        query_states = query_states.view(bsz, -1, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim)
        
        # 更新key和value cache
        # torch.cuda.synchronize()
        start = time.perf_counter()
        nvtx.range_push("decode_update_kv_cache")
        key_states, value_states = self.kv_cache.decode_update_kv_cache(key_states, value_states, layer_idx)
        nvtx.range_pop()
        # torch.cuda.synchronize()
        self.timing_stats["decode_update_kv_cache"] += time.perf_counter() - start
        # 计算attention输出, 这里包含了找topk
        
        # torch.cuda.synchronize()
        attn_start = time.perf_counter()
        nvtx.range_push("decode_attention")
        attn_out = self.decode_attention(query_states, key_states, value_states, layer_idx)
        nvtx.range_pop()
        # torch.cuda.synchronize()
        self.timing_stats["decode_attention"] += time.perf_counter() - attn_start
        # Reshape attention output from [bsz, seq_len, num_heads, head_dim] to [bsz, seq_len, num_heads * head_dim]
        # For Qwen3 MoE: num_heads * head_dim may not equal hidden_siz
        attn_out_dim = attn_out.shape[-2] * attn_out.shape[-1]  # num_heads * head_dim
        attn_out = attn_out.reshape(bsz, seq_len, attn_out_dim)
        hidden_states = self.wo(attn_out, layer, bsz, seq_len, attn_out_dim)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layernorm(hidden_states, layer.post_attention_layernorm_variance_epsilon, layer.post_attention_layernorm_weight)
        hidden_states = self.mlp(hidden_states, layer)
        hidden_states = residual + hidden_states

        return hidden_states


    def prefill_forward(self, inputs_ids):
        bsz, seq_len = inputs_ids.shape
        device = inputs_ids.device

        last_hidden_states = torch.empty((bsz, 1, self.hidden_size), dtype=self.dtype, device=device)
        chunk_size = min(self.prefill_chunk_size or 1, bsz)
        for start_bdx in range(0, bsz, chunk_size):
            end_bdx = min(bsz, start_bdx + chunk_size)
            hidden_states = self.word_embedding(inputs_ids[start_bdx:end_bdx])  # [chunk, seq_len, hidden_size]
            if self.num_gpus > 1:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                    hidden_states = self.parameter_move(hidden_states, ldx)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :].to(self.layers[0].device)
            else:
                for ldx in range(self.num_layers):
                    hidden_states = self.layer_prefill(ldx, start_bdx, hidden_states)
                last_hidden_states[start_bdx:end_bdx] = hidden_states[:, -1:, :]
        
        last_hidden_states = self.layernorm(last_hidden_states.contiguous(), self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(last_hidden_states)
        
        return logits
        

    def decode_forward(self, inputs_ids):
        hidden_states = self.word_embedding(inputs_ids)

        if self.num_gpus > 1:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
                hidden_states = self.parameter_move(hidden_states, ldx)
            hidden_states = hidden_states.to(self.layers[0].device)
        else:
            for ldx in range(self.num_layers):
                hidden_states = self.layer_decode(ldx, hidden_states)
        
        hidden_states = self.layernorm(hidden_states[:, -1:, :], self.norm_variance_epsilon, self.norm_weight)
        logits = self.lm(hidden_states)
        
        return logits




    def _is_stop_token(self, output_ids):
        """Return True when every item in the batch generated a chat stop token."""
        if not hasattr(self, 'tokenizer'):
            return False

        eos_token_id = self.tokenizer.eos_token_id
        stop_texts = {"<|eot_id|>", "<|im_end|>", "<|endoftext|>", "<|end|>"}
        stop_ids = {151329, 151336, 151338}  # GLM stop ids used by ShadowKV.

        token_ids = output_ids.detach().view(-1).tolist()
        if not token_ids:
            return False

        for token_id in token_ids:
            if eos_token_id is not None and token_id == eos_token_id:
                continue
            if token_id in stop_ids:
                continue
            if self.tokenizer.decode([token_id]) in stop_texts:
                continue
            return False
        return True

    def _decode_progress_interval(self):
        try:
            return int(os.environ.get("ZOOMKV_GEN_PROGRESS_EVERY", "0"))
        except ValueError:
            return 0

    def _log_decode_progress(self, n_generated, decode_start):
        interval = self._decode_progress_interval()
        if interval <= 0:
            return
        if n_generated != 1 and (n_generated % interval) != 0:
            return
        elapsed = time.time() - decode_start
        tps = (n_generated / elapsed) if elapsed > 0 else 0.0
        limit = max(int(getattr(self, "max_new_length", 0)) - 1, 0)
        print(
            f"[decode] tokens={n_generated}/{limit} elapsed={elapsed:.1f}s "
            f"{tps:.1f} tok/s",
            flush=True,
        )

    def inference(self, inputs_ids):
        outputs_ids = []    # multi iteration, multi request
        output_ids = []     # single iteration, multi request
        
        torch.cuda.synchronize()
        prefill_start = time.time()
        logits = self.prefill_forward(inputs_ids=inputs_ids)
        output_ids = logits.argmax(dim=-1)
        outputs_ids.append(output_ids)
        self.move()
        torch.cuda.synchronize()
        prefill_end = time.time()



        decode_start = time.time()
        actual_generated = 0  # 实际生成的token数

        for step_idx in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            output_ids = logits.argmax(dim=-1)
            outputs_ids.append(output_ids)
            actual_generated += 1
            self._log_decode_progress(actual_generated, decode_start)
            
            if not getattr(self, 'disable_early_stop', False):
                if self._is_stop_token(output_ids):
                    break
        
        torch.cuda.synchronize()
        decode_end = time.time()
        
        # 分别计算 prefill 和 decode 时间
        prefill_time = prefill_end - prefill_start
        decode_time = decode_end - decode_start
        total_time = prefill_time + decode_time
        
        avg_latency = round(decode_time * 1000 / actual_generated, 2) if actual_generated > 0 else 0
        throughput = round(self.batch_size * actual_generated / decode_time, 2) if decode_time > 0 else 0
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()
        
        # 返回生成的 token ids 和详细的时间统计
        timing_stats = {
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'actual_generated': actual_generated,
            'avg_latency_ms': avg_latency,
            'throughput': throughput
        }
        
        return outputs_ids, timing_stats


    def inference_cot(self, inputs_ids):
        """
        CoT (Chain-of-Thought) 推理专用版本
        支持灵活的采样策略，适合需要多样性的推理任务
        """
        outputs_ids = []
        output_ids = []
        
        torch.cuda.synchronize()
        prefill_start = time.time()

        logits = self.prefill_forward(inputs_ids=inputs_ids)
        # 使用采样策略（sample_next_token 返回 [batch_size]，需要 unsqueeze 成 [batch_size, 1]）
        output_ids = self.sample_next_token(logits, self.temperature, self.top_p, self.top_k).unsqueeze(-1)
        outputs_ids.append(output_ids)
        self.move()

        torch.cuda.synchronize()
        prefill_end = time.time()
        
        decode_start = time.time()

        actual_generated = 0
        
        for step_idx in range(self.max_new_length-1):
            logits = self.decode_forward(inputs_ids=output_ids)
            # 使用采样策略（CoT核心区别，sample_next_token 返回 [batch_size]，需要 unsqueeze 成 [batch_size, 1]）
            output_ids = self.sample_next_token(logits, self.temperature, self.top_p, self.top_k).unsqueeze(-1)
            outputs_ids.append(output_ids)
            actual_generated += 1
            self._log_decode_progress(actual_generated, decode_start)
            
            if not getattr(self, 'disable_early_stop', False):
                if self._is_stop_token(output_ids):
                    break
                        
        torch.cuda.synchronize()
        decode_end = time.time()
        avg_latency = round((decode_end - decode_start) * 1000 / actual_generated, 2) if actual_generated > 0 else 0
        throughput = round(self.batch_size * actual_generated / (decode_end - decode_start), 2) if (decode_end - decode_start) > 0 else 0
        
        outputs_ids = torch.cat(outputs_ids, dim=-1).tolist()

        prefill_time = prefill_end - prefill_start
        decode_time = decode_end - decode_start
        timing_stats = {
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': prefill_time + decode_time,
            'actual_generated': actual_generated,
            'avg_latency_ms': avg_latency,
            'throughput': throughput,
        }

        return outputs_ids, timing_stats




    def generate(self, attention_type, inputs_ids, attention_masks, max_new_length, attn_config=None,
                 temperature=0.0, top_p=0.0, top_k=0, disable_early_stop=True):
        """ LLM Inference.
        Args:
            attention_type: str,
            input_ids (torch.tensor): The input of LLM.
            attention_masks (torch.tensor): The attention masks of LLM.
            max_new_length (int): The maximum length of generated sequences.
            attn_config: KV cache configuration.
            temperature (float): 温度参数 (0.0=greedy, >0使用采样)
            top_p (float): Nucleus sampling参数 (0.0=不使用)
            top_k (int): Top-k sampling参数 (0=不使用)
            disable_early_stop (bool): 禁用 EOS 提前停止（用于吞吐测试）
        """
        self.disable_early_stop = disable_early_stop

        bs, input_length = inputs_ids.shape
        assert input_length + max_new_length <= self.max_length, \
        f"Error: input_length({input_length}) + max_new_length({max_new_length}) exceeds max_length({self.max_length})"

        self.batch_size = bs
        self.input_length = input_length
        self.max_new_length = max_new_length
        self.attention_type = attention_type
        # 先确保 chunk size 不超过 batch size，之后 init_kv_cache 中还会根据配置更新
        self.prefill_chunk_size = min(self.prefill_chunk_size or 1, self.batch_size)
        
        # 保存采样参数
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k

        valid_start = attention_masks.shape[1] - torch.sum(attention_masks, dim=-1).detach().cpu().numpy()
        del attention_masks

        self.init_kv_cache(input_length, valid_start, attn_config)

        # 根据采样参数选择推理模式
        # 注意：只有 temperature > 0 时，top_p 和 top_k 才有意义
        # 所以判断是否使用 CoT Mode 只看 temperature
        use_cot_mode = (temperature > 1e-5)
        
        if use_cot_mode:
            # 采样模式：使用 CoT 推理路径（支持 temperature, top_p, top_k）
            outputs = self.inference_cot(inputs_ids)
        else:
            # 贪心模式：使用标准推理路径（argmax）
            outputs = self.inference(inputs_ids)

        return outputs
