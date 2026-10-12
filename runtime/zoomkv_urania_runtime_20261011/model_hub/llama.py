"""Llama / DeepSeek-R1-Distill-Llama 适配 (PolarANN density release).

这版基于另一台服务器上较老的 llama.py，做了三处与当前 density release 对齐的改动：
  1. 移除 SuCo/PQ/Polar_PQ/RetroInfer 相关 import 和分支 — 这些 cache/attn
     在 density release 里已经被剥离了 (见 cache_hub/__init__.py、attn_hub/__init__.py)。
     现在只支持 Full_Flash_Attn / PolarANN / ExactTopK 三种 attention_type。
  2. transformers 5.x 把 ``rope_theta`` 从顶层迁移到 ``rope_parameters`` dict；
     这里读取时做了 fallback。同时 YaRN 入口接 ``rope_scaling`` kwarg 并把
     ``rope_parameters`` / ``rope_scaling`` / ``rope_theta`` 三个字段都同步
     写到 AutoConfig 后再交给 ``from_pretrained``，避免 _compute_yarn_parameters
     拿到 None。
  3. ``enable_offload`` 的默认值改成 True（与 PolarANN 主路径一致：4-bit code 留
     GPU、原始 KV 在 CPU pinned）。

与 qwen.py 的差异：Llama 没有 per-head q_norm / k_norm，没有 head_dim 字段，
所以 init_layer 里也不再读 q_norm/k_norm，head_dim 走 hidden//num_heads 计算。
"""

import gc
import re
import os
import json

import torch
import torch.nn.functional as F
import flashinfer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from .LLM import LLM
from cache_hub import flash_attn_cache, ExactTopKCache
from attn_hub import (
    prefill_full_flash_attn,
    decode_full_flash_attn,
    polar_prefill_attn,
    polar_decode_attn,
    exact_topk_prefill_attn,
    exact_topk_decode_attn,
)


class LlamaLayer:
    """A class representing the Llama transformer block."""

    def __init__(self, layer_idx, device) -> None:
        self.layer_idx = layer_idx
        self.device = device

    def init_layer(self, hf_llama_layer):
        self.wq = hf_llama_layer.self_attn.q_proj.weight.detach()
        self.wk = hf_llama_layer.self_attn.k_proj.weight.detach()
        self.wv = hf_llama_layer.self_attn.v_proj.weight.detach()
        self.wqkv = torch.cat((self.wq, self.wk, self.wv), dim=0).to(self.device, non_blocking=True)
        self.wo = hf_llama_layer.self_attn.o_proj.weight.detach().to(self.device, non_blocking=True)

        # Llama 没有 q_norm/k_norm（Qwen3 才有）；显式置 None 以便后续 wqkv() 跳过
        self.q_norm_weight = None
        self.q_norm_variance_epsilon = None
        self.k_norm_weight = None
        self.k_norm_variance_epsilon = None

        self.gate_proj = hf_llama_layer.mlp.gate_proj.weight.detach()
        self.up_proj = hf_llama_layer.mlp.up_proj.weight.detach()
        self.gate_up_proj = torch.cat((self.gate_proj, self.up_proj), dim=0).to(self.device, non_blocking=True)
        self.down_proj = hf_llama_layer.mlp.down_proj.weight.detach().to(self.device, non_blocking=True)
        self.mlp_module = None  # Llama 这里没有 MoE

        self.input_layernorm_weight = hf_llama_layer.input_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.input_layernorm_variance_epsilon = hf_llama_layer.input_layernorm.variance_epsilon

        self.post_attention_layernorm_weight = hf_llama_layer.post_attention_layernorm.weight.detach().to(self.device, non_blocking=True)
        self.post_attention_layernorm_variance_epsilon = hf_llama_layer.post_attention_layernorm.variance_epsilon

        del self.wq, self.wk, self.wv, self.gate_proj, self.up_proj


class LlamaModel(LLM):
    """A class representing the Llama / DeepSeek-R1-Distill-Llama model."""

    def __init__(
        self,
        model_name: str,
        max_length: int,
        dtype: torch.dtype,
        device_map: str,
        rope_scaling: dict = None,
    ) -> None:
        super().__init__(model_name, max_length, dtype, device_map)
        self._rope_scaling = rope_scaling

        # 用 AutoConfig + AutoModelForCausalLM；AutoModelForCausalLM 看到 architectures=
        # LlamaForCausalLM 会自动 dispatch 到 transformers.models.llama.LlamaForCausalLM。
        from .tokenizer_utils import load_tokenizer
        self.tokenizer = load_tokenizer(model_name, trust_remote_code=True)
        self.config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        self.num_layers = self.config.num_hidden_layers
        self.num_heads = self.config.num_attention_heads
        self.num_key_value_heads = self.config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.hidden_size = self.config.hidden_size
        # Llama 不在 config 里写 head_dim
        self.head_dim = getattr(self.config, 'head_dim', self.hidden_size // self.num_heads)
        self.max_position_embeddings = self.config.max_position_embeddings
        self.vocab_size = self.config.vocab_size
        self.eos_tokens = [self.config.eos_token_id]

        # transformers 5.x 把 RoPE base 迁到 rope_parameters dict 下，做 fallback
        if hasattr(self.config, 'rope_theta') and getattr(self.config, 'rope_theta') is not None:
            self.base = self.config.rope_theta
        else:
            _rp = (
                getattr(self.config, 'rope_parameters', None)
                or getattr(self.config, 'rope_scaling', None)
                or {}
            )
            self.base = _rp.get('rope_theta', 10000.0)

        self.init_model()

    def _set_cos_sin_cache(self):
        t = torch.arange(self.max_length, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        if hasattr(self, 'attention_scaling') and self.attention_scaling is not None:
            return freqs.cos() * self.attention_scaling, freqs.sin() * self.attention_scaling
        else:
            return freqs.cos(), freqs.sin()

    def init_model(self):
        if self.device_map != 'auto' and self.device_map.startswith('cuda:'):
            device_id = int(self.device_map.split(':')[1])
            torch.cuda.set_device(device_id)

        # YaRN / 自定义 rope_scaling 路径：transformers 5.x _compute_yarn_parameters
        # 用 config.rope_theta，所以光传 rope_scaling kwarg 会让 rope_theta=None 炸；
        # 这里先 AutoConfig，把 rope_parameters/rope_scaling/rope_theta 三处都改齐。
        load_config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        if self._rope_scaling is not None:
            existing_rp = (
                getattr(load_config, "rope_parameters", None)
                or getattr(load_config, "rope_scaling", None)
                or {}
            )
            existing_theta = (
                getattr(load_config, "rope_theta", None)
                or existing_rp.get("rope_theta")
                or 10000.0
            )
            new_rope = dict(self._rope_scaling)
            new_rope.setdefault("rope_theta", existing_theta)
            for attr in ("rope_parameters", "rope_scaling"):
                try:
                    setattr(load_config, attr, new_rope)
                except Exception:
                    pass
            try:
                load_config.rope_theta = new_rope["rope_theta"]
            except Exception:
                pass
            print(f"[YaRN] applied rope config: {new_rope}", flush=True)

        if self.device_map != 'auto':
            hf_llama = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=load_config,
                dtype=self.dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                device_map=self.device_map,
            )
        else:
            hf_llama = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=load_config,
                dtype=self.dtype,
                trust_remote_code=True,
                device_map='auto',
            )

        self.num_gpus = torch.cuda.device_count() if self.device_map == 'auto' else 1
        if self.device_map == 'auto' and self.num_gpus == 1:
            self.device_map = 'cuda:0'

        if self.device_map != "auto":  # single GPU
            self.layer_mapping = {str(ldx): self.device_map for ldx in range(self.num_layers)}

            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(self.device_map, non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(self.device_map, non_blocking=True)

            self.norm_weight = hf_llama.model.norm.weight.detach().to(self.device_map, non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(self.device_map, non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(self.device_map, non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for idx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(idx, device=self.device_map)
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                hf_llama.model.layers[idx] = None
        else:  # multi GPU
            self.gpu_ids = list(range(self.num_gpus))
            self.layer_interval = (self.num_layers + self.num_gpus - 1) // self.num_gpus
            self.layer_mapping = {
                str(ldx): f'cuda:{ldx // self.layer_interval}' for ldx in range(self.num_layers)
            }

            self.embed_tokens = hf_llama.model.embed_tokens.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.lm_head = hf_llama.lm_head.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)

            self.norm_weight = hf_llama.model.norm.weight.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.norm_variance_epsilon = hf_llama.model.norm.variance_epsilon

            self.position_ids = torch.arange(0, self.max_length).to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.inv_freq = hf_llama.model.rotary_emb.inv_freq.detach().to(f'cuda:{self.gpu_ids[0]}', non_blocking=True)
            self.attention_scaling = hf_llama.model.rotary_emb.attention_scaling
            self.cos_cache, self.sin_cache = self._set_cos_sin_cache()
            self.cos_sin_cache = torch.cat((self.cos_cache, self.sin_cache), dim=-1)

            self.layers = []
            for ldx, hf_llama_layer in enumerate(hf_llama.model.layers):
                llama_layer = LlamaLayer(ldx, device=self.layer_mapping[str(ldx)])
                llama_layer.init_layer(hf_llama_layer)
                self.layers.append(llama_layer)
                hf_llama.model.layers[ldx] = None

        del self.inv_freq, self.cos_cache, self.sin_cache
        gc.collect()
        torch.cuda.empty_cache()

    # -------------------------------------------------------------- KV cache --
    def init_kv_cache(self, real_input_length, valid_start, attn_config=None):
        # Locate per-model PolarANN config
        if attn_config is None:
            CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
            PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
            CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
            MODEL_NAME_JSON = self.model_name.split("/")[-1] + '.json'
            CONFIG_FILE = os.path.join(CONFIG_DIR, MODEL_NAME_JSON)
            with open(CONFIG_FILE, "r") as f:
                llama_config = json.load(f)
        else:
            llama_config = attn_config

        if self.attention_type == 'Full_Flash_Attn':
            self.kv_cache = flash_attn_cache(
                valid_start=valid_start,
                layer_num=self.num_layers,
                batch_size=self.batch_size,
                max_length=self.max_new_length + real_input_length,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layer_mapping=self.layer_mapping,
                num_gpus=self.num_gpus,
                model_size=int(re.search(r'(\d+)[B]', self.model_name).group(1)),
                enable_decode_timing=True,
            )

        elif self.attention_type == 'PolarANN':
            polar_config = llama_config.get(self.attention_type)
            if polar_config is None:
                raise ValueError(f"PolarANN configuration not found in {CONFIG_FILE}")

            # Cache 模块选择优先级：config["cache_module"] > $POLAR_CACHE_MODULE > 默认
            cache_module_name = (
                polar_config.get("cache_module")
                or os.environ.get("POLAR_CACHE_MODULE")
                or "cache_hub.polar_cache_kivi_quest_density_seperate_kernel"
            )
            try:
                import importlib
                _cache_mod = importlib.import_module(cache_module_name)
                polar_cache_cls = getattr(_cache_mod, "polar_cache")
                print(f"[PolarANN] using cache module: {cache_module_name}", flush=True)
            except Exception as e:
                raise ImportError(
                    f"Failed to import polar_cache from '{cache_module_name}': {e}"
                ) from e

            polar_cache_kwargs = {
                "valid_start": valid_start,
                "layer_num": self.num_layers,
                "batch_size": self.batch_size,
                "max_length": self.max_new_length + real_input_length,
                "num_key_value_heads": self.num_key_value_heads,
                "num_heads": self.num_heads,
                "head_dim": self.head_dim,
                "dtype": self.dtype,
                "layer_mapping": self.layer_mapping,
                "max_new_length": self.max_new_length,
                "sink_size": polar_config.get("sink_size", 4),
                "local_size": polar_config.get("local_size", 512),
                "core": polar_config["core"],
                "nprobe": polar_config["nprobe"],
                "cache_unit_size": polar_config["cache_unit_size"],
                "cache_cluster_num": polar_config["cache_cluster_num"],
                "num_gpus": self.num_gpus,
                "model_size": int(re.search(r'(\d+)[B]', self.model_name).group(1)),
                "collision_ratio": polar_config.get("collision_ratio", 0.3),
                "candidate_ratio": polar_config.get("candidate_ratio", 0.2),
                # 默认 True：offload 是 PolarANN 的主路径（4-bit code 留 GPU，原始 KV 在 CPU pinned）
                "enable_offload": polar_config.get("enable_offload", True),
                "dynamic_update_interval": polar_config.get("dynamic_update_interval", 128),
                "enable_recall": polar_config.get("enable_recall", False),
            }
            if "codebook_path" in polar_config:
                polar_cache_kwargs["codebook_path"] = polar_config["codebook_path"]

            optional_keys = [
                "final_topk",
                "full_attention_threshold",
                "debug_offload_coords",
                "enable_rerank",
                "enable_alpha_correction",
                "enable_rerank_error_ablation",
                "enable_coarse_only",
                "enable_collision_1tier",
                "enable_coarse_recall_ablation",
                "enable_rerank_recall_ablation",
                "adaptive_topk_enabled",
                "adaptive_topk_threshold",
                "adaptive_topk_min_k",
                "adaptive_topk_max_k",
                "adaptive_topk_max_k_ratio",
                "hamming_candidate_ratio",
            ]
            try:
                import inspect
                supported = set(inspect.signature(polar_cache_cls.__init__).parameters.keys())
            except (TypeError, ValueError):
                supported = set()
            for key in optional_keys:
                if key in polar_config and key in supported:
                    polar_cache_kwargs[key] = polar_config[key]

            self.kv_cache = polar_cache_cls(**polar_cache_kwargs)

        elif self.attention_type == 'ExactTopK':
            exact_config = llama_config.get(self.attention_type, {})
            self.kv_cache = ExactTopKCache(
                valid_start=valid_start,
                layer_num=self.num_layers,
                batch_size=self.batch_size,
                max_length=self.max_new_length + real_input_length,
                num_key_value_heads=self.num_key_value_heads,
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                dtype=self.dtype,
                layer_mapping=self.layer_mapping,
                num_gpus=self.num_gpus,
                model_size=int(re.search(r'(\d+)[B]', self.model_name).group(1)),
                candidate_ratio=exact_config.get("candidate_ratio", 0.2),
                recent_size=exact_config.get("recent_size", 64),
                sink_size=exact_config.get("sink_size", 4),
                enable_decode_timing=True,
            )
        else:
            raise ValueError(
                f"Unsupported attention type: {self.attention_type}. "
                "Density release supports only Full_Flash_Attn / PolarANN / ExactTopK."
            )

    # ---------------------------------------------------------------- forward --
    def move(self):
        torch.cuda.empty_cache()
        if self.attention_type == 'Full_Flash_Attn':
            if hasattr(self.kv_cache, 'move_gpu'):
                self.kv_cache.move_gpu()
        torch.cuda.empty_cache()

    def word_embedding(self, inputs_id):
        return F.embedding(inputs_id, self.embed_tokens)

    def lm(self, hidden_states):
        return F.linear(hidden_states, self.lm_head).float()

    def wqkv(self, hidden_states, layer):
        # Llama-style 切分：q=hidden, k/v=hidden//num_kv_groups
        qkv = F.linear(hidden_states, layer.wqkv)
        q_dim = self.hidden_size
        kv_dim = self.hidden_size // self.num_key_value_groups
        query_states, key_states, value_states = qkv.split([q_dim, kv_dim, kv_dim], dim=-1)
        return query_states, key_states, value_states

    def wo(self, hidden_states, layer, bsz, seq_len, dim):
        hidden_states = hidden_states.reshape(bsz, seq_len, dim)
        return F.linear(hidden_states, layer.wo)

    def prefill_attention(self, query_states, key_states, value_states):
        if self.attention_type == 'Full_Flash_Attn':
            return prefill_full_flash_attn(query_states, key_states, value_states, causal=True)
        if self.attention_type == 'PolarANN':
            return polar_prefill_attn(query_states, key_states, value_states, causal=True)
        if self.attention_type == 'ExactTopK':
            return exact_topk_prefill_attn(query_states, key_states, value_states, causal=True)
        raise ValueError(
            f"Unsupported attention type: {self.attention_type}. "
            "Density release supports only Full_Flash_Attn / PolarANN / ExactTopK."
        )

    def decode_attention(self, query_states, key_states, value_states, layer_idx):
        if self.attention_type == 'Full_Flash_Attn':
            return decode_full_flash_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        if self.attention_type == 'PolarANN':
            return polar_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        if self.attention_type == 'ExactTopK':
            return exact_topk_decode_attn(query_states, key_states, value_states, layer_idx, self.kv_cache)
        raise ValueError(
            f"Unsupported attention type: {self.attention_type}. "
            "Density release supports only Full_Flash_Attn / PolarANN / ExactTopK."
        )

    def mlp(self, hidden_states, layer):
        hidden_states = F.linear(hidden_states, layer.gate_up_proj)
        dim = hidden_states.shape[-1] // 2
        hidden_shape = hidden_states.shape[:-1] + (dim,)
        out = torch.empty(hidden_shape, dtype=hidden_states.dtype, device=hidden_states.device)
        flashinfer.activation.silu_and_mul(hidden_states, out)
        return F.linear(out, layer.down_proj)

    def parameter_move(self, hidden_states, ldx):
        next_device = self.layer_mapping[str(ldx + 1)] if str(ldx + 1) in self.layer_mapping else self.layer_mapping[str(0)]
        torch.cuda.set_device(next_device)
        hidden_states = hidden_states.to(next_device)
        self.position_ids = self.position_ids.to(next_device)
        self.cos_sin_cache = self.cos_sin_cache.to(next_device)

        # Per-cache buffer migration (decode 阶段才需要)
        if hidden_states.shape[1] == 1:
            kv = self.kv_cache
            if self.attention_type in ('Full_Flash_Attn', 'ExactTopK'):
                if hasattr(kv, 'batch_indices'):
                    kv.batch_indices = kv.batch_indices.to(next_device)
                if hasattr(kv, 'valid_length'):
                    kv.valid_length = kv.valid_length.to(next_device)
            elif self.attention_type == 'PolarANN':
                if hasattr(kv, 'execution_buffer_keys'):
                    kv.execution_buffer_keys = kv.execution_buffer_keys.to(next_device)
                if hasattr(kv, 'execution_buffer_values'):
                    kv.execution_buffer_values = kv.execution_buffer_values.to(next_device)
        return hidden_states

    def layernorm(self, hidden_states, epsilon, weight):
        bsz, seq_len, dim = hidden_states.shape
        if hidden_states.dtype == torch.float32:
            variance = hidden_states.pow(2).mean(-1, keepdim=True)
            hidden_states = hidden_states * torch.rsqrt(variance + epsilon)
            hidden_states = hidden_states * weight
            return hidden_states
        hidden_states = hidden_states.reshape(bsz * seq_len, dim)
        hidden_states = flashinfer.rmsnorm(hidden_states, weight, epsilon)
        return hidden_states.reshape(bsz, seq_len, dim)

    def apply_rotary_pos_emb(self, query_states, key_states, position_ids):
        bsz, seq_len, hidden_dim = query_states.shape
        _, _, kv_dim = key_states.shape

        if query_states.dtype == torch.float32:
            q = query_states.view(bsz, seq_len, self.num_heads, self.head_dim)
            k = key_states.view(bsz, seq_len, self.num_key_value_heads, self.head_dim)
            positions = position_ids[0]

            if hasattr(self, 'cos_cache') and hasattr(self, 'sin_cache'):
                cos = self.cos_cache[positions].to(query_states.dtype)
                sin = self.sin_cache[positions].to(query_states.dtype)
            else:
                cos_sin = self.cos_sin_cache[positions].to(query_states.dtype)
                half_dim = cos_sin.shape[-1] // 2
                cos = cos_sin[..., :half_dim]
                sin = cos_sin[..., half_dim:]

            rope_dim = cos.shape[-1]
            q_rot, q_pass = q[..., :rope_dim], q[..., rope_dim:]
            k_rot, k_pass = k[..., :rope_dim], k[..., rope_dim:]
            cos = cos.unsqueeze(0).unsqueeze(2)
            sin = sin.unsqueeze(0).unsqueeze(2)

            def rotate_half(x):
                x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
                return torch.cat((-x2, x1), dim=-1)

            q_rot_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
            k_rot_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
            q_embed = torch.cat([q_rot_embed, q_pass], dim=-1)
            k_embed = torch.cat([k_rot_embed, k_pass], dim=-1)
            return q_embed.view(bsz, seq_len, hidden_dim), k_embed.view(bsz, seq_len, kv_dim)

        # bf16/fp16 走 flashinfer
        query_states = query_states.contiguous().view(-1, hidden_dim)
        key_states = key_states.contiguous().view(-1, kv_dim)
        positions = position_ids.reshape(-1).to(torch.int32)
        flashinfer.rope.apply_rope_with_cos_sin_cache_inplace(
            positions, query_states, key_states, self.head_dim, self.cos_sin_cache, True
        )
        query_states = query_states.view(bsz, -1, hidden_dim)
        key_states = key_states.view(bsz, -1, kv_dim)
        return query_states, key_states

    def position_embedd(self, query_states, key_states):
        bsz, seq_len, _ = key_states.shape
        position_ids = self.position_ids[self.kv_cache.context:self.kv_cache.context + seq_len].unsqueeze(0).repeat(bsz, 1)
        return self.apply_rotary_pos_emb(query_states, key_states, position_ids)
