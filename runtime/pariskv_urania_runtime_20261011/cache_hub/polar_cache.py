
"""
ParisKV

碰撞次数+2, candidate_ratio固定
打印碰撞阈值和

rabitq_4bit version
Polar ANN-based KV Cache for RetroInfer
使用RxΩ极坐标分解的data-independent量化方法替代K-means聚类
1. Prefill: 用encode()替代segment_k_means()
2. Decode: 用collision_based_topk()替代centroid检索
"""
import math
from pathlib import Path
from typing import Any
import torch
import torch.cuda.nvtx as nvtx
import numpy as np
import json
import os
import sys
import time
from datetime import datetime
import gc
# from retroinfer_kernels import ThreadPool, WaveBufferCPU
# from retroinfer_kernels import gather_copy_and_concat, gather_copy_and_scatter
try:
    from fast_hadamard_transform import hadamard_transform  # type: ignore
except ImportError:
    from cache_hub.hadamard_fallback import hadamard_transform
from cache_hub.rerank.rerank import partial_rerank_cuda

USE_RABITQ_RERANK = True    # True: RaBitQ 近似精排, False: 精确内积精排 (临时关闭验证 coarse recall)
DEFAULT_SRHT_ROUNDS = 1



# 导入基类
try:
    from .cache import KV_Cache
except ImportError:
    from cache import KV_Cache
from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat
# Steady Zone 动态更新的间隔已移至实例变量 self.dynamic_update_interval


# ==================== 内嵌 MultiDimBlockQuantizer 类 ====================
class MultiDimBlockQuantizer:
    """
    多维块结构化量化器(RxΩ分解)
    1. 将d维向量分成B个m维块
    2. 每个块分解为 radius x direction(极坐标形式)
    3. radius用解析Lloyd-Max量化,direction用球面CVT量化
    4. 支持ADC(Asymmetric Distance Computation)查询
    """
    
    def __init__(self, d, m, K_r, K_omega, codebook_path, seed=42, device=None):
        """
        初始化量化器(从预生成的码本加载到GPU)
        """
        assert d % m == 0, f"总维度{d}必须能被块维度{m}整除"
        if not torch.cuda.is_available():
            raise RuntimeError("GPU不可用!此版本只支持GPU模式")
    
        self.d = d
        self.m = m
        self.B = d // m
        self.K_r = K_r
        self.K_omega = K_omega
        self.K_total = K_r * K_omega
        self.seed = seed
                
        if device is None:
            device = torch.device('cuda:0')
        elif isinstance(device, str):
            device = torch.device(device)
        
        # 预生成 SRHT 对角符号矩阵 (GPU, bfloat16)
        # 保存当前随机状态，避免影响外部采样
        rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
        
        torch.manual_seed(self.seed)  # 使用固定种子确保 SRHT 矩阵可重复
        self.srht_diagonal_signs = []
        for _ in range(DEFAULT_SRHT_ROUNDS):
            signs = torch.randint(0, 2, (self.d,), device=device, dtype=torch.bfloat16) * 2 - 1
            self.srht_diagonal_signs.append(signs)
        
        # 恢复之前的随机状态（不影响外部 token 采样）
        torch.set_rng_state(rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state)
        
        # 加载码本到 GPU
        try:
            with open(codebook_path, 'r') as f:
                codebook = json.load(f)
            
            config = codebook['config']
            if (config['d'] != self.d or config['m'] != self.m or 
                config['K_r'] != self.K_r or config['K_omega'] != self.K_omega):
                raise ValueError(
                    f"码本参数不匹配!\n"
                    f"  期望: d={self.d}, m={self.m}, K_r={self.K_r}, K_omega={self.K_omega}\n"
                    f"  实际: d={config['d']}, m={config['m']}, K_r={config['K_r']}, K_omega={config['K_omega']}"
                )
            
            if "angular_signs" in codebook:
                angular_signs = torch.tensor(codebook["angular_signs"], dtype=torch.int8, device=device)
                angular_signs = torch.where(angular_signs >= 0,
                    torch.ones_like(angular_signs), -torch.ones_like(angular_signs)).to(torch.int8)
                # print("量化器初始化完成(K_r=1 优化:从符号码本加载)")
            else:
                centers = torch.tensor(codebook['angular_centers'], dtype=torch.bfloat16, device=device)
                angular_signs = torch.where(centers >= 0, 
                    torch.ones_like(centers), -torch.ones_like(centers)).to(torch.int8)
                # print("量化器初始化完成(K_r=1 优化:从 float 码本转换)")

            # 保存 bfloat16 版本（值只有 ±1）
            self.V_omega_gpu = angular_signs.to(torch.bfloat16)
            
        except FileNotFoundError:
            raise FileNotFoundError(f"码本文件不存在: {codebook_path}")
        except (KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"码本文件格式错误: {e}")


# ==================== 结束 MultiDimBlockQuantizer 类定义 ====================



class polar_cache(KV_Cache):

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
        max_new_length: int,
        sink_size: int,
        local_size: int,
        core: int,
        nprobe: int,
        cache_unit_size: int,
        cache_cluster_num: int,
        num_gpus: int,
        model_size: int,
        codebook_path: str = '/data/qwang/q/thalia/kvcache/ANNCache/turboquant/codebooks/codebook_d128_m8_Kr1_Kw256_rabitq_sign.json',
        collision_ratio: float = 0.3,
        candidate_ratio: float = 0.05,
        final_topk: int = 100,  # 精排阶段：从候选集中选 top-K 个 key
        dynamic_update_interval: int = 512,  # 每累积多少个 tokens 触发更新
        full_attention_threshold: int = 2000,  # prefill + decode < threshold 时使用全量 attention（与历史可复现版本对齐）
        enable_offload: bool = True,  # 消融实验：关闭 offload，key 都在 GPU
        debug_offload_coords: bool = False,  # 调试：打印 offload 坐标/写入区间（会产生大量日志）
        enable_rerank: bool = False,  # 是否启用 rerank（=exact 或者rabitq）（提升 recall，略微增加延迟
        enable_recall: bool = False,  # 是否启用 Recall@K 计算（用于调试，会增加延迟）
        # ===== 消融实验参数 =====
        enable_alpha_correction: bool = False,  # 是否在 block_weight 中除以 alpha（消融实验）
        enable_rerank_error_ablation: bool = False,  # 消融实验：开启对比精确内积（默认关闭）
        # ===== 粗排漂移消融参数 =====
        enable_coarse_only: bool = False,  # True: 禁用 RaBitQ rerank，只用碰撞结果
        enable_collision_1tier: bool = False,  # True: 1-tier 碰撞（和 SuCo 对齐），False: 6-tier
        enable_coarse_recall_ablation: bool = False,  # True: 启用粗排 Recall 消融实验
        enable_rerank_recall_ablation: bool = False,  # True: 启用精排后 Recall 消融实验
        # ===== 自适应 Top-k 参数 =====
        adaptive_topk_enabled: bool = False,  # True: 根据 attention mass 自适应确定 k
        adaptive_topk_threshold: float = 0.9,  # 累积 attention mass 阈值
        adaptive_topk_min_k: int = 100,  # per-head 最小 k
        adaptive_topk_max_k: int = 400,  # per-head 最大 k，默认等于 final_topk
    ) -> None:
        
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
         
        self.valid_start = valid_start  # 用于 prefill 时确定有效数据起始位置
        self.sink_size = sink_size  # Sink tokens 数量 = 4
        self.local_size = local_size   # Local window 大小 (通过配置文件设置,推荐512)
        self.steady_size = self.sink_size + self.local_size  # Steady Zone 总大小 = 4 + 512 = 516
        # 动态更新间隔：每累积多少个新 tokens 触发一次 local zone 更新
        self.dynamic_update_interval = dynamic_update_interval if dynamic_update_interval is not None else self.local_size
        self.group_size = self.num_heads // self.kv_head # 每个kv head对应的query head的group数量=32/8=4
        self.batch_groups = self.batch_size * self.kv_head # batch_size × kv_head = 1 × 8 = 8
        self.dtype = dtype
        # 保存 max_new_length,用于计算 retrieval zone capacity
        self.max_new_length = max_new_length
        # qwen.py 传入: max_length = real_input_length + max_new_length
        self.input_length = self.max_length - max_new_length  # 实际 prefill 长度 (通过反推得到)
        self.FULL_ATTENTION_THRESHOLD = full_attention_threshold  # prefill + decode < threshold 时全量计算
        self.current_seq_len = [0] * layer_num  # 每层独立跟踪，prefill 后会初始化为 input_length        
        self.codebook_path = codebook_path
        self.collision_ratio = collision_ratio
        self.candidate_ratio = candidate_ratio
        self.final_topk = final_topk  # 精排阶段的 top-K
        self.enable_offload = enable_offload  # 兼容性参数
        self.debug_offload_coords = debug_offload_coords
        self.enable_rerank = enable_rerank  # 是否启用精确内积 rerank
        self.enable_recall = enable_recall  # 是否启用 Recall@K 计算
        # 消融实验参数
        self.enable_alpha_correction = enable_alpha_correction  # 是否在 block_weight 中除以 alpha
        self.enable_rerank_error_ablation = enable_rerank_error_ablation  # 是否启用 rerank 消融实验
        # 粗排漂移消融参数
        self.enable_coarse_only = enable_coarse_only  # 禁用 RaBitQ rerank，只用碰撞
        self.enable_collision_1tier = enable_collision_1tier  # 1-tier vs 6-tier 碰撞
        self.enable_coarse_recall_ablation = enable_coarse_recall_ablation  # 粗排 Recall 消融
        self.enable_rerank_recall_ablation = enable_rerank_recall_ablation  # 精排后 Recall 消融

        self._allayer_recall_check = os.environ.get("POLAR_ALLAYER_RECALL", "0") not in ("0", "false", "")
        self._allayer_recall_log_every = int(os.environ.get("POLAR_ALLAYER_RECALL_LOG_EVERY", "100"))
        self._allayer_recall_k = int(os.environ.get("POLAR_ALLAYER_RECALL_K", "500"))
        self._allayer_recall_step = 0
        self._allayer_recall_accum = {}  # {tag: [hit_sum, total_sum]}

        # 自适应 Top-k 参数
        self.adaptive_topk_enabled = adaptive_topk_enabled
        self.adaptive_topk_threshold = adaptive_topk_threshold
        self.adaptive_topk_min_k = adaptive_topk_min_k
        self.adaptive_topk_max_k = adaptive_topk_max_k if adaptive_topk_max_k is not None else final_topk
        try:
            with open(codebook_path, 'r') as f:
                codebook_config = json.load(f)['config']
            self.polar_m = codebook_config['m']
            self.polar_K_r = codebook_config['K_r']
            self.polar_K_omega = codebook_config['K_omega']
            self.polar_B = head_dim // self.polar_m  # 块数
            # print(f"  ✓ 从 codebook 自动读取参数: m={self.polar_m}, K_r={self.polar_K_r}, K_omega={self.polar_K_omega}")
        except (FileNotFoundError, KeyError, json.JSONDecodeError) as e:
            raise ValueError(f"无法从 codebook 读取参数: {e}")
        # 初始化实际使用的 retrieval zone 长度
        self._actual_retrieval_length = None  # 将在 prefill 时设置
        # 添加decode步数计数器,用于控制Recall计算频率
        self._decode_step_counter = 0
        # 上次 buffer reset 时的 counter 值（长 prefill offload 用）
        self._buffer_base_counter = 0
        # 延迟 base 更新标志（确保 update 后的第一个 token 写到 offset 0）
        self._pending_buffer_reset = False
        

        # 初始化Polar ANN量化器(共享一个,码本数据无关)
        import time as _time
        _t0 = _time.time()
        print(f"[Init] polar_cache.__init__ start (input_length={self.input_length}, max_new={max_new_length}, layers={layer_num}, offload={self.enable_offload})", flush=True)
        self.device = torch.device(list[Any](layer_mapping.values())[0] if layer_mapping else 'cuda:0')
        self.quantizer = MultiDimBlockQuantizer(
            d=head_dim,
            m=self.polar_m,
            K_r=self.polar_K_r,
            K_omega=self.polar_K_omega,
            seed=42,
            codebook_path=codebook_path,
            device=self.device
        )

        # ============ 符号码本（用于 coarse retrieval）============
        # V_omega_gpu 是 ±1 的符号码本 [K_omega, m]
        self._V_omega_gpu_T = self.quantizer.V_omega_gpu.T.contiguous()  # [m, K_omega] 预转置
        # 统一 KV Cache 架构（按生成顺序存储所有 tokens）: 全量容量 = prefill + decode 的总长度
        self.unified_kv_capacity = self.input_length + self.max_new_length
        # ============ 4-bit 幅值量化参数（1-bit符号 + 3-bit幅值 = 8 levels）============
        # 由 Lloyd-Max 从球面均匀分布采样得到，对任何数据集通用 生成脚本: python run/generate_magnitude_levels.py --levels 8
        self._mag_thresholds = torch.tensor([0.0843, 0.1698, 0.2578, 0.3499, 0.4487, 0.5585, 0.6901],
            device=self.device, dtype=torch.bfloat16).contiguous()  # [7] 决策阈值（bf16，~2-3位小数精度）
        self._mag_centers = torch.tensor([0.0420, 0.1266, 0.2130, 0.3025, 0.3973, 0.5001, 0.6169, 0.7633],
            device=self.device, dtype=torch.bfloat16).contiguous()  # [8] 重构中心 a[t]
        # ============ 预缓存 shifts 张量（避免每次调用时创建）============
        # bitpack 符号用：[1, 2, 4, 8, 16, 32, 64, 128]
        self._bitpack_shifts = (1 << torch.arange(self.polar_m, device=self.device, dtype=torch.uint8))
        print(f"[Init] quantizer + codebook ready ({_time.time()-_t0:.1f}s)", flush=True)

        
        if self.enable_offload:
            # 【Offload 模式】GPU 只存 Sink + Local + Update Buffer（小容量，省显存）
            #  Retrieval Zone 的 KV offload 到 CPU（大容量，pinned memory）
            self.gpu_kv_capacity = self.sink_size + self.local_size + self.dynamic_update_interval
            print(f"[Init] Offload Mode: ENABLED (gpu_kv={self.gpu_kv_capacity}, cpu_kv={self.unified_kv_capacity - self.sink_size - self.local_size})", flush=True)
            
            # 创建专用 copy stream，实现 GPU→CPU 拷贝与 Attention 计算并行
            self._copy_stream = torch.cuda.Stream(device=self.device)
            # - main_event: 标记源数据准备好（default stream → copy stream 的依赖）
            # - copy_event: 标记该层 D2H 拷贝完成（CPU 数据可安全读取）
            self._main_events = [torch.cuda.Event() for _ in range(layer_num)]
            self._copy_events = [torch.cuda.Event() for _ in range(layer_num)]
            # 让 copy_event 初始处于“已完成”状态，避免第一次 synchronize() 卡住
            with torch.cuda.device(self.device):
                for e in self._copy_events:
                    e.record()

            # 保存仍在 D2H 拷贝中的源 tensor 引用，避免 allocator 复用导致潜在数据竞争
            # 每层一个 (keys, values) tuple；拷贝完成后会清理
            self._inflight_offload_buffers = [None for _ in range(layer_num)]
            self._layer_offload_inflight = [False for _ in range(layer_num)]
            # CPU 只存 Retrieval Zone，不含 Sink + Local
            self.cpu_kv_capacity = max(0, self.unified_kv_capacity - self.sink_size - self.local_size)
            # print(f"  ✓ GPU KV Capacity: {self.gpu_kv_capacity} tokens (sink={self.sink_size} + local={self.local_size} + update_buffer={self.dynamic_update_interval})")
            # print(f"  ✓ CPU KV Capacity: {self.cpu_kv_capacity} tokens (pinned memory for retrieval zone)")
            
            # GPU 端：小容量，只存 Sink + Local + Update Buffer
            print(f"[Init] allocating GPU KV ({layer_num} layers × {self.gpu_kv_capacity} tokens) ...", flush=True)
            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
            print(f"[Init] GPU KV done ({_time.time()-_t0:.1f}s)", flush=True)
            
            # CPU 端：只存 Retrieval Zone 的 KV（pinned memory 加速传输）
            # 分配前先清理内存碎片
            gc.collect()
            torch.cuda.empty_cache()
            
            self.unified_keys_cpu = []
            self.unified_values_cpu = []
            
            print(f"[Init] allocating CPU pinned KV ({layer_num} layers × {self.cpu_kv_capacity} tokens) ...", flush=True)
            for ldx in range(layer_num):
                # 每层分配前清理，帮助整理内存碎片
                if ldx % 7 == 0:
                    gc.collect()
                self.unified_keys_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))
                self.unified_values_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))
            print(f"[Init] CPU pinned KV done ({_time.time()-_t0:.1f}s)", flush=True)
                
            # 记录 CPU 上已存储的有效范围
            self.cpu_valid_end = [0] * layer_num
            # 预分配 TopK gather 结果的 GPU buffer（避免每次 gather 后分配新内存）
            # 使用 final_topk 参数作为 buffer 大小
            max_topk_size = self.final_topk
            # print(f"  ✓ TopK buffer size: {max_topk_size} tokens (final_topk={self.final_topk})")
            # print(f"enter _topk_keys_buffer", flush=True)
            self._topk_keys_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)
            self._topk_values_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)
            
            
            print(f"[Init] allocating offload staging buffers ...", flush=True)
            # 预分配 Offload staging buffers（避免 clone + 避免跨层复用导致数据竞争）
            # 每层一个 staging buffer：否则在 _update_polar_index 的 layer 循环里复用同一个 buffer，
            # 会和异步 copy_stream 产生 race（上一层还在 D2H，下一层就覆盖 src）。
            self._offload_keys_buffer = [
                torch.empty(
                    (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                    dtype=dtype, device=self.device
                )
                for _ in range(layer_num)
            ]
            self._offload_values_buffer = [
                torch.empty(
                    (batch_size, self.kv_head, self.dynamic_update_interval, head_dim),
                    dtype=dtype, device=self.device
                )
                for _ in range(layer_num)
            ]
            # 预分配 CPU gather buffer（避免每次 gather 分配新内存）
            # print(f"enter _topk_keys_cpu_buffer", flush=True)
            self._topk_keys_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True  # pin_memory 加速 CPU→GPU 传输
            )
            self._topk_values_cpu_buffer = torch.empty(
                (batch_size, self.kv_head, max_topk_size, head_dim),
                dtype=dtype, device='cpu', pin_memory=True
            )
            
        else:
            # ===========================================
            # 【非 Offload 模式】GPU 存全量
            self.gpu_kv_capacity = self.unified_kv_capacity
            print(f"[Init] No-Offload Mode: GPU full capacity = {self.unified_kv_capacity} tokens", flush=True)
            print(f"[Init] allocating GPU KV ({layer_num} layers × {self.unified_kv_capacity} tokens) ...", flush=True)
            self.unified_keys_gpu = []
            self.unified_values_gpu = []
            
            for ldx in range(layer_num):
                self.unified_keys_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim), dtype=dtype,
                        device=self.device).contiguous())
                self.unified_values_gpu.append(
                    torch.zeros((batch_size, self.kv_head, self.unified_kv_capacity, head_dim), dtype=dtype,
                        device=self.device).contiguous())
                if ldx % 10 == 0:
                    print(f"[Init]   GPU KV layer {ldx}/{layer_num} ({_time.time()-_t0:.1f}s)", flush=True)
            print(f"[Init] GPU KV done ({_time.time()-_t0:.1f}s)", flush=True)
            
            self.unified_keys_cpu = None
            self.unified_values_cpu = None
            self.cpu_valid_end = None
            self._topk_keys_buffer = None
            self._topk_values_buffer = None
        
        # 注意：在统一架构下，retrieval_zone_capacity 指的是 unified_kv_capacity
        self.retrieval_zone_capacity = self.unified_kv_capacity
        print(f"[Init] allocating codebook/weight/4bit ({layer_num} layers × {self.retrieval_zone_capacity} tokens) ...", flush=True)
        # RaBitQ rerank 预计算权重（已融合 key_norm）：
        # weight_{k,b} = (r_{k,b} / alpha_{k,b}) * ||k||  shape: [batch, kv_head, B, N] - 与 codebook 对齐
        self.key_block_weight_gpu = []
        for ldx in range(layer_num):
            self.key_block_weight_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),
                    dtype=torch.bfloat16,device=self.device,).contiguous())
        
        # ============ 4-bit packed 存储（1-bit sign + 3-bit magnitude）============
        # 每个 block（m=8 维）4-bit/维 × m维 = 32 bits = 4 bytes
        # shape: [batch, kv_head, B, N, bytes_per_block]
        self._4bit_bytes_per_block = self.polar_m // 2  # m=8 -> 4 bytes
        self.key_4bit_packed_gpu = []
        for ldx in range(layer_num):
            self.key_4bit_packed_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity, self._4bit_bytes_per_block),
                    dtype=torch.uint8,device=self.device).contiguous())

        # Codebook存储(GPU): [layer][batch, kv_head, B, kv_len]
        self.codebook_gpu = [torch.zeros((batch_size, self.kv_head, self.polar_B, self.retrieval_zone_capacity),dtype=torch.uint8, device=self.device) for _ in range(layer_num)]
        # cluster_key_counts: [layer][batch, kv_head, B, K_omega] - 每个cluster的key数量
        self.cluster_key_counts_gpu = [torch.zeros((batch_size, self.kv_head, self.polar_B, self.polar_K_omega), dtype=torch.int32, device=self.device) for _ in range(layer_num)]  
        self.context = 0
        self._preallocated_buffers = {}
        self._cache_cnt_buffer = None
        self._cluster_range_cache = None
        self._hadamard_scale = 1.0 / np.sqrt(self.head_dim)
        print(f"[Init] codebook/weight/4bit done ({_time.time()-_t0:.1f}s)", flush=True)
 
        
        
        
        # 设置 PyTorch CPU 线程数（只在第一次初始化时设置）
        if not hasattr(polar_cache, '_threads_initialized'):
            torch.set_num_threads(1)          # 8 关闭多线程，单线程运行
            try:
                torch.set_num_interop_threads(1)  # inter-op 也用 1
            except RuntimeError:
                pass  # 已经设置过或并行工作已启动，忽略
            polar_cache._threads_initialized = True
            # print(f"  [Thread Config] intra-op={torch.get_num_threads()}, inter-op={torch.get_num_interop_threads()}")
        
        # 预加载 C++ gather extension（避免 decode 循环内重复检查）
        print(f"[Init] loading CUDA extensions ...", flush=True)
        try:
            from .gather import load_gather_ext
            self._gather_ext = load_gather_ext()
            print(f"[Init]   gather ext loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   gather ext FAILED: {e}", flush=True)
            self._gather_ext = None
        
        # 预加载 Radix TopK CUDA extension (O(n) vs O(n log k))
        try:
            from .topk import load_radix_topk_ext
            self._radix_topk_ext = load_radix_topk_ext()
            print(f"[Init]   radix_topk loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   radix_topk FAILED: {e}", flush=True)
            self._radix_topk_ext = None
        
        # 预加载 Collision CUDA extension (update_cache_cnt)
        try:
            from .collision.collision import update_cache_cnt_cuda
            self._update_cache_cnt_cuda = update_cache_cnt_cuda
            print(f"[Init]   collision loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"Collision CUDA kernel compile failed: {e}")
        
        # 预加载融合版本的 Collision CUDA kernel（预编译，避免首次调用时卡住）
        try:
            from .collision_fused.collison_interface import update_cache_cnt_cuda_interface, load_kernel_module as load_collision_module
            self._update_cache_cnt_fused = update_cache_cnt_cuda_interface
            load_collision_module("collision.cu", "update_cache_cnt")
            print(f"[Init]   collision_fused loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"Collision Fused CUDA kernel compile failed: {e}")
        
        # 预加载 UVA H2D Gather kernel（利用 UVA 直接从 pinned CPU gather 到 GPU）
        try:
            from .gather_trans.trans_h2d import h2d_gather_kv
            self._h2d_gather_kv = h2d_gather_kv
            print(f"[Init]   h2d_gather loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"H2D Gather CUDA kernel compile failed: {e}")
        
        # 预加载 Rerank CUDA extension (partial_rerank)
        try:
            from .rerank.rerank import load_kernel_module
            self._rerank_kernel = load_kernel_module("rerank.cu", "rerank")
            print(f"[Init]   rerank loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"Rerank CUDA kernel compile failed: {e}")
        
        # __init__ 完成，清理初始化过程中的临时对象
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Init] polar_cache.__init__ DONE ({_time.time()-_t0:.1f}s)", flush=True)


    # 处理单层prefill阶段（单样本）
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill 阶段入口：
        - 兼容旧逻辑: bsz==1 时按 batch_idx 单样本处理
        - lockstep multi-batch: bsz>1 时一次性处理整个 batch(要求各样本有效长度一致)
        """
        bsz = key_states.shape[0]
        if bsz == 1:
            return self._prefill_single_sample(key_states, value_states, layer_idx, batch_idx)
        # 对 bsz>1：只在 batch_idx==0 时执行一次，避免外部仍按 batch_idx 循环时重复工作
        if batch_idx != 0:
            # 保持返回 shape 兼容：去掉 padding（按 batch_idx=0 的 valid_start 规则）
            vs = self.valid_start[0] if isinstance(self.valid_start, (list, tuple)) else int(self.valid_start)
            return key_states[:, vs:, :, :], value_states[:, vs:, :, :]
        return self._prefill_lockstep_batch(key_states, value_states, layer_idx)





    def _prefill_lockstep_batch(self, key_states, value_states, layer_idx: int):
        """
        lockstep multi-batch prefill：
        - 假设所有 batch 的有效 token 长度一致（典型 benchmark/padding 场景）
        - 当前实现要求 valid_start 在 batch 内一致（否则需要 per-batch region，改动更大）
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert bsz == self.batch_size, f"bsz({bsz}) must equal cache batch_size({self.batch_size}) for lockstep mode."
        assert seq_len <= self.input_length

        # valid_start：要求 batch 内一致（支持 list/tuple/numpy array）
        if hasattr(self.valid_start, '__len__') and len(self.valid_start) > 1:
            assert len(self.valid_start) >= bsz
            vs0 = int(self.valid_start[0])
            for b in range(1, bsz):
                assert int(self.valid_start[b]) == vs0, (
                    f"lockstep multi-batch currently requires same valid_start across batch. "
                    f"valid_start[0]={vs0}, valid_start[{b}]={self.valid_start[b]}"
                )
            valid_start = vs0
        else:
            valid_start = int(self.valid_start) if not hasattr(self.valid_start, '__len__') else int(self.valid_start[0])

        # 区域划分（与单样本一致，只是对整个 batch 一次性 copy/encode）
        sink_start = valid_start
        sink_end = valid_start + self.sink_size
        retrieval_start = sink_end
        retrieval_end_raw = seq_len - self.local_size
        retrieval_end = max(retrieval_start, retrieval_end_raw)
        local_start = retrieval_end
        local_end = seq_len
        retrieval_length = retrieval_end - retrieval_start
        valid_length = retrieval_length



        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length
            
            # 重新分配精确大小的 buffer（避免每次 slice）
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

        # 短 prefill：全部留在 GPU（不触发 retrieval/offload）
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                # print(f"[Short Prefill] prefill_length={seq_len} <= steady_size={self.steady_size}, no retrieval zone")
                self._short_prefill_printed = True
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            k_gpu = key_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu
            self.current_seq_len[layer_idx] = actual_len
            if layer_idx == self.layer_num - 1:
                # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # Step1: 分区存储 KV
        local_len = local_end - local_start
        if self.enable_offload:
            # Sink → GPU[:sink_size]
            sink_k = key_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            sink_v = value_states[:, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_k
            self.unified_values_gpu[layer_idx][:bsz, :, :self.sink_size, :] = sink_v

            # Local → GPU[sink_size : sink_size+local_len]
            local_k = key_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            local_v = value_states[:, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_k
            self.unified_values_gpu[layer_idx][:bsz, :, self.sink_size:self.sink_size+local_len, :] = local_v

            # Retrieval → CPU（异步 D2H）
            temp_retrieval_keys = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            temp_retrieval_values = value_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            self._main_events[layer_idx].record()
            
            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()
                self.unified_keys_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][:bsz, :, :retrieval_length, :].copy_(temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()
            self.cpu_valid_end[layer_idx] = retrieval_length

        else:
            # 非 Offload：全部存 GPU
            actual_len = seq_len - valid_start
            k_gpu = key_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            v_gpu = value_states[:, valid_start:seq_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_keys_gpu[layer_idx][:bsz, :, :actual_len, :] = k_gpu
            self.unified_values_gpu[layer_idx][:bsz, :, :actual_len, :] = v_gpu

        self.current_seq_len[layer_idx] = seq_len - valid_start

        # Step2: encode retrieval zone（一次性 batch_encode）
        keys_to_encode = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).contiguous()
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)
        
        if layer_idx == 0:
            allocated = torch.cuda.memory_allocated(self.device) / 1024**3
            reserved = torch.cuda.memory_reserved(self.device) / 1024**3
            # print(colored(f"[MEM DEBUG] batch_encode 完成后: allocated={allocated:.2f}GB, reserved={reserved:.2f}GB", 'yellow'))
        
        self.codebook_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = codebook_batch
        if USE_RABITQ_RERANK:
            self.key_block_weight_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end] = block_weight_batch
            self.key_4bit_packed_gpu[layer_idx][:bsz, :, :, retrieval_start:retrieval_end, :] = packed4b_batch

        # Step3: cluster counts（一次性处理整个 batch）
        self.cluster_key_counts_gpu[layer_idx][:bsz] = self._count_cluster_keys(codebook_batch, codebook_batch.device)

        if layer_idx == self.layer_num - 1:
            # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
            self.context += self.current_seq_len[layer_idx]
            if self.enable_offload:
                self._sync_offload_copies()
            # 确保所有 GPU 操作完成（避免异步错误延迟报告）
        else:
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]





    def _prefill_single_sample(self, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill阶段:使用Polar ANN编码keys(完整GPU版本)
        主要改动:
        1. 不用K-means,用encode()量化keys(在GPU上进行)
        2. 构建倒排索引而非centroids(在GPU上进行)
        3. 所有数据保持在GPU上,无 CPU offload
        """
        bsz, seq_len, group_num, head_dim = key_states.shape        
        assert bsz == 1, "Multi-batch prefilling only support single batch one by one."
        assert seq_len <= self.input_length        
        valid_start = self.valid_start[batch_idx]  
        # Sink Zone: [valid_start, sink_end)
        sink_start = valid_start # 0
        sink_end = valid_start + self.sink_size   # 4
        # 注意:当 seq_len < steady_size 时,会出现以下情况:
        #    - retrieval_end 会是负数(因为 seq_len - local_size < 0) 这时应该理解为:没有 Retrieval Zone, 所有数据都在 Steady Zone
        #    - Retrieval Zone: [retrieval_start, retrieval_end)
        retrieval_start = sink_end  # 4
        retrieval_end_raw = seq_len - self.local_size  # 139-512=-373可能为负数
        retrieval_end = max(retrieval_start, retrieval_end_raw)  # 4 确保 >= retrieval_start
        # Local Zone: [local_start, local_end)
        local_start = retrieval_end  # 4 如果 retrieval_end = retrieval_start,说明没有 retrieval zone
        local_end = seq_len  # 139
        retrieval_length = retrieval_end - retrieval_start  # 现在一定 >= 0
        valid_length = retrieval_length  # 保持变量名兼容性
        
        # 设置初始的 retrieval zone 长度
        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length
            
            # 重新分配精确大小的 buffer（避免每次 slice）
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
            
        # 如果 retrieval_length = 0,说明 prefill 长度太短, kvcache全部保存在 GPU(Steady Zone)中
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                # print(f"[Short Prefill] prefill_length={seq_len} <= steady_size={self.steady_size}, no retrieval zone")
                self._short_prefill_printed = True
            # 短序列全部放 GPU（不超过 gpu_kv_capacity）
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = key_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = value_states[0, valid_start:valid_start+actual_len, :, :].transpose(0, 1).to(self.dtype)
            self.current_seq_len[layer_idx] = actual_len
            if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]
        
        # ===========================================
        # Step1: 将 prefill tokens 分区存储
        # Offload 模式：Sink+Local → GPU, Retrieval → CPU
        # 非 Offload 模式：全部 → GPU
        local_len = local_end - local_start
        
        if self.enable_offload:
            # 【Offload 模式】GPU 只存 Sink + Local → GPU(Sink Zone + Local Zone)
            self.unified_keys_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = key_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :self.sink_size, :] = value_states[0, sink_start:sink_end, :, :].transpose(0, 1).to(self.dtype)
            
            # Local → GPU[sink_size:sink_size+local_len]
            self.unified_keys_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] = key_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, self.sink_size:self.sink_size+local_len, :] =  value_states[0, local_start:local_end, :, :].transpose(0, 1).to(self.dtype)
            
            # Retrieval Zone → CPU（异步拷贝）
            temp_retrieval_keys = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            temp_retrieval_values = value_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).to(self.dtype)
            # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
            self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
            self._layer_offload_inflight[layer_idx] = True
            # 主流记录 event：数据已准备好
            self._main_events[layer_idx].record()
            
            # Copy stream 异步拷贝：不阻塞主流，与后续 batch_encode 并行
            with torch.cuda.stream(self._copy_stream):
                self._main_events[layer_idx].wait()  # 等待数据准备好
                # 使用 copy_ + non_blocking 做异步拷贝（目标是 pinned memory）
                self.unified_keys_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(temp_retrieval_keys, non_blocking=True)
                self.unified_values_cpu[layer_idx][batch_idx, :, :retrieval_length, :].copy_(temp_retrieval_values, non_blocking=True)
                self._copy_events[layer_idx].record()  # 标记该层拷贝完成
            
            self.cpu_valid_end[layer_idx] = retrieval_length
            
            # if layer_idx == 0 and batch_idx == 0:
                # print(colored(f"[Offload Prefill] Sink({self.sink_size})+Local({local_len})→GPU, Retrieval({retrieval_length})→CPU (async)", 'yellow'))
                # pass
        else:
            # 【非 Offload 模式】全部存 GPU
            actual_len = seq_len - valid_start  # 有效 token 数（去除左侧 padding）
            self.unified_keys_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                key_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)
            self.unified_values_gpu[layer_idx][batch_idx, :, :actual_len, :] = \
                value_states[0, valid_start:seq_len, :, :].transpose(0, 1).to(self.dtype)
            
    
        
        # 记录有效序列长度（用于 decode 时确定写入位置）
        self.current_seq_len[layer_idx] = seq_len - valid_start
        

        #  step2: 计算方向码本(量化keys,K_r=1 优化:不再计算半径码) 提取 Retrieval Zone 用于编码（仅用于构建索引，不需要单独存储）
        # ----- Step2a: batch_encode -----
        # nvtx.range_push("prefill_step1_batch_encode")
        # key_states: [1, seq_len, kv_head, head_dim] -> [1, kv_head, retrieval_length, head_dim]
        keys_to_encode = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).contiguous().unsqueeze(0)
        codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
            keys_to_encode, layer_idx=layer_idx, return_rabitq=USE_RABITQ_RERANK)
        
        # codebook_batch: [1, kv_head, B, retrieval_length]（batch_encode 已转置）
        self.codebook_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = codebook_batch[0]
        if USE_RABITQ_RERANK:
            # block_weight_batch: [1, kv_head, B, retrieval_length] bf16
            # packed4b_batch: [1, kv_head, B, retrieval_length, 4]（batch_encode 已转置）
            self.key_block_weight_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end] = block_weight_batch[0]
            self.key_4bit_packed_gpu[layer_idx][batch_idx, :, :, retrieval_start:retrieval_end, :] = packed4b_batch[0]
        # nvtx.range_pop()
        
        # step3: 统计每个cluster包含的key数量（codebook_batch 是 [1, kv_heads, B, len]，取 [0]）
        # nvtx.range_push("prefill_step3_cluster_count")
        self.cluster_key_counts_gpu[layer_idx][batch_idx] = self._count_cluster_keys(codebook_batch, codebook_batch.device)[0]  # [1, kv_heads, B, K_omega] -> [kv_heads, B, K_omega]
        # nvtx.range_pop()
        
        
        # 更新context:目前处理的tokens数量（有效长度，不含 padding）
        if (layer_idx == self.layer_num - 1) and (batch_idx + bsz == self.batch_size):
            self.context += self.current_seq_len[layer_idx]  # 已在上面计算
            # 【同步点】Prefill 结束，等待所有异步 offload 完成
            if self.enable_offload:
                self._sync_offload_copies()
        else:
            # Prefill 的 key_states/value_states 往往是短生命周期张量（来自当前 layer forward）。
            # 为避免“跨层累计持有大张量引用”导致峰值显存暴涨，这里按层等待本层 D2H 完成并清理引用。
            if self.enable_offload:
                self._sync_layer_offload(layer_idx)
                
        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]
    
    
    
    
    
    
    def _sync_offload_copies(self):
        """等待所有异步 offload 拷贝完成"""
        if not self.enable_offload:
            return
        # 等待 copy stream 上的所有操作完成
        self._copy_stream.synchronize()
        # 清理临时引用
        if hasattr(self, "_inflight_offload_buffers"):
            for ldx in range(self.layer_num):
                self._inflight_offload_buffers[ldx] = None
                self._layer_offload_inflight[ldx] = False



    def _sync_layer_offload(self, layer_idx: int):
        """只等待指定层的异步 offload 完成（PQCache 风格：按层精确等待）。"""
        if not self.enable_offload:
            return
        if not hasattr(self, "_copy_events"):
            return
        if hasattr(self, "_layer_offload_inflight") and (not self._layer_offload_inflight[layer_idx]):
            return
        self._copy_events[layer_idx].synchronize()
        if hasattr(self, "_inflight_offload_buffers"):
            self._inflight_offload_buffers[layer_idx] = None
        if hasattr(self, "_layer_offload_inflight"):
            self._layer_offload_inflight[layer_idx] = False






    def get_region_indices(self, layer_idx, device=None):
        """Use the committed offload boundary, including a full unflushed buffer."""
        device = self.device if device is None else device
        total = self.current_seq_len[layer_idx]
        retrieval = self.cpu_valid_end[layer_idx]
        sink_end = min(self.sink_size, total)
        local_start = min(total, self.sink_size + retrieval)
        return {
            "sink_indices": torch.arange(sink_end, device=device, dtype=torch.long),
            "local_indices": torch.arange(local_start, total, device=device, dtype=torch.long),
            "retrieval_start": self.sink_size,
            "retrieval_length": retrieval,
            "total_len": total,
            "local_len": total - local_start,
        }


    def _compute_local_start(self, layer_idx=0):
        # Called at a token boundary with all layers' full buffers available.
        return max(self.sink_size, self.current_seq_len[layer_idx] - self.local_size)


    def _get_full_kv_for_attention(self, layer_idx, sink_indices, local_indices, 
                                    retrieval_start, retrieval_length, device, dtype):
        """
        获取完整的 KV 用于 attention 计算（Sink + Retrieval + Local）
        统一处理 Offload 和非 Offload 模式，避免代码重复
        
        Args:
            layer_idx: 层索引
            sink_indices: Sink 区域的全局索引
            local_indices: Local+Buffer 区域的全局索引  
            retrieval_start: Retrieval 区域起始位置
            retrieval_length: Retrieval 区域长度（理论值）
            device: 目标设备
            dtype: 数据类型
        Returns:
            (concat_keys, concat_values): 拼接后的 KV，形状 [bs, kv_head, total_len, head_dim]
        """
        local_len = local_indices.shape[0]
        
        if self.enable_offload and self.unified_keys_cpu is not None and retrieval_length > 0:
            self._sync_layer_offload(layer_idx)
            
            # Sink+Local 在 GPU 上（先提取出来，保证变量作用域）
            sink_local_end = self.sink_size + local_len
            sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
            sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
            
            cpu_retrieval_len = self.cpu_valid_end[layer_idx]
            if cpu_retrieval_len > 0:
                # Retrieval 数据已 offload 到 CPU，拼接: Sink+Local + Retrieval（causal=False，顺序无关）
                retrieval_keys = self.unified_keys_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(device, non_blocking=True)
                retrieval_values = self.unified_values_cpu[layer_idx][:, :, :cpu_retrieval_len, :].to(device, non_blocking=True)
                concat_keys = torch.cat([sink_local_keys, retrieval_keys], dim=2)
                concat_values = torch.cat([sink_local_values, retrieval_values], dim=2)
            else:
                # CPU 没有 offload 数据，所有 token 仍在 GPU，直接用 Sink+Local
                concat_keys = sink_local_keys
                concat_values = sink_local_values
                
        else:
            # 【非 Offload 模式】全部从 GPU 获取
            if retrieval_length > 0:
                retrieval_end = retrieval_start + retrieval_length
                retrieval_indices = torch.arange(retrieval_start, retrieval_end, device=device, dtype=torch.long)
                # 按物理顺序拼接：Sink → Retrieval → Local
                all_indices = torch.cat([sink_indices, retrieval_indices, local_indices])
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)
            else:
                # 没有 retrieval zone，只用 Sink + Local
                all_indices = torch.cat([sink_indices, local_indices]) if local_len > 0 else sink_indices
                concat_keys = torch.index_select(self.unified_keys_gpu[layer_idx], 2, all_indices)
                concat_values = torch.index_select(self.unified_values_gpu[layer_idx], 2, all_indices)
        
        return concat_keys, concat_values





    def _update_polar_index(self):
        """
        动态更新:对Steady Zone中旧的Local部分进行Polar编码并更新倒排索引(GPU版本)
        逻辑:
        1. 提取旧的Local tokens [4:68] → 编码 → 追加到统一KV缓存
        2. 把新生成的64个tokens [68:132] 移到Local位置 [4:68]
        3. Steady Zone重置为 [Sink(4) + Local(64)] = 68
        """
        # 计算需要编码的范围：[old_retrieval_end : new_retrieval_end]
        # 关键：这里用的是 _update_polar_index 被调用时的 current_seq_len[0]
        # 这个值是在写入当前 token 之前的值！
        new_retrieval_end = self._compute_local_start(0)
        old_retrieval_end = self._actual_retrieval_length + self.sink_size
        encode_start = old_retrieval_end
        encode_end = new_retrieval_end
        num_new_tokens = encode_end - encode_start
        
        # 边界检查（触发时序已修复，num_new_tokens 应该总是 > 0）
        assert num_new_tokens > 0, f"num_new_tokens={num_new_tokens} <= 0, 触发逻辑有 bug"
        assert encode_start >= self.sink_size, f"encode_start ({encode_start}) < sink_size ({self.sink_size})"
        assert encode_end <= self.current_seq_len[0], f"encode_end ({encode_end}) > current_seq_len ({self.current_seq_len[0]})"

        # 更新 _actual_retrieval_length（编码到的相对长度）
        # Retrieval Zone = [sink_size : encode_end]
        self._actual_retrieval_length = encode_end - self.sink_size
        
        for ldx in range(self.layer_num):
            # 获取需要编码的新增部分
            if self.enable_offload:
                # 【Offload 模式】从 GPU 获取即将 offload 的 tokens
                # GPU 布局: [sink | local | update_buffer]
                # 需要编码的是 [sink : sink + num_new_tokens]（旧 Local + Buffer 前半）
                offload_start = self.sink_size
                offload_end = self.sink_size + num_new_tokens
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, offload_start:offload_end, :]
            else:
                # 【非 Offload 模式】按全局位置获取
                new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, encode_start:encode_end, :]
                
            # 统一架构：批量编码新增部分，按绝对位置存储
            # 注意：codebook/weight/4bit 已预分配 input_length + max_new_length 容量
            # 一次性处理整个 batch（避免逐 batch 循环的性能开销）
            # new_tokens_to_encode: [batch_size, kv_heads, num_new_tokens, head_dim]
            codebook_batch, block_weight_batch, packed4b_batch = self.batch_encode(
                new_tokens_to_encode, layer_idx=ldx, return_rabitq=USE_RABITQ_RERANK)
            # 输出: [batch_size, kv_heads, B, num_new_tokens, ...]
            self.codebook_gpu[ldx][:, :, :, encode_start:encode_end] = codebook_batch
            if USE_RABITQ_RERANK:
                self.key_block_weight_gpu[ldx][:, :, :, encode_start:encode_end] = block_weight_batch
                self.key_4bit_packed_gpu[ldx][:, :, :, encode_start:encode_end, :] = packed4b_batch
            # 累加新增 keys 的统计（_count_cluster_keys 已支持 multi-batch，返回 [batch, kv_heads, B, K]）
            cluster_counts = self._count_cluster_keys(codebook_batch, codebook_batch.device)
            self.cluster_key_counts_gpu[ldx] += cluster_counts
            del codebook_batch, block_weight_batch, packed4b_batch, cluster_counts

            
            # ===========================================
            # 【Offload 模式】Update 时：
            # 1. 把 Local 最老的部分 offload 到 CPU（进入 retrieval zone）
            # 2. 滑动 GPU 上的 Local window（Local 新部分 + Update Buffer → 新 Local）
            if self.enable_offload and self.unified_keys_cpu is not None:
                # 1. 把 Local 最老的 num_new_tokens 个 tokens offload 到 CPU
                # 这些 tokens 从 GPU local 区域移出，进入 CPU retrieval zone
                # GPU 布局: [0:sink] [sink:sink+local] [sink+local:sink+local+update]
                # Local 最老部分: [sink : sink+num_new_tokens]
                
                cpu_write_start = self.cpu_valid_end[ldx]  # 第 ldx 层 CPU 缓存的有效结束位置+1，即本次 offload 的开始写入位置
                cpu_write_end = cpu_write_start + num_new_tokens  # 本次 offload 的结束写入位置
                
                # 【优化】使用预分配 buffer 替代 clone()，避免内存分配开销
                # 注意：num_new_tokens 应该等于 dynamic_update_interval
                src_keys = self._offload_keys_buffer[ldx][:, :, :num_new_tokens, :]
                src_values = self._offload_values_buffer[ldx][:, :, :num_new_tokens, :]
                src_keys.copy_(self.unified_keys_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                src_values.copy_(self.unified_values_gpu[ldx][:, :, self.sink_size:self.sink_size+num_new_tokens, :])
                # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
                self._inflight_offload_buffers[ldx] = (src_keys, src_values)
                self._layer_offload_inflight[ldx] = True
                
                # 在 copy stream 中异步拷贝到 CPU（与主流并行） 
                self._main_events[ldx].record()
                with torch.cuda.stream(self._copy_stream):
                    self._main_events[ldx].wait()
                    # 将 src_keys  拷贝到 dst_keys 
                    dst_keys = self.unified_keys_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_values = self.unified_values_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                    dst_keys.copy_(src_keys, non_blocking=True)
                    dst_values.copy_(src_values, non_blocking=True)
                    self._copy_events[ldx].record()
                    
                self.cpu_valid_end[ldx] = cpu_write_end
                
                # 2. 滑动 GPU Local window（可以与 CPU 拷贝并行）
                # 把 [sink+num_new_tokens : sink+local+num_new_tokens] 移动到 [sink : sink+local]
                # 即：原 Local 新部分 + 原 Update Buffer → 新 Local
                old_local_end = self.sink_size + self.local_size
                old_buffer_end = old_local_end + num_new_tokens
                new_local_start_src = self.sink_size + num_new_tokens  # old_local_start + num_new_tokens
                
                # 只有当 num_new_tokens < local_size 时源和目标才会重叠，需要 clone
                if num_new_tokens < self.local_size:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                else:
                    new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                    new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                self.unified_keys_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_keys
                self.unified_values_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_values
                # 显式释放 clone 的临时 tensor（如果是 clone 的话）
                if num_new_tokens < self.local_size:
                    del new_local_keys, new_local_values
                
                # if ldx == 0:
                #     print(colored(f"[Offload] GPU滑窗完成: 新Local=[{self.sink_size}:{old_local_end}] (来自[{new_local_start_src}:{old_buffer_end}]), CPU已存{cpu_write_end}个token", 'yellow'))
        
        # 等待所有层的 offload 完成，这样 retrieval 阶段不需要再检查
        if self.enable_offload:
            self._sync_offload_copies()
        
        self.steady_size = self.sink_size + self.local_size 





    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """Flush a full buffer only after all layers wrote the previous token."""
        assert self.enable_offload, "Revision runtime requires CPU pinned KV offload"
        current = self.current_seq_len[layer_idx]
        if layer_idx == 0:
            near = current - min(self.sink_size, current) - self.cpu_valid_end[0]
            if near >= self.local_size + self.dynamic_update_interval:
                assert all(n == current for n in self.current_seq_len)
                self._update_polar_index()
                self._revision_update_count = getattr(self, "_revision_update_count", 0) + 1
        write_pos = current - self.cpu_valid_end[layer_idx]
        assert 0 <= write_pos < self.gpu_kv_capacity, (current, write_pos)
        self.unified_keys_gpu[layer_idx][:, :, write_pos, :] = key_states[:, 0]
        self.unified_values_gpu[layer_idx][:, :, write_pos, :] = value_states[:, 0]
        self.current_seq_len[layer_idx] = current + 1
        if layer_idx == 0:
            self._decode_step_counter += 1
        # position_embedd reads this shared context before each layer append.
        if layer_idx == self.layer_num - 1:
            self.context += 1
        return None, None


    def _pack_4bit_codes(self, signs: torch.Tensor, magnitudes: torch.Tensor) -> torch.Tensor:
        """
        将 1-bit 符号 + 3-bit 幅值合并打包成 4-bit。
        Args:
            signs: [..., m] bool, True=正(>=0), False=负(<0)
            magnitudes: [..., m] uint8, 0-7 (3-bit 幅值码)
        Returns:
            packed: [..., m//2] uint8, 每个 byte 存 2 个 4-bit 值 (m=8 时为 4 bytes)
        """
        # 4-bit = (sign << 3) | magnitude, 范围 0-15
        v4bit = (signs.to(torch.uint8) << 3) | magnitudes  # [..., m], uint8
        # 每 2 个 4-bit 打包成 1 个 byte: (v_odd << 4) | v_even
        v_even = v4bit[..., 0::2]  # 偶数索引: v0, v2, v4, v6
        v_odd = v4bit[..., 1::2]   # 奇数索引: v1, v3, v5, v7
        packed = (v_odd << 4) | v_even  # [..., m//2]
        return packed.to(torch.uint8).contiguous()



    def _unpack_4bit_codes(self, packed: torch.Tensor, m: int = 8) -> tuple:
        """
        从 4-bit packed 解出符号和幅值。
        Args:
            packed: [..., m//2] uint8 (m=8 时为 4 bytes)
        Returns:
            s_sign: [..., m] int8, ±1
            t_codes: [..., m] int64, 0-7 (用于查表 _mag_centers)
        """
        # 解包 2 个 4-bit 值
        v_even = packed & 0x0F           # [..., m//2], 低 4 位
        v_odd = (packed >> 4) & 0x0F     # [..., m//2], 高 4 位
        # 交错合并: [v0, v1, v2, v3, ...] from [even0, even1, ...] and [odd0, odd1, ...]
        v4bit = torch.stack([v_even, v_odd], dim=-1).view(*packed.shape[:-1], m)  # [..., m]
        # 拆分符号和幅值
        sign_bits = (v4bit >> 3) & 1     # [..., m], 0/1
        s_sign = (2 * sign_bits - 1).to(torch.int8)  # 0->-1, 1->+1
        t_codes = (v4bit & 0x7).to(torch.int64)      # [..., m], 0-7
        return s_sign, t_codes



    def _encode_query_blocks(self, query: torch.Tensor) -> tuple:
        """
        将 query 编码为 SRHT 后的 blocks（用于 3-bit rerank）。
        Args:
            query: [bs, kv_heads, head_dim] 原始 query（未归一化）
        Returns:
            q_blocks: [bs, kv_heads, B, m] (来自单位化 query 的 SRHT 结果，包含 block 半径)
            q_norm: [bs, kv_heads, 1] query 的模长
        """
        q_norm = torch.linalg.vector_norm(query, ord=2, dim=-1, keepdim=True)  # [bs, kv, 1]
        query_normalized = query / q_norm.clamp(min=1e-8)
        Y = query_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y = Y * self.quantizer.srht_diagonal_signs[round_idx]
            Y = hadamard_transform(Y, scale=self._hadamard_scale)
        q_blocks = Y.view(query.shape[0], query.shape[1], self.polar_B, self.polar_m).contiguous()
        return q_blocks, q_norm




    # 批量版本的编码和索引构建函数
    def batch_encode(self, Y, layer_idx=None, return_rabitq: bool = True):
        """
        4-bit RaBitQ 编码：1-bit 符号 + 3-bit 幅值
        Args:
            Y: [bs, kv_heads, seq_len, head_dim] GPU tensor
            layer_idx: 层索引(可选,用于调试)
            return_rabitq: 是否返回完整编码（默认 True，False 时跳过 4-bit 量化）
        Returns:
            codebook: [bs, kv_heads, B, seq_len] uint8, bitpacked 符号（已转置，可直接存储）
            block_weight: [bs, kv_heads, B, seq_len] bf16, 已融合 (r/α) * ||k||（return_rabitq=False 时为 None）
            packed_4bit: [bs, kv_heads, B, seq_len, m//2] uint8, 4-bit packed（return_rabitq=False 时为 None）
        """
        bs, kv_heads, seq_len, head_dim = Y.shape
        key_norm = torch.linalg.vector_norm(Y, ord=2, dim=-1, keepdim=True)  # [bs, kv_heads, seq_len, 1]
        Y_normalized = Y / key_norm.clamp(min=1e-8)  # [bs, kv_heads, seq_len, head_dim]
        key_norm = key_norm.squeeze(-1)  # [bs, kv_heads, seq_len]
        
        # 步骤2:Layer-specific SRHT旋转(直接在4D tensor上操作)
        Y_rotated = Y_normalized
        for round_idx in range(DEFAULT_SRHT_ROUNDS):
            Y_rotated = Y_rotated * self.quantizer.srht_diagonal_signs[round_idx]
            Y_rotated = hadamard_transform(Y_rotated, scale=self._hadamard_scale)
                
        # 步骤3:分块 [bs, kv_heads, seq_len, head_dim] -> [bs, kv_heads, seq_len, B, m]
        Y_blocks = Y_rotated.view(bs, kv_heads, seq_len, self.quantizer.B, self.quantizer.m)
        
        # 步骤4: 方向量化 - 直接 bitpack sign(u)
        # 码本就是全体 256 个 sign pattern，直接 bitpack 比 256-way dot + argmax 更快、更数值稳定
        bits = (Y_blocks >= 0)  # [bs, kv, seq, B, m], bool
        angular_codes_u8 = (bits * self._bitpack_shifts).sum(dim=-1, dtype=torch.uint8)  # [bs, kv, seq, B]
        codebook = angular_codes_u8.permute(0, 1, 3, 2).contiguous()  # [bs, kv, B, seq]
        

        # 如果不需要 RaBitQ rerank，只返回 codebook（用于碰撞检索）
        if not return_rabitq:
            return codebook, None, None
        
        # ===== 以下是 4-bit RaBitQ rerank 需要的计算 =====
        # 计算每个 block 的半径 r_{k,b}
        block_radii = torch.linalg.vector_norm(Y_blocks, ord=2, dim=-1)  # [bs, kv_heads, seq_len, B]
        # 方向归一化
        denom = block_radii.unsqueeze(-1).clamp(min=1e-8)
        directions = Y_blocks / denom  # [bs, kv_heads, seq_len, B, m]
        del Y_blocks, denom 

        # ===== 4-bit 合并量化：1-bit 符号 + 3-bit 幅值 =====
        # t_j = bucketize(|u_j|, τ) ∈ {0,1,2,3,4,5,6,7}
        abs_u = directions.abs()  # 单位向量的大小，[bs, kv, seq, B, m] bf16
        del directions  # 释放 directions，只保留 abs_u
        t_codes = torch.bucketize(abs_u, self._mag_thresholds)  # 返回 int64
        
        # ===== 计算 alpha^(4b) = Σ a[t_j] * |u_j| =====
        # 先用 t_codes (int64) 做索引，再转 uint8，避免同时持有两个大 tensor
        a_vals = self._mag_centers[t_codes]  # [bs, kv, seq, B, m] bf16
        t_codes_u8 = t_codes.to(torch.uint8)
        del t_codes  # 释放 int64 tensor 以节省显存
        # 4-bit pack: 1-bit sign + 3-bit magnitude → 4 bytes per block (m=8)
        # bits: [bs, kv, seq, B, m] bool, t_codes_u8: [bs, kv, seq, B, m] uint8
        packed_4bit = self._pack_4bit_codes(bits, t_codes_u8)  # [bs, kv, seq, B, m//2]
        alpha4b = (a_vals * abs_u).sum(dim=-1, dtype=torch.float32)  # [bs, kv, seq, B]，abs_u 已是 bf16
        alpha4b = alpha4b.clamp(min=1e-6)
        # weight = (r / α) * ||k||（RaBitQ 公式，必须除以 alpha）
        block_weight = (block_radii / alpha4b) * key_norm.unsqueeze(-1)  # [bs, kv, seq, B]
        block_weight = block_weight.permute(0, 1, 3, 2).to(torch.bfloat16).contiguous()  # [bs, kv, B, seq]
        packed_4bit = packed_4bit.permute(0, 1, 3, 2, 4).contiguous()  # [bs, kv, B, seq, m//2]
        return codebook, block_weight, packed_4bit
    




    def _count_cluster_keys(self, codebook, device):
        """
        统计每个 cluster 包含的 key 数量（支持 multi-batch）
        Args:
            codebook: [batch, kv_heads, B, kv_len]
            device: 目标设备
        Returns:
            counts: [batch, kv_heads, B, K_omega] 每个 cluster 的 key 数量
        """
        batch_size, kv_heads, B, kv_len = codebook.shape
        actual_K = self.polar_K_omega
        num_groups = kv_heads * B
        
        # 逐 batch 处理（bincount 不支持 batch 维度）
        counts_list = []
        for b in range(batch_size):
            codebook_b = codebook[b]  # [kv_heads, B, kv_len]
            codebook_flat = codebook_b.reshape(-1).long()
            group_ids = torch.arange(num_groups, device=device).view(kv_heads, B, 1).expand(-1, -1, kv_len).reshape(-1)
            global_ids = codebook_flat + group_ids * actual_K
            counts_flat = torch.bincount(global_ids, minlength=num_groups * actual_K).to(torch.int32)
            counts_list.append(counts_flat.view(kv_heads, B, actual_K))
        
        return torch.stack(counts_list, dim=0)  # [batch, kv_heads, B, K_omega]
    
    
    
    # 单层Decode 阶段的 Attention 计算
    # 输入:query 向量,  输出:attention 结果
    def compute(self, queries, layer_idx):
        """
        输入: queries [batch_size, seq_len, num_heads, head_dim]  seq_len=1
        输出: attn_out [batch_size, seq_len, num_heads, head_dim] seq_len=1
        Decode计算:
        - 当 total_length < 1000 时:全量 attention(不检索)
        - 当 total_length >= 1000 时:使用Polar ANN检索top-k tokens
        """
        # nvtx.range_push(f"decode_layer_{layer_idx}")
        nvtx.range_push("decode_prepare")
        
        # 统一架构：动态计算各区域索引（预计算 sink/local indices，避免重复切片）
        device = self.device
        region_info = self.get_region_indices(layer_idx, device=device)
        total_length = region_info['total_len']
        sink_indices = region_info['sink_indices']
        local_indices = region_info['local_indices']
        retrieval_start = region_info['retrieval_start']
        retrieval_length = region_info['retrieval_length']
        local_len = region_info['local_len']
        
        # 计算有效检索长度（避免访问尚未编码/offload 的区域）
        encoded_len = self._actual_retrieval_length or 0
        cpu_len = self.cpu_valid_end[layer_idx] if (self.enable_offload and self.cpu_valid_end is not None) else 0
        if self.enable_offload and self.cpu_valid_end is not None:
            # Offload 模式：cpu_len 是实际限制（cpu_len <= encoded_len <= retrieval_len）
            valid_length = cpu_len
        else:
            # 非 Offload 模式：需要同时考虑理论值和实际编码值
            valid_length = min(retrieval_length, encoded_len)
        retrieval_end_effective = retrieval_start + valid_length
        
        # 准备 query: [bs, 1, num_heads, head_dim] -> [bs, kv_heads, head_dim]
        bs, seqlen_q, head_q, head_dim = queries.shape
        query_group = queries[:, 0, :, :]
        queries_reshaped = query_group.view(bs, self.kv_head, self.group_size, self.head_dim)
        query_avg_gpu = queries_reshaped.mean(dim=2)  # [bs, kv_heads, head_dim]
        query_batch = query_avg_gpu.unsqueeze(2)  # [bs, kv_heads, 1, head_dim]
        
        # 判断是否需要检索
        # total_length 至少要 > sink + local + final_topk*2，否则 retrieval zone 太小
        min_total_for_retrieval = max(self.FULL_ATTENTION_THRESHOLD, self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)
        # key_data_gpu 仅用于 exact rerank / recall 计算（延迟获取）
        key_data_gpu = None
        nvtx.range_pop()  # decode_prepare_query



        if use_full_attention:
            # ============ 全量 Attention 模式 ============
            concat_keys, concat_values = self._get_full_kv_for_attention(layer_idx, sink_indices, local_indices, retrieval_start, retrieval_length, device, queries.dtype)
            if layer_idx == 0 and not hasattr(self, '_full_attn_printed'):
                print(f"[Full Attention] threshold={min_total_for_retrieval} (FULL_ATTN={self.FULL_ATTENTION_THRESHOLD}, sink={self.sink_size}, local={self.local_size}, topk={self.final_topk})", flush=True)
                print(f"[Full Attention] enable_offload={self.enable_offload}, kv_shape={concat_keys.shape}, total_len={total_length}, retrieval_len={retrieval_length}, valid_len={valid_length}", flush=True)
                self._full_attn_printed = True
            if layer_idx == 0 and total_length % 100 == 0:
                print(f"[Mode] total_len={total_length}/{min_total_for_retrieval}, mode=FULL_ATTN, kv_len={concat_keys.shape[2]}, offload={self.enable_offload}", flush=True)
        else:
            if layer_idx == 0 and not hasattr(self, '_retrieval_printed'):
                print(f"[Retrieval Mode] total_len={total_length} >= {min_total_for_retrieval}, valid_len={valid_length}, retrieval_len={retrieval_length}", flush=True)
                self._retrieval_printed = True
            if layer_idx == 0 and total_length % 100 == 0:
                print(f"[Mode] step total_len={total_length}, threshold={min_total_for_retrieval}, mode=RETRIEVAL, valid_len={valid_length}", flush=True)
           
            # ============ 检索模式 ============
            # nvtx.range_push("get_retrieval_data")
                
            # 按绝对位置获取 codebook、weight 和 4-bit packed
            codebook_gpu = self.codebook_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective]
            if USE_RABITQ_RERANK:
                key_block_weight_gpu = self.key_block_weight_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective]
                key_4bit_gpu = self.key_4bit_packed_gpu[layer_idx][:, :, :, retrieval_start:retrieval_end_effective, :]
                # 消融实验需要原始 key 数据来计算精确内积
                if self.enable_rerank_error_ablation or self.enable_coarse_recall_ablation or self.enable_rerank_recall_ablation:
                    key_data_gpu = self.unified_keys_gpu[layer_idx][:, :, retrieval_start:retrieval_end_effective, :]
                else:
                    key_data_gpu = None
            else:
                # Exact rerank 需要原始 key 数据
                key_block_weight_gpu = None
                key_4bit_gpu = None
                key_data_gpu = self.unified_keys_gpu[layer_idx][:, :, retrieval_start:retrieval_end_effective, :]
            
            # 【Offload 模式】Retrieval 起点固定为 sink_size
            assert not self.enable_offload or retrieval_start == self.sink_size, f"Offload 模式下 retrieval_start({retrieval_start}) 必须等于 sink_size({self.sink_size})"
            # nvtx.range_pop()  # get_retrieval_data




            # 自适应调节 candidate_ratio 和 collision_ratio
            retrieval_len = valid_length  # retrieval zone 的实际长度
            _fixed_cr = os.environ.get("POLAR_FIXED_CANDIDATE_RATIO", "")
            if _fixed_cr:
                adaptive_candidate_ratio = float(_fixed_cr)
                adaptive_collision_ratio = float(_fixed_cr)
            else:
                adaptive_candidate_ratio, adaptive_collision_ratio = self._get_adaptive_ratios(retrieval_len)
            # print(f"[Adaptive] retrieval_len={retrieval_len}, candidate_ratio={adaptive_candidate_ratio}, collision_ratio={adaptive_collision_ratio}", flush=True)
            # 与 polar_cache_value_adaptive_k_cuda.compute 的 NVTX 区间命名对齐（见该文件 NVTX 说明）
            nvtx.range_push("collision_based_topk_batch")
            topk_indices_batch = self.collision_based_topk_batch(
                query=query_batch,
                codebook=codebook_gpu,
                key_data=key_data_gpu,
                key_block_weight=key_block_weight_gpu,
                key_4bit=key_4bit_gpu,
                collision_ratio=adaptive_collision_ratio,
                candidate_ratio=adaptive_candidate_ratio,
                layer_idx=layer_idx
            )
            nvtx.range_pop()
            # if layer_idx == 0:
            #     print(f"[Debug] topk_indices_batch.shape={topk_indices_batch.shape}, final_topk={self.final_topk}", flush=True)

            # 与 value_adaptive 中该 range 一致：仅 head 展宽索引（adaptive 版此处含 sink+local 准备 + 自适应选 k，本处无自适应）
            nvtx.range_push("fixed_topk_sl_pre")
            sink_indices_exp = sink_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)
            local_indices_exp = local_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)
            nvtx.range_pop()

            if self.enable_offload and self.unified_keys_cpu is not None:
                # 【Offload 模式】与 value_adaptive 相同：h2d_gather_kv → compute_attention(含 cat + flash)
                nvtx.range_push("h2d_gather_kv")
                self._h2d_gather_kv(
                    self.unified_keys_cpu[layer_idx],    # src keys: CPU pinned
                    self.unified_values_cpu[layer_idx],  # src values: CPU pinned
                    topk_indices_batch,                   # indices: GPU (无需传回 CPU)
                    self._topk_keys_buffer,               # dst keys: GPU
                    self._topk_values_buffer              # dst values: GPU
                )
                nvtx.range_pop()

                sink_local_end = self.sink_size + local_len
                sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
                sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]
                
                nvtx.range_push("compute_attention")
                concat_keys = torch.cat([sink_local_keys, self._topk_keys_buffer], dim=2)
                concat_values = torch.cat([sink_local_values, self._topk_values_buffer], dim=2)
                k_cache = concat_keys.transpose(1, 2)
                v_cache = concat_values.transpose(1, 2)
                attn_out = flash_attn_with_kvcache_compat(
                    q=queries,
                    k_cache=k_cache,
                    v_cache=v_cache,
                    causal=False
                )
                del sink_local_keys, sink_local_values, concat_keys, concat_values, k_cache, v_cache
                nvtx.range_pop()

                return attn_out
            else:
                # 【非 Offload】GPU gather top-k；NVTX 名与 h2d 版一致，便于同一张 timeline 上对比「取回 topk」阶段
                nvtx.range_push("h2d_gather_kv")
                topk_indices_abs = topk_indices_batch + retrieval_start
                all_indices = torch.cat([sink_indices_exp, topk_indices_abs, local_indices_exp], dim=2)
                all_indices_exp = all_indices.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
                concat_keys = torch.gather(self.unified_keys_gpu[layer_idx], 2, all_indices_exp)
                concat_values = torch.gather(self.unified_values_gpu[layer_idx], 2, all_indices_exp)
                del all_indices, all_indices_exp, topk_indices_abs
                nvtx.range_pop()

        # 7. 全量 Attention 或 非 Offload 检索：cat 已完成，仅 Flash
        k_cache = concat_keys.transpose(1, 2)
        v_cache = concat_values.transpose(1, 2)
        nvtx.range_push("compute_attention")
        attn_out = flash_attn_with_kvcache_compat(
            q=queries,  # [bs, 1, num_heads, head_dim]
            k_cache=k_cache,  # [bs, seq_len, kv_heads, head_dim]
            v_cache=v_cache,  # [bs, seq_len, kv_heads, head_dim]
            causal=False
        )
        del concat_keys, concat_values, k_cache, v_cache
        nvtx.range_pop()
        return attn_out






    # ============================================================================
 
    
    # def _get_adaptive_ratios(self, retrieval_len):
    #     """
    #     根据 retrieval zone 长度自适应调节 candidate_ratio 和 collision_ratio
    #     Args:
    #         retrieval_len: retrieval zone 的长度
    #     Returns:
    #         (candidate_ratio, collision_ratio) 元组
    #     """
    #     if retrieval_len < 5000:
    #         candidate_ratio, collision_ratio = 0.7, 0.7   # (0, 5k)
    #     elif retrieval_len < 10000:
    #         candidate_ratio, collision_ratio = 0.5, 0.5   # (5k, 10k)
    #     elif retrieval_len < 30000:
    #         candidate_ratio, collision_ratio = 0.4, 0.4   # (10k, 30k)
    #     elif retrieval_len < 100000:
    #         candidate_ratio, collision_ratio = 0.4, 0.4   # (30k, 100k)
    #     else:
    #         # (100k+) 超长样本：保持与候选比例一致，防止 collision 低于 candidate
    #         candidate_ratio, collision_ratio = 0.4, 0.4
    #     # Optional override for消融 / sweep（retrieve 路径里 self.collision_ratio 会被此处覆盖）
    #     cand_env = os.environ.get("POLAR_ADAPTIVE_FORCE_CANDIDATE_RATIO")
    #     coll_env = os.environ.get("POLAR_ADAPTIVE_FORCE_COLLISION_RATIO")
    #     if cand_env is not None:
    #         candidate_ratio = float(cand_env)
    #     if coll_env is not None:
    #         collision_ratio = float(coll_env)
    #     # Guardrail: collision_ratio 不应低于 candidate_ratio
    #     collision_ratio = max(collision_ratio, candidate_ratio)
    #     if (cand_env is not None or coll_env is not None) and not getattr(
    #         self, "_logged_adaptive_ratio_override", False
    #     ):
    #         print(
    #             f"[Adaptive override] retrieval_len={retrieval_len} -> "
    #             f"candidate_ratio={candidate_ratio}, collision_ratio={collision_ratio}",
    #             flush=True,
    #         )
    #         self._logged_adaptive_ratio_override = True
    #     return candidate_ratio, collision_ratio
    
    

    def _get_adaptive_ratios(self, retrieval_len):
        """
        根据 retrieval zone 长度自适应调节 candidate_ratio 和 collision_ratio
        Args:
            retrieval_len: retrieval zone 的长度
        Returns:
            (candidate_ratio, collision_ratio) 元组
        """
        dynamic_ratios_env = os.environ.get("POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS", "").strip()
        if dynamic_ratios_env:
            ratios = [float(x.strip()) for x in dynamic_ratios_env.split(",") if x.strip()]
            thresholds_env = os.environ.get("POLAR_ADAPTIVE_DYNAMIC_CAND_THRESHOLDS", "5000,10000")
            thresholds = [int(x.strip()) for x in thresholds_env.split(",") if x.strip()]
            if len(ratios) != len(thresholds) + 1:
                raise ValueError(
                    "POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS must have exactly one more "
                    "value than POLAR_ADAPTIVE_DYNAMIC_CAND_THRESHOLDS"
                )
            idx = 0
            while idx < len(thresholds) and retrieval_len >= thresholds[idx]:
                idx += 1
            candidate_ratio = ratios[idx]
            collision_ratio = candidate_ratio
            coll_ratios_env = os.environ.get("POLAR_ADAPTIVE_DYNAMIC_COLL_RATIOS", "").strip()
            if coll_ratios_env:
                coll_ratios = [float(x.strip()) for x in coll_ratios_env.split(",") if x.strip()]
                if len(coll_ratios) != len(ratios):
                    raise ValueError(
                        "POLAR_ADAPTIVE_DYNAMIC_COLL_RATIOS must match "
                        "POLAR_ADAPTIVE_DYNAMIC_CAND_RATIOS length"
                    )
                collision_ratio = coll_ratios[idx]
            if not getattr(self, "_logged_adaptive_dynamic_ratio", False):
                print(
                    f"[Adaptive dynamic] thresholds={thresholds}, cand_ratios={ratios}, "
                    f"coll_ratios={coll_ratios_env or 'same-as-cand'}",
                    flush=True,
                )
                self._logged_adaptive_dynamic_ratio = True
        elif retrieval_len < 5000:
            candidate_ratio, collision_ratio = 0.5, 0.5   #(0, 5k)
        elif retrieval_len < 10000:
            candidate_ratio, collision_ratio = 0.25, 0.3  #(5k, 10k)
        elif retrieval_len < 30000:
            candidate_ratio, collision_ratio = 0.2, 0.25  #(10k, 30k)   
        elif retrieval_len < 100000:
            # candidate_ratio, collision_ratio = 0.2, 0.2 
            candidate_ratio, collision_ratio = 0.12, 0.2  #(30k, 100k)   short,medium,
        else:
            # (100k+) 超长样本：沿用较保守的设置，避免候选集过大
            candidate_ratio, collision_ratio = 0.12, 0.2
        # Optional override for消融 / sweep（retrieve 路径里 self.collision_ratio 会被此处覆盖）
        cand_env = os.environ.get("POLAR_ADAPTIVE_FORCE_CANDIDATE_RATIO")
        coll_env = os.environ.get("POLAR_ADAPTIVE_FORCE_COLLISION_RATIO")
        if cand_env is not None:
            candidate_ratio = float(cand_env)
        if coll_env is not None:
            collision_ratio = float(coll_env)
        if (cand_env is not None or coll_env is not None) and not getattr(
            self, "_logged_adaptive_ratio_override", False
        ):
            print(
                f"[Adaptive override] retrieval_len={retrieval_len} -> "
                f"candidate_ratio={candidate_ratio}, collision_ratio={collision_ratio}",
                flush=True,
            )
            self._logged_adaptive_ratio_override = True
        return candidate_ratio, collision_ratio
    
    # ============================================================================
    # 批量并行的 collision_based_topk 函数    
    def collision_based_topk_batch(self, query, codebook, key_data=None, key_block_weight=None,
                                     key_4bit=None, collision_ratio=0.2, 
                                     candidate_ratio=0.2, layer_idx=None):
        """
        基于Codebook的碰撞Top-k检索（性能优化版：只返回索引）
        Args:
            query: [bs, kv_heads, 1, head_dim] =[1, 8, 1, 128]（原始 query，未归一化）
            codebook: [bs, kv_heads, B, kv_len] =[1, 8, 8, 165]
            key_data: [bs, kv_heads, kv_len, head_dim] 用于 exact rerank 的 key 数据（可选）
            key_block_weight: [bs, kv_heads, B, kv_len] bf16, 已融合 (r/α) * ||k||
            key_4bit: [bs, kv_heads, B, kv_len, 4] 4-bit packed (1-bit sign + 3-bit magnitude)
            collision_ratio: 碰撞比例（选择多少比例的 clusters）
            candidate_ratio: 候选集比例（返回候选集的数量）
        Returns:
            topk_indices: [bs, kv_heads, candidate_num] =[1, 8, candidate_ratio*kv_len]
        Note:
            性能优化：只返回 indices，调用方从 unified_keys_gpu 中 gather KV
            key_block_weight 已融合 key_norm，省去一次 gather
        """
        nvtx.range_push("phase1_suco")
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1, "Decode阶段每次只处理1个token"        
        query_squeezed = query.squeeze(2)  # [bs, kv_heads, head_dim]=[1, 8, 128]
        # 3-bit rerank: 预计算 q_blocks 和 q_norm（只做一次 SRHT）
        q_blocks, q_norm = self._encode_query_blocks(query_squeezed)  # [bs,kv,B,m], [bs,kv,1]
        # nvtx.range_pop()
        # Step 2+3 融合: 计算碰撞计数 (get_topk_clusters + update_cache_cnt)
        # nvtx.range_push("collision_fused")
        cache_cnt = self.get_cache_cnt_fused(q_blocks, codebook, collision_ratio, layer_idx)
        # nvtx.range_pop()
        
        # Step 4: coarse filter (topk on collision counts)
        # nvtx.range_push("coarse_topk")
        coarse_indices, max_candidates = self._coarse_filter_truncate(cache_cnt, candidate_ratio, layer_idx)
        nvtx.range_pop()

        # # 消融实验：碰撞粗排 candidate Recall（实际粗筛候选 vs 精确内积 top-K/top-C）
        # if self.enable_coarse_recall_ablation:
        #     self._run_coarse_recall_ablation(
        #         query=query_squeezed,
        #         key_data=key_data,
        #         cache_cnt=cache_cnt,
        #         layer_idx=layer_idx,
        #         candidate_ratio=candidate_ratio,
        #         coarse_indices=coarse_indices,
        #     )

        # Step 5: 获取候选集（根据 enable_rerank 选择 rerank 方式）                    
        if USE_RABITQ_RERANK and key_block_weight is not None and key_4bit is not None:
            # 4-bit RaBitQ rerank（key_block_weight 已融合 key_norm）
            nvtx.range_push("rabitq_rerank")
            topk_indices = self.get_candidate_cache_with_rabitq_rerank(
                coarse_indices=coarse_indices,
                q_blocks=q_blocks,
                key_4bit=key_4bit,
                key_block_weight=key_block_weight,
                q_norm=q_norm,
                key_data=key_data,
                layer_idx=layer_idx,
                query=query_squeezed
            )
            nvtx.range_pop()    
        else:
            # 【安全检查】Offload + USE_RABITQ_RERANK=False 会导致 key_data=None
            if self.enable_offload:
                raise RuntimeError(
                    "Offload 模式下 USE_RABITQ_RERANK=False 不被支持！"
                    "key_data_gpu 在 CPU 上，exact rerank 需要从 CPU 获取数据。"
                    "请使用 USE_RABITQ_RERANK=True 或关闭 offload。"
                )
            raise ValueError("enable_rerank=True 但未提供 key_data 或 key_block_weight")

        if getattr(self, '_allayer_recall_check', False):
            self._allayer_recall_ablation(
                topk_indices, query_squeezed, layer_idx, tag="Final",
            )
            self._allayer_recall_ablation(
                coarse_indices, query_squeezed, layer_idx, tag="CoarsePool",
            )

        return topk_indices
    
    

    
    def get_topk_clusters(self, q_blocks, codebook, collision_ratio, layer_idx):
        """
        Step 2: 选择前collision_ratio范围内的clusters(K_r=1 优化版本)
     # 6级权重策略（高碰撞优先，top 档位占比少）- 最佳配置 Recall=75.6%
        # Tier 1: 0 ~ 5% -> 在 tier 1~6 中命中 -> weight +6
        # Tier 2: 5 ~ 15% -> 在 tier 2~6 中命中 -> weight +5
        # Tier 3: 15 ~ 30% -> 在 tier 3~6 中命中 -> weight +4
        # Tier 4: 30 ~ 50% -> 在 tier 4~6 中命中 -> weight +3
        # Tier 5: 50 ~ 75% -> 在 tier 5~6 中命中 -> weight +2
        # Tier 6: 75 ~ 100% -> 只在 tier 6 中命中 -> weight +1
        Args:
            scores: [bs, kv_heads, B, K_omega] 每个cluster的得分
            codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
            collision_ratio: 每个子空间激活的keys占比
            layer_idx: 当前层索引(用于调试)
        Returns:
            sorted_cluster_ids: [bs, kv_heads, B, K_omega] 按得分排序的cluster IDs
            tier_cluster_cnts: tuple of 6 tensors [bs, kv_heads, B]，分别为6个层级的cluster数量
        """
        kv_len = codebook.shape[3]
        # 按得分排序clusters(降序)
        scores = torch.matmul(q_blocks, self._V_omega_gpu_T)  # [bs,kv,B,K_omega]
        sorted_cluster_ids = scores.argsort(dim=-1, descending=True)
        # 优化:直接使用prefill阶段预计算的cluster_key_counts(避免每次decode重新one_hot计算)
        cluster_key_counts = self.cluster_key_counts_gpu[layer_idx]  # [bs, kv_heads, B, num_clusters]
        # 根据sorted_cluster_ids重排cluster_key_counts
        sorted_key_counts = torch.gather(cluster_key_counts, dim=-1, index=sorted_cluster_ids.long())
        cumsum_counts = torch.cumsum(sorted_key_counts, dim=-1)  # [bs, kv_heads, B, num_clusters]
        before_cumsum = cumsum_counts - sorted_key_counts  # [bs, kv_heads, B, num_clusters]
        
        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        tier_cluster_cnts = []
        
        for ratio in tier_ratios:
            target_keys = math.ceil(kv_len * collision_ratio * ratio)
            valid_mask = before_cumsum < target_keys
            tier_cluster_cnts.append(valid_mask.sum(dim=-1))  # [bs, kv_heads, B] int64
        
        return sorted_cluster_ids, tuple(tier_cluster_cnts)
    
    
    
    
    def update_cache_cnt(self, sorted_cluster_ids, tier_cluster_cnts, codebook):
        """
        Step 3: 更新每个cache的计数器(碰撞次数)- 使用封装好的 CUDA 版本
        使用6级权重策略：
            - Tier 1 (0~5% of candidate): weight +6
            - Tier 2 (0~15% of candidate): weight +5
            - Tier 3 (0~30% of candidate): weight +4
            - Tier 4 (0~50% of candidate): weight +3
            - Tier 5 (0~75% of candidate): weight +2
            - Tier 6 (0~100% of candidate): weight +1
        Args:
            sorted_cluster_ids: [bs, kv_heads, B, num_clusters] 排序后的cluster IDs
            tier_cluster_cnts: tuple of 6 tensors [bs, kv_heads, B]，分别为6个层级的cluster数量
                - 当前 CUDA 实现只支持单个 topk_cluster_cnt，这里默认取最后一档(100%)
            codebook: [bs, kv_heads, B, kv_len] 每个key在每个子空间的cluster ID
        Returns:
            cache_cnt: [bs, kv_heads, kv_len] 每个key的碰撞次数(累积所有子空间)
        """
        if isinstance(tier_cluster_cnts, (list, tuple)):
            if len(tier_cluster_cnts) == 0:
                raise ValueError("tier_cluster_cnts 不能为空")
            topk_cluster_cnt = tier_cluster_cnts[-1]
        else:
            topk_cluster_cnt = tier_cluster_cnts
        return self._update_cache_cnt_cuda(
            sorted_cluster_ids.contiguous(),
            topk_cluster_cnt.contiguous(),
            codebook.contiguous(),
        )



    def get_cache_cnt_fused(self, q_blocks, codebook, collision_ratio, layer_idx):
        """
        融合版本: 将 get_topk_clusters + update_cache_cnt 合并为单次 CUDA 调用
        流程:
            1. 计算 query-cluster scores 并排序
            2. 获取预计算的 cluster_key_counts
            3. 计算 tier_counts 阈值
            4. 调用融合的 CUDA kernel (内部做 gather + cumsum + collision counting)
        Args:
            q_blocks: [bs, kv_heads, B, m] query 在各子空间的编码
            codebook: [bs, kv_heads, B, kv_len] 每个 key 的 cluster ID (uint8)
            collision_ratio: 碰撞检测范围比例
            layer_idx: 当前层索引
        Returns:
            cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
        """
        kv_len = codebook.shape[3]
        
        # Step 1: 计算 scores 并排序得到 sorted_cluster_ids
        scores = torch.matmul(q_blocks, self._V_omega_gpu_T)  # [bs, kv_heads, B, K_omega]
        sorted_cluster_ids = scores.argsort(dim=-1, descending=True).to(torch.int64)
        
        # Step 2: 获取预计算的 cluster_key_counts
        cluster_key_counts = self.cluster_key_counts_gpu[layer_idx]  # [bs, kv_heads, B, num_clusters]
        
        # Step 3: 计算各 tier 的 key 数量阈值
        tier_ratios = [0.05, 0.15, 0.30, 0.50, 0.75, 1.0]
        tier_counts = [int(math.ceil(kv_len * collision_ratio * ratio)) for ratio in tier_ratios]
        
        # Step 4: 调用融合的 CUDA kernel
        # codebook: [bs, kv_heads, B, kv_len] - 不需要 transpose
        cache_cnt = self._update_cache_cnt_fused(
            sorted_cluster_ids,
            cluster_key_counts,
            codebook,  # [bs, kv_heads, B, kv_len] uint8
            tier_counts
        )
        
        return cache_cnt


    
    def _coarse_filter_truncate(self, cache_cnt, candidate_ratio, layer_idx=None):
        """
        bs打到14，空隙可能消失，gpu利用率可能会升高
        简化版碰撞粗筛：直接按碰撞计数从高到低排序，取前 candidate_len 个
        Args:
            cache_cnt: [bs, kv_heads, kv_len] 碰撞计数 (值域 [0, 96])
            candidate_ratio: 候选集比例
            layer_idx: 当前层索引
        Returns:
            coarse_indices: [bs, kv_heads, candidate_len] 粗筛后的候选索引
            candidate_len: int - 候选数量
        """
        kv_len = cache_cnt.shape[-1]
        candidate_len = max(1, int(math.ceil(kv_len * candidate_ratio)))
        coarse_indices = self._radix_topk_ext.radix_topk_3d(cache_cnt, candidate_len)    
        return coarse_indices, candidate_len
    
    
    

    def get_candidate_cache_with_rabitq_rerank(
        self, coarse_indices, q_blocks, key_4bit, key_block_weight,
        q_norm, key_data=None, layer_idx=None, query=None
    ):
        """
        4-bit RaBitQ 近似内积精排（合并 1-bit 符号 + 3-bit 幅值）
        使用融合 CUDA kernel 加速 Step 1-3
        Args:
            coarse_indices: [bs, kv_heads, candidate_len] 粗筛后的候选索引
            q_blocks: [bs, kv_heads, B, m] query 的 SRHT 分块结果
            key_4bit: [bs, kv_heads, B, kv_len, m//2] 4-bit packed (1-bit sign + 3-bit magnitude)
            key_block_weight: [bs, kv_heads, B, kv_len] bf16, 已融合 (r/α) * ||k||
            q_norm: [bs, kv_heads, 1] query 的模长 ||q||
            key_data: [bs, kv_heads, kv_len, head_dim] 可选，用于调试对比
            query: [bs, kv_heads, head_dim] 原始 query（消融实验需要）
        """
        # 1-3. 融合 CUDA kernel: Gather + Unpack + Score
        # 目前 CUDA kernel 只支持 B=16, m=8，其他情况回退到 PyTorch 实现
        if self.polar_B == 16 and self.polar_m == 8:
            nvtx.range_push("rerank_qk_score")
            approx_scores, v4b_all = partial_rerank_cuda(
                coarse_indices.contiguous(),
                key_4bit,
                key_block_weight,
                q_blocks.contiguous(),
                q_norm.contiguous(),
                self._mag_centers.contiguous(),
                self.polar_B,
                self.polar_m,
                kernel=self._rerank_kernel  # 使用预加载的 kernel
            )
            nvtx.range_pop()
            
        else:
            # PyTorch 回退实现
            # nvtx.range_push("rerank_1_gather")
            idx_code = coarse_indices.unsqueeze(2).expand(-1, -1, self.polar_B, -1)
            cand_weight = torch.gather(key_block_weight, dim=3, index=idx_code)
            idx_4bit = coarse_indices.unsqueeze(2).unsqueeze(-1).expand(-1, -1, self.polar_B, -1, key_4bit.shape[-1])
            cand_4bit = torch.gather(key_4bit, dim=3, index=idx_4bit)
            # nvtx.range_pop()
            
            # nvtx.range_push("rerank_2_unpack_reconstruct")
            orig_shape = cand_4bit.shape[:-1]
            s_sign, t_codes = self._unpack_4bit_codes(cand_4bit.view(-1, cand_4bit.shape[-1]), m=self.polar_m)
            s_sign = s_sign.view(*orig_shape, self.polar_m)
            t_codes = t_codes.view(*orig_shape, self.polar_m)
            a_vals = self._mag_centers[t_codes]
            # nvtx.range_pop()
            
            # nvtx.range_push("rerank_3_score")
            v4b = a_vals * s_sign.to(a_vals.dtype)
            q_exp = q_blocks.unsqueeze(3)
            per_block_scores = (q_exp * v4b).sum(dim=-1)
            cosine_proxy = (per_block_scores * cand_weight).sum(dim=2)
            approx_scores = cosine_proxy * q_norm
            # nvtx.range_pop()
               
        # 4. Rerank (topk + gather)
        nvtx.range_push("topk_gather")
        actual_topk = min(self.final_topk, approx_scores.shape[-1])
        _, rerank_indices = torch.topk(approx_scores, k=actual_topk, dim=-1)
        final_indices = torch.gather(coarse_indices, dim=2, index=rerank_indices)
        nvtx.range_pop()
        
        #  消融实验：RaBitQ rerank 后 Recall@100（与全局精确 top-100 对比）
        # if self.enable_rerank_recall_ablation and key_data is not None and query is not None:
        #     self._run_rerank_recall_ablation(
        #         query=query,
        #         key_data=key_data,
        #         topk_indices=final_indices,
        #         layer_idx=layer_idx
        #     )

        # if self.enable_rerank_recall_ablation and query is not None:
        #     self._run_mass_ablation(
        #         query=query, key_data=key_data,
        #         approx_scores=approx_scores, coarse_indices=coarse_indices,
        #         rerank_indices=rerank_indices, layer_idx=layer_idx
        #     )

        return final_indices


    def _run_rerank_ablation(self, query, key_data, coarse_indices, approx_scores, 
                              final_indices, layer_idx):
        """
        Rerank 消融实验：对比 RaBitQ 近似内积 vs 精确内积（使用旋转后的向量）
        
        思路：
        - RaBitQ 估计的是旋转后分块向量的内积
        - 所以精确内积也应该用旋转后的 query 和 key 来算，才公平
        
        指标：
        1. 绝对误差：|approx_score - exact_score| 的平均值
        2. 召回率：RaBitQ top-K 与精确内积 top-K 的交集比例
        
        Args:
            query: [bs, kv_heads, head_dim] 原始 query
            key_data: [bs, kv_heads, kv_len, head_dim] 所有 key 数据
            coarse_indices: [bs, kv_heads, num_candidates] 粗排候选集索引
            approx_scores: [bs, kv_heads, num_candidates] RaBitQ 近似得分
            final_indices: [bs, kv_heads, final_topk] RaBitQ rerank 的最终索引
            layer_idx: 层索引
        """
        # 初始化统计变量（全局统计：所有层、所有 head）
        if not hasattr(self, '_ablation_stats'):
            self._ablation_stats = {
                'step_count': 0,
                'total_recall_sum': 0.0,      # 所有层所有 head 的 recall 累加
                'total_rel_err_sum': 0.0,     # 所有层所有 head 的相对误差累加
                'total_head_count': 0,        # 累计的 (layer, head) 次数
                'current_step_recalls': [],   # 当前 step 的所有 head recall
                'current_step_rel_errs': [],  # 当前 step 的所有 head 相对误差
            }
        
        # 统计所有层（Layer 0 时递增 step，清空当前 step 统计）
        if layer_idx == 0:
            self._ablation_stats['step_count'] += 1
            self._ablation_stats['current_step_recalls'] = []
            self._ablation_stats['current_step_rel_errs'] = []
        
        step = self._ablation_stats['step_count']
        
        with torch.no_grad():
            bs, kv_heads, num_candidates = coarse_indices.shape
            head_dim = query.shape[-1]
            
            # 候选集太小时不统计（没有说服力）
            MIN_CANDIDATES_FOR_ABLATION = 500
            if num_candidates < MIN_CANDIDATES_FOR_ABLATION:
                return  # 跳过统计
            
            # 1. Gather 候选集的 key 数据
            idx_expanded = coarse_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            candidate_keys = torch.gather(key_data, dim=2, index=idx_expanded)
            
            # 2. 对 query 和 candidate_keys 进行相同的 SRHT 旋转
            q_rotated = query.clone()
            for round_idx in range(DEFAULT_SRHT_ROUNDS):
                q_rotated = q_rotated * self.quantizer.srht_diagonal_signs[round_idx]
                q_rotated = hadamard_transform(q_rotated, scale=self._hadamard_scale)
            
            k_rotated = candidate_keys.clone()
            for round_idx in range(DEFAULT_SRHT_ROUNDS):
                k_rotated = k_rotated * self.quantizer.srht_diagonal_signs[round_idx]
                k_rotated = hadamard_transform(k_rotated, scale=self._hadamard_scale)
            
            # 3. 计算精确内积
            q_expanded = q_rotated.unsqueeze(2)
            exact_scores = (q_expanded * k_rotated).sum(dim=-1)  # [bs, kv_heads, num_candidates]
            
            # 4. 计算精确内积的 top-K
            # 保护：候选集可能小于 final_topk
            actual_topk = min(self.final_topk, num_candidates)
            _, exact_topk_indices = torch.topk(exact_scores, k=actual_topk, dim=-1)
            exact_final_indices = torch.gather(coarse_indices, dim=2, index=exact_topk_indices)
            
            # 5. 计算每个 head 的 Recall 和相对误差
            for b in range(bs):
                for h in range(kv_heads):
                    # Recall
                    rabitq_set = set(final_indices[b, h].cpu().tolist())
                    exact_set = set(exact_final_indices[b, h].cpu().tolist())
                    h_recall = len(rabitq_set & exact_set) / actual_topk
                    
                    # 相对误差 = MAE / score_range
                    h_approx = approx_scores[b, h]
                    h_exact = exact_scores[b, h]
                    h_mae = (h_approx - h_exact).abs().mean().item()
                    score_range = h_exact.max().item() - h_exact.min().item()
                    h_rel_err = h_mae / score_range if score_range > 1e-6 else 0.0
                    
                    # 累积到当前 step
                    self._ablation_stats['current_step_recalls'].append(h_recall)
                    self._ablation_stats['current_step_rel_errs'].append(h_rel_err)
                    
                    # 累积到全局
                    self._ablation_stats['total_recall_sum'] += h_recall
                    self._ablation_stats['total_rel_err_sum'] += h_rel_err
                    self._ablation_stats['total_head_count'] += 1
        
        # 最后一层时打印汇总（假设有 36 层，最后一层 idx=35）
        # 这里用一个简单的判断：当前 step 已经收集了足够多的 head 数据时打印
        num_layers = 36  # Qwen3-4B 有 36 层 (num_hidden_layers = 36)
        expected_heads_per_step = num_layers * kv_heads * bs
        
        if len(self._ablation_stats['current_step_recalls']) >= expected_heads_per_step:
            alpha_status = "WITH α correction" if self.enable_alpha_correction else "WITHOUT α correction"
            
            # 当前 step 的统计
            step_recalls = self._ablation_stats['current_step_recalls']
            step_rel_errs = self._ablation_stats['current_step_rel_errs']
            step_avg_recall = sum(step_recalls) / len(step_recalls)
            step_avg_rel_err = sum(step_rel_errs) / len(step_rel_errs)
            
            # 全局累积统计
            global_avg_recall = self._ablation_stats['total_recall_sum'] / self._ablation_stats['total_head_count']
            global_avg_rel_err = self._ablation_stats['total_rel_err_sum'] / self._ablation_stats['total_head_count']
            
            print(f"\n{'='*80}")
            print(f"🔬 [Rerank Ablation - Step {step}] {alpha_status}")
            print(f"  参数: candidate_ratio={self.candidate_ratio}, 候选集大小={num_candidates}, final_topk={self.final_topk}")
            print(f"  统计范围: 所有 {num_layers} 层 × {kv_heads} heads = {expected_heads_per_step} 个 (layer, head) 组合")
            print(f"  ├─ 当前 Step 平均 Recall@{self.final_topk}: {step_avg_recall*100:.2f}%")
            print(f"  ├─ 当前 Step 平均相对误差: {step_avg_rel_err*100:.4f}%")
            print(f"  ├─ 累积平均 Recall@{self.final_topk}: {global_avg_recall*100:.2f}%")
            print(f"  └─ 累积平均相对误差: {global_avg_rel_err*100:.4f}%")
            print(f"{'='*80}\n")


    def _allayer_recall_ablation(
        self,
        indices: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        tag: str = "Final",
    ):
        """Accumulate recall across all layers/heads, print average per step."""
        if layer_idx == 0 and tag == "Final":
            self._allayer_recall_step += 1
            self._allayer_recall_accum = {}
        if self._allayer_recall_step % self._allayer_recall_log_every != 0:
            return

        K = self._allayer_recall_k
        with torch.no_grad():
            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                cpu_len = int(self.cpu_valid_end[layer_idx])
                if cpu_len < K:
                    return
                keys_layer = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
                    self.device, non_blocking=True
                ).to(torch.float32)
                kv_len_eff = cpu_len
            else:
                return

            bs, kv_heads, _ = indices.shape
            q_fp32 = raw_q.to(torch.float32)

            if tag not in self._allayer_recall_accum:
                self._allayer_recall_accum[tag] = [0, 0, 0]  # [hit_sum, count, kv_len]

            for h in range(kv_heads):
                q_h = q_fp32[0, h]
                k_h = keys_layer[0, h, :kv_len_eff, :]
                exact_scores = torch.matmul(k_h, q_h)

                cand_set = set(indices[0, h].cpu().tolist())
                _, true_topK = torch.topk(exact_scores, k=K, dim=-1)
                true_set = set(true_topK.cpu().tolist())
                self._allayer_recall_accum[tag][0] += len(cand_set & true_set)
                self._allayer_recall_accum[tag][1] += K
                self._allayer_recall_accum[tag][2] = kv_len_eff

            if layer_idx == self.layer_num - 1 and tag == "CoarsePool":
                self._print_allayer_recall_summary()
                self._allayer_recall_accum = {}

    def _print_allayer_recall_summary(self):
        parts = []
        kv_len = 0
        for tag in ("Final", "CoarsePool"):
            if tag not in self._allayer_recall_accum:
                continue
            hit, total, kv = self._allayer_recall_accum[tag]
            recall = hit / total * 100 if total > 0 else 0
            parts.append(f"{tag} R@{self._allayer_recall_k}={recall:.1f}%")
            kv_len = max(kv_len, kv)
        if parts:
            print(
                f"[Pipeline Recall] step={self._allayer_recall_step} kv={kv_len} "
                f"(avg all layers/heads) | " + " | ".join(parts),
                flush=True,
            )


    def _run_coarse_recall_ablation(
        self,
        query,
        key_data,
        cache_cnt,
        layer_idx,
        candidate_ratio,
        coarse_indices=None,
    ):
        """
        Stage-1 SUCO collision Recall 消融实验（两个指标同时统计）
        共用变量：
            - candidate_set := topk(cache_cnt, candidate_num)   ← 第一阶段碰撞粗筛
            - exact_scores  := <q, k_i>  for i in 1..kv_len     ← 全量 retrieval key 精确 QK 内积
        指标 1 (与 user 指定 K=100 比，衡量是否保留 “真实 top-K”):
            Recall@K   := |candidate_set ∩ topk(exact_scores, K)| / K
        指标 2 (与同等大小 candidate_num 比，衡量碰撞排序与精确排序是否一致）：
            Recall@C   := |candidate_set ∩ topk(exact_scores, candidate_num)| / candidate_num
        打印：仅按 POLAR_RECALL_ABLATION_LOG_EVERY 抽样，每行「step + 各 L{l}H{h} 的 R@K / R@C」。

        Args:
            query: [bs, kv_heads, head_dim] 原始 query (mean over GQA group)
            key_data: [bs, kv_heads, kv_len, head_dim] non-offload 模式下的 retrieval keys
            cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
            layer_idx: 当前层索引
            candidate_ratio: 候选集比例 (candidate_num = ceil(kv_len * candidate_ratio))
            coarse_indices: [bs, kv_heads, candidate_num] 实际生产粗筛候选；传入时避免
                因重新 torch.topk(cache_cnt) 的 tie-breaking 与生产路径不一致。
        """
        # ------- 配置（首次调用时解析环境变量并缓存） -------
        if not hasattr(self, '_first_stage_recall_cfg'):
            def _parse_int_list(env_name, default):
                raw = os.environ.get(env_name)
                if not raw:
                    return tuple(default)
                try:
                    return tuple(int(x.strip()) for x in raw.split(',') if x.strip() != '')
                except ValueError:
                    print(f"[FirstStage Recall] WARN: 无法解析 {env_name}={raw!r}，使用默认 {default}", flush=True)
                    return tuple(default)
            self._first_stage_recall_cfg = {
                'target_layers': set(_parse_int_list('POLAR_RECALL_ABLATION_LAYERS', [4, 17, 30])),
                'target_heads':  tuple(_parse_int_list('POLAR_RECALL_ABLATION_HEADS', [0, 4])),
                'K':             int(os.environ.get('POLAR_RECALL_ABLATION_K', '100')),
                'log_every':     max(1, int(os.environ.get('POLAR_RECALL_ABLATION_LOG_EVERY', '50'))),
            }

        cfg = self._first_stage_recall_cfg
        target_layers = cfg['target_layers']
        if layer_idx not in target_layers:
            return
        target_heads = cfg['target_heads']
        K = cfg['K']

        # ------- 统计容器 -------
        if not hasattr(self, '_first_stage_recall_stats'):
            self._first_stage_recall_stats = {
                'step_count': 0,
                # 缓存当前 step 内、各目标层的 per-head 结果（最后一个 target layer 时打一行）
                'current_step_layer_recalls': {},   # layer -> [(head, recall_K, recall_C), ...]
            }
        stats = self._first_stage_recall_stats

        # 在最早的 target layer 处递增 step_count（一次 decode = 所有层各调用一次本函数）
        first_target_layer = min(target_layers)
        last_target_layer  = max(target_layers)
        if layer_idx == first_target_layer:
            stats['step_count'] += 1
            stats['current_step_layer_recalls'].clear()
        step = stats['step_count']

        with torch.no_grad():
            bs, kv_heads, kv_len = cache_cnt.shape
            # 候选集大小（与 _coarse_filter_truncate 完全一致）
            candidate_num = max(1, int(math.ceil(kv_len * candidate_ratio)))
            candidate_num = min(candidate_num, kv_len)

            # 真实 top-K 必须存在于 retrieval zone 中
            if kv_len < K or candidate_num < 1:
                return

            # ------- 取 retrieval keys（offload-aware）-------
            # offload 模式下 unified_keys_gpu 只含 sink+local+buffer，必须从 pinned CPU 取
            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                cpu_len = int(self.cpu_valid_end[layer_idx])
                if cpu_len < K:
                    return
                # 整层一次性 D2H→GPU 转 fp32（与 _run_*_ablation 保持一致：CPU compute 太慢，用 GPU）
                keys_layer = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(self.device, non_blocking=True).to(torch.float32)
                kv_len_eff = cpu_len
            else:
                # 非 offload：直接复用调用方传入的 GPU 切片
                if key_data is None:
                    return
                keys_layer = key_data.to(torch.float32)
                kv_len_eff = key_data.shape[2]

            # ------- 第一阶段候选集（topk by collision count）-------
            # 优先复用生产路径 _coarse_filter_truncate 的输出，避免 cache_cnt 分数相同
            # 时重新 torch.topk 产生不同 tie-breaking。
            if coarse_indices is not None:
                candidate_num = min(candidate_num, int(coarse_indices.shape[-1]))
                coarse_indices_all = coarse_indices[:, :, :candidate_num]
            else:
                _, coarse_indices_all = torch.topk(cache_cnt, k=candidate_num, dim=-1)

            # ------- 逐 head 计算 -------
            b = 0
            per_head_results = []  # [(head, recall_K, recall_C), ...]
            for h in target_heads:
                if h >= kv_heads:
                    continue

                # 精确 QK 一次性计算（两个 topk 共享）
                q_h = query[b, h].to(torch.float32)              # [d]
                k_h = keys_layer[b, h, :kv_len_eff, :]           # [kv_len_eff, d]
                exact_scores = torch.matmul(k_h, q_h)            # [kv_len_eff]

                # true top-K（默认 K=100）
                _, true_topK_indices = torch.topk(exact_scores, k=K, dim=-1)
                true_top_K_set = set(true_topK_indices.cpu().tolist())

                # true top-candidate_num（与候选集等大）
                _, true_topC_indices = torch.topk(exact_scores, k=candidate_num, dim=-1)
                true_top_C_set = set(true_topC_indices.cpu().tolist())

                cand_set = set(coarse_indices_all[b, h].cpu().tolist())
                recall_K = len(cand_set & true_top_K_set) / K
                recall_C = len(cand_set & true_top_C_set) / candidate_num
                per_head_results.append((h, recall_K, recall_C))

            stats['current_step_layer_recalls'][layer_idx] = per_head_results

            # 仅在最后一个 target layer、且按 log_every 抽样打一行：
            # 「step … K … | 各 L{l}H{h} R@K / R@C」无 global / kv_len / 累积块
            if layer_idx == last_target_layer and step % cfg['log_every'] == 0:
                parts = []
                for l in sorted(stats['current_step_layer_recalls'].keys()):
                    for h, rK, rC in stats['current_step_layer_recalls'][l]:
                        parts.append(f"L{l}H{h} R@K={rK*100:.1f}% R@C={rC*100:.1f}%")
                print(f"[Stage1] step={step} K={K} cand={candidate_num} " + " | ".join(parts),flush=True)









    def _run_rerank_recall_ablation(self, query, key_data, topk_indices, layer_idx):
        """
        PolarANN 精排 Recall@100 消融实验（固定 K=100）
        简化版：只计算 Layer 4, Head 4，每步立即打印
        
        Args:
            query: [bs, kv_heads, head_dim] 原始 query
            key_data: [bs, kv_heads, kv_len, head_dim] retrieval zone 的所有 key
            topk_indices: [bs, kv_heads, final_topk] 精排后的全局索引
            layer_idx: 层索引
        """
        # 只在 Layer 4 时计算和打印
        TARGET_LAYER = 4
        TARGET_HEAD = 4
        
        if layer_idx != TARGET_LAYER:
            return
        
        # 初始化统计变量
        if not hasattr(self, '_rerank_ablation_stats'):
            self._rerank_ablation_stats = {
                'step_count': 0,
                'total_recall_sum': 0.0,
                'total_head_count': 0,
            }
        
        self._rerank_ablation_stats['step_count'] += 1
        step = self._rerank_ablation_stats['step_count']
        FIXED_TOPK = 100  # 固定 K=100
        
        with torch.no_grad():
            # Offload 模式下 key_data 可能来自 unified_keys_gpu 的切片而被截断（只包含 local/buffer），
            # 这里用 cpu_valid_end 作为真实 kv_len，并在 CPU 上计算精确 top-100，保证 recall 统计可信。
            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                bs = topk_indices.shape[0]
                kv_heads = topk_indices.shape[1]
                kv_len = int(self.cpu_valid_end[layer_idx])
                head_dim = query.shape[-1]
            else:
                bs, kv_heads, kv_len, head_dim = key_data.shape
            
            # kv_len 太小时跳过
            if kv_len < FIXED_TOPK:
                print(f"[Rerank] Step {step}: kv_len={kv_len} < 100, skip")
                return
            
            # 只计算 head 4
            h = min(TARGET_HEAD, kv_heads - 1)
            b = 0
            
            # 计算全局精确 QK 内积的 Top-100（只针对目标 head）
            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                cpu_len = kv_len
                keys_cpu = self.unified_keys_cpu[layer_idx][b, h, :cpu_len, :].to(torch.float32)  # [cpu_len, d]
                q_cpu = query[b, h].detach().to('cpu').to(torch.float32)  # [d]
                exact_scores = torch.matmul(keys_cpu, q_cpu)  # [cpu_len]
                _, exact_top100_indices = torch.topk(exact_scores, k=FIXED_TOPK, dim=-1)  # [100]
            else:
                exact_scores = torch.einsum('d,kd->k', query[b, h], key_data[b, h])  # [kv_len]
                _, exact_top100_indices = torch.topk(exact_scores, k=FIXED_TOPK, dim=-1)  # [100]
            
            # 精排后的 top-100 索引
            num_reranked = min(topk_indices.shape[2], FIXED_TOPK)
            rerank_set = set(topk_indices[b, h, :num_reranked].cpu().tolist())
            exact_set = set(exact_top100_indices.cpu().tolist())
            
            # Recall@100 = |精排 top-100 ∩ 精确 top-100| / 100
            h_recall = len(rerank_set & exact_set) / FIXED_TOPK
            self._rerank_ablation_stats['total_recall_sum'] += h_recall
            self._rerank_ablation_stats['total_head_count'] += 1
            global_avg_recall = self._rerank_ablation_stats['total_recall_sum'] / self._rerank_ablation_stats['total_head_count']
            if step == 1:
                print(f"[Rerank Ablation] Layer={TARGET_LAYER}, Head={TARGET_HEAD}, collision_ratio={self.collision_ratio}, candidate_ratio={self.candidate_ratio}, final_topk={self.final_topk}", flush=True)            
            print(f"[Rerank] Step {step}: kv_len={kv_len}, Recall@100={h_recall*100:.2f}%", flush=True)




    def _run_mass_ablation(self, query, key_data, approx_scores, coarse_indices,
                           rerank_indices, layer_idx):
        """
        计算所有层所有 head 的平均 cand_covered_mass 和 approx_k_mass。

        每层调用一次，在最后一层汇总并打印该 step 的全局平均。
        - cand_covered_mass: 碰撞粗排选出的所有 candidate 在全量精确 attention 下的权重之和
        - approx_k_mass: rerank 固定 top-K (final_topk) 在全量精确 attention 下的权重之和
        """
        if not hasattr(self, '_mass_ablation_step'):
            self._mass_ablation_step = 0
            self._mass_buf_cand = []
            self._mass_buf_approx = []

        if layer_idx == 0:
            self._mass_ablation_step += 1
            self._mass_buf_cand.clear()
            self._mass_buf_approx.clear()

        with torch.no_grad():
            _scale = 1.0 / math.sqrt(self.head_dim)
            b = 0

            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                cpu_len = int(self.cpu_valid_end[layer_idx])
                all_keys = self.unified_keys_cpu[layer_idx][b:b+1, :, :cpu_len, :]
                q_cpu = query[b:b+1].detach().cpu().float()
                full_scores = torch.einsum('bhd,bhkd->bhk', q_cpu, all_keys.float())
            elif key_data is not None:
                full_scores = torch.einsum('bhd,bhkd->bhk', query[b:b+1].float(), key_data[b:b+1].float())
            else:
                return

            full_attn = torch.softmax(full_scores * _scale, dim=-1)
            kv_heads = full_attn.shape[1]
            dev = full_attn.device

            for h in range(kv_heads):
                coarse_idx_h = coarse_indices[b, h].long().to(dev)
                self._mass_buf_cand.append(full_attn[0, h][coarse_idx_h].sum().item())

                rerank_local = rerank_indices[b, h].long().to(dev)
                rerank_global = coarse_idx_h[rerank_local]
                self._mass_buf_approx.append(full_attn[0, h][rerank_global].sum().item())

        if layer_idx == self.layer_num - 1:
            step = self._mass_ablation_step
            n = len(self._mass_buf_cand)
            avg_cand = sum(self._mass_buf_cand) / max(n, 1)
            avg_approx = sum(self._mass_buf_approx) / max(n, 1)
            if step == 1:
                print(f"[Mass Ablation] all layers × {kv_heads} heads, "
                      f"final_topk={rerank_indices.shape[-1]}", flush=True)
            print(f"[Mass] Step {step}: avg_cand_mass={avg_cand:.4f}, "
                  f"avg_approx_k_mass={avg_approx:.4f} "
                  f"(n={n} layer×head pairs)", flush=True)

    def _compute_recall_at_k(self, query_avg_gpu, key_data_gpu, topk_indices_batch, layer_idx, valid_length, bs):
        """
        Compute Recall@K for multiple layers and all heads (简化版：只打印平均 recall)
        每200步计算一次
        """
        # Initialize tracking variables
        if not hasattr(self, '_layer_recalls'):
            self._layer_recalls = {}  # 存储当前 step 所有层的 recall
            self._all_step_recalls = []  # 累积所有 step 的平均 recall
        
        # 只在 layer 0 时重置当前 step 的数据
        if layer_idx == 0:
            self._layer_recalls = {}
        
        step = self._recall_step_counter
        actual_topk = topk_indices_batch.shape[-1]
        
        # 调试：检查数据一致性
        key_data_len = key_data_gpu.shape[2]
        if key_data_len != valid_length:
            # print(f"[Recall DEBUG] key_data_gpu.shape[2]={key_data_len} != valid_length={valid_length}")
            pass
        
        # 检查索引范围
        max_idx = topk_indices_batch.max().item()
        if max_idx >= key_data_len:
            # print(f"[Recall DEBUG] topk_indices max={max_idx} >= key_data_len={key_data_len}, 索引越界!")
            pass
        
        # Compute recall
        with torch.no_grad():
            similarities = torch.bmm(
                query_avg_gpu.view(bs * self.kv_head, 1, self.head_dim),
                key_data_gpu.view(bs * self.kv_head, key_data_len, self.head_dim).transpose(1, 2)
            )
            similarities = similarities.view(bs, self.kv_head, key_data_len)
            _, ground_truth_indices = torch.topk(similarities, k=actual_topk, dim=-1)
            
            # 计算所有 head 的 recall
            head_recalls = []
            for head_idx in range(self.kv_head):
                ann_indices = topk_indices_batch[0, head_idx]
                gt_indices = ground_truth_indices[0, head_idx]
                intersection_mask = torch.isin(ann_indices, gt_indices)
                recall = intersection_mask.sum().item() / actual_topk
                head_recalls.append(recall)
            
            avg_recall = sum(head_recalls) / len(head_recalls)
            self._layer_recalls[layer_idx] = avg_recall
            
            
            # 在最后一层统计并打印
            if layer_idx == self.layer_num - 1:
                overall_avg = sum(self._layer_recalls.values()) / len(self._layer_recalls) if self._layer_recalls else 0
                self._all_step_recalls.append(overall_avg)
                # 存储已打印的 recall 用于最后计算平均
                if hasattr(self, '_printed_recalls'):
                    self._printed_recalls.append(overall_avg)
   
 

    
    def sync(self, layer_idx, batch_idx):
        """
        GPU版本:数据已经在prefill阶段直接存储到GPU,无需额外同步
        """
        pass  # GPU版本中所有数据已在prefill时存储到GPU,无需额外操作

    def _coarse_filter(self, cache_cnt, candidate_ratio):
        """
        碰撞粗筛：基于碰撞阈值选出所有 >= 阈值的 key（公共逻辑）
        Args:
            cache_cnt: [bs, kv_heads, kv_len] 碰撞计数
            candidate_ratio: 候选集比例
        Returns:
            coarse_indices: [bs, kv_heads, max_candidates] 粗筛后的候选索引
            max_candidates: int - 实际候选数量
        """
        kv_len = cache_cnt.shape[-1]
        candidate_len = max(1, math.ceil(kv_len * candidate_ratio))
        candidate_len = min(candidate_len, kv_len)
        candidate_len = max(candidate_len, self.final_topk)
        
        vals, idx = torch.topk(cache_cnt, k=candidate_len, dim=-1)
        thr = vals[..., -1]
        
        thr_count = (cache_cnt == thr.unsqueeze(-1)).sum(dim=-1)
        gt_count = (cache_cnt > thr.unsqueeze(-1)).sum(dim=-1)
        ge_count = gt_count + thr_count
        max_candidates = min(int(ge_count.max().item()), kv_len)
        
        if max_candidates > candidate_len:
            _, coarse_indices = torch.topk(cache_cnt, k=max_candidates, dim=-1)
        else:
            coarse_indices = idx
            max_candidates = candidate_len
        return coarse_indices, max_candidates





    def get_candidate_cache_with_rerank(self, cache_cnt, candidate_ratio, query, key_data, layer_idx=None):
        """精确内积精排"""
        # coarse_indices, max_candidates = self._coarse_filter(cache_cnt, candidate_ratio)
        coarse_indices, max_candidates = self._coarse_filter_truncate(cache_cnt, candidate_ratio, layer_idx)

        if not hasattr(self, '_exact_rerank_candidate_printed'):
            # print(f"[精确rerank] max_candidates = {max_candidates}, final_topk = {self.final_topk}")
            self._exact_rerank_candidate_printed = True
        
        if max_candidates <= self.final_topk:
            return coarse_indices[:, :, :self.final_topk]
        
        # 精确内积 rerank
        head_dim = key_data.shape[-1]
        coarse_indices_exp = coarse_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        candidate_keys = torch.gather(key_data, dim=2, index=coarse_indices_exp)
        exact_scores = torch.einsum('bhd,bhcd->bhc', query, candidate_keys)
        
        final_k = min(self.final_topk, max_candidates)
        _, rerank_indices = torch.topk(exact_scores, k=final_k, dim=-1)
        return torch.gather(coarse_indices, dim=2, index=rerank_indices)
