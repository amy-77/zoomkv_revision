"""Polar ANN KV cache — Quest stage-1 + KIVI per-(quest-)chunk per-channel

ZoomKV production backend


4-bit stage-2 rerank (offload-aware, _seperate_kernel layout).
Drop-in counterpart of
``cache_hub.polar_cache_pure_rabitq_quest_density_seperate_kernel`` with
RaBitQ stage-2 replaced by KIVI 4-bit:

* Prefill / dynamic update: raw post-RoPE K → per-(quest-)chunk min/max
  (already maintained by ``_compute_and_store_chunk_minmax``) → KIVI 4-bit
  packed-along-head_dim int32 stored in ``key_kivi_packed_gpu`` in
  **chunk-major** layout (A-1):
  ``[bs, kv, n_chunks_max, n_pack, group_size]``.  Within each chunk, the
  ``group_size`` lanes for a fixed dim-block sit at consecutive int32
  positions, so the stage-2 kernel's per-warp packed-K read collapses to a
  single 64-byte coalesced load (vs. 16 strided loads in the older
  token-major layout).
* Decode stage-1: hierarchical or flat Quest chunk picking, only reads
  ``chunk_min`` / ``chunk_max`` / raw query.
* Decode stage-2: CDS dense/sparse classify (unchanged) →
  ``partial_chunk_kivi_qk_dense_sparse_cuda`` (replaces
  ``partial_rerank_chunk_dense_sparse_cuda``) → global top-K.

What got dropped vs. the RaBitQ pipeline:

* SRHT rotation (``fast_hadamard_transform``), diagonal sign matrix
  (``MultiDimBlockQuantizer``), and 3-bit magnitude codebook
  (``_mag_thresholds`` / ``_mag_centers``, ``polar_B`` / ``polar_m``).
* Query encoding (``fused_query_encode`` → ``q_blocks`` / ``q_norm``);
  KIVI uses the GQA-averaged raw query directly.
* ``key_4bit_packed_gpu`` and ``key_block_weight_gpu`` buffers; replaced
  by a single ``key_kivi_packed_gpu`` (int32, packed along head_dim).
* ``batch_encode`` / ``_encode_query_blocks`` /
  ``_rabitq_rerank_on_coarse_indices`` / RaBitQ-only debug branches.

What stayed identical:

* Offload plumbing (CPU pinned KV, copy stream, async D2H, H2D gather).
* Sink + local + retrieval geometry, ``current_seq_len`` bookkeeping,
  ``_update_polar_index`` flush trigger, live carry / skipped tokens.
* Quest stage-1 + CDS classify + fused global top-K.

Activate by setting
``POLAR_CACHE_MODULE=cache_hub.polar_cache_kivi_quest_density_seperate_kernel``
before launching the runner.  KIVI stage-2 is fixed to CUDA, 4-bit, head_dim=128.
"""
import hashlib
import json
import math


def _parse_cand_schedule(raw: str):
    """Parse ``4999:0.7,10000:0.5,0.4`` into ``[(4999, 0.7), (10000, 0.5), (inf, 0.4)]``.

    Each ``limit:ratio`` applies while the retrieval-zone length is ``<= limit``.
    A trailing bare ratio is the long-context tail.
    """
    raw = (raw or "").strip()
    if not raw:
        return []
    schedule = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            limit, ratio = part.split(":", 1)
            schedule.append((float(limit), float(ratio)))
        else:
            schedule.append((float("inf"), float(part)))
    schedule.sort(key=lambda item: item[0])
    return schedule
import os
from typing import Any, Optional

import torch
import torch.cuda.nvtx as nvtx
from cache_hub.kivi.quant_pack import _pack_along_last_dim
from cache_hub.rerank.rerank import (dense_mask_from_topk_cuda,
                                     partial_chunk_density_scores_cuda)
from cache_hub.topk.float_topk_gather_concat_ops import \
    float_topk_gather_concat_3d
from cache_hub.topk.float_topk_ops import float_topk_3d

try:
    from cache_hub.kivi.kivi_qk_dot_ops import \
        partial_chunk_kivi_qk_dense_sparse_cuda
    _HAS_KIVI_CUDA = True
    _KIVI_CUDA_IMPORT_ERR = ""
except Exception as _e:  # noqa: BLE001 — report a hard init failure below
    partial_chunk_kivi_qk_dense_sparse_cuda = None
    _HAS_KIVI_CUDA = False
    _KIVI_CUDA_IMPORT_ERR = repr(_e)


_KIVI_BITS_DEFAULT = int(os.environ.get("POLAR_KIVI_BITS", "4"))
if _KIVI_BITS_DEFAULT != 4:
    raise ValueError(
        f"KIVI speed path requires POLAR_KIVI_BITS=4; got {_KIVI_BITS_DEFAULT}",
    )


# 导入基类
try:
    from .cache import KV_Cache
except ImportError:
    from cache import KV_Cache

from attn_hub.flash_attn_compat import flash_attn_with_kvcache_compat

# Steady Zone 动态更新的间隔已移至实例变量 self.dynamic_update_interval

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
        codebook_path: str = '',  # ignored; kept for kwarg-compat with qwen.py
        # NOTE: SuCo 碰撞路径已移除，collision_ratio 仅作 backward-compat 占位
        # （qwen.py 始终往 polar_cache_kwargs 里塞这个键），运行期不再使用。
        collision_ratio: float = 0.3,
        candidate_ratio: float = 0.05,
        final_topk: int = 100,  # 精排阶段：从候选集中选 top-K 个 key
        dynamic_update_interval: int = 512,  # 每累积多少个 tokens 触发更新
        full_attention_threshold: int = 2000,  # prefill + decode < threshold 时使用全量 attention
        enable_offload: bool = True,
        debug_offload_coords: bool = False,  # ignored compatibility knob
        enable_rerank: bool = False,  # ignored compatibility knob
        enable_alpha_correction: bool = False,  # ignored compatibility knob
        enable_rerank_error_ablation: bool = False,  # ignored compatibility knob
        adaptive_topk_enabled: bool = False,  # ignored compatibility knob
        adaptive_topk_threshold: float = 0.9,  # ignored compatibility knob
        adaptive_topk_min_k: int = 100,  # ignored compatibility knob
        adaptive_topk_max_k: int = 400,  # ignored compatibility knob
        **_ignored_compat_kwargs,  # 吸收老调用方仍传入的兼容 kwarg
    ) -> None:
        
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)
         
        self.valid_start = valid_start  # 用于 prefill 时确定有效数据起始位置
        self.sink_size = sink_size  # Sink tokens 数量 = 4
        self.local_size = local_size   # Local window 大小 (通过配置文件设置,推荐512)
        # ---- aligned-offload (A 方案) ----
        # 只把 stage-1 对齐粒度整数倍的 retrieval token offload 到 CPU；不足一个对齐块
        # 的余数 R0 留在 GPU，当作 local 的延伸全量 attend（零 H2D，且 CPU retrieval 永远
        # 整除 → skipped=0）。余数恒定 = R0，所以环形缓冲机制不变，只是把 local_size 换成
        # 这里的 _eff_local_size = local_size + R0（prefill 时按真实 prefill 长度确定）。
        self._align_offload = os.environ.get("POLAR_ALIGN_OFFLOAD", "1") != "0"
        self._eff_local_size = self.local_size  # 实际生效的 GPU near-zone 窗口（prefill 时更新）
        self.steady_size = self.sink_size + self._eff_local_size  # Steady Zone 总大小
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
        _ = codebook_path  # explicitly ignored (KIVI has no codebook)
        _ = collision_ratio  # explicitly ignored; kept for kwarg compat with qwen.py
        _ = (
            debug_offload_coords,
            enable_rerank,
            enable_alpha_correction,
            enable_rerank_error_ablation,
            adaptive_topk_enabled,
            adaptive_topk_threshold,
            adaptive_topk_min_k,
            adaptive_topk_max_k,
        )
        self.candidate_ratio = candidate_ratio
        self.final_topk = final_topk  # 精排阶段的 top-K
        if not enable_offload:
            raise RuntimeError("KIVI speed path requires enable_offload=True")
        self.enable_offload = True  # speed path is offload-only
        # ----- Quest-inspired Stage-1 (this file's production retrieval path) -----
        self.quest_chunk_size = max(1, int(os.environ.get("POLAR_QUEST_CHUNK", "8")))
        # 当 retrieval 长度不是 chunk_size 的整数倍时，跨界 chunk 和 trailing partial 的
        # token 没有 chunk min/max 元数据，因此**不进 stage-1 评分**；但它们仍有合法的
        # RaBitQ 4-bit 编码，所以会以"额外候选"身份 cat 进 coarse_indices，**继续走 stage-2
        # rerank**。每次 prefill / decode update 最多产生 chunk_size 个跨界 + chunk_size-1
        # 个 trailing partial token。1024 decode @ dui=512 最多 2 次 flush ⇒ 上界 ~22；
        # slack 128 够 ~16k decode token。
        # live skipped 上界 = chunk_size * hq_factor - 1（hierarchical 时）或 chunk_size - 1（非分层）。
        # 默认 512 足够覆盖 chunk=8 × factor=64 的极端情况；遇到更激进配置时通过 env 调大。
        self.MAX_SKIPPED_INDICES = int(os.environ.get("POLAR_MAX_SKIPPED_INDICES", "512"))
        # ----- Quest coarse selection -----
        self._use_hierarchical_quest = (
            os.environ.get("POLAR_QUEST_HIERARCHICAL", "1")
            not in ("0", "false", "False", "")
        )
        self.quest_coarse_candidate_ratio = float(
            os.environ.get("POLAR_QUEST_CAND_RATIO", str(candidate_ratio))
        )
        # Optional length schedule, e.g. "4999:0.7,10000:0.5,0.4"
        # (<5k → 0.7, 5k–10k → 0.5, >10k → 0.4). Each entry applies while
        # the retrieval-zone length is <= the cutoff. A trailing bare ratio
        # is the long-context tail.
        self._cand_schedule = _parse_cand_schedule(
            os.environ.get("POLAR_QUEST_CAND_SCHEDULE", "")
        )
        _force_cand_raw = os.environ.get("POLAR_FORCE_CAND_TOKENS")
        self._force_cand_tokens = int(_force_cand_raw) if _force_cand_raw else None

        # ----- Hierarchical Quest (two-level chunk selection) -----
        self._hq_large_chunk_size = int(os.environ.get("POLAR_QUEST_LARGE_CHUNK", "256"))
        self._hq_large_ratio = float(os.environ.get("POLAR_QUEST_LARGE_RATIO", "0.3"))
        self._hq_small_ratio = float(os.environ.get("POLAR_QUEST_SMALL_RATIO", "0.3"))
        assert self._hq_large_chunk_size % self.quest_chunk_size == 0, (
            f"POLAR_QUEST_LARGE_CHUNK ({self._hq_large_chunk_size}) must be divisible "
            f"by POLAR_QUEST_CHUNK ({self.quest_chunk_size})"
        )
        self._hq_factor = self._hq_large_chunk_size // self.quest_chunk_size
        self._offload_align_chunk = (
            self._hq_large_chunk_size if self._use_hierarchical_quest else self.quest_chunk_size
        )
        # aligned-offload 依赖 decode 每次 offload 量是 large_chunk 的整数倍，否则
        # cpu_valid_end 会重新错位。dui 默认 = local_size；只要它整除 large_chunk 即可。
        if self._align_offload:
            assert self.dynamic_update_interval % self._offload_align_chunk == 0, (
                f"POLAR_ALIGN_OFFLOAD=1 requires dynamic_update_interval "
                f"({self.dynamic_update_interval}) be a multiple of align_chunk "
                f"({self._offload_align_chunk}); 否则 decode offload 会破坏对齐。"
                f" 可调 local_size / dui 或关闭 POLAR_ALIGN_OFFLOAD。"
            )
        # ----- CDS (Coarse → Dense/Sparse) pipeline (always on for this module) -----
        self.cds_dense_ratio = float(os.environ.get("POLAR_CDS_DENSE_RATIO", "0.1"))
        self.cds_dense_topk = int(os.environ.get("POLAR_CDS_DENSE_TOPK", "16"))
        self.cds_sparse_topk = int(os.environ.get("POLAR_CDS_SPARSE_TOPK", "8"))

        # KIVI 4-bit packed-along-head_dim stage-2 codec (no codebook needed).
        self._kivi_bits = _KIVI_BITS_DEFAULT
        self._kivi_feat_per_int = 32 // self._kivi_bits
        if head_dim % self._kivi_feat_per_int != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by "
                f"feat_per_int={self._kivi_feat_per_int} (bits={self._kivi_bits})"
            )
        self._kivi_n_pack = head_dim // self._kivi_feat_per_int
        if not _HAS_KIVI_CUDA:
            raise RuntimeError(
                "KIVI speed path requires the CUDA stage-2 kernel; "
                f"import failed with {_KIVI_CUDA_IMPORT_ERR}"
            )
        if head_dim != 128:
            raise ValueError(f"KIVI CUDA speed path requires head_dim=128; got {head_dim}")
        # 初始化实际使用的 retrieval zone 长度
        self._actual_retrieval_length = None  # 将在 prefill 时设置
        # Quest stage-1 缓存：仅在 prefill 和 _update_polar_index 时刷新
        self._cached_quest_n_chunks = 0
        self._cached_quest_nk = 0
        self._cached_quest_kv_len = -1
        self._cached_scores_view = None
        self._cached_chunk_min_views = None
        self._cached_chunk_max_views = None
        self._cached_chunk_valid_views = None
        # Optional experiment-only recall logging.  Measures retrieval-zone
        # exact q·k recall for the stage-1 chunk pool and final CDS output.
        self._allayer_recall_check = (
            os.environ.get("POLAR_PIPELINE_RECALL", "0") not in ("0", "false", "False", "")
            or os.environ.get("POLAR_ALLAYER_RECALL", "0") not in ("0", "false", "False", "")
        )
        self._allayer_recall_log_every = int(os.environ.get(
            "POLAR_PIPELINE_RECALL_LOG_EVERY",
            os.environ.get("POLAR_ALLAYER_RECALL_LOG_EVERY", "100"),
        ))
        _recall_k_raw = os.environ.get(
            "POLAR_PIPELINE_RECALL_K",
            os.environ.get("POLAR_ALLAYER_RECALL_K", str(final_topk)),
        )
        self._allayer_recall_k = int(str(_recall_k_raw).split(",")[-1])
        self._allayer_recall_step = 0
        self._allayer_recall_accum = {}
        # Optional experiment-only comparison of the current Quest max/min
        # bound against the exact maximum q·k inside every small/large chunk.
        self._quest_bound_check = (
            os.environ.get("POLAR_QUEST_BOUND_CHECK", "0")
            not in ("0", "false", "False", "")
        )
        self._quest_bound_steps = {
            int(v)
            for v in os.environ.get("POLAR_QUEST_BOUND_STEPS", "1").split(",")
            if v.strip()
        }
        self._quest_bound_layers = {
            int(v)
            for v in os.environ.get("POLAR_QUEST_BOUND_LAYERS", "0").split(",")
            if v.strip()
        }
        _bound_heads = os.environ.get("POLAR_QUEST_BOUND_HEADS", "all")
        self._quest_bound_heads = (
            None
            if _bound_heads.strip().lower() == "all"
            else {int(v) for v in _bound_heads.split(",") if v.strip()}
        )
        self._quest_bound_output = os.environ.get(
            "POLAR_QUEST_BOUND_OUTPUT", "results/quest_bound_gap.jsonl"
        )
        self._quest_bound_save_jsonl = (
            os.environ.get("POLAR_QUEST_BOUND_SAVE_JSONL", "1")
            not in ("0", "false", "False", "")
        )
        self._quest_bound_save_values = (
            os.environ.get("POLAR_QUEST_BOUND_SAVE_VALUES", "0")
            not in ("0", "false", "False", "")
        )
        self._quest_bound_pt_dir = os.environ.get("POLAR_QUEST_BOUND_PT_DIR", "")
        self._quest_bound_save_chunk_shards = (
            os.environ.get("POLAR_QUEST_BOUND_SAVE_CHUNK_SHARDS", "1")
            not in ("0", "false", "False", "")
        )
        self._quest_bound_token_dir = os.environ.get(
            "POLAR_QUEST_BOUND_TOKEN_DIR", ""
        )
        self._quest_bound_chunk_sizes = tuple(
            sorted(
                {
                    int(v)
                    for v in os.environ.get(
                        "POLAR_QUEST_BOUND_CHUNK_SIZES",
                        f"{self.quest_chunk_size},{self._hq_large_chunk_size}",
                    ).split(",")
                    if v.strip()
                }
            )
        )
        self._quest_bound_max_chunks = int(
            os.environ.get("POLAR_QUEST_BOUND_MAX_CHUNKS", "0")
        )
        self._quest_bound_full_splits = {
            value.strip()
            for value in os.environ.get(
                "POLAR_QUEST_BOUND_FULL_SPLITS",
                "val,calibration,test",
            ).split(",")
            if value.strip()
        }
        self._quest_bound_exit_after_probe = (
            os.environ.get("POLAR_QUEST_BOUND_EXIT_AFTER_PROBE", "0")
            not in ("0", "false", "False", "")
        )
        self._quest_bound_written = set()
        # Read-only GQA query aggregation diagnostic (does not affect retrieval).
        self._gqa_agg_check = (
            os.environ.get("POLAR_GQA_AGG_CHECK", "0")
            not in ("0", "false", "False", "")
        )
        self._gqa_agg_steps = {
            int(v)
            for v in os.environ.get("POLAR_GQA_AGG_STEPS", "1,4").split(",")
            if v.strip()
        }
        self._gqa_agg_layers = {
            int(v)
            for v in os.environ.get("POLAR_GQA_AGG_LAYERS", "0,9,18,35").split(",")
            if v.strip()
        }
        _gqa_heads = os.environ.get("POLAR_GQA_AGG_HEADS", "all")
        self._gqa_agg_heads = (
            None
            if _gqa_heads.strip().lower() == "all"
            else {int(v) for v in _gqa_heads.split(",") if v.strip()}
        )
        self._gqa_agg_output = os.environ.get(
            "POLAR_GQA_AGG_OUTPUT", "results/gqa_query_agg.jsonl"
        )
        self._gqa_agg_chunk_sizes = tuple(
            sorted(
                {
                    int(v)
                    for v in os.environ.get(
                        "POLAR_GQA_AGG_CHUNK_SIZES", "16,64,128,256"
                    ).split(",")
                    if v.strip()
                }
            )
        )
        self._gqa_agg_ratios = tuple(
            float(v)
            for v in os.environ.get(
                "POLAR_GQA_AGG_RATIOS", "0.01,0.05,0.10,0.20"
            ).split(",")
            if v.strip()
        )
        self._gqa_agg_exit_after_probe = (
            os.environ.get("POLAR_GQA_AGG_EXIT_AFTER_PROBE", "0")
            not in ("0", "false", "False", "")
        )
        self._gqa_agg_written = set()
        # Compact training shards: one file per sample/step/layer.
        self._quest_bound_compact = (
            os.environ.get("POLAR_QUEST_BOUND_COMPACT", "0")
            not in ("0", "false", "False", "")
        )
        # Decode counter used to decide when the local/update buffer flushes.
        self._decode_step_counter = 0
        # 上次 buffer reset 时的 counter 值（长 prefill offload 用）
        self._buffer_base_counter = 0
        # 延迟 base 更新标志（确保 update 后的第一个 token 写到 offset 0）
        self._pending_buffer_reset = False
        

        self.device = torch.device(list[Any](layer_mapping.values())[0] if layer_mapping else 'cuda:0')

        # 统一 KV Cache 架构（按生成顺序存储所有 tokens）: 全量容量 = prefill + decode 的总长度
        self.unified_kv_capacity = self.input_length + self.max_new_length

        # ── decode_prepare bubble killer ──
        # Pre-allocated [0, unified_kv_capacity) int64 arange used for both
        # ``sink_indices`` (= self._arange_buffer[:sink_size]) and
        # ``local_indices`` (= self._arange_buffer[local_start:total_len]).
        # Both slices are zero-copy views; previously each compute() call
        # spent ~10us launching two torch.arange kernels, summing to ~360us
        # per decode token across 36 layers and showing up as a pre-stage1
        # bubble in nsys.
        self._arange_buffer = torch.arange(
            0, self.unified_kv_capacity, dtype=torch.long, device=self.device,
        )
        # Per-step memoisation of get_region_indices() output.  All 36 layers
        # in a single decode step see the same ``current_seq_len[layer_idx]``
        # (update_kv_cache increments AFTER compute()), so layer 0 builds the
        # dict and layers 1..35 reuse it via dict-identity check.  Saves
        # ~2us python work + 1 conditional branch per layer.
        self._cached_region_total_len: int = -1
        self._cached_region_info = None
        # Per-layer cache of ``key_kivi_packed_gpu[l][:, :, :n_valid_chunks]``.
        # The slice itself is stride-only (zero kernel launch) but goes
        # through PyTorch's ``__getitem__`` Python dispatch (~3-5us).
        # ``n_valid_chunks`` only advances every CHUNK_SIZE (=16) tokens,
        # so cache hit rate is ~15/16; saves ~3-5us × 36 layers = ~150us
        # per decode token in steady state.
        self._cached_kivi_packed_n_chunks: list = [-1] * layer_num
        self._cached_kivi_packed_views: list = [None] * layer_num

        
        # 【Offload 模式（speed path 唯一支持的模式）】
        # GPU 只存 Sink + Local + Update Buffer（小容量，省显存）
        #  Retrieval Zone 的 KV offload 到 CPU（大容量，pinned memory）
        # aligned-offload 会把不足一个 stage-1 对齐块的余数留在 GPU near-zone，
        # 所以 GPU 容量要多预留 (align_chunk - 1)。
        self._eff_local_extra = (self._offload_align_chunk - 1) if self._align_offload else 0
        self.gpu_kv_capacity = (
            self.sink_size + self.local_size + self._eff_local_extra
            + self.dynamic_update_interval
        )

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
        self.cpu_kv_capacity = self.unified_kv_capacity - self.sink_size - self.local_size

        # GPU 端：小容量，只存 Sink + Local + Update Buffer
        self.unified_keys_gpu = []
        self.unified_values_gpu = []
        for ldx in range(layer_num):
            self.unified_keys_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
            self.unified_values_gpu.append(
                torch.zeros((batch_size, self.kv_head, self.gpu_kv_capacity, head_dim), dtype=dtype, device=self.device).contiguous())
        # CPU 端：只存 Retrieval Zone 的 KV（pinned memory 加速传输）
        self.unified_keys_cpu = []
        self.unified_values_cpu = []
        for ldx in range(layer_num):
            self.unified_keys_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))
            self.unified_values_cpu.append(torch.zeros((batch_size, self.kv_head, self.cpu_kv_capacity, head_dim), dtype=dtype, pin_memory=True))

        # 记录 CPU 上已存储的有效范围
        self.cpu_valid_end = [0] * layer_num
        # 预分配 TopK gather 结果的 GPU buffer（避免每次 gather 后分配新内存）
        # 使用 final_topk 参数作为 buffer 大小
        max_topk_size = self.final_topk + self.MAX_SKIPPED_INDICES
        self._topk_keys_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)
        self._topk_values_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),dtype=dtype, device=self.device)

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
        self._topk_keys_cpu_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),
            dtype=dtype, device='cpu', pin_memory=True  # pin_memory 加速 CPU→GPU 传输
        )
        self._topk_values_cpu_buffer = torch.empty(
            (batch_size, self.kv_head, max_topk_size, head_dim),
            dtype=dtype, device='cpu', pin_memory=True
        )

        # 注意：在统一架构下，retrieval_zone_capacity 指的是 unified_kv_capacity
        self.retrieval_zone_capacity = self.unified_kv_capacity
        # Pre-compute chunk capacity here so the chunk-major KIVI packed-K
        # allocation below can size axis-2 by chunks (rather than tokens).
        # ``_chunk_n_max`` is also (re)assigned in the Quest-coarse block
        # below; we set it eagerly here because the KIVI packed buffer
        # always needs it under the A-1 chunk-major layout.
        _kivi_chunk = self.quest_chunk_size
        self._chunk_n_max = (
            self.retrieval_zone_capacity + _kivi_chunk - 1
        ) // _kivi_chunk
        # ============ KIVI 4-bit packed-along-head_dim 存储 (chunk-major) ============
        # A-1 layout: [batch, kv_head, n_chunks_max, n_pack, group_size] int32.
        # Within one chunk, lane k (0..g-1) sits at the innermost axis, so a
        # warp's 16 lanes for a fixed dim-block read 16 consecutive int32
        # elements -> one 64-byte coalesced load (vs. 16 strided loads in
        # the previous token-major layout).  The retrieval-internal chunk
        # id from Quest stage-1 is the axis-2 index directly; the stage-2
        # kernel no longer needs a per-token offset / sink_size shift.
        self.key_kivi_packed_gpu = []
        for ldx in range(layer_num):
            self.key_kivi_packed_gpu.append(
                torch.zeros(
                    (batch_size, self.kv_head, self._chunk_n_max,
                     self._kivi_n_pack, _kivi_chunk),
                    dtype=torch.int32, device=self.device,
                ).contiguous()
            )

        # ============ Chunk-level raw K min/max metadata (Quest-style upper bound) ============
        # shape: [batch, kv_head, n_chunks_max, head_dim]
        # 数学：score(chunk) = Σ_d max(q_d * max_k_d, q_d * min_k_d)，对应 Quest paper Sec.3.1
        # 存储 ≈ KV cache 的 1/chunk_size（chunk=8 时约 12.5%）
        chunk = self.quest_chunk_size
        self._chunk_n_max = (self.retrieval_zone_capacity + chunk - 1) // chunk
        self._chunk_key_min_gpu = []
        self._chunk_key_max_gpu = []
        self._chunk_key_centroid_gpu = []
        self._chunk_valid_mask_gpu = []
        for ldx in range(layer_num):
            self._chunk_key_min_gpu.append(
                torch.zeros((batch_size, self.kv_head, self._chunk_n_max, head_dim),
                            dtype=dtype, device=self.device).contiguous()
            )
            self._chunk_key_max_gpu.append(
                torch.zeros((batch_size, self.kv_head, self._chunk_n_max, head_dim),
                            dtype=dtype, device=self.device).contiguous()
            )
            self._chunk_key_centroid_gpu.append(
                torch.zeros((batch_size, self.kv_head, self._chunk_n_max, head_dim),
                            dtype=dtype, device=self.device).contiguous()
            )
            self._chunk_valid_mask_gpu.append(
                torch.zeros(
                    (batch_size, self.kv_head, self._chunk_n_max),
                    dtype=torch.bool,
                    device=self.device,
                ).contiguous()
            )
        # 记录每层已写入的 chunk 数（== floor(retrieval_token_count / chunk_size)）
        self._chunk_valid_end = [0] * layer_num
        # carry: 把上次未满一个 small chunk 的尾巴 token 暂存下来；下次 update
        # 先与新 token 拼接再切块，避免每次都产生重复 head/tail skipped。
        self._chunk_carry_count = [0] * layer_num
        carry_cap = max(0, chunk - 1)
        self._chunk_carry_keys = []
        self._chunk_merge_buffer = []
        for ldx in range(layer_num):
            self._chunk_carry_keys.append(
                torch.zeros(
                    (batch_size, self.kv_head, carry_cap, head_dim),
                    dtype=dtype,
                    device=self.device,
                ).contiguous()
            )
            # 临时拼接 [carry + new_head] 用：固定 chunk 大小，避免 decode 热路径频繁 cat 分配。
            self._chunk_merge_buffer.append(
                torch.empty(
                    (batch_size, self.kv_head, chunk, head_dim),
                    dtype=dtype,
                    device=self.device,
                ).contiguous()
            )
        # 跨界 chunk + trailing partial 的 retrieval-zone 索引，每层独立维护。
        # shape: [batch, kv_head, MAX_SKIPPED_INDICES] int64；attention 阶段直接 cat 进 topk。
        # 注意：所有 batch / kv_head 共享同一组 skipped indices（chunk 边界与 batch 无关），
        # 但为了和 topk_indices [bs, kv, K] 形状对齐，我们把同一份索引广播到 [bs, kv]。
        self._chunk_skipped_indices_gpu = []
        for ldx in range(layer_num):
            self._chunk_skipped_indices_gpu.append(
                torch.zeros(
                    (batch_size, self.kv_head, self.MAX_SKIPPED_INDICES),
                    dtype=torch.int64,
                    device=self.device,
                ).contiguous()
            )
        self._chunk_skipped_count = [0] * layer_num
        # Pre-create chunk expansion offsets [0, 1, ..., chunk-1] used to fan-out
        # chunk indices to token indices in the hot path.
        self._chunk_offs = torch.arange(chunk, device=self.device, dtype=torch.int64).view(1, 1, 1, chunk)
        # --- Hierarchical Quest: large chunk buffers ---
        self._chunk_n_max_large = self._chunk_n_max // self._hq_factor
        self._chunk_key_min_gpu_large = []
        self._chunk_key_max_gpu_large = []
        self._chunk_valid_mask_gpu_large = []
        for ldx in range(layer_num):
            self._chunk_key_min_gpu_large.append(
                torch.zeros((batch_size, self.kv_head, self._chunk_n_max_large, head_dim),
                            dtype=dtype, device=self.device).contiguous()
            )
            self._chunk_key_max_gpu_large.append(
                torch.zeros((batch_size, self.kv_head, self._chunk_n_max_large, head_dim),
                            dtype=dtype, device=self.device).contiguous()
            )
            self._chunk_valid_mask_gpu_large.append(
                torch.zeros(
                    (batch_size, self.kv_head, self._chunk_n_max_large),
                    dtype=torch.bool, device=self.device,
                ).contiguous()
            )
        self._chunk_valid_end_large = [0] * layer_num
        self._quest_scores_buffer_small = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max,
            dtype=torch.float32, device=self.device,
        )
        self._quest_scores_buffer_large = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large,
            dtype=torch.float32, device=self.device,
        )
        self._quest_scores_buffer_sub = torch.empty(
            batch_size * self.kv_head * self._chunk_n_max_large * self._hq_factor,
            dtype=torch.float32, device=self.device,
        )

        self.context = 0
        self._preallocated_buffers = {}


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
        try:
            from .gather import load_gather_ext
            self._gather_ext = load_gather_ext()
        except Exception as e:
            print(f"[Init]   gather ext FAILED: {e}", flush=True)
            self._gather_ext = None
        
        # 预加载 Radix TopK CUDA extension (O(n) vs O(n log k))
        try:
            from .topk import load_radix_topk_ext
            self._radix_topk_ext = load_radix_topk_ext()
        except Exception as e:
            print(f"[Init]   radix_topk FAILED: {e}", flush=True)
            self._radix_topk_ext = None

        # 预加载 UVA H2D Gather kernel（利用 UVA 直接从 pinned CPU gather 到 GPU）
        try:
            from .gather_trans.trans_h2d import h2d_gather_kv
            self._h2d_gather_kv = h2d_gather_kv
        except Exception as e:
            raise RuntimeError(f"H2D Gather CUDA kernel compile failed: {e}")
        
        # 预加载 CDS density + dense-mask CUDA extension (rerank_topk shares
        # the same .so as the RaBitQ rerank kernel — we only use the
        # density/dense-mask entry points here).
        try:
            from .rerank.rerank import load_kernel_module
            self._rerank_topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
        except Exception as e:
            raise RuntimeError(f"rerank_topk CUDA kernel compile failed: {e}")

        # 预加载 KIVI stage-2 CUDA kernel.
        try:
            from .kivi.kivi_qk_dot_ops import _kivi_kernel as _load_kivi_kernel
            self._kivi_kernel = _load_kivi_kernel()
        except Exception as e:
            raise RuntimeError(f"kivi_qk_dot CUDA kernel compile failed: {e}")

        # 预加载 Quest chunk score CUDA kernel
        try:
            from .quest.quest_ops import \
                load_kernel_module as load_quest_kernel
            self._quest_kernel = load_quest_kernel()
        except Exception as e:
            print(f"[Init]   quest_chunk_score CUDA failed ({e}), falling back to PyTorch", flush=True)
            self._quest_kernel = None

        # Keep allocator state warm for speed tests; avoid gc/empty_cache here.

    # 处理单层 prefill 阶段
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, batch_idx):
        """
        Prefill 阶段入口：
        - bsz==1: 按 batch_idx 把单样本写入对应 cache slot（外部按 batch_idx 循环调用）。
        - bsz>1 (lockstep): 一次性处理整个 batch（要求各样本有效长度/valid_start 一致），
          只在 batch_idx==0 时执行，避免外部仍按 batch_idx 循环导致重复工作。
        两条路径只是 batch 维写入切片不同，统一走 ``_prefill_store``。
        """
        bsz = key_states.shape[0]
        if bsz == 1:
            return self._prefill_store(
                key_states, value_states, layer_idx, dst_offset=batch_idx, src_bsz=1
            )
        # bsz>1：只在 batch_idx==0 时执行一次
        if batch_idx != 0:
            # 保持返回 shape 兼容：去掉 padding（按 batch_idx=0 的 valid_start 规则）
            vs = self.valid_start[0] if isinstance(self.valid_start, (list, tuple)) else int(self.valid_start)
            return key_states[:, vs:, :, :], value_states[:, vs:, :, :]
        return self._prefill_store(
            key_states, value_states, layer_idx, dst_offset=0, src_bsz=bsz
        )

    def _resolve_valid_start(self, dst_offset: int, src_bsz: int) -> int:
        """Resolve the (batch-consistent) ``valid_start`` for this prefill call.

        For ``src_bsz>1`` (lockstep) we require an identical valid_start across
        the involved batch slots (per-batch region划分否则需要更大改动)。
        """
        vs = self.valid_start
        if not hasattr(vs, '__len__'):
            return int(vs)
        if src_bsz > 1:
            assert len(vs) >= dst_offset + src_bsz
            vs0 = int(vs[dst_offset])
            for b in range(dst_offset + 1, dst_offset + src_bsz):
                assert int(vs[b]) == vs0, (
                    f"lockstep multi-batch currently requires same valid_start across batch. "
                    f"valid_start[{dst_offset}]={vs0}, valid_start[{b}]={vs[b]}"
                )
            return vs0
        return int(vs[dst_offset])

    def _prefill_store(self, key_states, value_states, layer_idx, *, dst_offset: int, src_bsz: int):
        """Offload-only prefill：把 ``src_bsz`` 个样本按区域写入 unified KV cache。

        统一了原 ``_prefill_single_sample`` (dst_offset=batch_idx, src_bsz=1) 与
        ``_prefill_lockstep_batch`` (dst_offset=0, src_bsz=batch_size) 两条路径，
        行为完全一致——区别仅在于 batch 维的写入切片。

        区域划分（zone-internal offset 从 0 起）：
            Sink      = [valid_start, valid_start+sink_size)        → GPU
            Retrieval = [sink_end, max(sink_end, seq_len-local_size)) → 异步 offload 到 CPU
            Local     = [retrieval_end, seq_len)                    → GPU
        KIVI 4-bit packing 折叠进 ``_compute_and_store_chunk_minmax`` —— 每个 chunk
        的 (mn, mx, packed_codes) 三元组在同一处一次写齐，保证 stage-2 读到一致状态。
        """
        bsz, seq_len, group_num, head_dim = key_states.shape
        assert seq_len <= self.input_length
        b_sl = slice(dst_offset, dst_offset + src_bsz)
        is_last_layer = (layer_idx == self.layer_num - 1)
        is_last_chunk = (dst_offset + src_bsz == self.batch_size)

        valid_start = self._resolve_valid_start(dst_offset, src_bsz)
        sink_start = valid_start
        sink_end = valid_start + self.sink_size
        retrieval_start = sink_end
        # 默认 retrieval/local 边界
        raw_retrieval_end = max(retrieval_start, seq_len - self.local_size)
        align_chunk = self._offload_align_chunk
        raw_retrieval_len = raw_retrieval_end - retrieval_start
        if self._align_offload and raw_retrieval_len >= align_chunk:
            # 把 offload 到 CPU 的 retrieval 长度向下取整到 align_chunk；不足的余数 R0
            # 留在 GPU，并入 local（_eff_local_size = local_size + R0）。
            aligned_retrieval_len = (raw_retrieval_len // align_chunk) * align_chunk
            r0 = raw_retrieval_len - aligned_retrieval_len
            retrieval_end = retrieval_start + aligned_retrieval_len
            self._eff_local_size = self.local_size + r0
        else:
            retrieval_end = raw_retrieval_end
            self._eff_local_size = self.local_size
        self.steady_size = self.sink_size + self._eff_local_size
        local_start = retrieval_end
        local_end = seq_len
        retrieval_length = retrieval_end - retrieval_start
        valid_length = retrieval_length

        # 首次 prefill：按真实 batch 维重新分配精确大小的 topk buffer（避免每次 slice）
        if self._actual_retrieval_length is None:
            self._actual_retrieval_length = valid_length
            self._refresh_quest_chunk_cache(valid_length)
            bs = self.unified_keys_cpu[0].shape[0]
            max_topk = self.final_topk + self.MAX_SKIPPED_INDICES
            self._topk_keys_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device=self.device)
            self._topk_values_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device=self.device)
            self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
            self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, max_topk, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

        # 短 prefill：没有 retrieval zone，全部留在 GPU（不触发 offload）
        if valid_length == 0:
            if layer_idx == 0 and not hasattr(self, '_short_prefill_printed'):
                self._short_prefill_printed = True
            actual_len = min(seq_len - valid_start, self.gpu_kv_capacity)
            self.unified_keys_gpu[layer_idx][b_sl, :, :actual_len, :] = \
                key_states[:src_bsz, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.unified_values_gpu[layer_idx][b_sl, :, :actual_len, :] = \
                value_states[:src_bsz, valid_start:valid_start+actual_len, :, :].transpose(1, 2).to(self.dtype)
            self.current_seq_len[layer_idx] = actual_len
            if is_last_layer and is_last_chunk:
                # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
                self.context += actual_len
            return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]

        # Step1: 分区存储 —— Sink + Local → GPU，Retrieval → CPU（异步 D2H）
        local_len = local_end - local_start
        sink_k = key_states[:src_bsz, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
        sink_v = value_states[:src_bsz, sink_start:sink_end, :, :].transpose(1, 2).to(self.dtype)
        self.unified_keys_gpu[layer_idx][b_sl, :, :self.sink_size, :] = sink_k
        self.unified_values_gpu[layer_idx][b_sl, :, :self.sink_size, :] = sink_v

        local_k = key_states[:src_bsz, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
        local_v = value_states[:src_bsz, local_start:local_end, :, :].transpose(1, 2).to(self.dtype)
        self.unified_keys_gpu[layer_idx][b_sl, :, self.sink_size:self.sink_size+local_len, :] = local_k
        self.unified_values_gpu[layer_idx][b_sl, :, self.sink_size:self.sink_size+local_len, :] = local_v

        temp_retrieval_keys = key_states[:src_bsz, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
        temp_retrieval_values = value_states[:src_bsz, retrieval_start:retrieval_end, :, :].transpose(1, 2).to(self.dtype).contiguous()
        # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
        self._inflight_offload_buffers[layer_idx] = (temp_retrieval_keys, temp_retrieval_values)
        self._layer_offload_inflight[layer_idx] = True
        self._main_events[layer_idx].record()
        # Copy stream 异步拷贝：不阻塞主流，与后续 encode 并行
        with torch.cuda.stream(self._copy_stream):
            self._main_events[layer_idx].wait()
            self.unified_keys_cpu[layer_idx][b_sl, :, :retrieval_length, :].copy_(temp_retrieval_keys, non_blocking=True)
            self.unified_values_cpu[layer_idx][b_sl, :, :retrieval_length, :].copy_(temp_retrieval_values, non_blocking=True)
            self._copy_events[layer_idx].record()
        self.cpu_valid_end[layer_idx] = retrieval_length

        self.current_seq_len[layer_idx] = seq_len - valid_start

        # Step2: KIVI per-(quest-)chunk per-channel 4-bit encode（折叠进 chunk min/max）
        keys_to_encode = key_states[:src_bsz, retrieval_start:retrieval_end, :, :].transpose(1, 2).contiguous()
        self._compute_and_store_chunk_minmax(
            layer_idx, keys_to_encode, abs_token_start=0, batch_slice=b_sl,
        )

        if is_last_layer and is_last_chunk:
            # context 跟踪单个样本的位置（用于 position_ids），不是 batch * seq_len
            self.context += self.current_seq_len[layer_idx]
            self._sync_offload_copies()
        else:
            # 按层等待本层 D2H 完成并清理引用，避免跨层累计持有大张量导致峰值显存暴涨。
            self._sync_layer_offload(layer_idx)

        return key_states[:, valid_start:, :, :], value_states[:, valid_start:, :, :]


    def _sync_offload_copies(self):
        """等待所有异步 offload 拷贝完成"""
        # 等待 copy stream 上的所有操作完成
        self._copy_stream.synchronize()
        # 清理临时引用
        for ldx in range(self.layer_num):
            self._inflight_offload_buffers[ldx] = None
            self._layer_offload_inflight[ldx] = False



    def _sync_layer_offload(self, layer_idx: int):
        """只等待指定层的异步 offload 完成（PQCache 风格：按层精确等待）。"""
        if not self._layer_offload_inflight[layer_idx]:
            return
        self._copy_events[layer_idx].synchronize()
        self._inflight_offload_buffers[layer_idx] = None
        self._layer_offload_inflight[layer_idx] = False


    def get_region_indices(self, layer_idx, device=None):
        """
        动态计算各区域索引（统一 KV cache 架构）
        Args:
            layer_idx: 当前层索引
            device: 目标设备（保留以兼容旧 caller，本实现忽略：所有索引来自
                    pre-allocated ``self._arange_buffer``，已经在 ``self.device``）
        Returns:
            dict: {
                'sink_indices': Tensor - Sink 区域索引   (zero-copy view)
                'local_indices': Tensor - Local + UpdateBuffer 区域索引 (zero-copy view)
                'retrieval_start': int - Retrieval zone 起始 (= sink_size)
                'retrieval_length': int - Retrieval zone 长度
                'total_len': int - 当前总序列长度
                'local_len': int - Local + UpdateBuffer 长度
            }

        Bubble killer (vs. previous implementation):
          * Both ``sink_indices`` and ``local_indices`` are now slices of the
            pre-allocated ``self._arange_buffer``: zero kernel launches and
            zero allocator pressure (was 2× ``torch.arange`` per call =
            ~10us host launch + same on device).
          * Per-step memoisation: all 36 layers in a decode step see the
            same ``current_seq_len`` (update_kv_cache increments AFTER
            compute()).  Layer 0 builds the dict, layers 1..35 reuse it
            via the ``_cached_region_total_len`` fast path.
        """
        del device  # only the pre-allocated buffer's device matters

        total_len = self.current_seq_len[layer_idx]

        # Per-step fast path: same total_len → same region info across the
        # 36 layers within one decode step.  Saves ~2us python work per
        # layer × 35 = ~70us per token.
        if total_len == self._cached_region_total_len and self._cached_region_info is not None:
            return self._cached_region_info

        # 计算 update buffer 大小（区分长短 prefill）
        update_buffer_size = self._compute_update_buffer_size(total_len)

        # 计算 local_start（retrieval_end = local_start）
        # _eff_local_size = local_size + R0：aligned-offload 把不足一个 large_chunk 的余数
        # 留在 GPU near-zone，所以这里的窗口用 eff 而非 local_size，保证
        # local_start == sink + cpu_valid_end（GPU near-zone 紧接 CPU retrieval 尾部）。
        local_end = max(total_len - update_buffer_size, self.sink_size)
        local_start = max(self.sink_size, local_end - self._eff_local_size)
        retrieval_length = max(0, local_start - self.sink_size) # 是"理论值"（滑动窗口认为应该有多少），不是"实际值"（真正 offload 了多少）。

        # Zero-launch index views — slicing a contiguous arange returns a
        # view; downstream callers (``unsqueeze().expand()``) accept views.
        sink_indices = self._arange_buffer[:self.sink_size]
        local_len = total_len - local_start if local_start < total_len else 0
        if local_len > 0:
            local_indices = self._arange_buffer[local_start:total_len]
        else:
            # zero-length view (no allocation)
            local_indices = self._arange_buffer[:0]

        region_info = {
            'sink_indices': sink_indices,
            'local_indices': local_indices,
            'retrieval_start': self.sink_size,
            'retrieval_length': retrieval_length,
            'total_len': total_len,
            'local_len': local_len,
        }
        self._cached_region_total_len = total_len
        self._cached_region_info = region_info
        return region_info


    def _compute_update_buffer_size(self, total_len: int) -> int:
        """Update-buffer 内 token 数（区分长/短 prefill），供区域划分逻辑共享。

        长 prefill (input_length >= steady_size)：buffer = (decode 累计) % dui。
        短 prefill：buffer = (total_len - steady_size) % dui。
        steady 之前：无 buffer。
        """
        steady_threshold = self.steady_size  # sink_size + local_size
        if self.input_length >= steady_threshold:
            return max(0, total_len - self.input_length) % self.dynamic_update_interval
        if total_len >= steady_threshold:
            return (total_len - steady_threshold) % self.dynamic_update_interval
        return 0

    def _compute_local_start(self, layer_idx=0):
        """
        轻量版：只计算 local_start，不构建 tensor（用于 _update_polar_index）
        """
        total_len = self.current_seq_len[layer_idx]
        update_buffer_size = self._compute_update_buffer_size(total_len)
        local_end = max(total_len - update_buffer_size, self.sink_size)
        return max(self.sink_size, local_end - self._eff_local_size)


    def _get_full_kv_for_attention(self, layer_idx, sink_indices, local_indices, 
                                    retrieval_start, retrieval_length, device, dtype):
        """
        获取完整的 KV 用于 attention 计算（Sink + Retrieval + Local，offload-only）。

        Args:
            layer_idx: 层索引
            sink_indices: Sink 区域的全局索引
            local_indices: Local+Buffer 区域的全局索引
            retrieval_start / dtype: 保留以兼容旧 caller（offload 路径不再使用）
            retrieval_length: Retrieval 区域长度（理论值）
            device: 目标设备
        Returns:
            (concat_keys, concat_values): 拼接后的 KV，形状 [bs, kv_head, total_len, head_dim]
        """
        local_len = local_indices.shape[0]

        if retrieval_length > 0:
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
            # 没有 retrieval zone，只用 Sink + Local（仍在 GPU）
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
        self._refresh_quest_chunk_cache(self._actual_retrieval_length)
        
        for ldx in range(self.layer_num):
            # Offload-only: encode tokens that are about to leave GPU Local.
            # GPU layout: [sink | local | update_buffer].
            offload_start = self.sink_size
            offload_end = self.sink_size + num_new_tokens
            new_tokens_to_encode = self.unified_keys_gpu[ldx][:, :, offload_start:offload_end, :]

            # KIVI per-(quest-)chunk per-channel encode is folded into
            # _compute_and_store_chunk_minmax — it writes both the
            # chunk_min/max bookkeeping AND the 4-bit packed codes into
            # key_kivi_packed_gpu for every chunk it completes (including
            # the carry-merged chunk).  zone-internal offset = encode_start - sink_size.
            self._compute_and_store_chunk_minmax(
                ldx,
                new_tokens_to_encode,
                abs_token_start=encode_start - self.sink_size,
            )

            
            # ===========================================
            # Offload-only update:
            # 1. 把 Local 最老的部分 offload 到 CPU（进入 retrieval zone）
            # 2. 滑动 GPU 上的 Local window（Local 新部分 + Update Buffer → 新 Local）
            # 1. 把 Local 最老的 num_new_tokens 个 tokens offload 到 CPU
            # 这些 tokens 从 GPU local 区域移出，进入 CPU retrieval zone
            # GPU 布局: [0:sink] [sink:sink+local] [sink+local:sink+local+update]
            # Local 最老部分: [sink : sink+num_new_tokens]
            cpu_write_start = self.cpu_valid_end[ldx]
            cpu_write_end = cpu_write_start + num_new_tokens

            # 使用预分配 buffer 替代 clone()，避免内存分配开销
            src_keys = self._offload_keys_buffer[ldx][:, :, :num_new_tokens, :]
            src_values = self._offload_values_buffer[ldx][:, :, :num_new_tokens, :]
            src_keys.copy_(self.unified_keys_gpu[ldx][:, :, self.sink_size:self.sink_size + num_new_tokens, :])
            src_values.copy_(self.unified_values_gpu[ldx][:, :, self.sink_size:self.sink_size + num_new_tokens, :])
            # 保留引用：避免异步拷贝过程中源 tensor 被 allocator 复用
            self._inflight_offload_buffers[ldx] = (src_keys, src_values)
            self._layer_offload_inflight[ldx] = True

            # 在 copy stream 中异步拷贝到 CPU（与主流并行）
            self._main_events[ldx].record()
            with torch.cuda.stream(self._copy_stream):
                self._main_events[ldx].wait()
                dst_keys = self.unified_keys_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                dst_values = self.unified_values_cpu[ldx][:, :, cpu_write_start:cpu_write_end, :]
                dst_keys.copy_(src_keys, non_blocking=True)
                dst_values.copy_(src_values, non_blocking=True)
                self._copy_events[ldx].record()

            self.cpu_valid_end[ldx] = cpu_write_end

            # 2. 滑动 GPU Local window（可以与 CPU 拷贝并行）
            # near-zone 窗口大小 = _eff_local_size（= local_size + R0，含 aligned-offload
            # 留在 GPU 的余数）。把 [sink+num_new : sink+eff+num_new] 移到 [sink : sink+eff]
            # 即：原 near-zone 新部分 + 原 Update Buffer → 新 near-zone。
            old_local_end = self.sink_size + self._eff_local_size
            old_buffer_end = old_local_end + num_new_tokens
            new_local_start_src = self.sink_size + num_new_tokens

            # 只有当 num_new_tokens < eff 时源和目标才会重叠，需要 clone
            if num_new_tokens < self._eff_local_size:
                new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
                new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :].clone()
            else:
                new_local_keys = self.unified_keys_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
                new_local_values = self.unified_values_gpu[ldx][:, :, new_local_start_src:old_buffer_end, :]
            self.unified_keys_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_keys
            self.unified_values_gpu[ldx][:, :, self.sink_size:old_local_end, :] = new_local_values
            if num_new_tokens < self._eff_local_size:
                del new_local_keys, new_local_values
        
        # 等待所有层的 offload 完成，这样 retrieval 阶段不需要再检查
        self._sync_offload_copies()
        
        self.steady_size = self.sink_size + self._eff_local_size



    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        统一架构:Decode阶段直接append到unified_keys_gpu
        每生成1个token,直接追加到当前序列末尾
        当积累 dynamic_update_interval 个新tokens时,触发动态更新(编码新tokens到retrieval zone)
        """
        # nvtx.range_push(f"decode_update_layer_{layer_idx}")

        # ============ Step 1: 先写入当前 token ============
        # nvtx.range_push("kv_write")
        key_to_add = key_states[:, 0, :, :]
        value_to_add = value_states[:, 0, :, :]

        # Offload-only GPU layout: [Sink][Local][UpdateBuffer]
        current_len = self.current_seq_len[layer_idx]
        steady_threshold = self.steady_size
        if current_len < steady_threshold:
            gpu_write_pos = current_len
        else:
            # Local 已满，写入 Update Buffer
            if self.input_length >= steady_threshold:
                tokens_in_buffer = self._decode_step_counter - self._buffer_base_counter
            else:
                tokens_in_buffer = current_len - steady_threshold
            buffer_offset = tokens_in_buffer % self.dynamic_update_interval
            gpu_write_pos = steady_threshold + buffer_offset

        self.unified_keys_gpu[layer_idx][:, :, gpu_write_pos, :] = key_to_add
        self.unified_values_gpu[layer_idx][:, :, gpu_write_pos, :] = value_to_add
        
        # 更新全局序列长度
        self.current_seq_len[layer_idx] += 1
        # nvtx.range_pop()  # kv_write
        
        # ============ Step 2: 在第一层检查是否触发 update（写入后检查！）============
        if layer_idx == 0:
            # print(f"[AFTER WRITE] current_seq_len[0]={self.current_seq_len[0]} (compute 会用这个值)")
            # 递增计数器（在写入 token 之后）
            self._decode_step_counter += 1
            self.context += 1
            
            # 处理延迟的 buffer base 更新（base 应设为触发时的 counter 值，即当前 counter - 1）
            if self._pending_buffer_reset:
                self._buffer_base_counter = self._decode_step_counter - 1
                self._pending_buffer_reset = False
                # print(f"[BASE RESET] old_base={old_base} → new_base={self._buffer_base_counter} (counter={self._decode_step_counter})")
            
            # 检查是否需要触发 update（Buffer 刚好写满 dynamic_update_interval 个）
            current_total_len = self.current_seq_len[0]  # 现在包含刚写入的 token！
            steady_threshold = self.steady_size  # 预计算：sink_size + local_size
            
            if current_total_len > steady_threshold:
                if self.input_length >= steady_threshold:
                    # 长 prefill：用 decode counter 计算 buffer 内 token 数
                    buffer_tokens = self._decode_step_counter - self._buffer_base_counter
                else:
                    # 短 prefill：用序列长度溢出计算
                    buffer_tokens = current_total_len - steady_threshold
                
                # Buffer 刚好满（256 个）时触发
                if buffer_tokens > 0 and buffer_tokens % self.dynamic_update_interval == 0:
                    # print(f"[Decode] 触发 _update_polar_index, buffer_tokens={buffer_tokens}, step={self._decode_step_counter}")
                    # nvtx.range_push("update_polar_index")
                    self._update_polar_index()
                    self._pending_buffer_reset = True
                    # nvtx.range_pop()

        # nvtx.range_pop()  # decode_update_layer_{layer_idx}
        return None, None


    # ============================================================================
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
        # nvtx.range_push("decode_prepare")

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
        # Offload-only：cpu_valid_end 是实际限制（<= encoded_len <= retrieval_len）
        valid_length = self.cpu_valid_end[layer_idx]
        
        # 准备 query: [bs, 1, num_heads, head_dim] -> [bs, kv_heads, head_dim]
        bs, seqlen_q, head_q, head_dim = queries.shape
        query_group = queries[:, 0, :, :]
        queries_reshaped = query_group.view(bs, self.kv_head, self.group_size, self.head_dim)
        query_avg_gpu = queries_reshaped.mean(dim=2)  # [bs, kv_heads, head_dim]
        query_batch = query_avg_gpu.unsqueeze(2)  # [bs, kv_heads, 1, head_dim]
        # nvtx.range_pop()  # decode_prepare_query

        # Read-only GQA aggregation diagnostics use the raw 4-query group.
        # Production retrieval continues to use query_avg_gpu only.
        if self._gqa_agg_check:
            self._gqa_query_agg_probe(queries_reshaped, layer_idx)
        if self._quest_bound_check and self._quest_bound_compact:
            self._save_quest_compact_layer_shards(queries_reshaped, layer_idx)
        
        # 判断是否需要检索
        # total_length 至少要 > sink + local + final_topk*2，否则 retrieval zone 太小
        min_total_for_retrieval = max(self.FULL_ATTENTION_THRESHOLD, self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)
        # nvtx.range_pop()


        if use_full_attention:
            # ============ 全量 Attention 模式 ============
            concat_keys, concat_values = self._get_full_kv_for_attention(layer_idx, sink_indices, local_indices, retrieval_start, retrieval_length, device, queries.dtype)
        else:

            # ============ 检索模式 ============
            # KIVI stage-2 reads `key_kivi_packed_gpu` (chunk-major after
            # A-1) directly by retrieval-internal chunk id.  We slice on
            # the chunk axis to expose only the chunks that have been
            # written so far — Quest stage-1 already filters by
            # ``_chunk_valid_mask_gpu``, but slicing here lets the kernel
            # bounds-check ``real_chunk_id < n_chunks_packed`` cheaply.
            chunk_size = self.quest_chunk_size
            n_valid_chunks = max(0, valid_length // chunk_size)
            # n_valid_chunks grows only when new retrieval chunks become valid
            # after prefill/update flushes. Most decode steps reuse this view.
            if self._cached_kivi_packed_n_chunks[layer_idx] == n_valid_chunks:
                key_kivi_packed = self._cached_kivi_packed_views[layer_idx]
            else:
                key_kivi_packed = self.key_kivi_packed_gpu[layer_idx][:, :, :n_valid_chunks, :, :]
                self._cached_kivi_packed_n_chunks[layer_idx] = n_valid_chunks
                self._cached_kivi_packed_views[layer_idx] = key_kivi_packed

            # nvtx.range_push("collision_based_topk_batch")
            topk_indices_batch = self.collision_based_topk_batch(
                query=query_batch,
                key_kivi_packed=key_kivi_packed,
                layer_idx=layer_idx,
            )
            # nvtx.range_pop()
            
            # if layer_idx == 0:
            #     print(f"[Debug] topk_indices_batch.shape={topk_indices_batch.shape}, final_topk={self.final_topk}", flush=True)

            sink_local_end = self.sink_size + local_len
            sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
            sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]

            # 【Offload 模式】h2d_gather_kv → cat → 单次 flash attention
            nvtx.range_push("h2d_gather_kv")
            self._h2d_gather_kv(
                self.unified_keys_cpu[layer_idx],    # src keys: CPU pinned
                self.unified_values_cpu[layer_idx],  # src values: CPU pinned
                topk_indices_batch,                   # indices: GPU (无需传回 CPU)
                self._topk_keys_buffer,               # dst keys: GPU
                self._topk_values_buffer              # dst values: GPU
            )
            nvtx.range_pop()

            nvtx.range_push("compute_attention")
            n_topk = topk_indices_batch.shape[-1]
            concat_keys = torch.cat([sink_local_keys, self._topk_keys_buffer[:, :, :n_topk, :]], dim=2)
            concat_values = torch.cat([sink_local_values, self._topk_values_buffer[:, :, :n_topk, :]], dim=2)
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

        # 7. 全量 Attention：cat 已完成，仅 Flash
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
    # Hierarchical Quest cache refresh
    # ============================================================================
    def _candidate_ratio_for_length(self, kv_len: int) -> float:
        """Stage-1 keep ratio. A schedule lowers it as the retrieval zone grows."""
        if not self._cand_schedule:
            return self.quest_coarse_candidate_ratio
        for limit, ratio in self._cand_schedule:
            if kv_len <= limit:
                return ratio
        return self._cand_schedule[-1][1]

    def _quest_aligned_candidate_tokens(self, kv_len: int) -> int:
        """Flat stage-1 token budget aligned to ``quest_chunk_size``."""
        chunk = self.quest_chunk_size
        if kv_len <= 0:
            return 0
        if kv_len < chunk:
            return kv_len
        cap = (kv_len // chunk) * chunk
        ratio = self._candidate_ratio_for_length(kv_len)
        target = (
            self._force_cand_tokens
            if self._force_cand_tokens is not None
            else math.ceil(kv_len * ratio)
        )
        target_aligned = max(chunk, ((target + chunk - 1) // chunk) * chunk)
        return min(cap, target_aligned)

    def _refresh_quest_chunk_cache(self, kv_len: int):
        """Recompute and cache all stage-1 invariants.

        Called at prefill end and _update_polar_index; the decode hot path
        reads only cached values — no shape queries, no slicing, no arithmetic.
        """
        chunk = self.quest_chunk_size
        self._refresh_live_skipped_indices(kv_len)
        if kv_len < chunk:
            self._cached_quest_n_chunks = 0
            self._cached_quest_nk = 0
            self._cached_quest_kv_len = kv_len
            self._cached_cds_n_dense = 0
            self._cached_cds_density_scores = None
            self._cached_hq_n_large = 0
            self._cached_hq_nk_large = 0
            self._cached_hq_nk_small = 0
            return
        n_chunks = kv_len // chunk
        self._cached_quest_n_chunks = n_chunks
        self._cached_quest_kv_len = kv_len
        bs = self.batch_size
        kv_heads = self.kv_head
        self._cached_chunk_min_views = [
            t[:bs] for t in self._chunk_key_min_gpu
        ]
        self._cached_chunk_max_views = [
            t[:bs] for t in self._chunk_key_max_gpu
        ]
        self._cached_chunk_valid_views = [
            t[:bs] for t in self._chunk_valid_mask_gpu
        ]
        if not self._use_hierarchical_quest:
            target_tokens = self._quest_aligned_candidate_tokens(kv_len)
            nk = max(1, min(n_chunks, target_tokens // chunk))
            self._cached_quest_nk = nk
            self._cached_scores_view = (
                self._quest_scores_buffer_small[:bs * kv_heads * n_chunks]
                .view(bs, kv_heads, n_chunks)
            )
            nk_small_for_cds = nk
            self._cached_cds_n_dense = max(1, int(nk_small_for_cds * self.cds_dense_ratio))
            self._cached_dense_mask = torch.zeros(
                bs, kv_heads, nk_small_for_cds, dtype=torch.bool, device=self.device,
            )
            self._cached_cds_out_scores = torch.empty(
                bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
                dtype=torch.float32, device=self.device,
            )
            self._cached_cds_out_indices = torch.empty(
                bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
                dtype=torch.int64, device=self.device,
            )
            self._cached_cds_density_scores = torch.empty(
                bs, kv_heads, nk_small_for_cds, dtype=torch.float32, device=self.device,
            )
            self._cached_cds_final_indices = torch.empty(
                bs, kv_heads, self.final_topk + self.MAX_SKIPPED_INDICES,
                dtype=torch.int64, device=self.device,
            )
            return
        # --- Hierarchical Quest cached views ---
        n_large = n_chunks // self._hq_factor
        nk_large = max(1, int(math.ceil(n_large * self._hq_large_ratio)))
        nk_large = min(nk_large, n_large)
        nk_sub_total = nk_large * self._hq_factor
        nk_small = max(1, int(math.ceil(nk_sub_total * self._hq_small_ratio)))
        nk_small = min(nk_small, nk_sub_total)
        self._cached_hq_n_large = n_large
        self._cached_hq_nk_large = nk_large
        self._cached_hq_nk_small = nk_small
        nk_small_for_cds = nk_small
        # Output buffer for hq_map_back; reused every decode step to avoid
        # per-call torch.empty_like allocation on the hot path.
        self._cached_hq_chunk_idx = torch.empty(
            bs, kv_heads, nk_small, dtype=torch.int64, device=self.device,
        )
        self._cached_hq_scores_large_view = (
            self._quest_scores_buffer_large[:bs * kv_heads * n_large]
            .view(bs, kv_heads, n_large)
        )
        self._cached_hq_scores_sub_view = (
            self._quest_scores_buffer_sub[:bs * kv_heads * nk_large * self._hq_factor]
            .view(bs, kv_heads, nk_large * self._hq_factor)
        )
        self._cached_hq_large_min_views = [
            t[:bs] for t in self._chunk_key_min_gpu_large
        ]
        self._cached_hq_large_max_views = [
            t[:bs] for t in self._chunk_key_max_gpu_large
        ]
        self._cached_hq_large_valid_views = [
            t[:bs] for t in self._chunk_valid_mask_gpu_large
        ]

        # --- CDS cache (shared by hierarchical + single-level) ---
        self._cached_cds_n_dense = max(1, int(nk_small_for_cds * self.cds_dense_ratio))
        self._cached_dense_mask = torch.zeros(
            bs, kv_heads, nk_small_for_cds, dtype=torch.bool, device=self.device,
        )
        self._cached_cds_out_scores = torch.empty(
            bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
            dtype=torch.float32, device=self.device,
        )
        self._cached_cds_out_indices = torch.empty(
            bs, kv_heads, nk_small_for_cds * self.quest_chunk_size,
            dtype=torch.int64, device=self.device,
        )
        self._cached_cds_density_scores = torch.empty(
            bs, kv_heads, nk_small_for_cds, dtype=torch.float32, device=self.device,
        )
        # Output buffer for the fused topk_gather_concat kernel.  Sized for
        # [bs, kv, final_topk + MAX_SKIPPED_INDICES] so the same buffer fits
        # all sk_count values (caller slices to actual region after launch).
        self._cached_cds_final_indices = torch.empty(
            bs, kv_heads, self.final_topk + self.MAX_SKIPPED_INDICES,
            dtype=torch.int64, device=self.device,
        )

    def _refresh_live_skipped_indices(self, kv_len: int):
        """Recompute the live skipped-token set for current retrieval length.

        skipped 定义为：当前 stage-1 chunk 选择路径无法触达、但必须并入 stage-2
        rerank 的 retrieval-zone token 索引（= retrieval 尾部未对齐到 chunk 边界
        的那点 "live tail"）。每次 _refresh_quest_chunk_cache 都按当前 kv_len
        重算并覆写 buffer，不做历史累计。
        """
        kv_len = max(0, int(kv_len))
        chunk = int(self.quest_chunk_size)
        n_chunks = kv_len // chunk if chunk > 0 else 0
        if self._use_hierarchical_quest:
            covered_small = (n_chunks // self._hq_factor) * self._hq_factor
        else:
            covered_small = n_chunks
        covered_tokens = covered_small * chunk
        skipped_count = max(0, kv_len - covered_tokens)
        # 诊断：看 "多余 / skipped" 这段到底多大（= retrieval_len mod align_chunk）。
        # 只在 refresh（每次 update / prefill 末，非 decode 热路径）打印一次。
        large_chunk = self._offload_align_chunk
        n_trailing_chunks = n_chunks - covered_small        # 完整 small chunk 数
        carry_tail = kv_len - n_chunks * chunk              # 不足 1 个 small chunk 的尾巴
        if os.environ.get("POLAR_DEBUG_SKIPPED", "0") != "0":
            print(
                f"[skipped] retrieval_len(kv_len)={kv_len} align_chunk={large_chunk} "
                f"=> skipped={skipped_count} (= kv_len % align_chunk = {kv_len % large_chunk}); "
                f"其中 trailing 完整 small chunk={n_trailing_chunks} 个 ({n_trailing_chunks * chunk} tok) "
                f"+ carry 残块={carry_tail} tok",
                flush=True,
            )
        if skipped_count > self.MAX_SKIPPED_INDICES:
            raise RuntimeError(
                "[chunk_minmax] live skipped indices overflow: "
                f"kv_len={kv_len}, covered_tokens={covered_tokens}, "
                f"skipped={skipped_count} > MAX={self.MAX_SKIPPED_INDICES}. "
                "增大 POLAR_MAX_SKIPPED_INDICES 或减小 align_chunk。"
            )
        idx_row = None
        if skipped_count > 0:
            idx_row = torch.arange(
                covered_tokens, kv_len, device=self.device, dtype=torch.int64
            ).unsqueeze(0).unsqueeze(0)
        for ldx in range(self.layer_num):
            self._chunk_skipped_count[ldx] = skipped_count
            if idx_row is not None:
                self._chunk_skipped_indices_gpu[ldx][:, :, :skipped_count].copy_(idx_row)

    # ------------------------------------------------------------------
    # Chunk-level raw K min/max metadata maintenance (Quest-style)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _kivi_pack_chunks(
        self,
        layer_idx: int,
        b_sl: slice,
        K_chunked: torch.Tensor,           # [bs_local, kv, n_chunks, chunk, D] bf16
        chunk_start: int,
        chunk_end: int,
    ) -> None:
        """KIVI-encode ``K_chunked`` and write into ``key_kivi_packed_gpu``.

        Storage layout (A-1 chunk-major):

            ``key_kivi_packed_gpu[layer]`` :
                ``[bs, kv, n_chunks_max, n_pack, group_size]`` int32

        ``chunk_start`` / ``chunk_end`` are retrieval-internal chunk
        indices (same as ``_chunk_key_min_gpu`` axis-2).  We write the
        packed codes for the freshly-completed chunks at
        ``[:, :, chunk_start:chunk_end, :, :]`` directly — no token-level
        offset / sink_size shift any more.
        """
        if chunk_end <= chunk_start:
            return
        chunk = self.quest_chunk_size
        n_chunks = chunk_end - chunk_start
        bs_local, kv, n_chunks_ck, g_check, D = K_chunked.shape
        assert n_chunks_ck == n_chunks and g_check == chunk, (
            f"kivi pack shape mismatch: K_chunked={K_chunked.shape}, "
            f"n_chunks={n_chunks}, chunk={chunk}"
        )
        # Per-(quest-)chunk per-channel min/max we just wrote.
        mn = self._chunk_key_min_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :
        ].unsqueeze(-2)                                # [bs, kv, n_chunks, 1, D]
        mx = self._chunk_key_max_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :
        ].unsqueeze(-2)
        levels = (1 << self._kivi_bits) - 1
        # bf16 ladder (matches kivi_simulate_dequant and the CUDA kernel).
        scale = (mx - mn) / float(levels)
        scale_safe = scale.clamp(min=1e-8)
        codes = ((K_chunked - mn) / scale_safe).clamp_(
            0.0, float(levels),
        ).round_().to(torch.int32)
        # Pack along head_dim axis: [bs, kv, n_chunks, chunk, n_pack] int32.
        packed = _pack_along_last_dim(codes, bits=self._kivi_bits)
        # Re-layout to chunk-major: [bs, kv, n_chunks, n_pack, chunk] so
        # that within one chunk the 16 lanes for a fixed dim-block live at
        # consecutive int32 positions.  ``.contiguous()`` materialises the
        # permuted tensor; ``.copy_()`` then does a straight memcpy into
        # the destination slice (which is also stride-1 on its inner axis).
        packed_cm = packed.transpose(-1, -2).contiguous()
        self.key_kivi_packed_gpu[layer_idx][
            b_sl, :, chunk_start:chunk_end, :, :
        ].copy_(packed_cm)




    @torch.inference_mode()
    def _compute_and_store_chunk_minmax(
        self,
        layer_idx: int,
        key_block: torch.Tensor,
        abs_token_start: int,
        batch_slice: Optional[slice] = None,
    ):
        """把一段连续 raw-K 写入 chunk 元数据（small chunk）。

        live-carry 语义：
            - 每层维护 <chunk_size 的 carry tail（上次不足一块的尾巴 token）。
            - 新 update 先用 carry + 本次开头 token 拼满 1 个 chunk，再继续处理。
            - 这样可避免 decode 长跑时周期性 head/tail hole 累积；当前 retrieval 末尾
              不足一块的 token 由 `_refresh_live_skipped_indices(kv_len)` 在下一次
              `_refresh_quest_chunk_cache` 时按 kv_len 重算后写入 skipped buffer。
        Args:
            layer_idx:        目标层
            key_block:        [bs_local, kv, n_tokens, head_dim] 一段连续的 RoPE-后 raw K
                              （要求是写入 retrieval zone 的那批 K，不含 sink/local）
            abs_token_start:  这批 token 在 retrieval zone 内的绝对起始偏移
                              （= 写入 unified_keys_*[..., retrieval_start:retrieval_end] 时的
                              retrieval_start - sink_size）
            batch_slice:      None  → 写入 buffer[:bs_local, ...]（lockstep 全 batch）
                              slice → 写入 buffer[batch_slice, ...]（单样本写入指定 slot）
        """
        chunk = self.quest_chunk_size
        bs_local, kv, n_tokens, dh = key_block.shape
        if n_tokens <= 0:
            return
        b_sl = batch_slice if batch_slice is not None else slice(0, bs_local)

        # ---- 先尝试用 carry 拼出一个完整 chunk ----
        carry_cnt = int(self._chunk_carry_count[layer_idx])
        if carry_cnt > 0:
            need = chunk - carry_cnt
            if n_tokens < need:
                if carry_cnt + n_tokens > self.quest_chunk_size - 1:
                    raise RuntimeError(
                        f"[chunk_minmax] invalid carry append: carry={carry_cnt} + n={n_tokens} > chunk-1={self.quest_chunk_size - 1}"
                    )
                self._chunk_carry_keys[layer_idx][b_sl, :, carry_cnt:carry_cnt + n_tokens, :].copy_(key_block[:, :, :n_tokens, :])
                self._chunk_carry_count[layer_idx] = carry_cnt + n_tokens
                return

            merge = self._chunk_merge_buffer[layer_idx][b_sl, :, :, :]
            merge[:, :, :carry_cnt, :].copy_(self._chunk_carry_keys[layer_idx][b_sl, :, :carry_cnt, :])
            merge[:, :, carry_cnt:, :].copy_(key_block[:, :, :need, :])

            chunk_start = (abs_token_start - carry_cnt) // chunk
            chunk_end = chunk_start + 1
            if chunk_end > self._chunk_n_max:
                raise RuntimeError(
                    f"[chunk_minmax] layer={layer_idx} chunk_end={chunk_end} exceeds "
                    f"preallocated _chunk_n_max={self._chunk_n_max}; check retrieval_zone_capacity."
                )
            self._chunk_key_min_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.amin(dim=2, keepdim=True)
            )
            self._chunk_key_max_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.amax(dim=2, keepdim=True)
            )
            self._chunk_key_centroid_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(
                merge.mean(dim=2, keepdim=True)
            )
            self._chunk_valid_mask_gpu[layer_idx][b_sl, :, chunk_start:chunk_end].fill_(True)
            if chunk_end > self._chunk_valid_end[layer_idx]:
                self._chunk_valid_end[layer_idx] = chunk_end
            self._update_large_chunk_minmax(layer_idx, chunk_start, chunk_end, b_sl)

            # KIVI pack the freshly-completed (carry + new_head) chunk.
            self._kivi_pack_chunks(
                layer_idx, b_sl,
                merge.view(bs_local, kv, 1, chunk, dh),
                chunk_start=chunk_start, chunk_end=chunk_end,
            )

            key_block = key_block[:, :, need:, :]
            abs_token_start = abs_token_start + need
            n_tokens -= need
            self._chunk_carry_count[layer_idx] = 0
            carry_cnt = 0
            if n_tokens <= 0:
                return

        # 若无 carry 但起点不对齐，说明上游时序/状态不一致（carry 丢失）。
        if abs_token_start % chunk != 0:
            raise RuntimeError(
                "[chunk_minmax] misaligned abs_token_start without carry: "
                f"layer={layer_idx}, abs={abs_token_start}, chunk={chunk}, carry={carry_cnt}"
            )

        # ---- 现在 abs_token_start 一定 chunk-aligned ----
        usable = (n_tokens // chunk) * chunk
        if usable > 0:
            n_chunks = usable // chunk
            chunk_start = abs_token_start // chunk
            chunk_end = chunk_start + n_chunks
            if chunk_end > self._chunk_n_max:
                raise RuntimeError(
                    f"[chunk_minmax] layer={layer_idx} chunk_end={chunk_end} exceeds "
                    f"preallocated _chunk_n_max={self._chunk_n_max}; check retrieval_zone_capacity."
                )
            K_chunked = key_block[:, :, :usable, :].view(bs_local, kv, n_chunks, chunk, dh)
            # amin/amax over the chunk dim → [bs_local, kv, n_chunks, head_dim]
            self._chunk_key_min_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.amin(dim=3))
            self._chunk_key_max_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.amax(dim=3))
            self._chunk_key_centroid_gpu[layer_idx][b_sl, :, chunk_start:chunk_end, :].copy_(K_chunked.mean(dim=3))
            self._chunk_valid_mask_gpu[layer_idx][b_sl, :, chunk_start:chunk_end].fill_(True)
            if chunk_end > self._chunk_valid_end[layer_idx]:
                self._chunk_valid_end[layer_idx] = chunk_end

            self._update_large_chunk_minmax(layer_idx, chunk_start, chunk_end, b_sl)

            # KIVI pack the aligned full-chunk region.
            self._kivi_pack_chunks(
                layer_idx, b_sl, K_chunked,
                chunk_start=chunk_start, chunk_end=chunk_end,
            )

        # ---- 尾部不足一个 small chunk：写入 carry（live，不进历史累计 skipped）----
        tail = n_tokens - usable
        if tail > 0:
            if tail > self.quest_chunk_size - 1:
                raise RuntimeError(
                    f"[chunk_minmax] invalid tail={tail} for chunk={self.quest_chunk_size}"
                )
            self._chunk_carry_keys[layer_idx][b_sl, :, :tail, :].copy_(
                key_block[:, :, usable:usable + tail, :]
            )
            self._chunk_carry_count[layer_idx] = tail
        else:
            self._chunk_carry_count[layer_idx] = 0

    @torch.inference_mode()
    def _update_large_chunk_minmax(
        self,
        layer_idx: int,
        chunk_start: int,
        chunk_end: int,
        b_sl: slice,
    ):
        """Derive large-chunk min/max from base-level (small chunk) results.

        For each large chunk group that has any newly written small chunks,
        re-derive its bbox from the full range of small-chunk min/max values.
        """
        factor = self._hq_factor
        g_start = chunk_start // factor
        g_end = (chunk_end + factor - 1) // factor
        g_end = min(g_end, self._chunk_n_max_large)
        if g_start >= g_end:
            return

        base_min = self._chunk_key_min_gpu[layer_idx][b_sl]
        base_max = self._chunk_key_max_gpu[layer_idx][b_sl]
        valid_end = self._chunk_valid_end[layer_idx]

        for g in range(g_start, g_end):
            b_lo = g * factor
            b_hi = min(b_lo + factor, valid_end)
            if b_hi <= b_lo:
                continue
            self._chunk_key_min_gpu_large[layer_idx][b_sl, :, g, :] = base_min[:, :, b_lo:b_hi, :].amin(dim=2)
            self._chunk_key_max_gpu_large[layer_idx][b_sl, :, g, :] = base_max[:, :, b_lo:b_hi, :].amax(dim=2)
            self._chunk_valid_mask_gpu_large[layer_idx][b_sl, :, g].fill_(True)

        if g_end > self._chunk_valid_end_large[layer_idx]:
            self._chunk_valid_end_large[layer_idx] = g_end



    def _topk_3d(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k on last dim [bs, kv_heads, L] → indices [bs, kv_heads, k].

        Production speed path always uses the fused float top-k CUDA backend.

        Caller contract:
        - ``k`` MUST be a Python int (all current call sites source from
          cached ints set at init).
        - ``scores`` MUST already be fp32; callers pass pre-allocated
          fp32 buffers (``_quest_scores_buffer_large``, ``_quest_scores_buffer_sub``,
          ``_cached_cds_density_scores``, ``_cached_cds_out_scores``).

        Defensive ``int(k)`` / ``k > 0`` / ``scores.float()`` were removed:
        the int cast is no-op (~30 ns) but ``scores.float()`` carried ~1-2 μs
        of PyTorch dispatch overhead even on already-fp32 inputs (≈200-400 μs
        per decode token over 5 × 36 calls).  ``float_topk.cu`` itself
        auto-casts non-fp32 inputs and asserts ``k > 0``, so real misuse is
        still caught loudly.
        """
        return float_topk_3d(scores, k)




    def _quest_pick_coarse_indices_flat(
        self,
        *,
        raw_q: torch.Tensor,
        layer_idx: int,
    ):
        """Single-level Quest: score all small chunks once, then pick top chunks."""
        n_chunks = self._cached_quest_n_chunks
        nk = self._cached_quest_nk
        if n_chunks <= 0 or nk <= 0:
            chunk_idx = torch.zeros(
                raw_q.shape[0], raw_q.shape[1], 1,
                dtype=torch.int64, device=raw_q.device,
            )
            return chunk_idx, 0

        scores_out = self._cached_scores_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        chunk_valid = self._cached_chunk_valid_views[layer_idx]
        self._quest_kernel.quest_chunk_score(
            raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid,
        )
        chunk_idx = self._topk_3d(scores_out, nk).clamp(min=0, max=n_chunks - 1)
        return chunk_idx, nk




    def _quest_pick_coarse_indices_hierarchical(
        self,
        *,
        raw_q: torch.Tensor,
        layer_idx: int,
    ):
        """Two-level hierarchical Quest: large chunk → sub-chunk scoring.

        Stage 1: score all large chunks, pick top nk_large.
        Stage 2: for each selected large chunk, score its sub-chunks via
                 offset-based CUDA kernel, then pick top nk_small globally.
        """
        n_chunks = self._cached_quest_n_chunks
        n_large = self._cached_hq_n_large
        nk_large = self._cached_hq_nk_large
        nk_small = self._cached_hq_nk_small
        factor = self._hq_factor

        if n_large <= 0:
            chunk_idx = torch.zeros(raw_q.shape[0], raw_q.shape[1], 1,
                                    dtype=torch.int64, device=raw_q.device)
            return chunk_idx, 0

        # --- Stage 1: score large chunks ---
        # nvtx.range_push("hq1_large")
        large_scores = self._cached_hq_scores_large_view
        large_min = self._cached_hq_large_min_views[layer_idx]
        large_max = self._cached_hq_large_max_views[layer_idx]
        large_valid = self._cached_hq_large_valid_views[layer_idx]
        self._quest_kernel.quest_chunk_score(
            raw_q, large_min, large_max, large_scores, n_large, large_valid,
        )
        # nvtx.range_pop()

        # nvtx.range_push("large_topk")
        large_idx = self._topk_3d(large_scores, nk_large)
        large_idx = large_idx.clamp(min=0, max=n_large - 1)
        # nvtx.range_pop()  # hq_stage1_large_topk

        # --- Stage 2: score sub-chunks within selected large chunks ---
        # nvtx.range_push("hq2_sub")
        sub_scores = self._cached_hq_scores_sub_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        self._quest_kernel.quest_sub_chunk_score(
            raw_q, chunk_min, chunk_max,
            large_idx, sub_scores,
            nk_large, factor,
        )
        # nvtx.range_pop()

        # nvtx.range_push("sub_topk")
        sub_topk_pos = self._topk_3d(sub_scores, nk_small)
        # nvtx.range_pop()

        # --- Map back to global small chunk indices ---
        # nvtx.range_push("hq_map_back")
        chunk_idx = self._cached_hq_chunk_idx
        self._quest_kernel.quest_map_back(
            large_idx,
            sub_topk_pos,
            chunk_idx,
            factor,
            n_chunks,
        )
        # nvtx.range_pop()

        return chunk_idx, nk_small




    # ------------------------------------------------------------------
    # CDS: Coarse → Dense/Sparse rerank + final topk
    # ------------------------------------------------------------------
    def _cds_dense_sparse_select(
        self,
        chunk_idx: torch.Tensor,
        bs: int,
        key_kivi_packed: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        skipped_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """KIVI fused dense/sparse rerank + global top-k.

        1. Classify dense chunks via float_topk on centroid @ q
        2. Per-chunk KIVI on-the-fly dequant + raw_q dot + bitonic top-K
        3. Global top-K across nk*chunk_size token candidates.

        NVTX names match the RaBitQ pipeline (``cds_classify`` /
        ``cds_fused_kernel`` / ``cds_global_topk`` / ``cds_skipped``) so
        existing nsys reports line up.
        """
        chunk = self.quest_chunk_size
        # KIVI's dense/sparse "topk" is per-(quest-)chunk, must fit within
        # ``chunk``.  RaBitQ defaults dense_topk=16/sparse_topk=8 are for
        # chunk=16; clamp for smaller chunks.
        sparse_topk = max(1, min(int(self.cds_sparse_topk), chunk))
        dense_topk = max(1, min(int(self.cds_dense_topk), chunk))

        # --- Step 1: classify dense/sparse by centroid @ q ---
        nvtx.range_push("cds_classify")
        centroid_all = self._chunk_key_centroid_gpu[layer_idx][:bs]
        density_scores = self._cached_cds_density_scores
        partial_chunk_density_scores_cuda(
            chunk_idx, centroid_all, raw_q, density_scores,
            kernel=self._rerank_topk_kernel,
        )
        n_dense = self._cached_cds_n_dense
        dense_positions = self._topk_3d(density_scores, n_dense)
        dense_mask = self._cached_dense_mask
        dense_mask_from_topk_cuda(
            dense_positions, dense_mask, kernel=self._rerank_topk_kernel,
        )
        nvtx.range_pop()

        # --- Step 2: fused KIVI dequant + dot + per-chunk top-K ---
        nvtx.range_push("cds_fused_kernel")
        out_scores = self._cached_cds_out_scores
        out_indices = self._cached_cds_out_indices
        chunk_min_layer = self._chunk_key_min_gpu[layer_idx][:bs]
        chunk_max_layer = self._chunk_key_max_gpu[layer_idx][:bs]
        partial_chunk_kivi_qk_dense_sparse_cuda(
            chunk_ids=chunk_idx,
            dense_mask=dense_mask,
            packed_K=key_kivi_packed,
            chunk_min=chunk_min_layer,
            chunk_max=chunk_max_layer,
            raw_q=raw_q,
            dense_topk=dense_topk,
            sparse_topk=sparse_topk,
            group_size=chunk,
            out_scores=out_scores,
            out_indices=out_indices,
            kernel=self._kivi_kernel,
        )
        nvtx.range_pop()

        # --- Step 3: fused global top-k + gather + skipped concat ---
        nvtx.range_push("cds_global_topk_fused")
        actual_topk = min(self.final_topk, out_scores.shape[-1])
        if skipped_indices is None:
            skipped_for_kernel = self._cached_cds_final_indices[:bs, :, :0]
            sk_count = 0
        else:
            skipped_for_kernel = skipped_indices
            sk_count = skipped_indices.shape[-1]
            
        out_buf = self._cached_cds_final_indices[:bs]
        float_topk_gather_concat_3d(out_scores, out_indices, skipped_for_kernel, out_buf, actual_topk)
        
        final_indices = out_buf[:, :, :actual_topk + sk_count].contiguous()
        nvtx.range_pop()

        return final_indices





    def collision_based_topk_batch(
        self,
        query,
        key_kivi_packed=None,
        layer_idx=None,
    ):
        """Quest stage-1 + CDS dense/sparse + KIVI 4-bit final topk.

        Name kept for backward compatibility with `compute()` / timing keys.

        Args:
            query:             [bs, kv_heads, 1, head_dim] raw query
            key_kivi_packed:   [bs, kv_heads, n_chunks_pkd, n_pack, group_size]
                               int32 — KIVI packed K (A-1 chunk-major) sliced
                               on the chunk axis to ``n_chunks_pkd``
            layer_idx:         layer index (used by chunk-metadata lookup)
        Returns:
            topk_indices:      [bs, kv_heads, final_topk] retrieval-zone indices
        """
        assert key_kivi_packed is not None, \
            "KIVI path requires key_kivi_packed"

        # KIVI has no query encode work; fold the trivial shape ops into
        # stage1 chunk picking.
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1, "Decode 阶段每次只处理 1 个 token"
        query_squeezed = query.squeeze(2)  # [bs, kv_heads, head_dim]

        nvtx.range_push("stage1_coarse")
        sk_count = self._chunk_skipped_count[layer_idx]
        skipped = None
        if sk_count > 0:
            # _chunk_skipped_indices_gpu is filled by torch.arange over the
            # in-range token span (see the ``_compute_and_store_chunk_minmax``
            # tail-skip path), so values are already in [0, kv_len).  No
            # defensive clamp needed.
            skipped = self._chunk_skipped_indices_gpu[layer_idx][:bs, :, :sk_count]

        if self._use_hierarchical_quest:
            chunk_idx, _nk = self._quest_pick_coarse_indices_hierarchical(
                raw_q=query_squeezed,
                layer_idx=layer_idx,
            )
        else:
            chunk_idx, _nk = self._quest_pick_coarse_indices_flat(
                raw_q=query_squeezed,
                layer_idx=layer_idx,
            )
        nvtx.range_pop()  # stage1_chunk_pick

        if self._quest_bound_check:
            self._quest_bound_probe(query_squeezed, layer_idx)

        nvtx.range_push("stage2_rerank")
        topk_indices = self._cds_dense_sparse_select(
            chunk_idx, bs,
            key_kivi_packed,
            query_squeezed, layer_idx,
            skipped_indices=skipped,
        )
        nvtx.range_pop()  # stage2_cds

        if self._allayer_recall_check:
            self._allayer_recall_ablation(
                topk_indices, query_squeezed, layer_idx, tag="Final",
            )
            chunk = self.quest_chunk_size
            offsets = torch.arange(chunk, device=chunk_idx.device, dtype=chunk_idx.dtype)
            coarse_tokens = (
                (chunk_idx * chunk).unsqueeze(-1) + offsets
            ).reshape(bs, kv_heads, -1).contiguous()
            self._allayer_recall_ablation(
                coarse_tokens, query_squeezed, layer_idx, tag="CoarsePool",
            )   

        return topk_indices


    @staticmethod
    def _quest_bound_summary(
        upper_bound: torch.Tensor,
        true_max: torch.Tensor,
    ):
        gap = upper_bound - true_max
        relative_gap = gap / true_max.abs().clamp_min(1e-6)
        quantile_points = torch.tensor(
            [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0],
            dtype=torch.float32,
            device=gap.device,
        )
        gap_quantiles = torch.quantile(gap, quantile_points)
        relative_quantiles = torch.quantile(relative_gap, quantile_points)
        upper_centered = upper_bound - upper_bound.mean()
        true_centered = true_max - true_max.mean()
        corr_denom = torch.sqrt(
            upper_centered.square().sum() * true_centered.square().sum()
        ).clamp_min(1e-12)
        correlation = (upper_centered * true_centered).sum() / corr_denom
        labels = ("min", "p01", "p05", "p25", "p50", "p75", "p95", "p99", "max")
        return {
            "num_chunks": int(gap.numel()),
            "upper_bound_mean": float(upper_bound.mean().item()),
            "true_max_mean": float(true_max.mean().item()),
            "gap_mean": float(gap.mean().item()),
            "gap_std": float(gap.std(unbiased=False).item()),
            "relative_gap_mean": float(relative_gap.mean().item()),
            "pearson_correlation": float(correlation.item()),
            "violation_count": int((gap < -1e-3).sum().item()),
            "violation_rate": float((gap < -1e-3).float().mean().item()),
            "gap_quantiles": {
                label: float(value)
                for label, value in zip(labels, gap_quantiles.cpu().tolist())
            },
            "relative_gap_quantiles": {
                label: float(value)
                for label, value in zip(labels, relative_quantiles.cpu().tolist())
            },
        }


    @staticmethod
    def _quest_training_chunk_indices(
        true_max: torch.Tensor,
        upper_bound: torch.Tensor,
        max_chunks: int,
    ) -> torch.Tensor:
        """Keep a deterministic mix of uniform, high-target, and high-bound chunks."""
        num_chunks = int(true_max.numel())
        if max_chunks <= 0 or num_chunks <= max_chunks:
            return torch.arange(num_chunks, device=true_max.device, dtype=torch.long)

        top_target_count = max(1, max_chunks // 4)
        top_bound_count = max(1, max_chunks // 4)
        uniform_count = max(1, max_chunks - top_target_count - top_bound_count)
        top_target = torch.topk(true_max, k=top_target_count).indices
        top_bound = torch.topk(upper_bound, k=top_bound_count).indices
        uniform = torch.linspace(
            0,
            num_chunks - 1,
            steps=uniform_count,
            device=true_max.device,
        ).round().long()
        return torch.unique(
            torch.cat((top_target, top_bound, uniform)),
            sorted=True,
        )


    @staticmethod
    def _quest_training_sample_split(sample_id: str) -> str:
        forced = os.environ.get("POLAR_QUEST_BOUND_SAMPLE_SPLIT", "").strip().lower()
        if forced in ("train", "val", "calibration", "test"):
            return forced
        bucket = int.from_bytes(
            hashlib.blake2b(sample_id.encode(), digest_size=2).digest(),
            "little",
        ) % 10
        if bucket <= 6:
            return "train"
        if bucket == 7:
            return "val"
        if bucket == 8:
            return "calibration"
        return "test"


    def _save_quest_training_shards(
        self,
        q: torch.Tensor,
        keys: torch.Tensor,
        step: int,
        layer_idx: int,
        head_idx: int,
    ):
        """Save per-chunk features and exact targets for offline predictor training."""
        if (
            not self._quest_bound_pt_dir
            or not self._quest_bound_save_chunk_shards
            or self._quest_bound_compact
        ):
            return

        sample_id = os.environ.get("POLAR_QUEST_BOUND_SAMPLE_ID", "unknown")
        sample_idx = int(os.environ.get("POLAR_QUEST_BOUND_SAMPLE_IDX", "-1"))
        sample_split = self._quest_training_sample_split(sample_id)
        safe_sample_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in sample_id
        )
        os.makedirs(self._quest_bound_pt_dir, exist_ok=True)

        for chunk_size in self._quest_bound_chunk_sizes:
            aligned_tokens = (int(keys.shape[0]) // chunk_size) * chunk_size
            if aligned_tokens <= 0:
                continue
            chunk_keys = keys[:aligned_tokens].view(-1, chunk_size, self.head_dim)
            chunk_min = chunk_keys.amin(dim=1)
            chunk_max = chunk_keys.amax(dim=1)
            chunk_mean = chunk_keys.mean(dim=1)
            token_scores = torch.matmul(chunk_keys, q)
            true_max = token_scores.amax(dim=1)
            upper_bound = torch.maximum(chunk_min * q, chunk_max * q).sum(dim=-1)
            selected = self._quest_training_chunk_indices(
                true_max,
                upper_bound,
                (
                    0
                    if sample_split in self._quest_bound_full_splits
                    else self._quest_bound_max_chunks
                ),
            )

            shard = {
                "meta": {
                    "sample_idx": sample_idx,
                    "sample_id": sample_id,
                    "sample_split": sample_split,
                    "context_tokens": int(self.input_length),
                    "retrieval_tokens": int(keys.shape[0]),
                    "step": int(step),
                    "layer": int(layer_idx),
                    "kv_head": int(head_idx),
                    "chunk_size": int(chunk_size),
                    "num_chunks_total": int(true_max.numel()),
                    "num_chunks_saved": int(selected.numel()),
                    "head_dim": int(self.head_dim),
                    "query_type": "post_rope_gqa_mean",
                    "score_type": "raw_qk",
                },
                "chunk_indices": selected.to(torch.int32).cpu(),
                "q": q.to(torch.float16).cpu(),
                "chunk_min": chunk_min[selected].to(torch.float16).cpu(),
                "chunk_max": chunk_max[selected].to(torch.float16).cpu(),
                "chunk_mean": chunk_mean[selected].to(torch.float16).cpu(),
                "true_max": true_max[selected].to(torch.float32).cpu(),
                "upper_bound": upper_bound[selected].to(torch.float32).cpu(),
            }
            filename = (
                f"{safe_sample_id}_idx{sample_idx}_step{step}_"
                f"L{layer_idx}_h{head_idx}_cs{chunk_size}.pt"
            )
            torch.save(shard, os.path.join(self._quest_bound_pt_dir, filename))


    def _save_quest_token_scores(
        self,
        exact_scores: torch.Tensor,
        step: int,
        layer_idx: int,
        head_idx: int,
    ):
        """Save exact per-key QK scores for held-out key-level recall."""
        if not self._quest_bound_token_dir:
            return
        sample_id = os.environ.get("POLAR_QUEST_BOUND_SAMPLE_ID", "unknown")
        sample_idx = int(os.environ.get("POLAR_QUEST_BOUND_SAMPLE_IDX", "-1"))
        safe_sample_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in sample_id
        )
        os.makedirs(self._quest_bound_token_dir, exist_ok=True)
        payload = {
            "meta": {
                "sample_idx": sample_idx,
                "sample_id": sample_id,
                "sample_split": self._quest_training_sample_split(sample_id),
                "context_tokens": int(self.input_length),
                "retrieval_tokens": int(exact_scores.numel()),
                "step": int(step),
                "layer": int(layer_idx),
                "kv_head": int(head_idx),
                "score_type": "exact_raw_qk_per_key",
            },
            "exact_qk": exact_scores.to(torch.float32).cpu(),
        }
        filename = (
            f"{safe_sample_id}_idx{sample_idx}_step{step}_"
            f"L{layer_idx}_h{head_idx}_token_qk.pt"
        )
        torch.save(
            payload,
            os.path.join(self._quest_bound_token_dir, filename),
        )


    def _quest_bound_probe(self, raw_q: torch.Tensor, layer_idx: int):
        """Compare Quest bounds with exact per-chunk q·k maxima."""
        step = int(self._decode_step_counter)
        probe_key = (step, int(layer_idx))
        if (
            step not in self._quest_bound_steps
            or layer_idx not in self._quest_bound_layers
            or probe_key in self._quest_bound_written
        ):
            return

        n_small = int(self._cached_quest_n_chunks)
        n_large = int(self._cached_hq_n_large) if self._use_hierarchical_quest else 0
        if n_small <= 0:
            return
        self._quest_bound_written.add(probe_key)

        # Compact shards are written from compute(); skip redundant per-head work
        # unless jsonl or token-score dumps are requested.
        if (
            self._quest_bound_compact
            and not self._quest_bound_save_jsonl
            and not self._quest_bound_token_dir
        ):
            if (
                self._quest_bound_exit_after_probe
                and layer_idx == max(self._quest_bound_layers)
                and step == max(self._quest_bound_steps)
            ):
                print("[Quest Bound] requested probes completed", flush=True)
                raise SystemExit(0)
            return

        if self._quest_bound_save_jsonl:
            output_dir = os.path.dirname(self._quest_bound_output)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

        heads = range(self.kv_head)
        if self._quest_bound_heads is not None:
            heads = sorted(h for h in self._quest_bound_heads if 0 <= h < self.kv_head)

        small_chunk = int(self.quest_chunk_size)
        large_chunk = int(self._hq_large_chunk_size)
        small_token_count = n_small * small_chunk
        large_token_count = n_large * large_chunk

        with torch.no_grad():
            output = (
                open(self._quest_bound_output, "a", encoding="utf-8")
                if self._quest_bound_save_jsonl
                else None
            )
            for head_idx in heads:
                q = raw_q[0, head_idx].to(torch.float32)
                keys = self.unified_keys_cpu[layer_idx][
                    0, head_idx, :small_token_count, :
                ].to(self.device, dtype=torch.float32, non_blocking=True)
                exact_scores = torch.matmul(keys, q)
                self._save_quest_token_scores(
                    exact_scores,
                    step,
                    layer_idx,
                    head_idx,
                )
                self._save_quest_training_shards(
                    q,
                    keys,
                    step,
                    layer_idx,
                    head_idx,
                )

                small_upper = torch.maximum(
                    self._cached_chunk_min_views[layer_idx][0, head_idx, :n_small].float()
                    * q,
                    self._cached_chunk_max_views[layer_idx][0, head_idx, :n_small].float()
                    * q,
                ).sum(dim=-1)
                small_true = exact_scores.view(n_small, small_chunk).amax(dim=-1)
                small_valid = self._cached_chunk_valid_views[layer_idx][
                    0, head_idx, :n_small
                ]
                small_upper = small_upper[small_valid]
                small_true = small_true[small_valid]
                records = [("small", small_chunk, small_upper, small_true)]

                if n_large > 0:
                    large_upper = torch.maximum(
                        self._cached_hq_large_min_views[layer_idx][
                            0, head_idx, :n_large
                        ].float()
                        * q,
                        self._cached_hq_large_max_views[layer_idx][
                            0, head_idx, :n_large
                        ].float()
                        * q,
                    ).sum(dim=-1)
                    large_true = exact_scores[:large_token_count].view(
                        n_large, large_chunk
                    ).amax(dim=-1)
                    large_valid = self._cached_hq_large_valid_views[layer_idx][
                        0, head_idx, :n_large
                    ]
                    records.append(
                        (
                            "large",
                            large_chunk,
                            large_upper[large_valid],
                            large_true[large_valid],
                        )
                    )

                for level, chunk_size, upper_bound, true_max in records:
                    record = {
                        "step": step,
                        "layer": int(layer_idx),
                        "kv_head": int(head_idx),
                        "level": level,
                        "chunk_size": chunk_size,
                        "context_tokens": int(self.input_length),
                        "query_norm": float(q.norm().item()),
                        **self._quest_bound_summary(upper_bound, true_max),
                    }
                    if self._quest_bound_save_values:
                        record["upper_bound"] = upper_bound.cpu().tolist()
                        record["true_max"] = true_max.cpu().tolist()
                    if output is not None:
                        output.write(json.dumps(record) + "\n")
                if output is not None:
                    output.flush()
            if output is not None:
                output.close()

        print(
            f"[Quest Bound] step={step} layer={layer_idx} "
            f"small={n_small} large={n_large} "
            f"jsonl={self._quest_bound_output if self._quest_bound_save_jsonl else 'off'} "
            f"pt={self._quest_bound_pt_dir or 'off'}",
            flush=True,
        )
        if (
            self._quest_bound_exit_after_probe
            and layer_idx == max(self._quest_bound_layers)
            and step == max(self._quest_bound_steps)
        ):
            print("[Quest Bound] requested probes completed", flush=True)
            raise SystemExit(0)

    @staticmethod
    def _topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
        k = min(int(k), int(scores.numel()))
        if k <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)
        return torch.topk(scores, k=k, largest=True, sorted=False).indices

    @staticmethod
    def _set_overlap_metrics(a: torch.Tensor, b: torch.Tensor):
        a_set = set(int(x) for x in a.detach().cpu().tolist())
        b_set = set(int(x) for x in b.detach().cpu().tolist())
        inter = len(a_set & b_set)
        union = len(a_set | b_set)
        return {
            "overlap": inter / max(len(a_set), 1),
            "jaccard": inter / max(union, 1),
            "intersection": inter,
            "union": union,
            "size_a": len(a_set),
            "size_b": len(b_set),
        }

    def _gqa_query_agg_probe(self, query_group: torch.Tensor, layer_idx: int):
        """Compare Avg / PerQueryUnion / GroupMax Top-K without changing retrieval."""
        step = int(self._decode_step_counter)
        sample_id = os.environ.get("POLAR_QUEST_BOUND_SAMPLE_ID", "unknown")
        probe_key = (sample_id, step, int(layer_idx))
        if (
            step not in self._gqa_agg_steps
            or layer_idx not in self._gqa_agg_layers
            or probe_key in self._gqa_agg_written
        ):
            return
        if not (
            self.enable_offload
            and self.unified_keys_cpu is not None
            and self.cpu_valid_end is not None
        ):
            return
        token_count = int(self.cpu_valid_end[layer_idx])
        if token_count <= 0:
            return
        self._gqa_agg_written.add(probe_key)

        output_dir = os.path.dirname(self._gqa_agg_output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        heads = range(self.kv_head)
        if self._gqa_agg_heads is not None:
            heads = sorted(h for h in self._gqa_agg_heads if 0 <= h < self.kv_head)

        sample_idx = int(os.environ.get("POLAR_QUEST_BOUND_SAMPLE_IDX", "-1"))
        q_group = query_group[0].to(torch.float32)  # [kv, G, D]
        group_size = int(q_group.shape[1])

        with torch.no_grad():
            with open(self._gqa_agg_output, "a", encoding="utf-8") as output:
                for head_idx in heads:
                    keys = self.unified_keys_cpu[layer_idx][
                        0, head_idx, :token_count, :
                    ].to(self.device, dtype=torch.float32, non_blocking=True)
                    q_heads = q_group[head_idx]  # [G, D]
                    q_avg = q_heads.mean(dim=0)
                    scores = torch.matmul(keys, q_heads.transpose(0, 1))  # [S, G]
                    scores_avg = torch.matmul(keys, q_avg)  # [S]
                    scores_gmax = scores.amax(dim=-1)

                    for ratio in self._gqa_agg_ratios:
                        token_k = max(1, int(round(float(ratio) * token_count)))
                        avg_idx = self._topk_indices(scores_avg, token_k)
                        gmax_idx = self._topk_indices(scores_gmax, token_k)
                        per_query_idx = [
                            self._topk_indices(scores[:, g], token_k)
                            for g in range(group_size)
                        ]
                        union_set = set()
                        for idxs in per_query_idx:
                            union_set.update(int(x) for x in idxs.cpu().tolist())
                        avg_set = set(int(x) for x in avg_idx.cpu().tolist())
                        gmax_set = set(int(x) for x in gmax_idx.cpu().tolist())
                        per_query_recall = [
                            len(avg_set & set(int(x) for x in idxs.cpu().tolist()))
                            / max(token_k, 1)
                            for idxs in per_query_idx
                        ]
                        avg_gmax = self._set_overlap_metrics(avg_idx, gmax_idx)
                        record = {
                            "kind": "token",
                            "sample_id": sample_id,
                            "sample_idx": sample_idx,
                            "step": step,
                            "layer": int(layer_idx),
                            "kv_head": int(head_idx),
                            "retrieval_tokens": token_count,
                            "group_size": group_size,
                            "ratio": float(ratio),
                            "budget_k": token_k,
                            "avg_vs_groupmax": avg_gmax,
                            "union_size": len(union_set),
                            "avg_covers_union": len(avg_set & union_set)
                            / max(len(union_set), 1),
                            "groupmax_covers_union": len(gmax_set & union_set)
                            / max(len(union_set), 1),
                            "per_query_recall_vs_avg": per_query_recall,
                            "per_query_recall_mean": float(
                                sum(per_query_recall) / max(len(per_query_recall), 1)
                            ),
                            "per_query_recall_min": float(min(per_query_recall)),
                            "key_recall_avg_vs_gmax_gt": avg_gmax["overlap"],
                            "key_recall_delta_gmax_minus_avg": 0.0,
                        }
                        output.write(json.dumps(record) + "\n")

                        # Chunk-level equal-budget comparison + chunk-pool key recall.
                        gt_keys = gmax_set
                        for chunk_size in self._gqa_agg_chunk_sizes:
                            aligned = (token_count // chunk_size) * chunk_size
                            if aligned <= 0:
                                continue
                            n_chunks = aligned // chunk_size
                            chunk_budget = max(
                                1, int(round(float(ratio) * n_chunks))
                            )
                            avg_chunk = (
                                scores_avg[:aligned]
                                .view(n_chunks, chunk_size)
                                .amax(dim=-1)
                            )
                            gmax_chunk = (
                                scores_gmax[:aligned]
                                .view(n_chunks, chunk_size)
                                .amax(dim=-1)
                            )
                            per_q_chunk = [
                                scores[:aligned, g]
                                .view(n_chunks, chunk_size)
                                .amax(dim=-1)
                                for g in range(group_size)
                            ]
                            avg_c_idx = self._topk_indices(avg_chunk, chunk_budget)
                            gmax_c_idx = self._topk_indices(gmax_chunk, chunk_budget)
                            per_q_c_idx = [
                                self._topk_indices(pc, chunk_budget)
                                for pc in per_q_chunk
                            ]
                            union_c = set()
                            for idxs in per_q_c_idx:
                                union_c.update(int(x) for x in idxs.cpu().tolist())
                            avg_c_set = set(int(x) for x in avg_c_idx.cpu().tolist())
                            gmax_c_set = set(
                                int(x) for x in gmax_c_idx.cpu().tolist()
                            )
                            per_q_c_recall = [
                                len(
                                    avg_c_set
                                    & set(int(x) for x in idxs.cpu().tolist())
                                )
                                / max(chunk_budget, 1)
                                for idxs in per_q_c_idx
                            ]
                            avg_gmax_c = self._set_overlap_metrics(
                                avg_c_idx, gmax_c_idx
                            )

                            def _chunk_pool(chunk_ids):
                                pool = set()
                                for cid in chunk_ids:
                                    start = int(cid) * chunk_size
                                    pool.update(range(start, start + chunk_size))
                                return pool

                            avg_pool = _chunk_pool(avg_c_set)
                            gmax_pool = _chunk_pool(gmax_c_set)
                            key_recall_avg = len(avg_pool & gt_keys) / max(
                                len(gt_keys), 1
                            )
                            key_recall_gmax = len(gmax_pool & gt_keys) / max(
                                len(gt_keys), 1
                            )
                            chunk_record = {
                                "kind": "chunk",
                                "sample_id": sample_id,
                                "sample_idx": sample_idx,
                                "step": step,
                                "layer": int(layer_idx),
                                "kv_head": int(head_idx),
                                "retrieval_tokens": token_count,
                                "group_size": group_size,
                                "chunk_size": int(chunk_size),
                                "num_chunks": int(n_chunks),
                                "ratio": float(ratio),
                                "budget_k": int(chunk_budget),
                                "avg_vs_groupmax": avg_gmax_c,
                                "union_size": len(union_c),
                                "avg_covers_union": len(avg_c_set & union_c)
                                / max(len(union_c), 1),
                                "groupmax_covers_union": len(gmax_c_set & union_c)
                                / max(len(union_c), 1),
                                "per_query_recall_vs_avg": per_q_c_recall,
                                "per_query_recall_mean": float(
                                    sum(per_q_c_recall)
                                    / max(len(per_q_c_recall), 1)
                                ),
                                "per_query_recall_min": float(min(per_q_c_recall)),
                                "key_recall_avg": key_recall_avg,
                                "key_recall_groupmax": key_recall_gmax,
                                "key_recall_delta_gmax_minus_avg": (
                                    key_recall_gmax - key_recall_avg
                                ),
                            }
                            output.write(json.dumps(chunk_record) + "\n")

        print(
            f"[GQA Agg] step={step} layer={layer_idx} tokens={token_count} "
            f"jsonl={self._gqa_agg_output}",
            flush=True,
        )
        if (
            self._gqa_agg_exit_after_probe
            and layer_idx == max(self._gqa_agg_layers)
            and step == max(self._gqa_agg_steps)
        ):
            print("[GQA Agg] requested probes completed", flush=True)
            raise SystemExit(0)

    def _save_quest_compact_layer_shards(
        self,
        query_group: torch.Tensor,
        layer_idx: int,
    ):
        """Save merged q_group + shared K stats for one sample/step/layer."""
        if (
            not self._quest_bound_pt_dir
            or not self._quest_bound_save_chunk_shards
            or not self._quest_bound_compact
        ):
            return
        step = int(self._decode_step_counter)
        sample_id = os.environ.get("POLAR_QUEST_BOUND_SAMPLE_ID", "unknown")
        probe_key = ("compact", sample_id, step, int(layer_idx))
        if (
            step not in self._quest_bound_steps
            or layer_idx not in self._quest_bound_layers
            or probe_key in self._quest_bound_written
        ):
            return
        if not (
            self.enable_offload
            and self.unified_keys_cpu is not None
            and self.cpu_valid_end is not None
        ):
            return
        token_count = int(self.cpu_valid_end[layer_idx])
        if token_count <= 0:
            return
        self._quest_bound_written.add(probe_key)

        sample_idx = int(os.environ.get("POLAR_QUEST_BOUND_SAMPLE_IDX", "-1"))
        sample_split = self._quest_training_sample_split(sample_id)
        safe_sample_id = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in sample_id
        )
        os.makedirs(self._quest_bound_pt_dir, exist_ok=True)

        q_group = query_group[0].to(torch.float32)  # [kv, G, D]
        heads = range(self.kv_head)
        if self._quest_bound_heads is not None:
            heads = sorted(h for h in self._quest_bound_heads if 0 <= h < self.kv_head)
        heads = list(heads)

        heads_payload = {}
        with torch.no_grad():
            for head_idx in heads:
                keys = self.unified_keys_cpu[layer_idx][
                    0, head_idx, :token_count, :
                ].to(self.device, dtype=torch.float32, non_blocking=True)
                q_h = q_group[head_idx]  # [G, D]
                q_avg = q_h.mean(dim=0)
                head_entry = {
                    "q_group": q_h.to(torch.float16).cpu(),  # [4, D]
                    "q_avg": q_avg.to(torch.float16).cpu(),
                    "chunk_sizes": {},
                }
                for chunk_size in self._quest_bound_chunk_sizes:
                    aligned = (token_count // chunk_size) * chunk_size
                    if aligned <= 0:
                        continue
                    chunk_keys = keys[:aligned].view(-1, chunk_size, self.head_dim)
                    chunk_min = chunk_keys.amin(dim=1)
                    chunk_max = chunk_keys.amax(dim=1)
                    chunk_mean = chunk_keys.mean(dim=1)
                    token_scores = torch.einsum(
                        "ncd,gd->ngc", chunk_keys, q_h
                    )  # [N, G, C]
                    true_max_per_query = token_scores.amax(dim=-1)  # [N, G]
                    true_max_avg = torch.matmul(
                        chunk_keys, q_avg
                    ).amax(dim=-1)  # [N]
                    true_max_groupmax = true_max_per_query.amax(dim=-1)
                    true_max_logsumexp = torch.logsumexp(
                        true_max_per_query, dim=-1
                    )
                    upper_bound_avg = torch.maximum(
                        chunk_min * q_avg, chunk_max * q_avg
                    ).sum(dim=-1)
                    selected = self._quest_training_chunk_indices(
                        true_max_groupmax,
                        upper_bound_avg,
                        (
                            0
                            if sample_split in self._quest_bound_full_splits
                            else self._quest_bound_max_chunks
                        ),
                    )
                    head_entry["chunk_sizes"][str(int(chunk_size))] = {
                        "chunk_indices": selected.to(torch.int32).cpu(),
                        "chunk_min": chunk_min[selected].to(torch.float16).cpu(),
                        "chunk_max": chunk_max[selected].to(torch.float16).cpu(),
                        "chunk_mean": chunk_mean[selected].to(torch.float16).cpu(),
                        "true_max_per_query": true_max_per_query[selected]
                        .to(torch.float32)
                        .cpu(),
                        "true_max_avg": true_max_avg[selected]
                        .to(torch.float32)
                        .cpu(),
                        "true_max_groupmax": true_max_groupmax[selected]
                        .to(torch.float32)
                        .cpu(),
                        "true_max_logsumexp": true_max_logsumexp[selected]
                        .to(torch.float32)
                        .cpu(),
                        "upper_bound_avg": upper_bound_avg[selected]
                        .to(torch.float32)
                        .cpu(),
                        "num_chunks_total": int(true_max_groupmax.numel()),
                        "num_chunks_saved": int(selected.numel()),
                    }
                heads_payload[str(int(head_idx))] = head_entry

        shard = {
            "meta": {
                "sample_idx": sample_idx,
                "sample_id": sample_id,
                "sample_split": sample_split,
                "context_tokens": int(self.input_length),
                "retrieval_tokens": int(token_count),
                "step": int(step),
                "layer": int(layer_idx),
                "kv_heads": heads,
                "group_size": int(q_group.shape[1]),
                "head_dim": int(self.head_dim),
                "chunk_sizes": list(self._quest_bound_chunk_sizes),
                "query_type": "post_rope_gqa_group",
                "score_type": "raw_qk",
                "format": "compact_v1",
            },
            "heads": heads_payload,
        }
        filename = (
            f"{safe_sample_id}_idx{sample_idx}_step{step}_L{layer_idx}_compact.pt"
        )
        torch.save(shard, os.path.join(self._quest_bound_pt_dir, filename))
        print(
            f"[Quest Compact] saved {filename} heads={len(heads)} "
            f"tokens={token_count}",
            flush=True,
        )

    def _allayer_recall_ablation(
        self,
        indices: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        tag: str = "Final",
    ):
        """Accumulate exact q·k recall across all layers/heads for one decode step."""
        if layer_idx == 0 and tag == "Final":
            self._allayer_recall_step += 1
            self._allayer_recall_accum = {}
        if self._allayer_recall_step % self._allayer_recall_log_every != 0:
            return

        K = int(self._allayer_recall_k)
        with torch.no_grad():
            if not (self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None):
                return
            cpu_len = int(self.cpu_valid_end[layer_idx])
            if cpu_len < K:
                return
            keys_layer = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
                self.device, non_blocking=True,
            ).to(torch.float32)
            q_fp32 = raw_q.to(torch.float32)

            if tag not in self._allayer_recall_accum:
                self._allayer_recall_accum[tag] = [0, 0, 0]  # hit, total, kv_len

            kv_heads = indices.shape[1]
            for h in range(kv_heads):
                exact_scores = torch.matmul(keys_layer[0, h, :cpu_len, :], q_fp32[0, h])
                _, true_topk = torch.topk(exact_scores, k=K, dim=-1)
                true_set = set(true_topk.cpu().tolist())
                cand_set = set(indices[0, h].cpu().tolist())
                self._allayer_recall_accum[tag][0] += len(cand_set & true_set)
                self._allayer_recall_accum[tag][1] += K
                self._allayer_recall_accum[tag][2] = cpu_len

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


    def sync(self, layer_idx, batch_idx):
        """
        GPU版本:数据已经在prefill阶段直接存储到GPU,无需额外同步
        """
        pass  # GPU版本中所有数据已在prefill时存储到GPU,无需额外操作









