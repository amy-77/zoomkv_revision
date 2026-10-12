"""Polar ANN KV cache — single-layer (flat) Quest stage-1 + KIVI 4-bit stage-2.

ParisKV2 — single-level variant.

This is a **complete, self-contained** single-layer counterpart of
``cache_hub.polar_cache_kivi_quest_density_seperate_kernel``.  It is NOT a
subclass: the entire KIVI pipeline is inlined here with the two-level
(hierarchical) Quest path removed, so stage-1 is always a single flat
chunk selection.  Keeping it standalone lets the single-layer kernels /
geometry be tuned independently from the hierarchical module.

Pipeline:

* Prefill / dynamic update: raw post-RoPE K → per-(quest-)chunk min/max
  (``_compute_and_store_chunk_minmax``) → KIVI 4-bit packed-along-head_dim
  int32 stored in ``key_kivi_packed_gpu`` in **chunk-major** layout (A-1):
  ``[bs, kv, n_chunks_max, n_pack, group_size]``.  Within each chunk, the
  ``group_size`` lanes for a fixed dim-block sit at consecutive int32
  positions, so the stage-2 kernel's per-warp packed-K read collapses to a
  single 64-byte coalesced load.
* Decode stage-1: flat Quest chunk picking — score all chunks via
  ``quest_chunk_score`` on raw-K chunk min/max, pick top-``nk`` with
  ``_topk_3d``.  No quantization in stage-1.
* Decode stage-2: CDS dense/sparse classify →
  ``partial_chunk_kivi_qk_dense_sparse_cuda`` (KIVI on-the-fly dequant +
  raw_q dot + per-chunk top-K) → global top-K.

Differences vs. the hierarchical KIVI module:

* No ``POLAR_QUEST_HIERARCHICAL`` / ``POLAR_QUEST_LARGE_CHUNK`` /
  ``POLAR_QUEST_LARGE_RATIO`` / ``POLAR_QUEST_SMALL_RATIO`` — there is only
  one chunk level (``POLAR_QUEST_CHUNK``, default 8).
* The two-level large/sub-chunk min/max buffers, ``quest_sub_chunk_score`` /
  ``quest_map_back`` kernels, and the hierarchical recall logging are all
  dropped.
* Stage-2 KIVI kernel (``group_size`` ∈ {8, 16}) and the CDS density-classify
  kernel (``rerank_topk.cu``, per chunk centroid) are both chunk-size
  parametric, so chunk=8 needs no dedicated kernel; dense/sparse top-k is
  clamped to the chunk size.

Activate by setting
``POLAR_CACHE_MODULE=cache_hub.polar_cache_quest_density_single_layer_kivi``
before launching the runner.  Tunables: ``POLAR_QUEST_CHUNK`` (default 8),
``POLAR_QUEST_CAND_RATIO`` / ``POLAR_FORCE_CAND_TOKENS`` or dynamic candidate
ratios via ``POLAR_QUEST_DYNAMIC_CAND_RATIO=1``,
``POLAR_QUEST_DYNAMIC_CAND_THRESHOLDS`` and ``POLAR_QUEST_DYNAMIC_CAND_RATIOS``,
``POLAR_CDS_DENSE_TOPK`` / ``POLAR_CDS_SPARSE_TOPK``, ``POLAR_KIVI_KERNEL``
(cuda/python, default cuda), ``POLAR_KIVI_BITS`` (default 4).
"""
import math
import json
import gc
import os
from typing import Any, Optional, Tuple

import torch
import torch.cuda.nvtx as nvtx

from cache_hub.rerank.rerank import (
    partial_chunk_density_scores_cuda,
    dense_mask_from_topk_cuda,
)
from cache_hub.topk.float_topk_ops import float_topk_3d
from cache_hub.topk.float_topk_gather_concat_ops import float_topk_gather_concat_3d

from cache_hub.kivi.quant_pack import _pack_along_last_dim
from cache_hub.kivi.qk_dot import partial_chunk_kivi_qk_ref

try:
    from cache_hub.kivi.kivi_qk_dot_ops import (
        partial_chunk_kivi_qk_dense_sparse_cuda,
    )
    _HAS_KIVI_CUDA = True
    _KIVI_CUDA_IMPORT_ERR = ""
except Exception as _e:  # noqa: BLE001 — import-time fallback to Python ref
    partial_chunk_kivi_qk_dense_sparse_cuda = None
    _HAS_KIVI_CUDA = False
    _KIVI_CUDA_IMPORT_ERR = repr(_e)


# Stage-2 kernel selection: "cuda" (default) | "python" (ref, debug only).
_KIVI_KERNEL_DEFAULT = os.environ.get("POLAR_KIVI_KERNEL", "cuda").lower()
if _KIVI_KERNEL_DEFAULT not in ("cuda", "python"):
    raise ValueError(
        f"POLAR_KIVI_KERNEL must be 'cuda' or 'python'; got {_KIVI_KERNEL_DEFAULT!r}",
    )

_KIVI_BITS_DEFAULT = int(os.environ.get("POLAR_KIVI_BITS", "4"))
if _KIVI_BITS_DEFAULT not in (2, 4, 8):
    raise ValueError(
        f"POLAR_KIVI_BITS must be one of 2/4/8; got {_KIVI_BITS_DEFAULT}",
    )
_KIVI_NEG_INF = -3.0e38


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
        enable_offload: bool = True,  # 消融实验：关闭 offload，key 都在 GPU
        debug_offload_coords: bool = False,  # 调试：打印 offload 坐标/写入区间（会产生大量日志）
        enable_rerank: bool = False,  # 是否启用 rerank（=exact 或者rabitq）（提升 recall，略微增加延迟
        enable_recall: bool = False,  # 是否启用 Recall@K 计算（用于调试，会增加延迟）
        # ===== 消融实验参数 =====
        enable_alpha_correction: bool = False,  # 是否在 block_weight 中除以 alpha（消融实验）
        enable_rerank_error_ablation: bool = False,  # 消融实验：开启对比精确内积（默认关闭）
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
        _ = codebook_path  # explicitly ignored (KIVI has no codebook)
        _ = collision_ratio  # explicitly ignored; kept for kwarg compat with qwen.py
        self.candidate_ratio = candidate_ratio
        self.final_topk = final_topk  # 精排阶段的 top-K
        self.enable_offload = enable_offload  # 兼容性参数
        self.debug_offload_coords = debug_offload_coords
        self._debug_flush = os.environ.get('ZOOMKV_DEBUG_FLUSH', '0') == '1'
        self._debug_flush_topk = {}
        self._debug_flush_writes = {}
        self.enable_rerank = enable_rerank  # 是否启用精确内积 rerank
        self.enable_recall = enable_recall  # 是否启用 Recall@K 计算
        # 消融实验参数
        self.enable_alpha_correction = enable_alpha_correction
        self.enable_rerank_error_ablation = enable_rerank_error_ablation
        self.enable_coarse_recall_ablation = enable_coarse_recall_ablation  # Quest stage-1 candidate Recall 消融
        self.enable_rerank_recall_ablation = enable_rerank_recall_ablation  # 精排后 Recall@K 消融
        # ----- Quest-inspired Stage-1 (this file's main retrieval path) -----
        # POLAR_QUEST_STAGE1=0 disables Stage-1, in which case decode falls back
        # to a full-retrieval-zone 4-bit RaBitQ scoring (no chunk pre-selection).
        self._use_quest_chunk_coarse = (
            os.environ.get("POLAR_QUEST_STAGE1", "1") not in ("0", "false", "False", "")
        )
        self.quest_chunk_size = max(1, int(os.environ.get("POLAR_QUEST_CHUNK", "8")))
        self.quest_coarse_candidate_ratio = float(
            os.environ.get("POLAR_QUEST_CAND_RATIO", "0.05")
        )
        self._dynamic_cand_enabled = (
            os.environ.get("POLAR_QUEST_DYNAMIC_CAND_RATIO", "0")
            not in ("0", "false", "False", "")
        )
        self._dynamic_cand_thresholds = []
        self._dynamic_cand_ratios = []
        if self._dynamic_cand_enabled:
            try:
                self._dynamic_cand_thresholds = [
                    int(x.strip())
                    for x in os.environ.get(
                        "POLAR_QUEST_DYNAMIC_CAND_THRESHOLDS",
                        "5000,10000",
                    ).split(",")
                    if x.strip()
                ]
                self._dynamic_cand_ratios = [
                    float(x.strip())
                    for x in os.environ.get(
                        "POLAR_QUEST_DYNAMIC_CAND_RATIOS",
                        "0.7,0.5,0.4",
                    ).split(",")
                    if x.strip()
                ]
                if len(self._dynamic_cand_ratios) != len(self._dynamic_cand_thresholds) + 1:
                    raise ValueError(
                        "need len(ratios) == len(thresholds) + 1 "
                        f"(got ratios={self._dynamic_cand_ratios}, "
                        f"thresholds={self._dynamic_cand_thresholds})"
                    )
            except Exception as exc:  # noqa: BLE001 - fall back to fixed ratio with a clear log.
                print(
                    "[polar_cache_sl_kivi] WARN: invalid dynamic candidate ratio config "
                    f"({exc}); fallback to fixed cand_ratio={self.quest_coarse_candidate_ratio}",
                    flush=True,
                )
                self._dynamic_cand_enabled = False
                self._dynamic_cand_thresholds = []
                self._dynamic_cand_ratios = []
        # Quest stage-1 score variants:
        #   "minmax" (default): bbox upper bound on the dequantized 4-bit key
        #                       (decode 4-bit + reconstruct + amin/amax per chunk)
        #   "minmax_raw_cached": same bbox bound but on raw K cached at prefill /
        #                       decode-update; cheaper and mathematically equivalent
        #                       to Quest's paper formulation
        #   "mean":             debug-only fallback (chunk-mean dot product)
        _quest_mode = os.environ.get("POLAR_QUEST_MODE", "minmax_raw_cached").strip()
        if _quest_mode not in ("minmax", "minmax_raw_cached", "mean"):
            print(
                f"[polar_cache_sl_kivi] WARN: unsupported POLAR_QUEST_MODE='{_quest_mode}', "
                "fallback to 'minmax_raw_cached'",
                flush=True,
            )
            _quest_mode = "minmax_raw_cached"
        self.quest_mode = _quest_mode
        # 当 retrieval 长度不是 chunk_size 的整数倍时，跨界 chunk 和 trailing partial 的
        # token 没有 chunk min/max 元数据，因此**不进 stage-1 评分**；但它们仍有合法的
        # RaBitQ 4-bit 编码，所以会以"额外候选"身份 cat 进 coarse_indices，**继续走 stage-2
        # rerank**。每次 prefill / decode update 最多产生 chunk_size 个跨界 + chunk_size-1
        # 个 trailing partial token。1024 decode @ dui=512 最多 2 次 flush ⇒ 上界 ~22；
        # slack 128 够 ~16k decode token。
        # live skipped 上界 = chunk_size - 1（单层）。默认 512 远超所需；遇到更激进配置时通过 env 调大。
        self.MAX_SKIPPED_INDICES = int(os.environ.get("POLAR_MAX_SKIPPED_INDICES", "512"))
        self._chunk_quality_log = os.environ.get("POLAR_CHUNK_QUALITY_LOG", "0") not in ("0", "false", "False", "")
        self._chunk_quality_log_every = int(os.environ.get("POLAR_CHUNK_QUALITY_LOG_EVERY", "1"))
        self._dump_keys_for_analysis = os.environ.get("POLAR_DUMP_KEYS", "0") not in ("0", "false", "False", "")
        self._dump_keys_dir = os.environ.get("POLAR_DUMP_KEYS_DIR", "./chunk_analysis_data")
        # Cache POLAR_FORCE_CAND_TOKENS once: _quest_aligned_candidate_tokens is on
        # the per-token / per-layer hot path and used to read env every call.
        _force_cand_raw = os.environ.get("POLAR_FORCE_CAND_TOKENS")
        self._force_cand_tokens = int(_force_cand_raw) if _force_cand_raw else None
        if self._use_quest_chunk_coarse:
            _cand_info = (
                f"force_cand_tokens={self._force_cand_tokens}"
                if self._force_cand_tokens is not None
                else (
                    "dynamic_cand="
                    f"{self._dynamic_cand_ratios}@{self._dynamic_cand_thresholds}"
                    if self._dynamic_cand_enabled
                    else f"cand_ratio={self.quest_coarse_candidate_ratio}"
                )
            )
            print(
                "[polar_cache_sl_kivi] Quest Stage-1 ON | "
                f"chunk={self.quest_chunk_size}, {_cand_info}, "
                f"mode={self.quest_mode}, max_skipped={self.MAX_SKIPPED_INDICES} "
                "(env: POLAR_QUEST_STAGE1/POLAR_QUEST_CHUNK/POLAR_QUEST_CAND_RATIO/"
                "POLAR_FORCE_CAND_TOKENS/POLAR_QUEST_DYNAMIC_CAND_*/POLAR_QUEST_MODE)",
                flush=True,
            )
        # ----- Single-layer (flat) Quest: no hierarchical two-level selection -----
        # This module is the single-level variant; hierarchical state is pinned
        # off so every stage-1 call uses the flat chunk-scoring path.
        self._use_hierarchical_quest = False
        # ----- CDS (Coarse → Dense/Sparse) pipeline (always on for this module) -----
        self.cds_dense_ratio = float(os.environ.get("POLAR_CDS_DENSE_RATIO", "0.1"))
        self.cds_dense_topk = int(os.environ.get("POLAR_CDS_DENSE_TOPK", "16"))
        self.cds_sparse_topk = int(os.environ.get("POLAR_CDS_SPARSE_TOPK", "8"))
        if self._use_quest_chunk_coarse:
            print(
                f"[polar_cache_sl_kivi] CDS pipeline ON | "
                f"dense_ratio={self.cds_dense_ratio}, dense_topk={self.cds_dense_topk}, "
                f"sparse_topk={self.cds_sparse_topk} "
                f"(env: POLAR_CDS_DENSE_RATIO/POLAR_CDS_DENSE_TOPK/POLAR_CDS_SPARSE_TOPK)",
                flush=True,
            )
        self._cds_recall_check = os.environ.get("POLAR_CDS_RECALL_CHECK", "0") not in ("0", "false", "False", "")
        self._cds_recall_log_every = int(os.environ.get("POLAR_CDS_RECALL_LOG_EVERY", "50"))
        self._cds_recall_layers = set()
        _cds_layers_raw = os.environ.get("POLAR_CDS_RECALL_LAYERS", "0")
        if _cds_layers_raw:
            self._cds_recall_layers = {int(x.strip()) for x in _cds_layers_raw.split(",") if x.strip()}
        _cds_heads_raw = os.environ.get("POLAR_CDS_RECALL_HEADS", "")
        if _cds_heads_raw:
            self._cds_recall_heads = [int(x.strip()) for x in _cds_heads_raw.split(",") if x.strip()]
        else:
            self._cds_recall_heads = None  # None = all heads up to kv_heads
        self._cds_recall_step = 0
        self._topk_accuracy_check = os.environ.get("POLAR_TOPK_ACCURACY_CHECK", "0") not in ("0", "false", "False", "")
        self._topk_accuracy_step = 0
        # Top-k backend: torch.topk vs float_topk_3d (radix CUDA).
        # NOTE: default is 'float' (cache_hub.topk.float_topk_3d) — empirically
        # ~2-3x faster than torch.topk for the shapes used in this pipeline
        # (e.g. stage1 large_topk drops 118us → 46us, cds_global_topk 91us → 86us;
        # stage1 tail variance drops by an order of magnitude).
        # Set POLAR_TOPK_BACKEND=torch only for ablation / debugging.
        #   POLAR_TOPK_BACKEND=float   (default) — cache_hub.topk.float_topk_3d
        #   POLAR_TOPK_BACKEND=torch   — PyTorch topk on last dim (slower)
        _topk_backend = os.environ.get("POLAR_TOPK_BACKEND", "float").strip().lower()
        if _topk_backend in ("float", "cuda", "radix", "float_topk"):
            self._use_torch_topk = False
            self._topk_backend_name = "float_topk_3d"
        elif _topk_backend in ("torch", "pytorch", "1", "true"):
            self._use_torch_topk = True
            self._topk_backend_name = "torch.topk"
        else:
            raise ValueError(
                f"POLAR_TOPK_BACKEND must be 'torch' or 'float', got {_topk_backend!r}"
            )
        if self._use_quest_chunk_coarse:
            print(
                f"[polar_cache_sl_kivi] TopK backend: {self._topk_backend_name} "
                f"(env POLAR_TOPK_BACKEND)",
                flush=True,
            )
        # Stage-2 global topk backend.  Keep this separate from POLAR_TOPK_BACKEND
        # so we can debug large final candidate sets without changing Quest topk.
        _global_topk_backend = os.environ.get("POLAR_CDS_GLOBAL_TOPK_BACKEND", "auto").strip().lower()
        if _global_topk_backend in ("auto", ""):
            self._use_torch_cds_global_topk = False
            self._cds_global_topk_backend_name = self._topk_backend_name
        elif _global_topk_backend in ("torch", "pytorch", "1", "true"):
            self._use_torch_cds_global_topk = True
            self._cds_global_topk_backend_name = "torch.topk"
        elif _global_topk_backend in ("float", "cuda", "radix", "float_topk"):
            self._use_torch_cds_global_topk = False
            self._cds_global_topk_backend_name = "float_topk_3d"
        else:
            raise ValueError(
                "POLAR_CDS_GLOBAL_TOPK_BACKEND must be 'auto', 'torch', or 'float', "
                f"got {_global_topk_backend!r}"
            )
        if self._use_quest_chunk_coarse:
            print(
                f"[polar_cache_sl_kivi] CDS global TopK backend: {self._cds_global_topk_backend_name} "
                "(env POLAR_CDS_GLOBAL_TOPK_BACKEND)",
                flush=True,
            )
        # Fused stage-2 global topk + indirect gather + skipped concat.
        # Replaces the 3-launch chain (float_topk → gather → cat) with a
        # single CUDA kernel writing into a pre-allocated buffer (CUDA-Graph
        # friendly, ~1.6-2x speedup, ~10us saved per stage-2 call × 36
        # layers ≈ 360us per token).  Only safe with the radix backend; the
        # torch.topk and RaBitQ-ablation paths fall back to the 3-step chain.
        # Set POLAR_FUSED_CDS_GLOBAL_TOPK=0 for debugging / ablation.
        _fused_env = os.environ.get("POLAR_FUSED_CDS_GLOBAL_TOPK", "1").strip().lower()
        _fused_enabled = _fused_env not in ("0", "false", "off", "no")
        self._use_fused_global_topk = (
            _fused_enabled
            and not self._use_torch_topk
            and not self._use_torch_cds_global_topk
        )
        # ----- Full pipeline recall logging (aggregated across all layers/heads) -----
        # Stage-1 recall (coarse): tokens expanded from chunk_idx (flat chunk pick)
        # Stage-2 recall (final):  final topk_indices (after CDS dense/sparse + global topk + skipped)
        # env:
        #   POLAR_PIPELINE_RECALL=1                          打开聚合 recall 日志
        #   POLAR_PIPELINE_RECALL_LOG_EVERY=50               每 N 个 decode step 汇总一次
        #   POLAR_PIPELINE_RECALL_K=final|auto|100,500|...   默认对齐 final_topk；可显式给 K 列表
        #   POLAR_PIPELINE_RECALL_USE_SELECTED_K=1           K 自动 min(K, len(final_indices))
        self._pipeline_recall_check = os.environ.get("POLAR_PIPELINE_RECALL", "0") not in ("0", "false", "False", "")
        self._pipeline_recall_log_every = int(os.environ.get("POLAR_PIPELINE_RECALL_LOG_EVERY", "50"))
        self._pipeline_recall_step = 0
        self._pipeline_recall_use_selected_k = os.environ.get(
            "POLAR_PIPELINE_RECALL_USE_SELECTED_K", "1"
        ) not in ("0", "false", "False", "")
        _pl_k_raw = os.environ.get("POLAR_PIPELINE_RECALL_K", "").strip()
        if not _pl_k_raw:
            # Default to a single fixed recall@K aligned with final retrieval K.
            self._pipeline_recall_ks = [int(final_topk)]
        elif _pl_k_raw.lower() in ("final", "auto"):
            self._pipeline_recall_ks = [int(final_topk)]
        else:
            self._pipeline_recall_ks = [int(x.strip()) for x in _pl_k_raw.split(",") if x.strip()]
            if not self._pipeline_recall_ks:
                self._pipeline_recall_ks = [int(final_topk)]
        # Keep Ks deterministic and strictly positive.
        self._pipeline_recall_ks = sorted({int(k) for k in self._pipeline_recall_ks if int(k) > 0})
        if not self._pipeline_recall_ks:
            self._pipeline_recall_ks = [int(final_topk)]
        self._pr_logging_this_step = False
        self._pr_step_accum = {}
        self._pr_step_kv_len = 0
        # Optional per-step diag meta (None when not wired by callers; print falls back to bare format).
        self._step_diag_meta = None
        # Optional PyTorch reference for CDS classify (centroid @ q).
        self._cds_density_torch = os.environ.get(
            "POLAR_CDS_DENSITY_TORCH", "0"
        ) not in ("0", "false", "False", "")
        if self._cds_density_torch:
            print(
                "[polar_cache_sl_kivi] CDS classify: PyTorch density scores "
                "(POLAR_CDS_DENSITY_TORCH)",
                flush=True,
            )
        if self._pipeline_recall_check:
            print(
                f"[polar_cache_sl_kivi] Pipeline Recall ON (aggregated all layers/heads) | "
                f"log_every={self._pipeline_recall_log_every}, Ks={self._pipeline_recall_ks}, "
                f"use_selected_k={self._pipeline_recall_use_selected_k}",
                flush=True,
            )
        # 自适应 Top-k 参数
        self.adaptive_topk_enabled = adaptive_topk_enabled
        self.adaptive_topk_threshold = adaptive_topk_threshold
        self.adaptive_topk_min_k = adaptive_topk_min_k
        self.adaptive_topk_max_k = adaptive_topk_max_k if adaptive_topk_max_k is not None else final_topk
        
        # KIVI 4-bit packed-along-head_dim stage-2 codec (no codebook needed).
        self._kivi_bits = _KIVI_BITS_DEFAULT
        self._kivi_feat_per_int = 32 // self._kivi_bits
        if head_dim % self._kivi_feat_per_int != 0:
            raise ValueError(
                f"head_dim ({head_dim}) must be divisible by "
                f"feat_per_int={self._kivi_feat_per_int} (bits={self._kivi_bits})"
            )
        self._kivi_n_pack = head_dim // self._kivi_feat_per_int
        self._kivi_stage2_kernel = _KIVI_KERNEL_DEFAULT
        if self._kivi_stage2_kernel == "cuda":
            if not _HAS_KIVI_CUDA:
                print(
                    f"[polar_cache_sl_kivi] WARN: KIVI CUDA kernel import failed "
                    f"({_KIVI_CUDA_IMPORT_ERR}); falling back to Python ref.",
                    flush=True,
                )
                self._kivi_stage2_kernel = "python"
            elif head_dim != 128:
                print(
                    f"[polar_cache_sl_kivi] WARN: CUDA kernel requires head_dim=128 "
                    f"(got {head_dim}); falling back to Python ref.",
                    flush=True,
                )
                self._kivi_stage2_kernel = "python"
            elif self._kivi_bits != 4:
                print(
                    f"[polar_cache_sl_kivi] WARN: CUDA kernel requires bits=4 "
                    f"(got {self._kivi_bits}); falling back to Python ref.",
                    flush=True,
                )
                self._kivi_stage2_kernel = "python"
        # 初始化实际使用的 retrieval zone 长度
        self._actual_retrieval_length = None  # 将在 prefill 时设置
        # Quest stage-1 缓存：仅在 prefill 和 _update_polar_index 时刷新
        self._cached_quest_nk = 0
        self._cached_quest_n_chunks = 0
        self._cached_quest_kv_len = -1
        self._cached_scores_view = None
        self._cached_chunk_min_views = None
        self._cached_chunk_max_views = None
        self._cached_chunk_valid_views = None
        # 添加decode步数计数器,用于控制Recall计算频率
        self._decode_step_counter = 0
        

        import time as _time
        _t0 = _time.time()
        print(
            f"[Init] polar_cache_kivi.__init__ start "
            f"(input_length={self.input_length}, max_new={max_new_length}, "
            f"layers={layer_num}, offload={self.enable_offload}, "
            f"kivi_bits={self._kivi_bits}, kivi_kernel={self._kivi_stage2_kernel})",
            flush=True,
        )
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
        # Per-step memoisation of get_region_indices() output.  Layer 0
        # builds the dict after appending the current token; later layers
        # reach the same length and reuse it.  A flush explicitly invalidates
        # this cache before the next region lookup.
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
        print(f"[Init] KIVI codec ready ({_time.time()-_t0:.1f}s)", flush=True)

        
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
            self.cpu_kv_capacity = self.unified_kv_capacity - self.sink_size - self.local_size
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
            max_topk_size = self.final_topk + self.MAX_SKIPPED_INDICES
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
        # Pre-compute chunk capacity here so the chunk-major KIVI packed-K
        # allocation below can size axis-2 by chunks (rather than tokens).
        # ``_chunk_n_max`` is also (re)assigned in the Quest-coarse block
        # below; we set it eagerly here because the KIVI packed buffer
        # always needs it under the A-1 chunk-major layout.
        _kivi_chunk = self.quest_chunk_size
        self._chunk_n_max = (
            self.retrieval_zone_capacity + _kivi_chunk - 1
        ) // _kivi_chunk
        print(
            f"[Init] allocating KIVI packed K ({layer_num} layers × bs={batch_size} × "
            f"kv={self.kv_head} × n_chunks={self._chunk_n_max} × "
            f"n_pack={self._kivi_n_pack} × g={_kivi_chunk}, bits={self._kivi_bits}) "
            f"[A-1 chunk-major layout] ...",
            flush=True,
        )
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
        # 仅在 Quest stage-1 + raw-K-cached 模式下分配，其他模式按需关掉以省显存
        # shape: [batch, kv_head, n_chunks_max, head_dim]
        # 数学：score(chunk) = Σ_d max(q_d * max_k_d, q_d * min_k_d)，对应 Quest paper Sec.3.1
        # 存储 ≈ KV cache 的 1/chunk_size（chunk=8 时约 12.5%）
        if self._use_quest_chunk_coarse:
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
            # Pre-allocate Quest stage-1 chunk-score buffer to avoid per-decode torch.empty

            self._quest_scores_buffer = torch.empty(
                batch_size * self.kv_head * self._chunk_n_max,
                dtype=torch.float32,
                device=self.device,
            )
            # Pre-create chunk expansion offsets [0, 1, ..., chunk-1] used to fan-out
            # chunk indices to token indices in the hot path.
            self._chunk_offs = torch.arange(chunk, device=self.device, dtype=torch.int64).view(1, 1, 1, chunk)
            print(
                f"[Init] chunk_minmax allocated: chunk={chunk}, "
                f"n_chunks_max={self._chunk_n_max}, "
                f"per-layer={(batch_size * self.kv_head * self._chunk_n_max * head_dim * 2 * 2) / 1024**2:.1f}MB "
                f"(min+max), total≈{(batch_size * self.kv_head * self._chunk_n_max * head_dim * 2 * 2 * layer_num) / 1024**3:.2f}GB",
                flush=True,
            )
        else:
            self._chunk_key_min_gpu = []
            self._chunk_key_max_gpu = []
            self._chunk_key_centroid_gpu = []
            self._chunk_valid_mask_gpu = []
            self._chunk_valid_end = [0] * layer_num
            self._chunk_n_max = 0
            self._chunk_carry_count = [0] * layer_num
            self._chunk_carry_keys = []
            self._chunk_merge_buffer = []
            self._chunk_skipped_indices_gpu = []
            self._chunk_skipped_count = [0] * layer_num
            self._quest_scores_buffer = None
            self._chunk_offs = None

        self.context = 0
        self._preallocated_buffers = {}
        print(f"[Init] KIVI buffer done ({_time.time()-_t0:.1f}s)", flush=True)


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

        # 预加载 UVA H2D Gather kernel（利用 UVA 直接从 pinned CPU gather 到 GPU）
        try:
            from .gather_trans.trans_h2d import h2d_gather_kv
            self._h2d_gather_kv = h2d_gather_kv
            print(f"[Init]   h2d_gather loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"H2D Gather CUDA kernel compile failed: {e}")
        
        # 预加载 CDS density + dense-mask CUDA extension (rerank_topk shares
        # the same .so as the RaBitQ rerank kernel — we only use the
        # density/dense-mask entry points here).
        try:
            from .rerank.rerank import load_kernel_module
            self._rerank_topk_kernel = load_kernel_module("rerank_topk.cu", "rerank_topk")
            print(f"[Init]   rerank_topk (CDS density) loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            raise RuntimeError(f"rerank_topk CUDA kernel compile failed: {e}")

        # 预加载 KIVI stage-2 CUDA kernel (only when cuda mode is active).
        if self._kivi_stage2_kernel == "cuda":
            try:
                from .kivi.kivi_qk_dot_ops import _kivi_kernel as _load_kivi_kernel
                self._kivi_kernel = _load_kivi_kernel()
                print(
                    f"[Init]   kivi_qk_dot loaded ({_time.time()-_t0:.1f}s)",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[Init]   kivi_qk_dot CUDA failed ({e}); using Python ref",
                    flush=True,
                )
                self._kivi_kernel = None
                self._kivi_stage2_kernel = "python"
        else:
            self._kivi_kernel = None

        # 预加载 Quest chunk score CUDA kernel
        try:
            from .quest.quest_ops import load_kernel_module as load_quest_kernel
            self._quest_kernel = load_quest_kernel()
            print(f"[Init]   quest_chunk_score loaded ({_time.time()-_t0:.1f}s)", flush=True)
        except Exception as e:
            print(f"[Init]   quest_chunk_score CUDA failed ({e}), falling back to PyTorch", flush=True)
            self._quest_kernel = None

        # __init__ 完成，清理初始化过程中的临时对象
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[Init] polar_cache_kivi.__init__ DONE ({_time.time()-_t0:.1f}s)", flush=True)

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
            self._refresh_quest_chunk_cache(valid_length)

            # 重新分配精确大小的 buffer（避免每次 slice）
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

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

        # Step2: encode retrieval zone — KIVI per-(quest-)chunk per-channel
        # 4-bit packing is folded into _compute_and_store_chunk_minmax (it
        # writes both chunk_min/max AND key_kivi_packed_gpu in lockstep,
        # so chunks always have a consistent (mn, mx, packed_codes) triple).
        keys_to_encode = key_states[:, retrieval_start:retrieval_end, :, :].transpose(1, 2).contiguous()

        # retrieval zone 起点 = sink_size，对应 zone-internal offset = 0
        if self._use_quest_chunk_coarse:
            self._compute_and_store_chunk_minmax(
                layer_idx, keys_to_encode, abs_token_start=0
            )

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
            self._refresh_quest_chunk_cache(valid_length)

            # 重新分配精确大小的 buffer（避免每次 slice）
            if self.enable_offload:
                bs = self.unified_keys_cpu[0].shape[0]
                self._topk_keys_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_values_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device=self.device)
                self._topk_keys_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)
                self._topk_values_cpu_buffer = torch.empty((bs, self.kv_head, self.final_topk + self.MAX_SKIPPED_INDICES, self.head_dim), dtype=self.dtype, device='cpu', pin_memory=True)

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
        

        # Step2: KIVI 4-bit encode retrieval zone (per-(quest-)chunk, per-channel).
        # Folded into _compute_and_store_chunk_minmax — each chunk gets its
        # (mn, mx, packed_codes) written in lockstep, so stage-2 always reads
        # a consistent triple.
        # key_states: [1, seq_len, kv_head, head_dim] -> [1, kv_head, retrieval_length, head_dim]
        keys_to_encode = key_states[0, retrieval_start:retrieval_end, :, :].transpose(0, 1).contiguous().unsqueeze(0)
        if self._use_quest_chunk_coarse:
            self._compute_and_store_chunk_minmax(
                layer_idx,
                keys_to_encode,
                abs_token_start=0,
                batch_slice=slice(batch_idx, batch_idx + 1),
            )
        
        
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
          * Per-step memoisation: after each layer appends the current token,
            it reaches the same total length.  Layer 0 builds the dict and
            later layers reuse it via the ``_cached_region_total_len`` fast
            path.  Cache maintenance explicitly invalidates this on flush.
        """
        del device  # only the pre-allocated buffer's device matters

        total_len = self.current_seq_len[layer_idx]

        if total_len == self._cached_region_total_len and self._cached_region_info is not None:
            return self._cached_region_info

        cpu_len = (
            int(self.cpu_valid_end[layer_idx])
            if self.enable_offload and self.cpu_valid_end is not None
            else -1
        )

        steady_threshold = self.steady_size  # 预计算：sink_size + local_size

        if cpu_len >= 0:
            # Offload mode has an authoritative physical boundary:
            #   CPU retrieval = [sink_size, sink_size + cpu_valid_end)
            #   GPU near zone = [sink_size + cpu_valid_end, total_len)
            # The near zone contains local + the live update buffer.  In
            # particular, when the buffer contains exactly one full update
            # interval, it must remain visible until the last layer commits
            # the flush; modulo arithmetic would incorrectly turn it into 0.
            retrieval_length = cpu_len
            local_start = self.sink_size + retrieval_length
            local_len = max(0, total_len - local_start)
        else:
            # Non-offload mode has no CPU boundary, so retain the logical
            # window calculation used by the all-GPU layout.
            if self.input_length >= steady_threshold:
                update_buffer_size = max(0, total_len - self.input_length) % self.dynamic_update_interval
            elif total_len >= steady_threshold:
                update_buffer_size = (total_len - steady_threshold) % self.dynamic_update_interval
            else:
                update_buffer_size = 0
            local_end = max(total_len - update_buffer_size, self.sink_size)
            local_start = max(self.sink_size, local_end - self.local_size)
            retrieval_length = max(0, local_start - self.sink_size)
            local_len = total_len - local_start if local_start < total_len else 0

        # Zero-launch index views — slicing a contiguous arange returns a
        # view; downstream callers (``unsqueeze().expand()``) accept views.
        sink_indices = self._arange_buffer[:self.sink_size]
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


    def _compute_local_start(self, layer_idx=0):
        """
        轻量版：只计算 local_start，不构建 tensor（用于 _update_polar_index）
        """
        total_len = self.current_seq_len[layer_idx]
        steady_threshold = self.steady_size  # 预计算：sink_size + local_size
        
        if self.input_length >= steady_threshold:
            update_buffer_size = max(0, total_len - self.input_length) % self.dynamic_update_interval
        elif total_len >= steady_threshold:
            update_buffer_size = (total_len - steady_threshold) % self.dynamic_update_interval
        else:
            update_buffer_size = 0
        
        local_end = max(total_len - update_buffer_size, self.sink_size)
        return max(self.sink_size, local_end - self.local_size)


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


    def _debug_flush_near_boundary(self, layer_idx):
        if not self._debug_flush:
            return False
        # Per-layer length is authoritative before the last-layer commit.
        progress = self.current_seq_len[layer_idx] - max(
            self.input_length, self.sink_size + self.local_size)
        interval = self.dynamic_update_interval
        return progress >= interval - 1 and progress % interval in (interval - 1, 0, 1)

    def _debug_flush_log(self, phase, layer_idx, indices=None,
                         write_range=None, **extra):
        if not self._debug_flush:
            return
        step = int(self.current_seq_len[layer_idx] - self.input_length)
        if indices is not None:
            # Debug only: scalar reads synchronize the index-producing stream.
            count = int(indices.shape[-1])
            self._debug_flush_topk[layer_idx] = dict(
                topk_step=step,
                topk_indices_min=int(indices.min().item()) if indices.numel() else None,
                topk_indices_max=int(indices.max().item()) if indices.numel() else None,
                n_topk=count,
            )
        topk = self._debug_flush_topk.get(layer_idx, dict(
            topk_step=None, topk_indices_min=None, topk_indices_max=None, n_topk=None))
        if write_range is None:
            write_range = self._debug_flush_writes.get(layer_idx, (None, None, None))
        cpu_end = int(self.cpu_valid_end[layer_idx]) if self.cpu_valid_end is not None else None
        record = dict(
            phase=phase, layer=layer_idx, step=step, context=int(self.context),
            decode_step_counter=int(self._decode_step_counter),
            current_seq_len=[int(n) for n in self.current_seq_len],
            cpu_valid_end=cpu_end,
            cpu_valid_end_all=[int(n) for n in self.cpu_valid_end] if self.cpu_valid_end is not None else None,
            cpu_write_start=write_range[0], cpu_write_end=write_range[1],
            cpu_write_step=write_range[2],
            topk_source='current_selection' if indices is not None else 'last_selection',
            **topk,
        )
        record.update(extra)
        print('[ZOOMKV_FLUSH] ' + json.dumps(record, sort_keys=True, default=int), flush=True)

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

        if self._debug_flush:
            for debug_layer in range(self.layer_num):
                start = int(self.cpu_valid_end[debug_layer]) if self.cpu_valid_end is not None else None
                end = start + num_new_tokens if start is not None else None
                self._debug_flush_writes[debug_layer] = (start, end, int(self._decode_step_counter))
                self._debug_flush_log('before_flush', debug_layer,
                                      num_new_tokens=int(num_new_tokens))

        # 更新 _actual_retrieval_length（编码到的相对长度）
        # Retrieval Zone = [sink_size : encode_end]
        self._actual_retrieval_length = encode_end - self.sink_size
        self._refresh_quest_chunk_cache(self._actual_retrieval_length)
        
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
                
            # KIVI per-(quest-)chunk per-channel encode is folded into
            # _compute_and_store_chunk_minmax — it writes both the
            # chunk_min/max bookkeeping AND the 4-bit packed codes into
            # key_kivi_packed_gpu for every chunk it completes (including
            # the carry-merged chunk).  zone-internal offset = encode_start - sink_size.
            if self._use_quest_chunk_coarse:
                self._compute_and_store_chunk_minmax(
                    ldx,
                    new_tokens_to_encode,
                    abs_token_start=encode_start - self.sink_size,
                )

            
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
        # cpu_valid_end and the physical GPU near-zone boundary changed.
        # Force the next layer-0 lookup to rebuild region metadata; the
        # remaining layers in that decode step will share the rebuilt entry.
        self._cached_region_total_len = -1
        self._cached_region_info = None
        if self._debug_flush:
            for debug_layer in range(self.layer_num):
                self._debug_flush_log('after_flush', debug_layer,
                                      num_new_tokens=int(num_new_tokens))



    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        Append one layer's KV for the current decode token.

        A full update buffer is flushed at the start of the *next* decode
        step, before layer 0 writes.  At that point every layer has finished
        the previous token and therefore sees the same physical layout for
        the entire current step.  This matches the ParisKV commit order.
        """
        # nvtx.range_push(f"decode_update_layer_{layer_idx}")

        current_len = self.current_seq_len[layer_idx]

        # Flush only at a cross-layer synchronization point.  The physical
        # GPU near zone is [local | update buffer], so its length is derived
        # from the authoritative CPU boundary rather than a decode counter.
        if self.enable_offload and layer_idx == 0:
            near_len = (
                current_len
                - min(self.sink_size, current_len)
                - int(self.cpu_valid_end[0])
            )
            if near_len >= self.local_size + self.dynamic_update_interval:
                if not all(n == current_len for n in self.current_seq_len):
                    raise RuntimeError(
                        "cache flush requires all layer lengths to match: "
                        f"{self.current_seq_len}"
                    )
                self._update_polar_index()

        # ============ Step 1: 写入当前 token ============
        # nvtx.range_push("kv_write")
        key_to_add = key_states[:, 0, :, :]
        value_to_add = value_states[:, 0, :, :]

        if self.enable_offload:
            # 【Offload 模式】GPU 布局: [Sink][Local][UpdateBuffer]
            gpu_write_pos = current_len - int(self.cpu_valid_end[layer_idx])
            if not 0 <= gpu_write_pos < self.gpu_kv_capacity:
                raise RuntimeError(
                    "GPU KV write is outside the physical cache: "
                    f"layer={layer_idx}, current={current_len}, "
                    f"cpu={self.cpu_valid_end[layer_idx]}, pos={gpu_write_pos}, "
                    f"capacity={self.gpu_kv_capacity}"
                )
            self.unified_keys_gpu[layer_idx][:, :, gpu_write_pos, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, gpu_write_pos, :] = value_to_add
        else:
            # 【非 Offload 模式】按全局位置追加
            self.unified_keys_gpu[layer_idx][:, :, current_len, :] = key_to_add
            self.unified_values_gpu[layer_idx][:, :, current_len, :] = value_to_add

        # 更新全局序列长度
        self.current_seq_len[layer_idx] = current_len + 1
        # nvtx.range_pop()  # kv_write

        # ============ Step 2: 最后一层提交 decode step ============
        if layer_idx == self.layer_num - 1:
            if not all(n == self.current_seq_len[0] for n in self.current_seq_len):
                raise RuntimeError(
                    "decode step layer lengths diverged before cache commit: "
                    f"{self.current_seq_len}"
                )
            self._decode_step_counter += 1
            self.context += 1

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
        # nvtx.range_pop()  # decode_prepare_query
        
        # 判断是否需要检索
        # total_length 至少要 > sink + local + final_topk*2，否则 retrieval zone 太小
        min_total_for_retrieval = max(self.FULL_ATTENTION_THRESHOLD, self.sink_size + self.local_size + self.final_topk * 2)
        use_full_attention = (total_length < min_total_for_retrieval) or (valid_length <= 0)
        # key_data_gpu 仅用于 exact rerank / recall 计算（延迟获取）
        key_data_gpu = None
        nvtx.range_pop()


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
            # KIVI stage-2 reads `key_kivi_packed_gpu` (chunk-major after
            # A-1) directly by retrieval-internal chunk id.  We slice on
            # the chunk axis to expose only the chunks that have been
            # written so far — Quest stage-1 already filters by
            # ``_chunk_valid_mask_gpu``, but slicing here lets the kernel
            # bounds-check ``real_chunk_id < n_chunks_packed`` cheaply.
            chunk_size = self.quest_chunk_size
            n_valid_chunks = max(0, valid_length // chunk_size)
            # Reuse the previous slice view when ``n_valid_chunks`` is
            # unchanged for this layer; ~15/16 of decode steps hit this
            # branch (chunk size = 16, n_valid_chunks advances 1 per chunk
            # commit).  Slicing itself is stride-only but the __getitem__
            # dispatch alone costs ~3-5us per layer.
            if self._cached_kivi_packed_n_chunks[layer_idx] == n_valid_chunks:
                key_kivi_packed = self._cached_kivi_packed_views[layer_idx]
            else:
                key_kivi_packed = self.key_kivi_packed_gpu[layer_idx][:, :, :n_valid_chunks, :, :]
                self._cached_kivi_packed_n_chunks[layer_idx] = n_valid_chunks
                self._cached_kivi_packed_views[layer_idx] = key_kivi_packed
            # Recall/error ablations may want fp32 raw K of the retrieval zone;
            # the normal KIVI path itself doesn't use it.
            if (
                self.enable_rerank_error_ablation
                or self.enable_coarse_recall_ablation
                or self.enable_rerank_recall_ablation
            ):
                if not (self.enable_offload and self.unified_keys_cpu is not None):
                    key_data_gpu = self.unified_keys_gpu[layer_idx][:, :, retrieval_start:retrieval_end_effective, :]
                else:
                    key_data_gpu = None
            else:
                key_data_gpu = None

            # 【Offload 模式】Retrieval 起点固定为 sink_size
            assert not self.enable_offload or retrieval_start == self.sink_size, f"Offload 模式下 retrieval_start({retrieval_start}) 必须等于 sink_size({self.sink_size})"

            nvtx.range_push("collision_based_topk_batch")
            topk_indices_batch = self.collision_based_topk_batch(
                query=query_batch,
                key_data=key_data_gpu,
                key_kivi_packed=key_kivi_packed,
                layer_idx=layer_idx,
            )
            nvtx.range_pop()
            # if layer_idx == 0:
            #     print(f"[Debug] topk_indices_batch.shape={topk_indices_batch.shape}, final_topk={self.final_topk}", flush=True)

        
            # fixed_topk_sl_pre NVTX removed — only stride-only ``expand``
            # views (no kernel work) lived inside; the push/pop pair was
            # pure overhead in the offload hot path.
            sink_indices_exp = sink_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)
            local_indices_exp = local_indices.unsqueeze(0).unsqueeze(0).expand(bs, self.kv_head, -1)


            if self.enable_offload and self.unified_keys_cpu is not None:
                sink_local_end = self.sink_size + local_len
                sink_local_keys = self.unified_keys_gpu[layer_idx][:, :, :sink_local_end, :]
                sink_local_values = self.unified_values_gpu[layer_idx][:, :, :sink_local_end, :]

                if self._debug_flush_near_boundary(layer_idx):
                    self._debug_flush_log(
                        'before_gather', layer_idx, indices=topk_indices_batch,
                        retrieval_length=int(retrieval_length), valid_length=int(valid_length),
                        local_len=int(local_len),
                        local_start=int(total_length - local_len),
                        physical_local_start=int(self.sink_size + cpu_len),
                    )
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
                if self._debug_flush_near_boundary(layer_idx):
                    self._debug_flush_log('after_gather', layer_idx)
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
    # Quest Stage-1（量化 key 版）：与 Quest-main 的 ``compute_max_possible`` 同构
    # ----------------------------------------------------------------------------
    # 公式（每 chunk c）：
    #     score(c) = Σ_{b,d} max(q_{b,d}·u_max[b,d,c], q_{b,d}·u_min[b,d,c])
    # 其中：
    #   v[b,d,t] = sign[b,d,t] · mag_center[t_code[b,d,t]]      ← 4-bit 反量化
    #   u[b,d,t] = key_block_weight[b,t] · v[b,d,t]             ← 与 partial_rerank 同构
    #   u_min/u_max = chunk 内沿 token 维的 per-dim 极值（bbox）
    # 这是该 chunk 内任意 token 的 RaBitQ 近似内积 ``Σ q·u`` 的紧上界，与 Quest 论文一致。
    # 仅用 GPU 上的 4-bit `key_4bit` + `key_block_weight`，不读浮点 key、无 D2H/H2D。
    # ============================================================================
    def _quest_aligned_candidate_tokens(self, kv_len: int) -> int:
        """候选 token 数 = ceil_to_chunk(ceil(candidate_ratio·kv_len))，并裁到 (kv_len//chunk)·chunk。
        若设置环境变量 ``POLAR_FORCE_CAND_TOKENS``，则使用绝对 budget（仍按 chunk 向上对齐），
        优先级高于 dynamic/fixed ratio，便于和 polar_cache.py 跑同 budget 对照实验。
        """
        c = self.quest_chunk_size
        if kv_len <= 0:
            return 0
        if kv_len < c:
            return kv_len
        cap = (kv_len // c) * c
        if self._force_cand_tokens is not None:
            target = self._force_cand_tokens
        else:
            ratio = self._quest_candidate_ratio(kv_len)
            target = math.ceil(kv_len * ratio)
        target_aligned = max(c, ((target + c - 1) // c) * c)
        return min(cap, target_aligned)


    def _quest_candidate_ratio(self, kv_len: int) -> float:
        """Return dynamic Quest candidate ratio for the current retrieval length."""
        if not self._dynamic_cand_enabled:
            return self.quest_coarse_candidate_ratio
        for threshold, ratio in zip(self._dynamic_cand_thresholds, self._dynamic_cand_ratios):
            if kv_len < threshold:
                return ratio
        return self._dynamic_cand_ratios[-1]




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
            return
        n_chunks = kv_len // chunk
        target_tokens = self._quest_aligned_candidate_tokens(kv_len)
        nk = max(1, min(n_chunks, target_tokens // chunk))
        self._cached_quest_n_chunks = n_chunks
        self._cached_quest_nk = nk
        self._cached_quest_kv_len = kv_len
        bs = self.batch_size
        kv_heads = self.kv_head
        self._cached_scores_view = (
            self._quest_scores_buffer[:bs * kv_heads * n_chunks]
            .view(bs, kv_heads, n_chunks)
        )
        self._cached_chunk_min_views = [
            t[:bs] for t in self._chunk_key_min_gpu
        ]
        self._cached_chunk_max_views = [
            t[:bs] for t in self._chunk_key_max_gpu
        ]
        self._cached_chunk_valid_views = [
            t[:bs] for t in self._chunk_valid_mask_gpu
        ]
        # Single-layer: stage-1 picks `nk` flat chunks; CDS operates on those.
        nk_small_for_cds = nk

        # --- CDS cache ---
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
        # Skipped via env var when on torch.topk path or RaBitQ ablation —
        # see ``_use_fused_global_topk`` below.
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
        if not self._use_quest_chunk_coarse:
            return
        kv_len = max(0, int(kv_len))
        chunk = int(self.quest_chunk_size)
        n_chunks = kv_len // chunk if chunk > 0 else 0
        covered_tokens = n_chunks * chunk
        skipped_count = max(0, kv_len - covered_tokens)
        if skipped_count > self.MAX_SKIPPED_INDICES:
            raise RuntimeError(
                "[chunk_minmax] live skipped indices overflow: "
                f"kv_len={kv_len}, covered_tokens={covered_tokens}, "
                f"skipped={skipped_count} > MAX={self.MAX_SKIPPED_INDICES}. "
                "增大 POLAR_MAX_SKIPPED_INDICES。"
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
    def _record_skipped_range(
        self,
        layer_idx: int,
        zone_start: int,
        zone_end: int,
    ):
        """把 retrieval-zone-内偏移 [zone_start, zone_end) 一段 token 索引追加到
        `_chunk_skipped_indices_gpu[layer_idx]`，并更新 `_chunk_skipped_count[layer_idx]`。

        这些索引会在 dispatch 阶段 cat 到 stage-1 `coarse_indices` 末尾，参与 stage-2 rerank。

        注意：在 live-carry 语义下此方法已不再被 `_compute_and_store_chunk_minmax`
        调用，保留是为了向后兼容外部 caller / 调试场景。
        """
        if zone_end <= zone_start:
            return
        n_add = zone_end - zone_start
        cur = self._chunk_skipped_count[layer_idx]
        if cur + n_add > self.MAX_SKIPPED_INDICES:
            raise RuntimeError(
                f"[chunk_minmax] layer={layer_idx} skipped indices buffer overflow: "
                f"cur={cur} + add={n_add} > MAX={self.MAX_SKIPPED_INDICES}. "
                f"提高 self.MAX_SKIPPED_INDICES（每次 flush 最多 ~2*chunk_size，"
                f"长 decode 需要按上界估）。"
            )
        # 一次性写入 [bs, kv] 个相同的 row（chunk 边界与 batch/head 无关）
        idx_row = torch.arange(zone_start, zone_end, device=self.device, dtype=torch.int64)
        self._chunk_skipped_indices_gpu[layer_idx][:, :, cur:cur + n_add].copy_(
            idx_row.unsqueeze(0).unsqueeze(0)
        )
        self._chunk_skipped_count[layer_idx] = cur + n_add




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
                              slice → 写入 buffer[batch_slice, ...]（_prefill_single_sample 用）
        """
        if not self._use_quest_chunk_coarse:
            return
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

    @staticmethod
    def _torch_topk_3d(scores: torch.Tensor, k: int) -> torch.Tensor:
        """Always torch.topk on last dim — used by ablation / POLAR_CDS_DENSITY_TORCH paths."""
        k = int(k)
        if k <= 0:
            raise ValueError(f"topk k must be positive, got {k}")
        _, indices = torch.topk(scores.float(), k=k, dim=-1)
        return indices

    def _topk_3d(self, scores: torch.Tensor, k: int) -> torch.Tensor:
        """Top-k on last dim [bs, kv_heads, L] → indices [bs, kv_heads, k].

        Backend selected at init via POLAR_TOPK_BACKEND (torch | float).

        Caller contract:
        - ``k`` MUST be a Python int (all current call sites source from
          cached ints set at init).
        - ``scores`` MUST already be fp32; all call sites pass pre-allocated
          fp32 buffers (``_quest_scores_buffer``,
          ``_cached_cds_density_scores``, ``_cached_cds_out_scores``).

        Defensive ``int(k)`` / ``k > 0`` / ``scores.float()`` were removed:
        the int cast is no-op (~30 ns) but ``scores.float()`` carried ~1-2 μs
        of PyTorch dispatch overhead even on already-fp32 inputs (≈200-400 μs
        per decode token over 5 × 36 calls).  ``float_topk.cu`` itself
        auto-casts non-fp32 inputs and asserts ``k > 0``, so real misuse is
        still caught loudly.
        """
        if self._use_torch_topk:
            _, indices = torch.topk(scores, k=k, dim=-1)
            return indices
        return float_topk_3d(scores, k)




    def _quest_pick_coarse_indices(
        self,
        kv_len: int,
        *,
        raw_q: torch.Tensor,
        layer_idx: int,
    ):
        """Quest stage-1: score chunks via min/max upper bound and pick top-K.
        Args:
            kv_len:    retrieval zone 长度（仅供 callers 做 shape 一致性校验，
                       kernel 本身用 ``self._cached_quest_n_chunks``）
            raw_q:     [bs, kv, head_dim] raw post-GQA-mean query
            layer_idx: layer index (used to look up chunk_min/max cache)
        Returns:
            (chunk_idx, nk): chunk_idx [bs, kv, nk] int64;
            nk: number of selected chunks
        """
        del kv_len  # cached shape views already encode this; kept in signature for callers
        # stage1_prepare NVTX removed — wrapped only Python attribute reads
        # (no kernel work) and added a push/pop pair to the hot path.
        n_chunks = self._cached_quest_n_chunks
        nk = self._cached_quest_nk
        scores_out = self._cached_scores_view
        chunk_min = self._cached_chunk_min_views[layer_idx]
        chunk_max = self._cached_chunk_max_views[layer_idx]
        chunk_valid = self._cached_chunk_valid_views[layer_idx]

        nvtx.range_push("chunk_minmax")
        self._quest_kernel.quest_chunk_score(raw_q, chunk_min, chunk_max, scores_out, n_chunks, chunk_valid)
        nvtx.range_pop()

        nvtx.range_push("chunk_topk")
        chunk_idx = self._topk_3d(scores_out, nk)
        nvtx.range_pop()

        if self._topk_accuracy_check:
            _, torch_chunk_idx = torch.topk(scores_out.float(), k=nk, dim=-1)
            for h in range(min(chunk_idx.shape[1], 4)):
                radix_set = set(chunk_idx[0, h].cpu().tolist())
                torch_set = set(torch_chunk_idx[0, h].cpu().tolist())
                recall = len(radix_set & torch_set) / nk if nk > 0 else 1.0
                if recall < 1.0:
                    print(f"[TopK Check] chunk_topk H{h} L={scores_out.shape[-1]} k={nk} recall={recall*100:.1f}%", flush=True)

        chunk_idx = chunk_idx.clamp(min=0, max=n_chunks - 1)
        return chunk_idx, nk


    # ------------------------------------------------------------------
    # CDS: Coarse → Dense/Sparse rerank + final topk
    # ------------------------------------------------------------------
    @staticmethod
    def _partial_chunk_density_scores_torch(
        chunk_idx: torch.Tensor,
        centroid_all: torch.Tensor,
        raw_q: torch.Tensor,
        out_scores: torch.Tensor,
    ) -> None:
        """PyTorch reference for partial_chunk_density_scores_cuda: centroid[chunk_id] @ q."""
        n_chunks = centroid_all.shape[2]
        head_dim = centroid_all.shape[3]
        gather_idx = chunk_idx.clamp(min=0, max=n_chunks - 1).unsqueeze(-1).expand(
            -1, -1, -1, head_dim,
        )
        cent = torch.gather(centroid_all, 2, gather_idx)
        scores = (cent.float() * raw_q.float().unsqueeze(2)).sum(dim=-1)
        invalid = (chunk_idx < 0) | (chunk_idx >= n_chunks)
        out_scores.copy_(scores.masked_fill(invalid, -1e30))


    def _get_real_keys_fp32_for_retrieval(
        self,
        layer_idx: int,
        kv_len_rb: int,
        key_data: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], int]:
        """Retrieval-zone 全精度原始 key（与 pipeline recall GT 同源），用于 exact q·k。

        Offload：始终用 unified_keys_cpu[:cpu_valid_end]（GPU 上无完整 retrieval key）。
        非 Offload：优先用 compute 传入的 key_data（unified_keys_gpu retrieval 切片），否则回退切片。
        """
        if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
            cpu_len = int(self.cpu_valid_end[layer_idx])
            if cpu_len <= 0:
                return None, 0
            keys = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
                self.device, non_blocking=True,
            ).float()
            return keys, cpu_len
        if key_data is not None:
            eff = min(int(kv_len_rb), int(key_data.shape[2]))
            if eff <= 0:
                return None, 0
            return key_data[:, :, :eff, :].contiguous().float(), eff
        retrieval_start = self.sink_size
        if kv_len_rb <= 0:
            return None, 0
        keys = self.unified_keys_gpu[layer_idx][
            :, :, retrieval_start:retrieval_start + kv_len_rb, :
        ].float()
        return keys, kv_len_rb



    def _cds_build_dense_mask_torch(
        self,
        chunk_idx: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        bs: int,
    ) -> torch.Tensor:
        """Classify dense chunks: PyTorch centroid@q + torch.topk (ablation / density-torch)."""
        centroid_all = self._chunk_key_centroid_gpu[layer_idx][:bs]
        density_scores = self._cached_cds_density_scores
        self._partial_chunk_density_scores_torch(
            chunk_idx, centroid_all, raw_q, density_scores,
        )
        n_dense = self._cached_cds_n_dense
        dense_positions = self._torch_topk_3d(density_scores, n_dense)
        dense_mask = self._cached_dense_mask
        dense_mask.fill_(False)
        dense_mask.scatter_(2, dense_positions, True)
        return dense_mask


    def _cds_dense_sparse_select(
        self,
        chunk_idx: torch.Tensor,
        nk: int,
        kv_len: int,
        bs: int,
        kv_heads: int,
        key_kivi_packed: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        skipped_indices: Optional[torch.Tensor] = None,
        dense_mask: Optional[torch.Tensor] = None,
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
        if dense_mask is None:
            if self._cds_density_torch:
                dense_mask = self._cds_build_dense_mask_torch(
                    chunk_idx, raw_q, layer_idx, bs,
                )
            else:
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
        if self._kivi_stage2_kernel == "cuda":
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
        else:
            partial_chunk_kivi_qk_ref(
                chunk_ids=chunk_idx,
                dense_mask=dense_mask,
                packed_K=key_kivi_packed,
                chunk_min=chunk_min_layer,
                chunk_max=chunk_max_layer,
                raw_q=raw_q,
                group_size=chunk,
                dense_topk=dense_topk,
                sparse_topk=sparse_topk,
                bits=self._kivi_bits,
                out_scores=out_scores,
                out_indices=out_indices,
                neg_inf=_KIVI_NEG_INF,
            )
        nvtx.range_pop()

        # --- Step 3: global top-k (+ optional fused gather + skipped concat) ---
        actual_topk = min(self.final_topk, out_scores.shape[-1])
        if self._use_fused_global_topk:
            # Single-launch fused path: radix topk + indirect gather + skipped
            # tail copy.  ~1.6-2x faster than the 3-step chain; writes into the
            # pre-allocated _cached_cds_final_indices buffer.  The RaBitQ
            # ablation path forces torch.topk for parity, so it can't use this.
            nvtx.range_push("cds_global_topk_fused")
            if skipped_indices is None:
                # Empty placeholder so the kernel can iterate sk_count == 0
                # without a None branch on the C++ side.
                skipped_for_kernel = self._cached_cds_final_indices[:bs, :, :0]
                sk_count = 0
            else:
                skipped_for_kernel = skipped_indices
                sk_count = skipped_indices.shape[-1]
            out_buf = self._cached_cds_final_indices[:bs]

            float_topk_gather_concat_3d(
                out_scores, out_indices, skipped_for_kernel, out_buf, actual_topk,
            )
            # Slicing dim 2 of a [bs, kv, K_max] tensor yields a non-contiguous view (stride of dim 1 stays K_max, but the kernel formula in
            # h2d_gather_kv assumes dense [bs, kv, topk] packing).  The 3-step fallback below also returns a contiguous tensor (gather + cat
            # both materialise), so we mirror that here to keep callers (esp. h2d_gather_kv whose CUDA kernel does not respect strides) safe.
            # ``.contiguous()`` on int64 [bs, kv, K] is ~2-5us, dominated by the 8-13us savings vs the 3-step path.
            final_indices = out_buf[:, :, :actual_topk + sk_count].contiguous()
            nvtx.range_pop()

        else:
            # Fallback 3-step path (torch.topk backend or RaBitQ ablation):
            #   topk_pos = topk(out_scores, k)            ← rank by score
            #   final    = gather(out_indices, ..., pos)  ← map pos → token id
            #   final    = cat([final, skipped], dim=-1)  ← append skipped tail
            # NVTX gates the cat on actual work to avoid empty push/pop in
            # steady state (sk_count == 0).

            nvtx.range_push("cds_global_topk")
            if self._use_torch_cds_global_topk:
                topk_pos = self._torch_topk_3d(out_scores, actual_topk)
            else:
                topk_pos = self._topk_3d(out_scores, actual_topk)
            final_indices = torch.gather(out_indices, dim=2, index=topk_pos)
            nvtx.range_pop()

            if skipped_indices is not None and skipped_indices.shape[-1] > 0:
                nvtx.range_push("cds_skipped")
                final_indices = torch.cat([final_indices, skipped_indices], dim=-1)
                nvtx.range_pop()


        if self._cds_recall_check and layer_idx in self._cds_recall_layers:
            n_dense_log = int(getattr(self, "_cached_cds_n_dense", 0))
            self._cds_merged_recall_log(
                out_indices, final_indices, raw_q, layer_idx, nk, n_dense_log,
            )

        return final_indices





    def _cds_recall_log(self, final_indices, raw_q, layer_idx, nk, n_dense):
        if layer_idx == min(self._cds_recall_layers):
            self._cds_recall_step += 1
        if self._cds_recall_step % self._cds_recall_log_every != 0:
            return
        with torch.no_grad():
            if not (self.enable_offload and self.unified_keys_cpu is not None
                    and self.cpu_valid_end is not None):
                return
            cpu_len = int(self.cpu_valid_end[layer_idx])
            if cpu_len < 100:
                return
            keys_fp32 = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
                self.device).float()
            q_fp32 = raw_q.float()
            heads = self._cds_recall_heads or list(range(final_indices.shape[1]))
            for h in heads:
                if h >= final_indices.shape[1]:
                    continue
                scores = keys_fp32[0, h] @ q_fp32[0, h]
                cds_set = set(final_indices[0, h].cpu().tolist())
                parts = []
                for K in (100, 500):
                    if cpu_len < K:
                        continue
                    _, topk_idx = torch.topk(scores, k=K)
                    recall = len(cds_set & set(topk_idx.cpu().tolist())) / K
                    parts.append(f"R@{K}={recall*100:.1f}%")
                print(
                    f"[CDS Recall] step={self._cds_recall_step} L{layer_idx}H{h} "
                    f"nk={nk} n_dense={n_dense} n_cds={len(cds_set)} kv={cpu_len}"
                    f" | " + " ".join(parts), flush=True,
                )



    def _cds_merged_recall_log(self, merged_indices, final_indices, raw_q, layer_idx, nk, n_dense):
        if layer_idx == min(self._cds_recall_layers):
            self._cds_recall_step += 1
        if self._cds_recall_step % self._cds_recall_log_every != 0:
            return
        with torch.no_grad():
            if not (self.enable_offload and self.unified_keys_cpu is not None
                    and self.cpu_valid_end is not None):
                return
            cpu_len = int(self.cpu_valid_end[layer_idx])
            if cpu_len < 100:
                return
            keys_fp32 = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
                self.device).float()
            q_fp32 = raw_q.float()
            heads = self._cds_recall_heads or list(range(merged_indices.shape[1]))
            for h in heads:
                if h >= merged_indices.shape[1]:
                    continue
                scores = keys_fp32[0, h] @ q_fp32[0, h]
                merged_set = set(merged_indices[0, h].cpu().tolist())
                final_set = set(final_indices[0, h].cpu().tolist())
                merged_parts = []
                final_parts = []
                for K in (100, 500):
                    if cpu_len < K:
                        continue
                    _, topk_idx = torch.topk(scores, k=K)
                    gt_set = set(topk_idx.cpu().tolist())
                    merged_recall = len(merged_set & gt_set) / K
                    final_recall = len(final_set & gt_set) / K
                    merged_parts.append(f"R@{K}={merged_recall*100:.1f}%")
                    final_parts.append(f"R@{K}={final_recall*100:.1f}%")
                print(
                    f"[Merged Recall] step={self._cds_recall_step} L{layer_idx}H{h} "
                    f"nk={nk} n_dense={n_dense} n_merged={len(merged_set)} n_final={len(final_set)} kv={cpu_len}"
                    f" | merged: " + " ".join(merged_parts)
                    + f" | final: " + " ".join(final_parts), flush=True,
                )




    def _full_pipeline_recall_log(
        self,
        chunk_idx: torch.Tensor,
        nk_small: int,
        final_indices: torch.Tensor,
        raw_q: torch.Tensor,
        layer_idx: int,
        kv_len: int,
        skipped_indices: Optional[torch.Tensor] = None,
    ):
        """Accumulate recall across all layers/heads, print one aggregated summary per step.

        纯 "chunks → rerank" 路径的 recall（skipped 旁路被完全剔除，不参与计算）：
          coarse: tokens expanded from `chunk_idx`（仅 stage-1 chunk 选出的 token，不含 skipped）
          final:  `topk_indices` 砍掉末尾 sk_count 个 skipped token 后的纯 rerank 输出
                  （`_cds_dense_sparse_select` 把 skipped cat 在最后，这里反向剥掉）
        这样两侧都只衡量 "chunk_idx → rerank 选 K" 这条主路径，
        排除掉 skipped 自动拼接带来的虚高，能真实反映 rerank kernel 的能力。
        final ⊂ coarse 严格成立 → final_recall ≤ coarse_recall。

        Aggregation strategy:
          - layer_idx == 0: flush previous step's accum, decide whether THIS step logs.
          - layer_idx == last: flush current step immediately, reset.
          - 中间层: 只累加，不打印。
        Env: POLAR_PIPELINE_RECALL=1, POLAR_PIPELINE_RECALL_LOG_EVERY,
             POLAR_PIPELINE_RECALL_K (single/comma-separated Ks, or "final"/"auto")
        """
        if layer_idx == 0:
            if self._pr_logging_this_step and self._pr_step_accum:
                self._print_pipeline_recall_summary()
            self._pipeline_recall_step += 1
            self._pr_logging_this_step = (self._pipeline_recall_step % self._pipeline_recall_log_every == 0)
            if self._pr_logging_this_step:
                self._pr_step_accum = {}
                self._pr_step_kv_len = 0

        if not self._pr_logging_this_step:
            return

        with torch.no_grad():
            min_k = min(self._pipeline_recall_ks)

            if self.enable_offload and self.unified_keys_cpu is not None and self.cpu_valid_end is not None:
                cpu_len = int(self.cpu_valid_end[layer_idx])
                if cpu_len < min_k:
                    return
                keys_fp32 = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(self.device).float()
                eff_len = cpu_len
            else:
                retrieval_start = self.sink_size
                eff_len = kv_len
                if eff_len < min_k:
                    return
                keys_fp32 = self.unified_keys_gpu[layer_idx][:, :, retrieval_start:retrieval_start + eff_len, :].float()

            self._pr_step_kv_len = eff_len
            q_fp32 = raw_q.float()
            chunk = self.quest_chunk_size
            selected_k = int(final_indices.shape[-1])

            small_bases = (chunk_idx * chunk).unsqueeze(-1)
            small_offs = torch.arange(chunk, device=chunk_idx.device)
            small_tokens = (small_bases + small_offs).clamp(max=eff_len - 1)

            # 剥掉 final_indices 尾部的 sk_count 个 skipped token：
            # `_cds_dense_sparse_select` 内部 `torch.cat([rerank_topk, skipped], dim=-1)`，
            # 反向切掉末尾即可得到纯 rerank 结果。
            sk_count = 0
            if skipped_indices is not None and skipped_indices.numel() > 0:
                sk_count = int(skipped_indices.shape[-1])
            if sk_count > 0 and final_indices.shape[-1] > sk_count:
                final_for_recall = final_indices[..., :-sk_count]
            else:
                final_for_recall = final_indices

            num_heads = chunk_idx.shape[1]
            for h in range(num_heads):
                scores = keys_fp32[0, h] @ q_fp32[0, h]
                coarse_set = set(small_tokens[0, h].reshape(-1).cpu().tolist())   # 纯 stage-1 chunks
                final_set = set(final_for_recall[0, h].cpu().tolist())             # 纯 rerank, 不含 skipped

                for K_req in self._pipeline_recall_ks:
                    K = int(K_req)
                    if self._pipeline_recall_use_selected_k:
                        K = min(K, selected_k)
                    if K <= 0 or eff_len < K:
                        continue
                    _, gt_topk = torch.topk(scores, k=K)
                    gt_set = set(gt_topk.cpu().tolist())
                    if K not in self._pr_step_accum:
                        self._pr_step_accum[K] = [0] * 4
                    acc = self._pr_step_accum[K]
                    acc[0] += len(coarse_set & gt_set)
                    acc[1] += K
                    acc[2] += len(final_set & gt_set)
                    acc[3] += K

            # Last layer of this step: flush immediately so summary lands in the
            # right log line (avoid carrying accum into next step's gap).
            if layer_idx == self.layer_num - 1 and self._pr_step_accum:
                self._print_pipeline_recall_summary()
                self._pr_logging_this_step = False
                self._pr_step_accum = {}



    def _print_pipeline_recall_summary(self):
        parts = []
        # When use_selected_k is enabled, effective K may differ from requested K.
        # Print what was actually accumulated in this step.
        for K in sorted(self._pr_step_accum.keys()):
            acc = self._pr_step_accum[K]
            ch, ct, fh, ft = acc[0], acc[1], acc[2], acc[3]
            coarse_r = ch / ct * 100 if ct > 0 else 0
            final_r = fh / ft * 100 if ft > 0 else 0
            part = f"R@{K}: coarse={coarse_r:.1f}% final={final_r:.1f}%"
            parts.append(part)
        meta = ""
        if self._step_diag_meta:
            meta = (
                f" step={self._step_diag_meta.get('step', '?')} "
                f"retrieval(valid/theory)={self._step_diag_meta.get('valid_len', '?')}/{self._step_diag_meta.get('retrieval_len', '?')} "
                f"skipped={self._step_diag_meta.get('skipped', '?')} "
                f"chunks(s/l)={self._step_diag_meta.get('n_small', '?')}/{self._step_diag_meta.get('n_large', '?')} "
                f"K(core/total)={self._step_diag_meta.get('k_core', '?')}/{self._step_diag_meta.get('k_total', '?')} "
            )
        print(
            f"[Pipeline Recall]{meta} step={self._pipeline_recall_step} kv={self._pr_step_kv_len} "
            f"(agg all layers/heads) | " + " | ".join(parts),
            flush=True,
        )
        self._step_diag_meta = None




    def collision_based_topk_batch(
        self,
        query,
        key_data=None,
        key_kivi_packed=None,
        layer_idx=None,
    ):
        """Quest stage-1 + CDS dense/sparse + KIVI 4-bit final topk.

        Name kept for backward compatibility with `compute()` / timing keys /
        existing NVTX ranges (`encode_query` is now a no-op range to
        preserve the same nsys timeline tag count vs the RaBitQ baseline).

        Args:
            query:             [bs, kv_heads, 1, head_dim] raw query
            key_data:          optional [bs, kv_heads, kv_len, head_dim], used by
                               ablation paths needing exact inner product
            key_kivi_packed:   [bs, kv_heads, n_chunks_pkd, n_pack, group_size]
                               int32 — KIVI packed K (A-1 chunk-major) sliced
                               on the chunk axis to ``n_chunks_pkd``
            layer_idx:         layer index (used by chunk-metadata lookup)
        Returns:
            topk_indices:      [bs, kv_heads, final_topk] retrieval-zone indices
        """
        assert key_kivi_packed is not None, \
            "KIVI path requires key_kivi_packed"

        # KIVI has no query encode work (RaBitQ's old `encode_query` NVTX
        # range was empty here); fold the trivial shape ops into
        # stage1_chunk_pick to remove an empty NVTX segment + a pair of
        # push/pop launches from the hot path.
        bs, kv_heads, qlen, head_dim = query.shape
        assert qlen == 1, "Decode 阶段每次只处理 1 个 token"
        query_squeezed = query.squeeze(2)  # [bs, kv_heads, head_dim]
        # A-1 chunk-major: shape[2] is now n_chunks, not kv_len.  Recover
        # the token count from chunk count for downstream consumers
        # (skipped_indices clamp / final_indices clamp) which expect
        # token-coordinate sizing.
        n_chunks_pkd_view = key_kivi_packed.shape[2]
        kv_len_rb = n_chunks_pkd_view * self.quest_chunk_size

        nvtx.range_push("stage1_chunk_pick")
        sk_count = self._chunk_skipped_count[layer_idx]
        skipped = None
        if sk_count > 0:
            # _chunk_skipped_indices_gpu is filled by torch.arange over the in-range token span (see ``_record_skipped_range`` /
            # ``_compute_and_store_chunk_minmax`` tail-skip path), so values are already in [0, kv_len_rb).  No defensive clamp needed.
            skipped = self._chunk_skipped_indices_gpu[layer_idx][:bs, :, :sk_count]
        
        chunk_idx, nk = self._quest_pick_coarse_indices(
            kv_len_rb,
            raw_q=query_squeezed,
            layer_idx=layer_idx,
        )
        nvtx.range_pop()  # stage1_chunk_pick



        nvtx.range_push("stage2_cds")
        topk_indices = self._cds_dense_sparse_select(
            chunk_idx, nk, kv_len_rb, bs, kv_heads,
            key_kivi_packed,
            query_squeezed, layer_idx,
            skipped_indices=skipped,
        )
        nvtx.range_pop()  # stage2_cds


        if self._pipeline_recall_check:
            try:
                self._full_pipeline_recall_log(
                    chunk_idx, nk, topk_indices,
                    query_squeezed, layer_idx, kv_len_rb,
                    skipped_indices=skipped,
                )
            except Exception as _pr_err:
                # recall log 内部 .cpu() 会 sync GPU；若上游 kernel 越界，错误会在此爆发。
                # 不要把整个 eval 拖崩 —— 关掉 recall log，继续跑。
                import traceback as _tb
                print(
                    f"[Pipeline Recall] ERROR at layer={layer_idx} step={self._pipeline_recall_step}: "
                    f"{type(_pr_err).__name__}: {_pr_err}\n"
                    f"  chunk_idx range  = [{int(chunk_idx.min())}, {int(chunk_idx.max())}] vs n_chunks={kv_len_rb // self.quest_chunk_size}\n"
                    f"  topk_indices rng = [{int(topk_indices.min())}, {int(topk_indices.max())}] vs kv_len_rb={kv_len_rb}\n"
                    f"  skipped_count    = {self._chunk_skipped_count[layer_idx]}\n"
                    f"  → 关闭 POLAR_PIPELINE_RECALL 以避免后续 sync 崩溃；真正的越界来源请用 CUDA_LAUNCH_BLOCKING=1 复跑。",
                    flush=True,
                )
                _tb.print_exc()
                self._pipeline_recall_check = False  # 关掉后续调用，否则每 step 都会崩

        return topk_indices









    def sync(self, layer_idx, batch_idx):
        """
        GPU版本:数据已经在prefill阶段直接存储到GPU,无需额外同步
        """
        pass  # GPU版本中所有数据已在prefill时存储到GPU,无需额外操作









    # ------------------------------------------------------------------
    # Chunk distribution stats（only on true top-K indices）
    # ------------------------------------------------------------------
    # 三个最有信息量的指标，用来回答："真 top-K 在 chunk 上是集中还是散乱？"
    #   n_chunks_hit : 含至少 1 个真 top-K 的 chunk 数 (越小 = 越集中)
    #   top1_share   : max(chunk 内 top-K 数) / K     (单 chunk 集中度峰值)
    #   oracle_at_nk : 在 chunk_size + nk 预算下，挑"top-K 数最多"的 nk 个 chunk
    #                  能拿到的 R@K —— 这是 chunk 路由的理论天花板
    # @staticmethod
    # def _compute_chunk_dist_stats(
    #     true_topK_indices: torch.Tensor,
    #     kv_len_eff: int,
    #     chunk_size: int,
    #     nk_chunks: int,
    # ) -> tuple:
    #     K = int(true_topK_indices.numel())
    #     if K == 0 or kv_len_eff <= 0 or chunk_size <= 0:
    #         return 0, 0.0, 0.0
    #     n_chunks_total = (kv_len_eff + chunk_size - 1) // chunk_size
    #     chunk_ids = (true_topK_indices.to(torch.long) // chunk_size).cpu()
    #     counts = torch.bincount(chunk_ids, minlength=n_chunks_total)
    #     n_chunks_hit = int((counts > 0).sum().item())
    #     top1_share = float(counts.max().item()) / K
    #     nk_eff = max(1, min(int(nk_chunks), int(counts.numel())))
    #     top_counts, _ = torch.topk(counts, k=nk_eff)
    #     oracle_at_nk = min(1.0, float(top_counts.sum().item()) / K)
    #     return n_chunks_hit, top1_share, oracle_at_nk





    # ------------------------------------------------------------------
    # Quest Stage-1 candidate Recall@K / Recall@C 消融
    # ------------------------------------------------------------------
    # 与 _run_coarse_recall_ablation 完全对齐 env：
    #   POLAR_RECALL_ABLATION_LAYERS / HEADS / K / LOG_EVERY
    # 区别：直接吃 Quest 已选好的 token 索引（coarse_indices），不再走 cache_cnt 路径。
    # 额外：POLAR_CHUNK_DIST_LOG=1 时，同步打印真 top-K 的 chunk 分布统计。
    # def _run_quest_coarse_recall_ablation(
    #     self,
    #     query: torch.Tensor,
    #     key_data,
    #     coarse_indices: torch.Tensor,
    #     layer_idx: int,
    #     candidate_num: int,
    # ) -> None:
    #     if not hasattr(self, '_first_stage_recall_cfg'):
    #         def _parse_int_list(env_name, default):
    #             raw = os.environ.get(env_name)
    #             if not raw:
    #                 return tuple(default)
    #             try:
    #                 return tuple(int(x.strip()) for x in raw.split(',') if x.strip() != '')
    #             except ValueError:
    #                 print(
    #                     f"[Quest Stage1] WARN 无法解析 {env_name}={raw!r}，使用默认 {default}",
    #                     flush=True,
    #                 )
    #                 return tuple(default)
    #         self._first_stage_recall_cfg = {
    #             'target_layers': set(_parse_int_list('POLAR_RECALL_ABLATION_LAYERS', [4, 17, 30])),
    #             'target_heads':  tuple(_parse_int_list('POLAR_RECALL_ABLATION_HEADS', [0, 4])),
    #             'K':             int(os.environ.get('POLAR_RECALL_ABLATION_K', '100')),
    #             'log_every':     max(1, int(os.environ.get('POLAR_RECALL_ABLATION_LOG_EVERY', '50'))),
    #             'chunk_dist_log': bool(int(os.environ.get('POLAR_CHUNK_DIST_LOG', '0'))),
    #         }
    #     cfg = self._first_stage_recall_cfg
    #     target_layers = cfg['target_layers']
    #     if layer_idx not in target_layers:
    #         return
    #     target_heads = cfg['target_heads']
    #     K = cfg['K']

    #     if not hasattr(self, '_first_stage_recall_stats'):
    #         self._first_stage_recall_stats = {
    #             'step_count': 0,
    #             'current_step_layer_recalls': {},
    #             'current_step_layer_chunkdist': {},
    #         }
    #     stats = self._first_stage_recall_stats
    #     first_target_layer = min(target_layers)
    #     last_target_layer = max(target_layers)
    #     if layer_idx == first_target_layer:
    #         stats['step_count'] += 1
    #         stats['current_step_layer_recalls'].clear()
    #         stats['current_step_layer_chunkdist'].clear()
    #     step = stats['step_count']

    #     with torch.no_grad():
    #         bs, kv_heads, _ = coarse_indices.shape

    #         # ---- 取 retrieval keys (offload-aware) ----
    #         if (
    #             self.enable_offload
    #             and self.unified_keys_cpu is not None
    #             and self.cpu_valid_end is not None
    #         ):
    #             cpu_len = int(self.cpu_valid_end[layer_idx])
    #             if cpu_len < K:
    #                 return
    #             keys_layer = self.unified_keys_cpu[layer_idx][:, :, :cpu_len, :].to(
    #                 self.device, non_blocking=True
    #             ).to(torch.float32)
    #             kv_len_eff = cpu_len
    #         else:
    #             if key_data is None:
    #                 return
    #             keys_layer = key_data.to(torch.float32)
    #             kv_len_eff = key_data.shape[2]

    #         cand_eff = max(1, min(int(candidate_num), kv_len_eff))
    #         if kv_len_eff < K:
    #             return

    #         chunk_size = max(1, int(getattr(self, 'quest_chunk_size', 8)))
    #         nk_chunks = max(1, cand_eff // chunk_size)

    #         b = 0
    #         per_head_results = []
    #         per_head_chunkdist = []
    #         for h in target_heads:
    #             if h >= kv_heads:
    #                 continue
    #             q_h = query[b, h].to(torch.float32)
    #             k_h = keys_layer[b, h, :kv_len_eff, :]
    #             exact_scores = torch.matmul(k_h, q_h)

    #             _, true_topK_indices = torch.topk(exact_scores, k=K, dim=-1)
    #             true_top_K_set = set(true_topK_indices.cpu().tolist())

    #             _, true_topC_indices = torch.topk(exact_scores, k=cand_eff, dim=-1)
    #             true_top_C_set = set(true_topC_indices.cpu().tolist())

    #             cand_set = set(coarse_indices[b, h].cpu().tolist())
    #             recall_K = len(cand_set & true_top_K_set) / K
    #             recall_C = len(cand_set & true_top_C_set) / cand_eff
    #             per_head_results.append((h, recall_K, recall_C))

    #             if cfg['chunk_dist_log']:
    #                 n_hit, top1, oracle_nk = self._compute_chunk_dist_stats(
    #                     true_topK_indices, kv_len_eff, chunk_size, nk_chunks,
    #                 )
    #                 per_head_chunkdist.append((h, n_hit, top1, oracle_nk))

    #         stats['current_step_layer_recalls'][layer_idx] = per_head_results
    #         if cfg['chunk_dist_log']:
    #             stats['current_step_layer_chunkdist'][layer_idx] = per_head_chunkdist

    #         if layer_idx == last_target_layer and step % cfg['log_every'] == 0:
    #             parts = []
    #             for l in sorted(stats['current_step_layer_recalls'].keys()):
    #                 for h, rK, rC in stats['current_step_layer_recalls'][l]:
    #                     parts.append(
    #                         f"L{l}H{h} R@K={rK*100:.1f}% R@C={rC*100:.1f}%"
    #                     )
    #             print(
    #                 f"[Quest Stage1] step={step} K={K} cand={cand_eff} | "
    #                 + " | ".join(parts),
    #                 flush=True,
    #             )

    #             if cfg['chunk_dist_log']:
    #                 n_chunks_total = (kv_len_eff + chunk_size - 1) // chunk_size
    #                 cd_parts = []
    #                 for l in sorted(stats['current_step_layer_chunkdist'].keys()):
    #                     for h, n_hit, top1, oracle_nk in stats['current_step_layer_chunkdist'][l]:
    #                         cd_parts.append(
    #                             f"L{l}H{h} hit={n_hit}/{K} top1={top1*100:.1f}% "
    #                             f"oracle@nk={oracle_nk*100:.1f}%"
    #                         )
    #                 print(
    #                     f"[ChunkDist] step={step} K={K} chunk={chunk_size} "
    #                     f"n_chunks={n_chunks_total} nk={nk_chunks} | "
    #                     + " | ".join(cd_parts),
    #                     flush=True,
    #                 )
