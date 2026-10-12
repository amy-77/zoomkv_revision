import torch
from .cache import KV_Cache


class flash_attn_cache(KV_Cache):
    """
    A class representing the KV Cache of Full flash-attn.
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
        enable_decode_timing: bool = True  # 🔧 默认启用性能追踪
    ) -> None:
        super().__init__(layer_num, batch_size, max_length, num_key_value_heads, num_heads, head_dim, dtype, layer_mapping, num_gpus, model_size)

        self.valid_start = valid_start  # start index of valid tokens for each batch
        self.valid_length = None        # valid seq length of each batch
        self.batch_indices = torch.arange(self.batch_size, dtype=torch.int32, device=self.layer_mapping[str(0)])
        
        # 🔧 启用 decode 阶段计时 - 用于性能分析
        self.enable_decode_timing = enable_decode_timing
        self.decode_times = []  # 存储每步解码的 attention 时间（ms）
        self.decode_step_count = 0  # decode 步数计数器

        print(f"\n[KV Cache 初始化] Step 1/4: 计算内存需求...")
        import sys
        sys.stdout.flush()
        
        self.allocated = self.pre_allocate_decision()
        
        # 强制使用 GPU 分配（避免 CPU pin_memory 的慢速分配）
        # 原因：CPU pin_memory 对大缓存（>10GB）分配极慢，可能需要数分钟
        # GPU 分配几乎是瞬时的，A100 80GB 有足够空间
        kv_consumption = (2 * self.layer_num * self.batch_size * self.max_length * self.kv_head * self.head_dim * 2) / 1024 / 1024 / 1024
        print(f"[KV Cache 初始化] 计算结果:")
        print(f"  - 层数: {self.layer_num}")
        print(f"  - 最大长度: {self.max_length}")
        print(f"  - KV头数: {self.kv_head}")
        print(f"  - 头维度: {self.head_dim}")
        print(f"  - 数据类型: {self.dtype}")
        print(f"  - 预计消耗: {kv_consumption:.2f} GB")
        print(f"  - 可用内存: {self.free_memory:.2f} GB")
        sys.stdout.flush()
        
        print(f"\n[KV Cache 初始化] Step 2/4: 决定分配策略...")
        print(f"  ✓ 强制使用 GPU 分配（避免 CPU pin_memory 慢速分配）")
        self.allocated = True  # 强制 GPU 分配
        sys.stdout.flush()
        
        print(f"\n[KV Cache 初始化] Step 3/4: 开始分配 {self.layer_num} 层缓存...")
        print(f"  每层大小: {kv_consumption / self.layer_num:.3f} GB")
        sys.stdout.flush()
        if self.allocated:
            import time
            import sys
            start_time = time.time()
            
            self.key_cache = []
            self.value_cache = []
            
            for ldx in range(self.layer_num):
                layer_start = time.time()
                
                # 分配 Key cache
                self.key_cache.append(torch.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device=self.layer_mapping[str(ldx)],
                    dtype=self.dtype
                ))
                
                # 分配 Value cache
                self.value_cache.append(torch.empty(
                    self.batch_size,
                    self.max_length,
                    self.kv_head,
                    self.head_dim,
                    device=self.layer_mapping[str(ldx)],
                    dtype=self.dtype
                ))
                
                layer_time = time.time() - layer_start
                
                # 每5层打印一次详细进度（更频繁的反馈）
                if (ldx + 1) % 5 == 0 or ldx == self.layer_num - 1:
                    elapsed = time.time() - start_time
                    print(f"  进度: {ldx+1}/{self.layer_num} "
                          f"({100*(ldx+1)/self.layer_num:.1f}%) | "
                          f"本层耗时: {layer_time*1000:.1f}ms | "
                          f"总耗时: {elapsed:.2f}s")
                    sys.stdout.flush()
        else:
            # 不应该到这里，因为已经强制 GPU 分配
            raise RuntimeError("不应该使用 CPU pin_memory 分配")
        
        total_time = time.time() - start_time
        print(f"\n[KV Cache 初始化] Step 4/4: 创建 CUDA 流和事件...")
        sys.stdout.flush()
        
        self.copystream = torch.cuda.Stream()
        self.copyevents = {}
        self.KVreadyevents = {}
        
        print(f"  - 创建 CUDA Stream...")
        sys.stdout.flush()
        
        device_list = sorted(set(self.layer_mapping.values()), key=lambda x: int(x.split(':')[-1]))
        print(f"  - 为 {len(device_list)} 个设备创建 CUDA Events: {device_list}")
        sys.stdout.flush()
        
        for device_idx in device_list:
            with torch.cuda.device(device_idx):
                self.copyevents[device_idx] = torch.cuda.Event()
                self.KVreadyevents[device_idx] = torch.cuda.Event()
        
        print(f"\n✓ [KV Cache 初始化完成] 总耗时: {total_time:.2f}s")
        print(f"  - 已分配 {self.layer_num * 2} 个缓存 (Key + Value)")
        print(f"  - 总显存占用: {kv_consumption:.2f} GB")
        print(f"  - 平均每层: {total_time/self.layer_num*1000:.1f}ms\n")
        sys.stdout.flush()
    
    # decide whether to pre-allocate GPU memory before prefilling
    def pre_allocate_decision(self):
        kv_consumption = (2 * self.layer_num * self.batch_size * self.max_length * self.kv_head * self.head_dim * 2) / 1024 / 1024 / 1024
        return self.free_memory > kv_consumption * 1.3
    
    def move_gpu(self):
        if not self.allocated:
            for ldx in range(self.layer_num):
                self.key_cache[ldx] = self.key_cache[ldx].to(self.layer_mapping[str(ldx)])
                self.value_cache[ldx] = self.value_cache[ldx].to(self.layer_mapping[str(ldx)])

    
    def prefill_update_kv_cache(self, query_states, key_states, value_states, layer_idx, start_bdx):
        """
        update part of batches keys and values, start from start_bdx
        Args:
            query_states: (bsz, seq_len, num_heads, head_dim)
            key_states: (bsz, seq_len, kv_head, head_dim)
            start_bdx: the start index of the batch (=batch_idx)
        """
        bsz, seq_len, _, _ = key_states.shape
        assert bsz == 1, f"Multi-batch prefilling only support prefill single batch one by one."
        assert seq_len <= self.max_length, f"Prefilling sequence length {seq_len} exceeds max length {self.max_length}."

        self.KVreadyevents[self.layer_mapping[str(layer_idx)]].record()

        if self.valid_length is None:
            self.valid_length = torch.from_numpy(seq_len - self.valid_start).to(torch.int32).to(self.layer_mapping[str(0)])
        
        _valid_start = self.valid_start[start_bdx]
        _valid_length = seq_len - _valid_start

        with torch.cuda.stream(self.copystream):
            self.KVreadyevents[self.layer_mapping[str(layer_idx)]].wait()    # wait for KV ready
            self.key_cache[layer_idx][start_bdx:start_bdx+bsz, :_valid_length, :, :].copy_(key_states[:, _valid_start:, :, :], non_blocking=True)
            self.value_cache[layer_idx][start_bdx:start_bdx+bsz, :_valid_length, :, :].copy_(value_states[:, _valid_start:, :, :], non_blocking=True)
            self.copyevents[self.layer_mapping[str(layer_idx)]].record()
        
        if (layer_idx == self.layer_num - 1) and (start_bdx + bsz == self.batch_size):
            self.context += seq_len
        
        return key_states[:, _valid_start:, :, :], value_states[:, _valid_start:, :, :]
    
    
    
    
    def sync(self, layer_idx, start_bdx):
        self.copyevents[self.layer_mapping[str(layer_idx)]].wait()  # wait for copy done

    
    def decode_update_kv_cache(self, key_states, value_states, layer_idx):
        """
        update all batch of the key and value cache for decoding
        Args:
            key_states: (bsz, seq_len(=1), kv_head, head_dim)
            value_states: (bsz, seq_len(=1), kv_head, head_dim)
            layer_idx: the index of the layer
        """
        
        # 🔧 修复时序不一致问题：在第一层立即更新计数器
        # 这样所有后续层都能看到正确的 valid_length
        if layer_idx == 0:
            self.context += 1
            self.valid_length += 1
        
        # 使用 -1 补偿获取正确的插入位置
        # 因为 valid_length 已经在第一层 +1 了，所以需要 -1 来获取当前要写入的位置
        insert_position = self.valid_length - 1
        
        self.key_cache[layer_idx][self.batch_indices, insert_position, :, :] = key_states[:, 0, :, :]
        self.value_cache[layer_idx][self.batch_indices, insert_position, :, :] = value_states[:, 0, :, :]
        
        # ⚡ 性能关键优化：直接返回整个 cache，不做切片
        # Flash Attention 会通过 cache_seqlens 参数知道有效长度
        # 避免每层每步的切片开销（48层 × 每步 = 大量tensor metadata操作）
        return self.key_cache[layer_idx], self.value_cache[layer_idx]
