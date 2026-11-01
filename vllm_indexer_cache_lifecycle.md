# vLLM Indexer Cache 源码生命周期

> 仓库：`vllm-project/vllm`  
> 基线 commit：`06f36c12229751e749116b4cf51d4511e51df99d`（2026-10-31）  
> 每个代码块上方都标出对应的原文件；代码块内只放该文件中的实际代码。
>
> 覆盖：
> 1. DeepSeek V3.2 `DeepseekV32IndexerCache`
> 2. DeepSeek V4 `DeepseekV4IndexerCache`
> 3. DeepSeek V4.1 同名 cache + source layer 共享 + candidate block
> 4. MiniMax M3 / MSA `MiniMaxM3IndexerCache`

---

## 0. 整条链路先看一遍

vLLM 的 indexer cache 复用了普通 KV cache 的 block 生命周期，但它是独立的 layer cache：

```text
模型构造 indexer cache 对象
  -> get_kv_cache_spec() 告诉 KV cache manager 需要什么格式
  -> KV cache manager 按 block 分配一块 layer cache
  -> bind_kv_cache() 把分配到的 tensor 绑到 indexer cache
  -> indexer 前向按 slot_mapping 写入 index K
  -> indexer 用 block_table/page_table 读取 index K
  -> 计算 MQA logits，做 top-k
  -> 主 attention 读取 top-k 选中的主 KV page
  -> index cache 所属 KV cache group 的 block 被释放时，index page 一起不可达
```

上面每一步在代码里对应的入口：

| 阶段 | 主要函数 / 类 | 调用者 |
|---|---|---|
| 声明 cache 格式 | `DeepseekV32IndexerCache.get_kv_cache_spec` / `DeepseekV4IndexerCache.get_kv_cache_spec` / `MiniMaxM3IndexerCache.get_kv_cache_spec` | KV cache manager |
| 绑定分配结果 | `bind_kv_cache` | model runner |
| 写入 index K | `ops.indexer_k_quant_and_cache` / compressor / `indexer_k_norm_rope_store` / MSA fused insert | indexer attention 前向 |
| 读取 index K | `SparseAttnIndexer` / `MiniMaxM3IndexerImpl.forward` | model 每层 attention |
| top-k | `get_indexer_topk` / `minimax_m3_index_decode` | indexer 内部 |
| 驱逐 | `BlockPool.free_blocks` | scheduler / KV cache manager |

---

## 1. 公共抽象：`MLAAttentionSpec` 和 `SparseCacheRole`

### 1.1 `SparseCacheRole`

**源码文件**：`vllm/v1/kv_cache_interface.py`  
**所在类**：`SparseCacheRole`

```python
class SparseCacheRole(str, Enum):
    SPARSE = "sparse"
    INDEXER = "indexer"
```

作用：

- `SPARSE`：主 sparse attention 的 KV cache（例如 MLA latent）；
- `INDEXER`：lightning indexer 的 key-only side cache。

调用关系：

- V3.2 的 `DeepseekV32IndexerCache.get_kv_cache_spec()` 显式返回 `cache_role=SparseCacheRole.INDEXER`；
- MSA 的 `MiniMaxM3IndexerCache.get_kv_cache_spec()` 也显式返回 `INDEXER`；
- V4 / V4.1 的 `DeepseekV4IndexerCache.get_kv_cache_spec()` 当前代码没有显式设置 `cache_role`，默认是 `SPARSE`；这一点在阅读时不能想当然。

### 1.2 `MLAAttentionSpec` 的关键字段

**源码文件**：`vllm/v1/kv_cache_interface.py`  
**所在类**：`MLAAttentionSpec`

```python
@dataclass(frozen=True)
class MLAAttentionSpec(FullAttentionSpec):
    # TODO(Lucas/Chen): less hacky way to do this
    cache_dtype_str: str | None = None
    topk_buffer_width: int | None = None
    """Width of the model's sparse top-k indices buffer, including padding."""
    # DeepseekV4 only fields. Non-DeepseekV4 MLA models leave these at defaults.
    alignment: int | None = None  # Default to None for no padding.
    model_version: str | None = None
    cache_role: SparseCacheRole = SparseCacheRole.SPARSE
    is_index_group_leader: bool = False
    # Group capability enabled when any member flattens a non-causal query block
    # into decode rows. Runtime metadata still selects causal vs. non-causal mode.
    non_causal_multi_token_decode: bool = False
    # MLA stores a single latent vector per state; there is no separate V.
    head_size_v: int = 0
```

变量全称与含义：

| 字段 | 全称 / 含义 |
|---|---|
| `cache_dtype_str` | cache data type string，e.g. `"fp8_ds_mla"` |
| `topk_buffer_width` | top-k indices buffer width，包含 padding 的列数 |
| `alignment` | page/block 字节对齐要求 |
| `model_version` | 模型版本标记，用于区分 V4/V4.1 的布局 |
| `cache_role` | 这个 layer cache 是 sparse 主 cache 还是 indexer cache |
| `is_index_group_leader` | 该 layer 是否是 index group 的 leader |
| `non_causal_multi_token_decode` | 是否支持非因果的多 token decode |
| `head_size_v` | value head size；MLA 没有独立 V，所以通常为 0 |

### 1.3 index page 大小如何算出来

**源码文件**：`vllm/v1/kv_cache_interface.py`  
**所在类**：`AttentionSpec`

```python
@property
def state_content_size_bytes(self) -> int:
    """Bytes per (head slot, stored state) cell of the page."""
    if self.state_content_bytes is not None:
        return self.state_content_bytes
    return (self.head_size + self.head_size_v) * get_dtype_size(self.dtype)

@property
def unpadded_page_size_bytes(self) -> int:
    return self.num_heads * self.num_states * self.state_content_size_bytes

@property
def page_size_bytes(self) -> int:
    if self.page_size_padded is not None:
        assert self.page_size_padded >= self.unpadded_page_size_bytes
        return self.page_size_padded
    return self.unpadded_page_size_bytes
```

`num_states` 的来源：

**源码文件**：`vllm/v1/kv_cache_interface.py`  
**所在类**：`KVCacheSpec`

```python
def get_num_kernel_states(self, kernel_block_size: int) -> int:
    if self.tokens_per_state > 0:
        return kernel_block_size // self.tokens_per_state
    return 1
```

因此：

```text
page_size_bytes
  = num_heads
  × (block_size / tokens_per_state)
  × state_content_size_bytes
```

调用链：

```text
KVCacheManager
  -> spec.page_size_bytes
  -> 计算每个 block 的字节数
  -> 分配 KV cache tensor
  -> 按 layer 切 view
  -> bind_kv_cache
```

---

## 2. DeepSeek V3.2：`DeepseekV32IndexerCache`

### 2.1 cache 类本身

**源码文件**：`vllm/model_executor/models/deepseek_v2.py`  
**所在类**：`DeepseekV32IndexerCache`  
**调用者**：`Indexer.__init__` 构造；`get_kv_cache_spec` 由 KV cache manager 调用；`bind_kv_cache` 由 model runner 调用。

```python
class DeepseekV32IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self, head_dim: int, dtype: torch.dtype, prefix: str, cache_config: CacheConfig
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]: the indexer kernels and
        # kv_cache_as_quant_view index a 3-D block-major cache.
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
        )  # Only has one vector instead of K + V

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV32IndexerBackend
```

为什么这样设计：

- `self.kv_cache` 一开始是空 tensor；
- `get_kv_cache_spec()` 只声明格式，不分配；
- `bind_kv_cache()` 接收 shape 为 `[B, H=1, N, C]` 的 layer cache，去掉 head 维变成 3D；
- `head_dim=132`，其中 128 B 是 FP8 index K，4 B 是 scale；
- `block_size=64`，与 DSA 主 attention 的 block 对齐。

### 2.2 `Indexer` 如何创建 cache

**源码文件**：`vllm/model_executor/models/deepseek_v2.py`  
**所在类**：`Indexer.__init__`  
**调用者**：`DeepseekV32Attention` 初始化时创建 `Indexer`，`Indexer` 再创建 `DeepseekV32IndexerCache`。

```python
        # NOTE: (zyongye) we use fp8 naive cache,
        #       where we store value in fp8 and scale in fp32
        #       per self.quant_block_size element
        assert cache_config is not None
        self.k_cache = DeepseekV32IndexerCache(
            head_dim=self.head_dim + self.head_dim // self.quant_block_size * 4,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.prefix = prefix
        from vllm.v1.attention.backends.mla.indexer import get_max_prefill_buffer_size

        self.max_total_seq_len = get_max_prefill_buffer_size(vllm_config)
        self.indexer_op = SparseAttnIndexer(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
        )
```

变量全称：

| 变量 | 全称 / 含义 |
|---|---|
| `head_dim` | index head dimension = 128 |
| `quant_block_size` | quantization block size = 128 |
| `self.k_cache` | index key cache，`DeepseekV32IndexerCache` |
| `self.indexer_op` | indexer operator，`SparseAttnIndexer` |
| `self.topk_tokens` | top-k tokens，最终选多少个 token |
| `self.max_model_len` | 模型最大序列长度 |
| `self.max_total_seq_len` | 一个 prefill batch 的最大总长度 |
| `self.topk_indices_buffer` | top-k 结果 buffer，不是 KV cache |

为什么调用：

- `Indexer` 需要自己的 K cache 来保存 index K；
- `SparseAttnIndexer` 需要 `k_cache` 的 `kv_cache` tensor 在 forward 中写入/读取；
- `get_kv_cache_spec` 由上层 manager 在分配阶段调用，而不是在 forward 里动态分配。

### 2.3 写入：`indexer_k_quant_and_cache`

**源码文件**：`vllm/model_executor/layers/sparse_attn_indexer.py`  
**所在函数**：`sparse_attn_indexer`  
**调用者**：`SparseAttnIndexer.forward_cuda` → `torch.ops.vllm.sparse_attn_indexer` → 本函数。

```python
    if not skip_k_cache_insert:
        assert k is not None
        k, slot_mapping_for_cache = maybe_gather_indexer_k(
            k,
            slot_mapping,
            num_decode_tokens,
            use_pcp,
            pcp_shard_decode_requests=pcp_shard_decode_requests,
        )
        # scale_fmt can be None, but the function expects str
        assert scale_fmt is not None
        assert not use_fp4_cache, "Unfused FP4 Insert is not supported yet"
        ops.indexer_k_quant_and_cache(
            k,
            kv_cache,
            slot_mapping_for_cache,
            quant_block_size,
            "mxfp8" if mqa_logits_use_mx_scales() else scale_fmt,
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `skip_k_cache_insert` | 是否跳过 K cache 写入；V4 的 compressor 已经写过时为 True |
| `k` | 当前 token 的 index key，形状 `[num_tokens, head_dim]` |
| `slot_mapping` | 主 KV 的 token slot 映射 |
| `slot_mapping_for_cache` | 经过 PCP gather 后的 cache slot mapping |
| `kv_cache` | `DeepseekV32IndexerCache.kv_cache` |
| `quant_block_size` | 128 维一个 scale |
| `use_fp4_cache` | 是否使用 FP4 index cache |

对应的 CUDA kernel：

**源码文件**：`csrc/cache_kernels.cu`  
**所在函数**：`indexer_k_quant_and_cache_kernel`

```cpp
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
  const scalar_t* __restrict__ k,            // [num_tokens, head_dim]
  cache_t* __restrict__ kv_cache,            // [num_blocks, block_size, cache_stride]
  const int64_t* __restrict__ slot_mapping,  // [num_tokens]
  const int head_dim,                        // dimension of each head
  const int quant_block_size,                // quantization block size
  const int cache_block_size,                // cache block size
  const int cache_stride,                    // stride for each token in kv_cache
  const bool use_ue8m0                       // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx =
      (blockIdx.y * blockDim.y * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(k))[
      (token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }
  __syncwarp();

  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
  }
  __syncwarp();
  float scale = fmaxf(amax, 1e-4) / 448.0f;
  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset =
      block_idx * cache_block_size * cache_stride
      + block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride
        + cache_block_size * head_dim
        + (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}
```

关键点：

- `slot_idx` 来自主 KV 的 slot mapping，所以 index K 的写入位置和主 KV 的 slot 是同一套位置空间；
- index layer 是否与主 KV 落在同一个 KV cache group，取决于 `cache_role` / spec 的合并规则；
- 但无论 group 如何合并，index cache 都没有自己的 page allocator，也不维护独立 prefix tree；
- `block_idx = slot_idx / cache_block_size`，`block_offset = slot_idx % cache_block_size`；
- `scale` 写到 block 内的 scale 区域，而不是单独的 index page；
- `use_ue8m0` 时 scale 取 2 的幂。

### 2.4 读取：paged MQA logits + top-k

**源码文件**：`vllm/model_executor/layers/sparse_attn_indexer.py`  
**所在函数**：`sparse_attn_indexer`（decode 分支）  
**调用者**：`SparseAttnIndexer.forward_cuda` 的 custom op。

```python
            logits = fp8_fp4_paged_mqa_logits(
                (padded_q_quant_cast, padded_q_scale),
                kv_cache,
                decode_weights,
                seq_lens,
                decode_metadata.block_table,
                decode_metadata.schedule_metadata,
                max_model_len=max_model_len,
                indices=decode_metadata.indices,
            )
            # Keeping bf16 halves top-k/candidate reads. SM100 decode is varlen:
            # one context length per row.
            if dcp_world_size != 1 or not get_indexer_topk(
                topk_backend
            ).takes_bf16_logits(logits, topk_tokens, logits.shape[0]):
                logits = as_fp32_logits(logits, seq_lens.view(-1))
```

然后：

**源码文件**：`vllm/model_executor/layers/sparse_attn_indexer.py`  
**所在函数**：`sparse_attn_indexer`

```python
        topk_indices = topk_indices_buffer[:num_padded_tokens, :topk_tokens]

        # The backend comes from the layer (config is only readable at model
        # construction); dispatchers are cached per backend.
        get_indexer_topk(topk_backend)(
            logits,
            seq_lens,
            next_n,
            topk_indices,
            topk_tokens,
            attn_metadata_narrowed.max_seq_len,
        )
```

调用链：

```text
SparseAttnIndexer.forward_cuda
  -> torch.ops.vllm.sparse_attn_indexer
  -> sparse_attn_indexer
  -> fp8_fp4_paged_mqa_logits
  -> get_indexer_topk
  -> topk_indices_buffer
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `padded_q_quant_cast` | 补齐/量化后的 index query |
| `padded_q_scale` | FP4 query 的 scale |
| `decode_weights` | 每个 index head 的权重 |
| `seq_lens` | 每个 decode 请求的 context length |
| `block_table` | 主 KV 的 block table |
| `schedule_metadata` | DeepGEMM 的 paged MQA 调度元数据 |
| `logits` | 每个 query 对历史位置的打分，`[num_rows, max_seq_len]` |
| `topk_indices_buffer` | 最终 top-k 的临时输出 buffer |

### 2.5 驱逐：`BlockPool.free_blocks`

**源码文件**：`vllm/v1/core/block_pool.py`  
**所在类/函数**：`BlockPool.free_blocks`  
**调用者**：KV cache manager 在请求结束或 block 被驱逐时调用。

```python
    def free_blocks(self, ordered_blocks: Iterable[KVCacheBlock]) -> None:
        """Free a list of blocks. The blocks should be ordered by their
        eviction priority, where the first block will be evicted first.

        Args:
            ordered_blocks: A list of blocks to free ordered by their eviction
                priority.

        """
        # Identify blocks with hash (LRU cache) and without it (never match APC)
        blocks_to_evict_last = []
        blocks_to_evict_first = []
        other_pools: dict[BlockPool, list[KVCacheBlock]] = {}
        for block in ordered_blocks:
            if block.is_null:  # never pooled; its ref_cnt is not maintained
                continue
            if block.pool is not None and block.pool is not self:
                other_pools.setdefault(block.pool, []).append(block)
                continue
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if block.block_hash is None or not self.enable_caching:
                    # LIFO reuse of non-cached blocks for better GPU locality.
                    blocks_to_evict_first.append(block)
                else:
                    # FIFO reuse of cached blocks for LRU eviction behavior.
                    blocks_to_evict_last.append(block)

        # Blocks to reuse first are prepended to the front of the free queue.
        self.free_block_queue.prepend_n(blocks_to_evict_first)
        # Blocks to reuse last are appended to the end of the free queue.
        self.free_block_queue.append_n(blocks_to_evict_last)
        for pool, blocks in other_pools.items():
            pool.free_blocks(blocks)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `ordered_blocks` | 按驱逐优先级排序的 block 列表 |
| `block.ref_cnt` | block reference count，引用计数 |
| `block.block_hash` | prefix cache 的 block hash |
| `blocks_to_evict_first` | 先复用的非缓存 block，LIFO |
| `blocks_to_evict_last` | 后复用的缓存 block，LRU |
| `other_pools` | block 所属的其他 pool，例如分层 cache |

这段代码没有 index 分支：index page 不是单独 free，而是随它所属 KV cache group 的 block 一起释放。

---

## 3. DeepSeek V4：`DeepseekV4IndexerCache`

### 3.1 cache 类与 spec

**源码文件**：`vllm/models/deepseek_v4/attention.py`  
**所在类**：`DeepseekV4IndexerCache`  
**调用者**：`DeepseekV4Indexer.__init__`

```python
class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # tokens_per_state=1 for V3.2, >1 for DeepseekV4; same cache layout.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype == "fp8_ds_mla"
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            max_tp_shards=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            alignment=576 if uses_fp8_ds_mla_layout else 512,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return DeepseekV4IndexerBackend
```

和 V3.2 的差异：

| 项目 | V3.2 | V4 |
|---|---|---|
| `tokens_per_state` | 1 | `compress_ratio` |
| `cache_role` | `INDEXER` | 未显式设置，默认 `SPARSE` |
| `alignment` | 默认 | 576 / 512 |
| index K 来源 | indexer 自己的 `wk` | compressor |

### 3.2 `DeepseekV4Indexer` 创建 cache

**源码文件**：`vllm/models/deepseek_v4/attention.py`  
**所在类**：`DeepseekV4Indexer.__init__`

```python
        if self.use_fp4_kv:
            # MXFP4 stores two values per byte plus one UE8M0 byte per 32 values.
            # head_dim bytes = 64 packed values + 4 UE8M0 scales = 68.
            k_cache_head_dim = self.head_dim // 2 + self.head_dim // MXFP4_BLOCK_SIZE
        else:
            # NOTE(yifan): FP8 indexer cache uses the same layout as V3.2:
            # head_dim bytes = 128 fp8 + 4 fp32 scale = 132.
            k_cache_head_dim = (
                self.head_dim + self.head_dim // self.quant_block_size * 4
            )
        self.k_cache = self._get_cache_cls()(
            head_dim=k_cache_head_dim,
            dtype=torch.uint8,
            prefix=f"{prefix}.k_cache",
            cache_config=cache_config,
            compress_ratio=self.compress_ratio,
        )
        self.compressor = self._get_compressor_cls()(
            vllm_config=vllm_config,
            compress_ratio=self.compress_ratio,
            hidden_size=hidden_size,
            head_dim=self.head_dim,
            rotate=True,
            prefix=f"{prefix}.compressor",
            k_cache_prefix=self.k_cache.prefix,
            use_fp4_cache=self.use_fp4_kv,
        )
        self.indexer_op = self._get_attn_cls()(
            self.k_cache,
            self.quant_block_size,
            self.scale_fmt,
            self.topk_tokens,
            self.head_dim,
            self.max_model_len,
            self.max_total_seq_len,
            self.topk_indices_buffer,
            skip_k_cache_insert=True,
            use_fp4_cache=self.use_fp4_kv,
            compress_ratio=self.compress_ratio,
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `use_fp4_kv` | 是否使用 FP4 index K |
| `k_cache_head_dim` | index K cache 的每 state 字节宽度：68 / 132 |
| `self.k_cache` | `DeepseekV4IndexerCache` |
| `self.compressor` | 负责写 index K 的 compressor |
| `self.indexer_op` | `SparseAttnIndexer`，只读 cache |
| `compress_ratio` | 一个 index state 覆盖多少 token |

为什么 `skip_k_cache_insert=True`：

```text
V4 的 index K 由 compressor 写
indexer_op 不再自己执行 indexer_k_quant_and_cache
```

### 3.3 写入：compressor 写 index K

**源码文件**：`vllm/models/deepseek_v4/attention.py`  
**所在类/函数**：`DeepseekV4Indexer.forward`

```python
        if not skip_compressor:
            # compressor returns None and writes K to the indexer KV cache; the
            # join orders that write before indexer_op (skip_k_cache_insert=True).
            (q_quant, weights), _ = maybe_execute_in_parallel(
                wq_b_and_q_quant,
                lambda: compressor(compressed_kv_score, positions, rotary_emb),
                self.ln_events[0],
                self.ln_events[1],
                self.aux_stream,
            )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `skip_compressor` | 是否跳过 compressor 写 K |
| `q_quant` | 量化后的 index query |
| `weights` | index head weights |
| `compressed_kv_score` | compressor 的 KV score |
| `positions` | token positions |
| `rotary_emb` | RoPE 模块 |

### 3.4 读取：FP8/FP4 paged MQA

**源码文件**：`vllm/models/deepseek_v4/attention.py`  
**所在类**：`DeepseekV4Indexer`；实际读取在 `SparseAttnIndexer` / `DeepseekV4IndexerBackend` 的 indexer op 中完成。

关键维度来自 indexer cache 的 spec：

```text
FP8: head_size = 132
FP4: head_size = 68
tokens_per_state = compress_ratio
```

读取时使用：

```text
indexer_kv_cache.view(num_pages, 64, 1, head_dim_with_sf)
  -> fp8_fp4_paged_mqa_logits
  -> top-k
```

其中 `head_dim_with_sf = 68 if use_fp4_kv else 132`，page 内的 token 单位是 compressed state，不是原始 token。

### 3.5 V4 的 `IndexCache`

V4 还有 `use_index_cache` / `index_topk_freq` / `index_topk_pattern`。它复用 top-k 结果，但不复用 index K cache：

```text
skip_topk=True
  -> 当前层不跑 indexer
  -> 使用 prev_topk_indices
  -> index K cache 仍然已分配、已写入
```

因此：

- `IndexCache` 不改变 `DeepseekV4IndexerCache` 的分配；
- 它只减少某些层的 top-k 计算；
- 被跳过的层仍然有 index K page。

---

## 4. DeepSeek V4.1：共享 index K cache + candidate block

### 4.1 V4.1 的 cache 类

**源码文件**：`vllm/models/deepseek_v41/attention.py`  
**所在类**：`DeepseekV4IndexerCache`

```python
class DeepseekV4IndexerCache(torch.nn.Module, AttentionLayerBase):
    def __init__(
        self,
        head_dim: int,
        dtype: torch.dtype,
        prefix: str,
        cache_config: CacheConfig,
        compress_ratio: int = 1,
    ):
        super().__init__()
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.prefix = prefix
        self.cache_config = cache_config
        self.dtype = dtype
        self.compress_ratio = compress_ratio
        vllm_config = get_current_vllm_config()
        self.sparse_logits = vllm_config.attention_config.indexer_sparse_logits
        # aiter's paged MXFP4 kernels write this cache and read it in place.
        self.rocm_mxfp4 = current_platform.is_rocm() and dsa_indexer_uses_fp4(
            vllm_config
        )
        compilation_config = vllm_config.compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # head_dim already carries the fp8 scale padding
        # tokens_per_state=1 for V3.2, >1 for DeepseekV4; same cache layout.
        # nvfp4_ds_mla is packed too: its compressed record is NVFP4 but the
        # sliding-window record stays the V4.1 MXFP8 one, so the indexer page
        # takes the same alignment either way.
        uses_fp8_ds_mla_layout = vllm_config.cache_config.cache_dtype in (
            "fp8_ds_mla",
            "nvfp4_ds_mla",
        )
        page_alignment = (
            576 if uses_fp8_ds_mla_layout and not _use_v41_mxfp8_kv_record() else 512
        )
        return MLAAttentionSpec(
            block_size=self.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_dim,
            dtype=self.dtype,
            tokens_per_state=self.compress_ratio,
            # The indexer page shares a physical block with the main KV cache
            # (the block stride is the sum of every page in the group), so it
            # has to carry the same padding: the packed KV pages' alignment,
            # else 512B for FlashInfer sparse (#44577).
            alignment=page_alignment,
            # DeepGEMM's paged sparse logits need a 256B-aligned block stride;
            # V4 records keep their TMA stride (V4.1's is on the SWA spec).
            block_stride_alignment=(
                math.lcm(
                    256,
                    1
                    if uses_fp8_ds_mla_layout and _use_v41_mxfp8_kv_record()
                    else page_alignment,
                )
                if self.sparse_logits and not self.rocm_mxfp4
                else None
            ),
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        if self.rocm_mxfp4:
            from vllm.v1.attention.backends.mla.rocm_paged_mxfp4_indexer import (
                DeepseekV41RocmMxfp4IndexerBackend,
            )

            return DeepseekV41RocmMxfp4IndexerBackend
        if self.sparse_logits:
            return DeepseekV41SparseIndexerBackend
        return DeepseekV41IndexerBackend
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `sparse_logits` | 是否启用 sparse logits |
| `rocm_mxfp4` | ROCm MXFP4 index cache 路径 |
| `page_alignment` | index page 对齐 |
| `block_stride_alignment` | block stride 对齐 |
| `tokens_per_state` | 一个 index state 覆盖的 token 数 |

V4.1 没有 `DeepseekV41IndexerCache` 这个新类名；它是在 V4.1 模块中重新定义的同名类。

### 4.2 只有 source 层拥有 index K cache

**源码文件**：`vllm/models/deepseek_v41/attention.py`  
**所在类**：`DeepseekV4Attention.__init__`

```python
        self.indexer = None
        if self.is_index_source:
            index_k_cache: DeepseekV4IndexerCache | None
            # Index K cache: owned only by kv-source layers (their indexer has
            # wk/k_norm); non-owning index sources share the K cache of the
            # latest kv source below them (its indexer produces the keys).
            if self.is_kv_source:
                index_k_cache = DeepseekV4IndexerCache(
                    head_dim=_indexer_k_cache_head_dim(
                        config.index_head_dim, dsa_indexer_uses_fp4(vllm_config)
                    ),
                    dtype=torch.uint8,
                    prefix=f"{prefix}.indexer.k_cache",
                    cache_config=cache_config,
                    compress_ratio=self.compress_ratio,
                )
            else:
                assert self.kv_source_layer_id is not None
                k_cache_prefix = (
                    f"{_replace_layer_index(prefix, self.kv_source_layer_id)}"
                    ".indexer.k_cache"
                )
                index_k_cache = self._static_forward_context.get(k_cache_prefix)
                if index_k_cache is None:
                    raise NotImplementedError(
                        f"Indexer K cache source {k_cache_prefix} not found on "
                        "this rank; PP splits inside a v4.1 kv-sharing group "
                        "are not supported."
                    )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `is_index_source` | 当前层是否是 index source |
| `is_kv_source` | 当前层是否是 KV source |
| `kv_source_layer_id` | 最近的前驱 KV source layer id |
| `k_cache_prefix` | 共享 cache 在 static forward context 中的名字 |

调用关系：

```text
DeepseekV4Attention.__init__
  -> 只有 is_index_source 才创建 indexer
  -> 只有 is_kv_source 才创建 DeepseekV4IndexerCache
  -> 非 source 的 indexer 从 static_forward_context 取 source 的 cache 对象
```

### 4.3 V4.1 的 K 来源：compressor latent

**源码文件**：`vllm/models/deepseek_v41/attention.py`  
**所在类**：`DeepseekV4Indexer`  
**所在函数**：`_produce_k`

```python
    def _produce_k(
        self,
        latent: torch.Tensor | None,
        positions: torch.Tensor,
        rotary_emb: nn.Module,
    ) -> None:
        """Turn the compressor's pre-RoPE latent into paged index keys.

        ``k_norm(wk(latent))`` at group-boundary tokens, RoPE'd at the group
        position and MXFP4/FP8-quantized into the indexer K cache.
        """
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict) or latent is None:
            # Profile run: the indexer K cache is not bound yet and the
            # compressor skipped its latent output.
            return
        assert self.owns_k
        indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
        # ReplicatedLinear returns (output, bias); bias is None. Rows at
        # non-boundary tokens hold garbage latent and are skipped by the
        # store kernel.
        k_pre, _ = self.wk(latent)
        indexer_k_norm_rope_store(
            k_pre,
            positions,
            rotary_emb.cos_sin_cache,
            self.k_norm.weight,
            self.k_norm.variance_epsilon,
            self.k_cache.kv_cache,
            indexer_metadata.slot_mapping,
            self.compress_ratio,
            self.use_fp4_kv,
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `latent` | kv-source compressor 的 pre-RoPE latent |
| `wk` | latent → index head 的投影 |
| `k_norm` | index K 的 RMSNorm |
| `indexer_k_norm_rope_store` | RoPE + 量化 + 写入 paged K cache 的 kernel |
| `slot_mapping` | compressed state 的 slot |
| `compress_ratio` | 1 / 2 / 4 / 128 |

### 4.4 candidate block：真正的 vLLM 版本

**源码文件**：`vllm/model_executor/kernels/attention/dsa/candidate_blocks.py`  
**所在函数**：`select_candidate_blocks`  
**调用者**：`vllm/model_executor/layers/sparse_attn_indexer.py` 中的 `_select_candidate_blocks`。

```python
def select_candidate_blocks(
    logits: torch.Tensor,
    row_ks: torch.Tensor | None,
    row_ke: torch.Tensor,
    topk_blocks: int,
    block_size: int,
    out: torch.Tensor,
    row_repeat: int = 1,
    score_dtype: torch.dtype | None = None,
) -> None:
    """Select local block IDs by maximum score, pinning each row's newest block.

    Row bounds are in packed column space; absent starts mean zero.
    Decode rows share bounds in groups of ``row_repeat``. Output is -1 padded.
    """
    assert logits.is_cuda
    rows, width = logits.shape
    if not rows:
        return
    if not width:
        out.fill_(-1)
        return
    nblocks = triton.cdiv(width, block_size)
    score_dtype = score_dtype or logits.dtype
    # Logits are max_model_len wide (CUDA-graph static) but rows are shorter:
    # DeepSelect stops at each row's valid block count. Order does not matter.
    use_deep_select = has_deep_select() and nblocks >= topk_blocks
    if use_deep_select:
        in_align, out_align = get_deep_select_stride_requirement()
        padded = round_up(nblocks, in_align // score_dtype.itemsize)
        scores = logits.new_empty((rows, padded), dtype=score_dtype)[:, :nblocks]
        use_deep_select = (
            out.stride(1) == 1
            and out.stride(0) * out.element_size() % out_align == 0
            and is_deep_select_supported(scores, topk_blocks)
        )
    else:
        scores = logits.new_empty((rows, nblocks), dtype=score_dtype)
    valid_blocks = torch.empty(rows, dtype=torch.int32, device=logits.device)
    _block_scores_kernel[(rows, 32)](
        logits,
        row_ks,
        row_ke,
        scores,
        valid_blocks,
        *logits.stride(),
        row_ks.stride(0) if row_ks is not None else 0,
        row_ke.stride(0),
        scores.stride(0),
        width,
        nblocks,
        block_size,
        row_ks is not None,
        row_repeat,
        TILE=128,
        FULL_WIDTH=not use_deep_select,
    )
    if use_deep_select:
        deep_select_topk(scores, topk_blocks, end=valid_blocks, output_idx=out)
        return
    # Keep the existing top-k tie behavior.
    top = scores.topk(min(topk_blocks, nblocks), dim=-1)
    _store_candidates_kernel[(rows, triton.cdiv(topk_blocks, 256))](
        top.values,
        top.indices,
        out,
        *out.stride(),
        top.values.shape[1],
        topk_blocks,
        256,
    )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `logits` | indexer 的 MQA logits，`[rows, width]` |
| `row_ks` | row key start，packed 列空间的起点 |
| `row_ke` | row key end，packed 列空间的终点 |
| `topk_blocks` | 每个 query 选多少个 block |
| `block_size` | 每个 block 覆盖多少 compressed positions |
| `out` | 输出的 block ids，`[rows, topk_blocks]`，`-1` 填充 |
| `row_repeat` | decode 的若干行共享同一组边界 |
| `score_dtype` | block score 的数据类型 |
| `nblocks` | block 数量 |
| `valid_blocks` | 每行有效 block 数 |
| `deep_select_topk` | DeepSelect 的 top-k kernel |

调用链：

```text
SparseAttnIndexer.forward_cuda
  -> sparse_attn_indexer
  -> if candidate_blocks is not None
       candidate_write=True  -> _select_candidate_blocks(...)
       candidate_write=False -> _apply_candidate_mask(...)
```

`sparse_attn_indexer` 中的调用：

**源码文件**：`vllm/model_executor/layers/sparse_attn_indexer.py`  
**所在函数**：`sparse_attn_indexer`

```python
        if candidate_blocks is not None:
            # Two-level selection (v4.1) on the decode logits; columns are
            # request-local compressed positions. seq_lens is (B, next_n)
            # for native spec decode (per-row effective lens) and (B, 1)
            # otherwise.
            vis = seq_lens.reshape(-1)
            row_repeat = next_n if vis.numel() != num_rows else 1
            vis = vis[:num_rows]
            decode_candidates = candidate_blocks[:num_rows]
            if candidate_write:
                _select_candidate_blocks(
                    logits,
                    None,
                    vis,
                    decode_candidates.shape[1],
                    candidate_block_size,
                    decode_candidates,
                    row_repeat,
                    score_dtype=(
                        torch.float32
                        if topk_backend in ("auto", "persistent")
                        else None
                    ),
                )
            else:
                _apply_candidate_mask(
                    logits,
                    None,
                    vis,
                    decode_candidates,
                    candidate_block_size,
                    row_repeat,
                )
```

这里 `candidate_write=True` 是 candidate source，负责选候选 block；`False` 是 consumer，只把非候选位置的 logits 置为 `-inf`。

### 4.5 V4.1 的 source 共享对驱逐的影响

```text
kv_source S 拥有 index_k_cache
consumer L1 / L2 通过 static_forward_context 复用 S 的 cache
```

所以：

- free S 的 block，会让 L1/L2 的 index K 一起不可达；
- consumer 没有独立 free；
- index K cache 不是每层一份，而是 source 一份。

---

## 5. MiniMax M3 / MSA：`MiniMaxM3IndexerCache`

### 5.1 cache 类与 spec

**源码文件**：`vllm/models/minimax_m3/common/indexer.py`  
**所在类**：`MiniMaxM3IndexerCache`  
**调用者**：`MiniMaxM3IndexerImpl.__init__`

```python
class MiniMaxM3IndexerCache(nn.Module, AttentionLayerBase):
    """Side KV cache for the indexer's per-token index keys (key-only).

    Registers itself in the static forward context so the KV-cache manager
    allocates it (like ``DeepseekV32IndexerCache``).
    """

    def __init__(
        self,
        head_dim: int,
        prefix: str,
        cache_config: CacheConfig | None = None,
        indexer_kv_dtype: IndexerKVDType | str = "bf16",
        backend_cls: type[AttentionBackend] = MiniMaxM3IndexerBackend,
    ) -> None:
        super().__init__()
        # Per-token storage width in elements of ``dtype``.
        self.head_size = head_dim
        if indexer_kv_dtype in ("fp8", "fp8_e4m3"):
            cache_dtype = query_dtype = torch.float8_e4m3fn
        elif indexer_kv_dtype == "bf16":
            cache_dtype = query_dtype = torch.bfloat16
        elif indexer_kv_dtype == "nvfp4":
            # Packed NVFP4 pages scored by the MSA Q8KV4 indexer with e4m3 Q.
            cache_dtype, query_dtype = torch.uint8, torch.float8_e4m3fn
            self.head_size = head_dim // 2 + head_dim // 16
        else:
            raise NotImplementedError(
                f"indexer_kv_dtype={indexer_kv_dtype!r} is not supported by the "
                "MiniMax M3 indexer cache (only 'bf16', 'fp8'/'fp8_e4m3' or "
                "'nvfp4')."
            )
        self.kv_cache = torch.tensor([])
        self.head_dim = head_dim
        self.indexer_kv_dtype = indexer_kv_dtype
        # Side-cache storage dtype: bf16, e4m3 for the fp8 score path, or uint8
        # for packed NVFP4.
        self.dtype = cache_dtype
        # Index-query dtype the fused qk-norm/rope kernel emits.
        self.query_dtype = query_dtype
        self.prefix = prefix
        self.cache_config = cache_config
        # Impl-chosen backend -> each impl gets its own builder (get_attn_backend).
        self.backend_cls = backend_cls
        compilation_config = get_current_vllm_config().compilation_config
        if prefix in compilation_config.static_forward_context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        compilation_config.static_forward_context[prefix] = self

    def bind_kv_cache(self, kv_cache: torch.Tensor) -> None:
        # [B, H=1, N, C] -> [B, N, C]
        self.kv_cache = kv_cache.squeeze(1)

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # Key-only: MLAAttentionSpec budgets one vector/token (not 2x for K+V).
        return MLAAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=1,
            head_size=self.head_size,
            dtype=self.dtype,
            cache_role=SparseCacheRole.INDEXER,
        )

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.backend_cls
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `head_dim` | index head dimension，通常 128 |
| `indexer_kv_dtype` | index cache 的数据类型：bf16/fp8/nvfp4 |
| `cache_dtype` | cache 存储 dtype |
| `query_dtype` | index query dtype |
| `head_size` | 一个 token 的 index K 存储宽度 |
| `kv_cache` | 绑定后的 index cache tensor |
| `cache_role=INDEXER` | 告诉 manager 这是 index side cache |

### 5.2 写入：fused QK-norm + RoPE + KV insert

**源码文件**：`vllm/models/minimax_m3/nvidia/model.py`  
**所在函数**：`MiniMaxM3SparseAttention.forward`  
**调用者**：attention 前向，在主 KV 写入的同一阶段调用。

```python
            ops.fused_minimax_m3_qknorm_rope_kv_insert(
                qkv,
                self.q_norm.weight,
                self.k_norm.weight,
                cos_sin_cache,
                positions,
                self.num_heads,
                self.num_kv_heads,
                rotary_dim,
                eps,
                self.index_q_norm.weight,
                self.index_k_norm.weight,
                self.num_idx_heads,
                main_slot_mapping,
                index_slot_mapping,
                self.kv_cache,
                self.indexer.index_cache.kv_cache,
                self.kv_cache.size(2),  # paged-cache block size
                None if q_is_fp8 else q,
                index_q,
                self.kv_cache_dtype,
                q_fp8_out=q if q_is_fp8 else None,
                q_fp8_scale=self._q_scale_float,
                kv_k_scale=self._k_scale if self.use_nvfp4_kv else None,
                kv_v_scale=self._v_scale if self.use_nvfp4_kv else None,
                write_qkv=not q_is_fp8,
            )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `qkv` | fused q/k/v/index_q/index_k projection 输出 |
| `main_slot_mapping` | 主 GQA K/V 的 slot mapping |
| `index_slot_mapping` | index cache 的 slot mapping |
| `self.kv_cache` | 主 GQA K/V cache |
| `self.indexer.index_cache.kv_cache` | index side cache |
| `index_q` | 量化后的 index query |
| `write_qkv` | 是否同时写 Q/K/V 主 cache |

### 5.3 读取：`MiniMaxM3IndexerImpl.forward`

**源码文件**：`vllm/models/minimax_m3/common/indexer.py`  
**所在类**：`MiniMaxM3IndexerImpl.forward`

```python
    def forward(
        self,
        index_query: torch.Tensor,
        *,
        attention_block_table: torch.Tensor | None = None,
        sparse_block_table_out: torch.Tensor | None = None,
        sparse_context_lens_out: torch.Tensor | None = None,
        block_page_stride: int | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        attn_metadata = get_forward_context().attn_metadata
        if not isinstance(attn_metadata, dict):
            return None, None  # profiling run; caches unbound
        index_md = attn_metadata[self.index_cache.prefix]
        assert isinstance(index_md, MiniMaxM3IndexerMetadata)
        num_tokens = index_md.num_actual_tokens
        nd = index_md.num_decode_tokens
        iq = index_query[:num_tokens].view(
            -1, self.num_index_heads, self.index_head_dim
        )
        kv = self.index_cache.kv_cache

        # Both sides write into the single shared persistent topk_indices_buffer
        # (decode at [:, :nd], prefill at [:, nd:]) and return views into it; the
        # kernels' out= writes out[:, :total_q]. None -> allocate fresh.
        buf = self.topk_indices_buffer
        buf_htk = (
            buf if buf is None or current_platform.is_roc() else buf.transpose(0, 1)
        )
        decode_topk: torch.Tensor | None = None
        prefill_topk: torch.Tensor | None = None
        if index_md.num_decodes > 0:
            d = index_md.decode
            assert d is not None
            fused_sparse_kwargs = {}
            if attention_block_table is not None:
                fused_sparse_kwargs = {
                    "attention_block_table": attention_block_table,
                    "sparse_block_table_out": sparse_block_table_out,
                    "sparse_context_lens_out": sparse_context_lens_out,
                    "block_page_stride": block_page_stride,
                }
            decode_backend_kwargs = {}
            if current_platform.is_roc():
                decode_backend_kwargs["completion_counter"] = (
                    self.topk_completion_counter
                )
            decode_topk = minimax_m3_index_decode(
                iq[:nd],
                kv,
                d.block_table,
                d.seq_lens,
                d.max_seq_len,
                self.topk_blocks,
                self.init_blocks,
                self.local_blocks,
                self.num_kv_heads,
                d.decode_query_len,
                d.max_decode_query_len,
                out=buf_htk,
                **fused_sparse_kwargs,
                **decode_backend_kwargs,
            )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `index_query` | 当前 token 的 index query |
| `index_md` | indexer metadata |
| `iq` | reshape 后的 index query |
| `kv` | `MiniMaxM3IndexerCache.kv_cache` |
| `buf` | 共享的 top-k indices buffer |
| `buf_htk` | head-token-k 布局的 buffer view |
| `decode_topk` | decode 的 top-k 输出 |
| `topk_blocks` | 选多少个 KV block |
| `init_blocks` / `local_blocks` | 固定保留的初始/局部 block |

调用链：

```text
MiniMaxM3SparseAttention.forward
  -> self.indexer(index_q)
  -> MiniMaxM3Indexer.forward
  -> MiniMaxM3IndexerImpl.forward
  -> minimax_m3_index_decode / minimax_m3_index_score / minimax_m3_index_topk
  -> topk_indices_buffer
```

### 5.4 驱逐

MSA 的 index cache 没有独立 free/LRU。主 KV 的 block 被 `BlockPool.free_blocks` 释放后，`MiniMaxM3IndexerCache.kv_cache` 中同一 block 的 index page 也随之不可达。

---

## 6. 量化与数据大小

这一节只回答：

```text
一个 index state 多少字节？
一个 block/page 多少字节？
一个 layer 的 index cache 大概多大？
```

### 6.1 vLLM 的统一公式

源码依据：

- `vllm/v1/kv_cache_interface.py`：`state_content_size_bytes`
- `vllm/v1/kv_cache_interface.py`：`page_size_bytes`
- `vllm/v1/kv_cache_interface.py`：`get_num_kernel_states`

```text
state_content_size_bytes
  = (head_size + head_size_v) * dtype_size

num_states
  = block_size / tokens_per_state

page_size_bytes
  = num_heads * num_states * state_content_size_bytes
```

index cache 通常：

```text
num_heads = 1
head_size_v = 0
```

所以：

```text
page_size_bytes
  = (block_size / tokens_per_state) * head_size
```

`head_size` 已经包含 scale / padding；不要再额外加 K + V。

### 6.2 DeepSeek V3.2

源码依据：

- `DeepseekV32IndexerCache.get_kv_cache_spec`
- `Indexer.__init__`

参数：

```text
head_size = 132 B
dtype     = uint8
block_size = 64
tokens_per_state = 1
num_heads = 1
```

计算：

```text
每 token = 132 B
每 block = 64 * 132 = 8,448 B
```

| 项目 | 数值 |
|---|---:|
| index K | 128 B |
| scale | 4 B |
| 每 token | 132 B |
| block | 64 token |
| 每 block/page | 8,448 B |

### 6.3 DeepSeek V4

源码依据：

- `DeepseekV4IndexerCache.get_kv_cache_spec`
- `DeepseekV4Indexer.__init__`

```text
tokens_per_state = compress_ratio
head_size = 132 (FP8) 或 68 (FP4)
block_size = cache_config.block_size
```

如果 `block_size=64`，不同 ratio 的 page：

| ratio | states / block | FP8 page | FP4 page |
|---:|---:|---:|---:|
| 1 | 64 | 8,448 B | 4,352 B |
| 2 | 32 | 4,224 B | 2,176 B |
| 4 | 16 | 2,112 B | 1,088 B |

注意：

- V4 的 index entry 是 **compressed state**，不是原始 token；
- `ratio=4` 时，一个 page 覆盖的原始 token 数是 `64 * 4 = 256`；
- 但 index page 里只有 16 个 state；
- 实际 block_size 以模型 `cache_config.block_size` 为准，上表用 64 做示例。

### 6.4 DeepSeek V4.1

源码依据：

- `vllm/models/deepseek_v41/attention.py` 的 `DeepseekV4IndexerCache.get_kv_cache_spec`
- 同文件的 `DeepseekV4Indexer._produce_k`

V4.1 的 per-state 大小和 V4 一样：

```text
FP8 132 B / state
FP4  68 B / state
```

主要差异：

- `tokens_per_state = compress_ratio`，V4.1 常见 ratio 1/2；
- `block_stride_alignment` 可能是 576/512 或 256 的 lcm；
- 这会影响 **block stride 对齐**，不一定改变 state 大小；
- source 层共享 index K cache，consumer 不重复分配完整的 per-layer cache。

如果 `block_size=64`：

| ratio | states / block | FP8 page | FP4 page |
|---:|---:|---:|---:|
| 1 | 64 | 8,448 B | 4,352 B |
| 2 | 32 | 4,224 B | 2,176 B |

### 6.5 MiniMax M3 / MSA

源码依据：

- `vllm/models/minimax_m3/common/indexer.py`
- `MiniMaxM3IndexerCache.__init__`
- `MiniMaxM3IndexerCache.get_kv_cache_spec`

参数：

```text
block_size = 128
tokens_per_state = 1
num_heads = 1
```

三种缓存格式：

| dtype | head_size | 每 token | 每 block=128 |
|---|---:|---:|---:|
| bf16 | 128 | 256 B | 32,768 B = 32 KiB |
| fp8 | 128 | 128 B | 16,384 B = 16 KiB |
| nvfp4 | 72 | 72 B | 9,216 B = 9 KiB |

nvfp4 的 72 B：

```text
head_dim = 128
payload = 128 / 2 = 64 B
scale   = 128 / 16 = 8 B
total   = 72 B
```

注意：

- bf16 没有额外 scale；
- fp8 是 1 B/元素；
- nvfp4 的 `head_size` 已经包含 packed payload + scale；
- 实际 page 可能还有对齐 padding，以 `page_size_bytes` 为准。

### 6.6 汇总表

| 模型 | 每 state | block | 每 block/page 示例 |
|---|---:|---:|---:|
| V3.2 | 132 B | 64 | 8,448 B |
| V4 FP8 / ratio 4 | 132 B | 64 | 2,112 B |
| V4 FP4 / ratio 4 | 68 B | 64 | 1,088 B |
| V4.1 FP8 / ratio 2 | 132 B | 64 | 4,224 B |
| V4.1 FP4 / ratio 2 | 68 B | 64 | 2,176 B |
| MSA bf16 | 256 B | 128 | 32,768 B |
| MSA fp8 | 128 B | 128 | 16,384 B |
| MSA nvfp4 | 72 B | 128 | 9,216 B |

### 6.7 三个容易搞错的点

1. **per-token 不是 per-entry**  
   V4/V4.1 的 entry 是 compressed state；ratio=4 时一个 entry 覆盖 4 个原始 token。

2. **head_size 不等于 K 的维度**  
   V3.2 的 `head_size=132` 已经包含 128 B K + 4 B scale；不要再乘 2 或额外加 scale。

3. **page_size_bytes 可能被 padding 放大**  
   `MLAAttentionSpec.alignment` / `page_size_padded` 可能让实际物理 page 比公式值大；上表是未额外 padding 的计算值。

---

## 7. 四个实现的对照

| 维度 | V3.2 | V4 | V4.1 | MSA |
|---|---|---|---|---|
| cache 类 | `DeepseekV32IndexerCache` | `DeepseekV4IndexerCache` | `DeepseekV4IndexerCache`（V4.1 模块） | `MiniMaxM3IndexerCache` |
| spec | `MLAAttentionSpec(cache_role=INDEXER)` | `MLAAttentionSpec(tokens_per_state=compress_ratio)` | 同 V4 + `block_stride_alignment` | `MLAAttentionSpec(cache_role=INDEXER)` |
| 每 state 字节 | 132 | 132 / 68 | 132 / 68 | 256 / 128 / 72 |
| K 来源 | `Indexer` 的 `wk` | compressor | compressor latent + `wk` | fused qkv 的 index_k |
| 写入 | `indexer_k_quant_and_cache` | compressor | `indexer_k_norm_rope_store` | fused qkv insert |
| 读取 | paged MQA + top-k | paged MQA + top-k | candidate + sparse MQA + top-k | MSA index score/topk |
| 驱逐 | 主 block | 主 block | 主 block + source | 主 block |

---

## 8. 完整生命周期

```text
[声明与分配]
  model 构造 indexer cache 对象
    -> get_kv_cache_spec()
    -> KV cache manager 按 block 分配 layer cache
    -> bind_kv_cache(kv_cache)

[写入]
  V3.2: indexer_k_quant_and_cache(k, kv_cache, slot_mapping)
  V4: compressor -> k_cache
  V4.1: indexer_k_norm_rope_store -> k_cache
  MSA: fused_minimax_m3_qknorm_rope_kv_insert -> index_cache

[读取与 top-k]
  page_table + seq_lens
    -> indexer Q × index K
    -> paged MQA logits
    -> candidate block (V4.1)
    -> top-k indices

[主 attention]
  top-k indices
    -> 选中的主 KV pages
    -> sparse MLA / MSA attention

[驱逐]
  BlockPool.free_blocks
    -> ref_cnt 到 0
    -> block 回 free queue
    -> index page 所属 group 的 block 不可达
    -> 下次分配覆盖旧 index 数据
```
