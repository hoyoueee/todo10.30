# SGLang Indexer Pool 源码生命周期

> 仓库：`sgl-project/sglang`  
> 代码基线：`main` 分支，commit `b8d86350d24a5689283810c34889f4defc6eb3ef`（2026-10-31）。  
> 每个代码块上方都标出原文件；代码块内只放该文件中的实际代码。
>
> 阅读顺序：
> 1. DSA / V3.2 的 `IndexKeyCache`
> 2. DSA 的 `IndexerKPool`
> 3. MSA 的 `index_k_pool`
> 4. V4 的 `DeepSeekV4IndexerPool`
> 5. V4.1 的 source 共享与 candidate block

---

## 0. 先建立整体模型

SGLang 的 indexer pool 不是主 KV pool，而是按 page 组织的 index-K side buffer：

```text
主 KV pool：page id -> 主 attention 的 K/V
indexer pool：同一个 page id（或同一个 compressed state）-> index K
RadixCache：只管理 page id 和 token prefix
```

生命周期：

```text
启动时分配固定 page slab
  -> 前向写入 index K
  -> indexer 读取 K + scale
  -> 计算 paged MQA logits
  -> top-k
  -> 主 attention 读取选中的主 KV page
  -> RadixCache 释放 page id，index page 一起不可达
```

---

## 1. DSA / V3.2：`IndexKeyCache`

### 1.1 这个类负责什么

`IndexKeyCache` 是 DSA 的 **index-K side buffer**：

- 每个 layer 一个固定大小 tensor；
- 启动时按 page 分配，不随请求增长；
- page id 复用主 KV 的 allocator；
- 没有独立 prefix tree、free 或 LRU；
- index K 和 scale 打包在同一个 `uint8` buffer。

### 1.2 构造：一次性分配每层 page slab

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.__init__`  
**调用者**：`DSATokenToKVPool.__init__` 在 pool 初始化时构造 `IndexKeyCache`。

```python
class IndexKeyCache:
    def __init__(self, pool: DSATokenToKVPool, index_buf_size: int):
        self.pool = pool
        num_pages = (index_buf_size + pool.page_size + 1) // pool.page_size
        with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(pool.custom_mem_pool)
                if pool.custom_mem_pool
                else nullcontext()
            ):
                self.buffer = [
                    torch.zeros(
                        self._buffer_shape(self._layer_num_pages(i, num_pages)),
                        dtype=pool.index_k_with_scale_buffer_dtype,
                        device=pool.device,
                    )
                    for i in range(pool.layer_num)
                ]
```

变量全称：

| 变量 | 全称 / 含义 |
|---|---|
| `pool` | `DSATokenToKVPool`，提供 page 大小、层数、dtype 等 |
| `index_buf_size` | index buffer size，按 token 计的最大容量 |
| `num_pages` | 向上取整的 page 数，额外加一页作为余量 |
| `self.buffer` | 每层一个 `[num_pages, page_bytes]` tensor |
| `GPU_MEMORY_TYPE_KV_CACHE` | memory saver 的 KV cache 区域标记 |
| `custom_mem_pool` | 可选的自定义 CUDA memory pool |

为什么调用：

- `DSATokenToKVPool` 在初始化时需要为所有 DSA layer 预分配 index K 空间；
- 不能等到请求来了再 malloc，否则会破坏 CUDA graph 和显存规划。

### 1.3 每层 buffer 的形状

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache._buffer_shape`

```python
    def _buffer_shape(self, num_pages: int) -> tuple[int, int]:
        pool = self.pool
        return (
            num_pages,
            pool.page_size
            // pool.index_kpool
            * (pool.index_head_dim + pool.index_head_dim // pool.quant_block_size * 4),
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `page_size` | 主 KV page 覆盖的 token 数，DSA 通常 64 |
| `index_kpool` | index key pool 因子；1 表示每 token 一个 entry |
| `index_head_dim` | index head dimension，通常 128 |
| `quant_block_size` | quantization block size，通常 128 |
| `num_pages` | 该层的 page 数 |

标准 DSA：

```text
128 + 128/128*4 = 132 bytes / entry
64 * 132 = 8448 bytes / page
```

### 1.4 `_kernel_view`：给 kernel 的 page 视图

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache._kernel_view`

```python
    def _kernel_view(self, buffer: torch.Tensor) -> torch.Tensor:
        pool = self.pool
        kernel_pages_per_row = (
            pool.page_size // pool.index_kpool // pool.index_page_size
        )
        return buffer.view(-1, buffer.shape[1] // kernel_pages_per_row)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `buffer` | 某层的原始 `[num_pages, page_bytes]` tensor |
| `kernel_pages_per_row` | 一行里包含多少个 `index_page_size` 页 |
| `index_page_size` | index page 的 slot 粒度 |

标准 DSA：`64 // 1 // 64 = 1`，所以视图还是每行一个 page。

### 1.5 读取前等待层传输完成

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.get_local_buffer`

```python
    def get_local_buffer(self, layer_id: int) -> torch.Tensor:
        if self.pool.layer_transfer_counter is not None:
            self.pool.layer_transfer_counter.wait_until(
                layer_id - self.pool.start_layer
            )
        return self._kernel_view(self.buffer[layer_id - self.pool.start_layer])
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `layer_id` | 全局 layer id |
| `start_layer` | 当前 PP stage 的起始 layer |
| `layer_transfer_counter` | 层间传输/异步 load 的同步计数器 |
| `get_local_buffer` | 返回该层 index page 的 kernel view |

### 1.6 写入：`store_quantized`

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.store_quantized`  
**调用者**：`DSATokenToKVPool.set_index_k_scale_buffer` / `Indexer` 前向路径。

```python
    def store_quantized(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        buf = self._kernel_view(self.buffer[layer_id - self.pool.start_layer])
        dsa_index_k_cache.SetKAndS.execute(
            pool=self.pool,
            buf=buf,
            loc=loc,
            index_k=index_k,
            index_k_scale=index_k_scale,
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `layer_id` | 当前 layer |
| `loc` | location，主 KV 的 token slot |
| `index_k` | 当前 token 的 index key |
| `index_k_scale` | index key 的 scale |
| `buf` | 该层 index page 的 kernel view |
| `SetKAndS` | 写 K 和 scale 的 kernel |

为什么调用：

- `loc` 来自主 KV 的 `out_cache_loc`；
- index K 和主 KV 共享同一个 slot；
- 不需要单独分配 page id。

### 1.7 读取：`get_k_and_scale`

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.get_k_and_scale`  
**调用者**：`IndexerKPool` / DSA indexer 的 paged MQA 路径。

```python
    def get_k_and_scale(
        self,
        layer_id: int,
        seq_len_tensor: torch.Tensor,
        page_indices: torch.Tensor,
        seq_len_sum: int,
        max_seq_len: int,
    ):
        buf = self.get_buffer(layer_id)
        return dsa_index_k_cache.GetKAndS.execute(
            self.pool,
            buf,
            page_indices=page_indices,
            seq_len_tensor=seq_len_tensor,
            seq_len_sum=seq_len_sum,
            max_seq_len=max_seq_len,
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `seq_len_tensor` | 每个请求的序列长度 |
| `page_indices` | 主 KV 的 page id 列表 |
| `seq_len_sum` | 所有请求长度之和 |
| `max_seq_len` | 最长序列长度 |
| `GetKAndS` | 从 paged buffer 取 K 和 scale 的 kernel |

### 1.8 搬移：`move`

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.move`  
**调用者**：主 KV page 搬移/compaction 路径。

这个函数只支持 `index_kpool == 1`：

```python
        assert self.pool.index_kpool == 1, (
            "Moving pooled index keys requires regrouping tokens"
        )
        page_size = self.pool.index_page_size
        tgt_loc_flat = tgt_loc.view(-1).long()
        src_loc_flat = src_loc.view(-1).long()
        tgt_page, tgt_offset = tgt_loc_flat // page_size, tgt_loc_flat % page_size
        src_page, src_offset = src_loc_flat // page_size, src_loc_flat % page_size
        key_bytes = page_size * self.pool.index_head_dim
```

为什么只支持 1：

- `index_kpool=1` 时，一个 token 对应一个 index entry；
- `tgt_loc`/`src_loc` 的 token slot 可以直接换算成 page + offset；
- `index_kpool>1` 时，一个 pooled entry 覆盖多个 token，需要重新分组，不能按 page 直接搬。

### 1.9 offload / restore：`cpu_copy` / `load_cpu_copy`

**源码文件**：`python/sglang/srt/mem_cache/index_key_cache.py`  
**所在类**：`IndexKeyCache.cpu_copy`

```python
    def cpu_copy(self, indices):
        page_indices = indices[:: self.pool.page_size] // self.pool.page_size
        torch.cuda.synchronize()
        index_k_cpu = []
        chunk_size = self.pool.cpu_offloading_chunk_size
        page_chunk_size = max(1, chunk_size // self.pool.page_size)
        for layer_id in range(self.pool.layer_num):
            index_k_cpu.append([])
            if self.buffer[layer_id].shape[0] == 0:
                continue
            for i in range(0, len(page_indices), page_chunk_size):
                chunk_page_indices = page_indices[i : i + page_chunk_size]
                idx_cpu = self.buffer[layer_id][chunk_page_indices].to(
                    "cpu", non_blocking=True
                )
                index_k_cpu[-1].append(idx_cpu)
        torch.cuda.synchronize()
        return index_k_cpu
```

`load_cpu_copy` 是反向操作。调用者仍是 cache manager 的 offload/restore 路径；index K 跟随主 KV 一起走，不单独决定驱逐。

### 1.10 驱逐：不在 `IndexKeyCache` 里

`IndexKeyCache` 没有 `free()` 或 `evict()`。驱逐由 RadixCache/allocator 完成：

```text
RadixCache 释放 main page id
  -> IndexKeyCache 中同一 page 不可达
  -> 没有独立 index free
  -> page id 复用时下一次 store_quantized 覆盖
```

---

## 2. DSA：`IndexerKPool` 如何读和选 top-k

### 2.1 读 buffer：`_get_index_k_read_buffer`

**源码文件**：`python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`  
**所在类**：`IndexerKPool._get_index_k_read_buffer`  
**调用者**：`_get_topk_paged`

```python
    @staticmethod
    def _get_index_k_read_buffer(pool, layer_id: int) -> torch.Tensor:
        if hasattr(pool, "get_broadcastable_index_k_with_scale_buffer"):
            return pool.get_broadcastable_index_k_with_scale_buffer(layer_id)
        if hasattr(pool, "_get_broadcastable_index_buffer"):
            return pool._get_broadcastable_index_buffer(layer_id)
        return pool.get_index_k_with_scale_buffer(layer_id=layer_id)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `pool` | `DSATokenToKVPool` |
| `layer_id` | indexer 层 |
| `get_broadcastable...` | 兼容不同后端/page layout 的读视图 |

### 2.2 top-k：`_get_topk_paged`

**源码文件**：`python/sglang/srt/layers/attention/dsa/dsa_indexer_kpool.py`  
**所在类**：`IndexerKPool._get_topk_paged`  
**调用者**：`IndexerKPool` 的 forward / 主 attention indexer。

```python
    def _get_topk_paged(
        self,
        forward_batch: ForwardBatch,
        layer_id: int,
        q_fp8: torch.Tensor,
        weights: torch.Tensor,
        metadata: BaseIndexerMetadata,
    ) -> torch.Tensor:
        if TYPE_CHECKING:
            assert isinstance(get_token_to_kv_pool(), DSATokenToKVPool)

        pool = get_token_to_kv_pool()
        page_size = pool.index_page_size
        # DeepGEMM paged-MQA requires 64-token pages.
        assert page_size == 64, "only support page size 64"

        block_tables = metadata.get_page_table_64()

        kv_cache_fp8 = self._get_index_k_read_buffer(pool, layer_id)

        blocksize = page_size
        assert len(kv_cache_fp8.shape) == 2
        block_kv = 64
        num_heads_kv = 1
        head_dim_with_sf = 132
        kv_cache_fp8 = kv_cache_fp8.view(
            kv_cache_fp8.shape[0], block_kv, num_heads_kv, head_dim_with_sf
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `q_fp8` | 量化后的 index query |
| `weights` | 每个 index head 的权重 |
| `metadata` | indexer metadata |
| `block_tables` | 64-token page table |
| `kv_cache_fp8` | indexer pool 的读视图 |
| `block_kv` | 每个 page 的 64 个 token |
| `num_heads_kv` | 1，单 head index K |
| `head_dim_with_sf` | 132，128 K + 4 scale |

调用链：

```text
IndexerKPool.forward
  -> _get_topk_paged
  -> _get_index_k_read_buffer
  -> paged MQA logits
  -> topk
```

---

## 3. MSA：`MiniMaxSparseKVPool.index_k_pool`

### 3.1 main + index 两个物理 pool

**源码文件**：`python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py`  
**所在类**：`MiniMaxM3Nvfp4KVPool`  
**调用者**：MiniMax-M3 的 sparse attention 初始化。

```python
class MiniMaxM3Nvfp4KVPool(MiniMaxSparseKVPool):
    """MSA-v1 main K/V and single-head index K, all packed as NVFP4."""

    def __init__(self, **kwargs):
        if kwargs["dense_layer_ids"]:
            raise ValueError("M3.1 NVFP4 requires sparse attention in every layer")
        if set(kwargs["disable_value_sparse_layer_ids"]) != set(
            kwargs["sparse_layer_ids"]
        ):
            raise ValueError("M3.1 NVFP4 requires a K-only sparse index")
        super().__init__(
            **kwargs,
            main_pool_cls=MiniMaxM3Nvfp4MainKVPool,
            index_k_pool_cls=MiniMaxM3Nvfp4IndexKPool,
        )
        # The base fast path is a raw byte copy, which cannot serve a quantizing
        # cache; this pool dispatches to its own fused quantize+store instead.
        self.use_minimax_fused_kv_index_store = False
        # The fused kernel reproduces `flashinfer.nvfp4_kv_quantize`, which is
        # the branch `NVFP4KVQuantizeUtil.quantize` takes on SM100/SM103 only.
        # SM120 shares the API but has no bit-exact acceptance yet, and SM90
        # takes a different quantizer entirely.
        self.use_nvfp4_fused_quantize_store = (
            envs.SGLANG_OPT_USE_MINIMAX_FUSED_KV_INDEX_STORE.get()
            and get_platform().is_sm100
            and self.main_pool.head_dim == MiniMaxM3Nvfp4IndexKPool.INDEX_HEAD_DIM
            and self.main_pool.page_size == self.index_k_pool.page_size
            # One kernel argument carries the outer scale for all three
            # streams, so main and index must agree on it.
            and torch.equal(self.main_pool.global_scale, self.index_k_pool.global_scale)
        )
        logger.info(
            "MiniMax M3.1 NVFP4 KV store: %s",
            (
                "fused quantize+store"
                if self.use_nvfp4_fused_quantize_store
                else "per-buffer quantize then scatter"
            ),
        )
```

### 3.2 index page 组件

**源码文件**：`python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py`  
**所在函数**：`_index_page_components`

```python
def _index_page_components(head_dim: int, page_size: int):
    return (
        ("k_buffer", (page_size, head_dim // 2)),
        ("k_scale_buffer", (page_size, head_dim // _SCALE_GROUP_SIZE)),
    )
```

MSA index page：

```text
page_size=128
k_buffer: 128 * 64 = 8192 B
k_scale_buffer: 128 * 8 = 1024 B
row_bytes = 9216 B（按 128 B 对齐后）
```

### 3.3 分配：`_allocate_page_rows`

**源码文件**：`python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py`  
**所在函数**：`_allocate_page_rows`

```python
def _allocate_page_rows(pool: KVCache, components, num_pages: int, device: str):
    offsets, row_bytes = _page_row_layout(components)
    with pool.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
        with (
            torch.cuda.use_mem_pool(pool.custom_mem_pool)
            if pool.enable_custom_mem_pool
            else nullcontext()
        ):
            slab = torch.zeros(
                (pool.layer_num, num_pages, row_bytes),
                dtype=torch.uint8,
                device=device,
            )
    rows = list(slab.unbind(0))
    views = {
        name: [
            row[:, offset : offset + math.prod(shape)].view(num_pages, *shape)
            for row in rows
        ]
        for (name, shape), offset in zip(components, offsets)
    }
    return slab, rows, row_bytes, views
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `components` | page 内的 K / scale 组件 |
| `num_pages` | 固定 page 数 |
| `row_bytes` | 一页字节数，按 128 B 对齐 |
| `slab` | `[layers, pages, row_bytes]` 的 uint8 大块 |
| `rows` | 每层二维 page row |
| `views` | K / scale 的 per-layer 视图 |

### 3.4 读取：模型拿 index page buffer

**源码文件**：`python/sglang/srt/models/minimax_m3.py`  
**所在函数**：`MiniMaxM3SparseAttention.forward`

```python
        kv_pool = self._get_sparse_kv_pool()
        k_cache, v_cache = kv_pool.get_kv_buffer(layer_id)
        idx_k_cache = kv_pool.get_index_k_buffer(layer_id)
        page_size = int(kv_pool.page_size)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `kv_pool` | `MiniMaxSparseKVPool` |
| `k_cache` / `v_cache` | 主 GQA K/V |
| `idx_k_cache` | 该层 index K page |
| `page_size` | 128 |

### 3.5 offload：`get_cpu_copy` / `load_cpu_copy`

**源码文件**：`python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py`  
**所在类**：`MiniMaxM3Nvfp4KVPool`

```python
    def get_cpu_copy(self, indices, mamba_indices=None, req_pool_index=None):
        return {
            "main": self.main_pool.get_cpu_copy(indices),
            "index_k": self.index_k_pool.get_cpu_copy(indices),
        }

    def load_cpu_copy(
        self, kv_cache_cpu, indices, mamba_indices=None, req_pool_index=None
    ):
        self.main_pool.load_cpu_copy(kv_cache_cpu["main"], indices)
        self.index_k_pool.load_cpu_copy(kv_cache_cpu["index_k"], indices)
```

含义：

- main 和 index 一起 offload/restore；
- 没有独立 index free；
- page id 仍由主 RadixCache 管理。

---

## 4. DeepSeek V4：`DeepSeekV4IndexerPool`

### 4.1 按 compression ratio 建 pool

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在类**：`DeepSeekV4IndexerPool`

```python
class DeepSeekV4IndexerPool(KVCache):
    quant_block_size = 128
    index_k_with_scale_buffer_dtype = torch.uint8

    def __init__(
        self,
        size: int,
        page_size: int,
        dtype: torch.dtype,
        index_head_dim: int,
        layer_num: int,
        device: str,
        enable_memory_saver: bool,
        start_layer: Optional[int] = None,
        end_layer: Optional[int] = None,
        use_fp4_indexer: Optional[bool] = None,
        global_page_size: Optional[int] = None,
    ):
        super().__init__(
            size,
            page_size,
            dtype,
            layer_num,
            device,
            enable_memory_saver,
            start_layer,
            end_layer,
        )
        self.index_head_dim = index_head_dim
        self.index_page_size = self.page_size
        self.global_page_size = global_page_size or page_size
        if use_fp4_indexer is None:
            use_fp4_indexer = get_exec().kernel.enable_deepseek_v4_fp4_indexer
        self.use_fp4_indexer = use_fp4_indexer
        self.uses_aiter_fp4_layout = _is_hip and self.use_fp4_indexer
        # Low-ratio pools round to nearest even; c4 keeps threshold rounding.
        self.index_k_rne = False
        # aiter's row-group page-8 layout inside each page (pack_kv_cache): page size and shapes
        # are unchanged but the bytes are not, so PD peers must use the same layout and a
        # persisted cache written under the other one must be cleared before switching
        self.index_k_page8 = False

        self._create_buffer()
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `size` | 该 compression ratio 的 compressed state 容量 |
| `page_size` | index page 的 compressed-slot 粒度 |
| `index_head_dim` | 128 |
| `layer_num` | 该 pool 里的 owning layer 数 |
| `use_fp4_indexer` | 是否 FP4 index K |
| `index_page_size` | index page size |
| `global_page_size` | 模型全局 page size |

### 4.2 每 state 字节数

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在函数**：`get_dsv4_indexer_bytes_per_token`

```python
def get_dsv4_indexer_bytes_per_token(index_head_dim: int, use_fp4_indexer: bool) -> int:
    """Return payload and quant-scale bytes for one compressed indexer token."""
    if use_fp4_indexer:
        return index_head_dim // 2 + index_head_dim // 32
    return index_head_dim + index_head_dim // 128 * 4
```

```text
FP8: 128 + 4 = 132 B / compressed state
FP4:  64 + 4 =  68 B / compressed state
```

### 4.3 分配 page slab

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在类**：`DeepSeekV4IndexerPool._create_buffer`

```python
    def _create_buffer(self):
        page_bytes = self.page_size * self.get_bytes_per_token()
        # Same page count as the KV pool of this ratio: PD registers both by page.
        num_pages = _num_dsv4_physical_kv_pages(
            self.size, self.page_size, self.global_page_size
        )
        with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_KV_CACHE):
            with (
                torch.cuda.use_mem_pool(self.custom_mem_pool)
                if self.custom_mem_pool
                else nullcontext()
            ):
                if self.uses_aiter_fp4_layout:
                    self.index_k_payload_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size, 16),
                            dtype=torch.uint8,
                            device=self.device,
                        ).view(torch.float4_e2m1fn_x2)
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_scale_buffer = [
                        torch.zeros(
                            (num_pages, 1, 4, self.page_size),
                            dtype=torch.uint8,
                            device=self.device,
                        )
                        for _ in range(self.layer_num)
                    ]
                    self.index_k_with_scale_buffer = None
                    return

                self.index_k_with_scale_buffer = [
                    torch.zeros(
                        num_pages,
                        page_bytes,
                        dtype=self.index_k_with_scale_buffer_dtype,
                        device=self.device,
                    )
                    for _ in range(self.layer_num)
                ]
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `page_bytes` | 一个 index page 的字节数 |
| `num_pages` | 与该 ratio 的 compressed KV pool 对齐的 page 数 |
| `index_k_with_scale_buffer` | FP8/FP4 fused per-layer buffer |
| `index_k_payload_buffer` | AITER FP4 的 payload buffer |
| `index_k_scale_buffer` | AITER FP4 的 scale buffer |

### 4.4 layer -> ratio -> pool

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在类**：`DeepSeekV4TokenToKVPool`

```python
    def set_index_k_scale_buffer(
        self,
        layer_id: int,
        loc: torch.Tensor,
        index_k: torch.Tensor,
        index_k_scale: torch.Tensor,
    ) -> None:
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        self._indexer_pool(compress_ratio).set_index_k_scale_buffer(
            compress_layer_id, loc, index_k, index_k_scale
        )

    def get_index_k_with_scale_buffer(self, layer_id: int) -> torch.Tensor:
        self.wait_layer_transfer(layer_id)
        compress_ratio, compress_layer_id, _ = self.layer_mapping[layer_id]
        return self._indexer_pool(compress_ratio).get_index_k_with_scale_buffer(
            compress_layer_id
        )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `layer_mapping` | `layer_id -> (compress_ratio, compress_layer_id, pool)` |
| `compress_ratio` | 该层压缩率 |
| `compress_layer_id` | 在该 ratio pool 里的层下标 |
| `_indexer_pool` | 取对应 ratio 的 `DeepSeekV4IndexerPool` |

### 4.5 C4 indexer 的读取

**源码文件**：`python/sglang/srt/layers/attention/dsv4/indexer.py`  
**所在类**：`C4IndexerBackendMixin.forward_c4_indexer`

```python
            c4_indexer_kv_cache = token_to_kv_pool.get_index_k_with_scale_buffer(
                layer_id=c4_indexer.layer_id,
            )
            assert c4_indexer_kv_cache.dim() == 2
            head_dim_with_sf = 68 if use_fp4_indexer else 132
            c4_indexer_kv_cache = c4_indexer_kv_cache.view(
                c4_indexer_kv_cache.shape[0], 64, 1, head_dim_with_sf
            )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `c4_indexer_kv_cache` | C4 index K page buffer |
| `head_dim_with_sf` | 68 / 132，K + scale 宽度 |
| `64` | C4 index page 的 compressed-slot 粒度 |

### 4.6 驱逐

`DeepSeekV4IndexerPool` 没有独立 free/LRU。主 compressed KV pool 的 page 被 RadixCache 释放后，同 ratio 的 index page 不可达。

---

## 5. DeepSeek V4.1：source 共享与 candidate block

### 5.1 source layer 选择

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在类**：`DeepSeekV4TokenToKVPool.source_layer_of`

```python
    def source_layer_of(self, layer_id: int) -> int:
        """The layer owning this layer's compressed storage: itself for ratios 4/128,
        the nearest preceding kv_source layer for ratios 1/2."""
        ratio = self.compression_ratios[layer_id]
        sources = [l for l in self.sources_by_ratio[ratio] if l <= layer_id]
        assert sources, f"layer {layer_id} (ratio {ratio}) has no kv_source layer"
        return max(sources)
```

### 5.2 layer_mapping

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在类**：`DeepSeekV4TokenToKVPool._init_compressed_layer_mapping`

```python
    def _init_compressed_layer_mapping(self):
        layer_counts = {0: 0, **{ratio: 0 for ratio in self.kv_pools}}
        total_L = len(self.compression_ratios)
        self.layer_mapping: List[Optional[DeepSeekV4LayerItem]] = [None] * total_L

        for idx in range(self._stage_start, self._stage_end):
            ratio = self.compression_ratios[idx]
            if ratio not in layer_counts:
                raise ValueError(f"Unsupported compression ratio: {ratio}")
            if ratio in (1, 2):
                sources = self.sources_by_ratio[ratio]
                compress_layer_id = sources.index(self.source_layer_of(idx))
            else:
                compress_layer_id = layer_counts[ratio]
                layer_counts[ratio] += 1
            self.layer_mapping[idx] = DeepSeekV4LayerItem(
                compress_ratio=ratio,
                compress_layer_id=compress_layer_id,
                compress_kv_pool=self.kv_pools.get(ratio),
            )
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `layer_counts` | 每个 ratio 的 owning layer 计数 |
| `sources_by_ratio` | 每个 ratio 的 source layer 列表 |
| `compress_layer_id` | 在该 ratio pool 里的层下标 |
| `DeepSeekV4LayerItem` | 保存 ratio、pool 内层下标、compressed KV pool |

### 5.3 V4.1 的 index page size

**源码文件**：`python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  
**所在函数**：`dsv41_index_page_size`

```python
def dsv41_index_page_size() -> int:
    from sglang.srt.layers.deep_gemm_wrapper.configurer import (
        DEEPGEMM_PAGED_SPARSE_MQA_LOGITS,
    )
    if DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
        return 128
    return 64
```

```text
V4.1 + DeepGEMM paged sparse: index page = 128 compressed slots
V4.1 其它 backend:            index page = 64
```

### 5.4 V4.1 的 K 来源

**源码文件**：`python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py`  
**所在类**：`DeepseekV41Indexer._produce_k`

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
            return
        assert self.owns_k
        indexer_metadata = cast(Any, attn_metadata[self.k_cache.prefix])
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
| `latent` | compressor 的 pre-RoPE latent |
| `wk` | latent -> index head 的投影 |
| `k_norm` | index K 的 RMSNorm |
| `slot_mapping` | compressed state 的 slot |
| `compress_ratio` | 1 / 2 / 4 / 128 |
| `use_fp4_kv` | 是否 FP4 index K |

### 5.5 candidate block：SGLang main 的当前实现

**源码文件**：`python/sglang/kernels/ops/attention/dsv4/candidate_blocks.py`  
**所在函数**：`select_candidate_block_ids`  
**调用者**：`python/sglang/srt/layers/attention/dsv4/v41_indexer/dense_blocks.py` 的 `DenseBlocksBackend.publish_decode`。

```python
def select_candidate_block_ids(
    logits: torch.Tensor,
    compress_lens: Union[torch.Tensor, int],
    topk_blocks: int,
    block_size: int,
) -> torch.Tensor:
    """Per row the ids of the ``topk_blocks`` blocks with the largest score, the
    block of position ``compress_lens - 1`` always kept: int32
    ``[rows, min(topk_blocks, blocks)]``, unordered, ``-1`` past the finite blocks.
    ``logits`` must be ``-inf`` past each row's causal length."""
    width = logits.size(-1)
    padding = -width % block_size
    scores = F.pad(logits, (0, padding), value=-torch.inf) if padding else logits
    scores = scores.unflatten(-1, (-1, block_size)).amax(dim=-1)
    num_blocks = scores.size(-1)
    last = (compress_lens - 1) // block_size
    scores = scores.masked_fill(
        torch.arange(num_blocks, device=logits.device) == last, torch.inf
    )
    top = scores.topk(min(topk_blocks, num_blocks), dim=-1)
    return top.indices.to(torch.int32).masked_fill_(~(top.values > -torch.inf), -1)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `logits` | indexer 对 compressed positions 的分数 |
| `compress_lens` | 每个请求可见的 compressed position 数 |
| `topk_blocks` | 选多少个候选 block |
| `block_size` | 每个 block 含多少个 compressed position，当前为 8 |
| `scores` | 每个 block 的 max logit |
| `last` | 最新 block 的下标，强制保留 |
| `top.indices` | 选中的 block id；`-1` 表示不足 k 个 |

`DenseBlocksBackend.publish_decode` 的实际调用：

**源码文件**：`python/sglang/srt/layers/attention/dsv4/v41_indexer/dense_blocks.py`  
**所在类/函数**：`DenseBlocksBackend.publish_decode`

```python
        if isinstance(d, PagedDecodeScores):
            assert self.block_size == 8
            blocks = amax_topk_blocks(
                d.scores, d.lens, (d.lens + 7) // 8, self.topk_blocks
            )
            published = BlockIds(
                blocks=blocks,
                decode_mask=candidate_block_mask(blocks, d.lmax, self.block_size),
            )
        else:
            published = BlockIds(
                blocks=select_candidate_block_ids(
                    d.scores,
                    d.lens[:, None],
                    topk_blocks=self.topk_blocks,
                    block_size=self.block_size,
                )
            )
        select_decode(inputs, d, inputs.indexer.ind)
```

变量：

| 变量 | 全称 / 含义 |
|---|---|
| `d.scores` | 当前层的 indexer logits |
| `d.lens` | 每个 request 的 compressed length |
| `self.topk_blocks` | 要发布的候选 block 数 |
| `self.block_size` | 一个 candidate block 的 compressed position 数，当前为 8 |
| `blocks` | 输出的 block id，`[rows, topk_blocks]` |
| `BlockIds` | 发布的候选元数据，供后续 consumer 使用 |

说明：

- 当前 SGLang main 返回的是 **block id tensor**，不是 bool mask；
- 最新 block 永远保留；
- consumer 会在这些 block 内做第二级 sparse top-k；
- 旧 PR 版本里的 `select_candidate_blocks` 已不在当前 main，本文不再把它当作当前实现。

---

## 6. 量化与数据大小

这一节只回答两个问题：

```text
1. 一个 index entry 多少字节？
2. 一个 page / 一层 / 一个 source 占多少？
```

注意区分：

```text
entry：一个 token 或一个 compressed state 的 index K + scale
page：一组 entry 的物理存储单位
layer：每层一个 buffer（V4.1 的 consumer 可能共享 source）
```

### 6.1 DSA / V3.2

源码依据：

- `python/sglang/srt/mem_cache/index_key_cache.py`：`_buffer_shape`
- `python/sglang/srt/mem_cache/index_key_cache.py`：`index_page_size`、`index_kpool`

布局：

```text
entry = index_head_dim + index_head_dim / quant_block_size * 4
      = 128 + 128 / 128 * 4
      = 132 B
```

默认 `page_size=64`、`index_kpool=1`：

```text
entries per page = page_size / index_kpool = 64
page bytes       = 64 * 132 = 8,448 B
```

| 配置 | 每 entry | 每 page entry 数 | 每 page 字节 |
|---|---:|---:|---:|
| `index_kpool=1` | 132 B | 64 | 8,448 B |
| `index_kpool=2` | 132 B | 32 | 4,224 B |
| `index_kpool=4` | 132 B | 16 | 2,112 B |

注意：

- `index_kpool > 1` 是把多个 token 的 index K 合并成一个 pooled entry，**entry 仍是 132 B**，但 entry 数变少；
- `IndexKeyCache.move()` 只支持 `index_kpool == 1`，因为 pooled entry 需要重新分组；
- 每层总大小约为：
  ```text
  layer_bytes = num_pages * page_bytes
  ```

### 6.2 MSA / MiniMax-M3

源码依据：

- `python/sglang/srt/mem_cache/minimax_m3_nvfp4_pool.py`
- `MiniMaxM3Nvfp4IndexKPool.PAYLOAD_BYTES_PER_TOKEN = 64`
- `SCALE_BYTES_PER_TOKEN = 8`
- `MSA_V1_PAGE_SIZE = 128`

三种 dtype：

| dtype | 每 token | page=128 时每页 |
|---|---:|---:|
| bf16 | 128 × 2 = 256 B | 32,768 B = 32 KiB |
| fp8 | 128 × 1 = 128 B | 16,384 B = 16 KiB |
| nvfp4 | 64 B payload + 8 B scale = 72 B | 9,216 B = 9 KiB |

nvfp4 的 72 B 来源：

```text
head_dim = 128
payload = head_dim / 2 = 64 B   # 两个值打包进 1 字节
scale   = head_dim / 16 = 8 B   # 每 16 维一个 scale
total   = 72 B
```

注意：

- MSA 的 index page row 按 128 B 对齐；
- `9,216 B` 本身是 128 的倍数，不再额外补齐；
- bf16 模式没有额外的 per-block scale，只有 index K 本身。

### 6.3 DeepSeek V4

源码依据：

- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `get_dsv4_indexer_bytes_per_token`

```python
if use_fp4_indexer:
    return index_head_dim // 2 + index_head_dim // 32
return index_head_dim + index_head_dim // 128 * 4
```

计算：

```text
FP8: 128 + 128 / 128 * 4 = 132 B / compressed state
FP4: 128 / 2 + 128 / 32 = 64 + 4 = 68 B / compressed state
```

page 大小：

```text
page_bytes = index_page_size * bytes_per_state
```

V4 的 C4 index page：

- `DeepSeekV4IndexerPool` 构造时传入的 `page_size` 是 `page_size // ratio`；
- 所以 C4 的 index page 比 full page 更小；
- 具体页大小取决于模型的 `page_size`，不要脱离模型配置硬算。

C4 常见示例：

| dtype | 每 state | index page=64 时 | index page=128 时 |
|---|---:|---:|---:|
| FP8 | 132 B | 8,448 B | 16,896 B |
| FP4 | 68 B | 4,352 B | 8,704 B |

注意：

- V4 的 entry 是 **compressed state**，不是原始 token；
- `tokens_per_state = compress_ratio`；
- AITER FP4 可能把 payload 和 scale 拆成两个 buffer；
- page 中可能还有对齐 padding，实际分配以 `_create_buffer` 为准。

### 6.4 DeepSeek V4.1

源码依据：

- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`
- `dsv41_index_page_size`

```python
if DEEPGEMM_PAGED_SPARSE_MQA_LOGITS:
    return 128
return 64
```

V4.1 的 index page：

```text
DeepGEMM paged sparse：128 compressed slots / page
其它 backend：       64 compressed slots / page
```

每 state 仍是：

```text
FP8 132 B
FP4  68 B
```

page 大小：

| dtype | page=128 | page=64 |
|---|---:|---:|
| FP8 | 16,896 B | 8,448 B |
| FP4 | 8,704 B | 4,352 B |

V4.1 的 ratio：

```text
ratio=1：1 token -> 1 state
ratio=2：2 token -> 1 state
```

所以同样 page 大小下，ratio=2 覆盖的原始 token 数是 ratio=1 的两倍。

### 6.5 量化格式速查

| 格式 | 元素宽度 | 额外开销 | 说明 |
|---|---:|---|---|
| bf16 | 2 B | 无 | 精度高，MSA 默认 |
| FP8 e4m3 | 1 B | 每 128 维 4 B scale | DSA/V4/V4.1 常用 |
| FP4 e2m1 | 0.5 B | 每 32 维 4 B scale | V4/V4.1，AITER 可能拆 buffer |
| nvfp4 | 0.5 B | 每 16 维 1 B scale | MSA nvfp4，head_dim=128 -> 8 B scale |

### 6.6 快速结论

```text
DSA index entry：132 B
MSA index entry：256 B bf16 / 128 B fp8 / 72 B nvfp4
V4 FP8：132 B/state
V4 FP4：68 B/state
V4.1：同 V4，page 由 backend 决定 128 或 64
```

---

## 7. 代码对照

| 维度 | DSA / V3.2 | MSA | V4 | V4.1 |
|---|---|---|---|---|
| pool 对象 | `IndexKeyCache` | `MiniMaxSparseKVPool.index_k_pool` | `DeepSeekV4IndexerPool` | 同 V4，source 共享 |
| 组织方式 | 每层 side buffer | main + index 两个 pool | 按 compression ratio | 按 ratio + source layer |
| 一个 entry | 1 token | 1 token | 1 compressed state | 1/2/4/128 个 token |
| index page | 64 | 128 | 64 | 128 / 64 |
| FP8 | 132 B | 128 B | 132 B | 132 B |
| FP4 | 无 | 72 B nvfp4 | 68 B | 68 B |
| 驱逐 | 主 page | 主 page | 主 page | 主 page + source |
| 独立 free/LRU | 无 | 无 | 无 | 无 |

---

## 8. 生命周期总结

```text
[启动]
  pool 构造 index cache
    -> 按层/按 ratio 分配固定 page slab
    -> page id 复用主 KV allocator / 主 compressed pool

[写入]
  DSA: store_quantized -> SetKAndS
  MSA: fused QK-norm + RoPE + KV insert
  V4: compressor -> index k_cache
  V4.1: compressor latent -> wk -> k_norm -> RoPE -> quantize -> k_cache

[读取]
  page_indices / page_table
    -> get_k_and_scale / get_index_k_with_scale_buffer
    -> paged MQA logits
    -> top-k
    -> 主 attention

[驱逐]
  RadixCache 释放 main page id
    -> index page 同一 page 不可达
    -> 没有 index 独立 free
    -> page id 复用后下一次写入覆盖
```
