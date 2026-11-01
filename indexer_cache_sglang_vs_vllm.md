# SGLang vs vLLM：Index Cache 横向对比

> 基于两份源码笔记：
> - `indexer_pool_cache_lifecycle.md`（SGLang）
> - `vllm_indexer_cache_lifecycle.md`（vLLM）
>
> 本文只比较核心逻辑、关键变量和模型差异，不重复代码实现。

---

## 0. 一句话结论

两边都把 index K 当作 **side cache**：

```text
index cache：存 index K + scale
主 KV cache：存主 K/V 或 MLA latent
两者共享 page/block 生命周期
```

区别在组织方式：

| | SGLang | vLLM |
|---|---|---|
| 抽象方式 | 显式 pool / buffer | layer cache + `KVCacheSpec` |
| 物理组织 | 独立 pool 或独立 side buffer | 独立 layer cache/page，通常位于同一 block group |
| 逻辑绑定 | page id / compressed state | block id / `tokens_per_state` |
| 前缀与驱逐 | 由 RadixCache 统一管理 | 由 `KVCacheManager` / `BlockPool` 统一管理 |
| index 独立 prefix/evict | 没有 | 没有 |

---

## 1. 核心逻辑对比

| 维度 | SGLang | vLLM |
|---|---|---|
| index cache 身份 | `IndexKeyCache` / `index_k_pool` / `DeepSeekV4IndexerPool` | `DeepseekV32IndexerCache` / `DeepseekV4IndexerCache` / `MiniMaxM3IndexerCache` |
| 扩展点 | 构造 pool / side buffer | 实现 `get_kv_cache_spec()` |
| 物理存储 | 独立 pool / slab | 独立 layer page/region；可能在同一 block group |
| group/component | index 不是 component；只有需要 Unified Radix 的 owner 才使用 `FullComponent` / `SWAComponent` 等；DSA 可走普通 RadixCache | 没有 index 专用 group；使用通用 `KVCacheGroupSpec` / `UniformTypeKVCacheSpecs` |
| 分配单位 | page | block |
| 逻辑地址 | page id | block id |
| prefix 管理 | RadixCache / Unified Radix Cache | `KVCacheManager` / prefix cache |
| 驱逐单位 | 主 page | 主 block |
| index 独立 free/LRU | 无 | 无 |
| index 独立 prefix tree | 无 | 无 |
| 写入方式 | `SetKAndS` / fused insert / compressor | `indexer_k_quant_and_cache` / compressor / `indexer_k_norm_rope_store` / fused insert |
| 读取方式 | paged MQA / MSA index score | paged MQA / sparse MQA |
| top-k 复用 | `skip_topk` / candidate | `IndexCache` / candidate |
| source 共享 | `layer_mapping` / `source_layer_of` | `is_kv_source` / `is_index_source` + static context |

核心共同点：

```text
index 不参与独立的 prefix match
index 不参与独立的 eviction
index 只是 owner page/block 的附带数据
```

---

## 2. 关键变量对应

| SGLang | vLLM | 对应关系 |
|---|---|---|
| `page_size` | `block_size` | 基本对应，都是 owner cache 的 token 粒度 |
| `index_page_size` | `block_size / tokens_per_state` | SGLang 是 index page 的 slot 数；vLLM 是 block 内的 state 数 |
| `index_head_dim` | `head_size` | SGLang 是裸 head dim；vLLM 的 `head_size` 通常已包含 scale/padding |
| `quant_block_size` | 无直接变量 | vLLM 把量化布局吸收进 `head_size` / spec / kernel |
| `index_kpool` | `tokens_per_state` | 都是“几个 token 合成一个 entry”，但语义不完全一一对应 |
| `IndexKeyCache.buffer` | `DeepseekV32IndexerCache.kv_cache` | 都是 per-layer index buffer |
| `DSATokenToKVPool` | `DeepseekV32IndexerCache` + manager | SGLang 显式 pool；vLLM 是 layer cache |
| `MiniMaxSparseKVPool` | `MiniMaxM3IndexerCache` | 主 cache + index side cache |
| `DeepSeekV4IndexerPool` | `DeepseekV4IndexerCache` | 都是 index K 的 per-ratio / per-state buffer |
| `layer_mapping` | `kv_source_layer_ids` / static context | layer 到 index pool 的归属映射 |
| `source_layer_of()` | `is_kv_source` / `kv_source_layer_id` | V4.1 source 共享 |
| `loc` / `out_cache_loc` | `slot_mapping` | token 写入位置 |
| `page_indices` | `block_table` | 逻辑页/块到物理页/块的映射 |
| `seq_len_tensor` | `seq_lens` | 每个请求的序列长度 |
| `topk_indices_buffer` | `topk_indices_buffer` | 临时 top-k 输出，不是 cache |
| `IndexerKPool` | `SparseAttnIndexer` / indexer backend | index 打分与 top-k |
| `C4Indexer` | `DeepseekV4Indexer` | V4 C4 indexer |
| `DeepseekV41Indexer` | `DeepseekV4Indexer`（V4.1） | V4.1 indexer |
| `select_candidate_block_ids` | `select_candidate_blocks` | V4.1 两级 top-k 的第一级 |
| `SetKAndS` | `indexer_k_quant_and_cache` | 写 index K 和 scale |

### 2.1 最容易混的两个映射

#### `index_page_size` vs `block_size / tokens_per_state`

SGLang：

```text
index_page_size = 一个 index page 里有多少个 compressed slot
```

vLLM：

```text
一个 block 里的 state 数
  = block_size / tokens_per_state
```

两者不是同名变量，但都在描述“一个物理存储单位里有多少 index entry”。

#### `index_head_dim` vs `head_size`

SGLang：

```text
index_head_dim = 128
entry = 128 + 4 scale = 132 B
```

vLLM：

```text
head_size = 132
```

vLLM 的 `head_size` 已经包含了 scale/padding，所以：

```text
vLLM entry bytes = head_size
SGLang entry bytes = index_head_dim + scale
```

不能直接拿 `index_head_dim` 和 vLLM 的 `head_size` 做一一比较。

---

## 3. DSA / V3.2 对比

| 项目 | SGLang | vLLM |
|---|---|---|
| cache 类 | `IndexKeyCache` | `DeepseekV32IndexerCache` |
| 归属 | 每层 side buffer | 每层 layer cache |
| spec | `pool` 字段 | `MLAAttentionSpec(cache_role=INDEXER)` |
| 每 entry | 132 B | 132 B |
| page | 64 | 64 |
| page 字节 | 8,448 B | 8,448 B |
| 写入 | `store_quantized` → `SetKAndS` | `indexer_k_quant_and_cache` |
| 读取 | `get_k_and_scale` | `SparseAttnIndexer` paged MQA |
| 驱逐 | RadixCache 释放主 page | `BlockPool.free_blocks` 释放主 block |
| 独立 prefix/evict | 无 | 无 |

核心差异：

- SGLang 的 `IndexKeyCache` 是 `DSATokenToKVPool` 内部显式 side buffer；
- vLLM 的 `DeepseekV32IndexerCache` 是通用 KV cache manager 管理的一层 layer cache；
- 两边都用 132 B/token，都把 index K 和主 cache 绑定到同一个逻辑 page/block。

---

## 4. MSA 对比

| 项目 | SGLang | vLLM |
|---|---|---|
| cache 类 | `MiniMaxSparseKVPool.index_k_pool` | `MiniMaxM3IndexerCache` |
| 组织 | `main_pool` + `index_k_pool` | 主 cache + index layer cache |
| bf16 | 256 B/token | 256 B/token |
| fp8 | 128 B/token | 128 B/token |
| nvfp4 | 72 B/token | 72 B/token |
| page | 128 | block 128 |
| bf16 page | 32 KiB | 32 KiB |
| fp8 page | 16 KiB | 16 KiB |
| nvfp4 page | 9 KiB | 9 KiB |
| 写入 | fused QK-norm + RoPE + KV insert | fused QK-norm + RoPE + KV insert |
| 读取 | `MiniMaxM3IndexerImpl.forward` | `MiniMaxM3IndexerImpl.forward` |
| 驱逐 | 主 page | 主 block |
| 独立 prefix/evict | 无 | 无 |

核心差异：

- SGLang 明确是 `main_pool` + `index_k_pool` 两个物理 pool；
- vLLM 是主 cache + `MiniMaxM3IndexerCache` layer cache；
- 两者都强调 key-only，index cache 不是 K+V，也没有独立 prefix tree。

---

## 5. DeepSeek V4 对比

| 项目 | SGLang | vLLM |
|---|---|---|
| cache 类 | `DeepSeekV4IndexerPool` | `DeepseekV4IndexerCache` |
| 组织 | 按 compression ratio 建 pool，`layer_mapping` 映射 | 每层 layer cache，spec 带 `tokens_per_state` |
| FP8 | 132 B/state | 132 B/state |
| FP4 | 68 B/state | 68 B/state |
| page | 由 `page_size / ratio` 决定 | 由 `block_size / tokens_per_state` 决定 |
| 写入 | compressor → index pool | compressor → index layer cache |
| 读取 | `C4Indexer` paged MQA | `DeepseekV4Indexer` paged MQA |
| 驱逐 | 主 compressed page | 主 compressed block |
| 独立 prefix/evict | 无 | 无 |

核心差异：

- SGLang 显式按 ratio 建 `index_pools`，并用 `layer_mapping` 找 pool；
- vLLM 把 ratio 信息放进 `MLAAttentionSpec.tokens_per_state`，由通用 manager 分配；
- 两边都让 compressor 写 index K，indexer 只读；
- 两边都不为 index 建独立 prefix tree。

---

## 6. DeepSeek V4.1 对比

| 项目 | SGLang | vLLM |
|---|---|---|
| cache 类 | `DeepSeekV4IndexerPool` | `DeepseekV4IndexerCache` |
| source 共享 | `source_layer_of()` / `layer_mapping` | `is_kv_source` / `is_index_source` + static context |
| consumer 是否建 cache | 不建，复用 source | 不建，复用 source |
| index K 来源 | compressor latent → `wk` → norm → RoPE | compressor latent → `wk` → norm → RoPE |
| 候选 block | `select_candidate_block_ids` | `select_candidate_blocks` |
| 两级 top-k | 有 | 有 |
| top-k 复用 | candidate（V4.1）；`skip_topk` 属于 V4 路径 | candidate（V4.1）；IndexCache 属于 V4 路径 |
| 驱逐 | source page 释放，consumer 一起不可达 | source block 释放，consumer 一起不可达 |
| 独立 prefix/evict | 无 | 无 |

核心差异：

- SGLang 的共享拓扑写在 **pool 层**：`layer_mapping` + `source_layer_of`；
- vLLM 的共享拓扑写在 **attention 初始化层**：`is_kv_source` 时创建 cache，否则从 `static_forward_context` 取 source 的 cache；
- 语义相同：只有 source 拥有 index K，consumer 复用；
- 两边都把 candidate block 作为两级 top-k 的中间结果，不当作持久 cache。

---

## 7. 生命周期对比

```text
SGLang：
  pool 初始化
    -> 按层/按 ratio 分配固定 slab
    -> 前向写入
    -> indexer 读取
    -> RadixCache 释放主 page
    -> index page 不可达

vLLM：
  layer cache 构造
    -> get_kv_cache_spec
    -> KVCacheManager 按 block 分配
    -> 前向写入
    -> indexer 读取
    -> BlockPool 释放主 block
    -> index page 不可达
```

共同点：

- index cache 都是固定容量；
- 都不是 per-request malloc；
- 都没有独立 index allocator；
- 都没有独立 index LRU；
- 都没有独立 index prefix tree；
- page/block 复用后，旧 index 数据被覆盖。

不同点：

- SGLang 的池对象更显式，适合按 ratio/source 手工组织；
- vLLM 的 spec/group 更通用，适合把 index cache 纳入统一 KV cache manager。

---

## 8. 前缀匹配与驱逐时的优先级

两边都不是“按架构优先级”决定，而是：

```text
最终可复用前缀
  = min(full, SWA, linear/Mamba, 主 cache 约束)
  -> 按 page/block 对齐
```

- DSA / MSA 的 index cache **不额外降低**前缀边界；
- 它们跟随主 cache 的 page/block 生命周期；
- SWA / linear / Mamba 才会真正缩短边界；
- index 的 top-k 是重新算的，不参与 prefix 持久化。

---

## 9. 总表

| 维度 | SGLang | vLLM |
|---|---|---|
| 核心抽象 | pool / buffer | layer cache + spec |
| index 归属 | 显式 index pool | 独立 layer cache |
| 逻辑单位 | page id | block id |
| 分配方式 | 启动时按层/ratio 分配 | manager 按 spec/block 分配 |
| 写入 | pool kernel | layer cache kernel |
| 读取 | paged MQA / MSA score | paged MQA / sparse MQA |
| 前缀 | RadixCache | KVCacheManager prefix cache |
| 驱逐 | 主 page | 主 block |
| 独立 index free/LRU | 无 | 无 |
| 独立 index prefix tree | 无 | 无 |
| source 共享 | pool/layer_mapping | static context/`is_kv_source` |
| top-k 复用 | skip_topk / candidate | IndexCache / candidate |

---

## 10. 最终结论

> **SGLang 和 vLLM 的核心逻辑一致：index K 是主 cache 的 sidecar，不独立参与 prefix match 和 eviction；差异主要在“如何表达和组织这个 sidecar”。**

```text
SGLang：
  显式 pool + page id + RadixCache

vLLM：
  layer cache + KVCacheSpec + KVCacheManager
```

对于 DSA / MSA / V4 / V4.1：

- 两边都使用独立 index K，不与主 K 共用物理字节；
- 两边都让 index K 跟随主 page/block 生命周期；
- 两边都没有 index 独立 prefix tree / LRU；
- 两边都在 V4.1 引入 source 共享和 candidate block；
- 两边的核心差异是 **pool 组织方式** 和 **spec/manager 表达方式**，不是 index 的生命周期语义。
