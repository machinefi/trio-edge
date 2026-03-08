# trio-core VLM 推理引擎 — 全面审查报告

## 1. 整体引擎健康度评分：7.5 / 10 (修复后)

原始评分 6.5，修复后提升至 7.5。

**已修复的致命扣分项：**

| 原扣分 | 类别 | 修复状态 |
|--------|------|----------|
| ~~-1.5~~ | ~~代码重复~~ | **已修复**: 4 个 backend 的 decode loop 统一到 `MLXBackend._run_generate`/`_run_stream_generate`/`_run_ar_decode`/`_run_ar_stream_decode`。消除 ~240 行重复。 |
| **-1.0** | **generate_step 复杂度** | 未修复：仍是 360 行单函数，圈复杂度 > 25。 |
| **-1.0** | **无并发 / 无 batching** | 未修复：单请求串行。 |

---

## 2. 模块耦合 & 依赖分析

### 2.1 剩余高耦合点

| # | 耦合点 | 涉及模块 | 危害 |
|---|--------|---------|------|
| ~~1~~ | ~~**Engine 直接构造后端子类**~~ | ~~`engine.py`~~ | **已修复**: `resolve_backend(config)` + `register_backend()` |
| 3 | **generate_step 访问 PromptCache 内部** | `generate.py` | 直接读 `_prefill_offset`、`_last_embeds`，打破封装。 |
| 4 | **model.language_model._position_ids 突变** | `tome_backend.py`, `fastv_backend.py`, `compressed_backend.py` | MRoPE 位置编码通过 mutation 传递。当前 threading.Lock 保护下安全，但架构不良。 |
| 5 | **mlx-vlm 紧耦合** | `fastv_backend.py`, `generate.py` | `create_attention_mask`、`KVCache` 依赖第三方内部 API。 |

**已修复:** #2 Backend 子类重复 detokenize 循环 — 统一到基类方法。

---

## 3. VLM 特有问题清单

| 优先级 | 问题类型 | 影响模块/文件 | 描述 | 建议修复 | 工作量 |
|--------|---------|--------------|------|---------|--------|
| ~~P2~~ | ~~错误处理缺失~~ | ~~`backends.py`~~ | ~~`generate_step` 没有 OOM 处理~~ | **已修复**: `_check_memory()` prefill 前检查 Metal 可用内存 | ~~M~~ |
| ~~P2~~ | ~~量化死代码~~ | ~~`generate.py`, `backends.py`, `fastv_backend.py`~~ | ~~`quantize_cache_fn` 硬编码 `kv_bits=None`~~ | **已修复**: 删除所有 no-op quantize_cache_fn 调用 | ~~S~~ |
| **P2** | 可测试性差 | 全部 backend 文件 | `generate()` 强依赖 GPU，无 CI 测试 | Protocol 抽象 model forward | **L** |
| ~~P2~~ | ~~stream_generate 不是真 async~~ | ~~`engine.py`~~ | ~~`stream_analyze` 内部同步阻塞事件循环~~ | **已修复**: thread+queue 桥接，不阻塞 event loop | ~~M~~ |
| ~~P3~~ | ~~命名不一致~~ | ~~`model_adapter.py`~~ | ~~`VisionOutput.hidden_states` 声明为 `object`~~ | **已修复**: `object` → `Any` | ~~S~~ |
| ~~P3~~ | ~~hash 性能~~ | ~~`generate.py`~~ | ~~MD5 hash 大型 pixel_values 较慢~~ | **已修复**: strided sampling，大 tensor 只 hash shape+首尾+stride | ~~S~~ |

---

## 4. 推荐的重构 Roadmap（剩余项）

### Week 2: generate_step 拆分

| # | 目标 | 涉及模块 | 收益 | 风险 |
|---|------|---------|------|------|
| 3 | **拆分 `generate_step` 为 prefill + decode + cache_manager** | `generate.py` | 圈复杂度从 25+ 降到 <10 | 中 |
| 4 | **PromptCache 封装** | `generate.py` | 不再通过 `_` 前缀属性交互 | 低 |

### Week 3: 性能优化

| # | 目标 | 涉及模块 | 收益 | 风险 |
|---|------|---------|------|------|
| 6 | ~~**hash 优化**~~ | ~~`generate.py`~~ | **已修复**: strided sampling | 低 |

### Week 4: 健壮性

| # | 目标 | 涉及模块 | 收益 | 风险 |
|---|------|---------|------|------|
| 7 | ~~**OOM 防护 + token count guard**~~ | ~~`backends.py`~~ | **已修复**: `_check_memory()` | 低 |

---

## 已完成修复记录

| 修复项 | 优先级 | 文件 | 修改内容 |
|--------|--------|------|----------|
| 统一 4 个 decode loop | P0 | `backends.py`, `tome_backend.py`, `fastv_backend.py`, `compressed_backend.py` | 基类添加 `_run_generate`/`_run_stream_generate`/`_run_ar_decode`/`_run_ar_stream_decode`；3 个子类删除重复 decode loop |
| 修复假流式输出 | P0/P1 | `fastv_backend.py`, `compressed_backend.py` | `stream_generate` 从 generate-then-yield 改为真正逐 token 流式 |
| Server temp file leak | P0 | `api/server.py` | `_resolve_media` 返回 temp_path，调用方 try/finally 清理 |
| merge_tokens 向量化 | P1 | `tome.py` | 消除 Python for loop，改用 broadcast argmin + 批量 gather/scatter |
| 量化死代码清理 | P2 | `backends.py`, `fastv_backend.py` | 删除所有 `quantize_cache_fn(kv_bits=None)` no-op 调用和未使用的 `functools`/`maybe_quantize_kv_cache` import |
| OOM 防护 | P2 | `backends.py` | `_check_memory()` prefill 前估算 pixel_values 内存，超出 Metal 可用内存时抛出 `MemoryError`。self-review 修复：移入 `_prepare()` 覆盖所有子类路径 |
| 类型命名修复 | P3 | `model_adapter.py` | `VisionOutput.hidden_states: object` → `Any`，`MergeResult.embeds: object` → `Any` |
| stream_analyze async 桥接 | P2 | `engine.py` | `stream_analyze` 用 thread+queue 桥接同步 `stream_generate`，不再阻塞 event loop |
| Backend registry | P2 耦合 | `backends.py`, `engine.py` | `resolve_backend(config)` 统一后端选择逻辑，engine 不再直接构造子类。`register_backend()` 支持插件注册 |
| Hash 优化 | P3 | `generate.py` | `_hash_input` 大 pixel_values 用 strided sampling（shape+首尾+stride），<64K 元素仍全量 hash |
