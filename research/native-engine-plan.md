# Native Engine Plan — Replacing mlx-vlm Dependency

## Current State

TrioCore's MLX inference path depends entirely on mlx-vlm:

```
Current call chain:
  TrioCore.generate()
    → MLXBackend._prepare()
      → _frames_to_temp_video()      ← extra I/O overhead
      → mlx_vlm.process_vision_info  ← mlx-vlm
      → mlx_vlm.process_inputs       ← mlx-vlm
    → mlx_vlm.generate()             ← mlx-vlm (core)
      → model.get_input_embeddings() ← ViT forward
      → generate_step()              ← KV cache + decode loop
```

What TrioCore owns: video pipeline (frame extraction, dedup, motion gate), ToMe (monkey-patched into ViT blocks), multi-backend abstraction, API server.

**Core inference = mlx-vlm.**

## Inference Pipeline — 5 Stages

```
Input → [Vision Encoder + ToMe] → visual tokens → [LLM Prefill] → [KV Cache] → [Decode]
         ^^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^      ^^^^^^^^^^^^     ^^^^^^^^^    ^^^^^^^^
         ToMe here                 fewer tokens    quantization     caching      batching
```

### Stage-by-Stage Analysis

#### 1. Vision Encoder

**What mlx-vlm does:**
- Loads ViT weights (nn.Module subclass)
- ViT forward: patches → N transformer blocks → visual embeddings
- PatchMerger / MLP projector: maps ViT dimensions to LLM dimensions

**Problem: Runs full ViT from scratch on every call. In video, 80-90% of patches are unchanged between consecutive frames, yet all are recomputed.**

**Benefits of owning this:**
- Native ToMe: no monkey-patch, built into forward pass, zero overhead
- Inter-frame patch reuse: compare raw patches, only run ViT on changed patches
- Per-layer adaptive r: merge more aggressively in early layers, less in later layers

**Metric impact: Prefill Latency -5~15%**

#### 2. LLM Prefill

**What mlx-vlm does (from generate_step source):**
```python
embedding_output = model.get_input_embeddings(input_ids, pixel_values, mask, **kwargs)
inputs_embeds = embedding_output.inputs_embeds

# Chunked prefill (prefill_step_size=2048)
while inputs_embeds.shape[1] > 1:
    model.language_model(inputs_embeds=inputs_embeds[:, :n], cache=prompt_cache, ...)
```

Every generate() call prefills all tokens (visual + text) from scratch.

**Problem: In video, the text prompt is identical across frames and visual tokens are mostly the same, yet everything is recomputed every time.**

**Benefits of owning this:**
- Text prompt KV reuse: compute prompt KV entries once, reuse across all frames
- Visual prefix caching: reuse KV entries for unchanged visual tokens from previous frame

**Metric impact: Prefill Latency -20~40% (biggest single improvement for video)**

#### 3. KV Cache

**What mlx-vlm does:**
```python
# Creates fresh cache on every generate() call
if prompt_cache is None:
    prompt_cache = cache.make_prompt_cache(model.language_model, max_kv_size=max_kv_size)
```

Supports kv_bits quantization and max_kv_size limit, but no cross-request reuse.

**Problem: Fresh cache per request, discarded after use. Video frames share the vast majority of context.**

**Benefits of owning this:**
- Persistent cache: keep previous frame's KV, only update changed entries
- Streaming eviction: when cache is full, evict oldest/least-important entries (not clear all)
- Frame-level cache sharing: multiple video frames share scene context cache

**Metric impact: Peak Memory -15~25%, Prefill Latency -10~20%**

#### 4. Decode

**What mlx-vlm does:**
```python
def _step(y):
    outputs = model.language_model(y, cache=prompt_cache, ...)
    logits = outputs.logits[:, -1, :]
    y = sampler(logprobs)
    return y, logprobs
```

Standard auto-regressive: one token at a time → feed back → next token.

**Problems:**
- No speculative decoding (small draft model generates multiple tokens, verified in parallel)
- No early stopping (even when first token logprob > 0.95 for yes/no, keeps generating)
- No batching (multiple camera streams processed serially)

**Benefits of owning this:**
- Speculative decoding: 2-3x decode throughput (paper-validated)
- Confidence early stop: 80%+ fewer useless tokens in yes/no scenarios
- Continuous batching: multiple requests share prefill computation

**Metric impact: Decode Throughput +30~50%**

## Expected Core Metrics Impact

| Metric | Current (mlx-vlm) | After Phase 1 | After All Phases | Confidence |
|--------|-------------------|---------------|------------------|------------|
| Prefill Latency | baseline | -25~35% | -40~60% | High (KV reuse is proven) |
| Decode Throughput | baseline | +10~20% (early stop) | +30~50% | Medium (speculative needs testing) |
| Visual Tokens | ToMe r=4 | same | additional -30~50% | Medium (adaptive r needs tuning) |
| Peak Memory | baseline | -15% (persistent cache) | -20~30% | High |
| Quality | baseline | same | +1~3% | Low (content-aware gains are small) |

## Phased Execution Plan

### Phase 1: Custom Generate Loop (current target)

**Goal:** Replace mlx-vlm's generate_step, take ownership of KV cache.

**What to build:**
1. Own `generate_step()`: KV cache creation + sampling loop
2. Reuse mlx-vlm's `model.get_input_embeddings()` and `model.language_model()`
3. Implement persistent prompt KV (reuse text prompt KV entries across requests)
4. Implement confidence early stop

**What NOT to touch:**
- Vision Encoder (still mlx-vlm's ViT)
- Model loading (still mlx_vlm.load())
- ToMe implementation (still monkey-patch)

**Acceptance criteria (ALL must pass before moving to Phase 2):**
- [ ] POPE accuracy >= baseline (no regression from generate loop change)
- [ ] Prefill latency -20%+ (via prompt KV reuse)
- [ ] Consecutive-frame latency -30%+ (inter-frame KV reuse)
- [ ] Peak memory does not increase
- [ ] All existing tests pass

**Files to modify/create:**
- `src/trio_core/generate.py` (new: own generate loop)
- `src/trio_core/kv_cache.py` (new: persistent KV cache management)
- `src/trio_core/backends.py` (MLXBackend.generate uses own loop)

---

### Phase 2: Native Vision Encoder

**Prerequisite: Phase 1 benchmarks pass acceptance criteria.**

**Goal:** Own ViT forward pass, native ToMe integration.

**What to build:**
1. Own Qwen VL ViT (RotaryEmbedding + Attention + MLP + PatchMerger)
2. ToMe as a forward parameter, no monkey-patch
3. Per-layer adaptive r
4. Inter-frame patch-level reuse (optional — skip if Phase 1's inter-frame KV reuse is sufficient)

**What NOT to touch:**
- LLM decoder (still mlx-vlm's language_model)
- Weight loading format

**Acceptance criteria:**
- [ ] POPE accuracy >= Phase 1
- [ ] Prefill latency improves 10%+ over Phase 1
- [ ] ToMe adds zero latency overhead (vs Phase 1's monkey-patch)

---

### Phase 3: Full Native Engine

**Prerequisite: Phase 2 benchmarks pass acceptance criteria.**

**Goal:** Zero mlx-vlm dependency. Only depends on MLX framework (mx.array, nn.Module).

**What to build:**
1. Own LLM decoder forward
2. Own weight loading (safetensors → model state dict)
3. Own tokenizer wrapper
4. Speculative decoding
5. Continuous batching

**This phase is the heaviest and riskiest. Evaluate feasibility after Phase 1+2.**

## References

- mlx-vlm generate_step source: `mlx_vlm/chat.py`
- mlx-vlm stream_generate source: `mlx_vlm/__init__.py`
- AIM paper (ICCV 2025): training-free adaptive inference, 7x FLOPs reduction
- PVC paper (CVPR 2025): progressive visual token compression for video
- See [paper-research.md](../memory/paper-research.md) for full literature review
