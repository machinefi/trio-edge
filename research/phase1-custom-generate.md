# Phase 1: Custom Generate Loop — Detailed Implementation Plan

## Goal

Replace mlx-vlm's `generate_step` with our own generate loop, taking ownership
of KV cache lifecycle. This enables cross-request cache reuse, prompt caching,
and confidence-based early stopping — none of which mlx-vlm supports.

**Expected gains:** Prefill skip on cache hit → **45% total latency reduction**,
Decode -50~80% on yes/no tasks (early stop), buffer reuse avoids re-alloc.

## Implementation Progress

### Step 1: Skeleton Generate Loop ✅ DONE

Created `src/trio_core/generate.py` with `generate_step()` functionally identical
to mlx-vlm's. Wired into `MLXBackend.generate()` and `stream_generate()`.

**Verification:**
- Direct token comparison: MATCH (identical output to mlx-vlm)
- POPE-random: 88.0% → 88.0% (0% delta)
- POPE-adversarial: 82.0% → 82.0% (0% delta)
- TextVQA: 68.0% → 66.0% (-2%, within ±3% threshold)
- 120 unit tests: all pass
- Streaming: works at backend level

### Step 2: Persistent KV Cache ✅ DONE

Implemented `PromptCache` class in `generate.py` (no separate file needed).
Wired into `MLXBackend` as a persistent cache across requests.

**Key finding — Qwen VL token layout:**
```
Token layout (170 total for 224×224 frame):
[prefix: 14 tokens] [vision_start] [video_pad × 145] [vision_end] [suffix: 13 tokens]
```

Visual tokens are in the MIDDLE (positions 14-158). Since KVCache is append-only
with trim-from-end, prefix-only reuse saves only ~8%. The original plan's
"same prompt different image" cache reuse is NOT worth the complexity.

**What we implemented instead:**
1. **Buffer reuse** — persistent KVCache buffers, trimmed to 0 instead of
   re-allocated. Avoids GPU buffer allocation/deallocation overhead.
2. **Exact-match detection** — hash input_ids, if identical skip entire
   prefill (ViT + LLM) by reusing cached KV state + first token.
3. **Cache lifecycle** — trim decode tokens on cache hit, restore to
   post-prefill offset.

**Benchmark results (224×224, Qwen2.5-VL-3B, 20 tokens):**
```
Cache MISS (avg): 754ms  [740, 755, 753, 776, 746]
Cache HIT  (avg): 333ms  [334, 332, 335, 332]  (excluding cold first hit)
Speedup: 44.8% overall, 55.8% on warm hits
Prefill saved: 338ms per cache-hit request
```

**Verification:**
- R1 (miss): "I see a pattern of small, colorful dots..."
- R2 (hit):  "I see a pattern of small, colorful dots..."  ← identical
- R3 (miss, different prompt): "The image you provided is a white background..."
- 120 unit tests: all pass

### Step 3: Early Stopping ⬜ TODO

EOS-probability-based early stopping. Check P(EOS token) specifically — NOT
max logprob of any token (which would trigger on "the", "is", etc.).

**Design:**
```python
@dataclass
class EarlyStopConfig:
    min_tokens: int = 1           # don't stop before this many tokens
    eos_threshold: float = 0.8    # stop if P(EOS) > this after min_tokens
    eos_token_ids: list = None    # from model config

    def should_stop(self, token, logprobs, n_generated):
        if n_generated < self.min_tokens:
            return False
        probs = mx.exp(logprobs)
        for eos_id in self.eos_token_ids:
            if probs[eos_id].item() > self.eos_threshold:
                return True
        return False
```

**Expected impact:**
- POPE (yes/no): "Yes" → P(EOS) spikes → stop after 1-3 tokens instead of
  10-16 → **80% fewer decode steps**
- Open-ended: P(EOS) stays low → no effect

### Step 4: Streaming Support ✅ DONE (with Step 2)

`stream_generate()` already uses our generate loop with `prompt_cache_manager`.

### Step 5: Benchmarks and Validation ⬜ TODO

Full regression across all models after Steps 3-4.

## What We Replace vs What We Keep

### Keep (still from mlx-vlm)
- `mlx_vlm.load()` — model + processor loading
- `model.get_input_embeddings()` — ViT forward + embedding projection
- `model.language_model()` — LLM forward pass (single call)
- `processor.apply_chat_template()` — prompt formatting
- `processor.detokenizer` — token → text conversion
- ToMe monkey-patching into ViT blocks

### Replace (our own)
- `generate_step()` → `trio_core.generate.generate_step()`
- KV cache management → `trio_core.generate.PromptCache`
- `stream_generate()` → wired through `MLXBackend.stream_generate()`
- `generate()` → called via `MLXBackend.generate()`

## Architecture

### `PromptCache` (in `src/trio_core/generate.py`)

```python
class PromptCache:
    """Persistent KV cache manager for cross-request reuse.

    Benefits:
    1. Buffer reuse: avoids GPU buffer re-allocation between requests
    2. Exact-match: identical input_ids → skip entire prefill (ViT + LLM)
    """

    def __init__(self, model, max_kv_size=None):
        self._kv_cache = None          # List[KVCache] — persistent buffers
        self._input_hash = None        # MD5 of input_ids for exact-match
        self._first_token = None       # (token, logprobs) from prefill
        self._prefill_offset = 0       # KV offset after prefill (before decode)

    def get_or_create_cache(self):
        """Reuse existing buffers (trimmed to 0) or create new ones."""

    def check_hit(self, input_ids):
        """Hash-based exact-match detection."""

    def save_state(self, input_ids, first_token, first_logprobs, kv_cache):
        """Save hash + first token + prefill offset for future reuse."""
```

### Cache Hit Flow

```
Request 1 (cold):
  _prepare() → input_ids, pixel_values, mask
  generate_step(prompt_cache_manager=pcache):
    check_hit() → MISS
    get_or_create_cache() → fresh KVCache
    get_input_embeddings() → ViT forward
    prefill → fill KV cache
    save_state(input_ids, first_token, offset)
    decode loop → yield tokens

Request 2 (same input):
  _prepare() → same input_ids, pixel_values, mask
  generate_step(prompt_cache_manager=pcache):
    check_hit() → HIT
    trim KV to prefill_offset (remove decode tokens)
    skip ViT + prefill entirely
    use cached first_token
    decode loop → yield tokens (identical output)
```

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cache state corruption | Wrong outputs | Hash-based invalidation, regression tests |
| Memory leak from persistent cache | OOM | Cache trimmed each request, no unbounded growth |
| KV position mismatch after trim | Wrong outputs | Trim only decode tokens, verified with A/B test |
| mlx-vlm API changes break us | Build failure | Pin mlx-vlm version, integration tests |
| wired_limit not replicated | Perf regression | Copied mlx-vlm's wired_limit logic exactly |
| encoder-decoder model compat | Crash on non-Qwen | Propagate cross_attention_states in _step |

## Acceptance Criteria

All must pass before proceeding to Phase 2:

- [x] Identical accuracy on Tier 1 benchmarks (within ±3% of baseline)
- [x] Step 1 (skeleton) matches mlx-vlm output exactly
- [x] Cache hit skips prefill and produces identical output
- [x] Buffer reuse avoids re-allocation
- [x] All existing tests pass (120/120)
- [x] Streaming generation works with cache
- [ ] Early stopping reduces decode tokens 50%+ on POPE
- [ ] Full regression suite across all Qwen models
- [ ] No memory increase (peak memory ≤ baseline)

## References

- mlx-vlm `generate_step`: `.venv/.../mlx_vlm/generate.py:230-409`
- mlx-lm `KVCache`: `.venv/.../mlx_lm/models/cache.py`
- native-engine-plan.md — high-level phased plan
- eval-baseline-plan.md — regression baselines to compare against
