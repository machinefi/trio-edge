# Phase 1: Custom Generate Loop — Detailed Implementation Plan

## Goal

Replace mlx-vlm's `generate_step` with our own generate loop, taking ownership
of KV cache lifecycle. This enables cross-request cache reuse, prompt caching,
and confidence-based early stopping — none of which mlx-vlm supports.

**Expected gains:** Prefill -15~25% (prompt KV reuse), Decode latency -50~80%
on yes/no tasks (early stop), Memory -15% (persistent cache avoids re-alloc)

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
- `cache.make_prompt_cache()` → `trio_core.kv_cache.CacheManager`
- `stream_generate()` → `trio_core.generate.stream_generate()`
- `generate()` → called via our `MLXBackend.generate()`

## Current mlx-vlm Flow (What We're Replacing)

```
MLXBackend.generate()
  → mlx_vlm.generate(model, processor, prompt, **kwargs)
    → stream_generate(model, processor, prompt, **kwargs)
      → prepare_inputs() → input_ids, pixel_values, mask
      → generate_step(input_ids, model, pixel_values, mask, ...)
        → model.get_input_embeddings(input_ids, pixel_values, mask)
        → [chunked prefill loop] model.language_model(embeds, cache=prompt_cache)
        → [decode loop] model.language_model(y, cache=prompt_cache) → yield token
      → detokenizer → yield GenerationResult per token
```

### Key Observations from mlx-vlm Source

1. **Fresh cache every call** (line 309-313):
   ```python
   if prompt_cache is None:
       prompt_cache = cache.make_prompt_cache(model.language_model, max_kv_size=max_kv_size)
   ```
   Creates `KVCache()` per layer, discarded after generation completes.

2. **Chunked prefill** (line 370-387):
   Processes embeddings in chunks of `prefill_step_size=2048` tokens. This is
   good for peak memory but doesn't help with redundant computation.

3. **No cross-request state** — `generate_step` is stateless. No way to pass
   a previous cache in.

4. **Decode loop** (line 396-409):
   Standard autoregressive: `_step(y) → next_y`, yield token. No early stopping,
   no confidence threshold, no speculative decode.

5. **`wired_limit` context manager** — sets Metal memory limit based on model
   size. We should replicate this.

## New Flow (Phase 1)

```
MLXBackend.generate()
  → self._prepare(frames, prompt)         # same as before → input_ids, pixel_values, mask
  → trio_generate.generate(               # OUR generate
      model, processor, input_ids, pixel_values, mask,
      cache_manager=self._cache_manager,  # PERSISTENT
      early_stop_config=...,              # NEW
    )
    → model.get_input_embeddings(...)     # same (mlx-vlm's ViT)
    → cache_manager.get_or_create(...)    # reuse prompt KV if available
    → [prefill: only NEW tokens]          # skip cached prefix
    → [decode loop with early stop]       # stop on high-confidence EOS
    → cache_manager.save(...)             # persist for next request
```

## Components to Build

### 1. `src/trio_core/kv_cache.py` — Persistent KV Cache Manager

```python
class CacheManager:
    """Manages KV cache lifecycle across requests.

    Key features:
    - Prompt caching: text prompt KV computed once, reused across frames
    - Cache truncation: trim decode/visual tokens, keep prompt KV prefix
    - Cache invalidation: detect when prompt changes via hash
    """

    def __init__(self, model, max_kv_size=None):
        self._model = model
        self._max_kv_size = max_kv_size
        self._cache = None          # List[KVCache] — persistent
        self._prompt_hash = None    # hash of text prompt tokens
        self._prompt_seq_len = 0    # how many tokens are prompt-only KV (reusable)
        self._total_seq_len = 0     # total tokens in cache (prompt + visual + decode)

    def get_or_create(self, prompt_token_ids=None):
        """Get existing cache or create new one.

        If the prompt is the same as last time, truncate cache to prompt length
        (removing old visual + decode KV) so new visual tokens can be appended.

        Returns:
            (cache, skip_tokens): cache to use, number of embedding tokens to skip
        """
        prompt_hash = hash(prompt_token_ids.tobytes()) if prompt_token_ids is not None else None

        if self._cache is not None and self._prompt_hash == prompt_hash:
            # Same prompt — trim cache to prompt length, reuse prompt KV
            # KVCache.trim(n) removes the LAST n tokens
            trim_count = self._total_seq_len - self._prompt_seq_len
            if trim_count > 0:
                for c in self._cache:
                    c.trim(trim_count)
            self._total_seq_len = self._prompt_seq_len
            # Skip prompt tokens in embedding (they're already cached)
            return self._cache, self._prompt_seq_len

        # Different prompt or first request — create fresh cache
        self._cache = make_cache(self._model, self._max_kv_size)
        self._prompt_hash = prompt_hash
        self._prompt_seq_len = 0
        self._total_seq_len = 0
        return self._cache, 0

    def mark_prompt_boundary(self, prompt_len):
        """Mark how many tokens are the reusable prompt prefix."""
        self._prompt_seq_len = prompt_len
        self._total_seq_len = prompt_len

    def update_seq_len(self, n):
        """Update total sequence length after prefill/decode."""
        self._total_seq_len = n

    def reset(self):
        """Force cache invalidation."""
        self._cache = None
        self._prompt_hash = None
        self._prompt_seq_len = 0
        self._total_seq_len = 0
```

**Important: ViT still runs every time.** Even with cache reuse,
`model.get_input_embeddings(input_ids, pixel_values, mask)` runs the full ViT
forward pass. We only skip the **LLM prefill** for cached prompt tokens, not
the vision encoder. ViT optimization is Phase 2.

**Cache reuse flow (same prompt, new frame):**

```
Request 1 (cold):
  get_input_embeddings() → embeds = [prompt_tokens | visual_tokens]
  cache = new KVCache()
  prefill(all embeds) → cache = [prompt_KV | visual_KV]
  decode() → cache = [prompt_KV | visual_KV | decode_KV]
  mark_prompt_boundary(len(prompt_tokens))

Request 2 (warm, same prompt):
  get_input_embeddings() → embeds = [prompt_tokens | NEW_visual_tokens]
  cache.trim(visual_len + decode_len) → cache = [prompt_KV]  # keep prompt
  skip = prompt_len → embeds_to_prefill = [NEW_visual_tokens]  # skip prompt embeds
  prefill(NEW_visual_tokens) → cache = [prompt_KV | new_visual_KV]
  decode() → cache = [prompt_KV | new_visual_KV | decode_KV]
```

**Cache reuse scenarios:**

| Scenario | Prompt Same? | Visual Tokens | Action |
|----------|-------------|---------------|--------|
| Same prompt, new frame | Yes | Different | Trim to prompt, re-prefill visual only |
| Same prompt, same image | Yes | Same | Trim to prompt, re-prefill visual (ViT still runs) |
| New prompt | No | Any | Full prefill from scratch |

**Estimated prefill savings (LLM prefill only, ViT excluded):**
- Typical prompt: ~80 tokens, visual: ~300 tokens → skip 80/380 ≈ **21% LLM prefill**
- With ToMe (visual ~130 tokens): skip 80/210 ≈ **38% LLM prefill**
- Note: ViT forward is separate and unaffected (Phase 2 target)

### 2. `src/trio_core/generate.py` — Custom Generate Loop

```python
def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    cache_manager: CacheManager = None,
    early_stop: EarlyStopConfig = None,
    max_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    prefill_step_size: int = 2048,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Our generate_step — drop-in replacement for mlx-vlm's.

    Differences from mlx-vlm:
    1. Uses CacheManager for persistent KV cache
    2. Skips already-cached tokens in prefill
    3. Supports early stopping on confidence threshold
    """
    # Get embeddings (still mlx-vlm's ViT)
    embedding_output = model.get_input_embeddings(input_ids, pixel_values, mask=mask, **kwargs)
    inputs_embeds = embedding_output.inputs_embeds

    # Get or reuse cache
    if cache_manager is not None:
        cache, skip_tokens = cache_manager.get_or_create(input_ids, ...)
        if skip_tokens > 0:
            inputs_embeds = inputs_embeds[:, skip_tokens:]
            input_ids = input_ids[:, skip_tokens:]
    else:
        cache = make_cache(model.language_model)
        skip_tokens = 0

    # Chunked prefill (only for non-cached tokens)
    # ... same chunked prefill logic as mlx-vlm ...

    # Handle cross_attention_states / encoder_outputs (for encoder-decoder
    # models like Mllama). Qwen models don't use these, but Gemma 3 might.
    # Must propagate from first _step output to subsequent calls via kwargs.

    # Decode loop with early stopping
    for n in range(max_tokens):
        y, logprobs = _step(y, cache)

        # Early stop: check if EOS token probability exceeds threshold
        if early_stop and early_stop.should_stop(y, logprobs, n):
            break

        yield y.item(), logprobs

    # Update cache manager
    if cache_manager is not None:
        cache_manager.update_seq_len(...)
```

### 3. `src/trio_core/early_stop.py` — Confidence-Based Early Stopping

```python
@dataclass
class EarlyStopConfig:
    """Configuration for EOS-probability-based early stopping.

    For yes/no questions (POPE, monitoring alerts), the model often produces
    high-confidence answers in the first 1-2 tokens. Continuing to generate
    wastes compute on unnecessary explanation text.

    We check the EOS token's probability specifically — NOT the max logprob
    of any token (which would trigger on high-confidence content tokens
    like "the" or "is").
    """
    enabled: bool = True
    min_tokens: int = 1           # don't stop before this many tokens
    eos_threshold: float = 0.8    # stop if P(EOS) > this after min_tokens
    eos_token_ids: list = None    # EOS token IDs (from model config)

    def should_stop(self, token, logprobs, n_generated):
        if n_generated < self.min_tokens:
            return False
        if not self.eos_token_ids:
            return False
        # Check probability of EOS tokens
        probs = mx.exp(logprobs)
        for eos_id in self.eos_token_ids:
            if eos_id < probs.shape[0] and probs[eos_id].item() > self.eos_threshold:
                return True
        return False
```

**Why EOS probability, not max probability:**
- High max logprob (e.g., P("the") = 0.98) means the model is confident about
  the NEXT token, not that it wants to stop. This would cause premature truncation
  of valid responses.
- High EOS probability (e.g., P(EOS) = 0.85) means the model thinks the response
  is complete. This is the correct signal for early stopping.

**Use case impact:**
- POPE (yes/no): generates "Yes" then P(EOS) spikes → stop after 1-3 tokens
  instead of 10-16 → **80% fewer decode steps**
- Open-ended description: P(EOS) stays low until natural end → no effect

### 4. Modified `MLXBackend` — Wire It Together

```python
class MLXBackend(BaseBackend):
    def load(self):
        # Same model loading
        self._model, self._processor = mlx_vlm.load(self.model_name)
        # NEW: persistent cache manager
        self._cache_manager = CacheManager(self._model.language_model)

    def generate(self, frames, prompt, *, max_tokens=512, ...):
        formatted, kwargs = self._prepare(frames, prompt)

        # Extract prompt-only token IDs for cache key
        prompt_token_ids = self._get_prompt_tokens(prompt)

        # Use OUR generate instead of mlx_vlm.generate
        from trio_core.generate import generate
        result = generate(
            self._model, self._processor,
            cache_manager=self._cache_manager,
            prompt_token_ids=prompt_token_ids,
            early_stop=EarlyStopConfig(enabled=True),
            **kwargs,
        )
        return GenerationResult(text=result.text, ...)
```

## Implementation Steps

### Step 1: Skeleton Generate Loop (no cache reuse yet)

Create `src/trio_core/generate.py` with a `generate_step()` that is functionally
identical to mlx-vlm's. Wire it into `MLXBackend`. Run regression tests — must
produce identical results.

**Files:** `src/trio_core/generate.py`, `src/trio_core/backends.py`
**Test:** `python examples/run_regression.py` — all benchmarks must match baseline ±1%

### Step 2: Persistent KV Cache

Create `src/trio_core/kv_cache.py` with `CacheManager`. Implement prompt-hash
based cache reuse. Wire into generate loop.

**Files:** `src/trio_core/kv_cache.py`, `src/trio_core/generate.py`
**Test:** Run same-prompt consecutive inference, verify prefill time drops on 2nd+ call.
Measure with:
```bash
python examples/run_eval.py --resolution 1080 --tome 4 --consecutive 5
```

### Step 3: Early Stopping

Create `src/trio_core/early_stop.py`. Wire into decode loop. Enable by default
for benchmarks with known answer format (POPE = yes/no).

**Files:** `src/trio_core/early_stop.py`, `src/trio_core/generate.py`
**Test:** Run POPE benchmark, verify same accuracy with fewer generated tokens.

### Step 4: Streaming Support

Update `stream_generate()` to use our generate loop with cache reuse.
Wire into `MLXBackend.stream_generate()`.

**Files:** `src/trio_core/generate.py`, `src/trio_core/backends.py`
**Test:** Run webcam/screen narrator examples, verify streaming still works.

### Step 5: Benchmarks and Validation

Full regression suite across all models:
```bash
for model in Qwen2.5-VL-3B Qwen3-VL-4B Qwen3.5-0.8B; do
    python examples/run_regression.py -m mlx-community/${model}-Instruct-4bit
done
```

Compare against baselines from eval-baseline-plan.md.

## Risk Analysis

| Risk | Impact | Mitigation |
|------|--------|------------|
| Cache state corruption | Wrong outputs | Hash-based invalidation, regression tests |
| Memory leak from persistent cache | OOM | Cache size limits, explicit cleanup |
| Numerical differences from skip | Accuracy drop | Compare logprobs, not just text |
| mlx-vlm API changes break us | Build failure | Pin mlx-vlm version, integration tests |
| wired_limit not replicated | Perf regression | Copy mlx-vlm's wired_limit logic |
| KVCache.trim() changes positions | Position mismatch | Verify trim preserves offset correctly |
| encoder-decoder model compat | Crash on Gemma 3 | Propagate cross_attention_states in _step |

## Acceptance Criteria

All must pass before proceeding to Phase 2:

- [ ] Identical accuracy on all 5 Tier 1 benchmarks (within ±1% of baseline)
- [ ] LLM prefill latency -15%+ on consecutive same-prompt requests
- [ ] Decode tokens reduced 50%+ on POPE (early stop via EOS probability)
- [ ] No memory increase (peak memory ≤ baseline)
- [ ] All existing tests pass
- [ ] Streaming generation works (webcam/screen narrator)
- [ ] No regressions on Qwen models (non-Qwen: best-effort, may need fixes)
- [ ] Step 1 (skeleton) matches mlx-vlm output exactly before adding features

## Code Size Estimate

| File | Lines | Complexity |
|------|-------|------------|
| `src/trio_core/generate.py` | ~200 | Medium — core generate loop |
| `src/trio_core/kv_cache.py` | ~100 | Low — cache management |
| `src/trio_core/early_stop.py` | ~40 | Low — threshold check |
| `src/trio_core/backends.py` changes | ~50 | Low — wire new generate |
| Tests | ~150 | Medium — cache + generate tests |
| **Total** | **~540** | |

## Dependencies

- **mlx** (mx.array, nn.Module) — direct dependency, no change
- **mlx-vlm** — still needed for model loading + ViT + LLM forward, but
  `generate`/`stream_generate`/`generate_step` are no longer called
- **mlx-lm** — `make_sampler`, `make_logits_processors`, `maybe_quantize_kv_cache`
  are imported from here by mlx-vlm; we'll import them directly

## References

- mlx-vlm `generate_step`: `.venv/.../mlx_vlm/generate.py:230-409`
- mlx-vlm `stream_generate`: `.venv/.../mlx_vlm/generate.py:412-528`
- mlx-vlm `cache.make_prompt_cache`: `.venv/.../mlx_vlm/models/cache.py:16-42`
- mlx-lm `KVCache`: `.venv/.../mlx_lm/models/cache.py`
- native-engine-plan.md — high-level phased plan
- eval-baseline-plan.md — regression baselines to compare against
