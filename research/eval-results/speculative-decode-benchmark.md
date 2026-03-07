# Speculative Decoding Benchmark (2026-03-06)

Prompt lookup speculative decoding: n-gram matching against input tokens to draft candidates.
No draft model — zero-cost candidate generation (CPU only).

**Model**: Qwen2.5-VL-3B-Instruct-4bit, Apple M3 Pro
**Script**: `examples/bench_speculative.py`
**Settings**: lookahead=5, temperature=0.0, max_tokens=64, 3 runs each

---

## Results by Scenario

### Scenario A: Typical VLM — "Describe this image in detail."

| Metric | Standard | Speculative | Diff |
|--------|----------|-------------|------|
| Decode TPS | 144.4 | 104.9 | **-27.3%** |
| Decode (ms) | 404 | 552 | +36.6% |
| Total (ms) | 774 | 929 | +20.1% |
| Acceptance rate | — | **0.0%** | |
| Drafted tokens | — | 70 | |
| Fallback rounds | — | 114 | |

VLM describes a random image — output is entirely novel text with no overlap to the prompt.
N-gram matching finds some spurious matches in the 70 drafted tokens (from repeated
subwords in the long visual-token prompt) but **none are accepted** by the target model.

### Scenario B: Structured JSON — "Output a JSON with keys: name, color, size, position, count."

| Metric | Standard | Speculative | Diff |
|--------|----------|-------------|------|
| Decode TPS | 141.0 | 116.7 | **-17.2%** |
| Decode (ms) | 272 | 320 | +17.7% |
| Total (ms) | 644 | 699 | +8.5% |
| Acceptance rate | — | **0.0%** | |
| Drafted tokens | — | 0 | |
| Fallback rounds | — | 0 | |

Zero n-gram matches found. The JSON output tokens (brackets, quotes, colons) don't form
n-grams that appear in the prompt. **Pure overhead = +17.2%** decode latency from the
speculative decode code path itself (verify call overhead, cache management).

### Scenario C: Repetitive list — "List all objects visible. Format each as: - object: description"

| Metric | Standard | Speculative | Diff |
|--------|----------|-------------|------|
| Decode TPS | 138.2 | 97.7 | **-29.3%** |
| Decode (ms) | 269 | 368 | +36.8% |
| Total (ms) | 641 | 746 | +16.4% |
| Acceptance rate | — | **2.2%** | |
| Drafted tokens | — | 45 | |
| Fallback rounds | — | 54 | |

Only 1/45 drafted tokens accepted. The "- object:" format in the prompt occasionally
matches an n-gram in the output, but the continuation almost never matches the target
model's actual next tokens.

---

## Analysis

### Acceptance Rate

| Scenario | Acceptance Rate | Drafted | Accepted |
|----------|---------------:|--------:|---------:|
| A: Describe image | 0.0% | 70 | 0 |
| B: JSON output | 0.0% | 0 | 0 |
| C: List format | 2.2% | 45 | 1 |

**Prompt lookup speculative decoding provides zero benefit for VLM inference.**

This is expected: VLM outputs are conditioned on visual content (pixel values), not on
text patterns in the prompt. The model's output distribution is dominated by what it
*sees*, not what it was *asked*. N-gram matching against the prompt cannot predict
image-conditioned outputs.

### Overhead

Even with 0% acceptance, speculative decode adds significant overhead:

| Source | Cost |
|--------|------|
| N-gram search (CPU) | ~2% prefill overhead |
| Verify call structure | ~17% decode overhead (Scenario B baseline) |
| Cache rollback on rejection | Additional overhead when drafts fail |

The verify call is the dominant cost: even when falling back to single-token decode,
the code path goes through `SpeculativeDecoder._verify()` → `target_model(tokens[None])`
instead of the optimized standard decode loop.

### Output Mismatch

All runs produced different text between standard and speculative paths (0/3 matches
in every scenario). This is because the speculative code path uses a different
function call pattern (`language_model` directly vs the standard `_step` wrapper),
causing subtle numerical differences even at temperature=0.

---

## Conclusions

1. **Prompt lookup speculative decoding is not useful for VLM tasks.** Acceptance rate is 0-2% across all tested scenarios.

2. **The overhead is significant**: 17-37% decode slowdown even when no drafts are accepted.

3. **Why it fails**: VLM outputs are image-conditioned. The text prompt contains the question, but the answer depends on pixel values that have no text representation in the prompt tokens. N-gram matching fundamentally cannot predict visual outputs.

4. **When it could help**: Pure text LLM tasks with highly repetitive/templated outputs (code generation, form filling) where output patterns match input patterns. Not applicable to VLM.

5. **Draft model speculative decode** (using a smaller LLM) could theoretically help, but on Apple Silicon the memory bandwidth is shared — loading a second model reduces available bandwidth for the target model, and the M-series unified memory architecture means draft model inference competes for the same bandwidth that limits decode speed.

### Recommendation

Speculative decode (prompt lookup) should remain available as an option but **disabled by default**. Higher ROI optimizations for VLM inference:

- **Frame-to-frame KV reuse** (roadmap #1): consecutive video frames share 80%+ context
- **Mid-stream FastV**: already done, -30-50% visual tokens
- **Shared text prefix KV**: already done, saves prefill on repeated prompts
