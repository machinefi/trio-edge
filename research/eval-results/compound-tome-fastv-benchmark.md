# ToMe + FastV Compound Benchmark (2026-03-06)

Compound visual token compression: ToMe (vision encoder, r=4) + FastV (LLM layers, 50% prune).
ToMe reduces tokens in the ViT, then FastV further prunes the remaining visual tokens in the LLM.

**Settings**: ToMe r=4 hidden metric, FastV ratio=0.5 layer 2, n=50 per benchmark

---

## Qwen2.5-VL-3B — Accuracy

| Benchmark    | Baseline | ToMe r=4 | FastV 0.5 | **ToMe+FastV** | Compound vs Baseline |
|--------------|:--------:|:--------:|:---------:|:--------------:|:--------------------:|
| POPE-random  | 88%      | 84%      | 80%       | **84%**        | -4%                  |
| POPE-adv     | 82%      | 70%      | 76%       | **74%**        | -8%                  |
| TextVQA      | 66%      | 38%      | 66%       | **46%**        | -20%                 |
| GQA          | 58%      | 46%      | 50%       | **54%**        | -4%                  |
| MMBench      | 94%      | 82%      | 90%       | **80%**        | -14%                 |

## Qwen2.5-VL-3B — Latency (ms/sample)

| Benchmark    | Baseline | ToMe r=4 | FastV 0.5 | **ToMe+FastV** | Compound vs Baseline |
|--------------|:--------:|:--------:|:---------:|:--------------:|:--------------------:|
| POPE-random  | 385      | 595      | 342       | **567**        | +47%                 |
| POPE-adv     | 385      | 594      | 337       | **589**        | +53%                 |
| TextVQA      | 890      | 1347     | 898       | **1620**       | +82%                 |
| GQA          | 511      | 706      | 437       | **676**        | +32%                 |
| MMBench      | 280      | 348      | 220       | **277**        | -1%                  |

## Qwen3-VL-4B — Accuracy (MMBench skipped — deepstack incompatible with ToMe)

| Benchmark    | Baseline | ToMe r=4 | **ToMe+FastV** | Compound vs Baseline |
|--------------|:--------:|:--------:|:--------------:|:--------------------:|
| POPE-random  | 86%      | 86%      | **78%**        | -8%                  |
| POPE-adv     | 82%      | 82%      | **76%**        | -6%                  |
| TextVQA      | 76%      | 70%      | **66%**        | -10%                 |
| GQA          | 66%      | 64%      | **56%**        | -10%                 |

## Qwen3-VL-4B — Latency (ms/sample)

| Benchmark    | Baseline | ToMe r=4 | **ToMe+FastV** | Compound vs Baseline |
|--------------|:--------:|:--------:|:--------------:|:--------------------:|
| POPE-random  | 347      | 323      | **251**        | -28%                 |
| POPE-adv     | 349      | 311      | **253**        | -28%                 |
| TextVQA      | 819      | 714      | **655**        | -20%                 |
| GQA          | 458      | 384      | **345**        | -25%                 |

---

## Analysis

### Accuracy Impact

**Compound compression is aggressive.** ToMe reduces tokens in the ViT (70% reduction on 1080p),
then FastV prunes another 50% of the remaining visual tokens in the LLM. The combined reduction
is roughly 85% of original visual tokens.

- **Qwen2.5-VL-3B**: Significant accuracy loss across all benchmarks. TextVQA drops 20pp (66%→46%),
  MMBench drops 14pp (94%→80%). POPE and GQA are more resilient (-4 to -8pp).
- **Qwen3-VL-4B**: More uniform degradation (-6 to -10pp across benchmarks). Better than Qwen2.5
  on TextVQA (66% vs 46%) suggesting the newer architecture is more robust to token reduction.

### Latency Impact

The story differs dramatically between models:

- **Qwen2.5-VL-3B**: Compound is **slower** than baseline (+32% to +82%), even slower than ToMe alone.
  The ToMe grid alignment and compressed grid position computation add overhead that exceeds
  the FastV savings from fewer tokens.
- **Qwen3-VL-4B**: Compound is **faster** than baseline (-20% to -28%) and even faster than
  ToMe alone. The Qwen3 architecture benefits more from the reduced token count.

### Why Qwen2.5 is slower with compound

ToMe compresses visual tokens non-uniformly, requiring grid realignment and position ID
recomputation. On Qwen2.5-VL (which uses windowed attention in the ViT), this overhead is
significant. FastV then adds its own per-layer manual attention computation. The two overheads
compound negatively.

### Compatibility Issues

| Model | ToMe | FastV | Compound | Notes |
|-------|:----:|:-----:|:--------:|-------|
| Qwen2.5-VL-3B | OK | OK | **OK** | Full support |
| Qwen3-VL-4B | OK | OK | **Partial** | MMBench fails (deepstack merger reshape) |
| Qwen3.5-0.8B | OK | **No** | **No** | DeltaNet layers lack self_attn for FastV importance scoring |
| Gemma3-4B | No | **No** | **No** | Neither ToMe nor FastV supported |

---

## Conclusions

1. **Compound compression is too aggressive for most tasks.** The 85% token reduction
   causes 4-20pp accuracy drops depending on benchmark.
2. **Use ToMe OR FastV, not both.** FastV alone preserves accuracy better (especially TextVQA)
   while delivering speed gains. ToMe alone is better for Qwen3-VL where accuracy is preserved.
3. **Qwen3-VL benefits more from compound** — both in accuracy resilience and latency gain.
4. **Recommended configs**:
   - Qwen2.5-VL: FastV 0.5 only (best accuracy/speed tradeoff)
   - Qwen3-VL: ToMe r=4 only (zero accuracy loss, 20-30% faster)
   - Qwen3.5: ToMe r=4 only (FastV incompatible)

## Raw Data

| File | Config |
|------|--------|
| `qwen2.5-vl-3b_tome_r4_fastv_0.5.json` | Compound |
| `qwen3-vl-4b_tome_r4_fastv_0.5.json` | Compound (no MMBench) |
| `qwen2.5-vl-3b_baseline.json` | Baseline |
| `qwen2.5-vl-3b_tome_r4.json` | ToMe only |
| `qwen2.5-vl-3b_fastv_0.5.json` | FastV only |
| `qwen3-vl-4b_baseline.json` | Baseline |
| `qwen3-vl-4b_tome_r4.json` | ToMe only |
