# trio-core Research

Research notes on visual token compression for edge VLM inference.

## Documents

- [visual-token-compression.md](visual-token-compression.md) — Core research direction: three key papers, prototype results, compatibility analysis with KV cache / batch scheduling / quantization.
- [tome-implementation-plan.md](tome-implementation-plan.md) — Detailed step-by-step plan for implementing ToMe in Qwen2.5-VL's vision encoder.
- [native-engine-plan.md](native-engine-plan.md) — Phased plan to replace mlx-vlm dependency: per-stage analysis, core metrics projections, Phase 1/2/3 execution plan.
- [eval-results/](eval-results/) — Baseline, compressed, and ToMe eval JSON files.

## Key Thesis

> The biggest win in VLM inference isn't faster decoding — it's **fewer visual tokens**.
> Prefill cost is O(n²) in token count. KV cache is O(n). Reducing visual tokens
> attacks both simultaneously, and existing tokens have massive redundancy.

## Optimization Stack (orthogonal layers, all compound)

```
Frame-level:  Motion Gate → Temporal Dedup → fewer frames       [v0.1 ✓]
Token-level:  ToMe (inside ViT) ──────────→ fewer visual tokens [v0.2 ✓]
Compute:      Quantization (INT4) ────────→ cheaper per-token   [mlx-vlm handles]
Memory:       KV Cache / StreamingLLM ───→ bounded decode mem   [Phase 3]
Throughput:   Batch Scheduling ───────────→ concurrent requests  [Phase 4]
```

## Results Summary

### POPE Benchmark (Object Hallucination, 100 COCO images, Apple M3 Pro)

#### Qwen2.5-VL-3B-Instruct-4bit

| Config | Accuracy | F1 | Latency | Tokens |
|---|---|---|---|---|
| Baseline | 92.0% | 0.913 | 952ms | 363 |
| ToMe r=4, hidden | 81.0% | 0.800 | 752ms (-21%) | 131 (-64%) |
| ToMe r=4, keys+RoPE | 80.0% | 0.778 | 842ms (-12%) | 131 (-64%) |

#### Qwen3-VL-4B-Instruct-4bit

| Config | Accuracy | F1 | Latency | Tokens |
|---|---|---|---|---|
| Baseline | 91.0% | 0.901 | 795ms | 278 |
| **ToMe r=4, hidden** | **91.0% (0%)** | **0.901** | **777ms (-2%)** | **258 (-7%)** |

### 1080p Synthetic Eval (Qwen2.5-VL-3B, avg across complexity levels)

| Config | Prefill | Tokens | Memory |
|---|---|---|---|
| Baseline | 1808ms | 748 | 4.02GB |
| **ToMe r=4** | **490ms (-73%)** | **242 (-68%)** | **3.85GB (-4%)** |

### 480p Synthetic Eval (Qwen3-VL-4B)

| Config | Prefill | Tokens | Memory |
|---|---|---|---|
| Baseline | 835ms | 323 | 4.07GB |
| ToMe r=4 | 573ms (-31%) | 303 (-6%) | 3.71GB (-9%) |

### Key Findings

1. **Qwen3-VL: zero quality loss** — 91% accuracy with and without ToMe, plus 31% prefill speedup
2. **Qwen2.5-VL at 1080p: 73% prefill speedup** — 1808ms → 490ms with 68% token reduction
3. Prefill scales quadratically — fewer tokens → disproportionately faster prefill
4. K-matrix similarity (even with RoPE fix) doesn't beat hidden states — hidden states remain the best metric
5. Qwen3-VL has fewer tokens to start with (323 vs 419 at 480p), so absolute compression is smaller
6. ToMe is more impactful at higher resolutions where token counts are larger

### min_keep_ratio Sweep

| min_keep_ratio | Accuracy | Latency | Tokens |
|---|---|---|---|
| 0.30 (default) | 90.0% | 765ms | 131 |
| 0.25 | 88.0% | 743ms | 117 |
| 0.20 | 75.0% (-17%) | 734ms | 97 |

Finding: 0.25 is viable (~2% more loss for ~11% more compression). 0.2 is too aggressive.

### K-matrix Similarity

- **v1 (without RoPE)**: 82% accuracy — worse than hidden states (90%). K was recomputed post-block without rotary position embeddings, producing different similarity structure than actual attention.
- **v2 (with RoPE)**: Fixed to apply rotary position embeddings to K before similarity computation, matching what the actual attention mechanism computes. Needs re-benchmarking.

### Next Optimization Directions

1. **Re-benchmark K-matrix with RoPE fix** — v2 should now match paper's recommendation
2. **High-resolution eval** — 1080p where prefill dominates and compression matters most (--resolution flag added)
3. **Fine-grained min_keep_ratio** — sweep 0.26-0.29 range

## Status

- [x] Eval framework built (synthetic + POPE + TextVQA benchmarks)
- [x] Baseline metrics collected (Qwen2.5-VL-3B, M3 Pro)
- [x] Naive compression prototype (uniform stride, post-encoder)
- [x] Research documented, implementation plan written
- [x] **ToMe v1 implemented** (bipartite soft matching in vision encoder)
- [x] POPE benchmark comparison (baseline vs ToMe r=4 vs r=8)
- [x] min_keep_ratio sweep (0.3 → 0.25 → 0.2)
- [x] K-matrix v2 with RoPE (implemented, benchmarked — still worse than hidden)
- [x] High-resolution 1080p benchmark (73% prefill speedup)
- [x] **Qwen3-VL ToMe — zero quality loss** (91% → 91% on POPE, 31% prefill speedup)
- [x] Qwen3.5 support verified (uses ToMeQwen3VisionWrapper, 36-50% prefill speedup)
- [x] Multi-model profiles (Gemma 3, SmolVLM) + generalized Transformers backend
- [x] mlx-vlm dependency analysis + phased replacement plan (native-engine-plan.md)
- [ ] **Phase 1: Custom generate loop** — KV cache reuse, prompt caching, early stop
- [ ] Phase 2: Native Vision Encoder — built-in ToMe, adaptive r
- [ ] Phase 3: Full native engine — zero mlx-vlm dependency
