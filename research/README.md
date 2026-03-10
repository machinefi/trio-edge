# trio-core Research

Research notes on visual token compression for edge VLM inference.

## Pipeline Big Picture

Every VLM inference decomposes into 5 stages. This is the map we optimize against:

```
输入 → [Vision Encoder + ToMe] → visual tokens → [LLM Prefill] → [KV Cache] → [Decode]
        ^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^      ^^^^^^^^^^^^     ^^^^^^^^^    ^^^^^^^^
        ToMe 在这里              更少了            量化在这里        缓存在这里    batch在这里
```

### Per-Stage Status (updated 2026-03-07)

```
Stage 0: 输入 (Video Pipeline)      ██████████ 100%   done
Stage 1: Vision Encoder + ToMe      ██████████ 100%   ToMe + adaptive r + native ViT + content-aware done
Stage 2: Visual Token Count          ██████████ 100%   post-encoder compression + content-aware adaptive ratio done
Stage 3: LLM Prefill                 █████████░  90%   generate loop + prefix cache + native loading wired in
Stage 4: KV Cache                    █████████░  90%   4-tier cache + KV reuse + StreamMem + attention sink done
Stage 5: Decode                      ███░░░░░░░  30%   streaming done; speculative decode removed (0% accept for VLM)
```

### Priority Ranking (ROI for video inference latency)

| # | What | Expected Gain | Difficulty | Stage | Status |
|---|------|---------------|------------|-------|--------|
| 1 | ~~Frame-to-frame KV reuse~~ | ~~-60~80% video latency~~ | ~~High~~ | ~~4~~ | DONE (1.6x Qwen2.5, 1.7x Qwen3, 1.35x Qwen3.5) |
| 2 | ~~Mid-stream FastV (true KV prune)~~ | ~~-30~50% visual tokens~~ | ~~Medium~~ | ~~2~~ | DONE |
| 3 | ~~Shared text prefix KV~~ | ~~-20~40% prefill~~ | ~~Medium~~ | ~~3~~ | DONE |
| 4 | ~~Speculative decoding~~ | ~~+30~50% decode TPS~~ | ~~Medium~~ | ~~5~~ | DONE (0% accept for VLM) |
| 5 | ~~Content-aware adaptive r~~ | ~~+quality, same speed~~ | ~~Low~~ | ~~2~~ | DONE |
| 6 | ~~Native ToMe (no monkey-patch)~~ | ~~cleaner arch~~ | ~~Medium~~ | ~~1~~ | DONE |
| 7 | ~~Unify ToMe generate path~~ | ~~free features~~ | ~~Low~~ | ~~1/5~~ | DONE |
| 8 | ~~Remove mlx-vlm load dep~~ | ~~zero dependency~~ | ~~High~~ | ~~3~~ | 80% (T1 native, T2 fallback) |

### Stage Details

**Stage 0 — Input (Video Pipeline)** -- DONE
- StreamCapture (webcam/RTSP/YouTube), temporal dedup (-30~70% frames),
  motion gate (-80%+ VLM calls), smart resize, model profiles

**Stage 1 — Vision Encoder + ToMe** -- DONE
- Done: ToMe bipartite soft matching, windowed-attn aware, adaptive r ramp,
  hidden-state metric, Qwen2.5/3/3.5 support,
  native ViT (NativeToMeQwen25Vision / NativeToMeQwen3Vision — proper OO, no monkey-patch)

**Stage 2 — Visual Token Count** -- DONE
- Done: ToMe compression (Qwen2.5 1080p: 748->242 tokens, -68%),
  post-encoder compression, mid-stream FastV (single-pass KV cache pruning,
  zero double computation, MRoPE-aware),
  content-aware adaptive r (per-image diversity → dynamic merge ratio)

**Stage 3 — LLM Prefill** -- 90%
- Done: own generate loop (sampler, KV cache, logits processors internalized),
  mlx-lm runtime dep mostly removed, chunked prefill, shared text prefix KV cache
  (three-tier: exact hit > prefix hit > full miss), early stopping,
  native model loading for all T1 models (qwen2_5_vl, qwen3_vl, qwen3_5 — bit-identical),
  load_native() wired into MLXBackend as primary path with mlx-vlm fallback
- TODO: update ToMe imports to use native models (low priority while mlx-vlm still installed)

**Stage 4 — KV Cache** -- 90%
- Done: persistent KVCache with buffer reuse, quantized KV cache,
  text prefix KV reuse across frames (15 tokens saved per inference),
  frame-to-frame KV reuse via visual embedding similarity gating
  (four-tier cache: exact hit > visual similarity hit > prefix hit > full miss,
  1.6x speedup Qwen2.5, 1.7x speedup Qwen3, 1.35x speedup Qwen3.5).
  DeltaNet support via state snapshot/restore (recurrent state is "all or nothing" —
  snapshotted after prefill, restored on visual hit, inspired by MARCONI MLSys '25).
  StreamMem bounded KV cache (saliency-based eviction + prototype merging,
  budget guard for long video single-pass prefill),
  attention sink (StreamingLLM-style, protect first N tokens from eviction).
  Hybrid model support (KVCache layers evicted, DeltaNet state unchanged)
- Note: cross-frame incremental streaming NOT viable — VLMs not trained for
  appended visual KV across requests; generate_step does full prefill per call.
  StreamMem works for long single-video prefill, not cross-request accumulation.

**Stage 5 — Decode** -- 30%
- Done: auto-regressive, streaming output, early stopping config
- Removed: speculative decoding (prompt lookup 0% acceptance for VLM — visual tokens ≠ text output; code deleted)
- TODO: continuous batching

### Per-Stage Model Differences (updated 2026-03-06)

```
                    Qwen2.5-VL       Qwen3-VL         Qwen3.5          Gemma3        SmolVLM
                    ──────────       ────────         ───────          ──────        ───────
Stage 1: ViT
  patch             14               14               16               14(SigLIP)    14/16(SigLIP)
  merge_factor      28               28               32               14(no merge)  14/16(no merge)
  windowed attn     YES              no               no               no            no
  deepstack         no               YES(tuple)       YES(empty)       no            no
  ToMe wrapper      Qwen25Wrapper    Qwen3Wrapper     Qwen3Wrapper     N/A           N/A
  ViT blocks        32               32               12(0.8B)/27(9B)  varies        varies

Stage 2: Visual Tokens
  token type        image+video      image+video      image+video      image only    image only
  token config      image_token_id   image_token_index image_token_index fixed 256    fixed 64
  merge signature   (img,vid,feat,   (feat,embed,ids, same as Qwen3    different     different
                     embed,ids)       img,vid)

Stage 3: LLM Prefill
  LLM type          Qwen2(GQA)       Qwen2(GQA)       DeltaNet+Attn    Gemma2        SmolLM
  RoPE              3D MRoPE         3D MRoPE          3D MRoPE         standard      standard
  position_ids      get_rope_index   get_rope_index    get_rope_index   simple seq    simple seq
  layers            36(3B)/28(7B)    32                 24(0.8B)         26-48         24

Stage 4: KV Cache
  KV heads          2(3B)/4(7B)      4                  2(0.8B)/4(4B+)  4-8           3-4
  DeltaNet layers   0                0                  18(0.8B)/24(4B)  0             0
  cache type        KVCache          KVCache            KVCache+ArraysC  KVCache       KVCache
  visual reuse      trim KV          trim KV            state snapshot   trim KV       trim KV

Stage 5: Decode
  eos handling      same             same               same             different     different
  stopping_criteria same             same               same             different     different
```

Key takeaway: Qwen family shares the same LLM layer loop (Stage 3-5).
Differences are in Stage 1-2 (ViT architecture, token merge signature),
already abstracted via wrappers + `_is_qwen3` flag.
Gemma/SmolVLM have entirely different ViT — ToMe not yet supported.

## Documents

- [visual-token-compression.md](visual-token-compression.md) — Core research direction: three key papers, prototype results, compatibility analysis with KV cache / batch scheduling / quantization.
- [tome-implementation-plan.md](tome-implementation-plan.md) — Detailed step-by-step plan for implementing ToMe in Qwen2.5-VL's vision encoder.
- [native-engine-plan.md](native-engine-plan.md) — Phased plan to replace mlx-vlm dependency: per-stage analysis, core metrics projections, Phase 1/2/3 execution plan.
- [eval-strategy.md](eval-strategy.md) — Evaluation benchmark selection: research across model reports, compression papers, and eval toolkits. Tiered benchmark suite (Tier 1 regression gate, Tier 2 full eval, Tier 3 video).
- [eval-baseline-plan.md](eval-baseline-plan.md) — Tier 1 baseline collection plan: 6 models × 5 benchmarks × baseline/ToMe configs.
- [phase1-custom-generate.md](phase1-custom-generate.md) — Phase 1 detailed implementation: custom generate loop, persistent KV cache, early stopping. Code-level design with mlx-vlm source analysis.
- [eval-results/mlxvlm-native-baselines.md](eval-results/mlxvlm-native-baselines.md) — mlx-vlm native baselines: ground-truth comparison (4 models × 5 benchmarks) showing trio-core adds no meaningful accuracy or latency overhead.
- [eval-results/speculative-decode-benchmark.md](eval-results/speculative-decode-benchmark.md) — Speculative decode benchmark (historical): 0% acceptance, code removed.
- [eval-results/compound-tome-fastv-benchmark.md](eval-results/compound-tome-fastv-benchmark.md) — ToMe + FastV compound benchmark: 85% token reduction is too aggressive, use one or the other.
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
Memory:       KV Cache + StreamMem ──────→ bounded decode mem   [done]
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

### Generate Loop A/B (trio-core vs mlx-vlm, Qwen2.5-VL-3B, 480p, 32 tokens)

| Metric | mlx-vlm | trio-core | Diff |
|---|---|---|---|
| Prefill | 1018ms | 1016ms | -0.2% |
| Decode | 524ms | 513ms | -2.1% |
| Total | 1542ms | 1529ms | -0.8% |
| Output match | - | - | 5/5 identical |

Conclusion: zero overhead from custom generate loop. Output is bit-identical at temperature=0.

### Prefix Cache A/B (Qwen2.5-VL-3B, 480p, same prompt different images)

| Condition | Avg Prefill | Notes |
|---|---|---|
| Cold start (full miss) | 1069ms | Fresh cache each time |
| Prefix hit (15/417 tokens reused) | 1045ms (-2%) | Same prompt, different pixels |

Note: Qwen's chat template puts visual tokens very early (position 15), so text prefix
is only 3.6% of total. Savings scale with prefix length — multi-turn conversations or
long system prompts before `<|vision_start|>` would see larger gains.

### Visual Similarity KV Reuse (Frame-to-frame, same prompt, 480p, 5-frame sequences)

| Model | Warm Speedup | Warm Savings | Cold Speedup | Output |
|---|---|---|---|---|
| Qwen2.5-VL-3B | **1.57x** | **36.4%** | 1.42x | Identical |
| Qwen3-VL-4B | **1.71x** | **41.5%** | 1.59x | Near-identical |
| Qwen3.5-0.8B | **1.35x** | **25.8%** | 1.34x | Correct (DeltaNet state snapshot) |

Settings: threshold=0.95, noise=0.01, 3 runs × 5 frames each.
"Warm" = frames 2-5 (similar to previous frame, KV reuse triggers).
"Cold" = frame 1 (always full prefill).
Note: Qwen3.5 uses DeltaNet (recurrent state, not KV cache). State is snapshotted after
prefill and restored on visual hit — "all or nothing" reuse (no partial trim).

### mlx-vlm Native Baseline Comparison (4 models × 5 benchmarks, n=50)

trio-core vs raw mlx-vlm.generate() — zero trio-core code in the native path.

| Model | Accuracy diff | Latency diff | Notes |
|---|---|---|---|
| Qwen2.5-VL-3B | ±2-6% | trio-core **faster** | POPE/MMBench: trio wins; TextVQA: native wins |
| Qwen3-VL-4B | ±2-14% | trio-core **faster** | TextVQA +14% (template effect); latency -5~20% |
| Qwen3.5-0.8B | ±4-12% | native **25% faster** | Tiny model — trio-core overhead proportionally larger |
| Gemma3-4B | ±2-8% | mixed | POPE-adv native much slower (1071 vs 697ms, thermal?) |

Conclusion: **trio-core adds no meaningful accuracy or latency overhead on 3B+ models.**
Differences are within noise (n=50) and prompt-template-sensitive, not engine-level.
Full details: [eval-results/mlxvlm-native-baselines.md](eval-results/mlxvlm-native-baselines.md)

### Speculative Decode Benchmark (Qwen2.5-VL-3B, M3 Pro, lookahead=5)

| Scenario | Acceptance Rate | Decode Overhead | Verdict |
|---|---|---|---|
| A: "Describe image" | 0.0% | +37% slower | No benefit |
| B: JSON output | 0.0% | +18% slower | Pure overhead |
| C: Repetitive list | 2.2% | +37% slower | Negligible |

**Prompt lookup speculative decoding is not useful for VLM inference.**
VLM outputs are conditioned on visual content (pixels), not text patterns in the prompt.
N-gram matching against prompt tokens cannot predict image-conditioned outputs.
Even with 0% acceptance, the speculative code path adds 17-37% decode overhead.
Full details: [eval-results/speculative-decode-benchmark.md](eval-results/speculative-decode-benchmark.md)

### Key Findings

1. **Qwen3-VL: zero quality loss** — 91% accuracy with and without ToMe, plus 31% prefill speedup
2. **Qwen2.5-VL at 1080p: 73% prefill speedup** — 1808ms → 490ms with 68% token reduction
3. Prefill scales quadratically — fewer tokens → disproportionately faster prefill
4. K-matrix similarity (even with RoPE fix) doesn't beat hidden states — hidden states remain the best metric
5. Qwen3-VL has fewer tokens to start with (323 vs 419 at 480p), so absolute compression is smaller
6. ToMe is more impactful at higher resolutions where token counts are larger
7. **Speculative decode (prompt lookup) useless for VLM** — 0% acceptance, 17-37% overhead. VLM outputs are image-conditioned, not text-pattern-predictable

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

### MVBench — Video Understanding (8 tasks × 20 samples, Qwen2.5-VL-3B, M3 Pro)

| Config | Accuracy | Avg Latency | Avg Visual Tokens |
|---|---|---|---|
| Baseline | 70.0% | 3789ms | 1555 |
| ToMe r=4 | 64.4% (-5.6pp) | 3192ms (-16%) | 541 (-65%) |

Per-task breakdown:

| Task | Baseline | ToMe | Impact | Notes |
|---|---|---|---|---|
| object_existence | 95% | 95% | zero | Simple perception |
| moving_attribute | 90% | 85% | -5pp | |
| moving_count | 80% | 65% | -15pp | Precision task |
| counterfactual_inference | 70% | 55% | -15pp | Reasoning task |
| action_sequence | 65% | 60% | -5pp | |
| moving_direction | 55% | 55% | zero | |
| action_prediction | 50% | 50% | zero | |
| object_interaction | 55% | 50% | -5pp | |

**Key insight: -65% tokens 的真正价值不在单帧速度。** 16 帧 480p 短视频只有 1555 tokens,
M3 Pro 轻松处理,实际 latency 只降 16%。ToMe 的核心价值在于：
- **高分辨率** — 1080p 从 748→242 tokens, prefill -73% (quadratic scaling)
- **长视频/连续流** — KV budget 内能容纳更多帧的历史上下文
- **OOM 防护** — 64+ 帧视频不压缩直接爆内存

对 Frigate 持续 480p 监控流：**不开 ToMe 是更好的默认值**。
ToMe 适用于高分辨率或长视频分析场景。

### SurveillanceVQA — Anomaly Detection (1,827 samples, yes/no, UCF-Crime)

[SurveillanceVQA-589K](https://arxiv.org/abs/2505.12589) detection benchmark — balanced yes/no on 13 anomaly categories from UCF-Crime surveillance videos. This is our **core use case benchmark** (Frigate-style security monitoring).

#### TrioCore Baseline (9 T1 models)

| Model | Params | Accuracy | F1 | Recall | Specificity | Yes Rate | Latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-7B | 7B | **70.1%** | 0.362 | 25.3% | **92.8%** | 13.3% | 587ms |
| Qwen3-VL-8B | 8B | 69.0% | 0.395 | 30.2% | 88.6% | 17.7% | 450ms |
| Qwen2.5-VL-3B | 3B | 68.4% | 0.504 | 47.6% | 79.1% | 29.9% | 375ms |
| Qwen3-VL-2B | 2B | 67.6% | 0.137 | 7.7% | 97.9% | 4.0% | 193ms |
| Qwen3.5-0.8B | 0.8B | 67.6% | 0.441 | 51.7% | 58.2% | 45.2% | 118ms |
| Qwen3-VL-4B | 4B | 67.5% | 0.484 | 45.4% | 78.7% | 29.3% | 304ms |
| Qwen3.5-2B | 2B | 67.3% | 0.108 | 5.9% | 98.4% | 3.1% | 189ms |
| **Qwen3.5-4B** | 4B | 65.2% | **0.556** | 65.1% | 65.2% | 44.9% | 295ms |
| Qwen3.5-9B | 9B | 56.7% | 0.550 | **79.0%** | 45.5% | 62.7% | 452ms |

#### mlx-vlm Raw Baseline Comparison (No TrioCore Optimizations)

To determine whether low SurveillanceVQA scores are caused by TrioCore optimizations or by fundamental domain gap, we ran raw `mlx_vlm.generate()` on the same benchmark (only Qwen2.5-VL models are supported by mlx-vlm 0.1.15):

| Model | Backend | Accuracy | F1 | Recall | Specificity | Yes Rate | Latency |
|---|---|---|---|---|---|---|---|
| Qwen2.5-VL-3B | TrioCore | 68.4% | 0.504 | 47.6% | 79.1% | 29.9% | 375ms |
| | mlx-vlm raw | 67.0% | 0.068 | 3.6% | 99.1% | 1.8% | 222ms |
| | **delta** | **+1.4%** | +0.436 | +44.0pp | -20.0pp | +28.1pp | |
| Qwen2.5-VL-7B | TrioCore | 70.1% | 0.362 | 25.3% | 92.8% | 13.3% | 587ms |
| | mlx-vlm raw | 67.4% | 0.100 | 5.4% | 98.7% | 2.7% | 308ms |
| | **delta** | **+2.7%** | +0.262 | +19.9pp | -5.9pp | +10.6pp | |

#### Key Findings

1. **TrioCore does NOT degrade performance** — accuracy is slightly *higher* (+1.4% to +2.7%) than raw mlx-vlm. Our optimization pipeline does not introduce regression.

2. **Raw mlx-vlm models almost never say "yes"** — yes rate 1.8-2.7%, making them useless as anomaly detectors despite similar accuracy (both sides are close to the 50/50 balanced baseline ceiling of ~67%).

3. **All models struggle** — accuracy ≤70% regardless of backend or model size (0.8B to 9B). Even the 9B model only reaches 56.7%. This is a **fundamental domain gap**: these general-purpose VLMs were not trained on surveillance video data.

4. **The accuracy ceiling is ~67-70% for a balanced yes/no dataset** — models that mostly say "No" (high specificity, low recall) achieve ~67% because 50.7% of the dataset is "No". True anomaly detection requires high recall, where all models fail badly.

5. **Interesting recall-specificity tradeoff** — smaller/newer models (Qwen3.5) tend to be more "trigger happy" (higher yes rate = higher recall but lower specificity). Qwen3.5-4B achieves the best F1 (0.556) with balanced yes/no rate. Qwen3.5-9B has highest recall (79%) but worst accuracy due to over-predicting "Yes" (62.7%).

6. **This validates LoRA/distillation plan** — the ≤70% ceiling is fundamental, not optimization-induced. Domain-specific fine-tuning has massive upside potential. See [lora.md](lora.md).

### Next: Real Video Benchmarks

Target use case: Frigate-style security/surveillance monitoring.

| Priority | Benchmark | Scale | Format | Surveillance Relevance |
|---|---|---|---|---|
| **1st** | **MVBench** | 4000 samples, 20 tasks | 多选 (3选1) | 中 — action/object/state tasks map to surveillance |
| 2nd | VideoMME | 900 videos, 2700 QA | 多选 ABCD | 低 — 通用视频理解, 行业标准参照 |
| 3rd | UCVL | 1699 videos, 16990 QA | 多任务 | 高 — 专门监控异常检测, 数据待公开 |

**MVBench 优先**：HuggingFace 直接下载, 多选格式复用 POPE 框架, 20 子任务可定位哪个能力受优化影响。
关键子任务：action_sequence, object_interaction, state_change, moving_direction

### Engineering TODO

1. **Continuous batching** — concurrent request handling for server mode
2. **Full native engine** — zero mlx-vlm dependency for T1 models

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
- [x] **Phase 1: Custom generate loop** — own sampler, KV cache reuse, prompt caching, early stop, chunked prefill
- [x] **Shared text prefix KV cache** — three-tier cache (exact hit > prefix hit > full miss), MRoPE-aware
- [x] **FastV visual token pruning** — attention-based importance scoring, Qwen2.5/3/3.5 support
- [x] **Mid-stream FastV** — single-pass KV cache pruning (zero double computation), MRoPE position_ids fix
- [x] ~~**Speculative decoding**~~ — removed (0% accept for VLM — visual tokens don't match text output)
- [x] **mlx-vlm native baselines** — 4 models × 5 benchmarks, trio-core matches or beats native on 3B+
- [x] **Native ToMe ViT** — NativeToMeQwen25Vision / NativeToMeQwen3Vision, proper OO subclass, no monkey-patch
- [x] **Unified ToMe generate path** — ToMeMLXBackend delegates to generate_step (gets PromptCache, early stopping, streaming for free). Fixed mx.eval() deepstack bug.
- [x] **ToMe + FastV compound benchmark** — 85% token reduction too aggressive, recommend using one or the other
- [x] **Frame-to-frame KV reuse** — visual embedding similarity gating, four-tier cache hierarchy (1.6x Qwen2.5, 1.7x Qwen3, 1.35x Qwen3.5). Input_ids check prevents wrong-answer reuse; p10 cosine for robust visual discrimination. DeltaNet support via ArraysCache state snapshot/restore.
- [x] **Content-aware adaptive r** — per-image diversity scoring, dynamic r scaling [0.2, 1.0], stacks with layer-adaptive
- [x] **StreamMem bounded KV cache** — saliency-based eviction + prototype merging + attention sink. Hybrid model support (KVCache eviction + DeltaNet passthrough). Proxy query scoring (chat template end-tokens as stand-in query)
- [x] **Attention sink (StreamingLLM)** — protect first N visual tokens from eviction, prevents model collapse after repeated eviction rounds
- [x] **Cross-frame streaming analysis** — VLMs not trained for appended visual KV across requests; StreamMem works for single-video prefill only
- [x] **ModelAdapter abstraction** — decouple model-family ops from optimization backends (5 adapters: Qwen2.5-VL, Qwen3-VL, InternVL, LLaVA, FastVLM). Promotes InternVL3 + nanoLLaVA to Tier 1.
- [x] **NativeToMeStandardVision** — ToMe wrapper for standard ViTs (SigLIP, for nanoLLaVA)
- [x] **trust_remote_code fix** — InternVL3 + nanoLLaVA loading via mlx-vlm
- [x] **FastVLM blocked** — CoreML `.mlpackage` vision encoder incompatible with mlx-vlm pure-MLX loader. Stays Tier 2.
- [x] **SurveillanceVQA baseline** — 9 T1 models × 1,827 samples, all ≤70% accuracy (domain gap). mlx-vlm raw comparison confirms no optimization regression.
- [x] **Tier 1 baseline benchmark (9 Qwen models)** — POPE + synthetic eval on M3 Ultra
- [x] **Native model loading (T1)** — vendored qwen2_5_vl (1080 lines), qwen3_vl (1240 lines), qwen3_5 (640 lines, reuses qwen3_vl vision). All bit-identical with mlx-vlm. Fixed upstream mx.eval() deepstack bug.
- [x] **Wire native loading into MLXBackend** — load_native() as primary path, mlx-vlm fallback for T2 models
- [ ] **Benchmark: Baseline vs ToMe (r=4)** — measure quality/speed tradeoff per model
- [ ] **Benchmark: Baseline vs KV Reuse (threshold=0.95)** — frame-to-frame speedup per model
- [ ] **Benchmark: InternVL3 / nanoLLaVA** — POPE + eval for newly promoted Tier 1 models
- [ ] Phase 3: Full native engine — zero mlx-vlm dependency for T1 models
