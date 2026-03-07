# trio-core Research

Research notes on visual token compression for edge VLM inference.

## Pipeline Big Picture

Every VLM inference decomposes into 5 stages. This is the map we optimize against:

```
输入 → [Vision Encoder + ToMe] → visual tokens → [LLM Prefill] → [KV Cache] → [Decode]
        ^^^^^^^^^^^^^^^^^^^      ^^^^^^^^^^^      ^^^^^^^^^^^^     ^^^^^^^^^    ^^^^^^^^
        ToMe 在这里              更少了            量化在这里        缓存在这里    batch在这里
```

### Per-Stage Status (updated 2026-03-06)

```
Stage 0: 输入 (Video Pipeline)      ██████████ 100%   done
Stage 1: Vision Encoder + ToMe      ██████████ 100%   ToMe + adaptive r + native ViT done
Stage 2: Visual Token Count          ████████░░  80%   mid-stream FastV done; adaptive ratio TODO
Stage 3: LLM Prefill                 ██████░░░░  60%   generate loop + prefix cache done; mlx-vlm dep remaining
Stage 4: KV Cache                    ████░░░░░░  40%   persistent cache + prefix reuse done; frame-to-frame TODO
Stage 5: Decode                      ████░░░░░░  40%   streaming + speculative decode done; benchmark pending
```

### Priority Ranking (ROI for video inference latency)

| # | What | Expected Gain | Difficulty | Stage | Status |
|---|------|---------------|------------|-------|--------|
| 1 | Frame-to-frame KV reuse | -60~80% video latency | High | 4 | TODO |
| 2 | ~~Mid-stream FastV (true KV prune)~~ | ~~-30~50% visual tokens~~ | ~~Medium~~ | ~~2~~ | DONE |
| 3 | ~~Shared text prefix KV~~ | ~~-20~40% prefill~~ | ~~Medium~~ | ~~3~~ | DONE |
| 4 | ~~Speculative decoding~~ | ~~+30~50% decode TPS~~ | ~~Medium~~ | ~~5~~ | DONE (0% accept for VLM) |
| 5 | Content-aware adaptive r | +quality, same speed | Low | 2 | TODO |
| 6 | ~~Native ToMe (no monkey-patch)~~ | ~~cleaner arch~~ | ~~Medium~~ | ~~1~~ | DONE |
| 7 | Remove mlx-vlm load dep | zero dependency | High | 3 | TODO |

### Stage Details

**Stage 0 — Input (Video Pipeline)** -- DONE
- StreamCapture (webcam/RTSP/YouTube), temporal dedup (-30~70% frames),
  motion gate (-80%+ VLM calls), smart resize, model profiles

**Stage 1 — Vision Encoder + ToMe** -- DONE
- Done: ToMe bipartite soft matching, windowed-attn aware, adaptive r ramp,
  hidden-state metric, Qwen2.5/3/3.5 support,
  native ViT (NativeToMeQwen25Vision / NativeToMeQwen3Vision — proper OO, no monkey-patch)

**Stage 2 — Visual Token Count** -- 80%
- Done: ToMe compression (Qwen2.5 1080p: 748->242 tokens, -68%),
  post-encoder compression, mid-stream FastV (single-pass KV cache pruning,
  zero double computation, MRoPE-aware)
- TODO: content-aware adaptive ratio

**Stage 3 — LLM Prefill** -- 60%
- Done: own generate loop (sampler, KV cache, logits processors internalized),
  mlx-lm runtime dep mostly removed, chunked prefill, shared text prefix KV cache
  (three-tier: exact hit > prefix hit > full miss), early stopping
- TODO: remove mlx-vlm model loading dep

**Stage 4 — KV Cache** -- 40%
- Done: persistent KVCache with buffer reuse, quantized KV cache,
  text prefix KV reuse across frames (15 tokens saved per inference)
- TODO: frame-to-frame KV reuse (consecutive frames share 80%+ context),
  streaming KV eviction for long video, cross-frame visual KV sharing

**Stage 5 — Decode** -- 40%
- Done: auto-regressive, streaming output, early stopping config,
  speculative decoding (draft model + prompt lookup, rejection sampling,
  KV cache rollback), integrated into generate loop via `speculative_lookahead` config
- Note: prompt lookup confirmed 0% acceptance for VLM (visual tokens ≠ text output).
  Draft model mode ready but needs separate model loading.
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
  cache type        KVCache          KVCache            KVCache+Delta    KVCache       KVCache

Stage 5: Decode
  eos handling      same             same               same             different     different
  stopping_criteria same             same               same             different     different
```

Key takeaway: Qwen family shares the same LLM layer loop (Stage 3-5).
Differences are in Stage 1-2 (ViT architecture, token merge signature),
already abstracted via wrappers + `_is_qwen3` flag.
Gemma/SmolVLM have entirely different ViT — ToMe/FastV not yet supported.

## Documents

- [visual-token-compression.md](visual-token-compression.md) — Core research direction: three key papers, prototype results, compatibility analysis with KV cache / batch scheduling / quantization.
- [tome-implementation-plan.md](tome-implementation-plan.md) — Detailed step-by-step plan for implementing ToMe in Qwen2.5-VL's vision encoder.
- [native-engine-plan.md](native-engine-plan.md) — Phased plan to replace mlx-vlm dependency: per-stage analysis, core metrics projections, Phase 1/2/3 execution plan.
- [eval-strategy.md](eval-strategy.md) — Evaluation benchmark selection: research across model reports, compression papers, and eval toolkits. Tiered benchmark suite (Tier 1 regression gate, Tier 2 full eval, Tier 3 video).
- [eval-baseline-plan.md](eval-baseline-plan.md) — Tier 1 baseline collection plan: 6 models × 5 benchmarks × baseline/ToMe configs.
- [phase1-custom-generate.md](phase1-custom-generate.md) — Phase 1 detailed implementation: custom generate loop, persistent KV cache, early stopping. Code-level design with mlx-vlm source analysis.
- [eval-results/mlxvlm-native-baselines.md](eval-results/mlxvlm-native-baselines.md) — mlx-vlm native baselines: ground-truth comparison (4 models × 5 benchmarks) showing trio-core adds no meaningful accuracy or latency overhead.
- [eval-results/speculative-decode-benchmark.md](eval-results/speculative-decode-benchmark.md) — Speculative decode (prompt lookup) benchmark: 0% acceptance rate across 3 scenarios, confirms prompt lookup is not useful for VLM inference.
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

1. **Frame-to-frame KV reuse** — consecutive video frames share 80%+ context (highest ROI, hardest)
2. **ToMe + FastV compound benchmark** — measure combined visual compression (both done, not measured together)
3. **Unify ToMe generate path** — ToMeMLXBackend should use generate_step (get prefix cache, speculative, early stop for free)
4. **Content-aware adaptive r** — quality-preserving compression per image
5. **Remove mlx-vlm model loading dep** — zero external dependency (Phase 3)

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
- [x] **Speculative decoding** — prompt lookup + draft model modes, rejection sampling, KV cache rollback, integrated into generate loop (0% accept for VLM — visual tokens don't match text output)
- [x] **mlx-vlm native baselines** — 4 models × 5 benchmarks, trio-core matches or beats native on 3B+
- [x] **Native ToMe ViT** — NativeToMeQwen25Vision / NativeToMeQwen3Vision, proper OO subclass, no monkey-patch
- [ ] ToMe + FastV compound benchmark — measure combined compression
- [ ] Phase 2: Native Vision Encoder — built-in ToMe, adaptive r
- [ ] Phase 3: Full native engine — zero mlx-vlm dependency
