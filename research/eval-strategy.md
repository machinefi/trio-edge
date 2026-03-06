# Evaluation Strategy — Benchmark Selection for TrioCore

## Goal

Select a comprehensive set of evaluation benchmarks to serve as regression gates
across Phase 1/2/3 engine changes. Must cover all capability axes that token
compression and generate-loop changes could affect.

## Research: What Others Use

### Model Technical Reports

| Model | Benchmarks Used |
|-------|----------------|
| **Qwen2.5-VL** | MMMU, MMBench, DocVQA, OCRBench, InfoVQA, ChartQA, TextVQA, GQA, POPE, MME, AI2D, RealWorldQA, MathVista, Video-MME, MVBench |
| **Qwen3-VL** | MMMU, MMBench, DocVQA, OCRBench, ChartQA, TextVQA, MathVista, Video-MME, MLVU |
| **SmolVLM2** | OCRBench, AI2D, ChartQA, TextVQA, DocVQA, ScienceQA, MMMU, MathVista, MMStar, Video-MME, MLVU, MVBench, WorldSense, TempCompass |
| **Gemma 3** | MMMU, MMBench, DocVQA, ChartQA, TextVQA, AI2D, MathVista, RealWorldQA |

### Token Compression Papers

| Paper | Benchmarks Used |
|-------|----------------|
| **AIM** (ICCV 2025) | GQA, POPE, ScienceQA, VizWiz, MME, MMBench, TextVQA, MLVU, Video-MME |
| **LLaVolta** (NeurIPS 2024) | GQA, POPE, ScienceQA, VizWiz, TextVQA, MME, MMBench |
| **PVC** (CVPR 2025) | POPE, MME, MMBench, GQA, TextVQA, DocVQA, ChartQA, OCRBench, Video-MME, MLVU, MVBench |
| **FlashVLM** (2025) | GQA, POPE, TextVQA, MME, MMBench, DocVQA, OCRBench, ChartQA |
| **VTC-Bench** (2025) | GQA, POPE, MME, MMBench, MMStar, OCRBench, ChartQA |

### Eval Toolkits

| Toolkit | Coverage |
|---------|----------|
| **VLMEvalKit** (OpenCompass) | 80+ benchmarks, 220+ models |
| **lmms-eval** (EvolvingLMMs) | 100+ tasks across text/image/video/audio |

### VTC-Bench Finding (Critical)

The paper "Are We Using the Right Benchmark" (2025) found that **simple image
downsampling outperforms many advanced token compression methods** on standard
benchmarks like GQA, MMBench, POPE. This means these benchmarks contain too
many "easy" samples that don't actually test compression quality.

Their recommendation: focus on **resolution-sensitive** benchmarks (OCRBench,
ChartQA, DocVQA) where compression actually matters, plus use their VTC-Bench
framework to filter "difficult" samples from standard benchmarks.

## Benchmark Frequency Analysis

Counting appearances across all sources above:

| Benchmark | Count | Category |
|-----------|-------|----------|
| **POPE** | 7 | Hallucination (yes/no) |
| **TextVQA** | 7 | OCR / text reading |
| **MMBench** | 7 | General multi-ability |
| **MME** | 6 | General perception + cognition |
| **GQA** | 6 | Visual reasoning (compositional) |
| **OCRBench** | 6 | OCR (resolution-sensitive) |
| **ChartQA** | 6 | Chart understanding (resolution-sensitive) |
| **DocVQA** | 5 | Document understanding (resolution-sensitive) |
| **MMMU** | 5 | Expert-level multi-discipline |
| **MathVista** | 4 | Mathematical reasoning with images |
| **ScienceQA** | 4 | Science QA with diagrams |
| **Video-MME** | 5 | Video understanding |
| **MLVU** | 3 | Long video understanding |
| **AI2D** | 3 | Diagram understanding |
| **MVBench** | 3 | Video temporal reasoning |

## Recommended Benchmark Suite for TrioCore

### Selection Criteria

1. **Coverage**: Must test all capability axes that our changes could break
2. **Speed**: Must be runnable in reasonable time (< 30 min for regression gate)
3. **Relevance**: Prioritize benchmarks sensitive to token compression
4. **Availability**: Must be freely available on HuggingFace

### Tier 1 — Regression Gate (run before every phase transition)

These run fast (~50 samples each, ~5 min total) and catch major regressions.

**Priority: real-world scene understanding** — TrioCore targets camera monitoring,
webcam analysis, and physical environment understanding. OCR/document benchmarks
are secondary.

| Benchmark | Tests | Why Critical |
|-----------|-------|-------------|
| **POPE-random** | Object hallucination | Baseline object recognition in real scenes |
| **POPE-adversarial** | Hallucination resistance | Robustness under adversarial objects |
| **GQA** | Compositional visual reasoning | Real-world scene understanding on COCO images — spatial relations, attributes, counting |
| **TextVQA** | OCR / text in images | Compression-sensitive (fine detail) |
| **MMBench** | 20 ability dimensions | Broadest general-purpose VLM benchmark |

### Tier 2 — Full Eval (run after each phase completes)

Run with more samples (~200 each, ~1-2 hours total):

| Benchmark | Tests | Why |
|-----------|-------|-----|
| **OCRBench** | Structured OCR | Most compression-sensitive per VTC-Bench |
| **ChartQA** | Chart/plot reading | Resolution-sensitive, fine detail |
| **DocVQA** | Document understanding | Real-world OCR use case |
| **MME** | Perception + cognition split | Standard in all papers |
| **ScienceQA** | Diagram + science reasoning | Different knowledge axis |
| **MathVista** | Math with visual input | Tests reasoning depth |

### Tier 3 — Video Benchmarks (when video pipeline changes)

| Benchmark | Tests | Why |
|-----------|-------|-----|
| **Video-MME** | Comprehensive video understanding | Gold standard, used by all model reports |
| **MLVU** | Long video comprehension | Tests temporal token handling |
| **MVBench** | Temporal reasoning | Fine-grained video QA |

### Not Included (and why)

| Benchmark | Reason |
|-----------|--------|
| MMMU | Expert-level, overkill for edge models, slow |
| VizWiz | Accessibility-focused, niche |
| AI2D | Covered by ScienceQA + ChartQA |
| RealWorldQA | Too easy, mostly saturated (consider for future if we need more scene benchmarks) |
| InfoVQA | Covered by DocVQA |

## Implementation Plan

### Phase 0 (now): Integrate Tier 1

1. Add OCRBench and ChartQA to `eval_benchmarks.py`
2. Update `run_regression.py` to run all 5 Tier 1 benchmarks
3. Save baselines for Qwen2.5-VL-3B and Qwen3-VL-4B
4. Save baseline for ToMe r=4 configurations

### Phase 0.5 (before Phase 1): Integrate Tier 2

5. Add MMBench, GQA, DocVQA, MME, ScienceQA, MathVista
6. Create `run_full_eval.py` for Tier 2 runs
7. Save full baselines for all model × config combinations

### Later: Integrate Tier 3

8. Video benchmarks when video pipeline changes in Phase 1+

## Baseline Matrix

Configurations to baseline:

| Model | Baseline | ToMe r=4 |
|-------|----------|----------|
| Qwen2.5-VL-3B | Tier 1 + Tier 2 | Tier 1 + Tier 2 |
| Qwen3-VL-4B | Tier 1 + Tier 2 | Tier 1 + Tier 2 |
| Qwen3.5-0.8B | Tier 1 | Tier 1 |

Total: 6 baseline configurations × 5 Tier 1 benchmarks = 30 baseline data points (minimum).

## References

- [VTC-Bench paper](https://arxiv.org/abs/2510.07143) — "Are We Using the Right Benchmark" (2025)
- [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) — 80+ benchmarks, 220+ models
- [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) — 100+ tasks
- [Qwen2.5-VL tech report](https://arxiv.org/abs/2502.13923)
- [Qwen3-VL tech report](https://arxiv.org/abs/2511.21631)
- [SmolVLM2 blog](https://huggingface.co/blog/smolvlm2)
- [AIM paper](https://arxiv.org/abs/2412.03248) — ICCV 2025
- [LLaVolta paper](https://arxiv.org/abs/2406.20092) — NeurIPS 2024
- [PVC paper](https://arxiv.org/abs/2412.09613) — CVPR 2025
- [Awesome Multimodal Token Compression](https://github.com/cokeshao/Awesome-Multimodal-Token-Compression) — TMLR 2026 survey
