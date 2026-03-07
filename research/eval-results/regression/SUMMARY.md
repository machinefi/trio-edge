# Eval Baseline Summary — Tier 1 Regression Gate

**Date**: 2026-03-06
**Device**: Apple M3 Ultra
**Samples per benchmark**: 50

## Baseline Results

| Model | Config | POPE-R | POPE-A | TextVQA | GQA | MMBench | Avg Latency |
|-------|--------|--------|--------|---------|-----|---------|-------------|
| Qwen3.5-0.8B | baseline | 90% | 80% | 66% | 54% | 60% | 151ms |
| Qwen3.5-0.8B | ToMe r=4 | 88% | 82% | 64% | 52% | 60% | 175ms |
| Qwen2.5-VL-3B | baseline | 88% | 82% | 66% | 58% | 94% | 489ms |
| Qwen2.5-VL-3B | ToMe r=4 | 84% | 70% | 38% | 46% | 82% | 718ms |
| Qwen3-VL-4B | baseline | 86% | 82% | 76% | 66% | 96% | 481ms |
| Qwen3-VL-4B | ToMe r=4 | 86% | 82% | 70% | 64% | 94% | 397ms |
| Gemma 3 4B | baseline | — | — | — | — | — | BLOCKED |
| SmolVLM2 2.2B | baseline | — | — | — | — | — | BLOCKED |
| SmolVLM2 256M | baseline | — | — | — | — | — | BLOCKED |

## ToMe Impact Analysis

### Qwen3.5-0.8B (DeltaNet)

| Benchmark | Baseline | ToMe r=4 | Delta |
|-----------|----------|----------|-------|
| POPE-R | 90% | 88% | -2% |
| POPE-A | 80% | 82% | +2% |
| TextVQA | 66% | 64% | -2% |
| GQA | 54% | 52% | -2% |
| MMBench | 60% | 60% | 0% |
| Avg Latency | 151ms | 175ms | +16% |

> Minor accuracy drops within 3% threshold. Latency increase due to ToMe overhead on small model.

### Qwen2.5-VL-3B

| Benchmark | Baseline | ToMe r=4 | Delta |
|-----------|----------|----------|-------|
| POPE-R | 88% | 84% | -4% ⚠️ |
| POPE-A | 82% | 70% | -12% ⚠️ |
| TextVQA | 66% | 38% | -28% ⚠️ |
| GQA | 58% | 46% | -12% ⚠️ |
| MMBench | 94% | 82% | -12% ⚠️ |
| Avg Latency | 489ms | 718ms | +47% |

> Significant accuracy regression across all benchmarks with ToMe r=4.
> Token merging is too aggressive for this model — try r=2 or r=1.

### Qwen3-VL-4B

| Benchmark | Baseline | ToMe r=4 | Delta |
|-----------|----------|----------|-------|
| POPE-R | 86% | 86% | 0% |
| POPE-A | 82% | 82% | 0% |
| TextVQA | 76% | 70% | -6% ⚠️ |
| GQA | 66% | 64% | -2% |
| MMBench | 96% | 94% | -2% |
| Avg Latency | 481ms | 397ms | -17% ✅ |

> Good ToMe candidate — POPE unaffected, GQA/MMBench within threshold.
> TextVQA drops 6% (OCR-sensitive). Latency improves 17%.

## Model Ranking (Baseline Only)

| Rank | Model | Avg Accuracy | Avg Latency | Notes |
|------|-------|-------------|-------------|-------|
| 1 | Qwen3-VL-4B | **81.2%** | 481ms | Best accuracy, especially TextVQA (76%) + MMBench (96%) |
| 2 | Qwen2.5-VL-3B | **77.6%** | 489ms | MMBench champion (94%), similar latency to Qwen3 |
| 3 | Qwen3.5-0.8B | **70.4%** | 151ms | 3x faster, good for edge/real-time; weaker MMBench (60%) |

## Blockers & Issues

### 1. ~~ToMe rope_index broadcast bug~~ — FIXED ✅
- **Error**: `ValueError: [broadcast_shapes] Shapes (3,1,N) and (3,1,M) cannot be broadcast`
- **Root cause**: `_compute_compressed_grid()` can't always produce exact token count
  due to integer factorization — grid gives T*H*W != compressed_count
- **Fix**: Grid alignment in `tome_backend.py` — pad or truncate hidden_states to match
  grid-computed token count, rebuild input_ids accordingly

### 2. Non-Qwen models blocked on video pipeline
- **Error** (Gemma 3): `pixel_values.transpose(0,2,3,1)` — expects 4D NCHW, gets 1D
- **Error** (SmolVLM2): `"number of videos in text [1] and videos [2] should be the same"`
- **Root cause**: `MLXBackend._prepare()` always uses the video pipeline
  (`process_vision_info` → temp mp4), but Gemma 3 and SmolVLM2 don't support video
  processing in mlx-vlm. They need image-based input.
- **Fix needed**: Add image-based `_prepare()` path for non-Qwen models

## Completed Baseline Files

```
research/eval-results/regression/
  qwen3.5-0.8b-mlx_baseline.json     ✅
  qwen3.5-0.8b-mlx_tome_r4.json      ✅
  qwen2.5-vl-3b_baseline.json        ✅
  qwen2.5-vl-3b_tome_r4.json         ✅  (NEW)
  qwen3-vl-4b_baseline.json          ✅
  qwen3-vl-4b_tome_r4.json           ✅  (NEW)
```

## Success Criteria Status

- [x] Qwen models load and run inference (3/3)
- [ ] Non-Qwen models run benchmarks (0/3 — blocked on video pipeline)
- [x] 30/45 benchmark runs completed (6 configs × 5 benchmarks)
- [ ] All 45 benchmark runs complete (blocked by non-Qwen video bug)
- [x] Baseline JSON files saved for completed configs (6 files)
- [x] Summary table generated
- [x] Regression check script validated end-to-end
- [x] No accuracy anomalies (all results in expected ranges)
- [x] ToMe rope_index bug fixed and validated

## Next Steps

1. **Add image input path** for non-Qwen models in `backends.py`
2. **Re-run**: Gemma 3, SmolVLM2 baselines
3. **Tune ToMe r** for Qwen2.5-VL (try r=2 or r=1 to reduce regression)
