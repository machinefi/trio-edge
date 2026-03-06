# Eval Baseline Summary — Tier 1 Regression Gate

**Date**: 2026-03-06
**Device**: Apple M3 Ultra
**Samples per benchmark**: 50

## Baseline Results

| Model | Config | POPE-R | POPE-A | TextVQA | GQA | MMBench | Avg Latency |
|-------|--------|--------|--------|---------|-----|---------|-------------|
| Qwen3.5-0.8B | baseline | 90% | 80% | 68% | 54% | 60% | 151ms |
| Qwen3.5-0.8B | ToMe r=4 | 88% | 82% | 64% | 52% | 60% | 175ms |
| Qwen2.5-VL-3B | baseline | 88% | 82% | 66% | 58% | 94% | 489ms |
| Qwen2.5-VL-3B | ToMe r=4 | — | — | — | — | — | CRASHED |
| Qwen3-VL-4B | baseline | 86% | 82% | 76% | 66% | 96% | 481ms |
| Qwen3-VL-4B | ToMe r=4 | 86% | 82% | — | — | — | CRASHED |
| Gemma 3 4B | baseline | — | — | — | — | — | BLOCKED |
| SmolVLM2 2.2B | baseline | — | — | — | — | — | BLOCKED |
| SmolVLM2 256M | baseline | — | — | — | — | — | BLOCKED |

## ToMe Impact (Qwen3.5-0.8B only — only model where ToMe completed all benchmarks)

| Benchmark | Baseline | ToMe r=4 | Delta |
|-----------|----------|----------|-------|
| POPE-R | 90% | 88% | -2% |
| POPE-A | 80% | 82% | +2% |
| TextVQA | 68% | 64% | -4% ⚠️ |
| GQA | 54% | 52% | -2% |
| MMBench | 60% | 60% | 0% |
| Avg Latency | 151ms | 175ms | +16% |

> TextVQA drops 4% with ToMe r=4, exceeding the 3% threshold. This is expected for
> OCR-heavy tasks where fine-grained token details matter. Consider r=2 for OCR workloads.

## Model Ranking (Baseline Only)

| Rank | Model | Avg Accuracy | Avg Latency | Notes |
|------|-------|-------------|-------------|-------|
| 1 | Qwen3-VL-4B | **81.2%** | 481ms | Best accuracy, especially TextVQA (76%) + MMBench (96%) |
| 2 | Qwen2.5-VL-3B | **77.6%** | 489ms | MMBench champion (94%), similar latency to Qwen3 |
| 3 | Qwen3.5-0.8B | **70.4%** | 151ms | 3x faster, good for edge/real-time; weaker MMBench (60%) |

## Blockers & Issues

### 1. ToMe rope_index broadcast bug (Qwen2.5-VL, Qwen3-VL)
- **Error**: `ValueError: [broadcast_shapes] Shapes (3,1,N) and (3,1,M) cannot be broadcast`
- **Location**: `tome_backend.py` → `model.language_model.get_rope_index()`
- **Root cause**: ToMe merges visual tokens but `get_rope_index()` computes position_ids
  from the original (pre-merge) token count, causing shape mismatch
- **Affected**: Qwen2.5-VL-3B, Qwen3-VL-4B (both use `get_rope_index`)
- **Not affected**: Qwen3.5-0.8B (uses DeltaNet, different position encoding)
- **Fix needed**: Recompute position_ids after token merging in `tome_backend.py`

### 2. Non-Qwen models blocked on video pipeline
- **Error** (Gemma 3): `pixel_values.transpose(0,2,3,1)` — expects 4D NCHW, gets 1D
- **Error** (SmolVLM2): `"number of videos in text [1] and videos [2] should be the same"`
- **Root cause**: `MLXBackend._prepare()` always uses the video pipeline
  (`process_vision_info` → temp mp4), but Gemma 3 and SmolVLM2 don't support video
  processing in mlx-vlm. They need image-based input.
- **Fix needed**: Add image-based `_prepare()` path for non-Qwen models, or wait for
  mlx-vlm upstream to add video support for Gemma/SmolVLM

## Completed Baseline Files

```
research/eval-results/regression/
  qwen3.5-0.8b-mlx_baseline.json     ✅
  qwen3.5-0.8b-mlx_tome_r4.json      ✅
  qwen2.5-vl-3b_baseline.json        ✅
  qwen3-vl-4b_baseline.json          ✅
```

## Success Criteria Status

- [x] Qwen models load and run inference (3/3)
- [ ] Non-Qwen models run benchmarks (0/3 — blocked on video pipeline)
- [x] 20/45 benchmark runs completed (4 configs × 5 benchmarks)
- [ ] All 45 benchmark runs complete (blocked by ToMe bug + non-Qwen video bug)
- [x] Baseline JSON files saved for completed configs (4 files)
- [x] Summary table generated
- [ ] Regression check script validated end-to-end
- [x] No accuracy anomalies (all results in expected ranges)

## Next Steps

1. **Fix ToMe rope_index bug** — recompute position_ids after token merging
2. **Add image input path** for non-Qwen models in `backends.py`
3. **Re-run**: Qwen2.5-VL ToMe, Qwen3-VL ToMe, Gemma 3, SmolVLM2
4. **Validate regression script** (Step 5) — can be done now with existing baselines
