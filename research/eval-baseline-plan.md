# Eval Baseline Plan — Tier 1 Regression Gate

## Goal

Collect Tier 1 benchmark baselines across all supported models before starting
Phase 1 engine changes. These baselines serve as regression gates — any Phase 1/2/3
change must not drop accuracy below threshold (default: 3%).

## Models

### Qwen Family (3 models — ToMe supported)

| # | Model ID | Profile | Size | Notes |
|---|----------|---------|------|-------|
| 1 | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | qwen2.5-vl-3b | ~2GB | Primary dev model, most benchmarked |
| 2 | `mlx-community/Qwen3-VL-4B-Instruct-4bit` | qwen3-vl-4b | ~2.5GB | Zero quality loss with ToMe |
| 3 | `mlx-community/Qwen3.5-VL-0.8B-Instruct-4bit` | qwen3.5-0.8b | ~0.6GB | Smallest, edge target |

### Non-Qwen Family (2-3 models — baseline only, no ToMe)

| # | Model ID | Profile | Size | Notes |
|---|----------|---------|------|-------|
| 4 | `mlx-community/gemma-3-4b-it-4bit` | gemma3-4b | ~2.5GB | Google, SigLIP ViT, 256 fixed tokens |
| 5 | `mlx-community/SmolVLM2-2.2B-Instruct` | smolvlm-2.2b | ~1.2GB | HuggingFace, smallest multi-modal |
| 6 | `mlx-community/SmolVLM2-256M-Video-Instruct` | smolvlm-256m | ~0.3GB | Ultra-small, stress test (optional) |

**Total: 5-6 models**

> Non-Qwen models run baseline only. ToMe wrappers only exist for Qwen2.5-VL,
> Qwen3-VL, and Qwen3.5 currently.

## Benchmarks (Tier 1)

| # | Benchmark | Type | Metric | Samples | Est. Time/Model |
|---|-----------|------|--------|---------|-----------------|
| 1 | POPE-random | Object hallucination (yes/no) | Accuracy, F1 | 50 | ~1 min |
| 2 | POPE-adversarial | Hallucination resistance | Accuracy, F1 | 50 | ~1 min |
| 3 | TextVQA | OCR / text reading | Accuracy | 50 | ~3 min |
| 4 | GQA | Real-world visual reasoning | Accuracy | 50 | ~2 min |
| 5 | MMBench | Multi-ability (20 dimensions) | Accuracy | 50 | ~2 min |

**Per model: ~9 min × 50 samples = ~9 min**

## Baseline Matrix

### Configurations to Run

| Model | Baseline | ToMe r=4 | Total Runs |
|-------|----------|----------|------------|
| Qwen2.5-VL-3B | 5 benchmarks | 5 benchmarks | 10 |
| Qwen3-VL-4B | 5 benchmarks | 5 benchmarks | 10 |
| Qwen3.5-0.8B | 5 benchmarks | 5 benchmarks | 10 |
| Gemma 3 4B | 5 benchmarks | — | 5 |
| SmolVLM2 2.2B | 5 benchmarks | — | 5 |
| SmolVLM2 256M | 5 benchmarks | — | 5 |
| **Total** | | | **45 runs** |

**Estimated total time: ~80 min** (including model load overhead)

### Output Format

Each run saves to `research/eval-results/regression/<config_key>.json`:

```
research/eval-results/regression/
  qwen2.5-vl-3b_baseline.json
  qwen2.5-vl-3b_tome_r4.json
  qwen3-vl-4b_baseline.json
  qwen3-vl-4b_tome_r4.json
  qwen3.5-0.8b_baseline.json
  qwen3.5-0.8b_tome_r4.json
  gemma3-4b_baseline.json
  smolvlm2-2.2b_baseline.json
  smolvlm2-256m_baseline.json
```

## Execution Plan

### Step 1: Verify Non-Qwen Models Load

Before running full baselines, verify each non-Qwen model loads and runs
a single inference correctly:

```bash
# Gemma 3
.venv/bin/python -c "
from trio_core import TrioCore, EngineConfig
e = TrioCore(EngineConfig(model='mlx-community/gemma-3-4b-it-4bit'))
e.load()
print(e.health())
"

# SmolVLM2 2.2B
.venv/bin/python -c "
from trio_core import TrioCore, EngineConfig
e = TrioCore(EngineConfig(model='mlx-community/SmolVLM2-2.2B-Instruct'))
e.load()
print(e.health())
"
```

If any model fails to load, fix the backend/profile integration first.

### Step 2: Run Qwen Baselines

These are the most important — they have existing partial baselines to compare.

```bash
# Qwen2.5-VL-3B baseline
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit

# Qwen2.5-VL-3B + ToMe r=4
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit --tome 4

# Qwen3-VL-4B baseline
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen3-VL-4B-Instruct-4bit

# Qwen3-VL-4B + ToMe r=4
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen3-VL-4B-Instruct-4bit --tome 4

# Qwen3.5-0.8B baseline
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen3.5-VL-0.8B-Instruct-4bit

# Qwen3.5-0.8B + ToMe r=4
python examples/run_regression.py --save-baseline \
  -m mlx-community/Qwen3.5-VL-0.8B-Instruct-4bit --tome 4
```

### Step 3: Run Non-Qwen Baselines

```bash
# Gemma 3 4B
python examples/run_regression.py --save-baseline \
  -m mlx-community/gemma-3-4b-it-4bit

# SmolVLM2 2.2B
python examples/run_regression.py --save-baseline \
  -m mlx-community/SmolVLM2-2.2B-Instruct

# SmolVLM2 256M (optional — stress test)
python examples/run_regression.py --save-baseline \
  -m mlx-community/SmolVLM2-256M-Video-Instruct
```

### Step 4: Generate Summary Report

After all baselines are collected, generate a cross-model comparison table:

| Model | Config | POPE-R | POPE-A | TextVQA | GQA | MMBench | Avg Latency |
|-------|--------|--------|--------|---------|-----|---------|-------------|
| Qwen2.5-VL-3B | baseline | | | | | | |
| Qwen2.5-VL-3B | ToMe r=4 | | | | | | |
| Qwen3-VL-4B | baseline | | | | | | |
| Qwen3-VL-4B | ToMe r=4 | | | | | | |
| Qwen3.5-0.8B | baseline | | | | | | |
| Qwen3.5-0.8B | ToMe r=4 | | | | | | |
| Gemma 3 4B | baseline | | | | | | |
| SmolVLM2 2.2B | baseline | | | | | | |
| SmolVLM2 256M | baseline | | | | | | |

Save as `research/eval-results/regression/SUMMARY.md`.

### Step 5: Validate Regression Script

Run a non-save regression check to verify the comparison logic works:

```bash
python examples/run_regression.py \
  -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit
# Should show "ALL PASSED" (comparing against just-saved baseline)
```

## Potential Issues

1. **Non-Qwen model loading** — mlx-vlm may not support all model IDs; need to
   verify exact HuggingFace IDs that work
2. **GQA dataset size** — first download is ~2GB (images), subsequent runs use cache
3. **MMBench format** — multiple-choice; small models may not follow instructions
   well, leading to low accuracy (this is expected, not a bug)
4. **Memory** — Gemma 3 12B won't fit on 18GB M3 Pro; stick with 4B
5. **SmolVLM2 answer format** — very small models may produce verbose/unexpected
   answers; may need per-model prompt tuning later

## Success Criteria

- [ ] All 5-6 models load and run inference successfully
- [ ] All 45 benchmark runs complete without errors
- [ ] Baseline JSON files saved for each model × config
- [ ] Summary table generated with cross-model comparison
- [ ] Regression check script works end-to-end (save → compare → pass/fail)
- [ ] No accuracy anomalies that indicate benchmark bugs (e.g., 0% or 100%)

## After This

Once baselines are solid, proceed to **Phase 1: Custom Generate Loop** with
confidence that any regression will be caught by:

```bash
python examples/run_regression.py -m <model>
# Compares against saved baseline, exits 1 if accuracy drops >3%
```
