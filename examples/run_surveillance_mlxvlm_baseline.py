#!/usr/bin/env python3
"""Compare mlx-vlm raw baseline vs TrioCore on SurveillanceVQA.

Only Qwen2.5-VL models work with mlx-vlm 0.1.15 (no qwen3_vl/qwen3_5 support).
"""

import gc
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    from trio_core.bench import (
        MODEL_REGISTRY, OPTIM_PRESETS, build_engine, _run_mlxvlm_baseline,
        get_benchmark, ResultStore, detect_anomalies,
    )

    # Only Qwen2.5-VL works with mlx-vlm 0.1.15
    VLM_MODELS = [
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
    ]

    data_dir = Path("surveillance_vqa")
    video_dir = data_dir / "videos"
    config = OPTIM_PRESETS["mlxvlm_baseline"]
    store = ResultStore(Path("research/bench-results"))

    all_results = {}

    for model_key in VLM_MODELS:
        model = MODEL_REGISTRY[model_key]

        print(f"\n{'='*60}")
        print(f"  MLX-VLM BASELINE: {model_key}")
        print(f"{'='*60}")

        try:
            print(f"  Loading via mlx_vlm.load()...")
            t_load = time.perf_counter()
            engine, is_mlxvlm = build_engine(model, config, max_tokens=16)
            assert is_mlxvlm, "Expected mlxvlm_baseline mode"
            print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

            benchmark = get_benchmark(
                "surveillance_vqa",
                max_samples=None,
                surveillance_dir=str(data_dir),
                surveillance_video_dir=str(video_dir),
                surveillance_qa_type="detection",
            )

            print(f"  Running benchmark...")
            result = _run_mlxvlm_baseline(engine, benchmark, max_tokens=16)
            result.print()

            filepath = store.save(model_key, "surveillance_vqa_detection", "mlxvlm_baseline", result)
            print(f"  Saved: {filepath}")

            all_results[model_key] = {
                "accuracy": result.accuracy,
                "f1": result.f1,
                "recall": result.recall,
                "specificity": result.specificity,
                "yes_rate": result.yes_rate,
                "avg_latency_ms": result.avg_latency_ms,
                "n_samples": result.n_samples,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"error": str(e)}

        finally:
            try:
                del engine
            except NameError:
                pass
            gc.collect()
            try:
                import mlx.core as mx
                mx.metal.clear_cache()
            except Exception:
                pass

    # Print comparison table
    print(f"\n\n{'='*90}")
    print(f"  MLX-VLM vs TRIOCORE — SurveillanceVQA Detection (1827 samples)")
    print(f"{'='*90}")

    # Load TrioCore results for comparison
    trio_results = {}
    if (data_dir / "baseline_results_full.json").exists():
        with open(data_dir / "baseline_results_full.json") as f:
            trio_results = json.load(f)

    print(f"{'Model':<20} {'Backend':<12} {'Acc':>6} {'F1':>6} {'Recall':>7} {'Spec':>6} {'YRate':>6} {'Lat':>7}")
    print(f"{'-'*80}")
    for model_key in VLM_MODELS:
        # TrioCore
        if model_key in trio_results and "error" not in trio_results[model_key]:
            t = trio_results[model_key]
            print(f"{model_key:<20} {'TrioCore':<12} {t['accuracy']:.1%} {t['f1']:.3f} {t['recall']:.1%}  {t['specificity']:.1%} {t['yes_rate']:.1%} {t['avg_latency_ms']:>6.0f}ms")
        # mlx-vlm
        if model_key in all_results and "error" not in all_results[model_key]:
            m = all_results[model_key]
            print(f"{'':<20} {'mlx-vlm':<12} {m['accuracy']:.1%} {m['f1']:.3f} {m['recall']:.1%}  {m['specificity']:.1%} {m['yes_rate']:.1%} {m['avg_latency_ms']:>6.0f}ms")
            # Delta
            if model_key in trio_results and "error" not in trio_results[model_key]:
                t = trio_results[model_key]
                da = m['accuracy'] - t['accuracy']
                df = m['f1'] - t['f1']
                print(f"{'':<20} {'delta':<12} {da:>+5.1%} {df:>+6.3f}")
        print()

    # Save mlxvlm results
    with open(data_dir / "mlxvlm_baseline_results.json", "w") as f:
        json.dump(all_results, f, indent=2)


if __name__ == "__main__":
    main()
