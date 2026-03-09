#!/usr/bin/env python3
"""Run SurveillanceVQA detection baseline on all T1 VLM models.

Usage:
    PYTHONPATH=src python3 examples/run_surveillance_baseline.py

Runs detection (yes/no anomaly) benchmark on all available T1 VLM models,
using whatever UCF-Crime video data is available locally.
"""

import gc
import json
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def main():
    from trio_core.bench import (
        MODEL_REGISTRY, OPTIM_PRESETS, build_engine, _run_mlxvlm_baseline,
        get_benchmark, ResultStore, detect_anomalies,
    )
    from trio_core.eval_benchmarks import BenchmarkRunner

    # Models to benchmark (T1 VLMs only, skip InternVL3 and text-only)
    VLM_MODELS = [
        "qwen2.5-vl-3b",
        "qwen2.5-vl-7b",
        "qwen3-vl-2b",
        "qwen3-vl-4b",
        "qwen3-vl-8b",
        "qwen3.5-0.8b",
        "qwen3.5-2b",
        "qwen3.5-4b",
        "qwen3.5-9b",
    ]

    data_dir = Path("surveillance_vqa")
    video_dir = data_dir / "videos"
    results_file = data_dir / "baseline_results_full.json"
    config = OPTIM_PRESETS["baseline"]
    store = ResultStore(Path("research/bench-results"))

    # Load existing results to skip completed models
    existing = {}
    if results_file.exists():
        with open(results_file) as f:
            existing = json.load(f)

    all_results = dict(existing)

    for model_key in VLM_MODELS:
        if model_key in existing and "error" not in existing[model_key]:
            print(f"\n{'='*60}")
            print(f"  SKIP {model_key} (already completed)")
            print(f"{'='*60}")
            continue

        model = MODEL_REGISTRY.get(model_key)
        if not model:
            print(f"Model {model_key} not in registry, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"  RUNNING: {model_key}")
        print(f"{'='*60}")

        try:
            # Build engine
            print(f"  Loading model...")
            t_load = time.perf_counter()
            engine, is_mlxvlm = build_engine(model, config, max_tokens=16)
            print(f"  Loaded in {time.perf_counter() - t_load:.1f}s")

            # Create benchmark (no max_samples = full test set)
            benchmark = get_benchmark(
                "surveillance_vqa",
                max_samples=None,
                surveillance_dir=str(data_dir),
                surveillance_video_dir=str(video_dir),
                surveillance_qa_type="detection",
            )

            # Run benchmark
            print(f"  Running benchmark...")
            if is_mlxvlm:
                result = _run_mlxvlm_baseline(engine, benchmark, max_tokens=16)
            else:
                runner = BenchmarkRunner(engine, max_tokens=16)
                result = runner.run(benchmark)

            result.print()

            # Save to bench-results
            filepath = store.save(model_key, "surveillance_vqa_detection", config.name, result)
            print(f"  Saved: {filepath}")

            # Check anomalies
            anomalies = detect_anomalies(result, "surveillance_vqa")
            if anomalies:
                print(f"  ANOMALIES: {anomalies}")

            # Record summary
            all_results[model_key] = {
                "accuracy": result.accuracy,
                "f1": result.f1,
                "yes_rate": result.yes_rate,
                "recall": result.recall,
                "specificity": result.specificity,
                "avg_latency_ms": result.avg_latency_ms,
                "n_samples": result.n_samples,
            }

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[model_key] = {"error": str(e)}

        finally:
            # Clean up to free memory
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

        # Save after each model
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"  Results saved to {results_file}")

    # Print summary table
    print(f"\n\n{'='*80}")
    print(f"  SURVEILLANCE VQA DETECTION BASELINE RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Acc':>6} {'F1':>6} {'Recall':>8} {'Spec':>6} {'YRate':>6} {'Lat(ms)':>8} {'N':>5}")
    print(f"{'-'*80}")
    for model_key in VLM_MODELS:
        if model_key not in all_results:
            continue
        r = all_results[model_key]
        if "error" in r:
            print(f"{model_key:<20} ERROR: {r['error'][:50]}")
        else:
            print(f"{model_key:<20} {r['accuracy']:>5.1%} {r['f1']:>6.3f} {r['recall']:>7.1%} {r['specificity']:>5.1%} {r['yes_rate']:>5.1%} {r['avg_latency_ms']:>7.1f} {r['n_samples']:>5}")


if __name__ == "__main__":
    main()
