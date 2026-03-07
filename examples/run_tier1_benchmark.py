#!/usr/bin/env python3
"""Comprehensive Tier 1 benchmark: all Qwen VL models, baseline vs TrioCore.

Runs POPE (accuracy) + synthetic eval (performance) for each model.

Usage:
    # Run all Tier 1 models
    python examples/run_tier1_benchmark.py

    # Run specific model only
    python examples/run_tier1_benchmark.py --model qwen3-vl-4b

    # Skip download (only run locally cached models)
    python examples/run_tier1_benchmark.py --no-download

    # POPE only (faster)
    python examples/run_tier1_benchmark.py --pope-only

    # Eval only (no accuracy benchmark)
    python examples/run_tier1_benchmark.py --eval-only

    # With ToMe compression (benchmark TrioCore optimizations)
    python examples/run_tier1_benchmark.py --tome 4
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# All Tier 1 models with their HuggingFace IDs
TIER1_MODELS = {
    # Qwen2.5-VL family
    "qwen2.5-vl-3b": "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "qwen2.5-vl-7b": "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    # Qwen3-VL family
    "qwen3-vl-2b": "mlx-community/Qwen3-VL-2B-Instruct-4bit",
    "qwen3-vl-4b": "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "qwen3-vl-8b": "mlx-community/Qwen3-VL-8B-Instruct-4bit",
    # Qwen3.5 family (DeltaNet hybrid)
    "qwen3.5-0.8b": "mlx-community/Qwen3.5-0.8B-MLX-4bit",
    "qwen3.5-2b": "mlx-community/Qwen3.5-2B-4bit",
    "qwen3.5-4b": "mlx-community/Qwen3.5-4B-MLX-4bit",
    "qwen3.5-9b": "mlx-community/Qwen3.5-9B-MLX-4bit",
    # InternVL3 family
    "internvl3-1b": "mlx-community/InternVL3-1B-4bit",
    "internvl3-2b": "mlx-community/InternVL3-2B-4bit",
    # FastVLM family
    "fastvlm-0.5b": "InsightKeeper/FastVLM-0.5B-MLX-4bit",
    "fastvlm-1.5b": "InsightKeeper/FastVLM-1.5B-MLX-4bit",
    # nanoLLaVA
    "nanollava-1.5": "mlx-community/nanoLLaVA-1.5-4bit",
}

# Order by size (smallest first) for benchmarking
MODEL_ORDER = [
    "fastvlm-0.5b",
    "qwen3.5-0.8b",
    "nanollava-1.5",
    "internvl3-1b",
    "fastvlm-1.5b",
    "qwen3.5-2b",
    "internvl3-2b",
    "qwen3-vl-2b",
    "qwen2.5-vl-3b",
    "qwen3-vl-4b",
    "qwen3.5-4b",
    "qwen2.5-vl-7b",
    "qwen3-vl-8b",
    "qwen3.5-9b",
]


def is_model_cached(hf_id: str) -> bool:
    """Check if model is already downloaded."""
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_dir / f"models--{hf_id.replace('/', '--')}"
    return model_dir.exists()


def download_model(hf_id: str) -> bool:
    """Download model using huggingface-cli."""
    print(f"  Downloading {hf_id}...")
    try:
        subprocess.run(
            ["huggingface-cli", "download", hf_id],
            check=True,
            capture_output=True,
            text=True,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  FAILED to download {hf_id}: {e.stderr[:200]}")
        return False
    except FileNotFoundError:
        print("  huggingface-cli not found. Install: pip install huggingface_hub")
        return False


def run_pope(model_id: str, output_path: str, samples: int = 100, tome: int = 0) -> dict | None:
    """Run POPE benchmark for a model."""
    cmd = [
        sys.executable, "examples/run_benchmark.py",
        "--bench", "pope",
        "--samples", str(samples),
        "--model", model_id,
        "--output", output_path,
    ]
    if tome > 0:
        cmd.extend(["--tome", str(tome)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  POPE failed: {result.stderr[:300]}")
            return None
        # Load results
        if Path(output_path).exists():
            with open(output_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print("  POPE timed out (10 min)")
    except Exception as e:
        print(f"  POPE error: {e}")
    return None


def run_eval(model_id: str, output_path: str, resolution: str = "480p",
             runs: int = 3, tome: int = 0) -> dict | None:
    """Run synthetic eval for a model."""
    cmd = [
        sys.executable, "examples/run_eval.py",
        "--model", model_id,
        "--output", output_path,
        "--resolution", resolution,
        "--runs", str(runs),
    ]
    if tome > 0:
        cmd.extend(["--tome", str(tome)])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"  Eval failed: {result.stderr[:300]}")
            return None
        if Path(output_path).exists():
            with open(output_path) as f:
                return json.load(f)
    except subprocess.TimeoutExpired:
        print("  Eval timed out (10 min)")
    except Exception as e:
        print(f"  Eval error: {e}")
    return None


def print_summary(results: dict):
    """Print a summary table of all benchmark results."""
    print("\n" + "=" * 90)
    print("TIER 1 BENCHMARK SUMMARY")
    print("=" * 90)

    # POPE results
    pope_results = {k: v for k, v in results.items() if v.get("pope")}
    if pope_results:
        print("\n--- POPE (Object Hallucination, 100 samples) ---")
        print(f"{'Model':<20} {'Accuracy':>10} {'F1':>8} {'Avg Latency':>14} {'Vis Tokens':>12}")
        print("-" * 70)
        for name in MODEL_ORDER:
            if name in pope_results and pope_results[name]["pope"]:
                p = pope_results[name]["pope"]
                acc = p.get("accuracy", 0) * 100
                f1 = p.get("f1", 0)
                lat = p.get("avg_latency_ms", 0)
                tok = p.get("avg_prompt_tokens", 0)
                print(f"{name:<20} {acc:>9.1f}% {f1:>8.3f} {lat:>12.0f}ms {tok:>12.0f}")

    # Eval results
    eval_results = {k: v for k, v in results.items() if v.get("eval")}
    if eval_results:
        print("\n--- Synthetic Eval (480p, 3 runs, averaged across complexity levels) ---")
        print(f"{'Model':<20} {'Prefill':>10} {'Decode TPS':>12} {'Vis Tokens':>12} {'Peak Mem':>10}")
        print("-" * 70)
        for name in MODEL_ORDER:
            if name in eval_results and eval_results[name]["eval"]:
                e = eval_results[name]["eval"]
                cases = e.get("cases", [])
                if not cases:
                    continue
                # Average across complexity levels
                prefills = [c["summary"]["prefill_ms"]["mean"] for c in cases if "summary" in c]
                decode_tps_list = [c["summary"]["generation_tps"]["mean"] for c in cases if "summary" in c]
                vis_toks = [c["summary"]["prompt_tokens"]["mean"] for c in cases if "summary" in c]
                mems = [c["summary"]["peak_memory_gb"]["mean"] for c in cases if "summary" in c]
                if prefills:
                    prefill = sum(prefills) / len(prefills)
                    decode_tps = sum(decode_tps_list) / len(decode_tps_list)
                    vis_tok = sum(vis_toks) / len(vis_toks)
                    mem = max(mems)
                    print(f"{name:<20} {prefill:>8.0f}ms {decode_tps:>12.1f} {vis_tok:>12.0f} {mem:>8.1f} GB")

    print("\n" + "=" * 90)


def main():
    parser = argparse.ArgumentParser(description="Tier 1 comprehensive benchmark")
    parser.add_argument("--model", default=None, help="Run specific model only")
    parser.add_argument("--no-download", action="store_true", help="Skip downloading, use cached only")
    parser.add_argument("--pope-only", action="store_true", help="Only run POPE benchmark")
    parser.add_argument("--eval-only", action="store_true", help="Only run synthetic eval")
    parser.add_argument("--pope-samples", type=int, default=100, help="POPE samples (default: 100)")
    parser.add_argument("--eval-runs", type=int, default=3, help="Eval runs (default: 3)")
    parser.add_argument("--resolution", default="480p", choices=["480p", "720p", "1080p"])
    parser.add_argument("--tome", type=int, default=0, help="ToMe r (0=disabled)")
    parser.add_argument("--output-dir", default="research/eval-results/tier1", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_tome_r{args.tome}" if args.tome > 0 else "_baseline"

    # Determine which models to run
    if args.model:
        if args.model in TIER1_MODELS:
            models_to_run = {args.model: TIER1_MODELS[args.model]}
        else:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(TIER1_MODELS.keys())}")
            sys.exit(1)
    else:
        models_to_run = TIER1_MODELS

    # Check availability
    print("=" * 60)
    print("TIER 1 MODEL AVAILABILITY")
    print("=" * 60)
    available = {}
    for name, hf_id in models_to_run.items():
        cached = is_model_cached(hf_id)
        status = "CACHED" if cached else "NOT CACHED"
        print(f"  {name:<20} {status}")
        if cached:
            available[name] = hf_id
        elif not args.no_download:
            if download_model(hf_id):
                available[name] = hf_id
            # else: skip this model

    if not available:
        print("\nNo models available. Use --no-download=false or download manually.")
        sys.exit(1)

    print(f"\nWill benchmark {len(available)} models: {', '.join(available.keys())}")
    print()

    # Run benchmarks
    all_results = {}
    total_start = time.monotonic()

    for i, (name, hf_id) in enumerate(available.items()):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(available)}] Benchmarking: {name} ({hf_id})")
        print(f"{'='*60}")

        model_results = {"model": name, "hf_id": hf_id, "pope": None, "eval": None}
        model_start = time.monotonic()

        # POPE benchmark
        if not args.eval_only:
            pope_path = str(output_dir / f"pope_{name}{suffix}.json")
            print(f"\n  Running POPE ({args.pope_samples} samples)...")
            pope = run_pope(hf_id, pope_path, samples=args.pope_samples, tome=args.tome)
            if pope:
                model_results["pope"] = pope
                acc = pope.get("accuracy", 0)
                print(f"  POPE done: accuracy={acc:.1%}")
            else:
                print("  POPE: SKIPPED/FAILED")

        # Synthetic eval
        if not args.pope_only:
            eval_path = str(output_dir / f"eval_{name}_{args.resolution}{suffix}.json")
            print(f"\n  Running synthetic eval ({args.resolution}, {args.eval_runs} runs)...")
            ev = run_eval(hf_id, eval_path, resolution=args.resolution,
                         runs=args.eval_runs, tome=args.tome)
            if ev:
                model_results["eval"] = ev
                print(f"  Eval done")
            else:
                print("  Eval: SKIPPED/FAILED")

        elapsed = time.monotonic() - model_start
        print(f"\n  {name} completed in {elapsed:.0f}s")
        all_results[name] = model_results

    total_elapsed = time.monotonic() - total_start
    print(f"\n\nTotal benchmark time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save combined results
    combined_path = output_dir / f"tier1_combined{suffix}.json"
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Combined results saved: {combined_path}")

    # Print summary
    print_summary(all_results)


if __name__ == "__main__":
    main()
