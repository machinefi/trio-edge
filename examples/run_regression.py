#!/usr/bin/env python3
"""Accuracy regression test — run before and after engine changes.

Runs a fixed set of benchmarks and compares against saved baselines.
Exits with code 1 if accuracy drops below threshold.

Usage:
    # Save baseline (run once per model, before making changes)
    python examples/run_regression.py --save-baseline

    # Run regression check (after changes)
    python examples/run_regression.py

    # Run with specific model
    python examples/run_regression.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit

    # Run with ToMe enabled
    python examples/run_regression.py --tome 4

    # Stricter threshold
    python examples/run_regression.py --threshold 0.02
"""

import argparse
import json
import sys
import time
from pathlib import Path

BASELINE_DIR = Path("research/eval-results/regression")
# Number of samples per benchmark — small enough to run in ~2 min
DEFAULT_SAMPLES = 50
# Maximum allowed accuracy drop before failing
DEFAULT_THRESHOLD = 0.03  # 3%


def build_config_key(args) -> str:
    """Build a unique key for this configuration."""
    parts = []
    model = args.model or "default"
    # Shorten model name
    model_short = model.split("/")[-1].lower().replace("-instruct", "").replace("-4bit", "")
    parts.append(model_short)
    if args.tome:
        parts.append(f"tome_r{args.tome}")
    else:
        parts.append("baseline")
    return "_".join(parts)


def run_benchmarks(args) -> dict:
    """Run all regression benchmarks, return results dict."""
    from trio_core import TrioCore, EngineConfig
    from trio_core.eval_benchmarks import (
        POPEBenchmark, TextVQABenchmark, GQABenchmark, MMBenchBenchmark,
        BenchmarkRunner,
    )

    # Setup engine
    config_kwargs = {
        "max_tokens": 16,
        "dedup_enabled": False,
        "motion_enabled": False,
    }
    if args.model:
        config_kwargs["model"] = args.model
    if args.tome:
        config_kwargs["tome_enabled"] = True
        config_kwargs["tome_r"] = args.tome
        config_kwargs["tome_metric"] = "hidden"
        config_kwargs["tome_min_keep_ratio"] = 0.3

    config = EngineConfig(**config_kwargs)
    engine = TrioCore(config)
    print(f"Loading model: {config.model}")
    engine.load()

    health = engine.health()
    print(f"Backend: {health['backend']['backend']}")
    print(f"Device: {health['backend']['device']}")
    print()

    runner = BenchmarkRunner(engine, max_tokens=16)
    n = args.samples

    results = {
        "model": config.model,
        "config_key": build_config_key(args),
        "tome_r": args.tome,
        "samples_per_bench": n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    # POPE — random split (object hallucination, yes/no)
    print("=" * 60)
    print(f"POPE-random (n={n})")
    print("=" * 60)
    pope_random = POPEBenchmark(split="random", max_samples=n)
    r = runner.run(pope_random)
    results["benchmarks"]["pope_random"] = {
        "accuracy": r.accuracy,
        "yes_rate": r.yes_rate,
        "avg_latency_ms": sum(p.latency_ms for p in r.predictions) / len(r.predictions),
        "n": len(r.predictions),
    }
    print(f"  Accuracy: {r.accuracy:.1%}  Yes-rate: {r.yes_rate:.1%}")
    print()

    # POPE — adversarial split (harder, tests hallucination resistance)
    print("=" * 60)
    print(f"POPE-adversarial (n={n})")
    print("=" * 60)
    pope_adv = POPEBenchmark(split="adversarial", max_samples=n)
    r = runner.run(pope_adv)
    results["benchmarks"]["pope_adversarial"] = {
        "accuracy": r.accuracy,
        "yes_rate": r.yes_rate,
        "avg_latency_ms": sum(p.latency_ms for p in r.predictions) / len(r.predictions),
        "n": len(r.predictions),
    }
    print(f"  Accuracy: {r.accuracy:.1%}  Yes-rate: {r.yes_rate:.1%}")
    print()

    # TextVQA — OCR capability
    print("=" * 60)
    print(f"TextVQA (n={n})")
    print("=" * 60)
    textvqa = TextVQABenchmark(max_samples=n)
    r = runner.run(textvqa)
    results["benchmarks"]["textvqa"] = {
        "accuracy": r.accuracy,
        "avg_latency_ms": sum(p.latency_ms for p in r.predictions) / len(r.predictions),
        "n": len(r.predictions),
    }
    print(f"  Accuracy: {r.accuracy:.1%}")
    print()

    # GQA — real-world visual reasoning (spatial relations, attributes, counting)
    if not args.skip_gqa:
        print("=" * 60)
        print(f"GQA (n={n})")
        print("=" * 60)
        gqa = GQABenchmark(max_samples=n)
        r = runner.run(gqa)
        results["benchmarks"]["gqa"] = {
            "accuracy": r.accuracy,
            "avg_latency_ms": sum(p.latency_ms for p in r.predictions) / len(r.predictions),
            "n": len(r.predictions),
        }
        print(f"  Accuracy: {r.accuracy:.1%}")
        print()

    # MMBench — multi-ability (20 dimensions, multiple choice)
    if not args.skip_mmbench:
        print("=" * 60)
        print(f"MMBench (n={n})")
        print("=" * 60)
        mmbench = MMBenchBenchmark(max_samples=n)
        r = runner.run(mmbench)
        results["benchmarks"]["mmbench"] = {
            "accuracy": r.accuracy,
            "avg_latency_ms": sum(p.latency_ms for p in r.predictions) / len(r.predictions),
            "n": len(r.predictions),
        }
        print(f"  Accuracy: {r.accuracy:.1%}")
        print()

    return results


def save_baseline(results: dict):
    """Save results as baseline for future regression checks."""
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    path = BASELINE_DIR / f"{results['config_key']}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"Baseline saved: {path}")


def check_regression(results: dict, threshold: float) -> bool:
    """Compare results against saved baseline. Returns True if passed."""
    path = BASELINE_DIR / f"{results['config_key']}.json"
    if not path.exists():
        print(f"No baseline found at {path}")
        print("Run with --save-baseline first.")
        return False

    baseline = json.loads(path.read_text())

    print("=" * 60)
    print("REGRESSION CHECK")
    print("=" * 60)
    print(f"Baseline: {baseline['timestamp']}")
    print(f"Current:  {results['timestamp']}")
    print(f"Threshold: {threshold:.1%} max accuracy drop")
    print()

    all_passed = True
    for bench_name in results["benchmarks"]:
        current = results["benchmarks"][bench_name]
        if bench_name not in baseline["benchmarks"]:
            print(f"  {bench_name}: SKIP (not in baseline)")
            continue

        base = baseline["benchmarks"][bench_name]
        acc_current = current["accuracy"]
        acc_baseline = base["accuracy"]
        delta = acc_current - acc_baseline

        latency_current = current["avg_latency_ms"]
        latency_baseline = base["avg_latency_ms"]
        lat_delta = (latency_current - latency_baseline) / max(latency_baseline, 1) * 100

        passed = delta >= -threshold
        status = "PASS" if passed else "FAIL"

        print(f"  {bench_name}:")
        print(f"    Accuracy: {acc_baseline:.1%} -> {acc_current:.1%} ({delta:+.1%}) [{status}]")
        print(f"    Latency:  {latency_baseline:.0f}ms -> {latency_current:.0f}ms ({lat_delta:+.1f}%)")

        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("RESULT: ALL PASSED")
    else:
        print(f"RESULT: FAILED — accuracy dropped more than {threshold:.1%}")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Accuracy regression test")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save current results as baseline")
    parser.add_argument("--model", "-m", default=None,
                        help="Model name (default: auto)")
    parser.add_argument("--tome", type=int, default=None, metavar="R",
                        help="Enable ToMe with r tokens merged per layer")
    parser.add_argument("--samples", "-n", type=int, default=DEFAULT_SAMPLES,
                        help=f"Samples per benchmark (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help=f"Max accuracy drop allowed (default: {DEFAULT_THRESHOLD})")
    parser.add_argument("--skip-gqa", action="store_true",
                        help="Skip GQA benchmark (large dataset download)")
    parser.add_argument("--skip-mmbench", action="store_true",
                        help="Skip MMBench benchmark")
    args = parser.parse_args()

    results = run_benchmarks(args)

    if args.save_baseline:
        save_baseline(results)
    else:
        passed = check_regression(results, args.threshold)
        # Also save current results for reference
        BASELINE_DIR.mkdir(parents=True, exist_ok=True)
        current_path = BASELINE_DIR / f"{results['config_key']}_latest.json"
        current_path.write_text(json.dumps(results, indent=2))
        print(f"Current results saved: {current_path}")

        if not passed:
            sys.exit(1)


if __name__ == "__main__":
    main()
