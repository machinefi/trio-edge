#!/usr/bin/env python3
"""Native mlx-vlm baseline — calls mlx_vlm.generate() directly, zero trio-core overhead.

Produces results in the same JSON schema as run_regression.py for direct comparison.

Usage:
    # Run baseline for default model
    uv run python examples/run_baseline_mlxvlm.py --save-baseline

    # Specific model
    uv run python examples/run_baseline_mlxvlm.py -m mlx-community/Qwen2.5-VL-3B-Instruct-4bit

    # Compare with trio-core baseline
    cat research/eval-results/regression/qwen2.5-vl-3b_mlxvlm_native.json
    cat research/eval-results/regression/qwen2.5-vl-3b_baseline.json
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

BASELINE_DIR = Path("research/eval-results/regression")
DEFAULT_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
DEFAULT_SAMPLES = 50


def frames_to_pil(frames: np.ndarray) -> Image.Image:
    """Convert (1, 3, H, W) float32 [0,1] array to PIL Image."""
    img = frames[0].transpose(1, 2, 0)  # (H, W, 3)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img)


def build_config_key(model: str) -> str:
    model_short = model.split("/")[-1].lower().replace("-instruct", "").replace("-4bit", "")
    return f"{model_short}_mlxvlm_native"


def run_benchmarks(args) -> dict:
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from trio_core.eval_benchmarks import (
        POPEBenchmark, TextVQABenchmark, GQABenchmark, MMBenchBenchmark,
    )

    model_id = args.model
    print(f"Loading model via mlx_vlm: {model_id}")
    model, processor = load(model_id)
    print("Model loaded.\n")

    n = args.samples
    config_key = build_config_key(model_id)

    results = {
        "model": model_id,
        "config_key": config_key,
        "runner": "mlx-vlm-native",
        "samples_per_bench": n,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmarks": {},
    }

    def run_single(benchmark, bench_name, *, track_yes_rate=False):
        print("=" * 60)
        print(f"{bench_name} (n={n})")
        print("=" * 60)

        samples = benchmark.load()
        correct = 0
        yes_count = 0
        latencies = []

        for sample in samples:
            pil_img = frames_to_pil(sample.image)

            # Apply prompt template if benchmark defines one
            question = sample.question
            if hasattr(benchmark, "PROMPT_TEMPLATE"):
                question = benchmark.PROMPT_TEMPLATE.format(question=question)

            # Build chat-template prompt via mlx_vlm (handles model-specific formatting)
            prompt = apply_chat_template(
                processor, model.config, question, num_images=1,
            )

            tic = time.perf_counter()
            result = generate(
                model, processor, prompt, image=[pil_img],
                max_tokens=16, temperature=0.0, verbose=False,
            )
            latency_ms = (time.perf_counter() - tic) * 1000
            latencies.append(latency_ms)

            text = result.text.strip()
            is_correct = benchmark.judge(sample, text)
            correct += int(is_correct)

            if track_yes_rate:
                norm = benchmark._normalize(text)
                if norm == "yes":
                    yes_count += 1

        total = len(samples)
        accuracy = correct / total if total else 0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0

        entry = {
            "accuracy": round(accuracy, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "n": total,
        }
        if track_yes_rate:
            entry["yes_rate"] = round(yes_count / total, 4) if total else 0

        results["benchmarks"][bench_name] = entry
        print(f"  Accuracy: {accuracy:.1%}" +
              (f"  Yes-rate: {entry['yes_rate']:.1%}" if track_yes_rate else ""))
        print(f"  Avg latency: {avg_latency:.0f}ms")
        print()

    # POPE — random
    run_single(POPEBenchmark(split="random", max_samples=n), "pope_random", track_yes_rate=True)

    # POPE — adversarial
    run_single(POPEBenchmark(split="adversarial", max_samples=n), "pope_adversarial", track_yes_rate=True)

    # TextVQA
    run_single(TextVQABenchmark(max_samples=n), "textvqa")

    # GQA
    if not args.skip_gqa:
        run_single(GQABenchmark(max_samples=n), "gqa")

    # MMBench
    if not args.skip_mmbench:
        run_single(MMBenchBenchmark(max_samples=n), "mmbench")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Native mlx-vlm baseline (no trio-core in inference path)")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL,
                        help=f"Model ID (default: {DEFAULT_MODEL})")
    parser.add_argument("--samples", "-n", type=int, default=DEFAULT_SAMPLES,
                        help=f"Samples per benchmark (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--save-baseline", action="store_true",
                        help="Save as permanent baseline (otherwise saves as _latest)")
    parser.add_argument("--skip-gqa", action="store_true",
                        help="Skip GQA benchmark")
    parser.add_argument("--skip-mmbench", action="store_true",
                        help="Skip MMBench benchmark")
    args = parser.parse_args()

    results = run_benchmarks(args)

    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    config_key = results["config_key"]

    if args.save_baseline:
        path = BASELINE_DIR / f"{config_key}.json"
    else:
        path = BASELINE_DIR / f"{config_key}_latest.json"

    path.write_text(json.dumps(results, indent=2))
    print(f"Results saved: {path}")


if __name__ == "__main__":
    main()
