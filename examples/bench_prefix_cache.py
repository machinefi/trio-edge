#!/usr/bin/env python3
"""A/B benchmark for prefix KV cache reuse.

Measures prefill latency savings when the same text prompt is used
with different images (prefix hit vs full miss).

Usage:
    python examples/bench_prefix_cache.py
    python examples/bench_prefix_cache.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python examples/bench_prefix_cache.py --runs 5
"""

import argparse
import time

import numpy as np


def make_random_frames(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """Generate random video frames as (T, C, H, W) float32."""
    rng = np.random.RandomState(seed)
    # Single frame, (1, 3, H, W) float32 in [0, 1]
    return rng.rand(1, 3, h, w).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Prefix cache A/B benchmark")
    parser.add_argument(
        "--model",
        "-m",
        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        help="Model to benchmark",
    )
    parser.add_argument("--runs", "-n", type=int, default=3, help="Runs per condition")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max generation tokens")
    parser.add_argument("--prompt", default="Describe this image briefly.", help="Prompt text")
    args = parser.parse_args()

    from trio_core.backends import MLXBackend

    print(f"Loading model: {args.model}")
    backend = MLXBackend(args.model)
    backend.load()
    print("Model loaded.\n")

    prompt = args.prompt
    n_runs = args.runs

    # Warmup: one full generation to JIT compile everything
    print("Warmup run...")
    img_warmup = make_random_frames(seed=999)
    _ = backend.generate(img_warmup, prompt, max_tokens=8)
    # Clear the prompt cache so first real run is a true cold start
    backend._prompt_cache = None
    print("Warmup done.\n")

    # --- Condition A: Cold start (full miss) ---
    # Each run uses a fresh prompt cache, so no reuse is possible
    cold_times = []
    print(f"=== Cold Start (full miss) x{n_runs} ===")
    for i in range(n_runs):
        backend._prompt_cache = None  # Force fresh cache
        img = make_random_frames(seed=i * 100)
        t0 = time.perf_counter()
        result = backend.generate(img, prompt, max_tokens=args.max_tokens)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        prefill_ms = result.prompt_tokens / max(result.prompt_tps, 1e-9) * 1000
        cold_times.append(prefill_ms)
        print(
            f"  Run {i + 1}: total={total_ms:.0f}ms  prefill={prefill_ms:.0f}ms  "
            f"tokens={result.prompt_tokens}  output={result.completion_tokens}tok"
        )

    # --- Condition B: Prefix hit (same prompt, different image) ---
    # First call seeds the prefix cache, subsequent calls should hit
    prefix_times = []
    print(f"\n=== Prefix Hit (same prompt, different images) x{n_runs} ===")
    backend._prompt_cache = None  # Start fresh
    # Seed the prefix cache with first image
    img_seed = make_random_frames(seed=500)
    t0 = time.perf_counter()
    result_seed = backend.generate(img_seed, prompt, max_tokens=args.max_tokens)
    t1 = time.perf_counter()
    seed_prefill_ms = result_seed.prompt_tokens / max(result_seed.prompt_tps, 1e-9) * 1000
    print(f"  Seed run: prefill={seed_prefill_ms:.0f}ms (full miss, saves prefix)")

    for i in range(n_runs):
        img = make_random_frames(seed=600 + i)
        t0 = time.perf_counter()
        result = backend.generate(img, prompt, max_tokens=args.max_tokens)
        t1 = time.perf_counter()
        total_ms = (t1 - t0) * 1000
        prefill_ms = result.prompt_tokens / max(result.prompt_tps, 1e-9) * 1000
        prefix_times.append(prefill_ms)
        print(
            f"  Run {i + 1}: total={total_ms:.0f}ms  prefill={prefill_ms:.0f}ms  "
            f"tokens={result.prompt_tokens}  output={result.completion_tokens}tok"
        )

    # --- Summary ---
    avg_cold = np.mean(cold_times)
    avg_prefix = np.mean(prefix_times)
    speedup = (avg_cold - avg_prefix) / avg_cold * 100

    print(f"\n{'=' * 50}")
    print(f"  Cold start avg prefill:  {avg_cold:.0f}ms")
    print(f"  Prefix hit avg prefill:  {avg_prefix:.0f}ms")
    print(f"  Savings:                 {speedup:+.1f}%")
    print(f"{'=' * 50}")

    if speedup > 0:
        print(f"\nPrefix cache saves ~{speedup:.0f}% prefill time!")
    else:
        print("\nNo savings observed (prefix cache may not be triggering).")


if __name__ == "__main__":
    main()
