#!/usr/bin/env python3
"""A/B benchmark: trio-core generate_step vs mlx-vlm generate_step.

Compares our custom generate loop against mlx-vlm's original to verify
we're not slower (and measure any gains from cache reuse, early stop, etc).

Usage:
    python examples/bench_generate_loop.py
    python examples/bench_generate_loop.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python examples/bench_generate_loop.py --runs 5 --max-tokens 64
"""

import argparse
import gc
import time

import mlx.core as mx
import numpy as np


def make_random_frames(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return rng.rand(1, 3, h, w).astype(np.float32)


def prepare_inputs(backend, frames, prompt):
    """Prepare model inputs (shared between both loops)."""
    formatted, kwargs = backend._prepare(frames, prompt)
    input_ids = kwargs.pop("input_ids")
    pixel_values = kwargs.pop("pixel_values")
    mask = kwargs.pop("mask")
    return input_ids, pixel_values, mask, kwargs


def bench_mlx_vlm(model, processor, input_ids, pixel_values, mask, kwargs,
                   max_tokens, temperature):
    """Benchmark mlx-vlm's original generate_step."""
    from mlx_vlm.generate import generate_step as mlxvlm_generate_step

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    detokenizer = processor.detokenizer
    detokenizer.reset()

    tic = time.perf_counter()
    n_tokens = 0
    prefill_time = 0

    for n, (token, logprobs) in enumerate(
        mlxvlm_generate_step(
            input_ids, model, pixel_values, mask,
            max_tokens=max_tokens, temperature=temperature,
            **kwargs,
        )
    ):
        if n == 0:
            prefill_time = time.perf_counter() - tic
            decode_tic = time.perf_counter()

        if tokenizer.stopping_criteria(token):
            break
        detokenizer.add_token(token)
        n_tokens = n + 1

    detokenizer.finalize()
    total_time = time.perf_counter() - tic
    decode_time = time.perf_counter() - decode_tic if n_tokens > 0 else 0

    return {
        "text": detokenizer.text,
        "prefill_ms": prefill_time * 1000,
        "decode_ms": decode_time * 1000,
        "total_ms": total_time * 1000,
        "prompt_tokens": input_ids.size,
        "gen_tokens": n_tokens,
        "prompt_tps": input_ids.size / max(prefill_time, 1e-9),
        "gen_tps": n_tokens / max(decode_time, 1e-9) if n_tokens > 0 else 0,
    }


def bench_trio_core(model, processor, input_ids, pixel_values, mask, kwargs,
                     max_tokens, temperature, use_cache_manager=False):
    """Benchmark trio-core's generate_step."""
    from trio_core.generate import generate_step as trio_generate_step, PromptCache

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    detokenizer = processor.detokenizer
    detokenizer.reset()

    cache_mgr = PromptCache(model) if use_cache_manager else None

    tic = time.perf_counter()
    n_tokens = 0
    prefill_time = 0

    for n, (token, logprobs) in enumerate(
        trio_generate_step(
            input_ids, model, pixel_values, mask,
            max_tokens=max_tokens, temperature=temperature,
            prompt_cache_manager=cache_mgr,
            **kwargs,
        )
    ):
        if n == 0:
            prefill_time = time.perf_counter() - tic
            decode_tic = time.perf_counter()

        if tokenizer.stopping_criteria(token):
            break
        detokenizer.add_token(token)
        n_tokens = n + 1

    detokenizer.finalize()
    total_time = time.perf_counter() - tic
    decode_time = time.perf_counter() - decode_tic if n_tokens > 0 else 0

    return {
        "text": detokenizer.text,
        "prefill_ms": prefill_time * 1000,
        "decode_ms": decode_time * 1000,
        "total_ms": total_time * 1000,
        "prompt_tokens": input_ids.size,
        "gen_tokens": n_tokens,
        "prompt_tps": input_ids.size / max(prefill_time, 1e-9),
        "gen_tps": n_tokens / max(decode_time, 1e-9) if n_tokens > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Generate loop A/B benchmark")
    parser.add_argument("--model", "-m",
                        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--runs", "-n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--prompt", default="Describe this image in detail.")
    args = parser.parse_args()

    from trio_core.backends import MLXBackend

    print(f"Loading model: {args.model}")
    backend = MLXBackend(args.model)
    backend.load()
    model = backend._model
    processor = backend._processor
    print("Model loaded.\n")

    # Prepare different random images for each run
    all_inputs = []
    for i in range(args.runs + 1):  # +1 for warmup
        frames = make_random_frames(seed=i * 42)
        inp = prepare_inputs(backend, frames, args.prompt)
        all_inputs.append(inp)

    # Warmup both paths
    print("Warmup (mlx-vlm)...")
    input_ids, pv, mask, kw = all_inputs[0]
    _ = bench_mlx_vlm(model, processor, input_ids, pv, mask, kw, 8, args.temperature)
    mx.clear_cache(); gc.collect()

    print("Warmup (trio-core)...")
    _ = bench_trio_core(model, processor, input_ids, pv, mask, kw, 8, args.temperature)
    mx.clear_cache(); gc.collect()
    print()

    # --- Benchmark ---
    mlxvlm_results = []
    trio_results = []

    for i in range(args.runs):
        input_ids, pv, mask, kw = all_inputs[i + 1]

        # Alternate order to reduce bias
        if i % 2 == 0:
            r_mlx = bench_mlx_vlm(model, processor, input_ids, pv, mask, kw,
                                   args.max_tokens, args.temperature)
            mx.clear_cache(); gc.collect()
            r_trio = bench_trio_core(model, processor, input_ids, pv, mask, kw,
                                     args.max_tokens, args.temperature)
            mx.clear_cache(); gc.collect()
        else:
            r_trio = bench_trio_core(model, processor, input_ids, pv, mask, kw,
                                     args.max_tokens, args.temperature)
            mx.clear_cache(); gc.collect()
            r_mlx = bench_mlx_vlm(model, processor, input_ids, pv, mask, kw,
                                   args.max_tokens, args.temperature)
            mx.clear_cache(); gc.collect()

        mlxvlm_results.append(r_mlx)
        trio_results.append(r_trio)

        print(f"Run {i+1}/{args.runs}:  "
              f"mlx-vlm prefill={r_mlx['prefill_ms']:.0f}ms decode={r_mlx['decode_ms']:.0f}ms  |  "
              f"trio-core prefill={r_trio['prefill_ms']:.0f}ms decode={r_trio['decode_ms']:.0f}ms")

    # --- Summary ---
    def avg(results, key):
        return np.mean([r[key] for r in results])

    print(f"\n{'='*60}")
    print(f"  {'Metric':<25} {'mlx-vlm':>12} {'trio-core':>12} {'diff':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for key, label in [
        ("prefill_ms", "Prefill (ms)"),
        ("decode_ms", "Decode (ms)"),
        ("total_ms", "Total (ms)"),
        ("prompt_tps", "Prefill TPS"),
        ("gen_tps", "Decode TPS"),
    ]:
        a = avg(mlxvlm_results, key)
        b = avg(trio_results, key)
        if "tps" in key.lower():
            diff_pct = (b - a) / max(a, 1e-9) * 100
            sign = "+" if diff_pct > 0 else ""
        else:
            diff_pct = (b - a) / max(a, 1e-9) * 100
            sign = "+" if diff_pct > 0 else ""
        print(f"  {label:<25} {a:>12.1f} {b:>12.1f} {sign}{diff_pct:>8.1f}%")

    print(f"{'='*60}")

    # Correctness check: same temperature=0 should produce same text
    if args.temperature == 0.0:
        matches = sum(1 for a, b in zip(mlxvlm_results, trio_results) if a["text"] == b["text"])
        print(f"\nOutput match: {matches}/{args.runs} runs produced identical text")


if __name__ == "__main__":
    main()
