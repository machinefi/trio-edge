#!/usr/bin/env python3
"""A/B benchmark: standard decode vs speculative decode (prompt lookup).

Compares decode TPS between standard autoregressive generation and
speculative decoding with prompt lookup (n-gram matching against input tokens).

Prompt lookup works best with structured/repetitive outputs — results may
vary depending on prompt and model output patterns.

Usage:
    python examples/bench_speculative.py
    python examples/bench_speculative.py --model mlx-community/Qwen2.5-VL-3B-Instruct-4bit
    python examples/bench_speculative.py --runs 5 --max-tokens 128 --lookahead 5
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
    formatted, kwargs = backend._prepare(frames, prompt)
    input_ids = kwargs.pop("input_ids")
    pixel_values = kwargs.pop("pixel_values")
    mask = kwargs.pop("mask")
    return input_ids, pixel_values, mask, kwargs


def bench_generate(model, processor, input_ids, pixel_values, mask, kwargs,
                   max_tokens, temperature, speculative_lookahead=0):
    """Benchmark generate_step with optional speculative decode."""
    from trio_core.generate import generate_step, PromptCache

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tokenizer.stopping_criteria.reset(model.config.eos_token_id)
    detokenizer = processor.detokenizer
    detokenizer.reset()

    cache_mgr = PromptCache(model)

    tic = time.perf_counter()
    n_tokens = 0
    prefill_time = 0
    spec_stats = {}

    for n, (token, logprobs) in enumerate(
        generate_step(
            input_ids, model, pixel_values, mask,
            max_tokens=max_tokens, temperature=temperature,
            prompt_cache_manager=cache_mgr,
            speculative_lookahead=speculative_lookahead,
            speculative_stats=spec_stats,
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
        "spec_stats": spec_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Speculative decode A/B benchmark")
    parser.add_argument("--model", "-m",
                        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--runs", "-n", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--lookahead", type=int, default=5,
                        help="Number of draft tokens for speculative decode")
    parser.add_argument("--prompt", default="Describe this image in detail.")
    args = parser.parse_args()

    from trio_core.backends import MLXBackend
    from trio_core.generate import _wired_limit

    print(f"Loading model: {args.model}")
    backend = MLXBackend(args.model)
    backend.load()
    model = backend._model
    processor = backend._processor
    print(f"Speculative lookahead: {args.lookahead}")
    print("Model loaded.\n")

    # Prepare inputs
    all_inputs = []
    for i in range(args.runs + 1):
        frames = make_random_frames(seed=i * 42)
        inp = prepare_inputs(backend, frames, args.prompt)
        all_inputs.append(inp)

    # Warmup
    print("Warmup (standard)...")
    input_ids, pv, mask, kw = all_inputs[0]
    with _wired_limit(model):
        _ = bench_generate(model, processor, input_ids, pv, mask, kw,
                           8, args.temperature, speculative_lookahead=0)
    mx.clear_cache(); gc.collect()

    print("Warmup (speculative)...")
    with _wired_limit(model):
        _ = bench_generate(model, processor, input_ids, pv, mask, kw,
                           8, args.temperature, speculative_lookahead=args.lookahead)
    mx.clear_cache(); gc.collect()
    print()

    # Benchmark
    std_results = []
    spec_results = []

    with _wired_limit(model):
        for i in range(args.runs):
            input_ids, pv, mask, kw = all_inputs[i + 1]

            # Alternate order to reduce bias
            if i % 2 == 0:
                r_std = bench_generate(model, processor, input_ids, pv, mask, kw,
                                       args.max_tokens, args.temperature, 0)
                mx.clear_cache(); gc.collect()
                r_spec = bench_generate(model, processor, input_ids, pv, mask, kw,
                                        args.max_tokens, args.temperature, args.lookahead)
                mx.clear_cache(); gc.collect()
            else:
                r_spec = bench_generate(model, processor, input_ids, pv, mask, kw,
                                        args.max_tokens, args.temperature, args.lookahead)
                mx.clear_cache(); gc.collect()
                r_std = bench_generate(model, processor, input_ids, pv, mask, kw,
                                       args.max_tokens, args.temperature, 0)
                mx.clear_cache(); gc.collect()

            std_results.append(r_std)
            spec_results.append(r_spec)

            ss = r_spec['spec_stats']
            acc_rate = ss.get('acceptance_rate', 0) * 100
            drafted = ss.get('drafted', 0)
            accepted = ss.get('accepted', 0)
            fallbacks = ss.get('fallbacks', 0)
            print(f"Run {i+1}/{args.runs}:  "
                  f"std decode={r_std['decode_ms']:.0f}ms ({r_std['gen_tps']:.1f} t/s, {r_std['gen_tokens']} tok)  |  "
                  f"spec decode={r_spec['decode_ms']:.0f}ms ({r_spec['gen_tps']:.1f} t/s, {r_spec['gen_tokens']} tok)  "
                  f"accept={acc_rate:.0f}% ({accepted}/{drafted} drafted, {fallbacks} fallbacks)")

    # Summary
    def avg(results, key):
        return np.mean([r[key] for r in results])

    print(f"\n{'='*70}")
    print(f"  {'Metric':<25} {'Standard':>12} {'Speculative':>12} {'diff':>10}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

    for key, label in [
        ("prefill_ms", "Prefill (ms)"),
        ("decode_ms", "Decode (ms)"),
        ("total_ms", "Total (ms)"),
        ("gen_tps", "Decode TPS"),
        ("gen_tokens", "Tokens generated"),
    ]:
        a = avg(std_results, key)
        b = avg(spec_results, key)
        diff_pct = (b - a) / max(a, 1e-9) * 100
        sign = "+" if diff_pct > 0 else ""
        print(f"  {label:<25} {a:>12.1f} {b:>12.1f} {sign}{diff_pct:>8.1f}%")

    # Acceptance rate summary
    all_drafted = sum(r['spec_stats'].get('drafted', 0) for r in spec_results)
    all_accepted = sum(r['spec_stats'].get('accepted', 0) for r in spec_results)
    all_fallbacks = sum(r['spec_stats'].get('fallbacks', 0) for r in spec_results)
    avg_acc = all_accepted / max(all_drafted, 1) * 100
    print(f"  {'Acceptance rate':<25} {'—':>12} {avg_acc:>11.1f}%")
    print(f"  {'Drafted/Accepted':<25} {'—':>12} {all_accepted:>5}/{all_drafted:<5}")
    print(f"  {'Fallback rounds':<25} {'—':>12} {all_fallbacks:>12}")
    print(f"{'='*70}")

    # Correctness check
    if args.temperature == 0.0:
        matches = sum(1 for a, b in zip(std_results, spec_results) if a["text"] == b["text"])
        print(f"\nOutput match: {matches}/{args.runs} runs produced identical text")
        if matches < args.runs:
            print("  (Mismatch is expected — speculative decode may produce different text")
            print("   if n-gram candidates cause different token orderings in verification)")

    # Sample outputs
    print(f"\nSample output (standard):    {std_results[0]['text'][:120]}...")
    print(f"Sample output (speculative): {spec_results[0]['text'][:120]}...")


if __name__ == "__main__":
    main()
