#!/usr/bin/env python3
"""A/B benchmark: visual similarity KV reuse for frame-to-frame inference.

Compares consecutive-frame inference latency with and without visual
similarity gating. Simulates video monitoring where consecutive frames
are similar (small noise perturbation).

Usage:
    python examples/bench_visual_similarity.py
    python examples/bench_visual_similarity.py --threshold 0.95 --runs 5
    python examples/bench_visual_similarity.py --noise 0.05  # more frame difference
"""

import argparse
import gc
import time

import numpy as np


def make_base_frame(seed: int, h: int = 480, w: int = 640) -> np.ndarray:
    """Generate a base video frame as (1, 3, H, W) float32."""
    rng = np.random.RandomState(seed)
    return rng.rand(1, 3, h, w).astype(np.float32)


def perturb_frame(frame: np.ndarray, noise_level: float, seed: int) -> np.ndarray:
    """Add small random noise to simulate a consecutive video frame."""
    rng = np.random.RandomState(seed)
    noise = rng.randn(*frame.shape).astype(np.float32) * noise_level
    return np.clip(frame + noise, 0.0, 1.0)


def run_sequence(backend, prompt, frames, max_tokens):
    """Run inference on a sequence of frames, return per-frame timings."""
    results = []
    for i, frame in enumerate(frames):
        tic = time.perf_counter()
        result = backend.generate(frame, prompt, max_tokens=max_tokens, temperature=0.0)
        elapsed = time.perf_counter() - tic
        results.append({
            "frame": i,
            "total_ms": elapsed * 1000,
            "prompt_tps": result.prompt_tps,
            "gen_tps": result.generation_tps,
            "text": result.text[:80],
            "prompt_tokens": result.prompt_tokens,
            "gen_tokens": result.completion_tokens,
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="Visual similarity KV reuse benchmark")
    parser.add_argument("--model", "-m",
                        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--runs", "-n", type=int, default=3,
                        help="Number of frame sequences to benchmark")
    parser.add_argument("--frames-per-seq", type=int, default=5,
                        help="Frames per sequence")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="Visual similarity threshold")
    parser.add_argument("--noise", type=float, default=0.01,
                        help="Noise level for frame perturbation (0.01=very similar)")
    parser.add_argument("--prompt", default="Describe what you see in this image.")
    args = parser.parse_args()

    import mlx.core as mx
    from trio_core.backends import MLXBackend
    from trio_core.generate import _wired_limit

    print(f"Loading model: {args.model}")
    backend = MLXBackend(args.model)
    backend.load()
    print(f"Visual similarity threshold: {args.threshold}")
    print(f"Noise level: {args.noise}")
    print(f"Frames per sequence: {args.frames_per_seq}")
    print("Model loaded.\n")

    # Generate frame sequences
    sequences = []
    for run_i in range(args.runs + 1):  # +1 for warmup
        base = make_base_frame(seed=run_i * 100)
        frames = [base]
        for j in range(1, args.frames_per_seq):
            frames.append(perturb_frame(base, args.noise, seed=run_i * 100 + j))
        sequences.append(frames)

    # Warmup
    print("Warmup (no similarity)...")
    backend.set_visual_similarity(0.0)
    with _wired_limit(backend._model):
        _ = run_sequence(backend, args.prompt, sequences[0][:2], args.max_tokens)
    mx.clear_cache(); gc.collect()

    print("Warmup (with similarity)...")
    backend.set_visual_similarity(args.threshold)
    with _wired_limit(backend._model):
        _ = run_sequence(backend, args.prompt, sequences[0][:2], args.max_tokens)
    mx.clear_cache(); gc.collect()
    print()

    # Benchmark
    all_baseline = []
    all_similarity = []

    with _wired_limit(backend._model):
        for run_i in range(args.runs):
            frames = sequences[run_i + 1]

            # Baseline: no visual similarity
            backend.set_visual_similarity(0.0)
            baseline = run_sequence(backend, args.prompt, frames, args.max_tokens)
            mx.clear_cache(); gc.collect()

            # With visual similarity
            backend.set_visual_similarity(args.threshold)
            similarity = run_sequence(backend, args.prompt, frames, args.max_tokens)
            mx.clear_cache(); gc.collect()

            all_baseline.append(baseline)
            all_similarity.append(similarity)

            # Print per-run summary
            avg_base = np.mean([r["total_ms"] for r in baseline])
            avg_sim = np.mean([r["total_ms"] for r in similarity])
            # Frame 0 is always cold; frames 1+ benefit from similarity
            avg_base_warm = np.mean([r["total_ms"] for r in baseline[1:]])
            avg_sim_warm = np.mean([r["total_ms"] for r in similarity[1:]])
            print(f"Run {run_i+1}/{args.runs}:  "
                  f"baseline avg={avg_base:.0f}ms (warm={avg_base_warm:.0f}ms)  |  "
                  f"similarity avg={avg_sim:.0f}ms (warm={avg_sim_warm:.0f}ms)  "
                  f"speedup={avg_base_warm/max(avg_sim_warm,1):.2f}x")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Visual Similarity KV Reuse Benchmark (threshold={args.threshold}, noise={args.noise})")
    print(f"  {'='*66}")

    # Aggregate warm frames (frame 1+) across all runs
    base_warm_times = [r["total_ms"] for run in all_baseline for r in run[1:]]
    sim_warm_times = [r["total_ms"] for run in all_similarity for r in run[1:]]
    base_cold_times = [run[0]["total_ms"] for run in all_baseline]
    sim_cold_times = [run[0]["total_ms"] for run in all_similarity]

    base_warm_avg = np.mean(base_warm_times)
    sim_warm_avg = np.mean(sim_warm_times)
    base_cold_avg = np.mean(base_cold_times)
    sim_cold_avg = np.mean(sim_cold_times)

    print(f"\n  {'Metric':<30} {'Baseline':>12} {'Similarity':>12} {'Speedup':>10}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*10}")
    print(f"  {'Cold frame (ms)':<30} {base_cold_avg:>12.0f} {sim_cold_avg:>12.0f} "
          f"{base_cold_avg/max(sim_cold_avg,1):>9.2f}x")
    print(f"  {'Warm frames avg (ms)':<30} {base_warm_avg:>12.0f} {sim_warm_avg:>12.0f} "
          f"{base_warm_avg/max(sim_warm_avg,1):>9.2f}x")

    savings_pct = (1 - sim_warm_avg / max(base_warm_avg, 1)) * 100
    print(f"\n  Warm frame savings: {savings_pct:.1f}%")
    print(f"  (Warm = frames 1+ in each sequence, where similarity KV reuse applies)")

    # Check output consistency
    print(f"\n  Output comparison (first sequence):")
    for i in range(min(3, len(all_baseline[0]))):
        base_text = all_baseline[0][i]["text"]
        sim_text = all_similarity[0][i]["text"]
        match = "✓" if base_text == sim_text else "≈"
        print(f"    Frame {i}: {match} baseline='{base_text[:50]}...'")
        if base_text != sim_text:
            print(f"             similarity='{sim_text[:50]}...'")

    print(f"{'='*70}")


if __name__ == "__main__":
    main()
