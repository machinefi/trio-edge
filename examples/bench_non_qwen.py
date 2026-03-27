#!/usr/bin/env python3
"""Benchmark T1 optimizations for InternVL3 and nanoLLaVA.

Runs synthetic eval suite + POPE accuracy for each model × optimization combo.
Results saved to research/eval-results/tier1/.

Models:
  - InternVL3-1B-4bit:  baseline, Compressed, KV reuse
  - nanoLLaVA-1.5-4bit: baseline, KV reuse, StreamMem
  - InternVL3-2B-4bit:  baseline, Compressed, KV reuse

Usage:
    python examples/bench_non_qwen.py
    python examples/bench_non_qwen.py --models internvl3-1b
    python examples/bench_non_qwen.py --pope-samples 50
"""

import argparse
import gc
import json
import time
import traceback
from pathlib import Path

import numpy as np

MODELS = {
    "internvl3-1b": "mlx-community/InternVL3-1B-4bit",
    "internvl3-2b": "mlx-community/InternVL3-2B-4bit",
    "nanollava": "mlx-community/nanoLLaVA-1.5-4bit",
}

# model → list of (config_name, config_overrides)
COMBOS = {
    "internvl3-1b": [
        ("baseline", {}),
        ("compressed_50", {"_compress": 0.5}),  # special key handled below
        ("kv_reuse", {"visual_similarity_threshold": 0.95}),
    ],
    "internvl3-2b": [
        ("baseline", {}),
        ("compressed_50", {"_compress": 0.5}),
        ("kv_reuse", {"visual_similarity_threshold": 0.95}),
    ],
    "nanollava": [
        ("baseline", {}),
        ("kv_reuse", {"visual_similarity_threshold": 0.95}),
        (
            "streammem",
            {
                "streaming_memory_enabled": True,
                "streaming_memory_budget": 6000,
            },
        ),
    ],
}

OUTPUT_DIR = Path("research/eval-results/tier1")


def run_perf_bench(engine, model_key, config_name, n_runs=3, max_tokens=64):
    """Run single-frame performance benchmark (for single-image models).

    These models don't support multi-frame video, so we use single synthetic
    frames instead of the EvalSuite which sends 2 frames.
    """
    import mlx.core as mx

    print(f"\n{'=' * 60}")
    print(f"  PERF: {model_key} / {config_name} ({n_runs} runs)")
    print(f"{'=' * 60}")

    prompt = "Describe what you see in this image in detail."
    results = []

    # Generate a single synthetic frame (1, 3, 480, 640)
    rng = np.random.RandomState(42)
    frame = rng.rand(1, 3, 480, 640).astype(np.float32)

    # Warmup
    print("  Warmup...", end=" ", flush=True)
    _ = engine.analyze_video(frame, prompt, max_tokens=16)
    mx.clear_cache()
    print("done")

    for i in range(n_runs):
        result = engine.analyze_video(frame, prompt, max_tokens=max_tokens)
        m = result.metrics
        results.append(
            {
                "run": i,
                "prefill_ms": m.prefill_ms,
                "decode_ms": m.decode_ms,
                "inference_ms": m.inference_ms,
                "latency_ms": m.latency_ms,
                "prompt_tps": m.prompt_tps,
                "generation_tps": m.tokens_per_sec,
                "prompt_tokens": m.prompt_tokens,
                "completion_tokens": m.completion_tokens,
                "peak_memory_gb": m.peak_memory_gb,
                "text": result.text[:100],
            }
        )
        print(
            f"  Run {i + 1}: prefill={m.prefill_ms:.0f}ms  "
            f"decode={m.decode_ms:.0f}ms  total={m.latency_ms:.0f}ms  "
            f"gen_tps={m.tokens_per_sec:.0f}"
        )
        mx.clear_cache()

    # Averages (skip first run for warm metrics)
    warm = results[1:] if len(results) > 1 else results
    summary = {
        "model": model_key,
        "config": config_name,
        "n_runs": n_runs,
        "avg_prefill_ms": float(np.mean([r["prefill_ms"] for r in warm])),
        "avg_decode_ms": float(np.mean([r["decode_ms"] for r in warm])),
        "avg_latency_ms": float(np.mean([r["latency_ms"] for r in warm])),
        "avg_prompt_tps": float(np.mean([r["prompt_tps"] for r in warm])),
        "avg_gen_tps": float(np.mean([r["generation_tps"] for r in warm])),
        "avg_peak_memory_gb": float(np.mean([r["peak_memory_gb"] for r in warm])),
        "prompt_tokens": results[0]["prompt_tokens"],
        "runs": results,
    }

    print(
        f"\n  Avg: prefill={summary['avg_prefill_ms']:.0f}ms  "
        f"decode={summary['avg_decode_ms']:.0f}ms  "
        f"total={summary['avg_latency_ms']:.0f}ms  "
        f"gen_tps={summary['avg_gen_tps']:.0f}  "
        f"mem={summary['avg_peak_memory_gb']:.2f}GB"
    )

    out = OUTPUT_DIR / f"perf_{model_key}_{config_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


def run_pope(engine, model_key, config_name, samples=100, max_tokens=16):
    """Run POPE benchmark, return result dict."""
    from trio_core.eval_benchmarks import BenchmarkRunner, POPEBenchmark

    print(f"\n{'=' * 60}")
    print(f"  POPE: {model_key} / {config_name} ({samples} samples)")
    print(f"{'=' * 60}")

    benchmark = POPEBenchmark(split="random", max_samples=samples)
    runner = BenchmarkRunner(engine, max_tokens=max_tokens)
    result = runner.run(benchmark)
    result.print()

    out = OUTPUT_DIR / f"pope_{model_key}_{config_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    result.save(str(out))
    print(f"  Saved: {out}")
    return result


def run_kv_reuse_bench(
    backend, model_key, threshold=0.95, n_runs=3, frames_per_seq=5, noise=0.01, max_tokens=32
):
    """Run visual similarity KV reuse benchmark, return summary dict."""
    import mlx.core as mx

    from trio_core.generate import _wired_limit

    print(f"\n{'=' * 60}")
    print(f"  KV REUSE: {model_key} (threshold={threshold}, noise={noise})")
    print(f"{'=' * 60}")

    prompt = "Describe what you see in this image."

    def make_frame(seed, h=480, w=640):
        rng = np.random.RandomState(seed)
        return rng.rand(1, 3, h, w).astype(np.float32)

    def perturb(frame, noise_level, seed):
        rng = np.random.RandomState(seed)
        noise = rng.randn(*frame.shape).astype(np.float32) * noise_level
        return np.clip(frame + noise, 0.0, 1.0)

    def run_seq(frames):
        results = []
        for i, frame in enumerate(frames):
            tic = time.perf_counter()
            result = backend.generate(frame, prompt, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - tic
            results.append(
                {
                    "frame": i,
                    "total_ms": elapsed * 1000,
                    "prompt_tps": result.prompt_tps,
                    "gen_tps": result.generation_tps,
                    "text": result.text[:80],
                }
            )
        return results

    # Build sequences
    sequences = []
    for run_i in range(n_runs + 1):
        base = make_frame(seed=run_i * 100)
        frames = [base] + [
            perturb(base, noise, seed=run_i * 100 + j) for j in range(1, frames_per_seq)
        ]
        sequences.append(frames)

    with _wired_limit(backend._model):
        # Warmup
        backend.set_visual_similarity(0.0)
        _ = run_seq(sequences[0][:2])
        mx.clear_cache()
        gc.collect()
        backend.set_visual_similarity(threshold)
        _ = run_seq(sequences[0][:2])
        mx.clear_cache()
        gc.collect()

        all_baseline = []
        all_sim = []
        for run_i in range(n_runs):
            frames = sequences[run_i + 1]
            backend.set_visual_similarity(0.0)
            bl = run_seq(frames)
            mx.clear_cache()
            gc.collect()
            backend.set_visual_similarity(threshold)
            sm = run_seq(frames)
            mx.clear_cache()
            gc.collect()
            all_baseline.append(bl)
            all_sim.append(sm)

            avg_bl_warm = np.mean([r["total_ms"] for r in bl[1:]])
            avg_sm_warm = np.mean([r["total_ms"] for r in sm[1:]])
            print(
                f"  Run {run_i + 1}: baseline_warm={avg_bl_warm:.0f}ms  "
                f"sim_warm={avg_sm_warm:.0f}ms  "
                f"speedup={avg_bl_warm / max(avg_sm_warm, 1):.2f}x"
            )

    base_warm = [r["total_ms"] for run in all_baseline for r in run[1:]]
    sim_warm = [r["total_ms"] for run in all_sim for r in run[1:]]
    base_cold = [run[0]["total_ms"] for run in all_baseline]
    sim_cold = [run[0]["total_ms"] for run in all_sim]

    summary = {
        "model": model_key,
        "threshold": threshold,
        "noise": noise,
        "baseline_cold_ms": float(np.mean(base_cold)),
        "baseline_warm_ms": float(np.mean(base_warm)),
        "similarity_cold_ms": float(np.mean(sim_cold)),
        "similarity_warm_ms": float(np.mean(sim_warm)),
        "speedup_warm": float(np.mean(base_warm) / max(np.mean(sim_warm), 1)),
        "savings_pct": float((1 - np.mean(sim_warm) / max(np.mean(base_warm), 1)) * 100),
    }
    print(
        f"\n  Cold:  baseline={summary['baseline_cold_ms']:.0f}ms  "
        f"sim={summary['similarity_cold_ms']:.0f}ms"
    )
    print(
        f"  Warm:  baseline={summary['baseline_warm_ms']:.0f}ms  "
        f"sim={summary['similarity_warm_ms']:.0f}ms  "
        f"speedup={summary['speedup_warm']:.2f}x  "
        f"savings={summary['savings_pct']:.1f}%"
    )

    out = OUTPUT_DIR / f"kv_reuse_{model_key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


def run_streammem_bench(backend, model_key, n_frames=10, budget=6000, max_tokens=32):
    """Run StreamMem continuous frames benchmark."""
    import mlx.core as mx

    from trio_core.generate import _wired_limit

    print(f"\n{'=' * 60}")
    print(f"  STREAMMEM: {model_key} ({n_frames} frames, budget={budget})")
    print(f"{'=' * 60}")

    prompt = "Describe what you see in this image."
    results = []

    with _wired_limit(backend._model):
        for i in range(n_frames):
            rng = np.random.RandomState(i * 42)
            frame = rng.rand(1, 3, 480, 640).astype(np.float32)
            tic = time.perf_counter()
            result = backend.generate(frame, prompt, max_tokens=max_tokens, temperature=0.0)
            elapsed = time.perf_counter() - tic
            results.append(
                {
                    "frame": i,
                    "total_ms": elapsed * 1000,
                    "prompt_tps": result.prompt_tps,
                    "gen_tps": result.generation_tps,
                    "text": result.text[:80],
                }
            )
            print(
                f"  Frame {i}: {elapsed * 1000:.0f}ms  "
                f"prefill={result.prompt_tps:.0f} t/s  "
                f"decode={result.generation_tps:.0f} t/s"
            )
            mx.clear_cache()

    summary = {
        "model": model_key,
        "n_frames": n_frames,
        "budget": budget,
        "avg_ms": float(np.mean([r["total_ms"] for r in results])),
        "first_ms": results[0]["total_ms"],
        "steady_ms": float(np.mean([r["total_ms"] for r in results[2:]])),
        "frames": results,
    }

    out = OUTPUT_DIR / f"streammem_{model_key}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {out}")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Non-Qwen T1 benchmark suite")
    parser.add_argument(
        "--models", nargs="*", default=None, help="Model keys to benchmark (default: all)"
    )
    parser.add_argument(
        "--pope-samples", type=int, default=100, help="POPE samples per config (default: 100)"
    )
    parser.add_argument(
        "--eval-runs", type=int, default=3, help="Eval suite runs per case (default: 3)"
    )
    parser.add_argument("--skip-pope", action="store_true", help="Skip POPE accuracy benchmarks")
    parser.add_argument("--skip-eval", action="store_true", help="Skip synthetic eval suite")
    args = parser.parse_args()

    model_keys = args.models or list(MODELS.keys())

    # Summary collector
    all_results = {}

    for model_key in model_keys:
        hf_id = MODELS.get(model_key)
        if not hf_id:
            print(f"Unknown model: {model_key}, skipping")
            continue

        combos = COMBOS.get(model_key, [("baseline", {})])

        print(f"\n{'#' * 70}")
        print(f"#  MODEL: {model_key} ({hf_id})")
        print(f"#  Configs: {[c[0] for c in combos]}")
        print(f"{'#' * 70}")

        for config_name, overrides in combos:
            try:
                import mlx.core as mx

                from trio_core import TrioCore
                from trio_core.config import EngineConfig

                is_compress = "_compress" in overrides
                is_kv_reuse = "visual_similarity_threshold" in overrides
                is_streammem = overrides.get("streaming_memory_enabled", False)

                # Build config
                config_kwargs = {
                    "model": hf_id,
                    "max_tokens": 64,
                    "dedup_enabled": False,
                    "motion_enabled": False,
                }
                # Add non-special overrides
                for k, v in overrides.items():
                    if not k.startswith("_"):
                        config_kwargs[k] = v

                config = EngineConfig(**config_kwargs)
                engine = TrioCore(config)

                print(f"\n  Loading {model_key} / {config_name}...")
                tic = time.perf_counter()

                if is_compress:
                    from trio_core.compressed_backend import CompressedMLXBackend
                    from trio_core.token_compression import TokenCompressor

                    ratio = overrides["_compress"]
                    compressor = TokenCompressor(strategy="similarity", ratio=ratio)
                    backend = CompressedMLXBackend(hf_id, compressor)
                    backend.load()
                    engine._backend = backend
                    engine._loaded = True
                else:
                    engine.load()

                load_time = time.perf_counter() - tic
                health = engine.health()
                print(f"  Loaded in {load_time:.1f}s  backend={health['backend']['backend']}")

                model_results = all_results.setdefault(model_key, {})

                # KV reuse: special bench (visual similarity A/B)
                if is_kv_reuse:
                    backend = engine._backend
                    kv_summary = run_kv_reuse_bench(
                        backend,
                        model_key,
                        threshold=overrides["visual_similarity_threshold"],
                    )
                    model_results[config_name] = {"kv_reuse": kv_summary}

                # StreamMem: special bench (continuous frames)
                elif is_streammem:
                    backend = engine._backend
                    sm_summary = run_streammem_bench(backend, model_key)
                    model_results[config_name] = {"streammem": sm_summary}

                else:
                    results = {}

                    # Single-frame performance benchmark
                    if not args.skip_eval:
                        try:
                            run_perf_bench(
                                engine,
                                model_key,
                                config_name,
                                n_runs=args.eval_runs,
                            )
                            results["perf"] = str(
                                OUTPUT_DIR / f"perf_{model_key}_{config_name}.json"
                            )
                        except Exception as e:
                            print(f"  PERF ERROR: {e}")
                            results["perf_error"] = str(e)

                    # POPE accuracy
                    if not args.skip_pope:
                        try:
                            run_pope(
                                engine,
                                model_key,
                                config_name,
                                samples=args.pope_samples,
                            )
                            results["pope"] = str(
                                OUTPUT_DIR / f"pope_{model_key}_{config_name}.json"
                            )
                        except Exception as e:
                            print(f"  POPE ERROR: {e}")
                            results["pope_error"] = str(e)

                    model_results[config_name] = results

                # Cleanup
                del engine
                gc.collect()
                mx.clear_cache()

            except Exception as e:
                print(f"\n  ERROR: {model_key}/{config_name}: {e}")
                traceback.print_exc()
                all_results.setdefault(model_key, {})[config_name] = {"error": str(e)}
                gc.collect()

    # Final summary
    print(f"\n\n{'=' * 70}")
    print("  BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    summary_path = OUTPUT_DIR / "non_qwen_benchmark_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"  Full results: {summary_path}")

    for model_key, configs in all_results.items():
        print(f"\n  {model_key}:")
        for config_name, result in configs.items():
            if "error" in result:
                print(f"    {config_name}: ERROR - {result['error']}")
            elif "kv_reuse" in result:
                kv = result["kv_reuse"]
                print(
                    f"    {config_name}: speedup={kv['speedup_warm']:.2f}x  "
                    f"savings={kv['savings_pct']:.1f}%"
                )
            elif "streammem" in result:
                sm = result["streammem"]
                print(
                    f"    {config_name}: first={sm['first_ms']:.0f}ms  "
                    f"steady={sm['steady_ms']:.0f}ms"
                )
            else:
                print(f"    {config_name}: ✓")


if __name__ == "__main__":
    main()
