#!/usr/bin/env python3
"""Real-model benchmark for attention sink in StreamMem.

Feeds N synthetic frames through the VLM with StreamMem enabled,
forcing repeated eviction rounds. Then asks a question about the
final frame and compares answer quality with/without sink tokens.

Follows StreamingLLM paper methodology:
- Measure generation quality after many eviction rounds
- Compare sink=0 (no protection) vs sink=4 (default) vs sink=8

Usage:
    python examples/bench_sink_real.py
    python examples/bench_sink_real.py --frames 50 --budget 200 --sinks 0 4 8
    python examples/bench_sink_real.py --model mlx-community/Qwen2.5-VL-7B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import gc
import logging
import time

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def create_test_frames(n_frames: int, h: int = 224, w: int = 224) -> list[np.ndarray]:
    """Create synthetic test frames with distinct visual content."""
    frames = []
    rng = np.random.RandomState(42)
    for i in range(n_frames):
        # Each frame has a distinct color + noise pattern
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Varying base color per frame
        frame[:, :, 0] = int(255 * (i / n_frames))  # R ramp
        frame[:, :, 1] = int(255 * (1 - i / n_frames))  # G inverse ramp
        frame[:, :, 2] = 128
        # Add some structure (rectangle at different positions)
        y = int((h - 40) * (i / n_frames))
        x = int((w - 40) * ((i * 7 % n_frames) / n_frames))
        frame[y : y + 40, x : x + 40, :] = 255
        # Add noise
        noise = rng.randint(0, 30, (h, w, 3), dtype=np.uint8)
        frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        frames.append(frame)
    return frames


def run_streaming_test(
    model_id: str,
    n_frames: int,
    budget: int,
    n_sink: int,
    prototype_ratio: float,
) -> dict:
    """Run streaming inference with given sink config."""
    from trio_core import TrioCore
    from trio_core.config import EngineConfig

    config = EngineConfig(
        model=model_id,
        max_tokens=50,
        dedup_enabled=False,
        motion_enabled=False,
        streaming_memory_enabled=True,
        streaming_memory_budget=budget,
        streaming_memory_prototype_ratio=prototype_ratio,
        streaming_memory_sink_tokens=n_sink,
    )

    engine = TrioCore(config)
    engine.load()

    frames = create_test_frames(n_frames)

    # Feed frames one by one through the engine, accumulating KV cache
    prompt = "Describe what you see in this image in one sentence."
    results = []
    t0 = time.monotonic()

    for i, frame in enumerate(frames):
        # Convert to NCHW format expected by analyze_frame
        result = engine.analyze_frame(frame, prompt, max_tokens=30)
        results.append(
            {
                "frame": i,
                "text": result.text,
                "prompt_tokens": result.metrics.prompt_tokens,
                "latency_ms": result.metrics.latency_ms,
            }
        )
        if (i + 1) % 10 == 0:
            # Check StreamMem state
            backend = engine._backend
            if hasattr(backend, "_get_prompt_cache"):
                pc = backend._get_prompt_cache()
                sm = pc.streaming_memory
                if sm:
                    print(
                        f"    frame {i + 1}: visual_tokens={sm._total_visual_tokens}, "
                        f"budget={sm.budget}, over={sm.over_budget}"
                    )

    total_time = time.monotonic() - t0

    # Final frame analysis — this is the quality we care about
    final_frame = frames[-1]
    final_result = engine.analyze_frame(
        final_frame,
        "Describe this image in detail. What colors, shapes, and objects do you see?",
        max_tokens=100,
    )

    # Check for degenerate outputs
    final_text = final_result.text.strip()
    is_degenerate = (
        len(final_text) < 5
        or final_text.count(final_text[:10]) > 3  # excessive repetition
        or len(set(final_text.split())) < 3  # too few unique words
    )

    # Collect generation stats from last 5 frames
    last_texts = [r["text"] for r in results[-5:]]
    unique_words = set()
    for t in last_texts:
        unique_words.update(t.lower().split())

    del engine
    gc.collect()

    return {
        "n_sink": n_sink,
        "n_frames": n_frames,
        "budget": budget,
        "total_time_s": round(total_time, 2),
        "final_text": final_text,
        "final_tokens": final_result.metrics.prompt_tokens,
        "final_latency_ms": round(final_result.metrics.latency_ms, 1),
        "is_degenerate": is_degenerate,
        "last5_unique_words": len(unique_words),
        "last5_avg_len": round(np.mean([len(t) for t in last_texts]), 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Real-model attention sink benchmark")
    parser.add_argument(
        "--model", "-m", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit", help="Model to use"
    )
    parser.add_argument(
        "--sinks",
        nargs="+",
        type=int,
        default=[0, 4],
        help="Sink token counts to compare (default: 0 4)",
    )
    parser.add_argument(
        "--frames", type=int, default=30, help="Number of frames to stream (default: 30)"
    )
    parser.add_argument(
        "--budget", type=int, default=200, help="StreamMem budget in visual tokens (default: 200)"
    )
    parser.add_argument("--prototype-ratio", type=float, default=0.1)
    args = parser.parse_args()

    print("=" * 70)
    print("StreamMem Attention Sink Benchmark (Real Model)")
    print("=" * 70)
    print(f"  model:   {args.model}")
    print(f"  frames:  {args.frames}")
    print(f"  budget:  {args.budget} visual tokens")
    print(f"  sinks:   {args.sinks}")
    print()

    all_results = []
    for n_sink in args.sinks:
        print(f"--- Running sink={n_sink} ---")
        result = run_streaming_test(
            model_id=args.model,
            n_frames=args.frames,
            budget=args.budget,
            n_sink=n_sink,
            prototype_ratio=args.prototype_ratio,
        )
        all_results.append(result)
        print(f"  Final text: {result['final_text'][:100]}...")
        print(f"  Degenerate: {result['is_degenerate']}")
        print()

    # Comparison table
    print("=" * 70)
    print("Results Comparison")
    print("=" * 70)
    print(
        f"{'sink':>6} {'time_s':>8} {'final_tok':>10} {'latency_ms':>11} "
        f"{'degenerate':>11} {'unique_w':>9} {'avg_len':>8}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"{r['n_sink']:>6} {r['total_time_s']:>8.1f} {r['final_tokens']:>10} "
            f"{r['final_latency_ms']:>11.1f} {str(r['is_degenerate']):>11} "
            f"{r['last5_unique_words']:>9} {r['last5_avg_len']:>8.1f}"
        )

    print()
    print("Final generated text per config:")
    for r in all_results:
        print(f"  sink={r['n_sink']}: {r['final_text'][:200]}")


if __name__ == "__main__":
    main()
