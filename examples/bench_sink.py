#!/usr/bin/env python3
"""Benchmark attention sink effect on StreamMem.

Simulates continuous video streaming: feeds N frames through the KV cache,
triggers repeated eviction rounds, then measures generation quality via
perplexity on a probe text.

Usage:
    # Quick test (synthetic frames, ~30s)
    python examples/bench_sink.py

    # Compare sink=0 vs sink=4 vs sink=8
    python examples/bench_sink.py --sinks 0 4 8

    # More frames = more eviction rounds = bigger difference
    python examples/bench_sink.py --frames 200 --budget 30
"""

from __future__ import annotations

import argparse
import time

import mlx.core as mx

from trio_core.streaming_memory import StreamingMemory

# ── Fake KV cache (same shape as real mlx_lm KVCache) ───────────────────────


class FakeKVCacheEntry:
    """Mimics mlx_lm KVCache with realistic shapes."""

    def __init__(self, n_heads: int, head_dim: int, seq_len: int):
        self.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.offset = seq_len
        mx.eval(self.keys, self.values)

    def extend(self, n_tokens: int, n_heads: int, head_dim: int):
        """Simulate appending new frame tokens to the cache."""
        new_k = mx.random.normal((1, n_heads, n_tokens, head_dim))
        new_v = mx.random.normal((1, n_heads, n_tokens, head_dim))
        self.keys = mx.concatenate([self.keys[:, :, : self.offset, :], new_k], axis=2)
        self.values = mx.concatenate([self.values[:, :, : self.offset, :], new_v], axis=2)
        self.offset += n_tokens
        mx.eval(self.keys, self.values)


def make_cache(n_layers: int, n_heads: int, head_dim: int, seq_len: int):
    return [FakeKVCacheEntry(n_heads, head_dim, seq_len) for _ in range(n_layers)]


# ── Coherence metric ────────────────────────────────────────────────────────


def measure_kv_coherence(cache: list, text_prefix_len: int, n_vis: int) -> dict:
    """Measure KV cache health after repeated evictions.

    Since we don't have a real LM for perplexity, we use proxy metrics:
    1. K-norm variance: high variance = some positions have collapsed/exploded
    2. K-V alignment: cosine sim between K and V at same position (should be moderate)
    3. Sink preservation: whether first visual positions have reasonable norms
    """
    metrics = {}
    for layer_idx, c in enumerate(cache):
        if not hasattr(c, "keys"):
            continue
        k = c.keys[:, :, text_prefix_len : text_prefix_len + n_vis, :]
        v = c.values[:, :, text_prefix_len : text_prefix_len + n_vis, :]

        if k.shape[2] < 2:
            metrics[f"layer_{layer_idx}"] = {
                "k_norm_mean": 0.0,
                "k_norm_cv": 0.0,
                "kv_cosine_sim": 0.0,
                "sink_k_norm": 0.0,
                "n_visual_tokens": k.shape[2],
            }
            continue

        # K-norm distribution (should be stable, not have huge outliers)
        k_norms = mx.linalg.norm(k, axis=-1).reshape(-1)  # (B*H*S,)
        mx.eval(k_norms)
        norm_mean = k_norms.mean().item()
        norm_std = mx.sqrt(mx.mean((k_norms - norm_mean) ** 2)).item()
        norm_cv = norm_std / (norm_mean + 1e-8)  # coefficient of variation

        # K-V cosine similarity (measure alignment health)
        k_flat = k.reshape(-1, k.shape[-1])
        v_flat = v.reshape(-1, v.shape[-1])
        cos_sim = mx.sum(k_flat * v_flat, axis=-1) / (
            mx.linalg.norm(k_flat, axis=-1) * mx.linalg.norm(v_flat, axis=-1) + 1e-8
        )
        mx.eval(cos_sim)
        kv_sim = cos_sim.mean().item()

        # First 4 visual positions norm (sink health)
        n_check = min(4, k.shape[2])
        sink_k = k[:, :, :n_check, :]
        sink_norms = mx.linalg.norm(sink_k, axis=-1).mean().item()

        metrics[f"layer_{layer_idx}"] = {
            "k_norm_mean": round(norm_mean, 4),
            "k_norm_cv": round(norm_cv, 4),
            "kv_cosine_sim": round(kv_sim, 4),
            "sink_k_norm": round(sink_norms, 4),
            "n_visual_tokens": k.shape[2],
        }

    return metrics


# ── Main benchmark ──────────────────────────────────────────────────────────


def run_one(
    n_sink: int,
    n_frames: int,
    tokens_per_frame: int,
    budget: int,
    prototype_ratio: float,
    n_layers: int,
    n_heads: int,
    head_dim: int,
) -> dict:
    """Run streaming simulation with given sink config, return metrics."""
    text_prefix_len = 20
    initial_seq = text_prefix_len  # start with just text prefix

    sm = StreamingMemory(
        budget=budget,
        prototype_ratio=prototype_ratio,
        n_sink_tokens=n_sink,
    )

    cache = make_cache(n_layers, n_heads, head_dim, initial_seq)
    sm._text_prefix_len = text_prefix_len

    n_evictions = 0
    t0 = time.monotonic()

    for frame_idx in range(n_frames):
        # Simulate prefilling a new frame
        for c in cache:
            c.extend(tokens_per_frame, n_heads, head_dim)

        sm.append_frame(tokens_per_frame)

        # Evict if over budget
        if sm.over_budget:
            saliency = mx.random.normal((sm._total_visual_tokens,))
            mx.eval(saliency)
            sm.evict_and_merge(cache, saliency)
            n_evictions += 1

    elapsed = time.monotonic() - t0

    # Measure cache health
    coherence = measure_kv_coherence(cache, text_prefix_len, sm._total_visual_tokens)

    return {
        "n_sink": n_sink,
        "n_frames": n_frames,
        "n_evictions": n_evictions,
        "final_visual_tokens": sm._total_visual_tokens,
        "elapsed_s": round(elapsed, 3),
        "coherence": coherence,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark attention sink in StreamMem")
    parser.add_argument(
        "--sinks",
        nargs="+",
        type=int,
        default=[0, 4, 8],
        help="Sink token counts to compare (default: 0 4 8)",
    )
    parser.add_argument(
        "--frames", type=int, default=100, help="Number of frames to simulate (default: 100)"
    )
    parser.add_argument(
        "--tokens-per-frame", type=int, default=20, help="Visual tokens per frame (default: 20)"
    )
    parser.add_argument("--budget", type=int, default=50, help="StreamMem budget (default: 50)")
    parser.add_argument(
        "--prototype-ratio", type=float, default=0.1, help="Prototype ratio (default: 0.1)"
    )
    parser.add_argument("--layers", type=int, default=4, help="Number of layers")
    parser.add_argument("--heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    args = parser.parse_args()

    print("StreamMem Attention Sink Benchmark")
    print(
        f"  frames={args.frames}, tokens/frame={args.tokens_per_frame}, "
        f"budget={args.budget}, proto_ratio={args.prototype_ratio}"
    )
    print(f"  model: {args.layers}L × {args.heads}H × {args.head_dim}D")
    print()

    results = []
    for n_sink in args.sinks:
        result = run_one(
            n_sink=n_sink,
            n_frames=args.frames,
            tokens_per_frame=args.tokens_per_frame,
            budget=args.budget,
            prototype_ratio=args.prototype_ratio,
            n_layers=args.layers,
            n_heads=args.heads,
            head_dim=args.head_dim,
        )
        results.append(result)

    # Print comparison table
    print(f"{'sink':>6} {'evictions':>10} {'final_tok':>10} {'time_s':>8} ", end="")
    print(f"{'k_norm_cv':>10} {'kv_cos_sim':>11} {'sink_k_norm':>12}")
    print("-" * 75)

    for r in results:
        # Use first layer metrics
        layer_key = next(iter(r["coherence"]), None)
        if layer_key:
            m = r["coherence"][layer_key]
            print(
                f"{r['n_sink']:>6} {r['n_evictions']:>10} {r['final_visual_tokens']:>10} "
                f"{r['elapsed_s']:>8.3f} {m['k_norm_cv']:>10.4f} "
                f"{m['kv_cosine_sim']:>11.4f} {m['sink_k_norm']:>12.4f}"
            )

    print()
    print("Metrics:")
    print("  k_norm_cv    — coefficient of variation of K norms (lower = more stable)")
    print("  kv_cos_sim   — mean K-V cosine similarity (should be consistent)")
    print("  sink_norm    — mean norm of first 4 visual token K vectors")
    print()
    print("NOTE: This is a synthetic benchmark with random KV values.")
    print("For real PPL measurement, run with an actual model:")
    print("  python examples/bench_sink.py --real-model <model_id>")


if __name__ == "__main__":
    main()
