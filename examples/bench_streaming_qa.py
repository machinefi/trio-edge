#!/usr/bin/env python3
"""Streaming QA benchmark — measures VLM accuracy after prolonged KV cache eviction.

Simulates a real surveillance scenario:
  1. Load model, establish KV cache with initial frame
  2. Incrementally append visual tokens from N warm-up frames → trigger eviction
  3. Run POPE QA on fresh images (fresh cache per QA, not polluted)
  4. After heavy eviction, also test QA with the accumulated cache

Usage:
    python examples/bench_streaming_qa.py --warmup 30 --samples 20
    python examples/bench_streaming_qa.py --warmup 100 --samples 30 --budget 100
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_pope_samples(n_samples: int) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/POPE", "Full", split="random")
    samples = []
    for i in range(min(n_samples, len(ds))):
        item = ds[i]
        img = np.array(item["image"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        frames = img.transpose(2, 0, 1).astype(np.float32) / 255.0
        frames = frames[np.newaxis]
        samples.append(
            {
                "id": str(item.get("question_id", i)),
                "image": frames,
                "question": item["question"],
                "answer": item["answer"].lower(),
            }
        )
    return samples


def create_warmup_frames(n: int) -> list[np.ndarray]:
    """Synthetic surveillance frames as (1,3,224,224) float32."""
    rng = np.random.RandomState(42)
    frames = []
    for i in range(n):
        h, w = 224, 224
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :, 0] = int(128 + 60 * np.sin(i * 0.1))
        frame[:, :, 1] = int(128 + 60 * np.cos(i * 0.15))
        frame[:, :, 2] = 100
        y = int((h - 50) * abs(np.sin(i * 0.08)))
        x = int((w - 50) * abs(np.cos(i * 0.06)))
        frame[y : y + 50, x : x + 50] = rng.randint(180, 255, (50, 50, 3), dtype=np.uint8)
        f32 = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
        frames.append(f32[np.newaxis])
    return frames


def prepare_inputs(backend, frames: np.ndarray, prompt: str) -> dict:
    """Use backend's _prepare to get input_ids, pixel_values, mask, kwargs."""
    formatted, kwargs = backend._prepare(frames, prompt)
    return kwargs


def _normalize_yesno(text: str) -> str:
    text = text.strip().lower()
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    for neg in ["there is no", "there are no", "don't see", "do not see", "no,"]:
        if neg in text:
            return "no"
    if "yes" in text:
        return "yes"
    return text


def run_baseline(model_id: str, pope_samples: list[dict], max_tokens: int) -> dict:
    """Baseline: fresh KV per sample."""
    from trio_core import TrioCore
    from trio_core.config import EngineConfig

    config = EngineConfig(
        model=model_id,
        max_tokens=max_tokens,
        dedup_enabled=False,
        motion_enabled=False,
    )
    engine = TrioCore(config)
    engine.load()

    correct = 0
    t0 = time.monotonic()
    for sample in pope_samples:
        result = engine.analyze_video(
            sample["image"],
            f"{sample['question']} Answer yes or no.",
            max_tokens=max_tokens,
        )
        pred = _normalize_yesno(result.text)
        correct += int(pred == sample["answer"])
    elapsed = time.monotonic() - t0

    del engine
    gc.collect()
    return {
        "label": "baseline",
        "n_samples": len(pope_samples),
        "correct": correct,
        "accuracy": round(correct / max(len(pope_samples), 1), 4),
        "time_s": round(elapsed, 1),
        "n_warmup": 0,
        "n_evictions": 0,
    }


def run_streaming(
    model_id: str,
    warmup_frames: list[np.ndarray],
    pope_samples: list[dict],
    budget: int,
    n_sink: int,
    prototype_ratio: float,
    max_tokens: int,
) -> dict:
    """Streaming: incrementally append visual KV, evict, then QA."""
    from trio_core.backends import auto_backend
    from trio_core.generate import make_prompt_cache
    from trio_core.streaming_memory import StreamingMemory

    backend = auto_backend(model_id)
    backend.load()
    model = backend._model
    processor = backend._processor

    sm = StreamingMemory(
        budget=budget,
        prototype_ratio=prototype_ratio,
        n_sink_tokens=n_sink,
    )

    # -- Phase 1: Initial prefill with first warm-up frame --
    kv_cache = make_prompt_cache(model.language_model)

    first_kwargs = prepare_inputs(backend, warmup_frames[0], "Describe this scene.")
    input_ids = first_kwargs.pop("input_ids")
    pixel_values = first_kwargs.pop("pixel_values")
    mask = first_kwargs.pop("mask")

    # Get embeddings
    emb_out = model.get_input_embeddings(input_ids, pixel_values, mask=mask, **first_kwargs)
    embeds = emb_out.inputs_embeds
    fwd_kwargs = {
        k: v for k, v in emb_out.to_dict().items() if k != "inputs_embeds" and v is not None
    }

    # Full prefill
    model.language_model(
        input_ids[:, :-1],
        inputs_embeds=embeds[:, :-1],
        cache=kv_cache,
        **fwd_kwargs,
    )
    mx.eval([c.state for c in kv_cache])

    # Count visual tokens — get token IDs from config JSON
    import json as _json

    import huggingface_hub

    _cfg_path = Path(huggingface_hub.snapshot_download(model_id)) / "config.json"
    with open(_cfg_path) as _f:
        _model_cfg = _json.load(_f)
    img_id = _model_cfg.get("image_token_id") or _model_cfg.get("image_token_index")
    vid_id = _model_cfg.get("video_token_id") or _model_cfg.get("video_token_index")
    # For video inputs, visual tokens use video_token_id
    vis_token_id = vid_id or img_id
    ids_flat = np.array(input_ids[0])
    n_visual = int((ids_flat == vis_token_id).sum()) if vis_token_id else 0
    vis_positions = np.where(ids_flat == vis_token_id)[0] if vis_token_id else np.array([])
    text_prefix_len = int(vis_positions[0]) if len(vis_positions) > 0 else 0

    sm.append_frame(n_visual, text_prefix_len=text_prefix_len)
    offset0 = kv_cache[0].offset if hasattr(kv_cache[0], "offset") else 0
    print(f"  Initial: {n_visual} vis tokens, prefix={text_prefix_len}, offset={offset0}")

    # -- Phase 2: Incrementally add visual tokens from more frames --
    n_evictions = 0
    t_warmup = time.monotonic()

    for i, frame in enumerate(warmup_frames[1:], 1):
        # Process frame through ViT
        frame_kwargs = prepare_inputs(backend, frame, "Describe.")
        f_ids = frame_kwargs.pop("input_ids")
        f_pix = frame_kwargs.pop("pixel_values")
        f_mask = frame_kwargs.pop("mask")

        f_emb_out = model.get_input_embeddings(f_ids, f_pix, mask=f_mask, **frame_kwargs)
        f_embeds = f_emb_out.inputs_embeds
        f_fwd_kwargs = {
            k: v for k, v in f_emb_out.to_dict().items() if k != "inputs_embeds" and v is not None
        }

        # Extract only visual token embeddings
        f_ids_flat = np.array(f_ids[0])
        vis_mask = (
            (f_ids_flat == vis_token_id) if vis_token_id else np.zeros(len(f_ids_flat), dtype=bool)
        )
        vis_indices = np.where(vis_mask)[0]

        if len(vis_indices) > 0:
            vs, ve = int(vis_indices[0]), int(vis_indices[-1] + 1)
            vis_embeds = f_embeds[:, vs:ve, :]

            # Append visual embeddings to KV cache (incremental prefill)
            model.language_model(
                f_ids[:, vs:ve],
                inputs_embeds=vis_embeds,
                cache=kv_cache,
                **f_fwd_kwargs,
            )
            mx.eval([c.state for c in kv_cache])

            n_vis = ve - vs
            sm.append_frame(n_vis)

            # Evict if over budget
            if sm.over_budget:
                proxy_ids = f_ids[:, -8:]
                stats = sm.maybe_evict(kv_cache, model.language_model, proxy_ids)
                if stats:
                    n_evictions += 1

        if (i + 1) % 10 == 0:
            off = kv_cache[0].offset if hasattr(kv_cache[0], "offset") else 0
            print(
                f"    frame {i}: vis={sm._total_visual_tokens}, offset={off}, "
                f"evictions={n_evictions}"
            )

    warmup_time = time.monotonic() - t_warmup
    final_off = kv_cache[0].offset if hasattr(kv_cache[0], "offset") else 0
    print(
        f"  Warm-up: {len(warmup_frames)} frames, {n_evictions} evictions, "
        f"vis={sm._total_visual_tokens}, offset={final_off}, "
        f"time={warmup_time:.1f}s"
    )

    # -- Phase 3: POPE QA using engine (proper MRoPE handling) --
    print(f"  Running {len(pope_samples)} POPE QA samples...")

    from trio_core import TrioCore
    from trio_core.config import EngineConfig

    qa_config = EngineConfig(
        model=model_id,
        max_tokens=max_tokens,
        dedup_enabled=False,
        motion_enabled=False,
    )
    qa_engine = TrioCore(qa_config)
    qa_engine.load()

    correct = 0
    t_qa = time.monotonic()

    for sample in pope_samples:
        result = qa_engine.analyze_video(
            sample["image"],
            f"{sample['question']} Answer yes or no.",
            max_tokens=max_tokens,
        )
        pred = _normalize_yesno(result.text)
        correct += int(pred == sample["answer"])

    qa_time = time.monotonic() - t_qa

    del model, processor, backend, qa_engine
    gc.collect()

    return {
        "label": f"sink={n_sink}",
        "budget": budget,
        "n_sink": n_sink,
        "n_samples": len(pope_samples),
        "correct": correct,
        "accuracy": round(correct / max(len(pope_samples), 1), 4),
        "time_s": round(warmup_time + qa_time, 1),
        "n_warmup": len(warmup_frames),
        "n_evictions": n_evictions,
        "final_vis_tokens": sm._total_visual_tokens,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", "-n", type=int, default=20)
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--sinks", nargs="+", type=int, default=[0, 4])
    parser.add_argument("--prototype-ratio", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    print("=" * 70)
    print("  Streaming QA Benchmark")
    print("=" * 70)
    print(f"  Model:    {args.model}")
    print(f"  Warm-up:  {args.warmup} frames")
    print(f"  Budget:   {args.budget} visual tokens")
    print(f"  Sinks:    {args.sinks}")
    print(f"  QA:       {args.samples} POPE samples")
    print()

    pope_samples = load_pope_samples(args.samples)
    warmup_frames = create_warmup_frames(args.warmup)
    print(f"Loaded {len(pope_samples)} QA + {len(warmup_frames)} warm-up frames")
    print()

    all_results = []

    # Baseline
    print("━" * 70)
    print("BASELINE")
    print("━" * 70)
    baseline = run_baseline(args.model, pope_samples, args.max_tokens)
    all_results.append(baseline)
    print(f"  Accuracy: {baseline['accuracy']:.1%}")
    print()

    # Streaming configs
    for n_sink in args.sinks:
        print("━" * 70)
        print(f"STREAMING: budget={args.budget}, sink={n_sink}")
        print("━" * 70)
        result = run_streaming(
            model_id=args.model,
            warmup_frames=warmup_frames,
            pope_samples=pope_samples,
            budget=args.budget,
            n_sink=n_sink,
            prototype_ratio=args.prototype_ratio,
            max_tokens=args.max_tokens,
        )
        all_results.append(result)
        print(f"  Accuracy: {result['accuracy']:.1%}")
        print()

    # Summary
    print("=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(
        f"{'Config':<16} {'Accuracy':>10} {'Correct':>9} {'Evictions':>10} "
        f"{'Vis Tok':>8} {'Time(s)':>8}"
    )
    print("-" * 70)
    for r in all_results:
        vt = r.get("final_vis_tokens", "-")
        print(
            f"{r['label']:<16} {r['accuracy']:>9.1%} "
            f"{r['correct']:>4}/{r['n_samples']:<4} "
            f"{r['n_evictions']:>10} {str(vt):>8} "
            f"{r['time_s']:>8.1f}"
        )

    base_acc = all_results[0]["accuracy"]
    print()
    for r in all_results[1:]:
        delta = r["accuracy"] - base_acc
        sign = "+" if delta >= 0 else ""
        print(f"  {r['label']}: {sign}{delta:.1%} vs baseline")

    out_path = (
        args.output
        or f"research/eval-results/bench_streaming_qa_w{args.warmup}_b{args.budget}.json"
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
