#!/usr/bin/env python3
"""StreamMem stress test — does KV cache memory grow unbounded without eviction?

Phase 1 (critical): Run N frames through the engine WITHOUT StreamMem.
  → If memory plateaus, StreamMem is solving a non-existent problem.
  → If memory grows linearly → OOM, StreamMem is needed.

Phase 2 (if needed): Same test WITH StreamMem, measure memory + accuracy.

Usage:
    # Phase 1: baseline memory growth (no StreamMem)
    python examples/streammem_stress.py --frames 200

    # Phase 2: with StreamMem
    python examples/streammem_stress.py --frames 200 --streaming-memory --budget 2000

    # Use a real video (movie, TV show) — much more realistic than synthetic frames
    python examples/streammem_stress.py --video movie.mp4 --frames 500

    # Full comparison
    python examples/streammem_stress.py --frames 500 --compare --budgets 2000 4000 6000

See research/streammem-validation.md for the full plan.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import mlx.core as mx
import numpy as np


@dataclass
class FrameSnapshot:
    frame_idx: int
    metal_active_mb: float
    metal_peak_mb: float
    kv_offset: int
    latency_ms: float
    eviction_happened: bool = False


@dataclass
class RunResult:
    label: str
    model: str
    n_frames: int
    streaming_memory: bool
    budget: int | None
    snapshots: list[FrameSnapshot] = field(default_factory=list)
    pope_accuracy: float | None = None
    pope_samples: int = 0
    total_evictions: int = 0
    total_time_s: float = 0.0


def make_synthetic_frame(idx: int, h: int = 336, w: int = 336) -> np.ndarray:
    """Synthetic surveillance frame — varying color + moving patch.

    Returns (1, 3, H, W) float32 in [0, 1].
    """
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = int(128 + 60 * np.sin(idx * 0.1))
    frame[:, :, 1] = int(128 + 60 * np.cos(idx * 0.15))
    frame[:, :, 2] = 100
    # Moving bright patch — ensures frames aren't identical (defeats dedup)
    rng = np.random.RandomState(idx)
    y = int((h - 60) * abs(np.sin(idx * 0.08)))
    x = int((w - 60) * abs(np.cos(idx * 0.06)))
    frame[y : y + 60, x : x + 60] = rng.randint(180, 255, (60, 60, 3), dtype=np.uint8)
    return frame.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0


class VideoFrameSource:
    """Extract frames from a video file at a target FPS."""

    def __init__(self, path: str, target_fps: float = 1.0, max_frames: int = 1000):
        import cv2

        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_src_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.step = max(1, int(self.src_fps / target_fps))
        self.max_frames = max_frames
        self.path = path
        duration_s = self.total_src_frames / self.src_fps
        avail = self.total_src_frames // self.step
        print(f"  Video: {path}")
        print(
            f"  Duration: {duration_s:.0f}s, src_fps={self.src_fps:.1f}, "
            f"step={self.step} → ~{avail} frames available (cap={max_frames})"
        )

    def __iter__(self):
        import cv2

        count = 0
        frame_no = 0
        while count < self.max_frames:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, bgr = self.cap.read()
            if not ret:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            # Resize to 336x336 for consistency
            rgb = cv2.resize(rgb, (336, 336))
            arr = rgb.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0
            yield count, arr
            count += 1
            frame_no += self.step
        self.cap.release()

    def __del__(self):
        if hasattr(self, "cap") and self.cap.isOpened():
            self.cap.release()


def get_metal_memory() -> tuple[float, float]:
    """Return (active_mb, peak_mb) for Metal GPU."""
    active = mx.metal.get_active_memory() / 1e6
    peak = mx.metal.get_peak_memory() / 1e6
    return active, peak


def run_stress(
    model_id: str,
    n_frames: int,
    streaming_memory: bool = False,
    budget: int | None = None,
    sink_tokens: int = 4,
    prototype_ratio: float = 0.1,
    max_tokens: int = 16,
    log_every: int = 10,
    video_path: str | None = None,
    video_fps: float = 1.0,
) -> RunResult:
    from trio_core import EngineConfig, TrioCore

    label = f"sm_budget={budget}" if streaming_memory else "baseline"
    print(f"\n{'━' * 60}")
    print(f"  {label}  |  {n_frames} frames  |  model={model_id.split('/')[-1]}")
    print(f"{'━' * 60}")

    config = EngineConfig(
        model=model_id,
        max_tokens=max_tokens,
        dedup_enabled=False,  # Don't skip frames — we want every frame to hit the cache
        motion_enabled=False,
        streaming_memory_enabled=streaming_memory,
        streaming_memory_budget=budget or 6000,
        streaming_memory_prototype_ratio=prototype_ratio,
        streaming_memory_sink_tokens=sink_tokens,
    )
    engine = TrioCore(config)
    engine.load()

    mx.metal.reset_peak_memory()
    result = RunResult(
        label=label,
        model=model_id,
        n_frames=n_frames,
        streaming_memory=streaming_memory,
        budget=budget,
    )

    # Frame source: video file or synthetic
    video_source = None
    if video_path:
        video_source = iter(VideoFrameSource(video_path, video_fps, n_frames))

    question = "Is there a person in this scene? Answer yes or no."
    t_total = time.monotonic()

    for i in range(n_frames):
        if video_source:
            try:
                _, frame = next(video_source)
            except StopIteration:
                print(f"  Video ended at frame {i}")
                break
        else:
            frame = make_synthetic_frame(i)
        t0 = time.monotonic()
        engine.analyze_video(frame, question, max_tokens=max_tokens)
        latency = (time.monotonic() - t0) * 1000

        active_mb, peak_mb = get_metal_memory()

        # Try to get KV cache offset
        kv_offset = 0
        try:
            pc = engine._backend._get_prompt_cache()
            if pc._cache and hasattr(pc._cache[0], "offset"):
                kv_offset = pc._cache[0].offset
        except Exception:
            pass

        # Check if eviction happened (StreamMem logs it)
        evicted = False
        if streaming_memory:
            try:
                sm = engine._backend._get_prompt_cache().streaming_memory
                if sm and sm._total_visual_tokens > (budget or 6000):
                    evicted = True
            except Exception:
                pass

        snap = FrameSnapshot(
            frame_idx=i,
            metal_active_mb=round(active_mb, 1),
            metal_peak_mb=round(peak_mb, 1),
            kv_offset=kv_offset,
            latency_ms=round(latency, 1),
            eviction_happened=evicted,
        )
        result.snapshots.append(snap)
        if evicted:
            result.total_evictions += 1

        if (i + 1) % log_every == 0 or i == 0 or i == n_frames - 1:
            print(
                f"  frame {i:>4d}  |  metal={active_mb:>7.1f}MB  "
                f"peak={peak_mb:>7.1f}MB  |  kv_off={kv_offset:>6d}  "
                f"|  {latency:>6.0f}ms  |  evict={evicted}"
            )

    result.total_time_s = round(time.monotonic() - t_total, 1)

    # Summary
    first = result.snapshots[0]
    last = result.snapshots[-1]
    mem_growth = last.metal_active_mb - first.metal_active_mb
    kv_growth = last.kv_offset - first.kv_offset

    print("\n  Summary:")
    print(
        f"    Memory: {first.metal_active_mb:.0f}MB → {last.metal_active_mb:.0f}MB "
        f"(+{mem_growth:.0f}MB)"
    )
    print(f"    KV offset: {first.kv_offset} → {last.kv_offset} (+{kv_growth})")
    print(f"    Peak metal: {last.metal_peak_mb:.0f}MB")
    print(f"    Latency: {first.latency_ms:.0f}ms → {last.latency_ms:.0f}ms")
    print(f"    Evictions: {result.total_evictions}")
    print(f"    Total: {result.total_time_s:.1f}s")

    del engine
    mx.metal.reset_peak_memory()
    return result


def run_pope_after_streaming(
    model_id: str,
    n_warmup: int,
    streaming_memory: bool,
    budget: int | None,
    n_samples: int = 20,
    max_tokens: int = 16,
) -> float:
    """Run POPE accuracy check after N warmup frames."""
    from trio_core import EngineConfig, TrioCore

    config = EngineConfig(
        model=model_id,
        max_tokens=max_tokens,
        dedup_enabled=False,
        motion_enabled=False,
        streaming_memory_enabled=streaming_memory,
        streaming_memory_budget=budget or 6000,
    )
    engine = TrioCore(config)
    engine.load()

    # Warm up with N synthetic frames
    for i in range(n_warmup):
        frame = make_synthetic_frame(i)
        engine.analyze_video(frame, "Describe this scene.", max_tokens=max_tokens)

    # Now run POPE
    from datasets import load_dataset

    ds = load_dataset("lmms-lab/POPE", "Full", split="random")
    correct = 0
    for i in range(min(n_samples, len(ds))):
        item = ds[i]
        img = np.array(item["image"])
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        frames = img.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0

        result = engine.analyze_video(
            frames,
            f"{item['question']} Answer yes or no.",
            max_tokens=max_tokens,
        )
        pred = result.text.strip().lower()
        gt = item["answer"].lower()
        if pred.startswith("yes") and gt.startswith("yes"):
            correct += 1
        elif pred.startswith("no") and gt.startswith("no"):
            correct += 1

    acc = correct / max(n_samples, 1)
    del engine
    return round(acc, 4)


def main():
    parser = argparse.ArgumentParser(
        description="StreamMem stress test — does KV cache memory grow unbounded?"
    )
    parser.add_argument("--model", "-m", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--frames", "-n", type=int, default=200)
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (movie, TV show) for real frames",
    )
    parser.add_argument(
        "--fps", type=float, default=1.0, help="Target FPS when sampling from video (default: 1)"
    )
    parser.add_argument("--streaming-memory", action="store_true", help="Enable StreamMem")
    parser.add_argument("--budget", type=int, default=2000)
    parser.add_argument("--compare", action="store_true", help="Run baseline + multiple budgets")
    parser.add_argument("--budgets", nargs="+", type=int, default=[2000, 4000, 6000])
    parser.add_argument("--pope", action="store_true", help="Also run POPE accuracy after warmup")
    parser.add_argument("--pope-samples", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output", "-o", default=None)
    args = parser.parse_args()

    results = []

    if args.compare:
        # Baseline
        r = run_stress(
            args.model,
            args.frames,
            streaming_memory=False,
            log_every=args.log_every,
            video_path=args.video,
            video_fps=args.fps,
        )
        if args.pope:
            r.pope_accuracy = run_pope_after_streaming(
                args.model, args.frames, False, None, args.pope_samples
            )
            r.pope_samples = args.pope_samples
        results.append(r)

        # StreamMem with various budgets
        for b in args.budgets:
            r = run_stress(
                args.model,
                args.frames,
                streaming_memory=True,
                budget=b,
                log_every=args.log_every,
                video_path=args.video,
                video_fps=args.fps,
            )
            if args.pope:
                r.pope_accuracy = run_pope_after_streaming(
                    args.model, args.frames, True, b, args.pope_samples
                )
                r.pope_samples = args.pope_samples
            results.append(r)
    else:
        r = run_stress(
            args.model,
            args.frames,
            streaming_memory=args.streaming_memory,
            budget=args.budget if args.streaming_memory else None,
            log_every=args.log_every,
            video_path=args.video,
            video_fps=args.fps,
        )
        if args.pope:
            r.pope_accuracy = run_pope_after_streaming(
                args.model,
                args.frames,
                args.streaming_memory,
                args.budget if args.streaming_memory else None,
                args.pope_samples,
            )
            r.pope_samples = args.pope_samples
        results.append(r)

    # Final comparison table
    print(f"\n{'=' * 70}")
    print("  COMPARISON")
    print(f"{'=' * 70}")
    print(
        f"{'Config':<22} {'Mem Start':>10} {'Mem End':>10} {'Growth':>8} "
        f"{'KV Start':>9} {'KV End':>9} {'Evict':>6}" + ("  POPE" if args.pope else "")
    )
    print("-" * (78 + (8 if args.pope else 0)))

    for r in results:
        first = r.snapshots[0]
        last = r.snapshots[-1]
        growth = last.metal_active_mb - first.metal_active_mb
        pope_str = f"  {r.pope_accuracy:.0%}" if r.pope_accuracy is not None else ""
        print(
            f"{r.label:<22} {first.metal_active_mb:>9.0f}M {last.metal_active_mb:>9.0f}M "
            f"{growth:>+7.0f}M {first.kv_offset:>9d} {last.kv_offset:>9d} "
            f"{r.total_evictions:>6d}{pope_str}"
        )

    # Save
    out_path = args.output or (f"research/eval-results/streammem_stress_{args.frames}f.json")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    # Serialize
    serializable = []
    for r in results:
        d = {
            "label": r.label,
            "model": r.model,
            "n_frames": r.n_frames,
            "streaming_memory": r.streaming_memory,
            "budget": r.budget,
            "total_evictions": r.total_evictions,
            "total_time_s": r.total_time_s,
            "pope_accuracy": r.pope_accuracy,
            "pope_samples": r.pope_samples,
            "mem_start_mb": r.snapshots[0].metal_active_mb,
            "mem_end_mb": r.snapshots[-1].metal_active_mb,
            "mem_peak_mb": r.snapshots[-1].metal_peak_mb,
            "kv_start": r.snapshots[0].kv_offset,
            "kv_end": r.snapshots[-1].kv_offset,
            "latency_first_ms": r.snapshots[0].latency_ms,
            "latency_last_ms": r.snapshots[-1].latency_ms,
            # Sample every 10th snapshot to keep file small
            "snapshots": [asdict(s) for s in r.snapshots[::10]],
        }
        serializable.append(d)

    Path(out_path).write_text(json.dumps(serializable, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
