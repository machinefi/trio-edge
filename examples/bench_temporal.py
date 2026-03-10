#!/usr/bin/env python3
"""Temporal video understanding benchmark — quantify StreamMem's temporal value.

Tests whether accumulated KV cache (StreamMem) provides meaningful temporal
understanding compared to independent per-frame inference and multiframe
(all-frames-at-once) inference.

Test categories:
  1. Positional recall: "What was shown at position N?"
  2. Temporal ordering: "Did X appear before or after Y?"
  3. State tracking: "Object appeared at frame A, disappeared at frame B"
  4. Long-range counting: "How many distinct Xs appeared across all frames?"
  5. Change detection: "What changed between the early and late frames?"

Modes:
  - independent:       Only the last frame + question (no temporal context)
  - multiframe:        All frames stacked as video (gold standard)
  - accumulated:       StreamMem, large budget (no eviction)
  - accumulated_evict: StreamMem with small budget (eviction active)

Usage:
    python examples/bench_temporal.py
    python examples/bench_temporal.py -m mlx-community/Qwen2.5-VL-7B-Instruct-4bit
    python examples/bench_temporal.py --modes accumulated multiframe --seq-lengths 5 10 20
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ---------------------------------------------------------------------------
# Frame generators
# ---------------------------------------------------------------------------

def _make_frame(img: Image.Image) -> np.ndarray:
    """PIL Image → (1, 3, H, W) float32 in [0,1]."""
    arr = np.array(img.convert("RGB"))
    return arr.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0


def _get_font(size: int = 120):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
    except (OSError, IOError):
        return ImageFont.load_default()


def _draw_text(text: str, size: int = 336, font_size: int = 120,
               bg: str = "white", fg: str = "black") -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((size - tw) // 2, (size - th) // 2), text, fill=fg, font=font)
    return img


def _draw_shape(shape: str, color: str, size: int = 336,
                bg: str = "white") -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    m = size // 4
    box = [m, m, size - m, size - m]
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon([(size // 2, m), (m, size - m), (size - m, size - m)],
                     fill=color)
    elif shape == "star":
        # Simple 5-point star
        cx, cy, r = size // 2, size // 2, size // 3
        points = []
        for i in range(10):
            angle = np.pi / 2 + i * np.pi / 5
            rad = r if i % 2 == 0 else r * 0.4
            points.append((cx + rad * np.cos(angle), cy - rad * np.sin(angle)))
        draw.polygon(points, fill=color)
    return img


COLORS = ["red", "blue", "green", "yellow", "purple", "orange", "pink", "cyan"]
SHAPES = ["circle", "square", "triangle", "star"]
LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ---------------------------------------------------------------------------
# Test case definitions
# ---------------------------------------------------------------------------

@dataclass
class TemporalTest:
    name: str
    frames: list[np.ndarray]
    question: str
    ground_truth: str           # expected answer content
    category: str               # positional / ordering / state / counting / change
    seq_length: int             # number of frames
    check_fn: str = "substring" # "substring" or "number" or "yesno" or "order"


def make_positional_recall(n: int) -> list[TemporalTest]:
    """Show N distinct letters, ask about a specific position."""
    tests = []
    letters = LETTERS[:n]

    # Ask about early, middle, and late positions
    positions = []
    if n >= 3:
        positions = [0, n // 2, n - 1]
    elif n >= 2:
        positions = [0, n - 1]
    else:
        positions = [0]

    for pos in positions:
        frames = [_make_frame(_draw_text(c, font_size=140)) for c in letters]
        # 1-indexed for human readability
        human_pos = pos + 1
        ordinal = {1: "1st", 2: "2nd", 3: "3rd"}.get(human_pos, f"{human_pos}th")
        tests.append(TemporalTest(
            name=f"pos_recall_n{n}_at{human_pos}",
            frames=frames,
            question=(
                f"You have seen {n} images in sequence, each showing a single letter. "
                f"What letter was shown in the {ordinal} image? "
                f"Answer with just the letter."
            ),
            ground_truth=letters[pos],
            category="positional",
            seq_length=n,
        ))
    return tests


def make_ordering_tests(n: int) -> list[TemporalTest]:
    """Show N colored shapes, ask about temporal ordering."""
    tests = []
    items = list(zip(SHAPES[:min(n, len(SHAPES))],
                     COLORS[:min(n, len(COLORS))]))
    # Pad if needed
    while len(items) < n:
        idx = len(items) % len(SHAPES)
        cidx = len(items) % len(COLORS)
        items.append((SHAPES[idx], COLORS[cidx]))

    frames = [_make_frame(_draw_shape(s, c)) for s, c in items]

    # Ask if item A came before item B
    if n >= 3:
        # Early vs late
        a_idx, b_idx = 0, n - 1
        a_desc = f"{items[a_idx][1]} {items[a_idx][0]}"
        b_desc = f"{items[b_idx][1]} {items[b_idx][0]}"
        tests.append(TemporalTest(
            name=f"order_n{n}_first_vs_last",
            frames=frames,
            question=(
                f"You saw {n} images in sequence, each with a different colored shape. "
                f"Did the {a_desc} appear before or after the {b_desc}? "
                f"Answer with just 'before' or 'after'."
            ),
            ground_truth="before",
            category="ordering",
            seq_length=n,
        ))

        # Middle vs early
        m_idx = n // 2
        m_desc = f"{items[m_idx][1]} {items[m_idx][0]}"
        tests.append(TemporalTest(
            name=f"order_n{n}_mid_vs_first",
            frames=frames,
            question=(
                f"You saw {n} images in sequence, each with a different colored shape. "
                f"Did the {m_desc} appear before or after the {a_desc}? "
                f"Answer with just 'before' or 'after'."
            ),
            ground_truth="after",
            category="ordering",
            seq_length=n,
        ))

    return tests


def make_state_tracking(n: int) -> list[TemporalTest]:
    """Object appears for some frames then disappears — track state."""
    tests = []

    # Object present in first half, absent in second half
    n_present = max(2, n // 2)
    frames = []
    for i in range(n):
        if i < n_present:
            frames.append(_make_frame(_draw_shape("circle", "red")))
        else:
            frames.append(_make_frame(Image.new("RGB", (336, 336), "white")))

    tests.append(TemporalTest(
        name=f"state_disappear_n{n}",
        frames=frames,
        question=(
            f"You saw {n} images in sequence. Some showed a red circle, "
            f"and some were blank white. Is the red circle present in the "
            f"current (last) image? Answer yes or no."
        ),
        ground_truth="no",
        category="state",
        seq_length=n,
        check_fn="yesno",
    ))

    # Was there ever a red circle?
    tests.append(TemporalTest(
        name=f"state_ever_present_n{n}",
        frames=frames,
        question=(
            f"You saw {n} images in sequence. The current image is blank. "
            f"Was there a red circle in any of the earlier images? "
            f"Answer yes or no."
        ),
        ground_truth="yes",
        category="state",
        seq_length=n,
        check_fn="yesno",
    ))

    # Object appears LATE (absent first, present at end)
    frames_late = []
    n_absent = max(2, n // 2)
    for i in range(n):
        if i >= n_absent:
            frames_late.append(_make_frame(_draw_shape("square", "blue")))
        else:
            frames_late.append(_make_frame(Image.new("RGB", (336, 336), "white")))

    tests.append(TemporalTest(
        name=f"state_appear_late_n{n}",
        frames=frames_late,
        question=(
            f"You saw {n} images in sequence. Were the early images blank "
            f"(showing nothing) before the blue square appeared? "
            f"Answer yes or no."
        ),
        ground_truth="yes",
        category="state",
        seq_length=n,
        check_fn="yesno",
    ))

    return tests


def make_counting_tests(n: int) -> list[TemporalTest]:
    """Show N distinct objects, ask total count."""
    tests = []

    n_distinct = min(n, len(COLORS))
    frames = []
    for i in range(n):
        color = COLORS[i % n_distinct]
        shape = SHAPES[i % len(SHAPES)]
        frames.append(_make_frame(_draw_shape(shape, color)))

    tests.append(TemporalTest(
        name=f"count_distinct_n{n}",
        frames=frames,
        question=(
            f"You saw {n} images in sequence, each showing a colored shape. "
            f"How many distinct colors did you see across all images? "
            f"Answer with just the number."
        ),
        ground_truth=str(n_distinct),
        category="counting",
        seq_length=n,
        check_fn="number",
    ))

    return tests


def make_change_detection(n: int) -> list[TemporalTest]:
    """Background color changes partway through sequence."""
    tests = []

    # First half: white background with red circle
    # Second half: gray background with red circle
    switch_point = n // 2
    frames = []
    for i in range(n):
        bg = "white" if i < switch_point else "lightgray"
        frames.append(_make_frame(_draw_shape("circle", "red", bg=bg)))

    tests.append(TemporalTest(
        name=f"change_bg_n{n}",
        frames=frames,
        question=(
            f"You saw {n} images in sequence, each showing a red circle. "
            f"Did the background color change at some point in the sequence? "
            f"Answer yes or no."
        ),
        ground_truth="yes",
        category="change",
        seq_length=n,
        check_fn="yesno",
    ))

    return tests


# ---------------------------------------------------------------------------
# Answer checking
# ---------------------------------------------------------------------------

def check_answer(answer: str, ground_truth: str, check_fn: str = "substring") -> bool:
    answer_lower = answer.lower().strip()
    gt_lower = ground_truth.lower().strip()

    if check_fn == "substring":
        # For comma-separated, check each element
        parts = [p.strip() for p in gt_lower.split(",")]
        if len(parts) > 1:
            return all(p in answer_lower for p in parts)
        return gt_lower in answer_lower

    elif check_fn == "number":
        # Extract first number from answer
        nums = re.findall(r'\d+', answer_lower)
        if nums:
            return nums[0] == gt_lower
        # Also check spelled-out numbers
        word_to_num = {"one": "1", "two": "2", "three": "3", "four": "4",
                       "five": "5", "six": "6", "seven": "7", "eight": "8"}
        for word, num in word_to_num.items():
            if word in answer_lower and num == gt_lower:
                return True
        return False

    elif check_fn == "yesno":
        if gt_lower == "yes":
            return answer_lower.startswith("yes")
        else:
            return answer_lower.startswith("no")

    elif check_fn == "order":
        return gt_lower in answer_lower

    return gt_lower in answer_lower


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def run_independent(engine, test: TemporalTest, max_tokens: int) -> tuple[str, dict]:
    """Only feed the last frame."""
    t0 = time.monotonic()
    result = engine.analyze_video(test.frames[-1], test.question, max_tokens=max_tokens)
    elapsed = time.monotonic() - t0
    return result.text.strip(), {"time_s": round(elapsed, 2)}


def run_multiframe(engine, test: TemporalTest, max_tokens: int) -> tuple[str, dict]:
    """Stack all frames as video."""
    video = np.concatenate(test.frames, axis=0)
    t0 = time.monotonic()
    result = engine.analyze_video(video, test.question, max_tokens=max_tokens)
    elapsed = time.monotonic() - t0
    return result.text.strip(), {"time_s": round(elapsed, 2)}


def run_accumulated(engine, test: TemporalTest, max_tokens: int) -> tuple[str, dict]:
    """Feed frames one-by-one with StreamMem KV accumulation."""
    observe_prompt = "Observe this image carefully and remember what you see."
    mem_snapshots = []

    for i, frame in enumerate(test.frames[:-1]):
        t0 = time.monotonic()
        engine.analyze_video(frame, observe_prompt, max_tokens=8)
        elapsed = (time.monotonic() - t0) * 1000

        # Snapshot memory + KV state
        active_mb = mx.metal.get_active_memory() / 1e6
        kv_offset = 0
        try:
            pc = engine._backend._get_prompt_cache()
            if pc._kv_cache and hasattr(pc._kv_cache[0], "offset"):
                kv_offset = pc._kv_cache[0].offset
        except Exception:
            pass
        mem_snapshots.append({
            "frame": i, "active_mb": round(active_mb, 1),
            "kv_offset": kv_offset, "latency_ms": round(elapsed, 1),
        })

    # Final frame with the real question
    t0 = time.monotonic()
    result = engine.analyze_video(test.frames[-1], test.question, max_tokens=max_tokens)
    elapsed = time.monotonic() - t0

    meta = {
        "time_s": round(elapsed, 2),
        "mem_snapshots": mem_snapshots,
    }
    if mem_snapshots:
        meta["kv_offset_final"] = mem_snapshots[-1]["kv_offset"]
        meta["mem_final_mb"] = mem_snapshots[-1]["active_mb"]

    return result.text.strip(), meta


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Temporal video understanding benchmark"
    )
    parser.add_argument("--model", "-m",
                        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--modes", nargs="+",
                        default=["independent", "multiframe",
                                 "accumulated", "accumulated_evict"],
                        choices=["independent", "multiframe",
                                 "accumulated", "accumulated_evict"])
    parser.add_argument("--seq-lengths", nargs="+", type=int,
                        default=[5, 10, 20],
                        help="Sequence lengths to test")
    parser.add_argument("--evict-budget", type=int, default=500,
                        help="StreamMem budget for accumulated_evict mode")
    args = parser.parse_args()

    from trio_core import EngineConfig, TrioCore

    # Build test suite for each sequence length
    all_tests: list[TemporalTest] = []
    for n in args.seq_lengths:
        all_tests.extend(make_positional_recall(n))
        all_tests.extend(make_ordering_tests(n))
        all_tests.extend(make_state_tracking(n))
        all_tests.extend(make_counting_tests(n))
        all_tests.extend(make_change_detection(n))

    n_tests = len(all_tests)
    by_cat = {}
    for t in all_tests:
        by_cat.setdefault(t.category, []).append(t)

    print(f"Model: {args.model}")
    print(f"Tests: {n_tests} total")
    for cat, tests in sorted(by_cat.items()):
        print(f"  {cat}: {len(tests)}")
    print(f"Seq lengths: {args.seq_lengths}")
    print(f"Modes: {args.modes}")
    print(f"Evict budget: {args.evict_budget}")
    print()

    results = []

    for mode in args.modes:
        print(f"\n{'=' * 70}")
        print(f"  MODE: {mode}")
        print(f"{'=' * 70}")

        # Configure engine per mode
        sm_enabled = mode in ("accumulated", "accumulated_evict")
        budget = args.evict_budget if mode == "accumulated_evict" else 10000
        config = EngineConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            dedup_enabled=False,
            motion_enabled=False,
            streaming_memory_enabled=sm_enabled,
            streaming_memory_budget=budget,
        )
        engine = TrioCore(config)
        engine.load()

        mx.metal.reset_peak_memory()
        correct = 0

        for test in all_tests:
            # Reset accumulated context between test cases
            if mode in ("accumulated", "accumulated_evict"):
                engine.reset_context()

            if mode == "independent":
                answer, meta = run_independent(engine, test, args.max_tokens)
            elif mode == "multiframe":
                answer, meta = run_multiframe(engine, test, args.max_tokens)
            elif mode in ("accumulated", "accumulated_evict"):
                answer, meta = run_accumulated(engine, test, args.max_tokens)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            passed = check_answer(answer, test.ground_truth, test.check_fn)
            correct += int(passed)
            status = "PASS" if passed else "FAIL"

            print(f"  [{status}] {test.name:<35s} n={test.seq_length:<3d} "
                  f"{meta.get('time_s', 0):>5.1f}s")
            if not passed:
                print(f"         GT: {test.ground_truth}")
                print(f"         Got: {answer[:120]}")

            results.append({
                "mode": mode,
                "test": test.name,
                "category": test.category,
                "seq_length": test.seq_length,
                "ground_truth": test.ground_truth,
                "answer": answer[:200],
                "passed": passed,
                **{k: v for k, v in meta.items() if k != "mem_snapshots"},
            })

        peak_mb = mx.metal.get_peak_memory() / 1e6
        acc = correct / n_tests
        print(f"\n  {mode}: {correct}/{n_tests} = {acc:.0%}  "
              f"(peak metal: {peak_mb:.0f}MB)")
        del engine

    # ---------------------------------------------------------------------------
    # Summary tables
    # ---------------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("  RESULTS BY CATEGORY")
    print(f"{'=' * 70}")

    categories = sorted(set(t.category for t in all_tests))
    header = f"{'Mode':<22}"
    for cat in categories:
        header += f" {cat:>12}"
    header += f" {'TOTAL':>10}"
    print(header)
    print("-" * len(header))

    for mode in args.modes:
        mode_results = [r for r in results if r["mode"] == mode]
        row = f"{mode:<22}"
        total_pass, total_n = 0, 0
        for cat in categories:
            cat_results = [r for r in mode_results if r["category"] == cat]
            n_pass = sum(1 for r in cat_results if r["passed"])
            n_total = len(cat_results)
            total_pass += n_pass
            total_n += n_total
            row += f" {n_pass}/{n_total:>9}"
        row += f" {total_pass}/{total_n:>7}"
        print(row)

    # By sequence length
    print(f"\n{'=' * 70}")
    print("  RESULTS BY SEQUENCE LENGTH")
    print(f"{'=' * 70}")

    header2 = f"{'Mode':<22}"
    for n in args.seq_lengths:
        header2 += f" {'n=' + str(n):>10}"
    header2 += f" {'TOTAL':>10}"
    print(header2)
    print("-" * len(header2))

    for mode in args.modes:
        mode_results = [r for r in results if r["mode"] == mode]
        row = f"{mode:<22}"
        total_pass, total_n = 0, 0
        for n in args.seq_lengths:
            n_results = [r for r in mode_results if r["seq_length"] == n]
            n_pass = sum(1 for r in n_results if r["passed"])
            n_total = len(n_results)
            total_pass += n_pass
            total_n += n_total
            row += f" {n_pass}/{n_total:>7}"
        row += f" {total_pass}/{total_n:>7}"
        print(row)

    # Save
    out_path = args.output or "research/eval-results/bench_temporal.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
