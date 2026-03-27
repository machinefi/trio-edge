#!/usr/bin/env python3
"""Cross-frame understanding test — does accumulated KV provide useful memory?

Tests whether a VLM can answer questions about PREVIOUS frames when KV cache
is accumulated across frames (StreamMem mode) vs independent per-frame inference.

Three test types:
  1. Sequence recall: "What letters did you see in order?"
  2. Disappearance: "Was there an object earlier that's now gone?"
  3. Counting: "How many distinct objects appeared across frames?"

Usage:
    python examples/streammem_crossframe.py
    python examples/streammem_crossframe.py -m mlx-community/Qwen2.5-VL-7B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Synthetic frame generators
# ---------------------------------------------------------------------------


def _make_frame(img: Image.Image) -> np.ndarray:
    """PIL Image → (1, 3, H, W) float32 in [0,1]."""
    arr = np.array(img.convert("RGB"))
    return arr.transpose(2, 0, 1).astype(np.float32)[np.newaxis] / 255.0


def _draw_text(
    text: str, size: int = 336, font_size: int = 120, bg: str = "white", fg: str = "black"
) -> Image.Image:
    """Draw large centered text on a solid background."""
    img = Image.new("RGB", (size, size), bg)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (size - tw) // 2
    y = (size - th) // 2
    draw.text((x, y), text, fill=fg, font=font)
    return img


def _draw_shape(shape: str, color: str, size: int = 336) -> Image.Image:
    """Draw a colored shape on white background."""
    img = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(img)
    margin = size // 4
    box = [margin, margin, size - margin, size - margin]
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        draw.polygon(
            [(size // 2, margin), (margin, size - margin), (size - margin, size - margin)],
            fill=color,
        )
    return img


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------


@dataclass
class TestCase:
    name: str
    frames: list[np.ndarray]  # list of (1,3,H,W) arrays
    question: str  # asked on the LAST frame
    ground_truth: str  # expected answer (substring match)
    category: str  # sequence / disappearance / counting


def make_sequence_tests() -> list[TestCase]:
    """Sequence recall: show letters one at a time, ask for the sequence."""
    tests = []

    # Test 1: A, B, C
    frames = [_make_frame(_draw_text(c)) for c in ["A", "B", "C"]]
    tests.append(
        TestCase(
            name="seq_ABC",
            frames=frames,
            question="You have seen several images in sequence, each showing a single letter. "
            "What were the letters in order? Just list them separated by commas.",
            ground_truth="A, B, C",
            category="sequence",
        )
    )

    # Test 2: X, Y, Z
    frames = [_make_frame(_draw_text(c)) for c in ["X", "Y", "Z"]]
    tests.append(
        TestCase(
            name="seq_XYZ",
            frames=frames,
            question="You have seen several images in sequence, each showing a single letter. "
            "What were the letters in order? Just list them separated by commas.",
            ground_truth="X, Y, Z",
            category="sequence",
        )
    )

    # Test 3: 1, 2, 3, 4, 5
    frames = [_make_frame(_draw_text(str(n), font_size=140)) for n in range(1, 6)]
    tests.append(
        TestCase(
            name="seq_12345",
            frames=frames,
            question="You have seen several images in sequence, each showing a single number. "
            "What were the numbers in order? Just list them separated by commas.",
            ground_truth="1, 2, 3, 4, 5",
            category="sequence",
        )
    )

    return tests


def make_disappearance_tests() -> list[TestCase]:
    """Object disappearance: show object, then remove it, ask about it."""
    tests = []

    # Red circle appears then disappears
    frames = [
        _make_frame(_draw_shape("circle", "red")),
        _make_frame(_draw_shape("circle", "red")),
        _make_frame(Image.new("RGB", (336, 336), "white")),  # empty
    ]
    tests.append(
        TestCase(
            name="disappear_circle",
            frames=frames,
            question="You have seen several images in sequence. The current image is blank. "
            "Was there an object in the earlier images? If yes, describe its shape and color.",
            ground_truth="red",
            category="disappearance",
        )
    )

    # Blue square appears then disappears
    frames = [
        _make_frame(_draw_shape("square", "blue")),
        _make_frame(_draw_shape("square", "blue")),
        _make_frame(Image.new("RGB", (336, 336), "white")),
    ]
    tests.append(
        TestCase(
            name="disappear_square",
            frames=frames,
            question="You have seen several images in sequence. The current image is blank. "
            "Was there an object in the earlier images? If yes, describe its shape and color.",
            ground_truth="blue",
            category="disappearance",
        )
    )

    # Text appears then disappears
    frames = [
        _make_frame(_draw_text("HELLO", font_size=80)),
        _make_frame(_draw_text("HELLO", font_size=80)),
        _make_frame(Image.new("RGB", (336, 336), "white")),
    ]
    tests.append(
        TestCase(
            name="disappear_text",
            frames=frames,
            question="You have seen several images in sequence. The current image is blank. "
            "Was there any text shown in the earlier images? If yes, what did it say?",
            ground_truth="HELLO",
            category="disappearance",
        )
    )

    return tests


def make_counting_tests() -> list[TestCase]:
    """Counting: different objects appear one per frame."""
    tests = []

    # 3 different colored shapes
    frames = [
        _make_frame(_draw_shape("circle", "red")),
        _make_frame(_draw_shape("square", "blue")),
        _make_frame(_draw_shape("triangle", "green")),
    ]
    tests.append(
        TestCase(
            name="count_3shapes",
            frames=frames,
            question="You have seen several images in sequence, each showing a different colored shape. "
            "How many distinct shapes did you see in total? Answer with just the number.",
            ground_truth="3",
            category="counting",
        )
    )

    return tests


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


def run_independent(engine, test: TestCase, max_tokens: int) -> str:
    """Independent mode: only feed the LAST frame + question."""
    result = engine.analyze_video(test.frames[-1], test.question, max_tokens=max_tokens)
    return result.text.strip()


def run_multiframe(engine, test: TestCase, max_tokens: int) -> str:
    """Multi-frame mode: feed ALL frames as a video (stacked along first dim)."""
    # Stack frames into (N, 3, H, W) — engine treats as video
    video = np.concatenate(test.frames, axis=0)
    result = engine.analyze_video(video, test.question, max_tokens=max_tokens)
    return result.text.strip()


def run_accumulated(engine, test: TestCase, max_tokens: int) -> str:
    """Accumulated KV mode: feed frames one-by-one, keeping KV cache.

    Frame 1..N-1: "Observe this image carefully."
    Frame N: actual question
    """
    observe_prompt = "Observe this image carefully and remember what you see."
    for frame in test.frames[:-1]:
        engine.analyze_video(frame, observe_prompt, max_tokens=8)

    # Final frame with the real question
    result = engine.analyze_video(test.frames[-1], test.question, max_tokens=max_tokens)
    return result.text.strip()


def check_answer(answer: str, ground_truth: str) -> bool:
    """Loose substring match — check if ground truth elements appear in answer."""
    answer_lower = answer.lower()
    # For comma-separated ground truth (e.g., "A, B, C"), check each element
    parts = [p.strip().lower() for p in ground_truth.split(",")]
    if len(parts) > 1:
        return all(p in answer_lower for p in parts)
    return ground_truth.lower() in answer_lower


def main():
    parser = argparse.ArgumentParser(description="Cross-frame understanding validation")
    parser.add_argument("--model", "-m", default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["independent", "multiframe", "accumulated"],
        choices=["independent", "multiframe", "accumulated"],
    )
    args = parser.parse_args()

    from trio_core import EngineConfig, TrioCore

    # Build all test cases
    tests = make_sequence_tests() + make_disappearance_tests() + make_counting_tests()
    print(
        f"Tests: {len(tests)} ({len(make_sequence_tests())} sequence, "
        f"{len(make_disappearance_tests())} disappearance, "
        f"{len(make_counting_tests())} counting)"
    )
    print(f"Model: {args.model}")
    print(f"Modes: {args.modes}")
    print()

    results = []

    for mode in args.modes:
        print(f"{'=' * 70}")
        print(f"  MODE: {mode}")
        print(f"{'=' * 70}")

        # Create engine per mode
        sm_enabled = mode == "accumulated"
        config = EngineConfig(
            model=args.model,
            max_tokens=args.max_tokens,
            dedup_enabled=False,
            motion_enabled=False,
            streaming_memory_enabled=sm_enabled,
            streaming_memory_budget=10000,  # large budget — we don't want eviction here
        )
        engine = TrioCore(config)
        engine.load()

        correct = 0
        for test in tests:
            # Reset accumulated context between test cases to prevent leaking
            if mode == "accumulated":
                engine.reset_context()

            t0 = time.monotonic()

            if mode == "independent":
                answer = run_independent(engine, test, args.max_tokens)
            elif mode == "multiframe":
                answer = run_multiframe(engine, test, args.max_tokens)
            elif mode == "accumulated":
                answer = run_accumulated(engine, test, args.max_tokens)
            else:
                raise ValueError(f"Unknown mode: {mode}")

            elapsed = time.monotonic() - t0
            passed = check_answer(answer, test.ground_truth)
            correct += int(passed)
            status = "PASS" if passed else "FAIL"

            print(f"  [{status}] {test.name:<25s}  {elapsed:>5.1f}s")
            print(f"         GT: {test.ground_truth}")
            print(f"         Got: {answer[:120]}")
            print()

            results.append(
                {
                    "mode": mode,
                    "test": test.name,
                    "category": test.category,
                    "ground_truth": test.ground_truth,
                    "answer": answer[:200],
                    "passed": passed,
                    "time_s": round(elapsed, 2),
                }
            )

        acc = correct / len(tests)
        print(f"  {mode}: {correct}/{len(tests)} = {acc:.0%}")
        print()
        del engine

    # Summary table
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Mode':<16} {'Sequence':>10} {'Disappear':>10} {'Counting':>10} {'Total':>10}")
    print("-" * 60)

    for mode in args.modes:
        mode_results = [r for r in results if r["mode"] == mode]
        by_cat = {}
        for r in mode_results:
            cat = r["category"]
            if cat not in by_cat:
                by_cat[cat] = {"pass": 0, "total": 0}
            by_cat[cat]["total"] += 1
            if r["passed"]:
                by_cat[cat]["pass"] += 1
        total_pass = sum(v["pass"] for v in by_cat.values())
        total = sum(v["total"] for v in by_cat.values())

        def fmt(cat):
            d = by_cat.get(cat, {"pass": 0, "total": 0})
            return f"{d['pass']}/{d['total']}"

        print(
            f"{mode:<16} {fmt('sequence'):>10} {fmt('disappearance'):>10} "
            f"{fmt('counting'):>10} {total_pass}/{total:>9}"
        )

    # Save
    out_path = args.output or "research/eval-results/crossframe_validation.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
