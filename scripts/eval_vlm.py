#!/usr/bin/env python3
"""Autoresearch-style VLM evaluation harness.

Inspired by Karpathy's autoresearch: fixed eval set, single metric,
iterate on prompts/params until quality converges.

Usage:
    python scripts/eval_vlm.py --prompt "Describe this person..." --frames data/eval_benchmark/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Default prompts to evaluate
PERSON_PROMPTS = {
    "v1_basic": "Describe this person briefly.",
    "v2_structured": (
        "Describe this person in one sentence: approximate age, "
        "gender, ethnicity, clothing, what they are carrying, "
        "and what they appear to be doing."
    ),
    "v3_detailed": (
        "You are a surveillance analyst. Describe this individual precisely: "
        "1) Gender and estimated age range "
        "2) Ethnicity/race "
        "3) Clothing (top, bottom, footwear, accessories) "
        "4) Items carried (bags, phones, etc) "
        "5) Activity (walking direction, standing, running, etc). "
        "Be factual and concise. One paragraph."
    ),
    "v4_concise": (
        "Describe this person: age/gender, clothing, items carried, activity. "
        "Example: '~30yo male, white, gray suit + red tie, carrying briefcase, walking east.' "
        "Be specific and concise — max 30 words."
    ),
}

VEHICLE_PROMPTS = {
    "v1_basic": "What vehicle is this?",
    "v2_structured": (
        "Identify this vehicle: type, color, make/model/brand if possible, "
        "and any distinguishing features. One sentence."
    ),
    "v3_detailed": (
        "You are a vehicle identification expert. Identify: "
        "1) Vehicle type (sedan, SUV, truck, bus, taxi, van) "
        "2) Color "
        "3) Make and model (e.g., Toyota Camry, Ford F-150) "
        "4) Approximate year range "
        "5) Any distinguishing features (damage, stickers, license plate visible). "
        "Be specific. One paragraph."
    ),
}


def evaluate_prompt(engine, frames_dir: str, prompt: str, class_filter: str = "person") -> dict:
    """Run a prompt against all benchmark frames and score it."""
    from trio_core.counter import PeopleCounter

    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")
    results = []
    total_latency = 0
    total_chars = 0
    descriptions = []

    for fname in sorted(Path(frames_dir).glob("frame_*.jpg")):
        frame = cv2.imread(str(fname))
        if frame is None:
            continue

        # Get crops
        result = counter.process(frame)
        crops = [c for c in result.new_crops if c.class_name == class_filter]

        # If no new crops (tracker already saw them), force re-detect
        if not crops:
            # Reset tracker to get fresh crops
            counter._seen_ids.clear()
            counter._tracker = None
            counter._initialized = False
            result = counter.process(frame)
            crops = [c for c in result.new_crops if c.class_name == class_filter]

        for crop_info in crops[:3]:  # max 3 per frame
            if crop_info.crop is None or crop_info.crop.size == 0:
                continue
            # VLM needs min 28x28 pixels; pad small crops
            ch, cw = crop_info.crop.shape[:2]
            crop = crop_info.crop
            if ch < 56 or cw < 56:
                scale = max(56 / ch, 56 / cw)
                crop = cv2.resize(crop, (max(56, int(cw * scale)), max(56, int(ch * scale))))

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            t0 = time.monotonic()
            vlm_result = engine.analyze_frame(crop_rgb, prompt)
            latency = (time.monotonic() - t0) * 1000

            text = vlm_result.text.strip() if vlm_result and vlm_result.text else ""
            total_latency += latency
            total_chars += len(text)
            descriptions.append(text)

            results.append({
                "frame": fname.name,
                "track_id": crop_info.track_id,
                "class": crop_info.class_name,
                "latency_ms": round(latency),
                "text_len": len(text),
                "text": text[:200],
            })

    if not results:
        return {"score": 0, "error": "no crops found"}

    # Score the prompt
    n = len(results)
    avg_latency = total_latency / n
    avg_chars = total_chars / n

    # Scoring criteria (autoresearch-style single metric)
    # - Specificity: does it mention age/gender/clothing? (check for keywords)
    specificity_keywords = {
        "person": ["male", "female", "man", "woman", "old", "young", "wearing",
                    "shirt", "dress", "suit", "jacket", "pants", "bag", "walking",
                    "standing", "running", "carrying", "~", "yo", "age"],
        "car": ["sedan", "suv", "truck", "bus", "taxi", "toyota", "ford", "honda",
                "bmw", "tesla", "white", "black", "red", "blue", "yellow", "model"],
    }
    keywords = specificity_keywords.get(class_filter, specificity_keywords["person"])

    keyword_hits = 0
    for desc in descriptions:
        desc_lower = desc.lower()
        hits = sum(1 for kw in keywords if kw in desc_lower)
        keyword_hits += hits

    avg_keyword_hits = keyword_hits / n if n > 0 else 0

    # Hallucination check: does it repeat the same description?
    unique_ratio = len(set(d[:50] for d in descriptions)) / max(len(descriptions), 1)

    # Conciseness: penalize too long or too short
    conciseness = 1.0
    if avg_chars > 300:
        conciseness = 300 / avg_chars  # penalty for verbosity
    elif avg_chars < 20:
        conciseness = avg_chars / 20  # penalty for too terse

    # Composite score (0-100)
    score = (
        avg_keyword_hits * 10  # specificity (max ~50)
        + unique_ratio * 20    # diversity (max 20)
        + conciseness * 20     # conciseness (max 20)
        + max(0, (5000 - avg_latency) / 500)  # speed bonus (max 10)
    )
    score = min(100, max(0, score))

    return {
        "score": round(score, 1),
        "n_crops": n,
        "avg_latency_ms": round(avg_latency),
        "avg_chars": round(avg_chars),
        "avg_keyword_hits": round(avg_keyword_hits, 1),
        "unique_ratio": round(unique_ratio, 2),
        "conciseness": round(conciseness, 2),
        "descriptions": descriptions[:5],
    }


def main():
    parser = argparse.ArgumentParser(description="VLM eval harness")
    parser.add_argument("--frames", default="data/eval_benchmark/")
    parser.add_argument("--class-filter", default="person")
    parser.add_argument("--prompt", help="Single prompt to evaluate")
    parser.add_argument("--all", action="store_true", help="Evaluate all built-in prompts")
    args = parser.parse_args()

    from trio_core import TrioCore, EngineConfig
    print("Loading VLM...")
    engine = TrioCore(EngineConfig())
    engine.load()
    print(f"Ready: {engine.config.model}\n")

    if args.prompt:
        print(f"Prompt: {args.prompt[:80]}...")
        result = evaluate_prompt(engine, args.frames, args.prompt, args.class_filter)
        print(f"Score: {result['score']}/100")
        print(f"  Crops evaluated: {result.get('n_crops', 0)}")
        print(f"  Avg latency: {result.get('avg_latency_ms', 0)}ms")
        print(f"  Avg chars: {result.get('avg_chars', 0)}")
        print(f"  Keyword hits: {result.get('avg_keyword_hits', 0)}")
        print(f"  Unique ratio: {result.get('unique_ratio', 0)}")
        print(f"\nSample descriptions:")
        for d in result.get("descriptions", [])[:3]:
            print(f"  - {d[:150]}")
        return

    # Evaluate all prompts
    prompts = PERSON_PROMPTS if args.class_filter == "person" else VEHICLE_PROMPTS
    if args.all or not args.prompt:
        print(f"{'Prompt':<15} {'Score':>6} {'Latency':>8} {'Chars':>6} {'Keywords':>9} {'Unique':>7}")
        print("-" * 60)

        best_name = ""
        best_score = 0

        for name, prompt in prompts.items():
            result = evaluate_prompt(engine, args.frames, prompt, args.class_filter)
            score = result["score"]
            print(f"{name:<15} {score:>6.1f} {result.get('avg_latency_ms', 0):>7}ms {result.get('avg_chars', 0):>5} {result.get('avg_keyword_hits', 0):>8.1f} {result.get('unique_ratio', 0):>7.2f}")

            if score > best_score:
                best_score = score
                best_name = name

        print(f"\nBest: {best_name} (score: {best_score:.1f})")
        print(f"\nSample from {best_name}:")
        result = evaluate_prompt(engine, args.frames, prompts[best_name], args.class_filter)
        for d in result.get("descriptions", [])[:3]:
            print(f"  - {d[:150]}")


if __name__ == "__main__":
    main()
