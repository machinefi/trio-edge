#!/usr/bin/env python3
"""Benchmark VLM models for crop-describe quality.

Tests each model on the same set of crops and scores them.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

MODELS = [
    "mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
    "mlx-community/Qwen3-VL-4B-Instruct-4bit",
    "mlx-community/Qwen2.5-VL-7B-Instruct-4bit",
    "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-4bit",
]

PERSON_PROMPT = (
    "Describe this person in one sentence: approximate age, "
    "gender, ethnicity, clothing, what they are carrying, "
    "and what they appear to be doing."
)

VEHICLE_PROMPT = (
    "Identify this vehicle in one sentence: type, color, "
    "make/model/brand if possible, and any distinguishing features."
)

KEYWORDS_PERSON = ["male", "female", "man", "woman", "wearing", "shirt", "dress",
                    "suit", "jacket", "pants", "bag", "walking", "standing",
                    "carrying", "age", "young", "old", "elderly", "middle"]

KEYWORDS_VEHICLE = ["sedan", "suv", "truck", "bus", "taxi", "toyota", "ford",
                     "honda", "bmw", "tesla", "white", "black", "red", "blue",
                     "yellow", "silver", "model", "pickup", "van"]


def get_test_crops():
    """Get crops from benchmark frames."""
    from trio_core.counter import PeopleCounter
    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")

    crops = []
    for fpath in sorted(Path("data/eval_benchmark").glob("frame_*.jpg"))[:3]:
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue
        counter._seen_ids.clear()
        counter._tracker = None
        counter._initialized = False
        result = counter.process(frame)
        for c in result.new_crops[:3]:
            if c.crop is not None and c.crop.size > 0:
                crop = c.crop
                ch, cw = crop.shape[:2]
                if ch < 56 or cw < 56:
                    scale = max(56 / ch, 56 / cw)
                    crop = cv2.resize(crop, (max(56, int(cw * scale)), max(56, int(ch * scale))))
                crops.append((c.class_name, crop))
    return crops


def bench_model(model_name: str, crops: list):
    """Benchmark a single model."""
    from trio_core import TrioCore, EngineConfig

    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    config = EngineConfig(model=model_name)
    engine = TrioCore(config)

    t0 = time.monotonic()
    try:
        engine.load()
    except Exception as e:
        print(f"  LOAD FAILED: {e}")
        return None
    load_time = time.monotonic() - t0
    print(f"  Load time: {load_time:.1f}s")

    results = []
    total_latency = 0
    total_keywords = 0
    descriptions = []

    for cls_name, crop in crops:
        prompt = PERSON_PROMPT if cls_name == "person" else VEHICLE_PROMPT
        keywords = KEYWORDS_PERSON if cls_name == "person" else KEYWORDS_VEHICLE

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        t0 = time.monotonic()
        try:
            r = engine.analyze_frame(rgb, prompt)
            text = r.text.strip() if r and r.text else ""
        except Exception as e:
            text = f"ERROR: {e}"
        latency = (time.monotonic() - t0) * 1000

        hits = sum(1 for kw in keywords if kw in text.lower())
        total_latency += latency
        total_keywords += hits
        descriptions.append(text)

        results.append({
            "class": cls_name,
            "latency_ms": round(latency),
            "text_len": len(text),
            "keyword_hits": hits,
            "text": text[:150],
        })

    n = len(results)
    if n == 0:
        return None

    avg_latency = total_latency / n
    avg_keywords = total_keywords / n
    unique_ratio = len(set(d[:50] for d in descriptions)) / max(len(descriptions), 1)

    # Score
    score = (
        avg_keywords * 10
        + unique_ratio * 20
        + min(20, max(0, (3000 - avg_latency) / 150))
    )
    score = min(100, max(0, score))

    print(f"  Avg latency: {avg_latency:.0f}ms")
    print(f"  Avg keywords: {avg_keywords:.1f}")
    print(f"  Unique ratio: {unique_ratio:.2f}")
    print(f"  Score: {score:.1f}/100")
    print(f"  Sample descriptions:")
    for r in results[:3]:
        print(f"    [{r['class']}] {r['text'][:120]}")

    return {
        "model": model_name,
        "load_time_s": round(load_time, 1),
        "avg_latency_ms": round(avg_latency),
        "avg_keywords": round(avg_keywords, 1),
        "unique_ratio": round(unique_ratio, 2),
        "score": round(score, 1),
        "n_crops": n,
    }


def main():
    crops = get_test_crops()
    print(f"Test crops: {len(crops)} ({sum(1 for c,_ in crops if c=='person')} person, {sum(1 for c,_ in crops if c!='person')} vehicle)")

    all_results = []
    for model in MODELS:
        result = bench_model(model, crops)
        if result:
            all_results.append(result)

    print(f"\n{'='*60}")
    print("FINAL RANKING")
    print(f"{'='*60}")
    print(f"{'Model':<50} {'Score':>6} {'Latency':>8} {'Keywords':>9} {'Load':>6}")
    print("-" * 85)
    for r in sorted(all_results, key=lambda x: x["score"], reverse=True):
        short = r["model"].split("/")[-1]
        print(f"{short:<50} {r['score']:>6.1f} {r['avg_latency_ms']:>7}ms {r['avg_keywords']:>8.1f} {r['load_time_s']:>5.1f}s")


if __name__ == "__main__":
    main()
