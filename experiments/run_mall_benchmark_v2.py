#!/usr/bin/env python3
"""Mall Dataset benchmark v2 — with new defaults (tiled=True, c=0.25, f=1.6).

Captures raw_count, velocity, kalman_confidence for temporal aggregation eval.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import scipy.io

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

FRAMES_DIR = Path("data/benchmarks/mall/mall_dataset/frames")
GT_PATH = Path("data/benchmarks/mall/mall_dataset/mall_gt.mat")
OUTPUT_PATH = Path("experiments/results/mall_v2_per_frame.json")
MODEL_PATH = "models/yolov10n/onnx/model.onnx"


def load_ground_truth() -> list[int]:
    gt = scipy.io.loadmat(str(GT_PATH))
    return gt["count"].flatten().tolist()


def run():
    from trio_core.counter import PeopleCounter

    gt_counts = load_ground_truth()
    print(f"GT: {len(gt_counts)} frames, range [{min(gt_counts)}-{max(gt_counts)}], mean {np.mean(gt_counts):.1f}")

    # New defaults: tiled=True, confidence=0.25, correction_factor=1.6
    counter = PeopleCounter(
        model_path=MODEL_PATH,
        auto_calibrate=False,  # use fixed correction_factor
        tiled=True,
        confidence=0.25,
        correction_factor=1.6,
    )

    results = []
    frame_files = sorted(FRAMES_DIR.glob("seq_*.jpg"))[:len(gt_counts)]
    print(f"Processing {len(frame_files)} frames (tiled=True, c=0.25, f=1.6)...")

    t_start = time.monotonic()
    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            continue

        result = counter.process(frame)
        gt = gt_counts[i]
        predicted = result.by_class.get("person", 0)

        results.append({
            "frame": i + 1,
            "gt": gt,
            "predicted": int(predicted),
            "raw_count": int(result.raw_count),
            "velocity": round(result.velocity, 3),
            "kalman_confidence": round(result.kalman_confidence, 3),
            "error": int(predicted) - gt,
            "abs_error": abs(int(predicted) - gt),
        })

        if (i + 1) % 500 == 0:
            elapsed = time.monotonic() - t_start
            mae = np.mean([r["abs_error"] for r in results])
            print(f"  {i+1}/{len(frame_files)} | MAE={mae:.2f} | last: gt={gt} pred={predicted} raw={result.raw_count} | {elapsed:.0f}s")

    # Compute metrics
    gt_arr = np.array([r["gt"] for r in results])
    pred_arr = np.array([r["predicted"] for r in results])
    raw_arr = np.array([r["raw_count"] for r in results])
    errors = pred_arr - gt_arr

    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.maximum(gt_arr, 1)) * 100)
    correlation = float(np.corrcoef(gt_arr, pred_arr)[0, 1])
    total_gt = int(gt_arr.sum())
    total_pred = int(pred_arr.sum())
    total_acc = 1.0 - abs(total_gt - total_pred) / max(total_gt, 1)

    print(f"\n{'='*60}")
    print(f"MALL v2 RESULTS (tiled=True, c=0.25, f=1.6)")
    print(f"{'='*60}")
    print(f"Per-frame MAPE: {mape:.1f}%")
    print(f"MAE: {mae:.2f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Total accuracy: {total_acc:.1%}")
    print(f"Total GT: {total_gt} | Predicted: {total_pred}")
    print(f"Avg raw detections: {raw_arr.mean():.1f}")
    print(f"Avg predicted (after Kalman+correction): {pred_arr.mean():.1f}")
    print(f"Time: {time.monotonic()-t_start:.0f}s")

    # Save per-frame results
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results))
    print(f"\nPer-frame data saved to {OUTPUT_PATH}")

    # Save summary
    summary = {
        "dataset": "Mall Dataset",
        "frames": len(results),
        "config": {"tiled": True, "confidence": 0.25, "correction_factor": 1.6, "kalman": True},
        "metrics": {
            "mae": round(mae, 2),
            "mape": round(mape, 1),
            "correlation": round(correlation, 4),
            "total_accuracy": round(total_acc, 4),
            "total_gt": total_gt,
            "total_predicted": total_pred,
        },
    }
    summary_path = OUTPUT_PATH.with_name("mall_v2_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    run()
