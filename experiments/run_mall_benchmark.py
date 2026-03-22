#!/usr/bin/env python3
"""Mall Dataset benchmark — measure YOLO+ByteTrack counting accuracy.

Processes 2000 frames from Mall Dataset, compares detection count
with ground truth, and produces accuracy metrics.

Usage:
    python experiments/run_mall_benchmark.py
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
OUTPUT_PATH = Path("experiments/results/mall_results.json")
MODEL_PATH = "models/yolov10n/onnx/model.onnx"


def load_ground_truth() -> list[int]:
    """Load per-frame people count from .mat file."""
    gt = scipy.io.loadmat(str(GT_PATH))
    counts = gt["count"].flatten().tolist()
    return counts


def run_benchmark():
    from trio_core.counter import PeopleCounter

    gt_counts = load_ground_truth()
    print(f"Ground truth: {len(gt_counts)} frames, range [{min(gt_counts)}-{max(gt_counts)}], mean {np.mean(gt_counts):.1f}")

    counter = PeopleCounter(model_path=MODEL_PATH, auto_calibrate=True)

    results = []
    total_time = 0
    frame_files = sorted(FRAMES_DIR.glob("seq_*.jpg"))[:len(gt_counts)]

    print(f"Processing {len(frame_files)} frames...")

    for i, fpath in enumerate(frame_files):
        frame = cv2.imread(str(fpath))
        if frame is None:
            print(f"  WARN: Cannot read {fpath}")
            continue

        t0 = time.monotonic()
        result = counter.process(frame)
        elapsed = (time.monotonic() - t0) * 1000

        total_time += elapsed
        gt = gt_counts[i]
        predicted = result.by_class.get("person", 0)

        results.append({
            "frame": i + 1,
            "gt": gt,
            "predicted": int(predicted),
            "error": int(predicted) - gt,
            "abs_error": abs(int(predicted) - gt),
            "latency_ms": round(elapsed, 1),
            "total_in": int(result.total_in),
            "total_out": int(result.total_out),
            "unique_tracks": len(counter._seen_ids),
        })

        if (i + 1) % 200 == 0:
            running_mae = np.mean([r["abs_error"] for r in results])
            print(f"  Frame {i+1}/{len(frame_files)} | MAE so far: {running_mae:.2f} | "
                  f"Last: gt={gt} pred={predicted} | Tracks: {len(counter._seen_ids)} | "
                  f"In: {result.total_in} Out: {result.total_out}")

    # Compute metrics
    gt_arr = np.array([r["gt"] for r in results])
    pred_arr = np.array([r["predicted"] for r in results])
    errors = pred_arr - gt_arr

    mae = float(np.mean(np.abs(errors)))
    mape = float(np.mean(np.abs(errors) / np.maximum(gt_arr, 1)) * 100)
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    total_gt = int(gt_arr.sum())
    total_pred = int(pred_arr.sum())
    total_accuracy = 1.0 - abs(total_gt - total_pred) / max(total_gt, 1)
    correlation = float(np.corrcoef(gt_arr, pred_arr)[0, 1])
    avg_latency = total_time / len(results)

    # Bias analysis
    under_count = int((errors < 0).sum())
    over_count = int((errors > 0).sum())
    exact_count = int((errors == 0).sum())

    metrics = {
        "mae": round(mae, 2),
        "mape": round(mape, 1),
        "rmse": round(rmse, 2),
        "total_accuracy": round(total_accuracy, 4),
        "correlation": round(correlation, 4),
        "total_gt": total_gt,
        "total_predicted": total_pred,
        "avg_gt_per_frame": round(float(gt_arr.mean()), 1),
        "avg_predicted_per_frame": round(float(pred_arr.mean()), 1),
        "avg_latency_ms": round(avg_latency, 1),
        "under_count_frames": under_count,
        "over_count_frames": over_count,
        "exact_frames": exact_count,
        "unique_tracks_total": results[-1]["unique_tracks"] if results else 0,
        "total_in": results[-1]["total_in"] if results else 0,
        "total_out": results[-1]["total_out"] if results else 0,
    }

    output = {
        "dataset": "Mall Dataset",
        "frames_processed": len(results),
        "metrics": metrics,
        "model": "YOLOv10n (Apache 2.0)",
        "tracker": "ByteTrack (supervision, MIT)",
        "confidence_threshold": 0.35,
        "auto_calibration": True,
        "calibration_frames": 30,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Print results
    print("\n" + "=" * 60)
    print("MALL DATASET BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Frames: {len(results)}")
    print(f"MAE:  {mae:.2f} people/frame")
    print(f"MAPE: {mape:.1f}%")
    print(f"RMSE: {rmse:.2f}")
    print(f"Total Accuracy: {total_accuracy:.1%}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Total GT: {total_gt} | Total Predicted: {total_pred}")
    print(f"Avg Latency: {avg_latency:.1f}ms/frame")
    print(f"Bias: under-count {under_count} | over-count {over_count} | exact {exact_count}")
    print(f"Unique tracks: {results[-1]['unique_tracks']}")
    print(f"Line crossings: IN={results[-1]['total_in']} OUT={results[-1]['total_out']}")
    print("=" * 60)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {OUTPUT_PATH}")

    # Also save per-frame data for plotting
    per_frame_path = OUTPUT_PATH.with_name("mall_per_frame.json")
    with open(per_frame_path, "w") as f:
        json.dump(results, f)
    print(f"Per-frame data saved to {per_frame_path}")


if __name__ == "__main__":
    run_benchmark()
