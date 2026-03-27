"""Exp9: Temporal Aggregation Benchmark on Mall Dataset.

Uses existing per-frame results (gt + predicted) to simulate temporal
aggregation at different window sizes and measure hourly MAPE improvement.

Simulates real deployment: 1 frame / 30s sampling, 15-min bins, hourly rollup.
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trio_core.analytics.aggregator import Aggregator, Sample
from trio_core.analytics.anomaly import AnomalyDetector


def load_per_frame() -> list[dict]:
    """Load per-frame Mall benchmark results (v2 with tiled+correction)."""
    # Prefer v2 (tiled=True, c=0.25, f=1.6) over v1
    p_v2 = Path(__file__).parent / "results" / "mall_v2_per_frame.json"
    p_v1 = Path(__file__).parent / "results" / "mall_per_frame.json"
    p = p_v2 if p_v2.exists() else p_v1
    data = json.loads(p.read_text())
    print(f"Loaded from: {p.name}")
    return data


def simulate_timestamps(frames: list[dict], fps: float = 10.0) -> list[dict]:
    """Assign realistic timestamps to frames.

    Mall Dataset: 2000 frames at ~10fps = 200 seconds of video.
    We simulate as if frames were captured at 1/30s intervals (real deployment rate)
    to get more time spread for meaningful aggregation.
    """
    base = datetime(2026, 3, 22, 8, 0, 0, tzinfo=timezone.utc)
    for i, f in enumerate(frames):
        # Simulate 1 frame per 30 seconds over ~16.7 hours
        f["timestamp"] = base + timedelta(seconds=i * 30)
    return frames


def frames_to_samples(frames: list[dict]) -> list[Sample]:
    """Convert per-frame results to Sample objects."""
    return [
        Sample(
            timestamp=f["timestamp"],
            count=f["predicted"],
            raw_count=f.get("raw_count", f["predicted"]),
            camera_id="cam_mall",
        )
        for f in frames
    ]


def compute_mape(predicted: list[float], gt: list[float]) -> float:
    """Compute MAPE between predicted and ground truth lists."""
    if not predicted or not gt:
        return 0.0
    errors = []
    for p, g in zip(predicted, gt):
        if g > 0:
            errors.append(abs(p - g) / g)
    return (sum(errors) / len(errors) * 100) if errors else 0.0


def run_window_sweep(frames: list[dict]) -> list[dict]:
    """Sweep different aggregation window sizes and measure MAPE reduction."""
    results = []
    gt_arr = np.array([f["gt"] for f in frames])
    pred_arr = np.array([f["predicted"] for f in frames])

    # Baseline: per-frame
    baseline_mape = float(np.mean(np.abs(pred_arr - gt_arr) / np.maximum(gt_arr, 1)) * 100)
    results.append(
        {
            "window": 1,
            "bins": len(frames),
            "mape": round(baseline_mape, 2),
            "label": "per-frame",
        }
    )

    # Aggregation windows
    for window in [5, 10, 20, 30, 50, 100, 200, 500]:
        n_bins = len(frames) // window
        if n_bins < 2:
            continue

        bin_pred = []
        bin_gt = []
        for i in range(n_bins):
            start = i * window
            end = start + window
            bin_pred.append(statistics.median(pred_arr[start:end]))
            bin_gt.append(statistics.median(gt_arr[start:end]))

        mape = compute_mape(bin_pred, bin_gt)
        results.append(
            {
                "window": window,
                "bins": n_bins,
                "mape": round(mape, 2),
                "label": f"window-{window}",
            }
        )

    return results


def run_aggregator_eval(frames: list[dict]) -> dict:
    """Run the actual Aggregator module on simulated timestamps."""
    frames = simulate_timestamps(frames)
    samples = frames_to_samples(frames)

    # Also create GT samples for comparison
    gt_by_ts = {f["timestamp"]: f["gt"] for f in frames}

    agg = Aggregator(bin_minutes=15, agg_method="mean")

    # 15-min bins
    bins_15m = agg.aggregate(samples, level="bin")
    # Hourly bins
    bins_hourly = agg.aggregate(samples, level="hourly")
    # Daily bins
    bins_daily = agg.aggregate(samples, level="daily")

    # Compute GT for each 15-min bin using same aggregation as predicted
    gt_15m = []
    for b in bins_15m:
        gt_counts = [gt_by_ts[f["timestamp"]] for f in frames if b.start <= f["timestamp"] < b.end]
        if gt_counts:
            gt_15m.append(round(statistics.mean(gt_counts)))
        else:
            gt_15m.append(0)

    pred_15m = [b.count for b in bins_15m]
    mape_15m = compute_mape(pred_15m, gt_15m)

    # Compute GT for hourly bins: mean of frames in each hour
    gt_hourly_vals = []
    pred_hourly_vals = []
    for b in bins_hourly:
        gt_in_bin = [gt_by_ts[f["timestamp"]] for f in frames if b.start <= f["timestamp"] < b.end]
        pred_in_bin = [f["predicted"] for f in frames if b.start <= f["timestamp"] < b.end]
        if gt_in_bin and pred_in_bin:
            gt_hourly_vals.append(round(statistics.mean(gt_in_bin)))
            pred_hourly_vals.append(round(statistics.mean(pred_in_bin)))

    mape_hourly = compute_mape(pred_hourly_vals, gt_hourly_vals)

    # Anomaly detection
    detector = AnomalyDetector(z_threshold=2.0)
    anomalies = detector.detect(bins_15m)

    return {
        "bins_15m": len(bins_15m),
        "bins_hourly": len(bins_hourly),
        "bins_daily": len(bins_daily),
        "mape_15m": round(mape_15m, 2),
        "mape_hourly": round(mape_hourly, 2),
        "anomalies_detected": len(anomalies),
        "anomaly_details": [
            {
                "time": a.timestamp.strftime("%H:%M"),
                "expected": a.expected,
                "actual": a.actual,
                "z": a.z_score,
                "severity": a.severity,
            }
            for a in anomalies[:5]
        ],
        "sample_15m_bins": [
            {
                "start": b.start.strftime("%H:%M"),
                "count": b.count,
                "samples": b.samples,
                "confidence": b.confidence,
            }
            for b in bins_15m[:8]
        ],
        "sample_hourly_bins": [
            {"start": b.start.strftime("%H:%M"), "count": b.count, "samples": b.samples}
            for b in bins_hourly[:6]
        ],
    }


def main():
    frames = load_per_frame()
    print("=" * 70)
    print("Exp9: Temporal Aggregation Benchmark")
    print("=" * 70)
    print(f"Frames: {len(frames)}")

    # Round 1: Window sweep (raw aggregation effect)
    print("\n--- WINDOW SWEEP (median aggregation vs per-frame) ---")
    sweep = run_window_sweep(frames)
    for r in sweep:
        indicator = " <-- TARGET" if r["mape"] < 5.0 and r["window"] > 1 else ""
        print(
            f"  Window={r['window']:>4d} | Bins={r['bins']:>4d} | MAPE={r['mape']:>6.2f}%{indicator}"
        )

    # Round 2: Full Aggregator pipeline
    print("\n--- AGGREGATOR PIPELINE (15-min bins → hourly → daily) ---")
    agg_result = run_aggregator_eval(frames)
    print(f"  15-min bins: {agg_result['bins_15m']}")
    print(f"  Hourly bins: {agg_result['bins_hourly']}")
    print(f"  Daily bins:  {agg_result['bins_daily']}")
    print(f"  MAPE (15-min): {agg_result['mape_15m']:.2f}%")
    print(f"  MAPE (hourly): {agg_result['mape_hourly']:.2f}%")
    print(f"  K1 hourly met: {'YES' if agg_result['mape_hourly'] < 5.0 else 'NO'}")
    print(f"  Anomalies: {agg_result['anomalies_detected']}")

    if agg_result["anomaly_details"]:
        print("  Top anomalies:")
        for a in agg_result["anomaly_details"]:
            print(
                f"    {a['time']}: expected={a['expected']}, actual={a['actual']}, z={a['z']}, severity={a['severity']}"
            )

    print("\n  Sample 15-min bins:")
    for b in agg_result["sample_15m_bins"]:
        print(
            f"    {b['start']}: count={b['count']}, samples={b['samples']}, conf={b['confidence']}"
        )

    print("\n  Sample hourly bins:")
    for b in agg_result["sample_hourly_bins"]:
        print(f"    {b['start']}: count={b['count']}, samples={b['samples']}")

    # Save results
    full_result = {
        "experiment": "exp9_temporal_aggregation",
        "dataset": "Mall Dataset (2000 frames)",
        "window_sweep": sweep,
        "aggregator_pipeline": agg_result,
        "k1_hourly_met": agg_result["mape_hourly"] < 5.0,
    }

    results_path = Path(__file__).parent / "results" / "exp9_temporal.json"
    results_path.write_text(json.dumps(full_result, indent=2))
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
