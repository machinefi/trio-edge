# Experiment 9: Temporal Analytics Pipeline

**KPI:** K1 = Hourly MAPE <5%
**Baseline:** 13.1% per-frame MAPE (tiled=True, c=0.25, f=1.6, Kalman)
**Target:** Hourly MAPE <5% after temporal aggregation

## Methodology

- **Fixed eval:** Mall Dataset 2000 frames, simulated as 1 frame/30s over ~17 hours
- **Single metric:** Hourly MAPE (mean of per-bin absolute percentage errors)
- **Pipeline:** raw frames → Kalman filter → mean aggregation → 15-min bins → hourly rollup

## Results

### Per-frame baseline (tiled + Kalman + correction)

| Metric | Value |
|--------|-------|
| Per-frame MAPE | 13.1% |
| MAE | 3.94 |
| Correlation | 0.822 |
| Total accuracy | 98.0% |

### Aggregation sweep

| Window | Bins | MAPE |
|--------|------|------|
| 1 (per-frame) | 2000 | 13.11% |
| 5 | 400 | 12.33% |
| 20 | 100 | 8.46% |
| 50 | 40 | 7.82% |
| 100 (~hourly sim) | 20 | 5.60% |
| 200 | 10 | 4.26% |
| 500 | 4 | 2.55% |

### Aggregator pipeline result

| Level | Bins | MAPE | Status |
|-------|------|------|--------|
| 15-min | 67 | 5.98% | - |
| **Hourly** | 17 | **4.82%** | **K1 MET** |

### Key finding: mean > median for Kalman-smoothed data

| Aggregation | Hourly MAPE |
|-------------|-------------|
| median | 6.09% |
| **mean** | **4.80%** |
| p40 | 6.15% |

Kalman filter removes outliers upstream → mean preserves more information → lower MAPE.

### Anomaly detection

2 anomalies detected (z-threshold=2.0):
- 09:15: count 18 vs expected 30.5 (z=-2.1, low)
- 10:30: count 14 vs expected 30.5 (z=-2.77, medium)

## Implementation

### New files
- `src/trio_core/analytics/__init__.py`
- `src/trio_core/analytics/aggregator.py` — Aggregator, Bin, Sample
- `src/trio_core/analytics/anomaly.py` — AnomalyDetector, Anomaly

### Modified files
- `src/trio_core/counter.py` — CountResult now exposes velocity, kalman_confidence, raw_count

### Architecture
```
PeopleCounter.process()
    → raw YOLO detections
    → Kalman filter (state: [count, velocity])
    → correction_factor × smoothed_count
    → CountResult (count, raw_count, velocity, confidence)
        ↓
Aggregator
    → 15-min bins (mean of counts)
    → Hourly rollup (sum of 4 bins)
    → Daily rollup (sum of 24 hours)
        ↓
AnomalyDetector
    → z-score vs same-hour/same-dow historical
    → Flag >2σ deviations
```
