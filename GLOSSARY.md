# Trio Enterprise — Metrics Glossary

Standard definitions aligned with Frigate NVR / Verkada / industry conventions.

## Core Concepts

```
Detection  = YOLO finds a bounding box in one frame (raw, high-frequency)
Tracking   = ByteTrack links detections across frames into a trajectory (per-object)
Event      = A tracked object's lifecycle (enter → dwell → exit) that meets criteria
Insight    = Multi-signal intelligence from Synthesis layer (the "wow" output)
```

**Detection ≠ Event ≠ Insight.** A busy scene has thousands of detections,
dozens of events, and maybe 1-2 insights per hour.

## Per-Layer Metrics

### Layer 1: Perception
| Metric | Definition | Unit |
|--------|-----------|------|
| `frames_processed` | Total frames read from stream | count |
| `motion_detected` | Frames where MOG2 found significant contours | count |
| `motion_rate` | motion_detected / frames_processed | % |
| `hash_skipped` | Frames rejected by perceptual hash (identical) | count |
| `stream_status` | connected / reconnecting / stalled | enum |

### Layer 2: Extraction
| Metric | Definition | Unit |
|--------|-----------|------|
| `yolo_triggers` | Times YOLO ran (motion confirmed) | count |
| `vlm_calls` | Times VLM ran (scene changed) | count |
| `vlm_skip_rate` | Frames where YOLO ran but VLM skipped (no change) | % |
| `avg_vlm_latency` | Mean VLM inference time | seconds |
| `detections` | Total bounding boxes produced by YOLO | count |

### Layer 3: Memory
| Metric | Definition | Unit |
|--------|-----------|------|
| `entities_learned` | Unique entities in semantic memory | count |
| `vehicles_known` | Known vehicles (regular + unknown) | count |
| `vehicles_regular` | Vehicles seen ≥3 times (trusted) | count |
| `persons_known` | Known persons | count |
| `episodes` | Events stored in episodic memory | count |
| `memory_actions` | ADD/UPDATE/DELETE decisions by memory agent | count |

### Layer 4: Synthesis
| Metric | Definition | Unit |
|--------|-----------|------|
| `baseline_coverage` | % of (hour × dow) slots with ≥10 samples | % |
| `baseline_slots` | Number of (camera, hour, dow, metric) baselines | count |
| `insights_generated` | Insights produced (anomaly + dwell + pattern) | count |
| `insights_high` | High/critical severity insights | count |
| `patterns_discovered` | Recurring patterns found by FP-Growth | count |
| `active_dwells` | Entities currently being tracked for dwell | count |

### System
| Metric | Definition | Unit |
|--------|-----------|------|
| `pipelines_running` | Active pipeline processes | count |
| `pipelines_total` | Registered cameras | count |
| `ram_used_gb` | System RAM used | GB |
| `ram_free_gb` | System RAM available | GB |
| `cpu_idle_pct` | CPU idle percentage | % |

## Status Indicators

| Status | Color | Meaning |
|--------|-------|---------|
| `active` | Green | Pipeline running, producing events |
| `learning` | Yellow | Baseline building, <10 samples per slot |
| `ready` | Green | Baseline ready, insights being generated |
| `idle` | Gray | Pipeline running but no motion detected |
| `offline` | Red | Pipeline not running |
| `stalled` | Red | Stream connected but no frames received |

## Event vs Insight

### Event (what happened)
```json
{
  "type": "vehicle_arrived",
  "timestamp": "2026-03-26T14:32:00Z",
  "camera_id": "cam_home_front",
  "description": "Black Tesla Model Y SUV parked in driveway",
  "people_count": 0,
  "vehicle_count": 3,
  "severity": "info"
}
```

### Insight (what it means)
```json
{
  "type": "security",
  "severity": "high",
  "confidence": 0.85,
  "narrative": "Dark Gray SUV appeared 4th time during quiet hours (22:00-06:00).
                Average dwell: 2.3h. NOT a regular vehicle. Recommend investigation.",
  "signals": ["frequency: 4x night", "dwell: 2.3h", "regular: false"],
  "actionable": "Investigate"
}
```
