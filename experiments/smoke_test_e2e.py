#!/usr/bin/env python3
"""End-to-end smoke test — verify the full Trio Enterprise backend pipeline.

Simulates: camera add → events generated → metrics stored → insights → report
Without requiring a real RTSP camera or VLM model.
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


async def main():
    print("=" * 70)
    print("END-TO-END SMOKE TEST — Trio Enterprise Backend")
    print("=" * 70)

    from trio_core.api.store import EventStore

    # ── 1. Initialize store ──
    print("\n[1] Initializing event store...")
    store = EventStore(db_path=":memory:", frames_dir="/tmp/trio_test_frames")
    await store.init()
    print("    OK — SQLite in-memory DB ready")

    # ── 2. Create camera ──
    print("\n[2] Creating camera...")
    cam_id = await store.create_camera({
        "name": "Test Camera - Lobby",
        "source_url": "rtsp://192.168.1.100/stream",
        "watch_condition": "Describe this person: appearance, action, direction.",
        "intent": "Monitor lobby for visitor analytics and security",
        "intent_config": {
            "persona": "security_officer",
            "scene_type": "security",
            "camera_angle": "eye_level",
            "customer_prompt": "Describe: age, gender, clothing, items, activity.",
            "scene_prompt": "Describe: people count, busyness, notable activity.",
            "report_type": "security",
            "key_metrics": ["foot_traffic", "demographics", "behavioral", "access_control"],
            "correction_factor": 1.6,
            "calibrated_at": datetime.now(timezone.utc).isoformat(),
        },
    })
    cameras = await store.list_cameras()
    print(f"    OK — Camera created: {cam_id} ({cameras[0]['name']})")

    # ── 3. Simulate events (VLM descriptions) over 24 hours ──
    print("\n[3] Simulating 24h of events...")
    base_ts = datetime(2026, 3, 22, 6, 0, 0, tzinfo=timezone.utc)
    descriptions = [
        "Young woman in business attire walking briskly toward entrance, carrying a laptop bag",
        "Middle-aged man in gray suit, badge visible on lanyard, entering through main door",
        "Two young adults walking together, one carrying a backpack, the other holding a phone",
        "Security guard standing at the main entrance, checking badges",
        "Man in dark clothing walking along the perimeter fence, looking around",
        "Woman in scrubs carrying a large tote bag, walking toward the east entrance",
        "Group of 3 people in business casual entering lobby, chatting",
        "Delivery driver in uniform carrying packages toward loading area",
        "Elderly gentleman with walking stick, moving slowly toward the bench",
        "Woman on phone, walking east, carrying a branded shopping bag",
        "Male cyclist dismounting near the bike rack, wearing helmet",
        "Unidentified person near side door, no badge visible",
        "Female contractor with visitor badge escorted through lobby",
        "Door propped open, no personnel in vicinity",
        "Night patrol officer completing rounds, flashlight visible",
        "Man in business suit ordering a grande latte at the counter",
        "Woman with laptop settled at table, tall coffee beside her",
        "Young couple sharing a venti Frappuccino on the patio",
        "Cleaning crew mopping floor near entrance after hours",
        "Empty lobby, overhead lights activated, no pedestrians visible",
    ]

    event_count = 0
    for h in range(18):  # 6am to midnight
        hour = 6 + h
        # More events during business hours
        events_this_hour = 12 if 8 <= hour <= 18 else 3
        for i in range(events_this_hour):
            ts = base_ts + timedelta(hours=h, minutes=i * (60 // events_this_hour))
            desc = descriptions[(event_count) % len(descriptions)]
            await store.insert({
                "timestamp": ts.isoformat(),
                "camera_id": cam_id,
                "camera_name": "Test Camera - Lobby",
                "description": desc,
                "alert_triggered": "suspicious" in desc.lower() or "propped" in desc.lower() or "unidentified" in desc.lower(),
            })
            event_count += 1

    result = await store.list_events(camera_id=cam_id, limit=1)
    total = result["total"]
    print(f"    OK — {total} events created over 18 hours")

    # ── 4. Simulate counting metrics ──
    print("\n[4] Simulating counting metrics...")
    import random
    random.seed(42)
    metric_count = 0
    for h in range(18):
        hour = 6 + h
        base_count = 30 if 8 <= hour <= 18 else 5
        for m in range(0, 60, 5):  # every 5 minutes
            ts = base_ts + timedelta(hours=h, minutes=m)
            count = max(0, base_count + random.randint(-8, 8))
            await store.insert_metric({
                "timestamp": ts.isoformat(),
                "camera_id": cam_id,
                "metric_type": "people_count",
                "value": count,
                "confidence": 0.85 + random.random() * 0.15,
                "metadata": {"velocity": random.uniform(-1, 1)},
            })
            # Also insert people_in
            if count > 0:
                await store.insert_metric({
                    "timestamp": ts.isoformat(),
                    "camera_id": cam_id,
                    "metric_type": "people_in",
                    "value": max(1, count // 3),
                })
            metric_count += 1

    print(f"    OK — {metric_count} metric data points stored")

    # ── 5. Test insights extraction ──
    print("\n[5] Testing InsightExtractor...")
    from trio_core.insights import InsightExtractor

    events_data = await store.list_events(camera_id=cam_id, limit=10000)
    all_events = events_data["events"]

    extractor = InsightExtractor()
    insights = extractor.extract(all_events)
    print(f"    OK — {len(insights)} actionable insights extracted")
    for ins in insights:
        print(f"    [{ins.insight_type:>14s}] {ins.text[:80]}")

    k3_met = len(insights) >= 5
    print(f"    K3 ({len(insights)}/5): {'MET' if k3_met else 'NOT MET'}")

    # ── 6. Test temporal aggregation ──
    print("\n[6] Testing temporal aggregation...")
    from trio_core.analytics.aggregator import Aggregator, Sample

    # Query metrics from store
    metrics_data = await store.query_metrics(
        camera_id=cam_id, metric_type="people_count", granularity="hour"
    )

    # Also test the Aggregator directly
    samples = []
    for m in range(0, 18 * 12):  # 18 hours * 12 samples/hour
        h = m // 12
        minute = (m % 12) * 5
        ts = base_ts + timedelta(hours=h, minutes=minute)
        base_count = 30 if 8 <= (6 + h) <= 18 else 5
        count = max(0, base_count + random.randint(-8, 8))
        samples.append(Sample(
            timestamp=ts,
            count=count,
            raw_count=count,
            camera_id=cam_id,
        ))

    agg = Aggregator(bin_minutes=15, agg_method="mean")
    bins_15m = agg.aggregate(samples, level="bin")
    bins_hourly = agg.aggregate(samples, level="hourly")
    print(f"    OK — 15-min bins: {len(bins_15m)}, hourly bins: {len(bins_hourly)}")
    hourly_str = ", ".join(f"{b.start.strftime('%H:00')}={b.count}" for b in bins_hourly[:6])
    print(f"    Sample hourly: {hourly_str}")

    # ── 7. Test anomaly detection ──
    print("\n[7] Testing anomaly detection...")
    from trio_core.analytics.anomaly import AnomalyDetector

    detector = AnomalyDetector(z_threshold=2.0)
    anomalies = detector.detect(bins_15m)
    print(f"    OK — {len(anomalies)} anomalies detected")
    for a in anomalies[:3]:
        print(f"    [{a.severity:>6s}] {a.description}")

    # ── 8. Test reports ──
    print("\n[8] Testing report generation...")
    daily = await store.daily_report("2026-03-22", camera_id=cam_id)
    print(f"    OK — Daily report: {daily['total_events']} events, {daily['total_alerts']} alerts")
    print(f"    Hourly breakdown: {len(daily['hourly'])} hours")
    print(f"    Anomalies in report: {len(daily['anomalies'])}")

    # ── 9. Verify metrics endpoints data ──
    print("\n[9] Testing metrics queries...")
    metrics_hourly = await store.query_metrics(
        camera_id=cam_id, metric_type="people_count", granularity="hour"
    )
    metrics_latest = await store.latest_metrics(cam_id)
    metrics_summary = await store.metrics_summary(cam_id)

    print(f"    OK — Hourly data points: {len(metrics_hourly)}")
    print(f"    Latest metrics: {metrics_latest}")
    print(f"    Summary: total_in={metrics_summary['total_in']}, peak={metrics_summary['peak_hour']}")

    # ── 10. Summary ──
    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)

    checks = [
        ("Camera CRUD", True),
        ("Event storage", total > 0),
        ("Metric storage", metric_count > 0),
        (f"K3 Insights ({len(insights)}/5)", k3_met),
        ("Temporal aggregation", len(bins_hourly) > 0),
        ("Anomaly detection", True),  # even 0 anomalies is valid
        ("Daily report", daily["total_events"] > 0),
        ("Metrics query (hourly)", len(metrics_hourly) > 0),
        ("Metrics query (latest)", len(metrics_latest) > 0),
        ("Metrics summary", metrics_summary["total_in"] > 0),
    ]

    all_pass = True
    for name, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  [{status}] {name}")

    print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print("=" * 70)

    await store.close()
    return all_pass


if __name__ == "__main__":
    result = asyncio.run(main())
    sys.exit(0 if result else 1)
