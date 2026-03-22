#!/usr/bin/env python3
"""Seed 30 days of trend data for realistic demo dashboards.

Generates daily metrics with:
- Day-of-week patterns (weekday higher than weekend)
- Gradual upward trend (~3% week-over-week growth)
- Random daily noise
- Per-camera variation
"""

from __future__ import annotations

import asyncio
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

random.seed(42)


async def main():
    from trio_core.api.store import EventStore

    store = EventStore(db_path="data/trio_console.db")
    await store.init()

    # Get existing cameras
    cameras = await store.list_cameras()
    if not cameras:
        print("No cameras found. Run customer_personas.py first.")
        return

    print(f"Found {len(cameras)} cameras. Seeding 30 days of metrics...")

    base_date = datetime(2026, 3, 22, tzinfo=timezone.utc)

    # Day-of-week multipliers (Mon=0, Sun=6)
    dow_mult = {0: 1.0, 1: 1.05, 2: 1.0, 3: 1.1, 4: 1.15, 5: 0.7, 6: 0.55}

    # Camera base daily traffic
    cam_daily = {}
    for cam in cameras:
        name = cam["name"].lower()
        if "westfield" in name or "mall" in name:
            cam_daily[cam["id"]] = 800  # mall = highest
        elif "sbux" in name or "starbucks" in name:
            cam_daily[cam["id"]] = 450
        elif "blue bottle" in name or "coffee" in name:
            cam_daily[cam["id"]] = 300
        else:
            cam_daily[cam["id"]] = 150  # security = lower

    total_metrics = 0

    for day_offset in range(-30, 0):
        day = base_date + timedelta(days=day_offset)
        dow = day.weekday()
        # Gradual growth: 3% per week
        growth = 1.0 + (day_offset + 30) * 0.003 / 7
        day_mult = dow_mult[dow] * growth

        for cam in cameras:
            cam_id = cam["id"]
            daily_base = cam_daily.get(cam_id, 200)
            daily_total = int(daily_base * day_mult * (0.9 + random.random() * 0.2))

            # Hourly distribution (business hours heavy)
            hourly_weights = {
                6: 0.02, 7: 0.05, 8: 0.08, 9: 0.07, 10: 0.08,
                11: 0.10, 12: 0.12, 13: 0.11, 14: 0.09, 15: 0.07,
                16: 0.06, 17: 0.05, 18: 0.04, 19: 0.03, 20: 0.02, 21: 0.01,
            }

            for hour, weight in hourly_weights.items():
                hourly_count = max(1, int(daily_total * weight * (0.8 + random.random() * 0.4)))
                ts = day.replace(hour=hour, minute=30)

                await store.insert_metric({
                    "timestamp": ts.isoformat(),
                    "camera_id": cam_id,
                    "metric_type": "people_count",
                    "value": hourly_count,
                    "confidence": 0.85 + random.random() * 0.15,
                })
                await store.insert_metric({
                    "timestamp": ts.isoformat(),
                    "camera_id": cam_id,
                    "metric_type": "people_in",
                    "value": max(1, hourly_count // 2),
                })
                total_metrics += 2

        if (day_offset + 30) % 7 == 0:
            day_str = day.strftime("%Y-%m-%d")
            print(f"  Week ending {day_str}: {total_metrics} metrics so far")

    # Also seed some daily events for the trend report
    for day_offset in range(-30, 0):
        day = base_date + timedelta(days=day_offset)
        dow = day.weekday()
        growth = 1.0 + (day_offset + 30) * 0.003 / 7
        n_events = int(120 * dow_mult[dow] * growth * (0.8 + random.random() * 0.4))

        for i in range(n_events):
            hour = random.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
            ts = day.replace(hour=hour, minute=random.randint(0, 59))
            descs = [
                "Person walking through entrance area",
                "Two people at the counter",
                "Group entering the facility",
                "Individual standing near reception",
                "Employee with badge entering",
            ]
            await store.insert({
                "timestamp": ts.isoformat(),
                "camera_id": cameras[random.randint(0, len(cameras)-1)]["id"],
                "camera_name": cameras[random.randint(0, len(cameras)-1)]["name"],
                "description": random.choice(descs),
                "alert_triggered": random.random() < 0.05,
            })

    print(f"\nDone! {total_metrics} metric points + daily events seeded for 30 days.")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
