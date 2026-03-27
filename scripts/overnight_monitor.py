#!/usr/bin/env python3
"""Overnight monitoring — loop local videos + RTSP, YOLO-only, store metrics.

Designed to run 8+ hours unattended on Mac Mini.
No VLM (saves ~5GB RAM). YOLO counting only.
Loops video files when they end.

Usage:
    python scripts/overnight_monitor.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trio_core.api.store import EventStore

from trio_core.counter import PeopleCounter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("overnight")

# Camera sources loaded dynamically from DB
CAMERAS = []  # Populated at startup from database

METRIC_INTERVAL = 300  # Store metrics every 5 minutes
FRAME_INTERVAL = 10.0  # Process frame every 10 seconds (video loops, no need for high freq)
LOOP_VIDEOS = True  # Restart video files when they end


async def monitor_camera(
    cam: dict, store: EventStore, counter: PeopleCounter, stop_event: asyncio.Event
):
    """Monitor a single camera source."""
    cam_id = cam["id"]
    cam_name = cam["name"]
    source = cam["source"]

    logger.info(f"[{cam_name}] Starting — {cam['type']}")

    while not stop_event.is_set():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.warning(f"[{cam_name}] Cannot open {source}. Retrying in 30s...")
            await asyncio.sleep(30)
            continue

        last_metric_time = 0
        frame_count = 0
        prev_gray = None
        MOTION_THRESHOLD = 0.003

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if cam["type"] == "file" and LOOP_VIDEOS:
                    logger.info(
                        f"[{cam_name}] Video ended. Looping. Total unique tracks: {len(counter._seen_ids)}"
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    prev_gray = None
                    # Reset tracker on loop — new "day" of observations
                    counter._seen_ids.clear()
                    counter._tracker = None
                    counter._initialized = False
                    continue
                else:
                    logger.warning(f"[{cam_name}] Stream ended.")
                    break

            frame_count += 1
            now = time.monotonic()

            # Motion detection — skip static frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (11, 11), 0)
            if prev_gray is not None:
                diff = cv2.absdiff(prev_gray, gray)
                motion = np.mean(diff) / 255.0
                if motion < MOTION_THRESHOLD:
                    prev_gray = gray
                    await asyncio.sleep(FRAME_INTERVAL)
                    continue
            prev_gray = gray

            # Keep tracker alive across frames for real person-counting
            # Only reset on video loop (handled above when cap resets to frame 0)
            result = counter.process(frame)

            # Store metrics only on motion + interval
            if now - last_metric_time > METRIC_INTERVAL:
                ts = datetime.now(timezone.utc).isoformat()
                metrics = [
                    {
                        "camera_id": cam_id,
                        "metric_type": "count_person",
                        "value": int(result.by_class.get("person", 0)),
                        "timestamp": ts,
                    },
                    {
                        "camera_id": cam_id,
                        "metric_type": "people_in",
                        "value": int(result.total_in),
                        "timestamp": ts,
                    },
                    {
                        "camera_id": cam_id,
                        "metric_type": "people_out",
                        "value": int(result.total_out),
                        "timestamp": ts,
                    },
                    {
                        "camera_id": cam_id,
                        "metric_type": "unique_tracks",
                        "value": int(len(counter._seen_ids)),
                        "timestamp": ts,
                    },
                ]
                for cls_name, count in result.by_class.items():
                    if cls_name != "person":
                        metrics.append(
                            {
                                "camera_id": cam_id,
                                "metric_type": f"count_{cls_name}",
                                "value": int(count),
                                "timestamp": ts,
                            }
                        )

                for m in metrics:
                    await store.insert_metric(m)

                last_metric_time = now

                # Log progress
                by_class = (
                    {k: int(v) for k, v in result.by_class.items()} if result.by_class else {}
                )
                logger.info(f"[{cam_name}] frame={frame_count} | {by_class}")

                # Save camera thumbnail periodically
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await store.save_camera_snapshot(cam_id, jpeg.tobytes())

            await asyncio.sleep(FRAME_INTERVAL)

        cap.release()

        if cam["type"] != "file":
            logger.warning(f"[{cam_name}] Disconnected. Reconnecting in 30s...")
            await asyncio.sleep(30)


async def main():
    store = EventStore()
    await store.init()

    # Load cameras from DB dynamically
    cams = await store.list_cameras()
    active_cams = []
    for c in cams:
        src = c.get("source_url", "")
        # Skip YouTube URLs (expire) — only local files + RTSP
        if "youtube.com" in src or "youtu.be" in src:
            logger.info(f"Skipping YouTube camera: {c['name']}")
            continue
        cam_type = "rtsp" if src.startswith("rtsp") else "file"
        active_cams.append(
            {
                "id": c["id"],
                "name": c["name"],
                "source": src,
                "type": cam_type,
            }
        )
    logger.info(f"Monitoring {len(active_cams)} cameras overnight")

    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")
    stop = asyncio.Event()

    # Run all cameras concurrently
    tasks = [asyncio.create_task(monitor_camera(c, store, counter, stop)) for c in active_cams]

    try:
        # Run for 8 hours
        await asyncio.sleep(8 * 3600)
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()

        # Final stats
        import sqlite3

        conn = sqlite3.connect("data/trio_console.db")
        events = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        metrics = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        conn.close()
        logger.info(f"OVERNIGHT COMPLETE — Events: {events}, Metrics: {metrics}")


if __name__ == "__main__":
    asyncio.run(main())
