#!/usr/bin/env python3
"""Scan the Starbucks 10-hour video at intervals to build a day's dataset.

Simulates a full day of monitoring by sampling frames every ~10 minutes
across the video. For each frame: YOLO detect + VLM crop-describe.
Stores high-quality events + metrics into the Event Store.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trio_core.api.store import EventStore
from trio_core.counter import PeopleCounter, COCO_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("starbucks_scan")

CAMERA_ID = "cam_5a1ff457"
CAMERA_NAME = "Starbucks Coffee Shop"
VIDEO_URL = "https://www.youtube.com/watch?v=LFFubUsxHug"

# Sample every 10 minutes across 10 hours = 60 samples
SAMPLE_INTERVAL_SEC = 600  # 10 minutes
TOTAL_DURATION_SEC = 36000  # 10 hours
STORE_OPEN_HOUR = 6  # Simulate store opens at 6 AM


async def main():
    import yt_dlp

    # Resolve stream URL
    logger.info("Resolving video URL...")
    ydl_opts = {"format": "best[height<=720]", "quiet": True}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(VIDEO_URL, download=False)
        stream_url = info["url"]
    logger.info("Stream resolved")

    # Init
    store = EventStore()
    await store.init()
    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")

    # Load VLM
    from trio_core import TrioCore, EngineConfig
    logger.info("Loading VLM...")
    engine = TrioCore(EngineConfig())
    engine.load()
    logger.info("VLM ready")

    # Scan the video at intervals
    samples = list(range(0, TOTAL_DURATION_SEC, SAMPLE_INTERVAL_SEC))
    logger.info(f"Scanning {len(samples)} timestamps across {TOTAL_DURATION_SEC/3600:.0f} hours")

    for i, offset_sec in enumerate(samples):
        # Simulate timestamp: today at store_open + offset
        sim_time = datetime.now(timezone.utc).replace(
            hour=STORE_OPEN_HOUR, minute=0, second=0, microsecond=0
        ) + timedelta(seconds=offset_sec)
        sim_ts = sim_time.isoformat()
        sim_hour = sim_time.strftime("%H:%M")

        # Capture frame at this offset
        proc = subprocess.run(
            ["ffmpeg", "-ss", str(offset_sec), "-i", stream_url,
             "-frames:v", "1", "-f", "image2pipe", "-vcodec", "mjpeg",
             "-q:v", "3", "-y", "pipe:1"],
            capture_output=True, timeout=30,
        )
        if not proc.stdout:
            logger.warning(f"[{sim_hour}] No frame at offset {offset_sec}s")
            continue

        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            continue

        # YOLO detection
        counter._seen_ids.clear()
        counter._tracker = None
        counter._initialized = False
        result = counter.process(frame)

        # Store metrics
        metrics = [
            {"camera_id": CAMERA_ID, "metric_type": "people_in", "value": int(result.detections), "timestamp": sim_ts},
            {"camera_id": CAMERA_ID, "metric_type": "count_person", "value": int(result.by_class.get("person", 0)), "timestamp": sim_ts},
        ]
        for cls_name, count in result.by_class.items():
            metrics.append({"camera_id": CAMERA_ID, "metric_type": f"count_{cls_name}", "value": int(count), "timestamp": sim_ts})
        for m in metrics:
            await store.insert_metric(m)

        # VLM describe each person crop (max 3 per frame)
        person_crops = [c for c in result.new_crops if c.class_name == "person" and c.crop is not None]

        for crop_info in person_crops[:3]:
            crop = crop_info.crop
            ch, cw = crop.shape[:2]
            if ch < 56 or cw < 56:
                scale = max(56 / ch, 56 / cw)
                crop = cv2.resize(crop, (max(56, int(cw * scale)), max(56, int(ch * scale))), interpolation=cv2.INTER_LANCZOS4)

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            try:
                vlm_result = engine.analyze_frame(crop_rgb,
                    "Describe this Starbucks customer precisely: "
                    "1) Age range and gender, "
                    "2) Drink size if visible (tall/grande/venti), "
                    "3) Food items if any (pastry, sandwich, none), "
                    "4) What they are doing (ordering, waiting, seated-working, seated-socializing, leaving), "
                    "5) Items carried (laptop, backpack, shopping bags, briefcase, phone). "
                    "Be factual and specific. One sentence.")
                desc = vlm_result.text.strip() if vlm_result and vlm_result.text else ""
            except Exception as e:
                logger.error(f"VLM error: {e}")
                desc = ""

            if not desc:
                continue

            event_id = await store.insert({
                "camera_id": CAMERA_ID,
                "camera_name": CAMERA_NAME,
                "description": desc,
                "timestamp": sim_ts,
                "metadata": {
                    "track_id": int(crop_info.track_id),
                    "confidence": round(float(crop_info.confidence), 3),
                    "by_class": {k: int(v) for k, v in result.by_class.items()},
                    "simulated_time": sim_hour,
                },
            })

            # Save full frame (not crop) for evidence
            _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            await store.save_frame(event_id, CAMERA_ID, jpeg.tobytes())

        # Also do a scene-level description every 30 minutes
        if offset_sec % 1800 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            try:
                scene = engine.analyze_frame(rgb,
                    "You are a retail analyst observing a Starbucks for an investment fund. "
                    "Rate and describe precisely: "
                    "1) OCCUPANCY: exact number of customers visible + busyness score 1-10 (1=empty, 5=moderate, 10=packed), "
                    "2) QUEUE: how many in line waiting to order (0 if none), "
                    "3) SEATED: how many seated vs standing, "
                    "4) ORDERS: visible cup sizes (tall/grande/venti) and any food items, "
                    "5) STAFF: how many employees visible and what they're doing, "
                    "6) REVENUE SIGNAL: estimate average ticket from what you see (low $3-5, medium $5-8, high $8+). "
                    "Be precise with numbers. This data drives investment decisions.")
                scene_desc = scene.text.strip() if scene and scene.text else ""
            except Exception:
                scene_desc = ""

            if scene_desc:
                event_id = await store.insert({
                    "camera_id": CAMERA_ID,
                    "camera_name": CAMERA_NAME,
                    "description": f"[SCENE {sim_hour}] {scene_desc}",
                    "timestamp": sim_ts,
                    "metadata": {"type": "scene_analysis", "simulated_time": sim_hour, "by_class": {k: int(v) for k, v in result.by_class.items()}},
                })
                _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                await store.save_frame(event_id, CAMERA_ID, jpeg.tobytes())

        people = result.by_class.get("person", 0)
        logger.info(f"[{i+1}/{len(samples)}] {sim_hour} — {people} people | {len(person_crops)} described | offset={offset_sec}s")

    # Final summary
    result = await store.list_events(limit=1)
    metrics_count = 0
    try:
        import sqlite3
        conn = sqlite3.connect("data/trio_console.db")
        metrics_count = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()[0]
        conn.close()
    except:
        pass

    logger.info(f"DONE — {result['total']} events, {metrics_count} metrics")
    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
