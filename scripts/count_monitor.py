#!/usr/bin/env python3
"""Continuous YOLO counting + VLM enrichment monitor.

Runs YOLO at ~2fps for counting, triggers VLM description for each new person/vehicle.
Feeds metrics + events into Trio Console API.

Usage:
    python scripts/count_monitor.py \
        --rtsp "rtsp://admin:pass@192.168.1.100:554/h264Preview_01_sub" \
        --camera-id cam_c8603989 \
        --camera-name "Office Camera"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trio_core.counter import PeopleCounter, COCO_CLASSES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("count_monitor")


def open_rtsp(url: str) -> cv2.VideoCapture:
    """Open RTSP stream with TCP transport."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open RTSP: {url}")
    return cap


async def run(args):
    from trio_core.api.store import EventStore

    store = EventStore(db_path=args.db)
    await store.init()

    # Init counter
    counter = PeopleCounter(
        model_path=args.model,
        confidence=args.confidence,
    )

    # Optionally load VLM for semantic enrichment
    engine = None
    if not args.no_vlm:
        try:
            from trio_core import TrioCore, EngineConfig
            logger.info("Loading VLM for semantic enrichment...")
            engine = TrioCore(EngineConfig())
            engine.load()
            logger.info("VLM ready: %s", engine.config.model)
        except Exception as e:
            logger.warning("VLM not available: %s. Counting-only mode.", e)

    logger.info("Opening RTSP stream: %s", args.rtsp)
    cap = open_rtsp(args.rtsp)

    frame_count = 0
    last_metric_time = 0
    last_vlm_time = 0
    prev_total_in = 0
    prev_total_out = 0
    prev_by_class: dict[str, int] = {}

    logger.info("Monitoring started. Press Ctrl+C to stop.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Frame read failed, reopening stream...")
                cap.release()
                await asyncio.sleep(2)
                cap = open_rtsp(args.rtsp)
                continue

            frame_count += 1
            now = time.monotonic()

            # YOLO counting (every frame from stream, ~2-5fps from RTSP)
            result = counter.process(frame)

            # Log when counts change
            if result.total_in != prev_total_in or result.total_out != prev_total_out or result.by_class != prev_by_class:
                parts = [f"IN:{result.total_in} OUT:{result.total_out}"]
                if result.by_class:
                    parts.append(" ".join(f"{k}:{v}" for k, v in sorted(result.by_class.items())))
                logger.info("Count: %s", " | ".join(parts))
                prev_total_in = result.total_in
                prev_total_out = result.total_out
                prev_by_class = dict(result.by_class)

            # Store metrics every 30 seconds
            if now - last_metric_time > 30:
                ts = datetime.now(timezone.utc).isoformat()
                metrics = [
                    {"camera_id": args.camera_id, "metric_type": "people_in", "value": result.total_in, "timestamp": ts},
                    {"camera_id": args.camera_id, "metric_type": "people_out", "value": result.total_out, "timestamp": ts},
                    {"camera_id": args.camera_id, "metric_type": "occupancy", "value": max(0, result.total_in - result.total_out), "timestamp": ts},
                ]
                for cls_name, count in result.by_class.items():
                    metrics.append({"camera_id": args.camera_id, "metric_type": f"count_{cls_name}", "value": count, "timestamp": ts})

                for m in metrics:
                    await store.insert_metric(m)
                last_metric_time = now

            # VLM enrichment for new targets (max once per 10 seconds)
            if engine and result.new_track_ids and (now - last_vlm_time > 10):
                last_vlm_time = now
                # Crop the newest detection for VLM
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

                # Build a specific prompt based on what's in frame
                class_list = ", ".join(f"{v} {k}(s)" for k, v in result.by_class.items())
                prompt = (
                    f"Describe what you see. Currently detected: {class_list}. "
                    "For each person, describe their apparent age, gender, clothing. "
                    "For vehicles, identify the make/model/color if possible. "
                    "Be concise — one line per subject."
                )

                try:
                    vlm_result = await asyncio.to_thread(engine.analyze_frame, rgb, prompt)
                    description = vlm_result.text.strip() if vlm_result and vlm_result.text else ""

                    if description:
                        # Store as event
                        ts = datetime.now(timezone.utc).isoformat()
                        event_id = await store.insert({
                            "camera_id": args.camera_id,
                            "camera_name": args.camera_name,
                            "description": description,
                            "timestamp": ts,
                            "metadata": {
                                "by_class": result.by_class,
                                "total_in": result.total_in,
                                "total_out": result.total_out,
                                "track_ids": result.new_track_ids[:5],
                            },
                        })

                        # Save frame
                        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                        await store.save_frame(event_id, args.camera_id, jpeg.tobytes())

                        logger.info("VLM: %s", description[:120])
                except Exception as e:
                    logger.error("VLM error: %s", e)

            # Throttle to ~2fps
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        await store.close()
        logger.info("Final counts — IN: %d, OUT: %d", result.total_in, result.total_out)


def main():
    parser = argparse.ArgumentParser(description="YOLO + VLM continuous monitor")
    parser.add_argument("--rtsp", required=True)
    parser.add_argument("--camera-id", default="cam_default")
    parser.add_argument("--camera-name", default="Camera")
    parser.add_argument("--model", default="models/yolov10n/onnx/model.onnx")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--db", default="data/trio_console.db")
    parser.add_argument("--no-vlm", action="store_true", help="Counting only, no VLM descriptions")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
