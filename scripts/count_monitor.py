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

            # VLM enrichment: crop-then-describe each new target individually
            if engine and result.new_crops and (now - last_vlm_time > 5):
                last_vlm_time = now

                # Describe up to 3 new targets per cycle
                for crop_info in result.new_crops[:3]:
                    if crop_info.crop is None or crop_info.crop.size == 0:
                        continue

                    # Build class-specific prompt
                    if crop_info.class_name == "person":
                        prompt = (
                            "Describe this person in one sentence: approximate age, "
                            "gender, ethnicity, clothing, what they are carrying, "
                            "and what they appear to be doing."
                        )
                    elif crop_info.class_name in ("car", "truck", "bus"):
                        prompt = (
                            "Identify this vehicle in one sentence: type, color, "
                            "make/model/brand if possible, and any distinguishing features."
                        )
                    elif crop_info.class_name in ("dog", "cat"):
                        prompt = (
                            "Describe this animal in one sentence: breed if identifiable, "
                            "size, color, and whether it appears to be with an owner."
                        )
                    else:
                        prompt = "Describe this object in one sentence."

                    try:
                        crop = crop_info.crop
                        ch, cw = crop.shape[:2]
                        if ch < 56 or cw < 56:
                            scale = max(56 / ch, 56 / cw)
                            crop = cv2.resize(crop, (max(56, int(cw * scale)), max(56, int(ch * scale))))
                        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                        vlm_result = await asyncio.to_thread(engine.analyze_frame, crop_rgb, prompt)
                        description = vlm_result.text.strip() if vlm_result and vlm_result.text else ""

                        if not description:
                            continue

                        # Store as event
                        ts = datetime.now(timezone.utc).isoformat()
                        event_id = await store.insert({
                            "camera_id": args.camera_id,
                            "camera_name": args.camera_name,
                            "description": f"[{crop_info.class_name} #{crop_info.track_id}] {description}",
                            "timestamp": ts,
                            "metadata": {
                                "class": crop_info.class_name,
                                "track_id": crop_info.track_id,
                                "confidence": round(crop_info.confidence, 3),
                                "bbox": list(crop_info.bbox),
                                "by_class": result.by_class,
                                "total_in": result.total_in,
                                "total_out": result.total_out,
                            },
                        })

                        # Save crop as thumbnail + full frame
                        _, crop_jpeg = cv2.imencode(".jpg", crop_info.crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        await store.save_frame(event_id, args.camera_id, crop_jpeg.tobytes())

                        logger.info("VLM: %s", description[:120])
                    except Exception as e:
                        logger.error("VLM error for %s #%d: %s", crop_info.class_name, crop_info.track_id, e)

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
