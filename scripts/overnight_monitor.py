#!/usr/bin/env python3
"""Overnight monitoring — loop local videos + RTSP, YOLO-only, log metrics.

Designed to run 8+ hours unattended on Mac Mini.
No VLM (saves ~5GB RAM). YOLO counting only.
Loops video files when they end.

Usage:
    python scripts/overnight_monitor.py /path/to/video.mp4
    python scripts/overnight_monitor.py rtsp://camera:554/stream
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

from trio_core.counter import PeopleCounter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("overnight")

METRIC_INTERVAL = 300  # Log metrics every 5 minutes
FRAME_INTERVAL = 10.0  # Process frame every 10 seconds
LOOP_VIDEOS = True  # Restart video files when they end


async def monitor_source(source: str, counter: PeopleCounter, stop_event: asyncio.Event):
    """Monitor a single video source."""
    is_file = not source.startswith("rtsp")
    name = Path(source).stem if is_file else source

    logger.info(f"[{name}] Starting — {'file' if is_file else 'rtsp'}")

    while not stop_event.is_set():
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            logger.warning(f"[{name}] Cannot open {source}. Retrying in 30s...")
            await asyncio.sleep(30)
            continue

        last_metric_time = 0
        frame_count = 0
        prev_gray = None
        MOTION_THRESHOLD = 0.003

        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if is_file and LOOP_VIDEOS:
                    logger.info(
                        f"[{name}] Video ended. Looping. "
                        f"Total unique tracks: {len(counter._seen_ids)}"
                    )
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    prev_gray = None
                    counter._seen_ids.clear()
                    counter._tracker = None
                    counter._initialized = False
                    continue
                else:
                    logger.warning(f"[{name}] Stream ended.")
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

            result = counter.process(frame)

            # Log metrics on interval
            if now - last_metric_time > METRIC_INTERVAL:
                ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
                by_class = (
                    {k: int(v) for k, v in result.by_class.items()} if result.by_class else {}
                )
                logger.info(
                    f"[{name}] {ts} frame={frame_count} | "
                    f"in={result.total_in} out={result.total_out} "
                    f"tracks={len(counter._seen_ids)} | {by_class}"
                )
                last_metric_time = now

            await asyncio.sleep(FRAME_INTERVAL)

        cap.release()

        if not is_file:
            logger.warning(f"[{name}] Disconnected. Reconnecting in 30s...")
            await asyncio.sleep(30)


async def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/overnight_monitor.py <source> [source2] ...")
        print("  source: path to video file or rtsp:// URL")
        sys.exit(1)

    sources = sys.argv[1:]
    logger.info(f"Monitoring {len(sources)} source(s) overnight")

    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")
    stop = asyncio.Event()

    tasks = [asyncio.create_task(monitor_source(s, counter, stop)) for s in sources]

    try:
        await asyncio.sleep(8 * 3600)  # Run for 8 hours
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        stop.set()
        for t in tasks:
            t.cancel()
        logger.info("OVERNIGHT COMPLETE")


if __name__ == "__main__":
    asyncio.run(main())
