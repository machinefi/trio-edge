#!/usr/bin/env python3
"""Seed an RTSP camera into the Event Store and run a monitoring loop.

Usage:
    GEMINI_API_KEY=... python scripts/seed_camera.py \
        --rtsp "rtsp://admin:REDACTED@192.168.1.100:554/h264Preview_01_sub" \
        --name "Office Camera" \
        --watch "Describe what you see"

This script:
1. Registers the camera in SQLite
2. Runs a motion-gated VLM monitoring loop
3. Stores each event (description + frame JPEG) in the Event Store
4. Prints events as they occur
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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from trio_core.api.store import EventStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("seed")


def capture_rtsp(url: str) -> np.ndarray | None:
    """Capture a single frame from RTSP via ffmpeg."""
    cmd = [
        "ffmpeg", "-rtsp_transport", "tcp",
        "-i", url,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "mjpeg",
        "-q:v", "5",
        "-y", "pipe:1",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=15)
        if proc.returncode != 0 or not proc.stdout:
            return None
        arr = np.frombuffer(proc.stdout, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        logger.error("Capture failed: %s", e)
        return None


def compute_motion(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Compute motion score between two grayscale frames."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    return np.mean(thresh) / 255.0


async def run_monitor(
    store: EventStore,
    camera_id: str,
    camera_name: str,
    rtsp_url: str,
    question: str = "Describe what you see in this image. Be specific about people, objects, and activities.",
    interval: float = 5.0,
    motion_threshold: float = 0.005,
):
    """Monitor camera, detect motion, run VLM, store events."""
    # Lazy import engine
    from trio_core import TrioCore, EngineConfig

    logger.info("Loading VLM engine...")
    config = EngineConfig()
    engine = TrioCore(config)
    engine.load()
    logger.info("Engine ready: %s", config.model)

    prev_gray = None
    frame_count = 0

    logger.info("Monitoring %s (interval=%.0fs, motion_threshold=%.3f)", camera_name, interval, motion_threshold)
    logger.info("Press Ctrl+C to stop.\n")

    while True:
        frame = await asyncio.to_thread(capture_rtsp, rtsp_url)
        if frame is None:
            logger.warning("Frame capture failed, retrying in %ds...", int(interval))
            await asyncio.sleep(interval)
            continue

        frame_count += 1
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.GaussianBlur(curr_gray, (11, 11), 0)

        # Motion gate
        if prev_gray is not None:
            motion = compute_motion(prev_gray, curr_gray)
            if motion < motion_threshold:
                prev_gray = curr_gray
                await asyncio.sleep(interval)
                continue
            logger.info("Motion detected: %.4f", motion)

        prev_gray = curr_gray

        # VLM inference
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        t0 = time.monotonic()
        result = await asyncio.to_thread(engine.analyze_frame, rgb, question)
        elapsed = (time.monotonic() - t0) * 1000

        description = result.text.strip() if result and result.text else ""
        if not description:
            await asyncio.sleep(interval)
            continue

        # Store event
        now = datetime.now(timezone.utc).isoformat()
        event_id = await store.insert({
            "camera_id": camera_id,
            "camera_name": camera_name,
            "description": description,
            "timestamp": now,
            "metadata": {"motion_score": motion if prev_gray is not None else 0, "latency_ms": round(elapsed)},
        })

        # Save frame
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        await store.save_frame(event_id, camera_id, jpeg.tobytes())

        logger.info("[%s] %s (%.0fms)", camera_name, description[:100], elapsed)

        await asyncio.sleep(interval)


async def main():
    parser = argparse.ArgumentParser(description="Seed camera + monitor")
    parser.add_argument("--rtsp", required=True, help="RTSP URL")
    parser.add_argument("--name", default="Camera", help="Camera display name")
    parser.add_argument("--watch", default="Describe what you see. Be specific about people, objects, and activities.", help="VLM prompt")
    parser.add_argument("--interval", type=float, default=5.0, help="Seconds between checks")
    parser.add_argument("--motion", type=float, default=0.005, help="Motion threshold")
    parser.add_argument("--db", default="data/trio_console.db", help="SQLite path")
    args = parser.parse_args()

    store = EventStore(db_path=args.db)
    await store.init()

    # Register camera
    cameras = await store.list_cameras()
    camera_id = None
    for cam in cameras:
        if cam["source_url"] == args.rtsp:
            camera_id = cam["id"]
            logger.info("Camera already registered: %s (%s)", cam["name"], camera_id)
            break

    if not camera_id:
        camera_id = await store.create_camera({
            "name": args.name,
            "source_url": args.rtsp,
            "watch_condition": args.watch,
        })
        logger.info("Registered camera: %s (%s)", args.name, camera_id)

    # Test capture
    logger.info("Testing capture from %s...", args.rtsp)
    frame = await asyncio.to_thread(capture_rtsp, args.rtsp)
    if frame is None:
        logger.error("Cannot capture from RTSP. Check URL and network.")
        return
    logger.info("Capture OK: %dx%d", frame.shape[1], frame.shape[0])

    # Run monitor
    try:
        await run_monitor(store, camera_id, args.name, args.rtsp, args.watch, args.interval, args.motion)
    except KeyboardInterrupt:
        pass
    finally:
        await store.close()
        logger.info("Done. Events stored in %s", args.db)


if __name__ == "__main__":
    asyncio.run(main())
