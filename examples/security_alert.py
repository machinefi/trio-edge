#!/usr/bin/env python3
"""Security Alert — minimal camera monitor with desktop notifications.

A 30-line example showing how to use trio-core as a security camera AI.
Watches a video source, asks a yes/no question every 2 seconds,
and prints an alert when the condition is met.

Usage:
    # RTSP camera
    python examples/security_alert.py --source "rtsp://admin:pass@192.168.1.100:554/stream1"

    # Local webcam
    python examples/security_alert.py --source 0

    # Video file (for demo / testing)
    python examples/security_alert.py --source test_videos/intruder_house.mp4

    # Custom condition
    python examples/security_alert.py --source 0 --condition "Is there a package at the door?"

Requires: pip install trio-core
"""

import argparse
import time

import cv2

from trio_core import TrioCore

parser = argparse.ArgumentParser()
parser.add_argument("--source", "-s", default="0",
                    help="RTSP URL, video file, or camera index (default: 0)")
parser.add_argument("--condition", "-c", default="Is there a person?")
parser.add_argument("--interval", type=float, default=2.0, help="Seconds between checks")
args = parser.parse_args()

# Accept "0", "1" as camera index, otherwise treat as URL/path
source = int(args.source) if args.source.isdigit() else args.source

engine = TrioCore()
engine.load()
prompt = f"Answer YES or NO: {args.condition}"

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    print(f"ERROR: Cannot open source: {args.source}")
    raise SystemExit(1)
print(f"Watching {args.source} — {args.condition}")

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = engine.analyze_frame(frame, prompt, max_tokens=32)
        answer = result.text.strip()
        if answer.upper().startswith("YES"):
            print(f"\n🚨 ALERT: {answer} ({result.metrics.latency_ms:.0f}ms)")
        else:
            print(".", end="", flush=True)
        time.sleep(args.interval)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    print("\nStopped.")
