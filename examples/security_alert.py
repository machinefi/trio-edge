#!/usr/bin/env python3
"""Security Alert — minimal camera monitor with desktop notifications.

A 30-line example showing how to use trio-core as a security camera AI.
Watches your camera, asks a yes/no question every 2 seconds,
and prints an alert when the condition is met.

Usage:
    python examples/security_alert.py
    python examples/security_alert.py --condition "Is there a package at the door?"
    python examples/security_alert.py --condition "Is anyone wearing a mask?" --camera 1

Requires: pip install trio-core
"""

import argparse
import time

import cv2

from trio_core import TrioCore

parser = argparse.ArgumentParser()
parser.add_argument("--condition", "-c", default="Is there a person?")
parser.add_argument("--camera", type=int, default=0)
parser.add_argument("--interval", type=float, default=2.0, help="Seconds between checks")
args = parser.parse_args()

engine = TrioCore()
engine.load()
prompt = f"Answer YES or NO: {args.condition}"

cap = cv2.VideoCapture(args.camera)
print(f"Watching camera {args.camera} — {args.condition}")

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
