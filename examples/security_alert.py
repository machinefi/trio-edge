#!/usr/bin/env python3
"""Security Alert — minimal camera monitor with desktop notifications.

A compact example showing how to use trio-core as a security camera AI.
Watches a video source, asks a yes/no question periodically,
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
import shutil
import subprocess

import cv2
import numpy as np

from trio_core import TrioCore


def open_source(source):
    """Open video source — handles RTSP via system ffmpeg to avoid macOS sandbox issues."""
    if isinstance(source, int) or not source.startswith("rtsp://"):
        cap = cv2.VideoCapture(source)
        if cap.isOpened():
            return cap, None
        raise IOError(f"Cannot open: {source}")

    # RTSP: use system ffmpeg subprocess (has Local Network permission on macOS)
    if not shutil.which("ffmpeg"):
        raise IOError("ffmpeg not found — install with: brew install ffmpeg")

    cmd = [
        "ffmpeg",
        "-rtsp_transport",
        "tcp",
        "-i",
        source,
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-an",
        "-sn",
        "-v",
        "error",
        "-",
    ]
    # First, probe resolution
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0",
            source,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    w, h = [int(x) for x in probe.stdout.strip().split(",")]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=w * h * 3 * 2
    )
    return (w, h, proc), proc


def read_frame(cap):
    """Read a frame from cv2.VideoCapture or ffmpeg pipe."""
    if isinstance(cap, cv2.VideoCapture):
        ret, frame = cap.read()
        return frame if ret else None

    w, h, proc = cap
    raw = proc.stdout.read(w * h * 3)
    if len(raw) != w * h * 3:
        return None
    return np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))


def release(cap, proc):
    """Release resources."""
    if isinstance(cap, cv2.VideoCapture):
        cap.release()
    elif proc:
        proc.terminate()
        proc.wait(timeout=5)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--source", "-s", default="0", help="RTSP URL, video file, or camera index (default: 0)"
)
parser.add_argument("--condition", "-c", default="Is there a person?")
parser.add_argument("--interval", type=float, default=2.0, help="Seconds between checks")
args = parser.parse_args()

# Accept "0", "1" as camera index, otherwise treat as URL/path
source = int(args.source) if args.source.isdigit() else args.source

engine = TrioCore()
engine.load()
prompt = f"Answer YES or NO: {args.condition}"

cap, proc = open_source(source)
print(f"Watching {args.source} — {args.condition}")

try:
    while True:
        frame = read_frame(cap)
        if frame is None:
            break

        result = engine.analyze_frame(frame, prompt, max_tokens=32)
        answer = result.text.strip()
        triggered = answer.upper().startswith("YES")

        # Draw status on frame
        color = (0, 0, 255) if triggered else (0, 200, 0)
        label = f"ALERT: {answer[:60]}" if triggered else "clear"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(
            frame,
            f"{result.metrics.latency_ms:.0f}ms",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.imshow("trio-core security", frame)

        if triggered:
            print(f"\n🚨 ALERT: {answer} ({result.metrics.latency_ms:.0f}ms)")
        else:
            print(".", end="", flush=True)

        # Press 'q' to quit, or wait for interval
        if cv2.waitKey(int(args.interval * 1000)) & 0xFF == ord("q"):
            break
except KeyboardInterrupt:
    pass
finally:
    release(cap, proc)
    cv2.destroyAllWindows()
    print("\nStopped.")
