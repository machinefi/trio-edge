#!/usr/bin/env python3
"""TrioClaw Integration — shows how trioclaw (Go) talks to trio-core.

trioclaw is the device layer (cameras, sensors, IoT) that calls trio-core
over HTTP for VLM inference. This script simulates what trioclaw does:

    1. Check health: GET /healthz
    2. Capture frame from camera
    3. Send base64 JPEG: POST /analyze-frame
    4. Act on the response (triggered / not triggered)

Start trio-core server first:
    trio serve

Then run this:
    python examples/trioclaw_integration.py
    python examples/trioclaw_integration.py --url http://remote-mac:8000
    python examples/trioclaw_integration.py --question "Is there a package at the door?"
"""

import argparse
import base64
import io
import json
import time
import urllib.request

import cv2


def health_check(base_url: str) -> bool:
    """Check if trio-core is ready."""
    try:
        r = urllib.request.urlopen(f"{base_url}/healthz", timeout=5)
        return r.status == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def analyze_frame(base_url: str, frame, question: str) -> dict:
    """Send a frame to trio-core and get the answer.

    This is the exact contract trioclaw uses:
        POST /analyze-frame
        { "frame_b64": "<base64 JPEG>", "question": "..." }
        → { "answer": "...", "triggered": true/false, "latency_ms": 123 }
    """
    # Encode frame as JPEG → base64 (same as trioclaw Go code)
    _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(jpeg.tobytes()).decode()

    payload = json.dumps({
        "frame_b64": b64,
        "question": question,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/analyze-frame",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read())


def main():
    parser = argparse.ArgumentParser(description="Simulate trioclaw → trio-core integration")
    parser.add_argument("--url", default="http://localhost:8000", help="trio-core server URL")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--question", "-q", default="Is there a person?")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between checks")
    parser.add_argument("--max-checks", type=int, default=0, help="Stop after N checks (0=unlimited)")
    args = parser.parse_args()

    # Step 1: Health check
    print(f"Connecting to trio-core at {args.url}...")
    if not health_check(args.url):
        print("ERROR: trio-core is not ready. Start it with: trio serve")
        return

    print(f"Connected. Watching camera {args.camera}")
    print(f"Question: {args.question}")
    print("-" * 50)

    # Step 2: Capture loop (simulates trioclaw's frame capture)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        return

    checks = 0
    triggers = 0
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Step 3: Send to trio-core
            result = analyze_frame(args.url, frame, args.question)
            checks += 1

            # Step 4: Act on response
            triggered = result.get("triggered")
            answer = result.get("answer", "")
            latency = result.get("latency_ms", 0)

            if triggered:
                triggers += 1
                print(f"[{checks}] 🚨 TRIGGERED ({latency}ms): {answer}")
                # In real trioclaw, this would:
                # - Send push notification
                # - Save clip to storage
                # - Update device state via W3bstream
            else:
                print(f"[{checks}] ✓ clear ({latency}ms): {answer[:80]}")

            if args.max_checks and checks >= args.max_checks:
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()

    print(f"\nSession: {checks} checks, {triggers} triggers")


if __name__ == "__main__":
    main()
