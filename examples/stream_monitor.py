#!/usr/bin/env python3
"""Stream Monitor — watch a live stream and fire webhooks on conditions.

Usage:
    python examples/stream_monitor.py "rtsp://camera/stream" --condition "Is there a person?"
    python examples/stream_monitor.py "https://youtube.com/watch?v=LIVE_ID" --condition "Is it raining?"
    python examples/stream_monitor.py 0 --condition "Is the door open?" --webhook https://webhook.site/xxx

Requires:
    pip install 'trio-core[mlx]' or pip install 'trio-core[cuda]'
    yt-dlp (for YouTube URLs)
"""

import argparse
import json
import time

from trio_core import EngineConfig, StreamCapture, TrioCore


def main():
    parser = argparse.ArgumentParser(description="Monitor a live stream with VLM")
    parser.add_argument("source", help="RTSP URL, YouTube URL, or camera index")
    parser.add_argument(
        "--condition",
        "-c",
        required=True,
        help="Yes/no condition to watch for (e.g., 'Is there a person?')",
    )
    parser.add_argument(
        "--webhook", "-w", default=None, help="Webhook URL to POST when condition triggers"
    )
    parser.add_argument(
        "--stride", type=int, default=60, help="Check every N-th frame (default: 60, ~2s at 30fps)"
    )
    parser.add_argument("--model", "-m", default=None, help="Model name")
    parser.add_argument("--backend", "-b", default=None, help="Force backend")
    parser.add_argument(
        "--motion", action="store_true", help="Enable motion gate (skip VLM on static scenes)"
    )
    parser.add_argument(
        "--max-checks", type=int, default=0, help="Stop after N checks (0=unlimited)"
    )
    args = parser.parse_args()

    # Configure
    config_kwargs = {"motion_enabled": args.motion, "max_tokens": 256}
    if args.model:
        config_kwargs["model"] = args.model
    config = EngineConfig(**config_kwargs)

    engine = TrioCore(config, backend=args.backend)
    print(f"Loading model: {config.model}")
    engine.load()

    # Build the monitoring prompt
    prompt = f"Answer YES or NO: {args.condition}\nThen explain briefly in one sentence."

    # Register callback for trigger detection
    triggers = []

    def on_vlm_end(e):
        if e.last_result and e.last_result.text:
            text = e.last_result.text.strip().upper()
            if text.startswith("YES"):
                triggers.append(e.last_result)
                print(f"  ** TRIGGERED: {e.last_result.text.strip()}")
                if args.webhook:
                    _send_webhook(args.webhook, args.condition, e.last_result.text)

    engine.add_callback("on_vlm_end", on_vlm_end)

    print(f"\nMonitoring: {args.source}")
    print(f"Condition: {args.condition}")
    print(f"Stride: every {args.stride} frames")
    if args.webhook:
        print(f"Webhook: {args.webhook}")
    print("-" * 60)

    check_count = 0
    try:
        with StreamCapture(args.source, vid_stride=args.stride) as cap:
            for frame in cap:
                check_count += 1
                t0 = time.monotonic()
                result = engine.analyze_frame(frame, prompt)
                elapsed = (time.monotonic() - t0) * 1000

                status = "TRIGGER" if result.text.strip().upper().startswith("YES") else "clear"
                print(f"[{check_count}] {status} ({elapsed:.0f}ms): {result.text.strip()[:100]}")

                if args.max_checks and check_count >= args.max_checks:
                    print(f"\nReached max checks ({args.max_checks}).")
                    break
    except KeyboardInterrupt:
        pass

    print(f"\nSession complete: {check_count} checks, {len(triggers)} triggers.")


def _send_webhook(url: str, condition: str, explanation: str):
    """POST trigger notification to webhook URL."""
    import urllib.request

    payload = json.dumps(
        {
            "type": "watch_trigger",
            "triggered": True,
            "condition": condition,
            "explanation": explanation,
            "timestamp": time.time(),
        }
    ).encode()

    try:
        req = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:
        print(f"  Webhook failed: {e}")


if __name__ == "__main__":
    main()
