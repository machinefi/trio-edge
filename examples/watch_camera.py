#!/usr/bin/env python3
"""Watch Camera — SSE client for trio-core /v1/watch API.

Connects to a running trio-core server, starts watching an RTSP camera,
and prints real-time alerts to the console. This is how trioclaw (or any
HTTP client) consumes the watch API.

Usage:
    # 1. Start trio-core server
    trio serve

    # 2. Run this example (in another terminal)
    python examples/watch_camera.py \\
        --source "rtsp://admin:pass@192.168.1.100:554/h264Preview_01_sub" \\
        --condition "Is there a person?"

    # Multiple conditions
    python examples/watch_camera.py \\
        --source "rtsp://admin:pass@192.168.1.100:554/stream" \\
        --condition "Is there a person?" \\
        --condition "Is there a package on the doorstep?"

    # Custom server URL
    python examples/watch_camera.py \\
        --server http://localhost:8000 \\
        --source "rtsp://camera:554/stream" \\
        --condition "Is the garage door open?"

Requires: pip install httpx  (or use requests)
"""

import argparse
import json
import sys
import time

import httpx


def watch(server: str, source: str, conditions: list[dict], fps: float = 1.0):
    """Start a watch and consume SSE events."""
    url = f"{server.rstrip('/')}/v1/watch"
    payload = {
        "source": source,
        "conditions": conditions,
        "fps": fps,
    }

    print(f"Connecting to {url}...")
    print(f"  Source: {source}")
    print(f"  Conditions: {[c['id'] for c in conditions]}")
    print(f"  FPS: {fps}")
    print()

    with httpx.stream("POST", url, json=payload, timeout=None) as resp:
        if resp.status_code != 200:
            print(f"Error: HTTP {resp.status_code}")
            print(resp.text)
            return

        watch_id = resp.headers.get("X-Watch-ID", "unknown")
        print(f"Watch started: {watch_id}")
        print("-" * 60)

        event_type = None
        for line in resp.iter_lines():
            if line.startswith("event: "):
                event_type = line[7:]
            elif line.startswith("data: "):
                raw = line[6:]
                if raw == "[DONE]":
                    print("\nStream ended.")
                    break
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event_type == "status":
                    state = data.get("state", "?")
                    model = data.get("model", "")
                    res = data.get("resolution", "")
                    extra = f" ({model}, {res})" if model else ""
                    print(f"  [{state}]{extra}")

                elif event_type == "result":
                    ts = data.get("ts", "")
                    latency = data["metrics"]["latency_ms"]
                    tok_s = data["metrics"]["tok_s"]
                    answers = ", ".join(
                        f"{c['id']}={'YES' if c['triggered'] else 'no'}"
                        for c in data["conditions"]
                    )
                    print(f"  {ts}  {answers}  ({latency}ms, {tok_s} tok/s)")

                elif event_type == "alert":
                    ts = data.get("ts", "")
                    latency = data["metrics"]["latency_ms"]
                    triggered = [c for c in data["conditions"] if c["triggered"]]
                    names = ", ".join(c["id"] for c in triggered)
                    answers = "; ".join(f"{c['id']}: {c['answer']}" for c in triggered)
                    has_frame = "frame_b64" in data
                    print(f"\n  ALERT [{ts}] {names}")
                    print(f"    {answers}")
                    print(f"    {latency}ms | frame={'yes' if has_frame else 'no'}")
                    print()

                elif event_type == "error":
                    print(f"\n  ERROR: {data.get('error', '?')}")
                    break


def list_watches(server: str):
    """List active watches."""
    resp = httpx.get(f"{server.rstrip('/')}/v1/watch")
    watches = resp.json()
    if not watches:
        print("No active watches.")
        return
    for w in watches:
        conds = ", ".join(c["id"] for c in w["conditions"])
        print(f"  {w['watch_id']}  {w['state']}  {w['source'][:40]}  "
              f"checks={w['checks']}  alerts={w['alerts']}  "
              f"uptime={w['uptime_s']}s  [{conds}]")


def stop_watch(server: str, watch_id: str):
    """Stop a watch."""
    resp = httpx.delete(f"{server.rstrip('/')}/v1/watch/{watch_id}")
    if resp.status_code == 200:
        data = resp.json()
        print(f"Stopped {watch_id}: {data['total_checks']} checks, {data['total_alerts']} alerts")
    else:
        print(f"Error: {resp.status_code} {resp.text}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="trio-core /v1/watch SSE client")
    parser.add_argument("--server", default="http://localhost:8000", help="trio-core server URL")

    sub = parser.add_subparsers(dest="command")

    # watch command (default)
    watch_p = sub.add_parser("watch", help="Start watching a camera")
    watch_p.add_argument("--source", "-s", required=True, help="RTSP URL")
    watch_p.add_argument("--condition", "-c", action="append", required=True,
                         help="Condition to monitor (can repeat)")
    watch_p.add_argument("--fps", type=float, default=1.0)

    # list command
    sub.add_parser("list", help="List active watches")

    # stop command
    stop_p = sub.add_parser("stop", help="Stop a watch")
    stop_p.add_argument("watch_id", help="Watch ID to stop")

    args = parser.parse_args()

    if args.command == "list":
        list_watches(args.server)
    elif args.command == "stop":
        stop_watch(args.server, args.watch_id)
    elif args.command == "watch" or args.command is None:
        if not hasattr(args, "source") or not args.source:
            parser.print_help()
            sys.exit(1)
        conditions = [
            {"id": c.lower().replace(" ", "_")[:20], "question": c}
            for c in args.condition
        ]
        try:
            watch(args.server, args.source, conditions, args.fps)
        except KeyboardInterrupt:
            print("\nStopped.")
        except httpx.ConnectError:
            print(f"\nCannot connect to {args.server}")
            print("Make sure trio-core server is running: trio serve")
