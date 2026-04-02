#!/usr/bin/env python3
"""Discover ONVIF cameras on the local network."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from dataclasses import asdict, replace

from trio_core.onvif import discover_cameras, get_rtsp_uri


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Discover ONVIF cameras on the local network")
    parser.add_argument("--user", help="Camera username for resolving RTSP via GetStreamUri")
    parser.add_argument("--password", default="", help="Camera password for resolving RTSP")
    parser.add_argument("--timeout", type=int, default=5, help="Discovery timeout in seconds")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON output")
    return parser


def _enrich_with_rtsp(
    cameras, user: str | None, password: str, timeout: int
):
    if not user:
        return [replace(camera, rtsp_url=None) for camera in cameras]

    resolved = []
    for camera in cameras:
        rtsp_url = None
        try:
            # Resolve RTSP from the discovered ONVIF service port without guessing a fallback.
            rtsp_url = get_rtsp_uri(camera.ip, camera.port, user, password, fallback=False)
        except Exception:
            rtsp_url = None
        resolved.append(replace(camera, rtsp_url=rtsp_url))
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Some WS-Discovery backends write interface warnings directly to stdout/stderr.
    # Suppress that noise so human output stays readable and JSON output stays valid.
    noise = io.StringIO()
    try:
        with contextlib.redirect_stdout(noise), contextlib.redirect_stderr(noise):
            cameras = discover_cameras(timeout=args.timeout)
    except Exception as exc:
        print(f"ONVIF discovery failed: {exc}", file=sys.stderr)
        return 1

    cameras = _enrich_with_rtsp(cameras, args.user, args.password, args.timeout)

    if args.json:
        print(json.dumps([asdict(camera) for camera in cameras], indent=2))
        return 0

    if not cameras:
        print("No ONVIF cameras found on the network.")
        return 0

    print(f"Found {len(cameras)} camera(s)\n")
    for index, camera in enumerate(cameras, start=1):
        print(f"[{index}] {camera.name}")
        print(f"    IP: {camera.ip}:{camera.port}")
        if camera.onvif_url:
            print(f"    ONVIF: {camera.onvif_url}")
        if camera.rtsp_url:
            print(f"    RTSP: {camera.rtsp_url}")
        if index != len(cameras):
            print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
