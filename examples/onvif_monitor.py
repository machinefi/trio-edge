#!/usr/bin/env python3
"""ONVIF Camera Monitor — discover IP cameras and watch with AI.

Auto-discovers ONVIF cameras on the local network via WS-Discovery,
gets the RTSP stream URI, and runs trio-core VLM analysis in a loop.
No opencv required — uses ffmpeg for RTSP capture.

Usage:
    # Auto-discover cameras and start monitoring
    python examples/onvif_monitor.py --user admin --password your_password

    # Discover only (list cameras, don't monitor)
    python examples/onvif_monitor.py --discover

    # Skip discovery, use known RTSP URL directly
    python examples/onvif_monitor.py --rtsp "rtsp://admin:pass@192.168.1.100:554/h264Preview_01_main"

    # Custom watch condition
    python examples/onvif_monitor.py --user admin --password pass --condition "Is there a person?"

    # Reolink camera with known IP (skip discovery)
    python examples/onvif_monitor.py --host 192.168.1.100 --user admin --password pass

Requires:
    pip install trio-edge[all]
    brew install ffmpeg
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time

import numpy as np

from trio_core.onvif import discover_cameras, get_rtsp_uri

try:
    from trio_core._rtsp_proxy import ensure_rtsp_url, stop_proxy
except ImportError:
    # Running standalone without trio-core installed — inline minimal proxy
    def ensure_rtsp_url(url: str) -> str:
        return url

    def stop_proxy():
        pass


# ── RTSP Frame Reader ────────────────────────────────────────────────────────


class RTSPReader:
    """Read frames from RTSP stream via ffmpeg subprocess.

    On macOS with Tailscale, automatically starts a TCP proxy using system Python
    to bypass network extension filtering.
    """

    def __init__(self, rtsp_url: str, fps: float = 1.0):
        self.url = rtsp_url
        self.fps = fps
        self.width = 0
        self.height = 0
        self._backend = None  # "ffmpeg" or "av"
        self._container = None
        self._proc = None
        self._frame_interval = 0
        self._frame_count = 0

    def start(self) -> RTSPReader:
        """Open the RTSP stream."""
        effective_url = ensure_rtsp_url(self.url)

        # Try PyAV first
        try:
            import av

            self._container = av.open(
                effective_url,
                options={"rtsp_transport": "tcp", "stimeout": "10000000"},
                timeout=10,
            )
            vstream = self._container.streams.video[0]
            self.width = vstream.width
            self.height = vstream.height
            native_fps = float(vstream.average_rate or 30)
            self._frame_interval = max(1, int(native_fps / self.fps))
            self._frame_count = 0
            self._backend = "av"
            print(f"  Stream: {self.width}x{self.height} @ {native_fps:.0f}fps (PyAV)")
            return self
        except ImportError:
            pass
        except Exception as e:
            print(f"  PyAV error: {e}")

        # Try ffmpeg
        ffprobe = shutil.which("ffprobe")
        ffmpeg = shutil.which("ffmpeg")
        if not ffprobe or not ffmpeg:
            raise RuntimeError(
                "Neither PyAV nor ffmpeg available. Install one:\n"
                "  pip install av    OR    brew install ffmpeg"
            )

        probe = subprocess.run(
            [
                ffprobe,
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
                effective_url,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode != 0 or not probe.stdout.strip():
            raise IOError(
                f"Cannot connect to RTSP stream.\n"
                f"Check: URL, credentials, camera RTSP enabled, network.\n"
                f"Error: {probe.stderr[:200]}"
            )

        parts = probe.stdout.strip().split(",")
        self.width, self.height = int(parts[0]), int(parts[1])
        print(f"  Stream: {self.width}x{self.height} (ffmpeg)")

        cmd = [
            ffmpeg,
            "-rtsp_transport",
            "tcp",
            "-i",
            effective_url,
            "-vf",
            f"fps={self.fps}",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-an",
            "-sn",
            "-v",
            "error",
            "-",
        ]
        self._proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=self.width * self.height * 3 * 2,
        )
        self._backend = "ffmpeg"
        return self

    def read(self) -> np.ndarray | None:
        """Read one RGB frame. Returns (H, W, 3) uint8 or None if stream ended."""
        if self._backend == "av":
            return self._read_av()
        return self._read_ffmpeg()

    def _read_av(self) -> np.ndarray | None:
        try:
            for packet in self._container.demux(video=0):
                for frame in packet.decode():
                    self._frame_count += 1
                    if self._frame_count % self._frame_interval != 0:
                        continue
                    return frame.to_ndarray(format="rgb24")
        except Exception:
            pass
        return None

    def _read_ffmpeg(self) -> np.ndarray | None:
        if self._proc is None:
            return None
        nbytes = self.width * self.height * 3
        raw = self._proc.stdout.read(nbytes)
        if len(raw) != nbytes:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3))

    def stop(self):
        if self._container:
            self._container.close()
            self._container = None
        if self._proc:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._proc.kill()
            self._proc = None
        stop_proxy()

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()


# ── Monitor Loop ─────────────────────────────────────────────────────────────


def monitor(rtsp_url: str, condition: str, interval: float, max_tokens: int):
    """Main monitoring loop — read frames, analyze with VLM, print results."""
    from trio_core import TrioCore

    engine = TrioCore()
    engine.load()

    prompt = f"Answer YES or NO: {condition}"

    with RTSPReader(rtsp_url, fps=1.0) as reader:
        print(f'\nMonitoring — "{condition}"')
        print("Press Ctrl+C to stop.\n")

        alert_count = 0
        check_count = 0

        while True:
            frame = reader.read()
            if frame is None:
                print("\nStream ended.")
                break

            frame_f = frame.astype(np.float32) / 255.0

            t0 = time.monotonic()
            result = engine.analyze_frame(frame_f, prompt, max_tokens=max_tokens)
            elapsed = time.monotonic() - t0

            answer = result.text.strip()
            triggered = answer.upper().startswith("YES")
            check_count += 1

            if triggered:
                alert_count += 1
                ts = time.strftime("%H:%M:%S")
                print(f"  [{ts}] ALERT #{alert_count}: {answer}  ({elapsed:.1f}s)")
            else:
                print(".", end="", flush=True)

            remaining = max(0, interval - elapsed)
            if remaining > 0:
                time.sleep(remaining)


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="ONVIF camera discovery + AI monitoring with trio-core",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  %(prog)s --discover                          # list cameras on LAN
  %(prog)s --user admin --password pass        # auto-discover + monitor
  %(prog)s --host 192.168.1.100 -u admin -p x  # known IP, skip discovery
  %(prog)s --rtsp "rtsp://admin:x@IP:554/..."  # direct RTSP URL
""",
    )
    p.add_argument("--discover", action="store_true", help="Discover cameras and exit")
    p.add_argument("--host", help="Camera IP (skip discovery)")
    p.add_argument("--port", type=int, default=8000, help="ONVIF port (default: 8000)")
    p.add_argument("--user", "-u", default="admin", help="Camera username")
    p.add_argument("--password", "-p", default="", help="Camera password")
    p.add_argument("--rtsp", help="Direct RTSP URL (skip discovery + ONVIF)")
    p.add_argument(
        "--condition",
        "-c",
        default="Is there a person?",
        help="What to watch for (yes/no question)",
    )
    p.add_argument(
        "--interval", type=float, default=3.0, help="Seconds between checks (default: 3.0)"
    )
    p.add_argument("--max-tokens", type=int, default=40)
    args = p.parse_args()

    # ── Mode 1: Direct RTSP URL ──
    if args.rtsp:
        print(f"Using RTSP URL: {args.rtsp}")
        try:
            monitor(args.rtsp, args.condition, args.interval, args.max_tokens)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    # ── Mode 2: Known host, get RTSP via ONVIF ──
    if args.host:
        rtsp_url = get_rtsp_uri(args.host, args.port, args.user, args.password)
        if not rtsp_url:
            print("Failed to get RTSP URI.")
            sys.exit(1)
        if args.discover:
            return
        try:
            monitor(rtsp_url, args.condition, args.interval, args.max_tokens)
        except KeyboardInterrupt:
            print("\nStopped.")
        return

    # ── Mode 3: Auto-discover ──
    cameras = discover_cameras()
    if not cameras:
        print("No ONVIF cameras found on the network.")
        print("Try: --host <IP> --user admin --password <pass>")
        sys.exit(1)

    print(f"\nFound {len(cameras)} camera(s):\n")
    for i, cam in enumerate(cameras):
        print(f"  [{i}] {cam.name}")
        print(f"      IP: {cam.ip}:{cam.port}")
        print(f"      ONVIF: {cam.onvif_url}")
        print()

    if args.discover:
        return

    cam = cameras[0]
    if not args.password:
        print("Password required: --password <pass>")
        sys.exit(1)

    rtsp_url = get_rtsp_uri(cam.ip, cam.port, args.user, args.password)
    if not rtsp_url:
        print("Failed to get RTSP URI.")
        sys.exit(1)

    try:
        monitor(rtsp_url, args.condition, args.interval, args.max_tokens)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
