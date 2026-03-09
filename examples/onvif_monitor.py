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
    pip install trio-core[mlx] WSDiscovery onvif-zeep
    brew install ffmpeg
"""

from __future__ import annotations

import argparse
import atexit
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from urllib.parse import quote, urlparse

import numpy as np


# ── ONVIF Discovery ─────────────────────────────────────────────────────────


def discover_cameras(timeout: int = 5) -> list[dict]:
    """Find ONVIF cameras on the local network via WS-Discovery."""
    try:
        from wsdiscovery.discovery import ThreadedWSDiscovery
    except ImportError:
        print("WSDiscovery not installed: pip install WSDiscovery")
        return []

    print(f"Scanning network for ONVIF cameras ({timeout}s)...")
    wsd = ThreadedWSDiscovery()
    wsd.start()

    ONVIF_SCOPE = "onvif://www.onvif.org"
    services = wsd.searchServices(timeout=timeout)
    wsd.stop()

    cameras = []
    for svc in services:
        scopes = [str(s) for s in svc.getScopes()]
        if not any(ONVIF_SCOPE in s or "NetworkVideoTransmitter" in s for s in scopes):
            continue

        xaddrs = svc.getXAddrs()
        if not xaddrs:
            continue

        addr = str(xaddrs[0])
        parsed = urlparse(addr)

        name = "Unknown"
        for s in scopes:
            if "/name/" in s:
                name = s.split("/name/")[-1]
                break
            elif "/hardware/" in s:
                name = s.split("/hardware/")[-1]
                break

        cameras.append({
            "name": name,
            "ip": parsed.hostname,
            "port": parsed.port or 80,
            "onvif_url": addr,
            "scopes": scopes,
        })

    return cameras


def get_rtsp_uri(host: str, port: int, user: str, password: str) -> str | None:
    """Get RTSP stream URI from an ONVIF camera."""
    try:
        from onvif import ONVIFCamera
    except ImportError:
        print("onvif-zeep not installed: pip install onvif-zeep")
        return None

    print(f"Connecting to ONVIF device {host}:{port}...")
    try:
        cam = ONVIFCamera(host, port, user, password)
        media = cam.create_media_service()
        profiles = media.GetProfiles()

        if not profiles:
            print("  No media profiles found")
            return None

        profile = profiles[0]
        stream_setup = {
            "Stream": "RTP-Unicast",
            "Transport": {"Protocol": "RTSP"},
        }
        uri_obj = media.GetStreamUri({
            "StreamSetup": stream_setup,
            "ProfileToken": profile.token,
        })

        rtsp_uri = uri_obj.Uri
        print(f"  Profile: {profile.Name}")
        print(f"  RTSP URI: {rtsp_uri}")

        # Inject credentials into URI if not present
        parsed = urlparse(rtsp_uri)
        if not parsed.username and user:
            enc_pw = quote(password, safe="")
            rtsp_uri = rtsp_uri.replace(
                f"{parsed.scheme}://",
                f"{parsed.scheme}://{user}:{enc_pw}@",
            )

        return rtsp_uri

    except Exception as e:
        print(f"  ONVIF error: {e}")
        enc_pw = quote(password, safe="")
        fallback = f"rtsp://{user}:{enc_pw}@{host}:554/h264Preview_01_main"
        print(f"  Trying Reolink fallback: {fallback}")
        return fallback


# ── TCP Proxy (Tailscale workaround) ────────────────────────────────────────

_proxy_proc = None


def _start_tcp_proxy(remote_host: str, remote_port: int, local_port: int = 15554) -> int:
    """Start a TCP proxy using /usr/bin/python3 (system-signed, bypasses Tailscale).

    Tailscale's network extension on macOS blocks unsigned binaries (Homebrew Python,
    ffmpeg) from accessing LAN devices. System Python (/usr/bin/python3) is Apple-signed
    and allowed through. This proxy forwards localhost:local_port → remote_host:remote_port.

    Returns the local port the proxy is listening on.
    """
    global _proxy_proc

    proxy_code = f'''
import socket, threading, sys, os
def proxy(src, dst):
    try:
        while True:
            data = src.recv(65536)
            if not data: break
            dst.sendall(data)
    except: pass
    try: src.close()
    except: pass
    try: dst.close()
    except: pass

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server.bind(("127.0.0.1", {local_port}))
server.listen(5)
sys.stdout.write("READY\\n")
sys.stdout.flush()
while True:
    client, _ = server.accept()
    remote = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    remote.settimeout(10)
    try:
        remote.connect(("{remote_host}", {remote_port}))
    except Exception as e:
        sys.stderr.write(f"proxy connect error: {{e}}\\n")
        client.close()
        continue
    remote.settimeout(None)
    threading.Thread(target=proxy, args=(client, remote), daemon=True).start()
    threading.Thread(target=proxy, args=(remote, client), daemon=True).start()
'''
    _proxy_proc = subprocess.Popen(
        ["/usr/bin/python3", "-c", proxy_code],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    # Wait for "READY"
    line = _proxy_proc.stdout.readline()
    if b"READY" not in line:
        raise RuntimeError("TCP proxy failed to start")

    atexit.register(_stop_tcp_proxy)
    return local_port


def _stop_tcp_proxy():
    global _proxy_proc
    if _proxy_proc:
        _proxy_proc.terminate()
        try:
            _proxy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _proxy_proc.kill()
        _proxy_proc = None


def _needs_proxy(host: str, port: int) -> bool:
    """Check if we need a TCP proxy (Tailscale blocking Homebrew Python)."""
    import socket as _socket

    # Try direct connection from current Python
    try:
        s = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.close()
        return False  # Direct works, no proxy needed
    except OSError:
        pass

    # Check if nc (system binary) can reach it
    if shutil.which("nc"):
        try:
            r = subprocess.run(
                ["nc", "-z", "-w", "3", host, str(port)],
                capture_output=True, timeout=5,
            )
            if r.returncode == 0:
                return True  # Reachable via system binary, need proxy
        except Exception:
            pass

    return False  # Not reachable at all


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

    def _get_effective_url(self) -> str:
        """Return the RTSP URL, possibly rewritten to go through a local TCP proxy."""
        parsed = urlparse(self.url)
        host = parsed.hostname
        port = parsed.port or 554

        if not _needs_proxy(host, port):
            return self.url

        print("  Tailscale detected — starting TCP proxy via system Python...")
        local_port = _start_tcp_proxy(host, port)
        # Rewrite URL to use localhost proxy
        proxy_url = self.url.replace(f"{host}:{port}", f"127.0.0.1:{local_port}")
        proxy_url = proxy_url.replace(f"{host}/", f"127.0.0.1:{local_port}/")
        print(f"  Proxy: 127.0.0.1:{local_port} → {host}:{port}")
        return proxy_url

    def start(self) -> RTSPReader:
        """Open the RTSP stream."""
        effective_url = self._get_effective_url()

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
            raise RuntimeError("Neither PyAV nor ffmpeg available. Install one:\n"
                               "  pip install av    OR    brew install ffmpeg")

        probe = subprocess.run(
            [ffprobe, "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "csv=p=0", effective_url],
            capture_output=True, text=True, timeout=15,
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
            ffmpeg, "-rtsp_transport", "tcp",
            "-i", effective_url,
            "-vf", f"fps={self.fps}",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-an", "-sn", "-v", "error", "-",
        ]
        self._proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            (self.height, self.width, 3)
        )

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
        _stop_tcp_proxy()

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
        print(f"\nMonitoring — \"{condition}\"")
        print(f"Press Ctrl+C to stop.\n")

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
    p.add_argument("--condition", "-c", default="Is there a person?",
                    help="What to watch for (yes/no question)")
    p.add_argument("--interval", type=float, default=3.0,
                    help="Seconds between checks (default: 3.0)")
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
        print(f"  [{i}] {cam['name']}")
        print(f"      IP: {cam['ip']}:{cam['port']}")
        print(f"      ONVIF: {cam['onvif_url']}")
        print()

    if args.discover:
        return

    cam = cameras[0]
    if not args.password:
        print("Password required: --password <pass>")
        sys.exit(1)

    rtsp_url = get_rtsp_uri(cam["ip"], cam["port"], args.user, args.password)
    if not rtsp_url:
        print("Failed to get RTSP URI.")
        sys.exit(1)

    try:
        monitor(rtsp_url, args.condition, args.interval, args.max_tokens)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
