"""TCP proxy for RTSP streams — works around Tailscale network extension on macOS.

Tailscale's macOS network extension blocks unsigned binaries (Homebrew Python,
ffmpeg) from accessing LAN devices. System Python (/usr/bin/python3) is Apple-signed
and allowed through. This module starts a local TCP proxy that tunnels traffic:

    localhost:local_port  →  remote_host:remote_port

Usage:
    from trio_core._rtsp_proxy import ensure_rtsp_url
    effective_url = ensure_rtsp_url("rtsp://admin:pass@192.168.1.100:554/stream")
    # If proxy is needed, returns "rtsp://admin:pass@127.0.0.1:15554/stream"
    # Otherwise returns the original URL unchanged.
"""

from __future__ import annotations

import atexit
import shutil
import subprocess
from urllib.parse import urlparse

_proxy_proc: subprocess.Popen | None = None


def _needs_proxy(host: str, port: int) -> bool:
    """Check if a TCP proxy is needed (Tailscale blocking current Python)."""
    import socket

    # Try direct connection
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        s.connect((host, port))
        s.close()
        return False
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

    return False


def start_proxy(remote_host: str, remote_port: int, local_port: int = 15554) -> int:
    """Start a TCP proxy using /usr/bin/python3 (Apple-signed).

    Returns the local port the proxy is listening on.
    """
    global _proxy_proc
    if _proxy_proc is not None:
        return local_port

    # Pass host/port/local_port as command-line arguments to avoid code injection
    # via user-controlled RTSP URLs (the host comes from user input).
    proxy_code = '''
import socket, sys, threading
remote_host = sys.argv[1]
remote_port = int(sys.argv[2])
local_port = int(sys.argv[3])

def relay(src, dst):
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

srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
srv.bind(("127.0.0.1", local_port))
srv.listen(5)
sys.stdout.write("READY\\n"); sys.stdout.flush()
while True:
    c, _ = srv.accept()
    r = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    r.settimeout(10)
    try:
        r.connect((remote_host, remote_port))
    except Exception as e:
        sys.stderr.write(f"proxy: {e}\\n")
        c.close(); continue
    r.settimeout(None)
    threading.Thread(target=relay, args=(c, r), daemon=True).start()
    threading.Thread(target=relay, args=(r, c), daemon=True).start()
'''
    _proxy_proc = subprocess.Popen(
        ["/usr/bin/python3", "-c", proxy_code, remote_host, str(remote_port), str(local_port)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    line = _proxy_proc.stdout.readline()
    if b"READY" not in line:
        raise RuntimeError("TCP proxy failed to start")

    atexit.register(stop_proxy)
    return local_port


def stop_proxy():
    """Stop the TCP proxy if running."""
    global _proxy_proc
    if _proxy_proc:
        _proxy_proc.terminate()
        try:
            _proxy_proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            _proxy_proc.kill()
        _proxy_proc = None


def ensure_rtsp_url(url: str) -> str:
    """Return an RTSP URL that works from this Python process.

    If Tailscale is blocking, rewrites the URL to go through a local TCP proxy.
    Otherwise returns the URL unchanged.
    """
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or 554

    if not _needs_proxy(host, port):
        return url

    print("  Tailscale detected — starting TCP proxy via system Python...")
    local_port = start_proxy(host, port)
    # Rewrite URL: replace host:port with 127.0.0.1:local_port
    proxy_url = url.replace(f"{host}:{port}", f"127.0.0.1:{local_port}")
    if f"{host}:{port}" not in url:
        # Port wasn't explicit in URL, just replace host
        proxy_url = url.replace(host, f"127.0.0.1:{local_port}")
    print(f"  Proxy: 127.0.0.1:{local_port} → {host}:{port}")
    return proxy_url
