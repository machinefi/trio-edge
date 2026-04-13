"""HTTP MPEG-TS relay transport for Trio Cloud."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")

_FFMPEG_READ_SIZE = 65536


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class RelayError(Exception):
    """Base exception for relay errors."""


class SourceError(RelayError):
    """Raised when the input source cannot be opened or normalized."""


class CameraRegistrationError(RelayError):
    """Raised when camera bootstrap against Trio Cloud fails."""


class IngestUploadError(RelayError):
    """Raised when HTTP MPEG-TS upload fails."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# ANSI escape codes for terminal output.
_CR = "\r"
_CLEAR = "\033[K"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_RESET = "\033[0m"


async def _read_until_timeout(
    stdout: asyncio.StreamReader,
    duration_seconds: float,
    chunk_size: int = 65536,
) -> bytes:
    """Read from *stdout* until *duration_seconds* elapses or EOF."""
    buf = bytearray()
    deadline = time.monotonic() + duration_seconds

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            data = await asyncio.wait_for(stdout.read(chunk_size), timeout=remaining)
        except asyncio.TimeoutError:
            break
        if not data:
            break
        buf.extend(data)

    return bytes(buf)


def _normalize_source_fingerprint(source: str) -> str:
    source_type = detect_source_type(source)
    if source_type == "file":
        return str(Path(source).expanduser().resolve())
    return source.strip()


def derive_camera_id(source: str, explicit_camera_id: str | None = None) -> str:
    if explicit_camera_id:
        return explicit_camera_id

    mac_hex = f"{uuid.getnode():012x}"
    source_hash = hashlib.sha256(_normalize_source_fingerprint(source).encode("utf-8")).hexdigest()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{mac_hex}:{source_hash}"))


def _join_api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _camera_name_from_source(source: str) -> str:
    source_type = detect_source_type(source)
    if source_type == "rtsp":
        return "Edge RTSP Camera"
    if source_type == "webcam":
        return f"Edge Webcam {source}"
    return Path(source).name or "Edge File Camera"


def _safe_int(raw: str) -> int:
    """Parse an ffmpeg integer field that may be ``""`` or ``"N/A"``."""
    return int(raw) if raw not in ("", "N/A") else 0


def _safe_float(raw: str) -> float:
    """Parse an ffmpeg float field that may be ``""`` or ``"N/A"``."""
    return float(raw) if raw not in ("", "N/A") else 0.0


def _parse_bitrate_kbps(raw: str) -> float:
    """Parse ffmpeg ``'bitrate=6200.0kbits/s'`` → ``6200.0``."""
    if raw in ("", "N/A"):
        return 0.0
    try:
        return float(raw.split("k")[0])
    except (ValueError, IndexError):
        return 0.0


def _format_uptime(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# FFmpeg command builder
# ---------------------------------------------------------------------------


class _FFmpegCommandBuilder:
    def __init__(
        self,
        source: str,
        framerate: int,
        resolution: tuple[int, int] | None,
        *,
        verbose: bool = False,
    ) -> None:
        self._source = source
        self._fps = framerate
        self._resolution = resolution
        self._verbose = verbose
        self._source_type = detect_source_type(source)
        self._gop = framerate * 2

    def build(self) -> list[str]:
        return [*self._base_args(), *self._input_args(), *self._encoding_args()]

    def _base_args(self) -> list[str]:
        level = "info" if self._verbose else "error"
        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            level,
            "-nostdin",
            "-stats_period",
            "5",
            "-progress",
            "pipe:2",
        ]

    def _input_args(self) -> list[str]:
        if self._source_type == "rtsp":
            return self._input_rtsp()
        if self._source_type == "webcam":
            return self._input_webcam()
        if self._source_type == "file":
            return self._input_file()
        raise SourceError(f"Unsupported source type: {self._source_type} for '{self._source}'")

    def _input_rtsp(self) -> list[str]:
        from trio_core._rtsp_proxy import ensure_rtsp_url

        return ["-rtsp_transport", "tcp", "-i", ensure_rtsp_url(self._source)]

    def _input_webcam(self) -> list[str]:
        system = platform.system()
        if system == "Linux":
            return self._input_v4l2()
        if system == "Darwin":
            return self._input_avfoundation()
        raise SourceError(f"Unsupported webcam platform: {system}")

    def _input_v4l2(self) -> list[str]:
        args: list[str] = ["-f", "v4l2"]
        args.extend(self._resolution_args())
        args.extend(["-framerate", str(self._fps), "-i", f"/dev/video{self._source}"])
        return args

    def _input_avfoundation(self) -> list[str]:
        args: list[str] = [
            "-f",
            "avfoundation",
            "-pixel_format",
            "nv12",
            "-framerate",
            str(self._fps),
        ]
        args.extend(self._resolution_args())
        args.extend(["-i", f"{self._source}:none"])
        return args

    def _input_file(self) -> list[str]:
        return ["-re", "-stream_loop", "-1", "-i", self._source]

    def _resolution_args(self) -> list[str]:
        if not self._resolution:
            return []
        return ["-video_size", f"{self._resolution[0]}x{self._resolution[1]}"]

    def _encoding_args(self) -> list[str]:
        args: list[str] = [
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-g",
            str(self._gop),
        ]
        # RTSP and file sources pass through at the target fps.
        if self._source_type in ("rtsp", "file"):
            args.extend(["-r", str(self._fps)])
        args.extend(["-an", "-f", "mpegts", "pipe:1"])
        return args


# ---------------------------------------------------------------------------
# Progress monitor
# ---------------------------------------------------------------------------


class _ProgressMonitor:
    _MAX_RECENT_ERRORS = 20

    def __init__(
        self,
        *,
        verbose: bool = False,
        json_mode: bool = False,
        target_fps: int = 30,
    ) -> None:
        self._verbose = verbose
        self._json_mode = json_mode
        self._target_fps = target_fps
        self.recent_errors: list[str] = []
        self._start_time = time.monotonic()
        self._prev_total_size = 0
        self._prev_time = time.monotonic()
        self._prev_drop_frames = 0

    async def run(self, stderr: asyncio.StreamReader) -> None:
        block: dict[str, str] = {}
        try:
            while True:
                line_bytes = await stderr.readline()
                if not line_bytes:
                    break
                line = line_bytes.decode(errors="replace").strip()
                if not line:
                    continue

                if self._verbose and not self._json_mode:
                    sys.stderr.write(line + "\n")
                    sys.stderr.flush()
                    continue

                if "=" in line:
                    key, _, value = line.partition("=")
                    key = key.strip()
                    value = value.strip()
                    block[key] = value

                    if key == "progress" and value in ("continue", "end"):
                        self._on_progress(block)
                        block = {}
                else:
                    self._record_error(line)
        finally:
            # Final newline so the shell prompt doesn't overwrite the last progress line.
            if not self._json_mode and not self._verbose:
                sys.stderr.write("\n")
                sys.stderr.flush()

    def _on_progress(self, block: dict[str, str]) -> None:
        if self._json_mode:
            self._emit_json(block)
        elif self._verbose:
            return
        else:
            self._emit_human(block)

    def _emit_human(self, block: dict[str, str]) -> None:
        total_size = _safe_int(block.get("total_size", "0"))
        fps = _safe_float(block.get("fps", "0"))
        drop = _safe_int(block.get("drop_frames", "0"))
        speed = block.get("speed", "N/A")

        now = time.monotonic()
        uptime = now - self._start_time
        elapsed = now - self._prev_time
        drop_delta = drop - self._prev_drop_frames

        rate_bps = (total_size - self._prev_total_size) / elapsed if elapsed > 0 else 0
        rate_mbps = rate_bps * 8 / 1_000_000
        total_mb = total_size / (1024 * 1024)

        state, color = self._health_state(fps, drop_delta)
        dot = f"\033[{color}m●{_RESET}"

        line = (
            f"{_CR}{dot} {state} | {_format_uptime(uptime)} | "
            f"{rate_mbps:.1f} Mbps | {fps:.1f} fps | drop: {drop_delta} | "
            f"speed: {speed} | {total_mb:.1f} MB sent{_CLEAR}"
        )
        sys.stderr.write(line)
        sys.stderr.flush()

        self._prev_total_size = total_size
        self._prev_time = now
        self._prev_drop_frames = drop

    def _emit_json(self, block: dict[str, str]) -> None:
        total_size = _safe_int(block.get("total_size", "0"))
        fps = _safe_float(block.get("fps", "0"))
        drop = _safe_int(block.get("drop_frames", "0"))
        dup = _safe_int(block.get("dup_frames", "0"))
        bitrate = _parse_bitrate_kbps(block.get("bitrate", "N/A"))
        speed = block.get("speed", "N/A")
        frame = _safe_int(block.get("frame", "0"))

        now = time.monotonic()
        drop_delta = drop - self._prev_drop_frames
        uptime = now - self._start_time

        obj = {
            "type": "progress",
            "ts": _iso_now(),
            "state": self._health_state(fps, drop_delta)[0].lower(),
            "uptime_s": round(uptime, 1),
            "frame": frame,
            "fps": round(fps, 1),
            "bitrate_kbps": round(bitrate, 1),
            "drop_frames": drop,
            "drop_delta": drop_delta,
            "dup_frames": dup,
            "total_bytes": total_size,
            "speed": speed,
        }
        sys.stderr.write(json.dumps(obj, separators=(",", ":")) + "\n")
        sys.stderr.flush()

        self._prev_total_size = total_size
        self._prev_time = now
        self._prev_drop_frames = drop

    def _health_state(self, fps: float, drop_delta: int) -> tuple[str, str]:
        # drop_delta is intentionally NOT used here. FFmpeg's drop_frames
        # counter tracks rate-conversion skips (e.g. 30fps→1fps drops 29
        # frames/sec), not actual network or decode failures. The output
        # fps compared against the target is the reliable health signal.
        if self._target_fps > 0 and fps < self._target_fps * 0.5:
            return "DEGRADED", "31"
        if self._target_fps > 0 and fps < self._target_fps * 0.9:
            return "LIVE", "33"
        return "LIVE", "32"

    def _record_error(self, line: str) -> None:
        self.recent_errors.append(line)
        if len(self.recent_errors) > self._MAX_RECENT_ERRORS:
            self.recent_errors = self.recent_errors[-self._MAX_RECENT_ERRORS :]

        if self._json_mode:
            obj = {"type": "error", "ts": _iso_now(), "message": line}
            sys.stderr.write(json.dumps(obj, separators=(",", ":")) + "\n")
            sys.stderr.flush()
        elif self._verbose:
            logger.warning("ffmpeg: %s", line)


# ---------------------------------------------------------------------------
# Segment uploader
# ---------------------------------------------------------------------------


class _SegmentUploader:
    def __init__(
        self,
        client: httpx.AsyncClient,
        ingest_url: str,
        headers: dict[str, str],
        segment_duration: float,
        *,
        json_mode: bool = False,
    ) -> None:
        self._client = client
        self._ingest_url = ingest_url
        self._headers = headers
        self._segment_duration = segment_duration
        self._json_mode = json_mode
        self.segments_ok = 0
        self.segments_fail = 0
        self._seg_num = 0

    async def upload_all(self, stdout: asyncio.StreamReader) -> None:
        while True:
            chunk = await _read_until_timeout(stdout, self._segment_duration)
            if not chunk:
                break
            self._seg_num += 1
            if not await self._upload_one(self._seg_num, chunk):
                break

    async def _upload_one(self, seg_num: int, chunk: bytes) -> bool:
        start = time.monotonic()
        try:
            resp = await self._client.post(self._ingest_url, content=chunk, headers=self._headers)
        except httpx.TransportError as exc:
            self.segments_fail += 1
            self._log_error(f"Transport error for seg #{seg_num}: {exc}")
            return False

        elapsed = time.monotonic() - start
        if resp.status_code == 204:
            self.segments_ok += 1
            self._log_success(seg_num, chunk, elapsed)
            return True

        self.segments_fail += 1
        self._log_error(f"HTTP {resp.status_code} for seg #{seg_num}")
        return False

    def _log_success(self, seg_num: int, chunk: bytes, elapsed: float) -> None:
        if self._json_mode:
            obj = {
                "type": "segment",
                "ts": _iso_now(),
                "seg_num": seg_num,
                "bytes": len(chunk),
                "elapsed_s": round(elapsed, 2),
                "ok": self.segments_ok,
                "fail": self.segments_fail,
            }
            sys.stderr.write(json.dumps(obj, separators=(",", ":")) + "\n")
            sys.stderr.flush()

    def _log_error(self, message: str) -> None:
        if self._json_mode:
            obj = {"type": "error", "ts": _iso_now(), "message": message}
            sys.stderr.write(json.dumps(obj, separators=(",", ":")) + "\n")
        else:
            sys.stderr.write(f"{_CR}{_CLEAR}{message}\n")
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Public relay class
# ---------------------------------------------------------------------------


@dataclass
class HttpIngestRelay:
    """Relays video from a local source to Trio Cloud over HTTP MPEG-TS."""

    source: str
    cloud_url: str
    bearer_token: str
    camera_id: str | None = None
    resolution: tuple[int, int] | None = None
    framerate: int = 1
    verbose: bool = False
    segment_duration: float = 10.0
    json_mode: bool = False

    def __post_init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    def resolved_camera_id(self) -> str:
        return derive_camera_id(self.source, self.camera_id)

    async def run(self) -> None:
        """Register camera, start ffmpeg, and relay segments until EOF or error."""
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as reg_client:
            camera_id = await self._register_camera(reg_client)

        ingest_url = _join_api_url(self.cloud_url, f"/api/stream/ingest/{camera_id}")
        headers = self._auth_headers(content_type="video/mp2t")

        await self._start_ffmpeg()
        assert self._process is not None
        assert self._process.stdout is not None

        monitor = _ProgressMonitor(
            verbose=self.verbose,
            json_mode=self.json_mode,
            target_fps=self.framerate,
        )
        monitor_task = asyncio.create_task(monitor.run(self._process.stderr))

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, write=30.0, read=30.0, pool=10.0),
                follow_redirects=True,
                limits=httpx.Limits(max_keepalive_connections=0),
            ) as client:
                uploader = _SegmentUploader(
                    client,
                    ingest_url,
                    headers,
                    self.segment_duration,
                    json_mode=self.json_mode,
                )
                await uploader.upload_all(self._process.stdout)

            return_code = await self._process.wait()
            if return_code != 0:
                recent = monitor.recent_errors[-5:]
                detail = "\n  ".join(recent) if recent else f"exit code {return_code}"
                raise SourceError(f"ffmpeg exited (code {return_code}):\n  {detail}")
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

    async def teardown(self) -> None:
        """Terminate the ffmpeg subprocess if still running."""
        if self._process is None:
            return
        if self._process.returncode is None:
            self._process.terminate()
            try:
                await asyncio.wait_for(self._process.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                self._process.kill()
                await self._process.wait()
        self._process = None

    def _auth_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"X-API-Key": self.bearer_token}
        if content_type is not None:
            headers["Content-Type"] = content_type
        return headers

    def _camera_payload(self, camera_id: str) -> dict[str, object]:
        return {
            "id": camera_id,
            "name": _camera_name_from_source(self.source),
            "source_url": self.source,
            "metadata": {
                "managed_by": "trio-edge",
                "ingest_transport": "http_mpegts",
                "source_type": detect_source_type(self.source),
            },
        }

    async def _register_camera(self, client: httpx.AsyncClient) -> str:
        camera_id = self.resolved_camera_id()

        response = await client.post(
            _join_api_url(self.cloud_url, "/api/cameras"),
            headers=self._auth_headers(),
            json=self._camera_payload(camera_id),
        )
        if response.status_code not in (200, 201):
            raise CameraRegistrationError(
                f"Camera registration failed (HTTP {response.status_code}): {response.text[:200]}"
            )

        returned_id = response.json().get("id")
        if not returned_id:
            raise CameraRegistrationError(
                "Camera registration succeeded but response did not include a camera id"
            )
        if returned_id != camera_id:
            logger.info(
                "Cloud assigned camera id %s instead of requested %s",
                returned_id,
                camera_id,
            )
        return returned_id

    def _build_ffmpeg_cmd(self) -> list[str]:
        return _FFmpegCommandBuilder(
            source=self.source,
            framerate=self.framerate,
            resolution=self.resolution,
            verbose=self.verbose,
        ).build()

    async def _start_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise SourceError(
                "ffmpeg not found. Install with: apt install ffmpeg (Linux) "
                "or brew install ffmpeg (macOS)"
            )
        self._process = await asyncio.create_subprocess_exec(
            *self._build_ffmpeg_cmd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
