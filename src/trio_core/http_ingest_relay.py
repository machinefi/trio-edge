"""HTTP MPEG-TS relay transport for Trio Cloud."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import platform
import shutil
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")
_FFMPEG_READ_SIZE = 65536


async def _read_until_timeout(
    stdout: asyncio.StreamReader, duration_seconds: float, chunk_size: int = 65536
) -> bytes:
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


class RelayError(Exception):
    """Base exception for relay errors."""


class SourceError(RelayError):
    """Raised when the input source cannot be opened or normalized."""


class CameraRegistrationError(RelayError):
    """Raised when camera bootstrap against Trio Cloud fails."""


class IngestUploadError(RelayError):
    """Raised when HTTP MPEG-TS upload fails."""


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


@dataclass
class HttpIngestRelay:
    source: str
    cloud_url: str
    bearer_token: str
    camera_id: str | None = None
    resolution: tuple[int, int] | None = None
    framerate: int = 30
    verbose: bool = False
    segment_duration: float = 10.0
    _recent_errors: list[str] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    def resolved_camera_id(self) -> str:
        return derive_camera_id(self.source, self.camera_id)

    def _auth_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = {"X-API-Key": self.bearer_token}
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

    def _base_ffmpeg_args(self) -> list[str]:
        level = "info" if self.verbose else "error"
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

    def _build_ffmpeg_cmd(self) -> list[str]:
        source_type = detect_source_type(self.source)
        fps = str(self.framerate)
        base = self._base_ffmpeg_args()

        if source_type == "rtsp":
            from trio_core._rtsp_proxy import ensure_rtsp_url

            return [
                *base,
                "-rtsp_transport",
                "tcp",
                "-i",
                ensure_rtsp_url(self.source),
                "-c:v",
                "copy",
                "-an",
                "-f",
                "mpegts",
                "pipe:1",
            ]

        if source_type == "webcam":
            resolution_args: list[str] = []
            if self.resolution:
                resolution_args = ["-video_size", f"{self.resolution[0]}x{self.resolution[1]}"]

            system = platform.system()
            if system == "Linux":
                return [
                    *base,
                    "-f",
                    "v4l2",
                    *resolution_args,
                    "-framerate",
                    fps,
                    "-i",
                    f"/dev/video{self.source}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-profile:v",
                    "baseline",
                    "-g",
                    str(self.framerate * 2),
                    "-an",
                    "-f",
                    "mpegts",
                    "pipe:1",
                ]
            if system == "Darwin":
                return [
                    *base,
                    "-f",
                    "avfoundation",
                    "-pixel_format",
                    "nv12",
                    "-framerate",
                    fps,
                    *resolution_args,
                    "-i",
                    f"{self.source}:none",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-profile:v",
                    "baseline",
                    "-g",
                    str(self.framerate * 2),
                    "-an",
                    "-f",
                    "mpegts",
                    "pipe:1",
                ]
            raise SourceError(f"Unsupported webcam platform: {system}")

        if source_type != "file":
            raise SourceError(f"Unsupported source type: {source_type} for '{self.source}'")

        return [
            *base,
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            self.source,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-g",
            str(self.framerate * 2),
            "-r",
            fps,
            "-an",
            "-f",
            "mpegts",
            "pipe:1",
        ]

    async def _register_camera(self, client: httpx.AsyncClient) -> str:
        requested_camera_id = self.resolved_camera_id()

        # Check if camera already exists (skip re-registration)
        try:
            check_url = _join_api_url(self.cloud_url, f"/api/cameras/{requested_camera_id}")
            check_resp = await client.get(check_url, headers=self._auth_headers())
            if check_resp.status_code == 200:
                existing_id = check_resp.json().get("id")
                if existing_id:
                    logger.info("Camera %s already registered, skipping registration", existing_id)
                    return existing_id
        except Exception:
            # GET check failed — fall through to POST (non-blocking)
            logger.debug("Camera existence check failed, proceeding with registration")

        # Existing POST logic
        response = await client.post(
            _join_api_url(self.cloud_url, "/api/cameras"),
            headers=self._auth_headers(),
            json=self._camera_payload(requested_camera_id),
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
        if returned_id != requested_camera_id:
            logger.info(
                "Cloud assigned camera id %s instead of requested %s",
                returned_id,
                requested_camera_id,
            )
        return returned_id

    async def _start_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise SourceError(
                "ffmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
            )
        self._process = await asyncio.create_subprocess_exec(
            *self._build_ffmpeg_cmd(),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _monitor_progress(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None

        prev_total_size = 0
        prev_time = time.monotonic()
        block: dict[str, str] = {}

        while True:
            line_bytes = await self._process.stderr.readline()
            if not line_bytes:
                break
            line = line_bytes.decode(errors="replace").strip()
            if not line:
                continue

            if self.verbose:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
                continue

            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip()
                block[key] = value

                if key == "progress" and value in ("continue", "end"):
                    self._emit_progress(block, prev_total_size, prev_time)
                    total_size_str = block.get("total_size", "0")
                    prev_total_size = (
                        int(total_size_str) if total_size_str not in ("", "N/A") else 0
                    )
                    prev_time = time.monotonic()
                    block = {}
            else:
                self._recent_errors.append(line)
                if len(self._recent_errors) > 20:
                    self._recent_errors = self._recent_errors[-20:]
                logger.warning("ffmpeg: %s", line)

    def _emit_progress(self, block: dict[str, str], prev_total_size: int, prev_time: float) -> None:
        total_size_str = block.get("total_size", "0")
        total_size = int(total_size_str) if total_size_str not in ("", "N/A") else 0
        speed = block.get("speed", "N/A")

        now = time.monotonic()
        elapsed = now - prev_time
        size_delta = total_size - prev_total_size
        rate_bps = size_delta / elapsed if elapsed > 0 else 0
        rate_mbps = rate_bps * 8 / 1_000_000
        total_mb = total_size / (1024 * 1024)

        ts = time.strftime("%H:%M:%S")
        sys.stdout.write(
            f"[{ts}] \u2191 {rate_mbps:.1f} Mbps | {total_mb:.1f} MB sent | {speed} speed\n"
        )
        sys.stdout.flush()

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as reg_client:
            camera_id = await self._register_camera(reg_client)

        ingest_url = _join_api_url(self.cloud_url, f"/api/stream/ingest/{camera_id}")
        headers = self._auth_headers(content_type="video/mp2t")

        await self._start_ffmpeg()
        assert self._process is not None
        assert self._process.stdout is not None

        monitor_task = asyncio.create_task(self._monitor_progress())

        seg_num = 0
        ok_count = 0
        fail_count = 0
        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(connect=10.0, write=30.0, read=30.0, pool=10.0),
                follow_redirects=True,
            ) as client:
                while True:
                    chunk = await _read_until_timeout(self._process.stdout, self.segment_duration)
                    if not chunk:
                        break

                    seg_num += 1
                    seg_start = time.monotonic()
                    try:
                        resp = await client.post(ingest_url, content=chunk, headers=headers)
                        elapsed = time.monotonic() - seg_start
                        if resp.status_code == 204:
                            ok_count += 1
                            size_kb = len(chunk) / 1024
                            ts = time.strftime("%H:%M:%S")
                            sys.stdout.write(
                                f"[{ts}] seg #{seg_num} OK {elapsed:.1f}s "
                                f"{size_kb:.0f}KB [ok={ok_count} fail={fail_count}]\n"
                            )
                            sys.stdout.flush()
                        else:
                            fail_count += 1
                            logger.error(
                                "Ingest POST failed: HTTP %s for seg #%d",
                                resp.status_code,
                                seg_num,
                            )
                            break
                    except httpx.TransportError as exc:
                        fail_count += 1
                        logger.error("Ingest POST transport error for seg #%d: %s", seg_num, exc)
                        break

            return_code = await self._process.wait()
            if return_code != 0:
                recent = self._recent_errors[-5:]
                detail = "\n  ".join(recent) if recent else f"exit code {return_code}"
                raise SourceError(f"ffmpeg exited (code {return_code}):\n  {detail}")
        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            await self.teardown()

    async def teardown(self) -> None:
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


def _join_api_url(base_url: str, path: str) -> str:
    return f"{base_url.rstrip('/')}{path}"


def _camera_name_from_source(source: str) -> str:
    source_type = detect_source_type(source)
    if source_type == "rtsp":
        return "Edge RTSP Camera"
    if source_type == "webcam":
        return f"Edge Webcam {source}"
    return Path(source).name or "Edge File Camera"
