"""HTTP MPEG-TS relay transport for Trio Cloud."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import platform
import shutil
import uuid
from dataclasses import dataclass
from pathlib import Path

import httpx

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")
_FFMPEG_READ_SIZE = 65536


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

    def __post_init__(self) -> None:
        self._process: asyncio.subprocess.Process | None = None

    def resolved_camera_id(self) -> str:
        return derive_camera_id(self.source, self.camera_id)

    def _auth_headers(self, *, content_type: str | None = None) -> dict[str, str]:
        headers = {"Authorization": f"Bearer {self.bearer_token}"}
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

    def _build_ffmpeg_cmd(self) -> list[str]:
        source_type = detect_source_type(self.source)
        fps = str(self.framerate)

        if source_type == "rtsp":
            from trio_core._rtsp_proxy import ensure_rtsp_url

            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
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

            common = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
            system = platform.system()
            if system == "Linux":
                return [
                    *common,
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
                    *common,
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
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
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
        if returned_id != camera_id:
            raise CameraRegistrationError(
                f"Cloud camera registration did not honor requested camera_id '{camera_id}'"
            )
        return camera_id

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

    async def _mpegts_body(self):
        if self._process is None or self._process.stdout is None:
            raise SourceError("ffmpeg process was not started")

        while True:
            chunk = await self._process.stdout.read(_FFMPEG_READ_SIZE)
            if chunk:
                yield chunk
                continue

            return_code = await self._process.wait()
            if return_code != 0:
                stderr = b""
                if self._process.stderr is not None:
                    stderr = await self._process.stderr.read()
                raise SourceError(
                    f"ffmpeg exited unexpectedly (code {return_code}): {stderr.decode(errors='replace')[:200]}"
                )
            break

    async def run(self) -> None:
        async with httpx.AsyncClient(timeout=None, follow_redirects=True) as client:
            camera_id = await self._register_camera(client)
            await self._start_ffmpeg()
            try:
                async with client.stream(
                    "POST",
                    _join_api_url(self.cloud_url, f"/api/stream/ingest/{camera_id}"),
                    headers=self._auth_headers(content_type="video/mp2t"),
                    content=self._mpegts_body(),
                ) as response:
                    if response.status_code not in (200, 201, 202, 204):
                        body = await response.aread()
                        raise IngestUploadError(
                            f"HTTP ingest failed (HTTP {response.status_code}): {body.decode(errors='replace')[:200]}"
                        )
            finally:
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
