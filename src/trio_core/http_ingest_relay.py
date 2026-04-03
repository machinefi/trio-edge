"""HTTP MPEG-TS relay transport for Trio Cloud."""

from __future__ import annotations

import hashlib
import logging
import platform
import uuid
from dataclasses import dataclass
from pathlib import Path

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")


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

    def resolved_camera_id(self) -> str:
        return derive_camera_id(self.source, self.camera_id)

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
