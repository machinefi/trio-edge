"""WHIP video relay built on ffmpeg and aiortc."""

from __future__ import annotations

import asyncio
import fractions
import logging
import platform
import shutil
import subprocess
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit, urlunsplit

import aiohttp
import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")

VIDEO_TIME_BASE = fractions.Fraction(1, 90000)
_ICE_GATHERING_TIMEOUT = 5.0
_FFMPEG_STARTUP_TIMEOUT = 0.5
_FFMPEG_READ_SIZE = 32768


class RelayError(Exception):
    """Base exception for relay errors."""


class SourceError(RelayError):
    """Raised when the input source cannot be opened or read."""


class WhipNegotiationError(RelayError):
    """Raised when WHIP SDP negotiation fails."""


class WhipAuthError(WhipNegotiationError):
    """Raised when the WHIP endpoint rejects authentication."""


class WhipAnalysisError(RelayError):
    """Raised when the analysis pipeline cannot be attached."""


class _AnnexBAccessUnitBuffer:
    """Buffer raw H.264 Annex B data and split it into access units."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._closed = False

    def push(self, data: bytes) -> None:
        if data:
            self._buffer.extend(data)

    def close(self) -> None:
        self._closed = True

    def pop(self) -> bytes | None:
        aud_offsets: list[int] = []
        for offset, prefix_len in self._iter_start_codes():
            nal_start = offset + prefix_len
            if nal_start >= len(self._buffer):
                break
            if self._buffer[nal_start] & 0x1F == 9:
                aud_offsets.append(offset)
                if len(aud_offsets) == 2:
                    chunk = bytes(self._buffer[:offset])
                    del self._buffer[:offset]
                    return chunk

        if self._closed and self._buffer:
            chunk = bytes(self._buffer)
            self._buffer.clear()
            return chunk

        return None

    def _iter_start_codes(self) -> list[tuple[int, int]]:
        starts: list[tuple[int, int]] = []
        i = 0
        data = self._buffer
        limit = len(data)
        while i + 3 < limit:
            if data[i] == 0 and data[i + 1] == 0:
                if data[i + 2] == 1:
                    starts.append((i, 3))
                    i += 3
                    continue
                if i + 3 < limit and data[i + 2] == 0 and data[i + 3] == 1:
                    starts.append((i, 4))
                    i += 4
                    continue
            i += 1
        return starts


class H264FfmpegTrack(MediaStreamTrack):
    """Video track that reads H.264 Annex B access units from ffmpeg."""

    kind = "video"

    def __init__(
        self,
        source: str,
        framerate: int = 30,
        resolution: tuple[int, int] | None = None,
    ) -> None:
        super().__init__()
        self._source = source
        self._framerate = framerate
        self._resolution = resolution
        self._process: subprocess.Popen | None = None
        self._pts = 0
        self._pts_increment = VIDEO_TIME_BASE.denominator // framerate
        self._started = False
        self._buffer = _AnnexBAccessUnitBuffer()

    def _build_ffmpeg_cmd(self) -> list[str]:
        source_type = detect_source_type(self._source)
        fps = str(self._framerate)

        if source_type == "rtsp":
            from trio_core._rtsp_proxy import ensure_rtsp_url

            url = ensure_rtsp_url(self._source)
            return [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-rtsp_transport",
                "tcp",
                "-i",
                url,
                "-c:v",
                "copy",
                "-an",
                "-f",
                "h264",
                "-bsf:v",
                "h264_mp4toannexb,h264_metadata=aud=insert",
                "pipe:1",
            ]

        if source_type == "webcam":
            resolution_args: list[str] = []
            if self._resolution:
                resolution_args = ["-video_size", f"{self._resolution[0]}x{self._resolution[1]}"]

            common = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
            ]

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
                    f"/dev/video{self._source}",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-profile:v",
                    "baseline",
                    "-x264-params",
                    "aud=1:repeat-headers=1",
                    "-g",
                    str(self._framerate * 2),
                    "-an",
                    "-f",
                    "h264",
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
                    f"{self._source}:none",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "ultrafast",
                    "-tune",
                    "zerolatency",
                    "-profile:v",
                    "baseline",
                    "-x264-params",
                    "aud=1:repeat-headers=1",
                    "-g",
                    str(self._framerate * 2),
                    "-an",
                    "-f",
                    "h264",
                    "pipe:1",
                ]

            raise SourceError(f"Unsupported webcam platform: {system}")

        if source_type != "file":
            raise SourceError(f"Unsupported source type: {source_type} for '{self._source}'")

        return [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-re",
            "-stream_loop",
            "-1",
            "-i",
            self._source,
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-tune",
            "zerolatency",
            "-profile:v",
            "baseline",
            "-x264-params",
            "aud=1:repeat-headers=1",
            "-g",
            str(self._framerate * 2),
            "-r",
            fps,
            "-an",
            "-f",
            "h264",
            "pipe:1",
        ]

    def _start_ffmpeg(self) -> None:
        if not shutil.which("ffmpeg"):
            raise SourceError(
                "ffmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
            )

        cmd = self._build_ffmpeg_cmd()
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))
        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,
        )

        try:
            self._process.wait(timeout=_FFMPEG_STARTUP_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._started = True
            return

        stderr = b""
        if self._process.stderr is not None:
            stderr = self._process.stderr.read()
        raise SourceError(f"ffmpeg exited immediately: {stderr.decode(errors='replace')[:300]}")

    async def recv(self) -> av.Packet:
        if self.readyState != "live":
            raise MediaStreamError

        if not self._started:
            self._start_ffmpeg()

        loop = asyncio.get_running_loop()
        access_unit = await loop.run_in_executor(None, self._read_next_access_unit)
        if not access_unit:
            self.stop()
            raise MediaStreamError

        packet = av.Packet(access_unit)
        packet.pts = self._pts
        packet.time_base = VIDEO_TIME_BASE
        self._pts += self._pts_increment
        return packet

    def _read_next_access_unit(self) -> bytes:
        assert self._process is not None
        assert self._process.stdout is not None

        while True:
            access_unit = self._buffer.pop()
            if access_unit is not None:
                return access_unit

            try:
                chunk = self._process.stdout.read(_FFMPEG_READ_SIZE)
            except (OSError, ValueError):
                self._buffer.close()
                return self._buffer.pop() or b""

            if not chunk:
                self._buffer.close()
                return self._buffer.pop() or b""

            self._buffer.push(chunk)

    def stop(self) -> None:
        super().stop()
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None


def _probe_rtsp_codec(rtsp_url: str) -> str:
    """Probe the first RTSP video stream codec name."""

    if not shutil.which("ffprobe"):
        raise SourceError(
            "ffprobe not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)"
        )

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-rtsp_transport",
            "tcp",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "csv=p=0",
            rtsp_url,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    codec = result.stdout.strip().splitlines()[0].strip() if result.stdout.strip() else ""
    if not codec:
        raise SourceError(
            f"Cannot probe RTSP stream: {rtsp_url}\n  ffprobe stderr: {result.stderr[:200]}"
        )
    return codec


@dataclass
class WhipRelay:
    """Orchestrate the lifecycle of a WHIP relay session."""

    source: str
    whip_url: str
    bearer_token: str | None = None
    resolution: tuple[int, int] | None = None
    framerate: int = 30

    _pc: RTCPeerConnection | None = None
    _session_url: str | None = None
    _track: H264FfmpegTrack | None = None

    async def run(self) -> None:
        source_type = detect_source_type(self.source)

        if source_type == "rtsp":
            from trio_core._rtsp_proxy import ensure_rtsp_url

            codec = _probe_rtsp_codec(ensure_rtsp_url(self.source))
            if codec.lower() != "h264":
                raise SourceError(
                    f"RTSP stream codec is '{codec}', but WHIP relay requires H.264.\n"
                    "  Configure your camera to use H.264 encoding."
                )
            logger.info("RTSP codec: %s (passthrough, no re-encoding)", codec)
        elif source_type == "webcam":
            logger.info("Webcam source detected; encoding to H.264 via ffmpeg")
        elif source_type == "file":
            logger.info("File source detected; encoding to H.264 via ffmpeg")
        else:
            raise SourceError(f"Unsupported source type: {source_type} for '{self.source}'")

        self._track = H264FfmpegTrack(
            self.source,
            framerate=self.framerate,
            resolution=self.resolution,
        )

        self._pc = RTCPeerConnection()
        self._pc.addTransceiver("video", direction="sendonly")
        self._pc.addTrack(self._track)

        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        await self._wait_for_ice_gathering()
        logger.info("SDP offer ready (%d bytes)", len(self._pc.localDescription.sdp))

        await self._negotiate()

        connected = asyncio.Event()

        @self._pc.on("iceconnectionstatechange")
        def on_ice_state() -> None:
            state = self._pc.iceConnectionState
            logger.info("ICE connection state: %s", state)
            if state in {"connected", "completed", "failed", "closed"}:
                connected.set()

        try:
            await asyncio.wait_for(connected.wait(), timeout=15.0)
        except asyncio.TimeoutError as exc:
            raise WhipNegotiationError(
                f"ICE connection timed out (state: {self._pc.iceConnectionState})"
            ) from exc

        if self._pc.iceConnectionState not in {"connected", "completed"}:
            raise WhipNegotiationError(
                f"ICE connection failed (state: {self._pc.iceConnectionState})"
            )

        await self._attach_analysis()
        logger.info("Connected; streaming H.264 video to %s", self.whip_url)

        while True:
            await asyncio.sleep(1)
            if self._pc.connectionState in {"failed", "closed"}:
                logger.warning("Connection lost (state: %s)", self._pc.connectionState)
                return

    async def _wait_for_ice_gathering(self) -> None:
        assert self._pc is not None
        if self._pc.iceGatheringState == "complete":
            return

        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_gathering() -> None:
            if self._pc is not None and self._pc.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=_ICE_GATHERING_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out (state: %s)", self._pc.iceGatheringState)

    async def _post_offer(
        self,
        session: aiohttp.ClientSession,
        url: str,
        sdp_offer: str,
        headers: dict[str, str],
    ) -> tuple[object, str]:
        async with session.post(
            url,
            data=sdp_offer,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=10),
        ) as resp:
            return resp, await resp.text()

    async def _negotiate(self) -> None:
        assert self._pc is not None
        assert self._pc.localDescription is not None

        headers = {"Content-Type": "application/sdp"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        sdp_offer = self._pc.localDescription.sdp

        async with aiohttp.ClientSession() as session:
            resp, body = await self._post_offer(session, self.whip_url, sdp_offer, headers)

            if resp.status == 307:
                redirect_url = resp.headers.get("Location")
                if not redirect_url:
                    raise WhipNegotiationError("Server returned 307 without Location header")
                redirect_url = urljoin(self.whip_url, redirect_url)
                logger.info("Following redirect to %s", redirect_url)
                resp, body = await self._post_offer(session, redirect_url, sdp_offer, headers)

            if resp.status in {401, 403}:
                raise WhipAuthError(
                    f"WHIP authentication failed (HTTP {resp.status}): {body[:200]}"
                )

            if resp.status != 201:
                raise WhipNegotiationError(
                    f"WHIP negotiation failed (HTTP {resp.status}): {body[:300]}"
                )

            location = resp.headers.get("Location", "")
            self._session_url = urljoin(self.whip_url, location) if location else None
            logger.info("WHIP session: %s", self._session_url)

            answer = RTCSessionDescription(sdp=body, type="answer")
            await self._pc.setRemoteDescription(answer)

    def _analysis_url(self) -> str:
        if not self._session_url:
            raise WhipAnalysisError("Cannot attach analysis before WHIP session is established")

        parsed = urlsplit(self._session_url)
        prefix, marker, session_id = parsed.path.rpartition("/whip/")
        if not marker or not session_id:
            raise WhipAnalysisError(f"Unexpected WHIP session URL: {self._session_url}")

        path = f"{prefix}/sessions/{session_id}/analyze"
        return urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    async def _attach_analysis(self) -> None:
        analyze_url = self._analysis_url()
        headers: dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        async with aiohttp.ClientSession() as session:
            async with session.post(
                analyze_url,
                json={},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()
                if resp.status != 200:
                    raise WhipAnalysisError(
                        f"Analysis attach failed (HTTP {resp.status}): {body[:300]}"
                    )
        logger.info("Attached analysis pipeline: %s", analyze_url)

    async def teardown(self) -> None:
        session_url = self._session_url
        self._session_url = None

        if self._track is not None:
            self._track.stop()
            self._track = None

        if self._pc is not None:
            await self._pc.close()
            self._pc = None

        if not session_url:
            return

        try:
            headers = {}
            if self.bearer_token:
                headers["Authorization"] = f"Bearer {self.bearer_token}"
            async with aiohttp.ClientSession() as session:
                async with session.delete(
                    session_url,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    logger.info("DELETE %s -> %d", session_url, resp.status)
        except Exception as exc:
            logger.warning("Failed to delete WHIP session: %s", exc)
