# Add `trio relay` — WHIP Video Relay Command (v3)

Publish video from a local webcam or LAN camera (RTSP) to trio-cortex's WHIP ingest server, per RFC 9725.

---

## User Review Required

> [!IMPORTANT]
> **New dependency: `aiortc>=1.9`** — used for `RTCPeerConnection`, SDP/ICE/DTLS/SRTP, and H.264 RTP packetization. Video capture and encoding is done by **ffmpeg** (subprocess), not aiortc's MediaPlayer. `aiohttp>=3.9` is used for the HTTP POST/DELETE to the WHIP endpoint.

> [!IMPORTANT]
> **H.264 only, video-only, no audio.** The trio-cortex server rejects non-H264 offers. RTSP sources that are not H.264 exit with a clear error.

---

## Target Server: trio-cortex

The WHIP server lives at `/home/haaai/iotex/trio-cortex`:

| Server component | What it does |
|---|---|
| [api/routers/stream.py#L63-L97](file:///home/haaai/iotex/trio-cortex/api/routers/stream.py#L63-L97) | `POST /api/stream/whip` — receives SDP offer, returns 201 + SDP answer + `Location` header |
| [cortex/stream/whip/sdp.py#L31-L32](file:///home/haaai/iotex/trio-cortex/cortex/stream/whip/sdp.py#L31-L32) | `_offer_supports_h264()` — **rejects non-H264 offers** (`"H264/90000" in sdp_offer.upper()`) |
| [cortex/stream/whip/sdp.py#L40-L45](file:///home/haaai/iotex/trio-cortex/cortex/stream/whip/sdp.py#L40-L45) | `_offer_direction_is_valid()` — rejects `a=recvonly` or `a=inactive` |
| [cortex/stream/whip/sdp.py#L48-L63](file:///home/haaai/iotex/trio-cortex/cortex/stream/whip/sdp.py#L48-L63) | `_add_preferred_transceivers()` — server adds video transceiver with H.264 codec preference |
| [cortex/stream/whip/ingest.py#L113-L179](file:///home/haaai/iotex/trio-cortex/cortex/stream/whip/ingest.py#L113-L179) | `consume_video_track()` — reads raw H.264 NAL units, appends to `MediaBuffer` |
| [cortex/stream/whip/aiortc_patch.py](file:///home/haaai/iotex/trio-cortex/cortex/stream/whip/aiortc_patch.py) | Server-side decoder bypass — forwards compressed H.264 frames without decoding |
| [api/routers/stream.py#L120-L126](file:///home/haaai/iotex/trio-cortex/api/routers/stream.py#L120-L126) | `DELETE /api/stream/whip/{session_id}` — session teardown |

---

## WHIP SDP/Signaling Flow

```
┌────────────┐                         ┌──────────────────────────┐
│ trio relay │                         │  trio-cortex             │
│  (client)  │                         │  POST /api/stream/whip   │
└─────┬──────┘                         └───────────┬──────────────┘
      │                                            │
      │  1. HTTP POST /api/stream/whip             │
      │     Content-Type: application/sdp          │
      │     Authorization: Bearer <token>          │
      │     Body: SDP offer (sendonly, H.264)      │
      ├───────────────────────────────────────────►│
      │                                            │
      │  2. 201 Created                            │
      │     Content-Type: application/sdp          │
      │     Location: /api/stream/whip/<session-id>│
      │     ETag: "<session-id>"                   │
      │     Body: SDP answer (recvonly)            │
      │◄───────────────────────────────────────────┤
      │                                            │
      │  3. ICE connectivity checks (STUN)         │
      │  4. DTLS handshake                         │
      │  5. SRTP — H.264 RTP packets (send only)   │
      ├═══════════════════════════════════════════►│
      │                                            │
      │  6. HTTP DELETE /api/stream/whip/<id>       │  (Ctrl+C / error)
      ├───────────────────────────────────────────►│
      │  200 OK                                    │
      │◄───────────────────────────────────────────┤
```

---

## Key Architecture Decision: Unified ffmpeg Pipeline

### Why ffmpeg for Both Webcam and RTSP

Both webcam and RTSP sources use **ffmpeg as a subprocess** to produce H.264 Annex B output on stdout:

| Source | ffmpeg command | Encoding |
|---|---|---|
| **Webcam** (Linux) | `ffmpeg -f v4l2 -video_size {W}x{H} -framerate {fps} -i /dev/video{N} -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -g {fps*2} -an -f h264 pipe:1` | **Encodes** raw frames to H.264 |
| **Webcam** (macOS) | `ffmpeg -f avfoundation -framerate {fps} -video_size {W}x{H} -i "{N}:none" -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -g {fps*2} -an -f h264 pipe:1` | **Encodes** raw frames to H.264 |
| **RTSP** (H.264) | `ffmpeg -rtsp_transport tcp -i {rtsp_url} -c:v copy -an -f h264 -bsf:v h264_mp4toannexb pipe:1` | **No re-encoding** (`-c:v copy`) |
| **File** | `ffmpeg -re -stream_loop -1 -i {path} -c:v libx264 -preset ultrafast -tune zerolatency -profile:v baseline -an -f h264 pipe:1` | Re-encodes if not H.264 |

**Benefits:**
- One code path for reading H.264 NAL units from ffmpeg stdout, regardless of source
- ffmpeg handles platform-specific capture (V4L2, AVFoundation) with well-tested code
- RTSP passthrough is trivial with `-c:v copy` — zero CPU encoding overhead
- No dependency on `aiortc.contrib.media.MediaPlayer`

### How Pre-encoded H.264 Flows Through aiortc (No Monkey-Patching)

aiortc **natively supports** pre-encoded packet passthrough. Here's the exact code flow:

1. Our custom `H264FfmpegTrack.recv()` returns an `av.Packet` (not `av.Frame`)
2. [`RTCRtpSender._next_encoded_frame()`](file:///home/haaai/iotex/trio-cortex/.venv/lib/python3.12/site-packages/aiortc/rtcrtpsender.py) checks `isinstance(data, Frame)` at line 311:
   ```python
   if isinstance(data, Frame):
       # encode the frame (NOT our path)
       payloads, timestamp = self.__encoder.encode(data, force_keyframe)
   else:
       # OUR PATH: pack pre-encoded data
       payloads, timestamp = self.__encoder.pack(data)
   ```
3. [`H264Encoder.pack()`](file:///home/haaai/iotex/trio-cortex/.venv/lib/python3.12/site-packages/aiortc/codecs/h264.py) at line 298:
   ```python
   def pack(self, packet: Packet) -> tuple[list[bytes], int]:
       packages = self._split_bitstream(bytes(packet))      # split Annex B NAL units
       timestamp = convert_timebase(packet.pts, packet.time_base, VIDEO_TIME_BASE)
       return self._packetize(packages), timestamp           # RTP packetize (FU-A / STAP-A)
   ```
4. The H.264 NAL units are split and RTP-packetized (FU-A fragmentation for large NALs, STAP-A aggregation for small ones)

**No encoder is invoked. No monkey-patching is needed.**

The `av.Packet` must have:
- `.data` = H.264 Annex B bytes (NAL units with `00 00 00 01` start codes)
- `.pts` = presentation timestamp (integer, in `time_base` units)
- `.time_base` = `fractions.Fraction(1, 90000)` (= `VIDEO_TIME_BASE`)

### RTSP Non-H264 Handling

Before starting the relay, we probe the RTSP stream with ffprobe:
```bash
ffprobe -v error -rtsp_transport tcp -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 {rtsp_url}
```
If the result is not `h264`, exit with error:
```
✗ RTSP stream codec is 'hevc', but WHIP relay requires H.264.
  Configure your camera to use H.264, or convert with:
  ffmpeg -i rtsp://... -c:v libx264 -f rtsp rtsp://...
```

---

## Proposed Changes

### File Summary

| File | Action | Purpose |
|------|--------|---------|
| [pyproject.toml](file:///home/haaai/iotex/trio-edge/pyproject.toml) | MODIFY | Add `relay` optional deps |
| [whip_relay.py](file:///home/haaai/iotex/trio-edge/src/trio_core/whip_relay.py) | NEW | WHIP relay engine |
| [cli.py](file:///home/haaai/iotex/trio-edge/src/trio_core/cli.py) | MODIFY | Add `trio relay` command |
| [test_whip_relay.py](file:///home/haaai/iotex/trio-edge/tests/test_whip_relay.py) | NEW | Unit tests |

---

### 1. [MODIFY] [pyproject.toml](file:///home/haaai/iotex/trio-edge/pyproject.toml)

Add `relay` to `[project.optional-dependencies]`:

```toml
relay = ["aiortc>=1.9", "aiohttp>=3.9"]
```

Update `all` to include it:

```toml
all = ["trio-edge[mlx,webcam,claw,relay]"]
```

---

### 2. [NEW] [whip_relay.py](file:///home/haaai/iotex/trio-edge/src/trio_core/whip_relay.py)

~350 lines. Detailed specification follows.

#### 2.1 Imports and Constants

```python
from __future__ import annotations

import asyncio
import fractions
import logging
import os
import platform
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass

import aiohttp
import av
from aiortc import RTCPeerConnection, RTCSessionDescription
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack

from trio_core.source_resolver import detect_source_type

logger = logging.getLogger("trio.relay")

VIDEO_TIME_BASE = fractions.Fraction(1, 90000)  # matches aiortc VIDEO_TIME_BASE
_ICE_GATHERING_TIMEOUT = 5.0   # seconds
_FFMPEG_STARTUP_TIMEOUT = 5.0  # seconds
```

#### 2.2 Exception Classes

```python
class RelayError(Exception):
    """Base exception for relay errors."""

class SourceError(RelayError):
    """Cannot open or read from the video source."""

class WhipNegotiationError(RelayError):
    """WHIP endpoint returned an error during SDP negotiation."""

class WhipAuthError(WhipNegotiationError):
    """WHIP endpoint returned 401 or 403."""
```

#### 2.3 `H264FfmpegTrack` — Custom MediaStreamTrack

This is the core abstraction. It reads H.264 Annex B data from ffmpeg stdout and yields `av.Packet` objects.

```python
class H264FfmpegTrack(MediaStreamTrack):
    """Video track that reads H.264 from an ffmpeg subprocess.

    recv() returns av.Packet objects (not av.Frame), so aiortc's
    RTCRtpSender will call H264Encoder.pack() for RTP packetization
    without re-encoding.
    """

    kind = "video"

    def __init__(self, source: str, framerate: int = 30,
                 resolution: tuple[int, int] | None = None) -> None:
        super().__init__()
        self._source = source
        self._framerate = framerate
        self._resolution = resolution
        self._process: subprocess.Popen | None = None
        self._pts = 0
        self._pts_increment = VIDEO_TIME_BASE.denominator // framerate  # 90000 / fps = 3000 at 30fps
        self._started = False

    def _build_ffmpeg_cmd(self) -> list[str]:
        """Build the ffmpeg command line based on source type."""
        source_type = detect_source_type(self._source)
        fps = str(self._framerate)

        if source_type == "rtsp":
            # RTSP H.264 passthrough — no re-encoding
            from trio_core._rtsp_proxy import ensure_rtsp_url
            url = ensure_rtsp_url(self._source)
            return [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-rtsp_transport", "tcp",
                "-i", url,
                "-c:v", "copy",       # no re-encoding
                "-an",                 # no audio
                "-f", "h264",          # output format: raw H.264 Annex B
                "-bsf:v", "h264_mp4toannexb",
                "pipe:1",
            ]

        if source_type == "webcam":
            idx = self._source  # "0", "1", etc.
            res_args = []
            if self._resolution:
                res_args = ["-video_size", f"{self._resolution[0]}x{self._resolution[1]}"]

            if platform.system() == "Linux":
                return [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "v4l2",
                    *res_args,
                    "-framerate", fps,
                    "-i", f"/dev/video{idx}",
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-profile:v", "baseline",
                    "-g", str(self._framerate * 2),  # keyframe every 2s
                    "-an",
                    "-f", "h264",
                    "pipe:1",
                ]
            else:
                # macOS (avfoundation)
                return [
                    "ffmpeg", "-hide_banner", "-loglevel", "error",
                    "-f", "avfoundation",
                    "-framerate", fps,
                    *res_args,
                    "-i", f"{idx}:none",  # video:audio (none = no audio)
                    "-c:v", "libx264",
                    "-preset", "ultrafast",
                    "-tune", "zerolatency",
                    "-profile:v", "baseline",
                    "-g", str(self._framerate * 2),
                    "-an",
                    "-f", "h264",
                    "pipe:1",
                ]

        # File source (video file — re-encode to H.264, loop, real-time pacing)
        return [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-re",                    # real-time pacing
            "-stream_loop", "-1",     # loop forever
            "-i", self._source,
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-g", str(self._framerate * 2),
            "-r", fps,
            "-an",
            "-f", "h264",
            "pipe:1",
        ]

    def _start_ffmpeg(self) -> None:
        """Start the ffmpeg subprocess."""
        if not shutil.which("ffmpeg"):
            raise SourceError("ffmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)")

        cmd = self._build_ffmpeg_cmd()
        logger.debug("ffmpeg cmd: %s", " ".join(cmd))

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=0,  # unbuffered for low latency
        )

        # Wait briefly and check ffmpeg didn't exit immediately
        try:
            self._process.wait(timeout=0.5)
            stderr = self._process.stderr.read().decode(errors="replace")
            raise SourceError(f"ffmpeg exited immediately: {stderr[:300]}")
        except subprocess.TimeoutExpired:
            pass  # Good — ffmpeg is running

        self._started = True

    async def recv(self) -> av.Packet:
        """Read the next H.264 access unit from ffmpeg stdout.

        Returns an av.Packet with:
          - .data = H.264 Annex B bytes
          - .pts  = monotonic timestamp
          - .time_base = Fraction(1, 90000)
        """
        if self.readyState != "live":
            raise MediaStreamError

        if not self._started:
            self._start_ffmpeg()

        loop = asyncio.get_event_loop()
        chunk = await loop.run_in_executor(None, self._read_next_au)

        if not chunk:
            self.stop()
            raise MediaStreamError

        packet = av.Packet(chunk)
        packet.pts = self._pts
        packet.time_base = VIDEO_TIME_BASE
        self._pts += self._pts_increment
        return packet

    def _read_next_au(self) -> bytes:
        """Read the next H.264 access unit (one frame's NAL units) from stdout.

        Reads from ffmpeg's raw H.264 Annex B output. An access unit boundary
        is detected by finding the next start code after we've accumulated
        at least one NAL unit. For simplicity, we read fixed-size chunks and
        let aiortc's H264Encoder._split_bitstream() handle NAL splitting.

        Returns raw H.264 bytes or b"" on EOF/error.
        """
        assert self._process is not None
        # Read a chunk of H.264 data. The size is chosen to be roughly
        # one frame at moderate bitrate (1 Mbps at 30fps ≈ 4KB/frame).
        # Larger reads are fine — aiortc splits NAL units internally.
        try:
            data = self._process.stdout.read(32768)  # 32KB chunks
        except (OSError, ValueError):
            return b""
        return data if data else b""

    def stop(self) -> None:
        """Stop the track and terminate ffmpeg."""
        super().stop()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None
```

> [!NOTE]
> **NAL unit framing**: We read arbitrary-sized chunks from ffmpeg stdout and return them as `av.Packet`. aiortc's `H264Encoder.pack()` calls `_split_bitstream()` which finds NAL unit boundaries (`00 00 00 01` / `00 00 01` start codes) in the raw bytes and packetizes them into RTP FU-A / STAP-A payloads. We don't need to do our own NAL parsing — aiortc handles it.

> [!NOTE]
> **Timing**: Each `recv()` call increments PTS by `90000 / fps` ticks. For 30fps, that's 3000 ticks per frame. This provides a monotonic clock for RTP timestamps. The actual frame pacing is controlled by ffmpeg's output rate (real-time for webcam, `-re` for files).

#### 2.4 RTSP Codec Probe

```python
def _probe_rtsp_codec(rtsp_url: str) -> str:
    """Probe the video codec of an RTSP stream. Returns codec name or raises SourceError."""
    result = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-rtsp_transport", "tcp",
            "-select_streams", "v:0",
            "-show_entries", "stream=codec_name",
            "-of", "csv=p=0",
            rtsp_url,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    codec = result.stdout.strip().split("\n")[0].strip()
    if not codec:
        raise SourceError(f"Cannot probe RTSP stream: {rtsp_url}\n  ffprobe stderr: {result.stderr[:200]}")
    return codec
```

#### 2.5 `WhipRelay` — Orchestrator Class

```python
@dataclass
class WhipRelay:
    """Orchestrates the WHIP relay lifecycle."""

    source: str
    whip_url: str
    bearer_token: str | None = None
    resolution: tuple[int, int] | None = None
    framerate: int = 30

    _pc: RTCPeerConnection | None = None
    _session_url: str | None = None
    _track: H264FfmpegTrack | None = None

    async def run(self) -> None:
        """Main entry point. Opens source, negotiates WHIP, streams until stopped."""
        source_type = detect_source_type(self.source)

        # Validate RTSP codec before starting
        if source_type == "rtsp":
            from trio_core._rtsp_proxy import ensure_rtsp_url
            probe_url = ensure_rtsp_url(self.source)
            codec = _probe_rtsp_codec(probe_url)
            if codec.lower() != "h264":
                raise SourceError(
                    f"RTSP stream codec is '{codec}', but WHIP relay requires H.264.\n"
                    f"  Configure your camera to use H.264 encoding."
                )
            logger.info("RTSP codec: %s (passthrough, no re-encoding)", codec)
        elif source_type == "webcam":
            logger.info("Webcam source — encoding to H.264 via ffmpeg libx264")
        elif source_type == "file":
            logger.info("File source — encoding to H.264 via ffmpeg libx264")
        else:
            raise SourceError(f"Unsupported source type: {source_type} for '{self.source}'")

        # Create track
        self._track = H264FfmpegTrack(
            self.source,
            framerate=self.framerate,
            resolution=self.resolution,
        )

        # Create PeerConnection, add video-only transceiver with H.264
        self._pc = RTCPeerConnection()
        self._pc.addTransceiver("video", direction="sendonly")
        self._pc.addTrack(self._track)

        # Create SDP offer, gather ICE candidates
        offer = await self._pc.createOffer()
        await self._pc.setLocalDescription(offer)
        await self._wait_for_ice_gathering()
        logger.info("SDP offer ready (%d bytes)", len(self._pc.localDescription.sdp))
        logger.debug("SDP offer:\n%s", self._pc.localDescription.sdp)

        # POST to WHIP endpoint
        await self._negotiate()

        # Wait for ICE connection
        connected = asyncio.Event()

        @self._pc.on("iceconnectionstatechange")
        def on_ice_state():
            state = self._pc.iceConnectionState
            logger.info("ICE connection state: %s", state)
            if state == "connected":
                connected.set()
            elif state == "failed":
                connected.set()  # unblock so we can handle failure

        try:
            await asyncio.wait_for(connected.wait(), timeout=15.0)
        except asyncio.TimeoutError:
            raise WhipNegotiationError(
                f"ICE connection timed out (state: {self._pc.iceConnectionState})"
            )

        if self._pc.iceConnectionState != "connected":
            raise WhipNegotiationError(
                f"ICE connection failed (state: {self._pc.iceConnectionState})"
            )

        logger.info("Connected! Streaming H.264 video to %s", self.whip_url)

        # Keep alive until cancelled
        try:
            while True:
                await asyncio.sleep(1)
                if self._pc.connectionState in ("failed", "closed"):
                    logger.warning("Connection lost (state: %s)", self._pc.connectionState)
                    break
        except asyncio.CancelledError:
            pass

    async def _wait_for_ice_gathering(self) -> None:
        """Wait for ICE candidate gathering to complete."""
        if self._pc.iceGatheringState == "complete":
            return
        done = asyncio.Event()

        @self._pc.on("icegatheringstatechange")
        def on_gathering():
            if self._pc.iceGatheringState == "complete":
                done.set()

        try:
            await asyncio.wait_for(done.wait(), timeout=_ICE_GATHERING_TIMEOUT)
        except asyncio.TimeoutError:
            logger.warning("ICE gathering timed out (state: %s)", self._pc.iceGatheringState)

    async def _negotiate(self) -> None:
        """Send SDP offer to WHIP endpoint via HTTP POST."""
        headers = {"Content-Type": "application/sdp"}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"

        sdp_offer = self._pc.localDescription.sdp

        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.whip_url,
                data=sdp_offer,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                body = await resp.text()

                if resp.status in (401, 403):
                    raise WhipAuthError(
                        f"WHIP authentication failed (HTTP {resp.status}): {body[:200]}"
                    )

                if resp.status == 307:
                    # Follow redirect (RFC 9725 §4.5)
                    redirect_url = resp.headers.get("Location")
                    if redirect_url:
                        logger.info("Following redirect to %s", redirect_url)
                        async with session.post(
                            redirect_url, data=sdp_offer, headers=headers,
                            timeout=aiohttp.ClientTimeout(total=10),
                        ) as redirect_resp:
                            body = await redirect_resp.text()
                            resp = redirect_resp
                    else:
                        raise WhipNegotiationError("Server returned 307 without Location header")

                if resp.status != 201:
                    raise WhipNegotiationError(
                        f"WHIP negotiation failed (HTTP {resp.status}): {body[:300]}"
                    )

                # Extract session URL from Location header
                self._session_url = resp.headers.get("Location", "")
                if self._session_url and not self._session_url.startswith("http"):
                    # Relative URL — resolve against WHIP endpoint
                    from urllib.parse import urljoin
                    self._session_url = urljoin(self.whip_url, self._session_url)

                logger.info("WHIP session: %s", self._session_url)

                # Set remote description (SDP answer)
                answer = RTCSessionDescription(sdp=body, type="answer")
                await self._pc.setRemoteDescription(answer)

    async def teardown(self) -> None:
        """Clean up: stop track, close PeerConnection, DELETE WHIP session."""
        if self._track:
            self._track.stop()
            self._track = None

        if self._pc:
            await self._pc.close()
            self._pc = None

        if self._session_url:
            try:
                headers = {}
                if self.bearer_token:
                    headers["Authorization"] = f"Bearer {self.bearer_token}"
                async with aiohttp.ClientSession() as session:
                    async with session.delete(
                        self._session_url,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=5),
                    ) as resp:
                        logger.info("DELETE %s → %d", self._session_url, resp.status)
            except Exception as e:
                logger.warning("Failed to delete WHIP session: %s", e)
            self._session_url = None
```

---

### 3. [MODIFY] [cli.py](file:///home/haaai/iotex/trio-edge/src/trio_core/cli.py)

Add the `relay` command. Insert before `_die_load_error()` (around line 733), following the existing pattern.

```python
@app.command()
def relay(
    whip_url: str = typer.Argument(
        ..., help="WHIP ingest endpoint URL (e.g. http://cortex:8000/api/stream/whip)"
    ),
    source: str = typer.Option(
        "0", "--source", "-s",
        help="Video source: camera index (0, 1), RTSP URL, or video file",
    ),
    token: str = typer.Option(
        None, "--token", "-t", help="Bearer token for WHIP authentication"
    ),
    resolution: str = typer.Option(
        None, "--resolution", "-r",
        help="Video resolution WxH (e.g. 1280x720). Default: camera native",
    ),
    framerate: int = typer.Option(
        30, "--framerate", "--fps", help="Target frame rate"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    json_logs: bool = typer.Option(
        False, "--json-logs",
        help="Structured JSON logging (or set TRIO_LOG_JSON=1)",
    ),
):
    """Relay video from a webcam or camera to a WHIP ingest server.

    Captures video from a local webcam or LAN camera (RTSP) and publishes
    it to a remote WebRTC server using the WHIP protocol (RFC 9725).
    Video is sent as H.264. RTSP streams already in H.264 are forwarded
    without re-encoding.

    Examples:
        trio relay http://cortex:8000/api/stream/whip
        trio relay http://cortex:8000/api/stream/whip -t mytoken
        trio relay http://cortex:8000/api/stream/whip -s rtsp://admin:pass@ip/stream
        trio relay http://cortex:8000/api/stream/whip -s 1 --fps 15
        trio relay http://cortex:8000/api/stream/whip -s video.mp4
    """
    _setup_logging(verbose, json_logs=json_logs)
    import asyncio

    # Parse resolution
    res_tuple = None
    if resolution:
        try:
            w, h = resolution.lower().split("x")
            res_tuple = (int(w), int(h))
        except ValueError:
            typer.echo(f"✗ Invalid resolution format: {resolution} (expected WxH, e.g. 1280x720)", err=True)
            raise typer.Exit(1)

    # Lazy import — relay deps are optional
    try:
        from trio_core.whip_relay import WhipRelay, RelayError
    except ImportError as e:
        typer.echo(f"✗ Missing dependency: {e}", err=True)
        typer.echo("  Install with: pip install 'trio-edge[relay]'", err=True)
        raise typer.Exit(1)

    # Check ffmpeg
    if not shutil.which("ffmpeg"):
        typer.echo("✗ ffmpeg not found. Install with: apt install ffmpeg (Linux) or brew install ffmpeg (macOS)", err=True)
        raise typer.Exit(1)

    relay_obj = WhipRelay(
        source=source,
        whip_url=whip_url,
        bearer_token=token,
        resolution=res_tuple,
        framerate=framerate,
    )

    typer.echo(f"Relay: {source} → {whip_url}")
    typer.echo(f"Codec: H.264 | FPS: {framerate} | Resolution: {resolution or 'native'}")
    typer.echo("Press Ctrl+C to stop.\n")

    async def _run():
        try:
            await relay_obj.run()
        finally:
            await relay_obj.teardown()

    try:
        asyncio.run(_run())
    except RelayError as e:
        typer.echo(f"\n✗ {e}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\nStopping relay...")
        asyncio.run(relay_obj.teardown())
        typer.echo("Disconnected.")
```

---

### 4. [NEW] [test_whip_relay.py](file:///home/haaai/iotex/trio-edge/tests/test_whip_relay.py)

~150 lines. Tests mock the network layer and subprocess calls.

```python
"""Tests for trio_core.whip_relay."""
from __future__ import annotations

import fractions
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

aiortc = pytest.importorskip("aiortc")
aiohttp = pytest.importorskip("aiohttp")

from trio_core.whip_relay import (
    H264FfmpegTrack,
    WhipRelay,
    WhipAuthError,
    WhipNegotiationError,
    SourceError,
    _probe_rtsp_codec,
)


class TestH264FfmpegTrack:
    def test_build_cmd_webcam_linux(self):
        track = H264FfmpegTrack("0", framerate=30)
        with patch("platform.system", return_value="Linux"):
            cmd = track._build_ffmpeg_cmd()
        assert "v4l2" in cmd
        assert "/dev/video0" in cmd
        assert "libx264" in cmd
        assert "copy" not in cmd  # webcam must encode

    def test_build_cmd_webcam_macos(self):
        track = H264FfmpegTrack("0", framerate=30)
        with patch("platform.system", return_value="Darwin"):
            cmd = track._build_ffmpeg_cmd()
        assert "avfoundation" in cmd
        assert "0:none" in cmd
        assert "libx264" in cmd

    def test_build_cmd_rtsp(self):
        track = H264FfmpegTrack("rtsp://admin:pass@192.168.1.1/stream", framerate=30)
        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://admin:pass@192.168.1.1/stream"):
            cmd = track._build_ffmpeg_cmd()
        assert "copy" in cmd       # RTSP must NOT re-encode
        assert "libx264" not in cmd
        assert "h264_mp4toannexb" in cmd

    def test_build_cmd_file(self):
        track = H264FfmpegTrack("video.mp4", framerate=15)
        cmd = track._build_ffmpeg_cmd()
        assert "-re" in cmd           # real-time pacing
        assert "-stream_loop" in cmd  # loop
        assert "libx264" in cmd

    def test_build_cmd_resolution(self):
        track = H264FfmpegTrack("0", framerate=30, resolution=(1280, 720))
        with patch("platform.system", return_value="Linux"):
            cmd = track._build_ffmpeg_cmd()
        assert "1280x720" in cmd

    def test_pts_increment_30fps(self):
        track = H264FfmpegTrack("0", framerate=30)
        assert track._pts_increment == 3000  # 90000 / 30

    def test_pts_increment_15fps(self):
        track = H264FfmpegTrack("0", framerate=15)
        assert track._pts_increment == 6000  # 90000 / 15


class TestProbeRtspCodec:
    def test_h264_detected(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="h264\n", stderr="")
            assert _probe_rtsp_codec("rtsp://test") == "h264"

    def test_hevc_detected(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="hevc\n", stderr="")
            assert _probe_rtsp_codec("rtsp://test") == "hevc"

    def test_empty_raises(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(stdout="", stderr="connection refused")
            with pytest.raises(SourceError, match="Cannot probe"):
                _probe_rtsp_codec("rtsp://bad")


class TestWhipNegotiation:
    @pytest.fixture
    def relay(self):
        return WhipRelay(source="0", whip_url="http://test:8000/api/stream/whip")

    @pytest.mark.asyncio
    async def test_negotiate_success(self, relay):
        """Mock 201 Created with SDP answer and Location header."""
        mock_resp = AsyncMock()
        mock_resp.status = 201
        mock_resp.text = AsyncMock(return_value="v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n")
        mock_resp.headers = {"Location": "/api/stream/whip/session-abc123"}

        relay._pc = MagicMock()
        relay._pc.localDescription = MagicMock()
        relay._pc.localDescription.sdp = "v=0\r\n"
        relay._pc.setRemoteDescription = AsyncMock()

        with patch("aiohttp.ClientSession") as mock_session:
            mock_ctx = AsyncMock()
            mock_ctx.__aenter__.return_value = mock_resp
            mock_session.return_value.__aenter__.return_value.post.return_value = mock_ctx
            # ... (full mock setup; exact form depends on aiohttp mock patterns)

    @pytest.mark.asyncio
    async def test_negotiate_auth_failure(self, relay):
        """Mock 401 → raises WhipAuthError."""
        # POST returns 401
        # assert raises WhipAuthError with "authentication failed"
        pass

    @pytest.mark.asyncio
    async def test_negotiate_server_error(self, relay):
        """Mock 503 → raises WhipNegotiationError."""
        pass


class TestSourceTypeDetection:
    """Verify detect_source_type() categories feed the right ffmpeg cmd."""

    def test_webcam_index(self):
        from trio_core.source_resolver import detect_source_type
        assert detect_source_type("0") == "webcam"
        assert detect_source_type("1") == "webcam"

    def test_rtsp_url(self):
        from trio_core.source_resolver import detect_source_type
        assert detect_source_type("rtsp://admin:pass@192.168.1.1/stream") == "rtsp"

    def test_file_path(self):
        from trio_core.source_resolver import detect_source_type
        assert detect_source_type("video.mp4") == "file"
        assert detect_source_type("/path/to/video.avi") == "file"
```

---

## Verification Plan

### Automated Tests
```bash
cd /home/haaai/iotex/trio-edge
pip install -e ".[relay,dev]"
pytest tests/test_whip_relay.py -v
```

### Manual E2E Verification
```bash
# 1. Start trio-cortex
cd /home/haaai/iotex/trio-cortex
python api.py

# 2. Relay webcam (in another terminal)
cd /home/haaai/iotex/trio-edge
trio relay http://localhost:8000/api/stream/whip -s 0 -v

# 3. Verify session exists
curl -s http://localhost:8000/api/stream/health | python -m json.tool

# 4. Test RTSP passthrough
trio relay http://localhost:8000/api/stream/whip -s "rtsp://admin:pass@camera/stream" -v

# 5. Test file source
trio relay http://localhost:8000/api/stream/whip -s ~/videos/test.mp4 -v

# 6. Test error cases
trio relay http://localhost:8000/api/stream/whip -s "rtsp://invalid" -v  # bad RTSP
trio relay http://bad-host:9999/whip -s 0 -v                            # bad WHIP URL
# Press Ctrl+C → verify DELETE sent, clean exit
```
