from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.cli import app

runner = CliRunner()


def _install_relay_stubs(monkeypatch: pytest.MonkeyPatch):
    av_module = types.ModuleType("av")

    class Packet:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self.pts = None
            self.time_base = None

        def __bytes__(self) -> bytes:
            return self._data

    av_module.Packet = Packet
    monkeypatch.setitem(sys.modules, "av", av_module)

    aiohttp_module = types.ModuleType("aiohttp")

    class ClientTimeout:
        def __init__(self, total: float | None = None) -> None:
            self.total = total

    class ClientSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

    aiohttp_module.ClientTimeout = ClientTimeout
    aiohttp_module.ClientSession = ClientSession
    monkeypatch.setitem(sys.modules, "aiohttp", aiohttp_module)

    aiortc_module = types.ModuleType("aiortc")

    class RTCPeerConnection:
        pass

    class RTCSessionDescription:
        def __init__(self, sdp: str, type: str) -> None:
            self.sdp = sdp
            self.type = type

    aiortc_module.RTCPeerConnection = RTCPeerConnection
    aiortc_module.RTCSessionDescription = RTCSessionDescription
    monkeypatch.setitem(sys.modules, "aiortc", aiortc_module)

    mediastreams_module = types.ModuleType("aiortc.mediastreams")

    class MediaStreamError(Exception):
        pass

    class MediaStreamTrack:
        kind = ""

        def __init__(self) -> None:
            self.readyState = "live"

        def stop(self) -> None:
            self.readyState = "ended"

    mediastreams_module.MediaStreamError = MediaStreamError
    mediastreams_module.MediaStreamTrack = MediaStreamTrack
    monkeypatch.setitem(sys.modules, "aiortc.mediastreams", mediastreams_module)

    monkeypatch.delitem(sys.modules, "trio_core.whip_relay", raising=False)
    try:
        return importlib.import_module("trio_core.whip_relay")
    except ModuleNotFoundError as exc:
        pytest.fail(f"trio_core.whip_relay should exist: {exc}")


def test_relay_command_help_registered():
    result = runner.invoke(app, ["relay", "--help"])

    assert result.exit_code == 0
    assert "WHIP ingest endpoint URL" in result.output
    assert "Video source" in result.output


def test_relay_invalid_resolution_returns_error():
    result = runner.invoke(
        app,
        ["relay", "http://cortex:8000/api/stream/whip", "--resolution", "bad"],
    )

    assert result.exit_code == 1
    assert "Invalid resolution format" in result.output


def test_access_unit_buffer_splits_aud_delimited_frames(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)
    buffer = relay._AnnexBAccessUnitBuffer()

    frame1 = (
        b"\x00\x00\x00\x01\x09\xf0"
        b"\x00\x00\x00\x01\x67\x64\x00\x1f"
        b"\x00\x00\x00\x01\x68\xee\x3c\x80"
        b"\x00\x00\x00\x01\x65\x88\x84"
    )
    frame2 = b"\x00\x00\x00\x01\x09\xf0\x00\x00\x00\x01\x41\x9a\x22"

    buffer.push(frame1 + frame2)

    assert buffer.pop() == frame1
    assert buffer.pop() is None

    buffer.close()
    assert buffer.pop() == frame2


def test_build_cmd_webcam_linux(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "webcam")

    track = relay.H264FfmpegTrack("0", framerate=30, resolution=(1280, 720))
    with patch("platform.system", return_value="Linux"):
        cmd = track._build_ffmpeg_cmd()

    assert "v4l2" in cmd
    assert "/dev/video0" in cmd
    assert "libx264" in cmd
    assert "1280x720" in cmd


def test_build_cmd_webcam_macos_sets_supported_pixel_format(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "webcam")

    track = relay.H264FfmpegTrack("0", framerate=30)
    with patch("platform.system", return_value="Darwin"):
        cmd = track._build_ffmpeg_cmd()

    assert "avfoundation" in cmd
    assert "0:none" in cmd
    assert "-pixel_format" in cmd
    assert "nv12" in cmd


def test_build_cmd_rtsp_uses_passthrough(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "rtsp")

    with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://camera/stream"):
        cmd = relay.H264FfmpegTrack("rtsp://camera/stream")._build_ffmpeg_cmd()

    assert "copy" in cmd
    assert "libx264" not in cmd
    assert "h264_mp4toannexb,h264_metadata=aud=insert" in cmd


def test_probe_rtsp_codec_detects_h264(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)

    with patch("shutil.which", return_value="/usr/bin/ffprobe"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="h264\n", stderr="", returncode=0)
        assert relay._probe_rtsp_codec("rtsp://camera/stream") == "h264"


def test_probe_rtsp_codec_empty_output_raises(monkeypatch: pytest.MonkeyPatch):
    relay = _install_relay_stubs(monkeypatch)

    with patch("shutil.which", return_value="/usr/bin/ffprobe"), patch("subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(stdout="", stderr="connection refused", returncode=1)
        with pytest.raises(relay.SourceError, match="Cannot probe RTSP stream"):
            relay._probe_rtsp_codec("rtsp://camera/stream")


class _FakeResponse:
    def __init__(self, status: int, text: str, headers: dict[str, str] | None = None) -> None:
        self.status = status
        self._text = text
        self.headers = headers or {}

    async def text(self) -> str:
        return self._text


class _FakeRequestContext:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeClientSession:
    def __init__(self, responses: list[_FakeResponse], calls: list[dict[str, object]]) -> None:
        self._responses = responses
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def post(self, url: str, **kwargs):
        self._calls.append({"url": url, **kwargs})
        return _FakeRequestContext(self._responses.pop(0))

    def delete(self, url: str, **kwargs):
        self._calls.append({"url": url, **kwargs})
        return _FakeRequestContext(self._responses.pop(0))


@pytest.mark.asyncio
async def test_negotiate_success_sets_remote_description(monkeypatch: pytest.MonkeyPatch):
    relay_module = _install_relay_stubs(monkeypatch)
    calls: list[dict[str, object]] = []
    responses = [
        _FakeResponse(
            201,
            "v=0\r\no=- 1 1 IN IP4 0.0.0.0\r\n",
            headers={"Location": "/api/stream/whip/session-abc123"},
        )
    ]
    monkeypatch.setattr(
        relay_module.aiohttp,
        "ClientSession",
        lambda: _FakeClientSession(list(responses), calls),
    )

    relay = relay_module.WhipRelay(source="0", whip_url="http://test:8000/api/stream/whip")
    relay._pc = MagicMock()
    relay._pc.localDescription = MagicMock(sdp="v=0\r\n")
    relay._pc.setRemoteDescription = AsyncMock()

    await relay._negotiate()

    assert calls[0]["url"] == "http://test:8000/api/stream/whip"
    assert calls[0]["data"] == "v=0\r\n"
    assert relay._session_url == "http://test:8000/api/stream/whip/session-abc123"
    relay._pc.setRemoteDescription.assert_awaited_once()
    description = relay._pc.setRemoteDescription.await_args.args[0]
    assert description.type == "answer"
    assert description.sdp.startswith("v=0")


@pytest.mark.asyncio
async def test_negotiate_auth_failure_raises(monkeypatch: pytest.MonkeyPatch):
    relay_module = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(
        relay_module.aiohttp,
        "ClientSession",
        lambda: _FakeClientSession([_FakeResponse(401, "unauthorized")], []),
    )

    relay = relay_module.WhipRelay(source="0", whip_url="http://test:8000/api/stream/whip")
    relay._pc = MagicMock()
    relay._pc.localDescription = MagicMock(sdp="v=0\r\n")

    with pytest.raises(relay_module.WhipAuthError, match="authentication failed"):
        await relay._negotiate()


@pytest.mark.asyncio
async def test_negotiate_server_error_raises(monkeypatch: pytest.MonkeyPatch):
    relay_module = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(
        relay_module.aiohttp,
        "ClientSession",
        lambda: _FakeClientSession([_FakeResponse(503, "downstream unavailable")], []),
    )

    relay = relay_module.WhipRelay(source="0", whip_url="http://test:8000/api/stream/whip")
    relay._pc = MagicMock()
    relay._pc.localDescription = MagicMock(sdp="v=0\r\n")

    with pytest.raises(relay_module.WhipNegotiationError, match="HTTP 503"):
        await relay._negotiate()


@pytest.mark.asyncio
async def test_attach_analysis_posts_to_session_endpoint(monkeypatch: pytest.MonkeyPatch):
    relay_module = _install_relay_stubs(monkeypatch)
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        relay_module.aiohttp,
        "ClientSession",
        lambda: _FakeClientSession([_FakeResponse(200, '{"attached": true}')], calls),
    )

    relay = relay_module.WhipRelay(
        source="0",
        whip_url="http://test:8000/api/stream/whip",
        bearer_token="secret-token",
    )
    relay._session_url = "http://test:8000/api/stream/whip/session-abc123"

    await relay._attach_analysis()

    assert calls[0]["url"] == "http://test:8000/api/stream/sessions/session-abc123/analyze"
    assert calls[0]["json"] == {}
    assert calls[0]["headers"]["Authorization"] == "Bearer secret-token"


@pytest.mark.asyncio
async def test_run_attaches_analysis_after_connection(monkeypatch: pytest.MonkeyPatch):
    relay_module = _install_relay_stubs(monkeypatch)
    monkeypatch.setattr(relay_module, "detect_source_type", lambda source: "file")

    class FakePeerConnection:
        def __init__(self) -> None:
            self.localDescription = None
            self.iceGatheringState = "complete"
            self.iceConnectionState = "connected"
            self.connectionState = "closed"
            self._handlers: dict[str, object] = {}

        def addTransceiver(self, kind: str, direction: str) -> None:
            return None

        def addTrack(self, track) -> None:
            return None

        async def createOffer(self):
            return types.SimpleNamespace(sdp="v=0\r\n", type="offer")

        async def setLocalDescription(self, offer) -> None:
            self.localDescription = offer

        async def close(self) -> None:
            return None

        def on(self, event: str):
            def register(callback):
                self._handlers[event] = callback
                if event == "iceconnectionstatechange":
                    callback()
                return callback

            return register

    fake_pc = FakePeerConnection()
    monkeypatch.setattr(relay_module, "RTCPeerConnection", lambda: fake_pc)
    monkeypatch.setattr(relay_module, "H264FfmpegTrack", lambda *args, **kwargs: object())

    relay = relay_module.WhipRelay(source="video.mp4", whip_url="http://test:8000/api/stream/whip")
    relay._wait_for_ice_gathering = AsyncMock()

    async def fake_negotiate() -> None:
        relay._session_url = "http://test:8000/api/stream/whip/session-abc123"

    relay._negotiate = AsyncMock(side_effect=fake_negotiate)
    relay._attach_analysis = AsyncMock()

    await relay.run()

    relay._attach_analysis.assert_awaited_once()
