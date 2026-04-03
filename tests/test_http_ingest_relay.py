from __future__ import annotations

import importlib
import sys
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.cli import app

runner = CliRunner()


def _relay_module():
    sys.modules.pop("trio_core.http_ingest_relay", None)
    return importlib.import_module("trio_core.http_ingest_relay")


def test_relay_command_help_mentions_cloud_http_ingest():
    result = runner.invoke(app, ["relay", "--help"])

    assert result.exit_code == 0
    assert "--cloud" in result.output
    assert "--camera-id" in result.output
    assert "HTTP MPEG-TS" in result.output


def test_relay_invalid_resolution_returns_error():
    result = runner.invoke(
        app,
        ["relay", "--cloud", "https://api.trio.ai", "--resolution", "bad"],
    )

    assert result.exit_code == 1
    assert "Invalid resolution format" in result.output


def test_derive_camera_id_is_deterministic(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay.uuid, "getnode", lambda: 0xAABBCCDDEEFF)

    cam_a = relay.derive_camera_id("rtsp://camera/stream")
    cam_b = relay.derive_camera_id("rtsp://camera/stream")
    cam_c = relay.derive_camera_id("video.mp4")

    assert cam_a == cam_b
    assert cam_a != cam_c
    assert len(cam_a) == 36


def test_build_ffmpeg_cmd_rtsp_uses_mpegts_copy(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "rtsp")

    with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://camera/stream"):
        cmd = relay.HttpIngestRelay(
            source="rtsp://camera/stream",
            cloud_url="https://api.trio.ai",
            bearer_token="token",
        )._build_ffmpeg_cmd()

    assert "-f" in cmd
    assert "mpegts" in cmd
    assert "copy" in cmd
    assert "h264_mp4toannexb" not in " ".join(cmd)


def test_build_ffmpeg_cmd_webcam_macos_uses_mpegts(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "webcam")

    with patch("platform.system", return_value="Darwin"):
        cmd = relay.HttpIngestRelay(
            source="0",
            cloud_url="https://api.trio.ai",
            bearer_token="token",
            framerate=30,
        )._build_ffmpeg_cmd()

    assert "avfoundation" in cmd
    assert "libx264" in cmd
    assert "mpegts" in cmd


class _FakeResponse:
    def __init__(self, status_code: int, json_body: dict | None = None, text: str = "") -> None:
        self.status_code = status_code
        self._json_body = json_body or {}
        self.text = text

    def json(self) -> dict:
        return self._json_body


class _FakeStreamContext:
    def __init__(self, response: _FakeResponse, calls: list[dict[str, object]]) -> None:
        self._response = response
        self._calls = calls

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


class _FakeAsyncClient:
    def __init__(self, responses: list[_FakeResponse], calls: list[dict[str, object]]) -> None:
        self._responses = responses
        self._calls = calls

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, **kwargs):
        self._calls.append({"method": "POST", "url": url, **kwargs})
        return self._responses.pop(0)

    def stream(self, method: str, url: str, **kwargs):
        self._calls.append({"method": method, "url": url, **kwargs})
        return _FakeStreamContext(self._responses.pop(0), self._calls)


@pytest.mark.asyncio
async def test_register_camera_posts_explicit_id_and_metadata(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    calls: list[dict[str, object]] = []
    responses = [_FakeResponse(201, {"id": "cam-123"})]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )

    client = relay.HttpIngestRelay(
        source="rtsp://camera/stream",
        cloud_url="https://api.trio.ai/",
        bearer_token="token-123",
        camera_id="cam-123",
    )

    async with relay.httpx.AsyncClient() as http_client:
        returned = await client._register_camera(http_client)

    assert returned == "cam-123"
    assert calls[0]["url"] == "https://api.trio.ai/api/cameras"
    assert calls[0]["headers"]["Authorization"] == "Bearer token-123"
    assert calls[0]["json"]["id"] == "cam-123"
    assert calls[0]["json"]["metadata"]["ingest_transport"] == "http_mpegts"
    assert calls[0]["json"]["metadata"]["managed_by"] == "trio-edge"


@pytest.mark.asyncio
async def test_register_camera_accepts_server_generated_id(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    calls: list[dict[str, object]] = []
    responses = [_FakeResponse(201, {"id": "server-generated"})]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://api.trio.ai",
        bearer_token="token-123",
        camera_id="synthetic-id",
    )

    async with relay.httpx.AsyncClient() as http_client:
        returned = await client._register_camera(http_client)

    assert returned == "server-generated"


@pytest.mark.asyncio
async def test_run_uploads_video_mp2t_to_server_returned_ingest_endpoint(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    calls: list[dict[str, object]] = []
    responses = [
        _FakeResponse(201, {"id": "server-generated"}),
        _FakeResponse(204, {}),
    ]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "file")

    fake_stdout = type("Stdout", (), {})()
    fake_stdout.read = AsyncMock(side_effect=[b"\x47" * 188, b""])
    fake_stderr = type("Stderr", (), {})()
    fake_stderr.read = AsyncMock(return_value=b"")
    fake_process = type("Process", (), {})()
    fake_process.stdout = fake_stdout
    fake_process.stderr = fake_stderr
    fake_process.returncode = 0
    fake_process.wait = AsyncMock(return_value=0)
    fake_process.terminate = lambda: None
    fake_process.kill = lambda: None

    monkeypatch.setattr(
        relay.asyncio,
        "create_subprocess_exec",
        AsyncMock(return_value=fake_process),
    )

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://api.trio.ai",
        bearer_token="token-123",
        camera_id="preferred-id",
    )

    await client.run()

    ingest_call = calls[1]
    assert ingest_call["url"] == "https://api.trio.ai/api/stream/ingest/server-generated"
    assert ingest_call["headers"]["Authorization"] == "Bearer token-123"
    assert ingest_call["headers"]["Content-Type"] == "video/mp2t"


def test_relay_cli_constructs_http_ingest_relay(monkeypatch: pytest.MonkeyPatch):
    import trio_core.cli as cli

    captured: dict[str, object] = {}

    class FakeRelay:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run(self) -> None:
            captured["run_called"] = True

        async def teardown(self) -> None:
            captured["teardown_called"] = True

    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli, "_setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "HttpIngestRelay", FakeRelay, raising=False)

    result = runner.invoke(
        app,
        [
            "relay",
            "--cloud",
            "https://api.trio.ai",
            "--camera",
            "rtsp://admin:pass@192.168.1.10/stream",
            "--token",
            "token-123",
            "--camera-id",
            "cam-123",
        ],
    )

    assert result.exit_code == 0
    assert captured["cloud_url"] == "https://api.trio.ai"
    assert captured["source"] == "rtsp://admin:pass@192.168.1.10/stream"
    assert captured["camera_id"] == "cam-123"
    assert captured["bearer_token"] == "token-123"
    assert captured["run_called"] is True
    assert captured["teardown_called"] is True
    assert "HTTP MPEG-TS" in result.output
