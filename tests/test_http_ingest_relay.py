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
    import re

    result = runner.invoke(app, ["relay", "--help"])

    # Strip ANSI escape codes (colors/styling) from the output for robust matching
    clean_output = re.sub(r"\x1b\[[0-9;]*[mG]", "", result.output)

    assert result.exit_code == 0
    assert "--cloud" in clean_output
    assert "--camera-id" in clean_output
    assert "Trio Cloud" in clean_output


def test_relay_invalid_resolution_returns_error():
    result = runner.invoke(
        app,
        ["relay", "--cloud", "https://trio-relay.machinefi.com", "--resolution", "bad"],
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


def test_build_ffmpeg_cmd_rtsp_applies_output_fps(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "rtsp")

    with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://camera/stream"):
        cmd = relay.HttpIngestRelay(
            source="rtsp://camera/stream",
            cloud_url="https://trio-relay.machinefi.com",
            bearer_token="token",
            framerate=15,
        )._build_ffmpeg_cmd()

    assert "-f" in cmd
    assert "mpegts" in cmd
    assert "libx264" in cmd
    assert "-r" in cmd
    assert cmd[cmd.index("-r") + 1] == "15"
    assert "pipe:1" in cmd
    assert "h264_mp4toannexb" not in " ".join(cmd)


def test_build_ffmpeg_cmd_webcam_macos_uses_mpegts(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "webcam")

    with patch("platform.system", return_value="Darwin"):
        cmd = relay.HttpIngestRelay(
            source="0",
            cloud_url="https://trio-relay.machinefi.com",
            bearer_token="token",
            framerate=30,
        )._build_ffmpeg_cmd()

    assert "avfoundation" in cmd
    assert "libx264" in cmd
    assert "mpegts" in cmd
    assert "pipe:1" in cmd


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

    async def get(self, url: str, **kwargs):
        self._calls.append({"method": "GET", "url": url, **kwargs})
        return self._responses.pop(0)

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
    responses = [
        _FakeResponse(201, {"id": "cam-123"}),
    ]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )

    client = relay.HttpIngestRelay(
        source="rtsp://camera/stream",
        cloud_url="https://trio-relay.machinefi.com/",
        bearer_token="token-123",
        camera_id="cam-123",
    )

    async with relay.httpx.AsyncClient() as http_client:
        returned = await client._register_camera(http_client)

    assert returned == "cam-123"
    assert len(calls) == 1
    post_call = calls[0]
    assert post_call["method"] == "POST"
    assert post_call["url"] == "https://trio-relay.machinefi.com/api/cameras"
    assert post_call["headers"]["X-API-Key"] == "token-123"
    assert post_call["json"]["id"] == "cam-123"
    assert post_call["json"]["metadata"]["ingest_transport"] == "http_mpegts"
    assert post_call["json"]["metadata"]["managed_by"] == "trio-edge"


@pytest.mark.asyncio
async def test_register_camera_accepts_server_generated_id(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    calls: list[dict[str, object]] = []
    responses = [
        _FakeResponse(201, {"id": "server-generated"}),
    ]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://trio-relay.machinefi.com",
        bearer_token="token-123",
        camera_id="synthetic-id",
    )

    async with relay.httpx.AsyncClient() as http_client:
        returned = await client._register_camera(http_client)

    assert returned == "server-generated"
    assert len(calls) == 1
    assert calls[0]["method"] == "POST"


@pytest.mark.asyncio
async def test_register_camera_returns_existing_on_200(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()
    calls: list[dict[str, object]] = []
    responses = [
        _FakeResponse(200, {"id": "cam-123"}),
    ]
    monkeypatch.setattr(
        relay.httpx,
        "AsyncClient",
        lambda **kwargs: _FakeAsyncClient(list(responses), calls),
    )

    client = relay.HttpIngestRelay(
        source="rtsp://camera/stream",
        cloud_url="https://trio-relay.machinefi.com/",
        bearer_token="token-123",
        camera_id="cam-123",
    )

    async with relay.httpx.AsyncClient() as http_client:
        returned = await client._register_camera(http_client)

    assert returned == "cam-123"
    assert len(calls) == 1
    assert calls[0]["method"] == "POST"


@pytest.mark.asyncio
async def test_run_launches_ffmpeg_with_pipe_output_and_posts_segment(
    monkeypatch: pytest.MonkeyPatch,
):
    relay = _relay_module()

    fake_stdout = type("Stdout", (), {})()
    fake_stdout.read = AsyncMock(side_effect=[b"fake-ts-data", b""])
    fake_stderr = type("Stderr", (), {})()
    fake_stderr.read = AsyncMock(return_value=b"")
    fake_stderr.readline = AsyncMock(return_value=b"")
    fake_process = type("Process", (), {})()
    fake_process.stdout = fake_stdout
    fake_process.stderr = fake_stderr
    fake_process.returncode = 0
    fake_process.wait = AsyncMock(return_value=0)
    fake_process.terminate = lambda: None
    fake_process.kill = lambda: None

    captured_cmd: list[list[str]] = []

    async def fake_create_subprocess_exec(*args, **kwargs):
        captured_cmd.append(list(args))
        return fake_process

    monkeypatch.setattr(relay.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    monkeypatch.setattr(relay.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "file")

    post_calls: list[dict[str, object]] = []

    class _SegmentClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            post_calls.append({"url": url, **kwargs})
            return _FakeResponse(204)

    reg_client_calls: list[dict[str, object]] = []

    class _RegClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            reg_client_calls.append({"method": "GET", "url": url, **kwargs})
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            reg_client_calls.append({"method": "POST", "url": url, **kwargs})
            return _FakeResponse(201, {"id": "server-generated"})

    client_instances = [_RegClient(), _SegmentClient()]

    def fake_async_client(**kwargs):
        return client_instances.pop(0)

    monkeypatch.setattr(relay.httpx, "AsyncClient", fake_async_client)

    monkeypatch.setattr(relay, "_read_until_timeout", AsyncMock(side_effect=[b"fake-ts-data", b""]))

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://trio-relay.machinefi.com",
        bearer_token="token-123",
        camera_id="preferred-id",
    )

    await client.run()

    assert len(captured_cmd) == 1
    cmd = captured_cmd[0]
    assert "pipe:1" in cmd
    assert "-method" not in cmd
    assert "-headers" not in cmd
    assert "https://trio-relay.machinefi.com/api/stream/ingest/server-generated" not in cmd

    assert len(post_calls) >= 1
    ingest_post = post_calls[0]
    assert (
        ingest_post["url"] == "https://trio-relay.machinefi.com/api/stream/ingest/server-generated"
    )
    assert ingest_post["headers"]["X-API-Key"] == "token-123"
    assert ingest_post["headers"]["Content-Type"] == "video/mp2t"
    assert ingest_post["content"] == b"fake-ts-data"


def test_relay_cli_constructs_http_ingest_relay(monkeypatch: pytest.MonkeyPatch):
    import trio_core.cli.relay as cli_relay

    captured: dict[str, object] = {}

    class FakeRelay:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        async def run(self) -> None:
            captured["run_called"] = True

        async def teardown(self) -> None:
            captured["teardown_called"] = True

    monkeypatch.setattr(cli_relay.shutil, "which", lambda name: "/usr/bin/ffmpeg")
    monkeypatch.setattr(cli_relay, "_setup_logging", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli_relay, "HttpIngestRelay", FakeRelay, raising=False)

    result = runner.invoke(
        app,
        [
            "relay",
            "--cloud",
            "https://trio-relay.machinefi.com",
            "--source",
            "rtsp://admin:pass@192.168.1.10/stream",
            "--token",
            "token-123",
            "--camera-id",
            "cam-123",
        ],
    )

    assert result.exit_code == 0
    assert captured["cloud_url"] == "https://trio-relay.machinefi.com"
    assert captured["source"] == "rtsp://admin:pass@192.168.1.10/stream"
    assert captured["camera_id"] == "cam-123"
    assert captured["bearer_token"] == "token-123"
    assert captured["run_called"] is True
    assert captured["teardown_called"] is True
    assert "HTTP MPEG-TS" in result.output


@pytest.mark.asyncio
async def test_segmented_post_loop_sends_chunks(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()

    fake_stderr = type("Stderr", (), {})()
    fake_stderr.readline = AsyncMock(return_value=b"")
    fake_process = type("Process", (), {})()
    fake_process.stderr = fake_stderr
    fake_process.returncode = 0
    fake_process.wait = AsyncMock(return_value=0)
    fake_process.terminate = lambda: None
    fake_process.kill = lambda: None

    monkeypatch.setattr(relay.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    async def fake_create_subprocess_exec(*args, **kwargs):
        fake_process.stdout = kwargs.get("stdout")
        return fake_process

    monkeypatch.setattr(relay.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "file")

    post_calls: list[dict[str, object]] = []

    class _SegClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            post_calls.append({"url": url, **kwargs})
            return _FakeResponse(204)

    class _RegClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            return _FakeResponse(201, {"id": "cam-xyz"})

    client_instances = [_RegClient(), _SegClient()]

    def fake_async_client(**kwargs):
        return client_instances.pop(0)

    monkeypatch.setattr(relay.httpx, "AsyncClient", fake_async_client)

    monkeypatch.setattr(
        relay,
        "_read_until_timeout",
        AsyncMock(side_effect=[b"chunk1", b"chunk2", b"chunk3", b""]),
    )

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://trio-relay.machinefi.com",
        bearer_token="tok",
        camera_id="cam-xyz",
    )

    await client.run()

    assert len(post_calls) == 3
    for call in post_calls:
        assert call["url"] == "https://trio-relay.machinefi.com/api/stream/ingest/cam-xyz"
        assert call["headers"]["X-API-Key"] == "tok"
        assert call["headers"]["Content-Type"] == "video/mp2t"
    assert post_calls[0]["content"] == b"chunk1"
    assert post_calls[1]["content"] == b"chunk2"
    assert post_calls[2]["content"] == b"chunk3"


@pytest.mark.asyncio
async def test_segmented_post_stops_on_server_error(monkeypatch: pytest.MonkeyPatch):
    relay = _relay_module()

    fake_stderr = type("Stderr", (), {})()
    fake_stderr.readline = AsyncMock(return_value=b"")
    fake_process = type("Process", (), {})()
    fake_process.stderr = fake_stderr
    fake_process.returncode = 0
    fake_process.wait = AsyncMock(return_value=0)
    fake_process.terminate = lambda: None
    fake_process.kill = lambda: None

    monkeypatch.setattr(relay.shutil, "which", lambda _: "/usr/bin/ffmpeg")

    async def fake_create_subprocess_exec(*args, **kwargs):
        fake_process.stdout = kwargs.get("stdout")
        return fake_process

    monkeypatch.setattr(relay.asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr(relay, "detect_source_type", lambda source: "file")

    post_calls: list[dict[str, object]] = []

    class _SegClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            post_calls.append({"url": url, **kwargs})
            return _FakeResponse(500, text="Internal Server Error")

    class _RegClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def get(self, url, **kwargs):
            return _FakeResponse(404)

        async def post(self, url, **kwargs):
            return _FakeResponse(201, {"id": "cam-err"})

    client_instances = [_RegClient(), _SegClient()]

    def fake_async_client(**kwargs):
        return client_instances.pop(0)

    monkeypatch.setattr(relay.httpx, "AsyncClient", fake_async_client)

    monkeypatch.setattr(
        relay,
        "_read_until_timeout",
        AsyncMock(side_effect=[b"chunk1", b"chunk2", b""]),
    )

    client = relay.HttpIngestRelay(
        source="video.mp4",
        cloud_url="https://trio-relay.machinefi.com",
        bearer_token="tok",
        camera_id="cam-err",
    )

    await client.run()

    assert len(post_calls) == 1
    assert post_calls[0]["content"] == b"chunk1"
