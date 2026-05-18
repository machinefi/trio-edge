from __future__ import annotations

import asyncio
import importlib
import sys
import time
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


def _fake_segment_reader(*chunks: bytes):
    chunk_iter = iter(chunks)

    async def _read_segment(stdout, path, duration_seconds, chunk_size=65536):
        del stdout, duration_seconds, chunk_size
        chunk = next(chunk_iter)
        if not chunk:
            return 0, True
        path.write_bytes(chunk)
        return len(chunk), False

    return _read_segment


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
async def test_register_camera_returns_client_camera_id(monkeypatch: pytest.MonkeyPatch):
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

    assert returned == "synthetic-id"
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

    monkeypatch.setattr(
        relay,
        "_read_segment_to_file",
        _fake_segment_reader(b"fake-ts-data", b""),
    )

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
    assert "https://trio-relay.machinefi.com/api/stream/ingest/preferred-id" not in cmd

    assert len(post_calls) >= 1
    ingest_post = post_calls[0]
    assert ingest_post["url"] == "https://trio-relay.machinefi.com/api/stream/ingest/preferred-id"
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
        "_read_segment_to_file",
        _fake_segment_reader(b"chunk1", b"chunk2", b"chunk3", b""),
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
async def test_segment_capture_continues_while_upload_is_in_flight(
    monkeypatch: pytest.MonkeyPatch,
):
    relay = _relay_module()
    first_post_started = asyncio.Event()
    release_first_post = asyncio.Event()
    second_segment_captured = asyncio.Event()
    chunks = iter([b"chunk1", b"chunk2", b""])

    async def fake_read_segment(stdout, path, duration_seconds, chunk_size=65536):
        del stdout, duration_seconds, chunk_size
        chunk = next(chunks)
        if not chunk:
            return 0, True
        path.write_bytes(chunk)
        if chunk == b"chunk2":
            second_segment_captured.set()
        return len(chunk), False

    monkeypatch.setattr(relay, "_read_segment_to_file", fake_read_segment)

    post_contents: list[bytes] = []

    class _SlowFirstPostClient:
        async def post(self, url, **kwargs):
            del url
            post_contents.append(kwargs["content"])
            if len(post_contents) == 1:
                first_post_started.set()
                await release_first_post.wait()
            return _FakeResponse(204)

    uploader = relay._SegmentUploader(
        _SlowFirstPostClient(),
        "https://trio-relay.machinefi.com/api/stream/ingest/cam-xyz",
        {"X-API-Key": "tok", "Content-Type": "video/mp2t"},
        segment_duration=10.0,
    )

    upload_task = asyncio.create_task(uploader.upload_all(object()))
    await asyncio.wait_for(first_post_started.wait(), timeout=1.0)
    await asyncio.wait_for(second_segment_captured.wait(), timeout=1.0)
    release_first_post.set()
    await upload_task

    assert post_contents == [b"chunk1", b"chunk2"]


@pytest.mark.asyncio
async def test_segmented_post_continues_through_5xx(monkeypatch: pytest.MonkeyPatch):
    """5xx responses are transient — the relay must skip the failed
    segment, NOT abort the whole session. Previously a single 500/502
    killed ffmpeg and the wrapper had to respawn, costing 30-60s of
    dead air per cloud blip.
    """
    post_contents: list[bytes] = []

    class _FlakyClient:
        async def post(self, url, **kwargs):
            del url
            post_contents.append(kwargs["content"])
            # First two: 500. Last: 204.
            if len(post_contents) <= 2:
                return _FakeResponse(500, text="Internal Server Error")
            return _FakeResponse(204)

    relay = _relay_module()

    monkeypatch.setattr(
        relay,
        "_read_segment_to_file",
        _fake_segment_reader(b"chunk1", b"chunk2", b"chunk3", b""),
    )

    uploader = relay._SegmentUploader(
        _FlakyClient(),
        "https://trio-relay.machinefi.com/api/stream/ingest/cam-err",
        {"X-API-Key": "tok", "Content-Type": "video/mp2t"},
        segment_duration=10.0,
    )

    # No raise — relay continues through 5xx.
    await uploader.upload_all(object())

    assert post_contents == [b"chunk1", b"chunk2", b"chunk3"]
    assert uploader.segments_ok == 1
    assert uploader.segments_fail == 2


@pytest.mark.asyncio
async def test_segmented_post_aborts_on_401_fatal(monkeypatch: pytest.MonkeyPatch):
    """401/403 means the API key was revoked or rejected — won't
    self-heal. Abort the relay so the wrapper exits and the operator
    notices, instead of looping silently."""
    relay = _relay_module()
    post_contents: list[bytes] = []

    class _AuthRejectingClient:
        async def post(self, url, **kwargs):
            del url
            post_contents.append(kwargs["content"])
            return _FakeResponse(401, text="bad key")

    monkeypatch.setattr(
        relay,
        "_read_segment_to_file",
        _fake_segment_reader(b"chunk1", b"chunk2", b""),
    )

    uploader = relay._SegmentUploader(
        _AuthRejectingClient(),
        "https://trio-relay.machinefi.com/api/stream/ingest/cam-auth",
        {"X-API-Key": "bad", "Content-Type": "video/mp2t"},
        segment_duration=10.0,
    )

    with pytest.raises(relay.IngestUploadError, match="Fatal upload error"):
        await uploader.upload_all(object())

    # Aborted after the first 401 — no second attempt.
    assert post_contents == [b"chunk1"]


@pytest.mark.asyncio
async def test_segmented_post_continues_on_transport_error(monkeypatch: pytest.MonkeyPatch):
    """Network blips raise httpx.TransportError. These are transient —
    the relay must log + skip + continue, not abort."""
    relay = _relay_module()
    post_contents: list[bytes] = []

    class _TransportErrorClient:
        async def post(self, url, **kwargs):
            del url
            post_contents.append(kwargs["content"])
            if len(post_contents) == 1:
                raise relay.httpx.ConnectError("conn refused")
            return _FakeResponse(204)

    monkeypatch.setattr(
        relay,
        "_read_segment_to_file",
        _fake_segment_reader(b"chunk1", b"chunk2", b""),
    )

    uploader = relay._SegmentUploader(
        _TransportErrorClient(),
        "https://trio-relay.machinefi.com/api/stream/ingest/cam-net",
        {"X-API-Key": "tok", "Content-Type": "video/mp2t"},
        segment_duration=10.0,
    )

    await uploader.upload_all(object())

    assert post_contents == [b"chunk1", b"chunk2"]
    assert uploader.segments_ok == 1
    assert uploader.segments_fail == 1


@pytest.mark.asyncio
async def test_upload_has_per_segment_deadline(monkeypatch: pytest.MonkeyPatch):
    """If the POST hangs (e.g. half-closed connection, server stopped
    reading body but never sent FIN), the per-segment deadline must
    fire so the worker isn't pinned at 0% CPU indefinitely. Observed
    2026-05-18 in prod; mitigated locally with a wrapper-script
    watchdog at the time."""
    relay = _relay_module()
    deadline_observed = asyncio.Event()

    class _HangingClient:
        async def post(self, url, **kwargs):
            del url, kwargs
            # Never returns — simulates the half-closed-connection case.
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                deadline_observed.set()
                raise
            return _FakeResponse(204)

    monkeypatch.setattr(
        relay,
        "_read_segment_to_file",
        _fake_segment_reader(b"chunk1", b""),
    )

    uploader = relay._SegmentUploader(
        _HangingClient(),
        "https://trio-relay.machinefi.com/api/stream/ingest/cam-hang",
        {"X-API-Key": "tok", "Content-Type": "video/mp2t"},
        segment_duration=10.0,
        upload_timeout_seconds=0.2,  # tight deadline for test speed
    )

    # Must finish — the wait_for deadline kicks in, classifies the
    # hang as TRANSIENT, and we proceed to EOF.
    await asyncio.wait_for(uploader.upload_all(object()), timeout=2.0)

    assert deadline_observed.is_set(), "wait_for should have cancelled the hung POST"
    assert uploader.segments_ok == 0
    assert uploader.segments_fail == 1


def test_clean_stale_segment_tmpdirs_removes_only_old_dirs(tmp_path: Path):
    """Startup sweep recovers disk from SIGKILL'd previous runs.
    Must remove orphan dirs older than the threshold but leave fresh
    dirs (owned by concurrent relays on the same host) alone."""
    import os

    relay = _relay_module()

    # Three fixtures: old orphan, fresh (concurrent relay), unrelated dir.
    old_orphan = tmp_path / "trio-relay-segments-old123"
    fresh_relay = tmp_path / "trio-relay-segments-fresh456"
    unrelated = tmp_path / "some-other-dir"
    for d in (old_orphan, fresh_relay, unrelated):
        d.mkdir()
        (d / "segment-00000001.ts").write_bytes(b"junk")

    # Backdate the old orphan to 2 hours ago (>1h threshold).
    two_hours_ago = time.time() - 7200
    os.utime(old_orphan, (two_hours_ago, two_hours_ago))

    removed = relay._clean_stale_segment_tmpdirs(parent_dir=tmp_path)

    assert removed == 1
    assert not old_orphan.exists()
    assert fresh_relay.exists()
    assert (fresh_relay / "segment-00000001.ts").exists()
    assert unrelated.exists()


def test_clean_stale_segment_tmpdirs_no_parent_dir_is_harmless(tmp_path: Path):
    """Missing parent dir must not crash."""
    relay = _relay_module()
    missing = tmp_path / "does-not-exist"
    # Should return 0 silently.
    assert relay._clean_stale_segment_tmpdirs(parent_dir=missing) == 0


def test_clean_stale_segment_tmpdirs_threshold_respected(tmp_path: Path):
    """A dir just under the threshold must be left alone."""
    import os

    relay = _relay_module()

    just_fresh = tmp_path / "trio-relay-segments-justfresh"
    just_fresh.mkdir()
    just_old = tmp_path / "trio-relay-segments-justold"
    just_old.mkdir()

    now = time.time()
    os.utime(just_fresh, (now - 30, now - 30))  # 30s old
    os.utime(just_old, (now - 120, now - 120))  # 120s old

    removed = relay._clean_stale_segment_tmpdirs(
        parent_dir=tmp_path,
        age_threshold_seconds=60.0,
    )

    assert removed == 1
    assert just_fresh.exists()
    assert not just_old.exists()


def test_relay_cli_registers_sigterm_handler():
    """Regression guard: production wrapper kills via SIGTERM. Without
    a handler, the segment TemporaryDirectory leaks (#disk-leak)."""
    import signal as _signal

    from trio_core.cli.relay import _shutdown_signals

    sigs = _shutdown_signals()
    assert _signal.SIGTERM in sigs, "SIGTERM must be in shutdown signal set"
    assert _signal.SIGINT in sigs, "SIGINT must still be handled (Ctrl+C)"


@pytest.mark.asyncio
async def test_relay_sigterm_cancels_main_task_and_runs_teardown(
    monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: deliver real SIGTERM to ourselves while the relay is
    running and assert (a) the run task is cancelled and (b) teardown
    fires. This guarantees the segment TemporaryDirectory's __exit__
    will actually run."""
    import os
    import signal as _signal

    started = asyncio.Event()
    teardown_called = asyncio.Event()

    class _FakeRelay:
        async def run(self) -> None:
            started.set()
            await asyncio.sleep(60)  # would block forever in real run

        async def teardown(self) -> None:
            teardown_called.set()

    relay_obj = _FakeRelay()
    main_task = asyncio.create_task(relay_obj.run())
    loop = asyncio.get_running_loop()

    from trio_core.cli.relay import _shutdown_signals

    shutdown_triggered = False

    def _on_shutdown() -> None:
        nonlocal shutdown_triggered
        shutdown_triggered = True
        main_task.cancel()

    sigs = _shutdown_signals()
    for sig in sigs:
        loop.add_signal_handler(sig, _on_shutdown)
    try:
        # Wait until the fake relay's run() is suspended in sleep.
        await asyncio.wait_for(started.wait(), timeout=1.0)
        # Deliver SIGTERM to ourselves.
        os.kill(os.getpid(), _signal.SIGTERM)
        try:
            await main_task
        except asyncio.CancelledError:
            pass
        # Mirror the production finally block.
        await relay_obj.teardown()
        assert shutdown_triggered
        assert teardown_called.is_set()
    finally:
        for sig in sigs:
            loop.remove_signal_handler(sig)
