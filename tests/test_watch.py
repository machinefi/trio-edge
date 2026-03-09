"""Tests for /v1/watch API — watch models, SSE helpers, and endpoint routing."""

import json
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from trio_core.api.models import (
    WatchCondition,
    WatchConditionResult,
    WatchInfo,
    WatchMetrics,
    WatchRequest,
)
from trio_core.api.server import (
    _detect_triggered,
    _parse_resolution,
    _sse_event,
    _strip_think_tags,
    _WatchState,
    _watches,
)
from trio_core.engine import InferenceMetrics, VideoResult


@asynccontextmanager
async def _noop_lifespan(app):
    yield


@pytest.fixture
def client():
    """Create test client with mocked engine (no model loading)."""
    from trio_core.api.server import create_app
    import trio_core.api.server as server_mod

    app = create_app()
    mock_engine = MagicMock()
    mock_engine._loaded = True
    mock_engine.config.model = "test-model"
    mock_engine.config.max_tokens = 512
    mock_engine.config.temperature = 0.0
    mock_engine.config.top_p = 1.0
    mock_engine.config.video_fps = 2.0
    mock_engine.config.video_max_frames = 128
    mock_engine.config.dedup_enabled = True
    mock_engine.config.dedup_threshold = 0.95
    mock_engine.config.motion_enabled = False
    mock_engine.health.return_value = {
        "status": "ok", "model": "test-model", "loaded": True, "config": {},
    }
    mock_engine.analyze_frame.return_value = VideoResult(
        text="No, the area is clear.",
        metrics=InferenceMetrics(latency_ms=100.0, tokens_per_sec=50.0),
    )

    server_mod._engine = mock_engine
    app.router.lifespan_context = _noop_lifespan

    with TestClient(app) as c:
        yield c

    server_mod._engine = None
    _watches.clear()


# ── Pydantic Model Tests ──────────────────────────────────────────────────


class TestWatchModels:
    def test_watch_request_defaults(self):
        req = WatchRequest(
            source="rtsp://test:554/stream",
            conditions=[WatchCondition(id="person", question="Is there a person?")],
        )
        assert req.fps == 1.0
        assert req.stream is True
        assert req.resolution == "672x448"
        assert len(req.conditions) == 1

    def test_watch_request_custom_resolution(self):
        req = WatchRequest(
            source="rtsp://test",
            conditions=[WatchCondition(id="p", question="?")],
            resolution="1280x720",
        )
        assert req.resolution == "1280x720"

    def test_watch_condition_result(self):
        r = WatchConditionResult(id="person", triggered=True, answer="Yes, a person is there.")
        assert r.triggered is True
        assert r.id == "person"

    def test_watch_metrics(self):
        m = WatchMetrics(latency_ms=242, tok_s=73.4, frames_analyzed=4)
        assert m.latency_ms == 242
        d = m.model_dump()
        assert d["tok_s"] == 73.4

    def test_watch_info(self):
        info = WatchInfo(
            watch_id="w_abc123",
            source="rtsp://test",
            state="running",
            conditions=[WatchCondition(id="person", question="Is there a person?")],
            uptime_s=3600,
            checks=100,
            alerts=5,
        )
        assert info.watch_id == "w_abc123"
        assert info.state == "running"


# ── Helper Function Tests ─────────────────────────────────────────────────


class TestStripThinkTags:
    def test_no_think_tags(self):
        assert _strip_think_tags("Yes, there is a person.") == "Yes, there is a person."

    def test_think_tags_with_content(self):
        text = "<think>The user is asking about a person. Let me check.</think> YES, there is a person."
        assert _strip_think_tags(text) == "YES, there is a person."

    def test_multiline_think(self):
        text = "<think>\nI see someone at the door.\nThey appear to be a delivery person.\n</think>\nYes, a person is at the door."
        result = _strip_think_tags(text)
        assert result.startswith("Yes")
        assert "<think>" not in result

    def test_orphan_closing_tag(self):
        text = "</think> YES."
        assert _strip_think_tags(text) == "YES."

    def test_empty_think(self):
        text = "<think></think>No."
        assert _strip_think_tags(text) == "No."

    def test_think_with_detect_triggered(self):
        """End-to-end: think tags + triggered detection."""
        text = "<think>reasoning here</think> Yes, there is a person at the door."
        cleaned = _strip_think_tags(text)
        assert _detect_triggered(cleaned) is True

    def test_think_with_no_answer(self):
        text = "<think>I see nothing unusual</think> No, the porch is empty."
        cleaned = _strip_think_tags(text)
        assert _detect_triggered(cleaned) is False


class TestSSEEvent:
    def test_format(self):
        event = _sse_event("status", {"watch_id": "w_123", "state": "connecting"})
        assert event.startswith("event: status\n")
        assert "data: " in event
        assert event.endswith("\n\n")
        data = json.loads(event.split("data: ")[1].strip())
        assert data["watch_id"] == "w_123"

    def test_alert_event(self):
        event = _sse_event("alert", {"watch_id": "w_x", "triggered": True})
        assert "event: alert\n" in event
        data = json.loads(event.split("data: ")[1].strip())
        assert data["triggered"] is True


class TestWatchState:
    def test_defaults(self):
        ws = _WatchState(
            watch_id="w_test",
            source="rtsp://test",
            conditions=[],
            fps=1.0,
        )
        assert ws.state == "connecting"
        assert ws.checks == 0
        assert ws.alerts == 0
        assert ws.error is None


# ── Endpoint Routing Tests ────────────────────────────────────────────────


class TestWatchEndpoints:
    def test_get_watches_empty(self, client):
        resp = client.get("/v1/watch")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_delete_watch_not_found(self, client):
        resp = client.delete("/v1/watch/w_nonexistent")
        assert resp.status_code == 404

    def test_delete_watch_existing(self, client):
        """Insert a fake watch state and delete it."""
        import asyncio
        import trio_core.api.server as server_mod

        ws = _WatchState(
            watch_id="w_test123",
            source="rtsp://test",
            conditions=[WatchCondition(id="p", question="?")],
            fps=1.0,
            started_at=1000.0,
            checks=42,
            alerts=3,
            stop_event=asyncio.Event(),
        )
        server_mod._watches["w_test123"] = ws

        resp = client.delete("/v1/watch/w_test123")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "stopped"
        assert data["total_checks"] == 42
        assert data["total_alerts"] == 3
        assert "w_test123" not in server_mod._watches

    def test_get_watches_with_active(self, client):
        """GET /v1/watch lists active watches."""
        import asyncio
        import time
        import trio_core.api.server as server_mod

        ws = _WatchState(
            watch_id="w_list1",
            source="rtsp://cam1",
            conditions=[WatchCondition(id="person", question="Is there a person?")],
            fps=1.0,
            state="running",
            started_at=time.time() - 60,
            checks=10,
            alerts=1,
            stop_event=asyncio.Event(),
        )
        server_mod._watches["w_list1"] = ws

        resp = client.get("/v1/watch")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["watch_id"] == "w_list1"
        assert data[0]["state"] == "running"
        assert data[0]["checks"] == 10
        assert data[0]["uptime_s"] >= 59

        # Cleanup
        server_mod._watches.pop("w_list1", None)

    def test_post_watch_returns_sse(self, client):
        """POST /v1/watch should return text/event-stream with status events.

        We mock ensure_rtsp_url + Popen so no real RTSP connection is made.
        The ffmpeg mock returns 2 frames then EOF.
        """
        import trio_core.api.server as server_mod

        frame_w, frame_h = 672, 448
        frame_bytes = frame_w * frame_h * 3
        # Create 2 synthetic frames (different content to pass motion gate)
        frame1 = np.zeros(frame_bytes, dtype=np.uint8)
        frame2 = np.full(frame_bytes, 128, dtype=np.uint8)

        mock_stdout = MagicMock()
        mock_stdout.read = MagicMock(side_effect=[
            frame1.tobytes(),
            frame2.tobytes(),
            b"",  # EOF
        ])
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("subprocess.Popen", return_value=mock_proc):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test:554/stream",
                "conditions": [
                    {"id": "person", "question": "Is there a person?"},
                ],
                "fps": 1.0,
            })

        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        # Parse SSE events
        events = _parse_sse(resp.text)
        assert len(events) >= 2  # at least status:connecting + status:running

        # First event should be status:connecting
        assert events[0]["type"] == "status"
        assert events[0]["data"]["state"] == "connecting"

        # Second event should be status:running
        assert events[1]["type"] == "status"
        assert events[1]["data"]["state"] == "running"
        assert "model" in events[1]["data"]

    def test_post_watch_with_triggered(self, client):
        """POST /v1/watch emits alert event when triggered."""
        import trio_core.api.server as server_mod

        # Mock engine to return "Yes" for triggered
        server_mod._engine.analyze_frame.return_value = VideoResult(
            text="Yes, a person is at the door.",
            metrics=InferenceMetrics(latency_ms=200.0, tokens_per_sec=60.0),
        )

        frame_w, frame_h = 672, 448
        frame_bytes = frame_w * frame_h * 3
        frame1 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)

        mock_stdout = MagicMock()
        mock_stdout.read = MagicMock(side_effect=[
            frame1.tobytes(),
            b"",  # EOF
        ])
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("subprocess.Popen", return_value=mock_proc):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test:554/stream",
                "conditions": [
                    {"id": "person", "question": "Is there a person?"},
                ],
            })

        events = _parse_sse(resp.text)

        # Should have an alert event
        alert_events = [e for e in events if e["type"] == "alert"]
        assert len(alert_events) >= 1
        alert = alert_events[0]["data"]
        assert alert["conditions"][0]["triggered"] is True
        assert "frame_b64" in alert
        assert "ts" in alert
        assert alert["metrics"]["latency_ms"] >= 0

    def test_post_watch_think_tags_stripped(self, client):
        """Verify <think> tags are stripped before triggered detection."""
        import trio_core.api.server as server_mod

        server_mod._engine.analyze_frame.return_value = VideoResult(
            text="<think>I should check carefully</think> Yes, there is a person.",
            metrics=InferenceMetrics(latency_ms=150.0, tokens_per_sec=55.0),
        )

        frame_w, frame_h = 672, 448
        frame_bytes = frame_w * frame_h * 3
        frame1 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)

        mock_stdout = MagicMock()
        mock_stdout.read = MagicMock(side_effect=[frame1.tobytes(), b""])
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("subprocess.Popen", return_value=mock_proc):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test",
                "conditions": [{"id": "person", "question": "Is there a person?"}],
            })

        events = _parse_sse(resp.text)
        alert_events = [e for e in events if e["type"] == "alert"]
        assert len(alert_events) == 1
        # Answer should be clean (no think tags)
        assert "<think>" not in alert_events[0]["data"]["conditions"][0]["answer"]
        assert alert_events[0]["data"]["conditions"][0]["triggered"] is True

    def test_post_watch_multi_conditions(self, client):
        """Multiple conditions per watch — each gets separate inference."""
        import trio_core.api.server as server_mod

        answers = iter([
            VideoResult(text="Yes, a person.", metrics=InferenceMetrics(latency_ms=100.0, tokens_per_sec=50.0)),
            VideoResult(text="No, no package.", metrics=InferenceMetrics(latency_ms=100.0, tokens_per_sec=50.0)),
        ])
        server_mod._engine.analyze_frame.side_effect = lambda *a, **kw: next(answers)

        frame_w, frame_h = 672, 448
        frame_bytes = frame_w * frame_h * 3
        frame1 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)

        mock_stdout = MagicMock()
        mock_stdout.read = MagicMock(side_effect=[frame1.tobytes(), b""])
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("subprocess.Popen", return_value=mock_proc):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test",
                "conditions": [
                    {"id": "person", "question": "Is there a person?"},
                    {"id": "package", "question": "Is there a package?"},
                ],
            })

        events = _parse_sse(resp.text)
        alert_events = [e for e in events if e["type"] == "alert"]
        assert len(alert_events) == 1  # any_triggered = True
        conds = alert_events[0]["data"]["conditions"]
        assert len(conds) == 2
        assert conds[0]["id"] == "person"
        assert conds[0]["triggered"] is True
        assert conds[1]["id"] == "package"
        assert conds[1]["triggered"] is False


# ── Parse Resolution Tests ────────────────────────────────────────────────


class TestParseResolution:
    def test_default(self):
        assert _parse_resolution("672x448") == (672, 448)

    def test_hd(self):
        assert _parse_resolution("1280x720") == (1280, 720)

    def test_invalid_falls_back(self):
        assert _parse_resolution("garbage") == (672, 448)

    def test_empty_falls_back(self):
        assert _parse_resolution("") == (672, 448)

    def test_case_insensitive(self):
        assert _parse_resolution("1280X720") == (1280, 720)

    def test_huge_resolution_clamped(self):
        """Absurdly large resolution is clamped to _MAX_RESOLUTION to prevent OOM."""
        w, h = _parse_resolution("999999x999999")
        assert w <= 3840
        assert h <= 3840

    def test_4k_allowed(self):
        """4K resolution (3840x2160) should be allowed."""
        assert _parse_resolution("3840x2160") == (3840, 2160)

    def test_negative_falls_back(self):
        """Negative dimensions fall back to default."""
        assert _parse_resolution("-1x-1") == (672, 448)

    def test_zero_falls_back(self):
        """Zero dimensions fall back to default."""
        assert _parse_resolution("0x0") == (672, 448)


# ── Custom Resolution Test ───────────────────────────────────────────────


class TestWatchCustomResolution:
    def test_custom_resolution_used_in_ffmpeg(self, client):
        """POST /v1/watch with custom resolution passes it to ffmpeg scale filter."""
        import trio_core.api.server as server_mod

        frame_w, frame_h = 1280, 720
        frame_bytes = frame_w * frame_h * 3
        frame1 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)

        mock_stdout = MagicMock()
        mock_stdout.read = MagicMock(side_effect=[frame1.tobytes(), b""])
        mock_proc = MagicMock()
        mock_proc.stdout = mock_stdout
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()

        popen_calls = []
        original_start_ffmpeg = None

        def capture_start_ffmpeg(url, fps, fw, fh):
            popen_calls.append((fw, fh))
            return mock_proc

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("trio_core.api.server._start_ffmpeg", side_effect=capture_start_ffmpeg):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test",
                "conditions": [{"id": "p", "question": "?"}],
                "resolution": "1280x720",
            })

        assert resp.status_code == 200
        events = _parse_sse(resp.text)

        # Check ffmpeg was called with custom resolution
        assert len(popen_calls) >= 1
        assert popen_calls[0] == (1280, 720)

        # Running status should report custom resolution
        running = [e for e in events if e["type"] == "status" and e["data"].get("state") == "running"]
        assert len(running) >= 1
        assert running[0]["data"]["resolution"] == "1280x720"


# ── Reconnect Tests ──────────────────────────────────────────────────────


class TestWatchReconnect:
    def test_reconnect_on_stream_loss(self, client):
        """Watch auto-reconnects when RTSP stream drops."""
        import trio_core.api.server as server_mod

        frame_w, frame_h = 672, 448
        frame_bytes = frame_w * frame_h * 3
        frame1 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)
        frame2 = np.random.randint(0, 255, frame_bytes, dtype=np.uint8)

        # First proc: 1 frame then EOF (simulating stream drop)
        mock_stdout1 = MagicMock()
        mock_stdout1.read = MagicMock(side_effect=[frame1.tobytes(), b""])
        mock_proc1 = MagicMock()
        mock_proc1.stdout = mock_stdout1
        mock_proc1.poll.return_value = None
        mock_proc1.terminate = MagicMock()
        mock_proc1.wait = MagicMock()

        # Second proc (reconnected): 1 frame then EOF
        mock_stdout2 = MagicMock()
        mock_stdout2.read = MagicMock(side_effect=[frame2.tobytes(), b""])
        mock_proc2 = MagicMock()
        mock_proc2.stdout = mock_stdout2
        mock_proc2.poll.return_value = None
        mock_proc2.terminate = MagicMock()
        mock_proc2.wait = MagicMock()

        start_calls = [0]

        def mock_start_ffmpeg(url, fps, fw, fh):
            start_calls[0] += 1
            if start_calls[0] == 1:
                return mock_proc1
            return mock_proc2

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("trio_core.api.server._start_ffmpeg", side_effect=mock_start_ffmpeg), \
             patch("trio_core.api.server._WATCH_RECONNECT_DELAY", 0):  # no sleep in test

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test",
                "conditions": [{"id": "person", "question": "Is there a person?"}],
            })

        events = _parse_sse(resp.text)

        # Should see reconnecting status
        reconnecting = [e for e in events if e["type"] == "status"
                        and e["data"].get("state") == "reconnecting"]
        assert len(reconnecting) >= 1
        assert reconnecting[0]["data"]["attempt"] == 1

        # Should see running again after reconnect
        running_events = [e for e in events if e["type"] == "status"
                         and e["data"].get("state") == "running"]
        assert len(running_events) >= 2  # initial + after reconnect

        # Should have processed frames from both connections
        assert start_calls[0] >= 2

    def test_reconnect_exhausted(self, client):
        """Watch emits error after max reconnect attempts."""
        frame_w, frame_h = 672, 448

        # All procs immediately EOF
        def mock_start_ffmpeg(url, fps, fw, fh):
            mock_stdout = MagicMock()
            mock_stdout.read = MagicMock(return_value=b"")
            mock_proc = MagicMock()
            mock_proc.stdout = mock_stdout
            mock_proc.poll.return_value = None
            mock_proc.terminate = MagicMock()
            mock_proc.wait = MagicMock()
            return mock_proc

        with patch("trio_core._rtsp_proxy.ensure_rtsp_url", return_value="rtsp://test"), \
             patch("trio_core.api.server._start_ffmpeg", side_effect=mock_start_ffmpeg), \
             patch("trio_core.api.server._WATCH_RECONNECT_DELAY", 0), \
             patch("trio_core.api.server._WATCH_MAX_RECONNECTS", 2):

            resp = client.post("/v1/watch", json={
                "source": "rtsp://test",
                "conditions": [{"id": "p", "question": "?"}],
            })

        events = _parse_sse(resp.text)

        # Should see error after exhausting retries
        errors = [e for e in events if e["type"] == "error"]
        assert len(errors) >= 1
        assert "reconnect" in errors[0]["data"]["error"].lower()


# ── Kill ffmpeg cleanup ──────────────────────────────────────────────────


class TestKillFfmpeg:
    def test_closes_pipes(self):
        """_kill_ffmpeg closes stdout and stderr pipes to prevent FD leaks."""
        from trio_core.api.server import _kill_ffmpeg

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        _kill_ffmpeg(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.stdout.close.assert_called_once()
        mock_proc.stderr.close.assert_called_once()

    def test_handles_already_dead_proc(self):
        """_kill_ffmpeg handles processes that already exited."""
        from trio_core.api.server import _kill_ffmpeg

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0  # already dead
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        _kill_ffmpeg(mock_proc)

        mock_proc.terminate.assert_not_called()
        mock_proc.stdout.close.assert_called_once()
        mock_proc.stderr.close.assert_called_once()

    def test_kills_on_timeout(self):
        """_kill_ffmpeg escalates to kill if terminate times out."""
        import subprocess
        from trio_core.api.server import _kill_ffmpeg

        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_proc.terminate = MagicMock()
        mock_proc.wait = MagicMock(side_effect=subprocess.TimeoutExpired("ffmpeg", 5))
        mock_proc.kill = MagicMock()
        mock_proc.stdout = MagicMock()
        mock_proc.stderr = MagicMock()

        _kill_ffmpeg(mock_proc)

        mock_proc.terminate.assert_called_once()
        mock_proc.kill.assert_called_once()


# ── Analyze Frame Think Tag Stripping ─────────────────────────────────────


class TestAnalyzeFrameThinkTags:
    def test_analyze_frame_strips_think_tags(self, client):
        """POST /analyze-frame strips <think> tags from answer."""
        import base64
        import io
        import trio_core.api.server as server_mod
        from PIL import Image

        server_mod._engine.analyze_video.return_value = VideoResult(
            text="<think>Let me check if there's a person</think> Yes, there is a person.",
            metrics=InferenceMetrics(latency_ms=100.0),
        )

        img = Image.new("RGB", (8, 8), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "Is there a person?",
        })
        data = resp.json()
        assert "<think>" not in data["answer"]
        assert data["answer"] == "Yes, there is a person."
        assert data["triggered"] is True

    def test_analyze_frame_think_no_triggered(self, client):
        """POST /analyze-frame: think tags stripped, negative answer detected."""
        import base64
        import io
        import trio_core.api.server as server_mod
        from PIL import Image

        server_mod._engine.analyze_video.return_value = VideoResult(
            text="<think>Checking the scene...</think> No, the area is empty.",
            metrics=InferenceMetrics(latency_ms=80.0),
        )

        img = Image.new("RGB", (8, 8), (0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "Is there a person?",
        })
        data = resp.json()
        assert data["answer"] == "No, the area is empty."
        assert data["triggered"] is False


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_sse(text: str) -> list[dict]:
    """Parse SSE text into list of {type, data} dicts."""
    events = []
    current_type = None
    for line in text.strip().split("\n"):
        if line.startswith("event: "):
            current_type = line[7:].strip()
        elif line.startswith("data: "):
            raw = line[6:].strip()
            try:
                data = json.loads(raw)
                events.append({"type": current_type, "data": data})
            except json.JSONDecodeError:
                pass  # skip [DONE] etc.
    return events
