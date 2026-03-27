"""Tests for trio_core.api — uses mocked engine."""

import json
from unittest.mock import MagicMock, patch

import pytest
from conftest import make_mock_engine, noop_lifespan
from fastapi import HTTPException
from fastapi.testclient import TestClient

from trio_core.engine import InferenceMetrics, VideoResult


@pytest.fixture
def client():
    """Create test client with mocked engine."""
    import trio_core.api.server as server_mod
    from trio_core.api.server import create_app

    app = create_app()
    mock_engine = make_mock_engine()
    mock_engine.analyze_video.return_value = VideoResult(
        text="Test analysis result",
        metrics=InferenceMetrics(
            frames_input=10,
            frames_after_dedup=6,
            prompt_tokens=100,
            completion_tokens=50,
            latency_ms=500.0,
        ),
    )

    server_mod._engine = mock_engine
    app.router.lifespan_context = noop_lifespan

    with TestClient(app) as c:
        yield c

    server_mod._engine = None


class TestHealthEndpoint:
    def test_health(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestModelsEndpoint:
    def test_list_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert len(data["data"]) == 1


class TestVideoAnalyze:
    def test_analyze(self, client):
        resp = client.post(
            "/v1/video/analyze",
            json={
                "video": "/tmp/test.mp4",
                "prompt": "What is happening?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data
        assert data["text"] == "Test analysis result"
        assert data["metrics"]["frames_input"] == 10


class TestFramesAnalyze:
    def test_upload_frames(self, client):
        """Multipart frame upload."""
        import io

        from PIL import Image

        # Create two 8x8 test images
        files = []
        for color in [(255, 0, 0), (0, 255, 0)]:
            img = Image.new("RGB", (8, 8), color)
            buf = io.BytesIO()
            img.save(buf, format="JPEG")
            buf.seek(0)
            files.append(("frames", ("frame.jpg", buf, "image/jpeg")))

        resp = client.post(
            "/v1/frames/analyze",
            data={"prompt": "What do you see?"},
            files=files,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["text"] == "Test analysis result"
        assert data["metrics"]["frames_input"] == 10

    def test_upload_frames_checks_engine_call(self, client):
        """Verify frames are passed as numpy array to engine."""
        import io

        import numpy as np
        from PIL import Image

        import trio_core.api.server as server_mod

        img = Image.new("RGB", (4, 4), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        resp = client.post(
            "/v1/frames/analyze",
            data={"prompt": "test", "max_tokens": "100"},
            files=[("frames", ("f.png", buf, "image/png"))],
        )
        assert resp.status_code == 200
        # Verify analyze_video was called with numpy array
        call_args = server_mod._engine.analyze_video.call_args
        video_arg = call_args.kwargs.get("video", call_args[1].get("video"))
        assert isinstance(video_arg, np.ndarray)
        assert video_arg.shape[0] == 1  # 1 frame
        assert video_arg.shape[1] == 3  # 3 channels


class TestTrioClawCompat:
    """Tests for TrioClaw-compatible endpoints (/healthz, /analyze-frame)."""

    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_analyze_frame(self, client):
        import base64
        import io

        from PIL import Image

        # Create a test JPEG
        img = Image.new("RGB", (8, 8), (255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post(
            "/analyze-frame",
            json={
                "frame_b64": frame_b64,
                "question": "what do you see?",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "latency_ms" in data
        assert data["answer"] == "Test analysis result"

    def test_analyze_frame_triggered_yes(self, client):
        """triggered=true when answer starts with 'Yes'."""
        import base64
        import io

        from PIL import Image

        import trio_core.api.server as server_mod

        # Mock answer starting with "Yes"
        server_mod._engine.analyze_video.return_value = VideoResult(
            text="Yes, there is a person at the door.",
            metrics=InferenceMetrics(latency_ms=100.0),
        )

        img = Image.new("RGB", (8, 8), (0, 0, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post(
            "/analyze-frame",
            json={
                "frame_b64": frame_b64,
                "question": "is there a person at the door?",
            },
        )
        data = resp.json()
        assert data["triggered"] is True

    def test_analyze_frame_triggered_no(self, client):
        """triggered=false when answer starts with 'No'."""
        import base64
        import io

        from PIL import Image

        import trio_core.api.server as server_mod

        server_mod._engine.analyze_video.return_value = VideoResult(
            text="No, the porch is empty.",
            metrics=InferenceMetrics(latency_ms=100.0),
        )

        img = Image.new("RGB", (8, 8), (0, 255, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post(
            "/analyze-frame",
            json={
                "frame_b64": frame_b64,
                "question": "is there a package?",
            },
        )
        data = resp.json()
        assert data["triggered"] is False

    def test_analyze_frame_triggered_null(self, client):
        """triggered=null when answer is descriptive (not yes/no)."""
        import base64
        import io

        from PIL import Image

        import trio_core.api.server as server_mod

        server_mod._engine.analyze_video.return_value = VideoResult(
            text="The image shows a red square on a white background.",
            metrics=InferenceMetrics(latency_ms=100.0),
        )

        img = Image.new("RGB", (8, 8), (255, 255, 255))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        frame_b64 = base64.b64encode(buf.getvalue()).decode()

        resp = client.post(
            "/analyze-frame",
            json={
                "frame_b64": frame_b64,
                "question": "describe this image",
            },
        )
        data = resp.json()
        assert data["triggered"] is None


class TestChatCompletions:
    def test_no_video_returns_400(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
            },
        )
        assert resp.status_code == 400

    def test_with_video(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": "/tmp/test.mp4"},
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Test analysis result"

    def test_with_image_url(self, client):
        """OpenAI-compatible image_url content part."""
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": "/tmp/test.jpg"}},
                            {"type": "text", "text": "What do you see?"},
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Test analysis result"
        assert data["object"] == "chat.completion"
        assert data["usage"]["prompt_tokens"] == 100

    def test_with_base64_image(self, client):
        """Base64 data URI image."""
        import base64

        # 1x1 red PNG
        pixel = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode()
        resp = client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{pixel}"},
                            },
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class TestGetEngineNotLoaded:
    """Test get_engine raises 503 when _engine is None (line 46)."""

    def test_get_engine_raises_503_none(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import get_engine

        saved = server_mod._engine
        try:
            server_mod._engine = None
            with pytest.raises(HTTPException) as exc_info:
                get_engine()
            assert exc_info.value.status_code == 503
        finally:
            server_mod._engine = saved

    def test_get_engine_raises_503_not_loaded(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import get_engine

        saved = server_mod._engine
        try:
            mock = MagicMock()
            mock._loaded = False
            server_mod._engine = mock
            with pytest.raises(HTTPException) as exc_info:
                get_engine()
            assert exc_info.value.status_code == 503
        finally:
            server_mod._engine = saved


class TestHealthNotLoaded:
    """Test /health and /healthz when engine is None (lines 90, 97)."""

    def test_health_not_loaded(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import create_app

        app = create_app()
        app.router.lifespan_context = noop_lifespan
        saved = server_mod._engine
        try:
            server_mod._engine = None
            with TestClient(app) as c:
                resp = c.get("/health")
                assert resp.status_code == 200
                data = resp.json()
                assert data["status"] == "not_loaded"
                assert data["loaded"] is False
        finally:
            server_mod._engine = saved

    def test_healthz_not_loaded(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import create_app

        app = create_app()
        app.router.lifespan_context = noop_lifespan
        saved = server_mod._engine
        try:
            server_mod._engine = None
            with TestClient(app) as c:
                resp = c.get("/healthz")
                assert resp.status_code == 503
        finally:
            server_mod._engine = saved


class TestDetectTriggered:
    """Test _detect_triggered with positive patterns and None (lines 293, 296)."""

    def test_positive_there_is_a(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("There is a person at the door.") is True

    def test_positive_i_can_see(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("I can see someone approaching.") is True

    def test_positive_someone(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("Someone is standing there.") is True

    def test_positive_a_package(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("A package is on the porch.") is True

    def test_negative_there_is_no(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("There is no one at the door.") is False

    def test_negative_cannot_see(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("I cannot see anything.") is False

    def test_none_for_ambiguous(self):
        from trio_core.api.server import _detect_triggered

        assert _detect_triggered("The image shows a red square on a white background.") is None


class TestResolveMedia:
    """Test _resolve_media with invalid data URI (line 342)."""

    def test_invalid_data_uri_no_comma(self):
        from trio_core.api.server import _resolve_media

        with pytest.raises(HTTPException) as exc_info:
            _resolve_media("data:image/jpeg;base64nocomma")
        assert exc_info.value.status_code == 400
        assert "missing comma" in exc_info.value.detail

    def test_plain_path_passthrough(self):
        from trio_core.api.server import _resolve_media

        path, temp = _resolve_media("/tmp/test.mp4")
        assert path == "/tmp/test.mp4"
        assert temp is None


class TestStreamVideoAnalyze:
    """Test streaming for /v1/video/analyze (lines 157, 367-378)."""

    def test_stream_video_analyze(self, client):
        import trio_core.api.server as server_mod

        metrics = InferenceMetrics(
            frames_input=5, prompt_tokens=50, completion_tokens=20, latency_ms=200.0
        )

        async def mock_stream_analyze(**kwargs):
            yield {"text": "hello ", "finished": False}
            yield {"text": "world", "finished": False}
            yield {"text": "", "finished": True, "metrics": metrics}

        server_mod._engine.stream_analyze = mock_stream_analyze

        resp = client.post(
            "/v1/video/analyze",
            json={"video": "/tmp/test.mp4", "prompt": "describe", "stream": True},
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data: ")]
        assert len(lines) >= 3  # 2 content + 1 metrics + DONE
        # Check first chunk
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["text"] == "hello "
        assert first["finished"] is False
        # Check final metrics chunk
        metrics_line = json.loads(lines[2].removeprefix("data: "))
        assert metrics_line["finished"] is True
        assert "metrics" in metrics_line
        # Check DONE sentinel
        assert lines[-1] == "data: [DONE]"


class TestStreamFramesAnalyze:
    """Test streaming for /v1/frames/analyze (lines 205, 389-400)."""

    def test_stream_frames_analyze(self, client):
        import io

        from PIL import Image

        import trio_core.api.server as server_mod

        metrics = InferenceMetrics(latency_ms=100.0)

        async def mock_stream_analyze(**kwargs):
            yield {"text": "a frame", "finished": False}
            yield {"text": "", "finished": True, "metrics": metrics}

        server_mod._engine.stream_analyze = mock_stream_analyze

        img = Image.new("RGB", (4, 4), (128, 128, 128))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        resp = client.post(
            "/v1/frames/analyze",
            data={"prompt": "describe", "stream": "true"},
            files=[("frames", ("f.jpg", buf, "image/jpeg"))],
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data: ")]
        assert len(lines) >= 2
        assert lines[-1] == "data: [DONE]"


class TestStreamChatCompletions:
    """Test streaming for /v1/chat/completions (lines 244, 412-451)."""

    def test_stream_chat_completions(self, client):
        import trio_core.api.server as server_mod

        metrics = InferenceMetrics(prompt_tokens=50, completion_tokens=10, latency_ms=150.0)

        async def mock_stream_analyze(**kwargs):
            yield {"text": "The video ", "finished": False}
            yield {"text": "shows a cat.", "finished": False}
            yield {"text": "", "finished": True, "metrics": metrics}

        server_mod._engine.stream_analyze = mock_stream_analyze

        resp = client.post(
            "/v1/chat/completions",
            json={
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "video", "video": "/tmp/test.mp4"},
                            {"type": "text", "text": "What is this?"},
                        ],
                    }
                ],
            },
        )
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers["content-type"]

        lines = [ln for ln in resp.text.strip().split("\n") if ln.startswith("data: ")]
        # First line should be role chunk
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["choices"][0]["delta"]["role"] == "assistant"
        # Content chunks
        second = json.loads(lines[1].removeprefix("data: "))
        assert second["choices"][0]["delta"]["content"] == "The video "
        # Last data line should be [DONE]
        assert lines[-1] == "data: [DONE]"
        # There should be a finish_reason="stop" chunk
        all_chunks = [json.loads(ln.removeprefix("data: ")) for ln in lines if ln != "data: [DONE]"]
        finish_chunks = [
            c for c in all_chunks if c.get("choices", [{}])[0].get("finish_reason") == "stop"
        ]
        assert len(finish_chunks) == 1


class TestLifespan:
    """Test app lifespan loads engine (lines 53-60)."""

    @pytest.mark.asyncio
    async def test_lifespan_loads_engine(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import create_app, lifespan

        app = create_app()
        saved = server_mod._engine
        try:
            server_mod._engine = None
            with patch("trio_core.api.server.TrioCore") as MockTrioCore:
                mock_instance = MagicMock()
                mock_instance._loaded = True
                mock_instance._backend.backend_name = "mock"
                mock_instance.health.return_value = {
                    "status": "ok",
                    "model": "test",
                    "loaded": True,
                    "config": {},
                }
                MockTrioCore.return_value = mock_instance

                # Directly invoke the lifespan context manager
                async with lifespan(app):
                    # Engine should now be loaded
                    assert server_mod._engine is mock_instance

                MockTrioCore.assert_called_once()
                mock_instance.load.assert_called_once()
        finally:
            server_mod._engine = saved


class TestMetricsEndpoint:
    """Test GET /metrics returns operational stats."""

    def test_metrics(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "uptime_s" in data
        assert "process" in data
        assert "rss_mb" in data["process"]
        assert "requests" in data
        assert data["requests"]["total"] >= 1  # this request counts
        assert "inference" in data
        assert "watches" in data
        assert "engine" in data
        assert data["engine"]["loaded"] is True


class TestBodySizeLimit:
    """Test request body size limit (413 on oversized payload)."""

    def test_oversized_body_rejected(self):
        import trio_core.api.server as server_mod
        from trio_core.api.server import _MAX_BODY_BYTES, create_app

        app = create_app()
        app.router.lifespan_context = noop_lifespan

        mock_engine = MagicMock()
        mock_engine._loaded = True
        mock_engine.config.model = "test-model"
        server_mod._engine = mock_engine

        try:
            with TestClient(app) as c:
                # Send a Content-Length header that exceeds the limit
                resp = c.post(
                    "/analyze-frame",
                    json={"frame_b64": "x", "question": "test"},
                    headers={"Content-Length": str(_MAX_BODY_BYTES + 1)},
                )
                assert resp.status_code == 413
                assert "payload_too_large" in resp.json()["error"]
        finally:
            server_mod._engine = None


class TestReloadEndpoint:
    """Test POST /v1/admin/reload hot-reloads the engine."""

    def test_reload_same_model(self, client):

        with patch("trio_core.api.server.TrioCore") as MockTrioCore:
            new_engine = MagicMock()
            new_engine._loaded = True
            new_engine._backend.backend_name = "mock"
            MockTrioCore.return_value = new_engine

            resp = client.post("/v1/admin/reload")
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "reloaded"

    def test_reload_failure_rolls_back(self, client):
        import trio_core.api.server as server_mod

        old_engine = server_mod._engine

        with patch("trio_core.api.server.TrioCore") as MockTrioCore:
            new_engine = MagicMock()
            new_engine.load.side_effect = RuntimeError("OOM")
            MockTrioCore.return_value = new_engine

            resp = client.post("/v1/admin/reload")
            assert resp.status_code == 500
            assert "OOM" in resp.json()["detail"]

            # Old engine should be restored
            assert server_mod._engine is old_engine


class TestStructuredLogging:
    """Test JSON log formatter."""

    def test_json_formatter(self):
        import json
        import logging

        from trio_core.cli import _JSONFormatter

        formatter = _JSONFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="hello %s",
            args=("world",),
            exc_info=None,
        )
        output = formatter.format(record)
        data = json.loads(output)
        assert data["level"] == "info"
        assert data["msg"] == "hello world"
        assert "ts" in data
