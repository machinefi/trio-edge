"""Tests for trio_core.api — uses mocked engine."""

from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from fastapi.testclient import TestClient

from trio_core.engine import VideoResult, InferenceMetrics


@pytest.fixture
def client():
    """Create test client with mocked engine."""
    from trio_core.api.server import create_app, _engine
    import trio_core.api.server as server_mod

    app = create_app()

    # Mock the engine
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
        "status": "ok",
        "model": "test-model",
        "loaded": True,
        "config": {},
    }
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

    # Skip lifespan (model loading)
    app.router.lifespan_context = _noop_lifespan

    with TestClient(app) as c:
        yield c

    server_mod._engine = None


from contextlib import asynccontextmanager

@asynccontextmanager
async def _noop_lifespan(app):
    yield


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
        resp = client.post("/v1/video/analyze", json={
            "video": "/tmp/test.mp4",
            "prompt": "What is happening?",
        })
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
        import io, numpy as np
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

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "what do you see?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "latency_ms" in data
        assert data["answer"] == "Test analysis result"

    def test_analyze_frame_triggered_yes(self, client):
        """triggered=true when answer starts with 'Yes'."""
        import base64, io
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

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "is there a person at the door?",
        })
        data = resp.json()
        assert data["triggered"] is True

    def test_analyze_frame_triggered_no(self, client):
        """triggered=false when answer starts with 'No'."""
        import base64, io
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

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "is there a package?",
        })
        data = resp.json()
        assert data["triggered"] is False

    def test_analyze_frame_triggered_null(self, client):
        """triggered=null when answer is descriptive (not yes/no)."""
        import base64, io
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

        resp = client.post("/analyze-frame", json={
            "frame_b64": frame_b64,
            "question": "describe this image",
        })
        data = resp.json()
        assert data["triggered"] is None


class TestChatCompletions:
    def test_no_video_returns_400(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 400

    def test_with_video(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "video", "video": "/tmp/test.mp4"},
                    {"type": "text", "text": "What is this?"},
                ],
            }],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["content"] == "Test analysis result"

    def test_with_image_url(self, client):
        """OpenAI-compatible image_url content part."""
        resp = client.post("/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "/tmp/test.jpg"}},
                    {"type": "text", "text": "What do you see?"},
                ],
            }],
        })
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
        resp = client.post("/v1/chat/completions", json={
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{pixel}"}},
                    {"type": "text", "text": "What is this?"},
                ],
            }],
        })
        assert resp.status_code == 200
