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
