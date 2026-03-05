"""Tests for trio_core.engine — uses mocks since we may not have the model."""

from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from trio_core.backends import GenerationResult
from trio_core.config import EngineConfig
from trio_core.engine import TrioCore, VideoResult, InferenceMetrics


def _mock_engine(backend_name="mock") -> TrioCore:
    """Create a TrioCore with a mocked backend."""
    engine = TrioCore()
    mock_backend = MagicMock()
    mock_backend.backend_name = backend_name
    mock_backend.device_info.device_name = "Test Device"
    mock_backend.device_info.accelerator = "cpu"
    mock_backend.device_info.memory_gb = 16.0
    mock_backend.loaded = True
    mock_backend.generate.return_value = GenerationResult(
        text="test output",
        prompt_tokens=10,
        completion_tokens=5,
        generation_tps=50.0,
        peak_memory=1.0,
    )
    mock_backend.health.return_value = {
        "backend": backend_name, "model": "test", "loaded": True,
        "device": "Test Device", "accelerator": "cpu", "memory_gb": 16.0,
    }
    engine._backend = mock_backend
    engine._loaded = True
    return engine


class TestTrioCoreInit:
    def test_default_config(self):
        engine = TrioCore()
        assert engine.config.model is not None
        assert not engine._loaded

    def test_custom_config(self):
        config = EngineConfig(model="test-model", max_tokens=256)
        engine = TrioCore(config)
        assert engine.config.model == "test-model"
        assert engine.config.max_tokens == 256

    def test_backend_override(self):
        engine = TrioCore(backend="transformers")
        assert engine._backend_override == "transformers"


class TestTrioCoreHealth:
    def test_health_not_loaded(self):
        engine = TrioCore()
        h = engine.health()
        assert h["status"] == "not_loaded"
        assert h["loaded"] is False

    def test_health_loaded_with_backend(self):
        engine = _mock_engine()
        h = engine.health()
        assert h["status"] == "ok"
        assert h["loaded"] is True
        assert "backend" in h


class TestTrioCoreLoad:
    @patch("trio_core.engine.auto_backend")
    def test_load_auto_selects_backend(self, mock_auto):
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_auto.return_value = mock_backend

        engine = TrioCore()
        engine.load()
        mock_auto.assert_called_once_with(engine.config.model, backend=None)
        mock_backend.load.assert_called_once()
        assert engine._loaded

    @patch("trio_core.engine.auto_backend")
    def test_load_respects_backend_override(self, mock_auto):
        mock_backend = MagicMock()
        mock_backend.backend_name = "transformers"
        mock_backend.device_info.device_name = "CPU"
        mock_auto.return_value = mock_backend

        engine = TrioCore(backend="transformers")
        engine.load()
        mock_auto.assert_called_once_with(engine.config.model, backend="transformers")


class TestAnalyzeVideo:
    def test_not_loaded_raises(self):
        engine = TrioCore()
        with pytest.raises(RuntimeError, match="not loaded"):
            engine.analyze_video(np.random.rand(4, 3, 64, 64).astype(np.float32), "test")

    def test_analyze_delegates_to_backend(self):
        engine = _mock_engine()
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "test")
        assert result.text == "test output"
        engine._backend.generate.assert_called_once()

    def test_analyze_frame_wraps_to_4d(self):
        engine = _mock_engine()
        frame = np.random.rand(64, 64, 3).astype(np.float32)  # (H, W, C)
        result = engine.analyze_frame(frame, "test")
        assert isinstance(result, VideoResult)
        assert result.text == "test output"

    def test_three_phase_timing(self):
        engine = _mock_engine()
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "test")
        m = result.metrics
        assert m.preprocess_ms >= 0
        assert m.inference_ms >= 0
        assert m.postprocess_ms >= 0
        assert m.latency_ms >= m.preprocess_ms + m.inference_ms

    def test_callbacks_fire_during_analyze(self):
        engine = _mock_engine()
        events_fired = []
        engine.add_callback("on_frame_captured", lambda e: events_fired.append("frame"))
        engine.add_callback("on_dedup_done", lambda e: events_fired.append("dedup"))
        engine.add_callback("on_vlm_start", lambda e: events_fired.append("vlm_start"))
        engine.add_callback("on_vlm_end", lambda e: events_fired.append("vlm_end"))

        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        engine.analyze_video(frames, "test")

        assert "frame" in events_fired
        assert "dedup" in events_fired
        assert "vlm_start" in events_fired
        assert "vlm_end" in events_fired
