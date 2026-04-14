"""Tests for trio_core.engine — uses mocks since we may not have the model."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from trio_core.backends import GenerationResult, StreamChunk
from trio_core.config import EngineConfig
from trio_core.engine import InferenceMetrics, TrioCore, VideoResult


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
        "backend": backend_name,
        "model": "test",
        "loaded": True,
        "device": "Test Device",
        "accelerator": "cpu",
        "memory_gb": 16.0,
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
    @patch("trio_core.engine.resolve_backend")
    def test_load_auto_selects_backend(self, mock_auto):
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_auto.return_value = mock_backend

        engine = TrioCore()
        engine.load()
        mock_auto.assert_called_once_with(engine.config, backend_override=None)
        mock_backend.load.assert_called_once()
        assert engine._loaded

    @patch("trio_core.engine.resolve_backend")
    def test_load_respects_backend_override(self, mock_auto):
        mock_backend = MagicMock()
        mock_backend.backend_name = "transformers"
        mock_backend.device_info.device_name = "CPU"
        mock_auto.return_value = mock_backend

        engine = TrioCore(backend="transformers")
        engine.load()
        mock_auto.assert_called_once_with(engine.config, backend_override="transformers")


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


class TestLoadIdempotency:
    """load() called twice — second call returns early (lines 131-132)."""

    @patch("trio_core.engine.resolve_backend")
    def test_load_twice_is_idempotent(self, mock_auto):
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_auto.return_value = mock_backend

        engine = TrioCore()
        engine.load()
        engine.load()  # second call — should short-circuit

        # auto_backend only called once
        mock_auto.assert_called_once()
        # backend.load() only called once
        mock_backend.load.assert_called_once()


class TestRemoteLoadPath:
    """When config.remote_vlm_url is set, load() bypasses resolve_backend entirely."""

    @patch("trio_core.engine.resolve_backend")
    def test_remote_url_creates_remote_backend(self, mock_resolve):
        config = EngineConfig(
            remote_vlm_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            remote_vlm_api_key="sk-test",
            remote_vlm_model="qwen-vl-plus",
        )
        engine = TrioCore(config)

        with patch("trio_core.remote_backend.RemoteHTTPBackend") as MockBackend:
            mock_instance = MagicMock()
            mock_instance.backend_name = "remote"
            MockBackend.return_value = mock_instance
            engine.load()

        MockBackend.assert_called_once_with(
            url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            api_key="sk-test",
            model="qwen-vl-plus",
        )
        mock_instance.load.assert_called_once()
        assert engine._loaded
        mock_resolve.assert_not_called()

    @patch("trio_core.engine.resolve_backend")
    def test_no_remote_url_falls_through_to_resolve_backend(self, mock_resolve):
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_resolve.return_value = mock_backend

        config = EngineConfig()
        assert config.remote_vlm_url is None
        engine = TrioCore(config)
        engine.load()

        mock_resolve.assert_called_once()


class TestBackendSwap:
    """Backend swap logic in resolve_backend: ToMe-only."""

    @patch("trio_core.backends.registry.auto_backend")
    def test_tome_only_swaps_to_tome_backend(self, mock_auto):
        """tome_enabled=True → ToMeMLXBackend."""
        from trio_core.backends.registry import resolve_backend

        base_backend = MagicMock()
        base_backend.backend_name = "mlx"
        base_backend.device_info.device_name = "Apple M3"
        mock_auto.return_value = base_backend

        mock_tome_instance = MagicMock()
        mock_tome_instance.backend_name = "tome_mlx"

        config = EngineConfig(tome_enabled=True, tome_r=4)

        mock_tb = MagicMock(return_value=mock_tome_instance)
        with patch.dict(
            "sys.modules",
            {
                "trio_core.tome_backend": MagicMock(ToMeMLXBackend=mock_tb),
                "trio_core.backends.tome": MagicMock(ToMeMLXBackend=mock_tb),
            },
        ):
            result = resolve_backend(config)

        assert result is mock_tome_instance
        mock_tb.assert_called_once()


class TestFeatureFlagMethods:
    """Feature flags: early_stop, visual_similarity, streaming_memory (lines 181, 185, 189)."""

    @patch("trio_core.engine.resolve_backend")
    def test_early_stop_flag(self, mock_auto):
        """early_stop=True calls backend.set_early_stop (line 181)."""
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_backend.set_early_stop = MagicMock()
        mock_auto.return_value = mock_backend

        config = EngineConfig(early_stop=True, early_stop_threshold=0.9)
        engine = TrioCore(config)
        engine.load()

        mock_backend.set_early_stop.assert_called_once_with(True, 0.9)

    @patch("trio_core.engine.resolve_backend")
    def test_visual_similarity_flag(self, mock_auto):
        """visual_similarity_threshold > 0 calls backend.set_visual_similarity (line 185)."""
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_backend.set_visual_similarity = MagicMock()
        mock_auto.return_value = mock_backend

        config = EngineConfig(visual_similarity_threshold=0.95)
        engine = TrioCore(config)
        engine.load()

        mock_backend.set_visual_similarity.assert_called_once_with(0.95)

    @patch("trio_core.engine.resolve_backend")
    def test_streaming_memory_flag(self, mock_auto):
        """streaming_memory_enabled calls backend.set_streaming_memory (line 189)."""
        mock_backend = MagicMock()
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_backend.set_streaming_memory = MagicMock()
        mock_auto.return_value = mock_backend

        config = EngineConfig(
            streaming_memory_enabled=True,
            streaming_memory_budget=8000,
            streaming_memory_prototype_ratio=0.2,
            streaming_memory_sink_tokens=8,
        )
        engine = TrioCore(config)
        engine.load()

        mock_backend.set_streaming_memory.assert_called_once_with(True, 8000, 0.2, 8)

    @patch("trio_core.engine.resolve_backend")
    def test_feature_flags_skip_when_backend_lacks_method(self, mock_auto):
        """Feature flags are skipped if backend doesn't have the method."""
        mock_backend = MagicMock(spec=["backend_name", "device_info", "load", "health"])
        mock_backend.backend_name = "mlx"
        mock_backend.device_info.device_name = "Apple M3"
        mock_auto.return_value = mock_backend

        config = EngineConfig(
            early_stop=True,
            visual_similarity_threshold=0.95,
            streaming_memory_enabled=True,
        )
        engine = TrioCore(config)
        # Should not raise even though backend lacks these methods
        engine.load()
        assert engine._loaded


class TestMotionGate:
    """Motion gate path in analyze_video (lines 255, 259-267)."""

    def test_dedup_disabled_still_sets_frames_after_dedup(self):
        """dedup_enabled=False → metrics.frames_after_dedup = input count (line 255)."""
        config = EngineConfig(dedup_enabled=False)
        engine = TrioCore(config)
        mock_backend = MagicMock()
        mock_backend.backend_name = "mock"
        mock_backend.generate.return_value = GenerationResult(
            text="ok",
            prompt_tokens=10,
            completion_tokens=5,
            generation_tps=50.0,
            peak_memory=1.0,
        )
        engine._backend = mock_backend
        engine._loaded = True

        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "test")
        assert result.metrics.frames_after_dedup == 4

    def test_motion_gate_no_motion_skips(self):
        """motion_enabled=True + no motion → returns early with skip text (lines 259-267)."""
        config = EngineConfig(motion_enabled=True, motion_threshold=100.0)
        engine = TrioCore(config)
        mock_backend = MagicMock()
        mock_backend.backend_name = "mock"
        engine._backend = mock_backend
        engine._loaded = True

        # Use identical frames so motion gate sees no motion
        frame = np.zeros((64, 64, 3), dtype=np.float32)
        frames = np.stack([frame] * 4)  # (4, 64, 64, 3) — identical
        # Patch the motion gate to always return False
        engine._motion_gate = MagicMock()
        engine._motion_gate.has_motion.return_value = False

        result = engine.analyze_video(frames, "describe")
        assert result.metrics.motion_skipped is True
        assert "[NO MOTION]" in result.text
        # Backend.generate should NOT have been called
        mock_backend.generate.assert_not_called()

    def test_motion_gate_with_motion_proceeds(self):
        """motion_enabled=True + motion detected → proceeds to inference."""
        config = EngineConfig(motion_enabled=True, motion_threshold=0.001)
        engine = TrioCore(config)
        mock_backend = MagicMock()
        mock_backend.backend_name = "mock"
        mock_backend.generate.return_value = GenerationResult(
            text="motion detected",
            prompt_tokens=10,
            completion_tokens=5,
            generation_tps=50.0,
            peak_memory=1.0,
        )
        engine._backend = mock_backend
        engine._loaded = True

        engine._motion_gate = MagicMock()
        engine._motion_gate.has_motion.return_value = True

        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "describe")
        assert result.metrics.motion_skipped is False
        assert result.text == "motion detected"
        mock_backend.generate.assert_called_once()


class TestPrefillTiming:
    """Prefill timing derived from prompt_tps (line 290)."""

    def test_prefill_ms_derived_from_prompt_tps(self):
        """prompt_tps > 0 → prefill_ms = (prompt_tokens / prompt_tps) * 1000."""
        engine = _mock_engine()
        engine._backend.generate.return_value = GenerationResult(
            text="result",
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tps=500.0,  # 500 tok/sec → 200ms for 100 tokens
            generation_tps=50.0,
            peak_memory=1.0,
        )
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "test")

        assert result.metrics.prefill_ms == pytest.approx(200.0, rel=1e-3)
        assert result.metrics.decode_ms == pytest.approx(200.0, rel=1e-3)

    def test_prefill_ms_zero_when_prompt_tps_zero(self):
        """prompt_tps == 0 → prefill_ms stays 0 (no division)."""
        engine = _mock_engine()
        engine._backend.generate.return_value = GenerationResult(
            text="result",
            prompt_tokens=100,
            completion_tokens=10,
            prompt_tps=0.0,
            generation_tps=50.0,
            peak_memory=1.0,
        )
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = engine.analyze_video(frames, "test")

        assert result.metrics.prefill_ms == 0.0


class TestStreamAnalyze:
    """stream_analyze() async generator (lines 313-351)."""

    @pytest.mark.asyncio
    async def test_stream_analyze_yields_chunks_and_final(self):
        """Iterate stream_analyze — gets text chunks then final with metrics."""
        engine = _mock_engine()

        # Mock stream_generate to yield 3 chunks
        chunks = [
            StreamChunk(text="Hello"),
            StreamChunk(text=" world"),
            StreamChunk(text="!"),
        ]
        engine._backend.stream_generate.return_value = iter(chunks)

        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        results = []
        async for item in engine.stream_analyze(frames, "test"):
            results.append(item)

        # Should have 3 text chunks + 1 final
        assert len(results) == 4

        # First 3 are text chunks
        for i, chunk in enumerate(results[:3]):
            assert chunk["finished"] is False
            assert chunk["metrics"] is None
        assert results[0]["text"] == "Hello"
        assert results[1]["text"] == " world"
        assert results[2]["text"] == "!"

        # Last is the final with metrics
        final = results[-1]
        assert final["finished"] is True
        assert final["metrics"] is not None
        assert isinstance(final["metrics"], InferenceMetrics)
        assert final["metrics"].latency_ms > 0

    @pytest.mark.asyncio
    async def test_stream_analyze_not_loaded_raises(self):
        """stream_analyze raises RuntimeError if not loaded."""
        engine = TrioCore()
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            async for _ in engine.stream_analyze(frames, "test"):
                pass

    @pytest.mark.asyncio
    async def test_stream_analyze_dedup_disabled(self):
        """stream_analyze with dedup_enabled=False still sets frames_after_dedup."""
        config = EngineConfig(dedup_enabled=False)
        engine = TrioCore(config)
        mock_backend = MagicMock()
        mock_backend.backend_name = "mock"
        mock_backend.stream_generate.return_value = iter([StreamChunk(text="ok")])
        engine._backend = mock_backend
        engine._loaded = True

        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        results = []
        async for item in engine.stream_analyze(frames, "test"):
            results.append(item)

        final = results[-1]
        assert final["finished"] is True
        assert final["metrics"].frames_after_dedup == 4
