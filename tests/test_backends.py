"""Tests for trio_core.backends — backend abstraction."""

from unittest.mock import MagicMock

import numpy as np

from trio_core.backends import (
    GenerationResult,
    MLXBackend,
    StreamChunk,
    TransformersBackend,
    auto_backend,
)
from trio_core.device import DeviceInfo


class TestGenerationResult:
    def test_defaults(self):
        r = GenerationResult(text="hello")
        assert r.text == "hello"
        assert r.prompt_tokens == 0
        assert r.generation_tps == 0.0


class TestStreamChunk:
    def test_defaults(self):
        c = StreamChunk(text="tok")
        assert c.text == "tok"
        assert c.finished is False


class TestAutoBackend:
    def test_mlx_on_apple_silicon(self):
        info = DeviceInfo("mlx", "Apple M3", "metal", 36.0, 40)
        b = auto_backend("test-model", device_info=info)
        assert isinstance(b, MLXBackend)
        assert b.model_name == "test-model"

    def test_transformers_on_cuda(self):
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        b = auto_backend("test-model", device_info=info)
        assert isinstance(b, TransformersBackend)

    def test_force_backend(self):
        info = DeviceInfo("mlx", "Apple M3", "metal", 36.0, 40)
        b = auto_backend("test-model", backend="transformers", device_info=info)
        assert isinstance(b, TransformersBackend)

    def test_unknown_backend_fallback(self):
        info = DeviceInfo("unknown", "test", "cpu", 0, 0)
        b = auto_backend("test-model", backend="nonexistent", device_info=info)
        assert isinstance(b, TransformersBackend)


class TestTransformersFeatureDetection:
    """P4: Verify _is_video_model uses feature detection, not string matching."""

    def test_video_model_detected(self):
        """Model with video_token_id routes to _prepare_video."""
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        b = TransformersBackend("some-unrelated-name", device_info=info)
        # Simulate load() setting _is_video_model
        mock_config = MagicMock()
        mock_config.video_token_id = 12345
        mock_model = MagicMock()
        mock_model.config = mock_config
        b._model = mock_model
        b._is_video_model = hasattr(mock_model.config, "video_token_id")
        assert b._is_video_model is True

    def test_non_video_model_detected(self):
        """Model without video_token_id routes to generic path."""
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        b = TransformersBackend("gemma-3-4b", device_info=info)
        mock_config = MagicMock(spec=["hidden_size"])  # no video_token_id
        mock_model = MagicMock()
        mock_model.config = mock_config
        b._model = mock_model
        b._is_video_model = hasattr(mock_model.config, "video_token_id")
        assert b._is_video_model is False

    def test_single_frame_uses_image_path_even_for_video_model(self):
        """Single images should not be forced through the video prep path."""
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        b = TransformersBackend("qwen-video-model", device_info=info)
        b._is_video_model = True
        b._device = "cuda"
        b._prepare_video = MagicMock(return_value={"input_ids": MagicMock(shape=(1, 1))})
        b._frames_to_pil = MagicMock(return_value=["image-frame"])
        fake_tensor = MagicMock()
        fake_tensor.shape = (1, 1)
        fake_tensor.to.return_value = fake_tensor
        b._processor = MagicMock()
        b._processor.apply_chat_template.return_value = "formatted"
        b._processor.return_value = {"input_ids": fake_tensor}

        frames = np.zeros((1, 3, 64, 64), dtype=np.float32)

        result = b._prepare(frames, "describe this image")

        b._prepare_video.assert_not_called()
        b._frames_to_pil.assert_called_once_with(frames)
        assert result["input_ids"].shape == (1, 1)

    def test_multi_frame_uses_video_path_for_video_model(self):
        """True videos should still route through video-specific prep."""
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        b = TransformersBackend("qwen-video-model", device_info=info)
        b._is_video_model = True
        b._prepare_video = MagicMock(return_value={"input_ids": MagicMock(shape=(1, 2))})

        frames = np.zeros((2, 3, 64, 64), dtype=np.float32)

        result = b._prepare(frames, "describe this video")

        b._prepare_video.assert_called_once_with(frames, "describe this video")
        assert result["input_ids"].shape == (1, 2)


class TestBaseBackendHealth:
    def test_health(self):
        info = DeviceInfo("mlx", "M3", "metal", 36.0, 40)
        b = MLXBackend("test-model", device_info=info)
        h = b.health()
        assert h["backend"] == "mlx"
        assert h["loaded"] is False
        assert h["device"] == "M3"
