"""Tests for trio_core.backends — backend abstraction."""

from unittest.mock import MagicMock, patch

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


class TestRemoteHTTPBackend:
    def test_init_sets_device_info(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(
            url="https://api.example.com/v1", api_key="sk-test", model="qwen-vl-plus"
        )
        assert b.backend_name == "remote"
        assert b.model_name == "qwen-vl-plus"
        assert b.device_info.backend == "remote"
        assert b.device_info.device_name == "Remote API"
        assert b.device_info.accelerator == "remote"
        assert not b.loaded

    def test_load_creates_client(self):
        import sys

        from trio_core.backends.remote import RemoteHTTPBackend

        mock_client = MagicMock()
        mock_openai_mod = MagicMock()
        mock_openai_mod.OpenAI.return_value = mock_client

        b = RemoteHTTPBackend(url="https://api.example.com/v1", api_key="sk-test")
        with patch.dict(sys.modules, {"openai": mock_openai_mod}):
            b.load()

        mock_openai_mod.OpenAI.assert_called_once_with(
            base_url="https://api.example.com/v1",
            api_key="sk-test",
        )
        assert b.loaded
        assert b._client is mock_client

    def test_load_with_no_api_key(self):
        import sys

        from trio_core.backends.remote import RemoteHTTPBackend

        mock_openai_mod = MagicMock()

        b = RemoteHTTPBackend(url="https://api.example.com/v1")
        with patch.dict(sys.modules, {"openai": mock_openai_mod}):
            b.load()

        mock_openai_mod.OpenAI.assert_called_once_with(
            base_url="https://api.example.com/v1",
            api_key="unused",
        )

    def test_generate_builds_correct_messages(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(url="https://api.example.com/v1", model="qwen-vl-plus")
        mock_client = MagicMock()
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 20
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "A cat on a mat"
        mock_response.usage = mock_usage
        mock_client.chat.completions.create.return_value = mock_response

        b._client = mock_client
        b._loaded = True

        frames = np.random.rand(2, 3, 64, 64).astype(np.float32)
        result = b.generate(frames, "describe this", max_tokens=256, temperature=0.1)

        assert result.text == "A cat on a mat"
        assert result.prompt_tokens == 50
        assert result.completion_tokens == 20
        assert result.generation_tps > 0

        call_kwargs = mock_client.chat.completions.create.call_args
        assert call_kwargs.kwargs["model"] == "qwen-vl-plus"
        assert call_kwargs.kwargs["max_tokens"] == 256
        assert call_kwargs.kwargs["temperature"] == 0.1

        messages = call_kwargs.kwargs["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        content = messages[0]["content"]
        image_blocks = [c for c in content if c["type"] == "image_url"]
        text_blocks = [c for c in content if c["type"] == "text"]
        assert len(image_blocks) == 2  # 2 frames → 2 image_url entries
        assert len(text_blocks) == 1
        assert text_blocks[0]["text"] == "describe this"

    def test_generate_handles_empty_content(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(url="https://api.example.com/v1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = None
        mock_response.usage = None
        MagicMock().chat.completions.create.return_value = mock_response

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        b._client = mock_client
        b._loaded = True

        frames = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = b.generate(frames, "test")

        assert result.text == ""
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_generate_handles_missing_usage(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(url="https://api.example.com/v1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "hello"
        mock_response.usage = None

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        b._client = mock_client
        b._loaded = True

        frames = np.random.rand(1, 3, 64, 64).astype(np.float32)
        result = b.generate(frames, "test")

        assert result.text == "hello"
        assert result.prompt_tokens == 0
        assert result.completion_tokens == 0

    def test_stream_generate_wraps_generate(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(url="https://api.example.com/v1")
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "streamed text"
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 10
        mock_usage.completion_tokens = 5
        mock_response.usage = mock_usage

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        b._client = mock_client
        b._loaded = True

        frames = np.random.rand(1, 3, 64, 64).astype(np.float32)
        chunks = list(b.stream_generate(frames, "test", max_tokens=100))

        assert len(chunks) == 1
        assert chunks[0].text == "streamed text"
        assert chunks[0].finished is True
        assert chunks[0].prompt_tokens == 10
        assert chunks[0].completion_tokens == 5

    def test_health(self):
        from trio_core.backends.remote import RemoteHTTPBackend

        b = RemoteHTTPBackend(url="https://api.example.com/v1", model="qwen-vl-plus")
        h = b.health()
        assert h["backend"] == "remote"
        assert h["model"] == "qwen-vl-plus"
        assert h["loaded"] is False
        assert h["device"] == "Remote API"
        assert h["accelerator"] == "remote"
        assert h["memory_gb"] == 0
