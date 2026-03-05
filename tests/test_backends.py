"""Tests for trio_core.backends — backend abstraction."""

from unittest.mock import patch, MagicMock

from trio_core.backends import (
    auto_backend,
    BaseBackend,
    MLXBackend,
    TransformersBackend,
    GenerationResult,
    StreamChunk,
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


class TestBaseBackendHealth:
    def test_health(self):
        info = DeviceInfo("mlx", "M3", "metal", 36.0, 40)
        b = MLXBackend("test-model", device_info=info)
        h = b.health()
        assert h["backend"] == "mlx"
        assert h["loaded"] is False
        assert h["device"] == "M3"
