"""Tests for trio_core.device — hardware detection."""

import platform

from trio_core.device import detect_device, recommend_model, DeviceInfo


class TestDetectDevice:
    def test_returns_device_info(self):
        info = detect_device()
        assert isinstance(info, DeviceInfo)
        assert info.backend in ("mlx", "transformers")
        assert info.accelerator in ("metal", "cuda", "cpu")

    def test_apple_silicon_detected_on_mac(self):
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info = detect_device()
            assert info.accelerator == "metal"
            assert info.memory_gb > 0
            assert "Apple" in info.device_name or "arm" in info.device_name

    def test_has_gpu_property(self):
        metal = DeviceInfo("mlx", "M3", "metal", 36.0, 40)
        assert metal.has_gpu is True
        cpu = DeviceInfo("transformers", "CPU", "cpu", 0, 0)
        assert cpu.has_gpu is False


class TestRecommendModel:
    def test_mlx_high_memory(self):
        info = DeviceInfo("mlx", "M3 Max", "metal", 64.0, 40)
        model = recommend_model(info)
        assert "7B" in model
        assert "mlx-community" in model

    def test_mlx_low_memory(self):
        info = DeviceInfo("mlx", "M1", "metal", 8.0, 8)
        model = recommend_model(info)
        assert "3B" in model

    def test_cuda_high_memory(self):
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        model = recommend_model(info)
        assert "7B" in model
        assert "Qwen/" in model

    def test_cpu_fallback(self):
        info = DeviceInfo("transformers", "CPU", "cpu", 0, 0)
        model = recommend_model(info)
        assert "3B" in model
