"""Tests for trio_core.device — hardware detection."""

import json
import platform
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from trio_core.device import (
    DeviceInfo,
    _detect_apple_silicon,
    _detect_nvidia,
    detect_device,
    recommend_model,
)


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
        assert "mlx-community" in model

    def test_cuda_high_memory(self):
        info = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        model = recommend_model(info)
        assert "8B" in model
        assert "Qwen/" in model

    def test_cpu_fallback(self):
        info = DeviceInfo("transformers", "CPU", "cpu", 0, 0)
        model = recommend_model(info)
        assert "2B" in model
        assert "Qwen/" in model

    def test_cuda_low_memory(self):
        info = DeviceInfo("transformers", "RTX 3060", "cuda", 12.0, 0)
        model = recommend_model(info)
        assert "2B" in model
        assert "Qwen/" in model


class TestDetectAppleSilicon:
    """Tests for _detect_apple_silicon with mocked platform and subprocess."""

    @patch("trio_core.device.platform")
    def test_returns_none_on_non_darwin(self, mock_platform):
        """Line 68: non-Darwin platform returns None immediately."""
        mock_platform.system.return_value = "Linux"
        mock_platform.machine.return_value = "x86_64"
        assert _detect_apple_silicon() is None

    @patch("trio_core.device.platform")
    def test_returns_none_on_darwin_x86(self, mock_platform):
        """Line 68: Darwin but x86_64 returns None."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "x86_64"
        assert _detect_apple_silicon() is None

    @patch("trio_core.device.platform")
    @patch("trio_core.device.subprocess.run")
    def test_full_apple_silicon_detection(self, mock_run, mock_platform):
        """Lines 51-56, 71-128: full successful Apple Silicon detection."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        profiler_data = {
            "SPDisplaysDataType": [
                {
                    "sppci_model": "Apple M3 Max",
                    "sppci_cores": "40",
                }
            ]
        }

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if cmd[1:] == ["-n", "machdep.cpu.brand_string"]:
                result.stdout = "Apple M3 Max\n"
            elif cmd[1:] == ["-n", "hw.memsize"]:
                result.stdout = str(64 * 1024**3) + "\n"
            elif cmd[0] == "system_profiler":
                result.stdout = json.dumps(profiler_data)
            return result

        mock_run.side_effect = side_effect

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            info = _detect_apple_silicon()

        assert info is not None
        assert info.backend == "mlx"
        assert info.device_name == "Apple M3 Max"
        assert info.accelerator == "metal"
        assert info.memory_gb == 64.0
        assert info.compute_units == 40

    @patch("trio_core.device.platform")
    @patch("trio_core.device.subprocess.run")
    def test_sysctl_chip_timeout(self, mock_run, mock_platform):
        """Lines 79-80: sysctl for chip name times out, fallback to default."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        def side_effect(cmd, **kwargs):
            if "machdep.cpu.brand_string" in cmd:
                raise subprocess.TimeoutExpired(cmd, 5)
            result = MagicMock()
            result.returncode = 0
            if "hw.memsize" in cmd:
                result.stdout = str(16 * 1024**3) + "\n"
            elif cmd[0] == "system_profiler":
                result.stdout = json.dumps({"SPDisplaysDataType": []})
            return result

        mock_run.side_effect = side_effect

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            info = _detect_apple_silicon()

        assert info is not None
        assert info.device_name == "Apple Silicon"  # fallback name

    @patch("trio_core.device.platform")
    @patch("trio_core.device.subprocess.run")
    def test_memsize_timeout(self, mock_run, mock_platform):
        """Lines 91-92: hw.memsize times out, memory stays 0."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        def side_effect(cmd, **kwargs):
            if "hw.memsize" in cmd:
                raise subprocess.TimeoutExpired(cmd, 5)
            result = MagicMock()
            result.returncode = 0
            if "machdep.cpu.brand_string" in cmd:
                result.stdout = "Apple M2\n"
            elif cmd[0] == "system_profiler":
                result.stdout = json.dumps({"SPDisplaysDataType": []})
            return result

        mock_run.side_effect = side_effect

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            info = _detect_apple_silicon()

        assert info is not None
        assert info.memory_gb == 0.0

    @patch("trio_core.device.platform")
    @patch("trio_core.device.subprocess.run")
    def test_system_profiler_timeout(self, mock_run, mock_platform):
        """Lines 111-112: system_profiler times out, cores stays 0."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        def side_effect(cmd, **kwargs):
            if cmd[0] == "system_profiler":
                raise subprocess.TimeoutExpired(cmd, 10)
            result = MagicMock()
            result.returncode = 0
            if "machdep.cpu.brand_string" in cmd:
                result.stdout = "Apple M2\n"
            elif "hw.memsize" in cmd:
                result.stdout = str(8 * 1024**3) + "\n"
            return result

        mock_run.side_effect = side_effect

        with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": MagicMock()}):
            info = _detect_apple_silicon()

        assert info is not None
        assert info.compute_units == 0

    @patch("trio_core.device.platform")
    @patch("trio_core.device.subprocess.run")
    def test_mlx_not_available(self, mock_run, mock_platform):
        """Lines 119-120: mlx import fails, backend falls back to transformers."""
        mock_platform.system.return_value = "Darwin"
        mock_platform.machine.return_value = "arm64"

        def side_effect(cmd, **kwargs):
            result = MagicMock()
            result.returncode = 0
            if "machdep.cpu.brand_string" in cmd:
                result.stdout = "Apple M1\n"
            elif "hw.memsize" in cmd:
                result.stdout = str(8 * 1024**3) + "\n"
            elif cmd[0] == "system_profiler":
                result.stdout = json.dumps({"SPDisplaysDataType": []})
            return result

        mock_run.side_effect = side_effect

        # Ensure mlx.core is NOT importable by removing it and patching import
        with patch.dict("sys.modules", {"mlx": None, "mlx.core": None}):
            # Patch builtins.__import__ to raise ImportError for mlx.core
            original_import = (
                __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__
            )

            def mock_import(name, *args, **kwargs):
                if name == "mlx.core" or name == "mlx":
                    raise ImportError("No module named 'mlx'")
                return original_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                info = _detect_apple_silicon()

        assert info is not None
        assert info.backend == "transformers"
        assert info.accelerator == "metal"


class TestDetectNvidia:
    """Tests for _detect_nvidia with mocked subprocess and torch."""

    @patch("trio_core.device.subprocess.run")
    def test_nvidia_smi_success(self, mock_run):
        """Lines 134-151: nvidia-smi returns valid GPU info."""
        result = MagicMock()
        result.returncode = 0
        result.stdout = "NVIDIA GeForce RTX 4090, 24564\n"
        mock_run.return_value = result

        info = _detect_nvidia()

        assert info is not None
        assert info.backend == "transformers"
        assert info.device_name == "NVIDIA GeForce RTX 4090"
        assert info.accelerator == "cuda"
        assert abs(info.memory_gb - 24564 / 1024) < 0.01
        assert info.compute_units == 0

    @patch("trio_core.device.subprocess.run")
    def test_nvidia_smi_multi_gpu(self, mock_run):
        """Lines 140: multi-GPU, picks first line."""
        result = MagicMock()
        result.returncode = 0
        result.stdout = "NVIDIA A100, 81920\nNVIDIA A100, 81920\n"
        mock_run.return_value = result

        info = _detect_nvidia()

        assert info is not None
        assert info.device_name == "NVIDIA A100"

    @patch("trio_core.device.subprocess.run")
    def test_nvidia_smi_timeout(self, mock_run):
        """Lines 152-153: nvidia-smi times out, falls through to torch."""
        mock_run.side_effect = subprocess.TimeoutExpired(["nvidia-smi"], 5)

        # Also no torch.cuda
        with patch.dict("sys.modules", {"torch": None}):

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                info = _detect_nvidia()

        assert info is None

    @patch("trio_core.device.subprocess.run")
    def test_nvidia_smi_not_found_torch_cuda_fallback(self, mock_run):
        """Lines 152-153, 156-167, 185: nvidia-smi not found, torch.cuda works."""
        mock_run.side_effect = FileNotFoundError()

        mock_props = MagicMock()
        mock_props.total_memory = 12 * (1024**3)  # 12 GB

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_name.return_value = "NVIDIA RTX 3060"
        mock_torch.cuda.get_device_properties.return_value = mock_props

        with patch.dict("sys.modules", {"torch": mock_torch}):
            info = _detect_nvidia()

        assert info is not None
        assert info.device_name == "NVIDIA RTX 3060"
        assert info.accelerator == "cuda"
        assert info.memory_gb == 12.0

    @patch("trio_core.device.subprocess.run")
    def test_nvidia_smi_not_found_no_torch(self, mock_run):
        """Line 168-171: nvidia-smi not found, torch not installed, returns None."""
        mock_run.side_effect = FileNotFoundError()

        with patch.dict("sys.modules", {"torch": None}):

            def mock_import(name, *args, **kwargs):
                if name == "torch":
                    raise ImportError("No module named 'torch'")
                return __import__(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                info = _detect_nvidia()

        assert info is None


class TestDetectDeviceIntegration:
    """Tests for detect_device with mocked internals."""

    @patch("trio_core.device._detect_apple_silicon", return_value=None)
    @patch("trio_core.device._detect_nvidia", return_value=None)
    @patch("trio_core.device.platform")
    def test_cpu_fallback(self, mock_platform, mock_nvidia, mock_apple):
        """Lines 51-56: no Apple Silicon, no NVIDIA → CPU fallback."""
        mock_platform.processor.return_value = "x86_64"
        info = detect_device()
        assert info.backend == "transformers"
        assert info.accelerator == "cpu"
        assert info.device_name == "x86_64"
        assert info.memory_gb == 0
        assert info.compute_units == 0

    @patch("trio_core.device._detect_apple_silicon", return_value=None)
    @patch("trio_core.device._detect_nvidia")
    def test_nvidia_path(self, mock_nvidia, mock_apple):
        """Lines 51-53: Apple Silicon returns None, NVIDIA detected."""
        mock_nvidia.return_value = DeviceInfo("transformers", "RTX 4090", "cuda", 24.0, 0)
        info = detect_device()
        assert info.accelerator == "cuda"
        assert info.device_name == "RTX 4090"

    @patch("trio_core.device._detect_apple_silicon")
    def test_apple_silicon_path(self, mock_apple):
        """Lines 46-48: Apple Silicon detected first, skips NVIDIA."""
        mock_apple.return_value = DeviceInfo("mlx", "Apple M3", "metal", 36.0, 30)
        info = detect_device()
        assert info.backend == "mlx"
        assert info.accelerator == "metal"

    @patch("trio_core.device._detect_apple_silicon", return_value=None)
    @patch("trio_core.device._detect_nvidia", return_value=None)
    @patch("trio_core.device.platform")
    def test_cpu_fallback_empty_processor(self, mock_platform, mock_nvidia, mock_apple):
        """Line 58: platform.processor() returns empty string → 'CPU'."""
        mock_platform.processor.return_value = ""
        info = detect_device()
        assert info.device_name == "CPU"
