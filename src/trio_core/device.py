"""Hardware detection and backend selection.

Auto-detects the best inference backend for the current hardware:
    Apple Silicon (M1/M2/M3/M4) → MLX
    NVIDIA GPU (CUDA)           → Transformers (PyTorch + CUDA)
    CPU-only                    → Transformers (PyTorch CPU)

Usage:
    from trio_core.device import detect_device, DeviceInfo

    info = detect_device()
    print(info)  # DeviceInfo(backend='mlx', device='Apple M3 Max', ...)
"""

from __future__ import annotations

import logging
import platform
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Detected hardware and recommended backend."""

    backend: str          # "mlx", "transformers", "cpu"
    device_name: str      # "Apple M3 Max", "NVIDIA RTX 4090", "CPU"
    accelerator: str      # "metal", "cuda", "cpu"
    memory_gb: float      # Available device memory in GB (0 if unknown)
    compute_units: int    # GPU cores / CUDA cores (0 if unknown)

    @property
    def has_gpu(self) -> bool:
        return self.accelerator in ("metal", "cuda")


def detect_device() -> DeviceInfo:
    """Auto-detect hardware and recommend the best backend.

    Priority: MLX (Apple Silicon) > CUDA (NVIDIA) > CPU
    """
    # 1. Check Apple Silicon (MLX)
    info = _detect_apple_silicon()
    if info is not None:
        return info

    # 2. Check NVIDIA GPU (CUDA)
    info = _detect_nvidia()
    if info is not None:
        return info

    # 3. Fallback: CPU
    return DeviceInfo(
        backend="transformers",
        device_name=platform.processor() or "CPU",
        accelerator="cpu",
        memory_gb=0,
        compute_units=0,
    )


def _detect_apple_silicon() -> DeviceInfo | None:
    """Detect Apple Silicon and check if MLX is available."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return None

    # Get chip name via sysctl
    chip_name = "Apple Silicon"
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            chip_name = result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Get unified memory
    memory_gb = 0.0
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            memory_gb = int(result.stdout.strip()) / (1024 ** 3)
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Get GPU core count
    cores = 0
    try:
        result = subprocess.run(
            ["system_profiler", "SPDisplaysDataType", "-json"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            displays = data.get("SPDisplaysDataType", [])
            for d in displays:
                name = d.get("sppci_model", "")
                if "Apple" in name:
                    cores_str = d.get("sppci_cores", "0")
                    cores = int(cores_str) if cores_str.isdigit() else 0
                    break
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    # Check if MLX is importable
    backend = "transformers"
    try:
        import mlx.core  # noqa: F401
        backend = "mlx"
    except ImportError:
        logger.info("Apple Silicon detected but mlx not installed, falling back to transformers")

    return DeviceInfo(
        backend=backend,
        device_name=chip_name,
        accelerator="metal",
        memory_gb=memory_gb,
        compute_units=cores,
    )


def _detect_nvidia() -> DeviceInfo | None:
    """Detect NVIDIA GPU via nvidia-smi or torch.cuda."""
    # Try nvidia-smi first (no Python deps)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(",")
            name = parts[0].strip()
            mem_mb = float(parts[1].strip()) if len(parts) > 1 else 0

            return DeviceInfo(
                backend="transformers",
                device_name=name,
                accelerator="cuda",
                memory_gb=mem_mb / 1024,
                compute_units=0,
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try torch.cuda
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return DeviceInfo(
                backend="transformers",
                device_name=name,
                accelerator="cuda",
                memory_gb=mem,
                compute_units=0,
            )
    except ImportError:
        pass

    return None


def recommend_model(info: DeviceInfo) -> str:
    """Recommend a default model based on detected hardware."""
    if info.backend == "mlx":
        if info.memory_gb >= 32:
            return "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"
        return "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"

    # Transformers (CUDA or CPU)
    if info.accelerator == "cuda":
        if info.memory_gb >= 16:
            return "Qwen/Qwen2.5-VL-7B-Instruct"
        return "Qwen/Qwen2.5-VL-3B-Instruct"

    # CPU — smallest model
    return "Qwen/Qwen2.5-VL-3B-Instruct"
