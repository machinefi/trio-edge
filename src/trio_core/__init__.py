"""TrioCore — Portable video inference engine for VLMs."""

__version__ = "1.1.5"

from trio_core.backends import BaseBackend, MLXBackend, TransformersBackend, auto_backend
from trio_core.callbacks import CallbackMixin
from trio_core.config import EngineConfig
from trio_core.device import DeviceInfo, detect_device, recommend_model
from trio_core.engine import InferenceMetrics, TrioCore, VideoResult
from trio_core.profiles import ModelProfile, get_profile
from trio_core.video import MotionGate, StreamCapture, TemporalDeduplicator

__all__ = [
    "TrioCore",
    "EngineConfig",
    "VideoResult",
    "InferenceMetrics",
    "BaseBackend",
    "MLXBackend",
    "TransformersBackend",
    "auto_backend",
    "DeviceInfo",
    "detect_device",
    "recommend_model",
    "StreamCapture",
    "TemporalDeduplicator",
    "MotionGate",
    "ModelProfile",
    "get_profile",
    "CallbackMixin",
    "__version__",
]
