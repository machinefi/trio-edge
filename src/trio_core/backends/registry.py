"""Backend registry — auto-selection, resolution, and plugin registration."""

from __future__ import annotations

import logging

from trio_core.backends.base import BaseBackend
from trio_core.backends.mlx import MLXBackend
from trio_core.backends.transformers import TransformersBackend
from trio_core.device import DeviceInfo, detect_device

logger = logging.getLogger(__name__)

_BACKEND_MAP: dict[str, type[BaseBackend]] = {
    "mlx": MLXBackend,
    "transformers": TransformersBackend,
}


def register_backend(name: str, cls: type[BaseBackend]) -> None:
    """Register a backend class by name (e.g. from a plugin)."""
    _BACKEND_MAP[name] = cls


def auto_backend(
    model_name: str,
    *,
    backend: str | None = None,
    device_info: DeviceInfo | None = None,
    adapter_path: str | None = None,
) -> BaseBackend:
    """Create the best backend for the current hardware.

    Args:
        model_name: HuggingFace model ID.
        backend: Force a specific backend ("mlx" or "transformers").
                 If None, auto-detect from hardware.
        device_info: Pre-detected device info. If None, auto-detect.
        adapter_path: Path to LoRA adapter directory (optional).

    Returns:
        Configured (but not loaded) backend instance.
    """
    if device_info is None:
        device_info = detect_device()

    chosen = backend or device_info.backend

    if chosen not in _BACKEND_MAP:
        logger.warning("Unknown backend '%s', falling back to transformers", chosen)
        chosen = "transformers"

    cls = _BACKEND_MAP[chosen]
    logger.info(
        "Auto-selected backend: %s (device=%s, accelerator=%s, memory=%.1fGB)",
        chosen,
        device_info.device_name,
        device_info.accelerator,
        device_info.memory_gb,
    )
    return cls(model_name, device_info=device_info, adapter_path=adapter_path)


def resolve_backend(config, *, backend_override: str | None = None) -> BaseBackend:
    """Create the right backend from an EngineConfig.

    Handles feature flags (ToMe, Compressed) so the engine
    doesn't need to know about backend subclasses.

    Args:
        config: EngineConfig instance.
        backend_override: Force "mlx" or "transformers". None = auto.

    Returns:
        Configured (but not loaded) backend instance.
    """
    base = auto_backend(
        config.model, backend=backend_override, adapter_path=getattr(config, "adapter_path", None)
    )

    if base.backend_name != "mlx":
        return base

    # Compress is exclusive to ToMe (no compound implementation)
    if config.compress_enabled and config.tome_enabled:
        logger.warning(
            "compress_enabled ignored: tome takes priority (no compound mode).",
        )

    # ToMe
    if config.tome_enabled:
        from trio_core.backends.tome import ToMeMLXBackend

        return ToMeMLXBackend(
            config.model,
            tome_r=config.tome_r,
            metric=config.tome_metric,
            min_keep_ratio=config.tome_min_keep_ratio,
            adaptive=config.tome_adaptive,
            content_aware=config.tome_content_aware,
            device_info=base.device_info,
            adapter_path=base.adapter_path,
        )

    # Compressed visual tokens
    if config.compress_enabled:
        from trio_core.backends.compressed import CompressedMLXBackend
        from trio_core.token_compression import TokenCompressor

        compressor = TokenCompressor(
            strategy="similarity",
            ratio=config.compress_ratio,
        )
        return CompressedMLXBackend(
            config.model,
            compressor,
            device_info=base.device_info,
            adapter_path=base.adapter_path,
        )

    return base
