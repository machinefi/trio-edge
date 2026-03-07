"""Native model loader — replaces mlx_vlm.load() for supported models.

Usage:
    from trio_core.models.loader import load_native
    model, processor = load_native("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
"""

from __future__ import annotations

import glob
import json
import logging
from pathlib import Path
from typing import Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)

# Model type → module mapping
_MODEL_REGISTRY = {
    "qwen2_5_vl": "trio_core.models.qwen2_5_vl",
    "qwen3_vl": "trio_core.models.qwen3_vl",
}


def load_native(
    path_or_hf_repo: str,
    lazy: bool = False,
) -> Tuple[nn.Module, any]:
    """Load model and processor without mlx-vlm dependency.

    Args:
        path_or_hf_repo: HuggingFace model ID or local path.
        lazy: If True, don't evaluate parameters immediately.

    Returns:
        (model, processor) tuple. Processor is from transformers AutoProcessor.

    Raises:
        ValueError: If model type is not supported natively.
    """
    import huggingface_hub
    from transformers import AutoProcessor

    # Download model files
    model_path = Path(huggingface_hub.snapshot_download(path_or_hf_repo))
    logger.info("Loading native model from %s", model_path)

    # Read config
    config_path = model_path / "config.json"
    with open(config_path) as f:
        config = json.load(f)

    model_type = config.get("model_type", "").lower()
    if model_type not in _MODEL_REGISTRY:
        supported = ", ".join(_MODEL_REGISTRY.keys())
        raise ValueError(
            f"Model type '{model_type}' not supported for native loading. "
            f"Supported: {supported}. Use mlx-vlm fallback."
        )

    # Import model module
    import importlib
    arch = importlib.import_module(_MODEL_REGISTRY[model_type])

    # Build config
    model_config = arch.ModelConfig.from_dict(config)

    # Update sub-configs
    if hasattr(arch, "VisionConfig") and "vision_config" in config:
        model_config.vision_config = arch.VisionConfig.from_dict(config["vision_config"])
    if hasattr(arch, "TextConfig"):
        if "text_config" in config and isinstance(config["text_config"], dict):
            # Qwen3-VL style: text config in dedicated sub-dict
            model_config.text_config = arch.TextConfig.from_dict(config["text_config"])
        else:
            # Qwen2.5-VL style: text config from root-level params
            model_config.text_config = arch.TextConfig.from_dict(
                {k: v for k, v in config.items() if k != "vision_config"}
            )

    # Instantiate model
    model = arch.Model(model_config)

    # Load weights from safetensors
    weight_files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Check if already in MLX format (has metadata marker)
    is_mlx_format = False
    try:
        from safetensors import safe_open
        with safe_open(weight_files[0], framework="numpy") as f:
            meta = f.metadata()
            is_mlx_format = meta is not None and meta.get("format") == "mlx"
    except (ImportError, Exception):
        # If safetensors metadata check fails, try sanitizing anyway
        pass

    # Sanitize weights if not in MLX format
    if not is_mlx_format:
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)
        if hasattr(model.vision_tower, "sanitize"):
            weights = model.vision_tower.sanitize(weights)

    # Apply quantization if specified in config
    quantization = config.get("quantization", None)
    if quantization is not None:
        skip_vision = config.get("vision_config", {}).get("skip_vision", False)

        def get_class_predicate(p, m):
            if skip_vision and _skip_multimodal_module(p):
                return False
            if p in quantization:
                return quantization[p]
            if not hasattr(m, "to_quantized"):
                return False
            if hasattr(m, "weight") and m.weight.size % 64 != 0:
                return False
            return f"{p}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=get_class_predicate,
        )

    # Load weights into model
    model.load_weights(list(weights.items()))

    if not lazy:
        mx.eval(model.parameters())

    model.eval()

    # Load processor (from transformers, not mlx-vlm)
    eos_token_id = getattr(model.config, "eos_token_id", None)
    processor = AutoProcessor.from_pretrained(str(model_path), use_fast=True)

    logger.info(
        "Native model loaded: type=%s, quantization=%s",
        model_type, quantization,
    )

    return model, processor


def _skip_multimodal_module(path: str) -> bool:
    """Check if a module path belongs to vision/audio (should skip quantization)."""
    skip_prefixes = ("vision_tower", "visual", "vision_model", "audio")
    return any(path.startswith(p) or f".{p}" in path for p in skip_prefixes)


def load_config_native(path_or_hf_repo: str) -> dict:
    """Load model config.json without mlx-vlm dependency."""
    import huggingface_hub
    model_path = Path(huggingface_hub.snapshot_download(path_or_hf_repo))
    with open(model_path / "config.json") as f:
        return json.load(f)
