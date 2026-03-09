"""Custom processors — torch-free replacements for transformers AutoProcessor."""

from trio_core.processors.qwen_vl import load_processor

__all__ = ["load_processor"]
