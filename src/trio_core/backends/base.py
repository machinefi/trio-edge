"""Base classes for VLM backend abstraction.

Defines the abstract interface (BaseBackend) and shared data types
(GenerationResult, StreamChunk) used by all concrete backends.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generator

import numpy as np

from trio_core.device import DeviceInfo, detect_device

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Unified generation result across all backends."""

    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0


@dataclass
class StreamChunk:
    """A single chunk from streaming generation."""

    text: str
    finished: bool = False
    prompt_tokens: int = 0
    completion_tokens: int = 0


class BaseBackend(ABC):
    """Abstract VLM inference backend.

    Subclasses must implement load(), generate(), and stream_generate().
    The engine calls these methods — it never touches mlx or torch directly.
    """

    def __init__(
        self,
        model_name: str,
        device_info: DeviceInfo | None = None,
        adapter_path: str | None = None,
    ):
        self.model_name = model_name
        self.device_info = device_info or detect_device()
        self.adapter_path = adapter_path
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Human-readable backend name."""
        ...

    @abstractmethod
    def load(self) -> None:
        """Load model and processor. Must set self._loaded = True."""
        ...

    @abstractmethod
    def generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Run inference on video frames.

        Args:
            frames: (T, C, H, W) float32 numpy array.
            prompt: User prompt / question about the video.

        Returns:
            GenerationResult with text and metrics.
        """
        ...

    @abstractmethod
    def stream_generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        """Stream inference token by token."""
        ...

    def health(self) -> dict:
        return {
            "backend": self.backend_name,
            "model": self.model_name,
            "loaded": self._loaded,
            "device": self.device_info.device_name,
            "accelerator": self.device_info.accelerator,
            "memory_gb": self.device_info.memory_gb,
        }

    @staticmethod
    def _frames_to_pil(frames: np.ndarray) -> list:
        """Convert (T, C, H, W) float32 numpy to list of PIL Images."""
        from PIL import Image

        images = []
        for i in range(frames.shape[0]):
            frame = (frames[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame))
        return images


# ── Shared token handling ────────────────────────────────────────────────────


class _TokenHandler:
    """Unified tokenizer init, EOS detection, and incremental detokenization.

    Eliminates repeated boilerplate across generate/stream/ar decode loops.
    """

    def __init__(self, processor, model_config):
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self._has_detokenizer = hasattr(processor, "detokenizer")
        if self._has_detokenizer:
            self._detokenizer = processor.detokenizer
            self._detokenizer.reset()

        if hasattr(self.tokenizer, "stopping_criteria"):
            self.tokenizer.stopping_criteria.reset(model_config.eos_token_id)

        self._eos_token_id = getattr(model_config, "eos_token_id", None)
        self._token_ids: list[int] = []
        self._prev_text = ""

    def should_stop(self, token: int) -> bool:
        """Check stopping criteria and EOS."""
        if hasattr(self.tokenizer, "stopping_criteria") and self.tokenizer.stopping_criteria(token):
            return True
        if self._eos_token_id is not None and not self._has_detokenizer:
            if isinstance(self._eos_token_id, list):
                return token in self._eos_token_id
            return token == self._eos_token_id
        return False

    def add_token(self, token: int) -> str:
        """Add token and return incremental text delta.

        For non-streaming callers the delta can be ignored — finalize()
        does a single full decode at the end, avoiding O(n²) repeated
        decodes in the non-detokenizer path.
        """
        if self._has_detokenizer:
            self._detokenizer.add_token(token)
            return self._detokenizer.last_segment
        self._token_ids.append(token)
        return ""

    def add_token_streaming(self, token: int) -> str:
        """Add token and return incremental text delta (streaming variant).

        Same as add_token() but always computes the delta text for
        immediate yield. For the non-detokenizer fallback path this
        decodes all accumulated tokens each call (O(n²) total, but
        this path is rare — most MLX models have a detokenizer).
        """
        if self._has_detokenizer:
            self._detokenizer.add_token(token)
            return self._detokenizer.last_segment
        self._token_ids.append(token)
        new_text = self.tokenizer.decode(self._token_ids, skip_special_tokens=True)
        delta = new_text[len(self._prev_text) :]
        self._prev_text = new_text
        return delta

    def finalize(self) -> str:
        """Return complete decoded text."""
        if self._has_detokenizer:
            self._detokenizer.finalize()
            return self._detokenizer.text
        return self.tokenizer.decode(self._token_ids, skip_special_tokens=True)

    def finalize_delta(self) -> str:
        """Return any remaining text not yet yielded (for streaming)."""
        if self._has_detokenizer:
            self._detokenizer.finalize()
            return self._detokenizer.last_segment
        return ""
