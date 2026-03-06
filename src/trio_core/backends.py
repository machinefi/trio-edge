"""VLM backend abstraction — one interface, multiple runtimes.

Backends:
    MLXBackend          → mlx-vlm on Apple Silicon (Metal GPU)
    TransformersBackend → HuggingFace Transformers on CUDA / CPU

Usage:
    from trio_core.backends import auto_backend

    backend = auto_backend("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    backend.load()
    result = backend.generate(frames, prompt, max_tokens=512)
"""

from __future__ import annotations

import logging
import time
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

    def __init__(self, model_name: str, device_info: DeviceInfo | None = None):
        self.model_name = model_name
        self.device_info = device_info or detect_device()
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


# ── MLX Backend ──────────────────────────────────────────────────────────────


class MLXBackend(BaseBackend):
    """VLM backend using mlx-vlm on Apple Silicon.

    Uses mlx_vlm's video_generate pipeline which handles:
    - process_vision_info: extracts video/image from messages
    - fetch_video: loads and resizes frames with smart_resize
    - process_inputs_with_fallback: converts to MLX tensors
    - generate/stream_generate: KV cache, token sampling
    """

    @property
    def backend_name(self) -> str:
        return "mlx"

    def load(self) -> None:
        from mlx_vlm import load
        from mlx_vlm.utils import load_config

        logger.info("[MLX] Loading model: %s", self.model_name)
        t0 = time.monotonic()
        self._model, self._processor = load(self.model_name)
        self._config = load_config(self.model_name)
        self._loaded = True
        logger.info("[MLX] Model loaded in %.1fs", time.monotonic() - t0)

    def generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> GenerationResult:
        formatted, kwargs = self._prepare(frames, prompt)

        from mlx_vlm import generate as mlx_generate
        result = mlx_generate(
            self._model, self._processor, formatted,
            max_tokens=max_tokens, temp=temperature, top_p=top_p, verbose=False,
            **kwargs,
        )

        return GenerationResult(
            text=result.text,
            prompt_tokens=getattr(result, "prompt_tokens", 0),
            completion_tokens=getattr(result, "generation_tokens", 0),
            prompt_tps=getattr(result, "prompt_tps", 0.0),
            generation_tps=getattr(result, "generation_tps", 0.0),
            peak_memory=getattr(result, "peak_memory", 0.0),
        )

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        formatted, kwargs = self._prepare(frames, prompt)

        from mlx_vlm import stream_generate as mlx_stream
        for chunk in mlx_stream(
            self._model, self._processor, formatted,
            max_tokens=max_tokens, temp=temperature,
            **kwargs,
        ):
            yield StreamChunk(
                text=chunk.text,
                prompt_tokens=getattr(chunk, "prompt_tokens", 0),
                completion_tokens=getattr(chunk, "generation_tokens", 0),
            )

    def _prepare(self, frames: np.ndarray, prompt: str) -> tuple[str, dict]:
        """Prepare prompt and pre-processed video tensors for mlx_vlm generate.

        Saves numpy frames as temp video, processes through mlx_vlm's
        video pipeline (process_vision_info → fetch_video → processor),
        then converts to MLX arrays.
        """
        import mlx.core as mx

        video_path = self._frames_to_temp_video(frames)

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }]

        formatted = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        # Process video through mlx_vlm's video pipeline
        from mlx_vlm.video_generate import process_vision_info, process_inputs_with_fallback
        image_inputs, video_inputs, _ = process_vision_info(
            messages, return_video_kwargs=True,
        )

        inputs = process_inputs_with_fallback(
            self._processor,
            prompts=formatted,
            images=image_inputs,
            audio=None,
            videos=video_inputs,
            return_tensors="np",
        )

        # Convert to MLX arrays for generate_step
        kwargs = {
            "input_ids": mx.array(np.asarray(inputs["input_ids"])),
            "pixel_values": mx.array(np.asarray(
                inputs.get("pixel_values_videos", inputs.get("pixel_values", np.array([])))
            )),
            "mask": mx.array(np.asarray(inputs["attention_mask"])),
        }
        for key in ("video_grid_thw", "image_grid_thw", "second_per_grid_ts"):
            if key in inputs:
                kwargs[key] = mx.array(np.asarray(inputs[key]))

        return formatted, kwargs

    @staticmethod
    def _frames_to_temp_video(frames: np.ndarray) -> str:
        """Save (T, C, H, W) float32 frames as a temp .mp4 for mlx_vlm pipeline."""
        import tempfile
        import cv2

        t, c, h, w = frames.shape
        # mlx_vlm requires at least 2 frames in a video
        if t < 2:
            frames = np.concatenate([frames, frames[-1:]], axis=0)
            t = frames.shape[0]
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        writer = cv2.VideoWriter(
            tmp.name, cv2.VideoWriter_fourcc(*"mp4v"), 2.0, (w, h),
        )
        for i in range(t):
            # (C, H, W) float32 [0,1] → (H, W, C) uint8 BGR
            frame = (frames[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()

        from trio_core.video import _TEMP_FILES
        _TEMP_FILES.append(tmp.name)
        return tmp.name


# ── Transformers Backend ─────────────────────────────────────────────────────


class TransformersBackend(BaseBackend):
    """VLM backend using HuggingFace Transformers on CUDA / CPU.

    Supports any VLM loadable via AutoModelForVision2Seq, including:
    Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 3, SmolVLM, Phi-4, InternVL, etc.
    """

    @property
    def backend_name(self) -> str:
        return "transformers"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForVision2Seq, AutoProcessor

        device = "cuda" if self.device_info.accelerator == "cuda" else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("[Transformers] Loading model: %s (device=%s)", self.model_name, device)
        t0 = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_name, torch_dtype=dtype, device_map=device,
        )
        self._device = device
        self._is_qwen = "qwen" in self.model_name.lower()
        self._loaded = True
        logger.info("[Transformers] Model loaded in %.1fs", time.monotonic() - t0)

    def generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> GenerationResult:
        import torch

        inputs = self._prepare(frames, prompt)

        t0 = time.monotonic()
        prompt_len = inputs["input_ids"].shape[1]
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                do_sample=temperature > 0,
            )

        new_tokens = outputs[0][prompt_len:]
        text = self._processor.decode(new_tokens, skip_special_tokens=True)

        elapsed = time.monotonic() - t0
        n_tokens = len(new_tokens)

        return GenerationResult(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=n_tokens,
            generation_tps=n_tokens / max(elapsed, 1e-6),
            peak_memory=0.0,
        )

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        from transformers import TextIteratorStreamer
        import torch
        import threading

        inputs = self._prepare(frames, prompt)

        streamer = TextIteratorStreamer(self._processor, skip_prompt=True, skip_special_tokens=True)

        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=max(temperature, 1e-6),
            top_p=top_p,
            do_sample=temperature > 0,
            streamer=streamer,
        )

        thread = threading.Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        for text in streamer:
            yield StreamChunk(text=text)

        thread.join(timeout=300)  # 5 min safety timeout

    def _prepare(self, frames: np.ndarray, prompt: str) -> dict:
        """Prepare inputs for generate().

        Qwen models use qwen_vl_utils for video processing.
        All other models get frames converted to PIL images.
        Returns a dict of tensors ready to pass to model.generate(**inputs).
        """
        import torch
        from PIL import Image

        if self._is_qwen:
            return self._prepare_qwen(frames, prompt)

        # Generic path: convert frames to PIL images
        images = self._frames_to_pil(frames)

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text], images=images,
            padding=True, return_tensors="pt",
        )

        return {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _prepare_qwen(self, frames: np.ndarray, prompt: str) -> dict:
        """Qwen-specific preparation using qwen_vl_utils."""
        import torch

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": frames},
                {"type": "text", "text": prompt},
            ],
        }]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        try:
            from qwen_vl_utils import process_vision_info
            image_inputs, video_inputs, _ = process_vision_info(
                messages, return_video_kwargs=True
            )
        except ImportError:
            video_inputs = [frames]
            image_inputs = None

        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        )

        return {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

    @staticmethod
    def _frames_to_pil(frames: np.ndarray) -> list:
        """Convert (T, C, H, W) float32 numpy to list of PIL Images."""
        from PIL import Image

        images = []
        for i in range(frames.shape[0]):
            # (C, H, W) float32 [0,1] → (H, W, C) uint8
            frame = (frames[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame))
        return images


# ── Auto Backend Selection ───────────────────────────────────────────────────

_BACKEND_MAP = {
    "mlx": MLXBackend,
    "transformers": TransformersBackend,
}


def auto_backend(
    model_name: str,
    *,
    backend: str | None = None,
    device_info: DeviceInfo | None = None,
) -> BaseBackend:
    """Create the best backend for the current hardware.

    Args:
        model_name: HuggingFace model ID.
        backend: Force a specific backend ("mlx" or "transformers").
                 If None, auto-detect from hardware.
        device_info: Pre-detected device info. If None, auto-detect.

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
        chosen, device_info.device_name, device_info.accelerator, device_info.memory_gb,
    )
    return cls(model_name, device_info=device_info)
