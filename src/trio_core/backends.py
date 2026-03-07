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
    """VLM backend using MLX on Apple Silicon.

    Uses mlx_vlm only for model loading (load + load_config).
    Preprocessing, generation, KV cache, and sampling are all internal.
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
        self._prompt_cache = None  # Lazily created on first generate
        self._early_stop = None   # Set via set_early_stop() after load
        self._speculative_lookahead = 0  # Set via set_speculative() after load
        self._visual_similarity_threshold = 0.0  # Set via set_visual_similarity() after load
        # Detect if model natively supports video tokens (Qwen2.5-VL, Qwen3-VL, etc.)
        # Models without video support (Gemma 3, SmolVLM2) use the image path instead.
        self._is_video_model = (
            hasattr(self._model.config, "video_token_id")
            or hasattr(self._model.config, "video_token_index")
        )
        self._loaded = True
        logger.info("[MLX] Model loaded in %.1fs (video_model=%s)", time.monotonic() - t0, self._is_video_model)

    def _get_prompt_cache(self):
        """Get or create the persistent PromptCache."""
        if self._prompt_cache is None:
            from trio_core.generate import PromptCache
            self._prompt_cache = PromptCache(self._model)
        return self._prompt_cache

    def _get_early_stop(self):
        """Get EarlyStopConfig if configured, else None."""
        return self._early_stop

    def set_early_stop(self, enabled: bool, threshold: float = 0.8) -> None:
        """Configure early stopping. Called by engine after load()."""
        if not enabled:
            self._early_stop = None
            return
        from trio_core.generate import EarlyStopConfig
        # Get EOS token IDs from model config
        eos_ids = self._model.config.eos_token_id
        if isinstance(eos_ids, int):
            eos_ids = [eos_ids]
        elif eos_ids is None:
            eos_ids = []
        self._early_stop = EarlyStopConfig(
            eos_threshold=threshold,
            eos_token_ids=list(eos_ids),
        )
        logger.info("[MLX] Early stopping enabled: threshold=%.2f, eos_ids=%s", threshold, eos_ids)

    def set_speculative(self, lookahead: int) -> None:
        """Configure speculative decoding. Called by engine after load()."""
        self._speculative_lookahead = max(0, lookahead)
        if self._speculative_lookahead > 0:
            logger.info("[MLX] Speculative decode enabled: lookahead=%d", self._speculative_lookahead)

    def set_visual_similarity(self, threshold: float) -> None:
        """Configure visual similarity KV reuse. Called by engine after load()."""
        self._visual_similarity_threshold = max(0.0, min(1.0, threshold))
        if self._visual_similarity_threshold > 0:
            logger.info(
                "[MLX] Visual similarity KV reuse enabled: threshold=%.2f",
                self._visual_similarity_threshold,
            )

    def generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> GenerationResult:
        import mlx.core as mx
        from trio_core.generate import generate_step, _wired_limit

        formatted, kwargs = self._prepare(frames, prompt)

        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        detokenizer = self._processor.detokenizer
        detokenizer.reset()

        # Reset stopping criteria
        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self._model.config.eos_token_id)

        text = ""
        prompt_tps = 0.0
        generation_tps = 0.0
        n_tokens = 0

        with _wired_limit(self._model):
            tic = time.perf_counter()
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids, self._model, pixel_values, mask,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    prompt_cache_manager=self._get_prompt_cache(),
                    early_stop=self._early_stop,
                    speculative_lookahead=self._speculative_lookahead,
                    visual_similarity_threshold=self._visual_similarity_threshold,
                    **kwargs,
                )
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = input_ids.size / max(prompt_time, 1e-9)
                    tic = time.perf_counter()

                if hasattr(tokenizer, "stopping_criteria") and tokenizer.stopping_criteria(token):
                    break

                detokenizer.add_token(token)
                n_tokens = n + 1

            detokenizer.finalize()
            text = detokenizer.text

            if n_tokens > 0:
                generation_tps = n_tokens / max(time.perf_counter() - tic, 1e-9)

        return GenerationResult(
            text=text,
            prompt_tokens=input_ids.size,
            completion_tokens=n_tokens,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else 0.0,
        )

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        import mlx.core as mx
        from trio_core.generate import generate_step, _wired_limit

        formatted, kwargs = self._prepare(frames, prompt)

        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

        tokenizer = (
            self._processor.tokenizer
            if hasattr(self._processor, "tokenizer")
            else self._processor
        )
        detokenizer = self._processor.detokenizer
        detokenizer.reset()

        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self._model.config.eos_token_id)

        with _wired_limit(self._model):
            tic = time.perf_counter()
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids, self._model, pixel_values, mask,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    prompt_cache_manager=self._get_prompt_cache(),
                    early_stop=self._early_stop,
                    speculative_lookahead=self._speculative_lookahead,
                    visual_similarity_threshold=self._visual_similarity_threshold,
                    **kwargs,
                )
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    tic = time.perf_counter()

                if hasattr(tokenizer, "stopping_criteria") and tokenizer.stopping_criteria(token):
                    break

                detokenizer.add_token(token)
                yield StreamChunk(
                    text=detokenizer.last_segment,
                    prompt_tokens=input_ids.size,
                    completion_tokens=n + 1,
                )

            detokenizer.finalize()
            if detokenizer.last_segment:
                yield StreamChunk(
                    text=detokenizer.last_segment,
                    prompt_tokens=input_ids.size,
                    completion_tokens=n + 1,
                    finished=True,
                )

            mx.clear_cache()

    def _prepare(self, frames: np.ndarray, prompt: str) -> tuple[str, dict]:
        """Route to video or image preparation based on model capability."""
        if self._is_video_model:
            return self._prepare_video(frames, prompt)
        return self._prepare_images(frames, prompt)

    def _prepare_video(self, frames: np.ndarray, prompt: str) -> tuple[str, dict]:
        """Prepare inputs via the video pipeline (Qwen2.5-VL, Qwen3-VL, etc.).

        Passes numpy frames directly to the processor as video — no temp
        file or mlx-vlm helpers needed.
        """
        import mlx.core as mx

        # Convert (T, C, H, W) float32 → list of PIL Images for video
        pil_frames = self._frames_to_pil(frames)

        # Ensure at least 2 frames (Qwen VL video requirement)
        if len(pil_frames) < 2:
            pil_frames = pil_frames + pil_frames[-1:]

        messages = [{
            "role": "user",
            "content": [
                {"type": "video", "video": pil_frames},
                {"type": "text", "text": prompt},
            ],
        }]

        formatted = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._call_processor(
            text=[formatted], videos=[pil_frames],
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

    def _prepare_images(self, frames: np.ndarray, prompt: str) -> tuple[str, dict]:
        """Prepare inputs via the image pipeline (Gemma 3, SmolVLM2, etc.).

        Converts numpy frames to PIL images and passes directly to processor.
        """
        import mlx.core as mx

        images = self._frames_to_pil(frames)

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        formatted = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

        inputs = self._call_processor(
            text=[formatted], images=images,
        )

        # Convert to MLX arrays for generate_step
        kwargs = {
            "input_ids": mx.array(np.asarray(inputs["input_ids"])),
            "pixel_values": mx.array(np.asarray(
                inputs.get("pixel_values", np.array([]))
            )),
            "mask": mx.array(np.asarray(inputs["attention_mask"])),
        }
        for key in ("image_grid_thw",):
            if key in inputs:
                kwargs[key] = mx.array(np.asarray(inputs[key]))

        return formatted, kwargs

    @staticmethod
    def _frames_to_pil(frames: np.ndarray) -> list:
        """Convert (T, C, H, W) float32 numpy to list of PIL Images."""
        from PIL import Image

        images = []
        for i in range(frames.shape[0]):
            frame = (frames[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(frame))
        return images

    def _call_processor(self, **kwargs) -> dict:
        """Call processor with numpy/PyTorch fallback."""
        try:
            inputs = self._processor(**kwargs, padding=True, return_tensors="np")
        except (ValueError, TypeError):
            inputs = self._processor(**kwargs, padding=True, return_tensors="pt")
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}
        return inputs


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
        self._is_video_model = hasattr(self._model.config, "video_token_id")
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

        if self._is_video_model:
            return self._prepare_video(frames, prompt)

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

    def _prepare_video(self, frames: np.ndarray, prompt: str) -> dict:
        """Video-capable model preparation using qwen_vl_utils."""
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

_BACKEND_MAP: dict[str, type[BaseBackend]] = {
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
