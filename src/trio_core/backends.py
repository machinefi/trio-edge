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
        logger.info("[MLX] Loading model: %s", self.model_name)
        t0 = time.monotonic()

        # Try native loading first (T1 models), fall back to mlx-vlm (T2)
        try:
            from trio_core.models.loader import load_native, load_config_native
            self._model, self._processor = load_native(self.model_name)
            self._config = load_config_native(self.model_name)
            logger.info("[MLX] Loaded via native path")
        except ValueError:
            from mlx_vlm import load
            from mlx_vlm.utils import load_config
            self._model, self._processor = load(
                self.model_name, trust_remote_code=True,
            )
            self._config = load_config(self.model_name)
            logger.info("[MLX] Loaded via mlx-vlm fallback")
        self._prompt_cache = None  # Lazily created on first generate
        self._early_stop = None   # Set via set_early_stop() after load
        self._visual_similarity_threshold = 0.0  # Set via set_visual_similarity() after load
        # Detect if model natively supports video input via processor.
        # Check processor signature for 'videos' param — only Qwen2.5-VL, Qwen3-VL, etc.
        # have this. InternVL3 has video_token_index in config but processor only takes images.
        import inspect
        proc_params = inspect.signature(self._processor.__call__).parameters
        self._is_video_model = "videos" in proc_params
        self._loaded = True
        logger.info("[MLX] Model loaded in %.1fs (video_model=%s)", time.monotonic() - t0, self._is_video_model)

    def _get_prompt_cache(self):
        """Get or create the persistent PromptCache."""
        if self._prompt_cache is None:
            from trio_core.generate import PromptCache
            self._prompt_cache = PromptCache(self._model)
            # Attach StreamingMemory if configured
            if getattr(self, '_streaming_memory_config', None):
                from trio_core.streaming_memory import StreamingMemory
                sm = StreamingMemory(**self._streaming_memory_config)
                self._prompt_cache.set_streaming_memory(sm)
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

    def set_visual_similarity(self, threshold: float) -> None:
        """Configure visual similarity KV reuse. Called by engine after load()."""
        self._visual_similarity_threshold = max(0.0, min(1.0, threshold))
        if self._visual_similarity_threshold > 0:
            logger.info(
                "[MLX] Visual similarity KV reuse enabled: threshold=%.2f",
                self._visual_similarity_threshold,
            )

    def set_streaming_memory(self, enabled: bool, budget: int = 6000, prototype_ratio: float = 0.1, n_sink_tokens: int = 4) -> None:
        """Configure StreamMem bounded KV cache. Called by engine after load()."""
        if not enabled:
            self._streaming_memory_config = None
            return
        self._streaming_memory_config = {
            "budget": budget,
            "prototype_ratio": prototype_ratio,
            "n_sink_tokens": n_sink_tokens,
        }
        logger.info(
            "[MLX] StreamMem enabled: budget=%d, prototype_ratio=%.2f, sink=%d",
            budget, prototype_ratio, n_sink_tokens,
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
        # Detokenizer: mlx-vlm processor has .detokenizer, native uses tokenizer.decode
        has_detokenizer = hasattr(self._processor, "detokenizer")
        if has_detokenizer:
            detokenizer = self._processor.detokenizer
            detokenizer.reset()

        # Reset stopping criteria
        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self._model.config.eos_token_id)

        text = ""
        prompt_tps = 0.0
        generation_tps = 0.0
        n_tokens = 0
        token_ids = []
        eos_token_id = getattr(self._model.config, "eos_token_id", None)

        with _wired_limit(self._model):
            tic = time.perf_counter()
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids, self._model, pixel_values, mask,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    prompt_cache_manager=self._get_prompt_cache(),
                    early_stop=self._early_stop,
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

                # Check EOS for native path
                if eos_token_id is not None and not has_detokenizer:
                    if isinstance(eos_token_id, list):
                        if token in eos_token_id:
                            break
                    elif token == eos_token_id:
                        break

                if has_detokenizer:
                    detokenizer.add_token(token)
                else:
                    token_ids.append(token)
                n_tokens = n + 1

            if has_detokenizer:
                detokenizer.finalize()
                text = detokenizer.text
            else:
                text = tokenizer.decode(token_ids, skip_special_tokens=True)

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
        has_detokenizer = hasattr(self._processor, "detokenizer")
        if has_detokenizer:
            detokenizer = self._processor.detokenizer
            detokenizer.reset()

        if hasattr(tokenizer, "stopping_criteria"):
            tokenizer.stopping_criteria.reset(self._model.config.eos_token_id)

        eos_token_id = getattr(self._model.config, "eos_token_id", None)
        token_ids = []

        with _wired_limit(self._model):
            tic = time.perf_counter()
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids, self._model, pixel_values, mask,
                    max_tokens=max_tokens, temperature=temperature, top_p=top_p,
                    prompt_cache_manager=self._get_prompt_cache(),
                    early_stop=self._early_stop,
                    visual_similarity_threshold=self._visual_similarity_threshold,
                    **kwargs,
                )
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    tic = time.perf_counter()

                if hasattr(tokenizer, "stopping_criteria") and tokenizer.stopping_criteria(token):
                    break

                if eos_token_id is not None and not has_detokenizer:
                    if isinstance(eos_token_id, list):
                        if token in eos_token_id:
                            break
                    elif token == eos_token_id:
                        break

                if has_detokenizer:
                    detokenizer.add_token(token)
                    yield StreamChunk(
                        text=detokenizer.last_segment,
                        prompt_tokens=input_ids.size,
                        completion_tokens=n + 1,
                    )
                else:
                    token_ids.append(token)
                    text = tokenizer.decode(token_ids, skip_special_tokens=True)
                    yield StreamChunk(
                        text=text,
                        prompt_tokens=input_ids.size,
                        completion_tokens=n + 1,
                    )

            if has_detokenizer:
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
        """Prepare inputs via the image pipeline (Gemma 3, SmolVLM2, InternVL, nanoLLaVA, etc.).

        Converts numpy frames to PIL images and passes directly to processor.
        Handles both multi-modal content blocks (Gemma, SmolVLM) and simple
        <image> token format (LLaVA, InternVL, nanoLLaVA).
        """
        import mlx.core as mx

        images = self._frames_to_pil(frames)

        # Try multi-modal content blocks first (Gemma 3, SmolVLM2, etc.)
        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        try:
            formatted = self._processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            formatted = None

        # Verify prompt text is actually included in the formatted output.
        # Some processors (InternVLChatProcessor) silently drop the prompt.
        if formatted is None or prompt[:20] not in formatted:
            tokenizer = getattr(self._processor, 'tokenizer', self._processor)
            image_tokens = "<image>\n" * len(images)
            text_messages = [
                {"role": "user", "content": f"{image_tokens}{prompt}"},
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    text_messages, tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                from mlx_vlm.prompt_utils import apply_chat_template
                formatted = apply_chat_template(
                    self._processor, self._config, prompt, num_images=len(images),
                )

        inputs = self._call_processor(
            text=[formatted], images=images,
        )

        # Convert to MLX arrays for generate_step
        kwargs = {
            "input_ids": mx.array(np.asarray(inputs["input_ids"])),
            "mask": mx.array(np.asarray(inputs["attention_mask"])),
        }

        # Get pixel_values: from processor output, or preprocess via image_processor
        if "pixel_values" in inputs:
            kwargs["pixel_values"] = mx.array(np.asarray(inputs["pixel_values"]))
        elif hasattr(self._processor, 'image_processor'):
            # LLaVA-style: image_processor returns list of (C, H, W) arrays
            pv_list = self._processor.image_processor.preprocess(images)
            kwargs["pixel_values"] = mx.array(np.stack(pv_list))
        else:
            kwargs["pixel_values"] = mx.array(np.array([]))

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
