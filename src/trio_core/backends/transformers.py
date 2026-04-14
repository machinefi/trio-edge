"""VLM backend using HuggingFace Transformers on CUDA / CPU.

Supports any VLM loadable via AutoModelForImageTextToText, including:
Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 3, SmolVLM, Phi-4, InternVL, etc.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import numpy as np

from trio_core.backends.base import BaseBackend, GenerationResult, StreamChunk

logger = logging.getLogger(__name__)


class TransformersBackend(BaseBackend):
    """VLM backend using HuggingFace Transformers on CUDA / CPU.

    Supports any VLM loadable via AutoModelForImageTextToText, including:
    Qwen2.5-VL, Qwen3-VL, Qwen3.5, Gemma 3, SmolVLM, Phi-4, InternVL, etc.
    """

    @property
    def backend_name(self) -> str:
        return "transformers"

    def load(self) -> None:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        device = "cuda" if self.device_info.accelerator == "cuda" else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32

        logger.info("[Transformers] Loading model: %s (device=%s)", self.model_name, device)
        t0 = time.monotonic()
        self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._device = device
        self._is_video_model = hasattr(self._model.config, "video_token_id")
        self._loaded = True
        logger.info("[Transformers] Model loaded in %.1fs", time.monotonic() - t0)

    def generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
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
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        import threading

        from transformers import TextIteratorStreamer

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

        # Single-frame requests should still use the image path even for
        # video-capable models. Qwen's video utils expect a list/tuple of
        # frames and can assert on one-frame numpy inputs that originate from
        # analyze_frame().
        if self._is_video_model and frames.shape[0] > 1:
            return self._prepare_video(frames, prompt)

        # Generic path: convert frames to PIL images
        images = self._frames_to_pil(frames)

        messages = [{"role": "user", "content": []}]
        for img in images:
            messages[0]["content"].append({"type": "image", "image": img})
        messages[0]["content"].append({"type": "text", "text": prompt})

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._processor(
            text=[text],
            images=images,
            padding=True,
            return_tensors="pt",
        )

        return {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}

    def _prepare_video(self, frames: np.ndarray, prompt: str) -> dict:
        """Video-capable model preparation using qwen_vl_utils."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        try:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs, _ = process_vision_info(messages, return_video_kwargs=True)
        except ImportError:
            video_inputs = [frames]
            image_inputs = None

        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        return {k: v.to(self._device) if hasattr(v, "to") else v for k, v in inputs.items()}
