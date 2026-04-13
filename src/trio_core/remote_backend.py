"""Remote VLM backend — routes inference to an OpenAI-compatible Chat API.

Supported providers:
    - DashScope / Qwen VL (Alibaba Cloud)
    - Any OpenAI-compatible vision API

Usage:
    Set environment variables:
        TRIO_REMOTE_VLM_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
        TRIO_REMOTE_VLM_API_KEY=sk-xxx
        TRIO_REMOTE_VLM_MODEL=qwen-vl-plus

    The engine will use this backend instead of local GPU inference.
"""

from __future__ import annotations

import base64
import logging
import time
from io import BytesIO
from typing import Generator

import numpy as np

from trio_core.backends import BaseBackend, GenerationResult, StreamChunk
from trio_core.device import DeviceInfo

logger = logging.getLogger(__name__)


def _frames_to_data_uris(frames: np.ndarray) -> list[str]:
    """Convert (T, C, H, W) float32 numpy array to base64 JPEG data URIs."""
    from PIL import Image

    uris: list[str] = []
    for i in range(frames.shape[0]):
        frame = (frames[i].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        uris.append(f"data:image/jpeg;base64,{b64}")
    return uris


class RemoteHTTPBackend(BaseBackend):
    def __init__(
        self,
        url: str,
        api_key: str | None = None,
        model: str = "qwen-vl-plus",
    ):
        device_info = DeviceInfo(
            backend="remote",
            device_name="Remote API",
            accelerator="remote",
            memory_gb=0,
            compute_units=0,
        )
        super().__init__(model_name=model, device_info=device_info)
        self._url = url
        self._api_key = api_key
        self._remote_model = model
        self._client = None

    @property
    def backend_name(self) -> str:
        return "remote"

    def load(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(
            base_url=self._url,
            api_key=self._api_key or "unused",
        )
        self._loaded = True
        logger.info(
            "[Remote] Backend ready: url=%s, model=%s",
            self._url,
            self._remote_model,
        )

    def generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        t0 = time.monotonic()
        uris = _frames_to_data_uris(frames)

        # Build OpenAI vision content blocks — multiple image_url entries
        # for multi-frame input (Qwen VL "video as image list" pattern)
        content: list[dict] = []
        for uri in uris:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": uri},
                }
            )
        content.append({"type": "text", "text": prompt})

        messages = [{"role": "user", "content": content}]

        response = self._client.chat.completions.create(
            model=self._remote_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        choice = response.choices[0]
        text = choice.message.content or ""
        elapsed = time.monotonic() - t0

        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        gen_tps = completion_tokens / max(elapsed, 1e-9)

        logger.info(
            "[Remote] generate: %d frames, %d+%d tokens, %.1f tps, %.0fms",
            frames.shape[0],
            prompt_tokens,
            completion_tokens,
            gen_tps,
            elapsed * 1000,
        )

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tps=0.0,  # Prefill happens server-side — not measurable
            generation_tps=gen_tps,
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
        """Stream from remote VLM API.

        Wraps generate() yielding a single chunk — remote API calls are
        atomic from our perspective (no incremental token delivery unless
        using stream=True on the remote side, which is a future enhancement).
        """
        result = self.generate(
            frames,
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        yield StreamChunk(
            text=result.text,
            finished=True,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )
