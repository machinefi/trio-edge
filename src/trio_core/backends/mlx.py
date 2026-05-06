"""VLM backend using MLX on Apple Silicon.

Uses mlx_vlm only for model loading (load + load_config).
Preprocessing, generation, KV cache, and sampling are all internal.
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import numpy as np

from trio_core.backends.base import BaseBackend, GenerationResult, StreamChunk, _TokenHandler

logger = logging.getLogger(__name__)


def compute_compressed_grid(
    grid_thw, original_count: int, compressed_count: int, spatial_merge_size: int = 2
):
    """Compute grid_thw that produces exactly compressed_count tokens.

    Shared between compressed and tome backends for MRoPE position ID
    computation after visual token reduction.

    grid_thw values are pre-merge (before PatchMerger's NxN spatial merge).
    Total visual tokens = T * (H/merge) * (W/merge).
    We scale H, W to match the compressed count.
    """
    import mlx.core as mx

    new_grids = []
    for i in range(grid_thw.shape[0]):
        t, h, w = [x.item() for x in grid_thw[i]]

        h_grid = h // spatial_merge_size
        w_grid = w // spatial_merge_size
        target_per_t = compressed_count // max(t, 1)

        if target_per_t <= 0:
            target_per_t = 1

        original_per_t = h_grid * w_grid
        ratio = (target_per_t / max(original_per_t, 1)) ** 0.5
        new_h_grid = max(1, round(h_grid * ratio))
        new_w_grid = max(1, target_per_t // new_h_grid)

        # Fine-tune to match exactly
        while new_h_grid * new_w_grid * t > compressed_count and new_w_grid > 1:
            new_w_grid -= 1
        while new_h_grid * new_w_grid * t < compressed_count and new_w_grid < w_grid:
            new_w_grid += 1

        new_h = new_h_grid * spatial_merge_size
        new_w = new_w_grid * spatial_merge_size
        new_grids.append([t, new_h, new_w])

    return mx.array(new_grids, dtype=grid_thw.dtype)


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
        if self.adapter_path:
            logger.info("[MLX] LoRA adapter: %s", self.adapter_path)
        t0 = time.monotonic()

        # Try native loading first (T1 models), fall back to mlx-vlm (T2)
        # If adapter_path is set, skip native loader and use mlx-vlm which has
        # built-in LoRA support via apply_lora_layers().
        if not self.adapter_path:
            try:
                from trio_core.models.loader import load_config_native, load_native

                self._model, self._processor = load_native(self.model_name)
                self._config = load_config_native(self.model_name)
                logger.info("[MLX] Loaded via native path")
            except ValueError:
                self._model = None  # Fall through to mlx-vlm

        if not hasattr(self, "_model") or self._model is None:
            try:
                from mlx_vlm import load
                from mlx_vlm.utils import load_config
            except ImportError:
                raise ImportError(
                    f"Model '{self.model_name}' is not natively supported. "
                    f"Install mlx-vlm for T2 model support: pip install 'trio-core[mlx-vlm]'"
                )
            load_kwargs = {"trust_remote_code": True}
            if self.adapter_path:
                load_kwargs["adapter_path"] = self.adapter_path
            self._model, self._processor = load(
                self.model_name,
                **load_kwargs,
            )
            self._config = load_config(self.model_name)
            if self.adapter_path:
                logger.info("[MLX] Loaded via mlx-vlm with LoRA adapter")
            else:
                logger.info("[MLX] Loaded via mlx-vlm fallback")
        self._prompt_cache = None  # Lazily created on first generate
        self._early_stop = None  # Set via set_early_stop() after load
        self._visual_similarity_threshold = 0.0  # Set via set_visual_similarity() after load
        # Detect if model natively supports video input via processor.
        # Check processor signature for 'videos' param — only Qwen2.5-VL, Qwen3-VL, etc.
        # have this. InternVL3 has video_token_index in config but processor only takes images.
        import inspect

        proc_params = inspect.signature(self._processor.__call__).parameters
        self._is_video_model = "videos" in proc_params
        self._loaded = True
        logger.info(
            "[MLX] Model loaded in %.1fs (video_model=%s)",
            time.monotonic() - t0,
            self._is_video_model,
        )

    def _get_prompt_cache(self):
        """Get or create the persistent PromptCache."""
        if self._prompt_cache is None:
            from trio_core.generate import PromptCache

            self._prompt_cache = PromptCache(self._model)
            # Attach StreamingMemory if configured
            if getattr(self, "_streaming_memory_config", None):
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

    def set_streaming_memory(
        self,
        enabled: bool,
        budget: int = 6000,
        prototype_ratio: float = 0.1,
        n_sink_tokens: int = 4,
    ) -> None:
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
            budget,
            prototype_ratio,
            n_sink_tokens,
        )

    # ── Token handling (shared by all decode loops) ─────────────────────

    def _make_token_handler(self):
        """Create a _TokenHandler for the current model/processor."""
        return _TokenHandler(self._processor, self._model.config)

    # ── Unified decode loop (shared by all MLX backend subclasses) ───────

    def _make_generate_config(self, max_tokens: int, temperature: float, top_p: float):
        """Build GenerateConfig from backend state + per-request sampling params."""
        from trio_core.generate import GenerateConfig

        return GenerateConfig(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            early_stop=self._early_stop,
            visual_similarity_threshold=self._visual_similarity_threshold,
        )

    def _run_generate(
        self,
        input_ids,
        pixel_values,
        mask,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        inputs_embeds=None,
        **extra_kwargs,
    ) -> GenerationResult:
        """Unified generate: generate_step + detokenize + TPS.

        Subclasses override generate() to prepare inputs, then call this.
        """
        import mlx.core as mx

        from trio_core.generate import _wired_limit, generate_step

        th = self._make_token_handler()
        cfg = self._make_generate_config(max_tokens, temperature, top_p)
        prompt_tps = 0.0
        generation_tps = 0.0
        n_tokens = 0

        with _wired_limit(self._model):
            tic = time.perf_counter()
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids,
                    self._model,
                    pixel_values,
                    mask,
                    config=cfg,
                    prompt_cache_manager=self._get_prompt_cache(),
                    inputs_embeds=inputs_embeds,
                    **extra_kwargs,
                )
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = input_ids.size / max(prompt_time, 1e-9)
                    tic = time.perf_counter()

                if th.should_stop(token):
                    break
                th.add_token(token)
                n_tokens = n + 1

            if n_tokens > 0:
                generation_tps = n_tokens / max(time.perf_counter() - tic, 1e-9)

        return GenerationResult(
            text=th.finalize(),
            prompt_tokens=input_ids.size,
            completion_tokens=n_tokens,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else 0.0,
        )

    def _run_stream_generate(
        self,
        input_ids,
        pixel_values,
        mask,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        inputs_embeds=None,
        **extra_kwargs,
    ) -> Generator[StreamChunk, None, None]:
        """Unified streaming: generate_step + yield chunks.

        Subclasses override stream_generate() to prepare inputs, then call this.
        """
        import mlx.core as mx

        from trio_core.generate import _wired_limit, generate_step

        th = self._make_token_handler()
        cfg = self._make_generate_config(max_tokens, temperature, top_p)
        n = 0

        with _wired_limit(self._model):
            for n, (token, logprobs) in enumerate(
                generate_step(
                    input_ids,
                    self._model,
                    pixel_values,
                    mask,
                    config=cfg,
                    prompt_cache_manager=self._get_prompt_cache(),
                    inputs_embeds=inputs_embeds,
                    **extra_kwargs,
                )
            ):
                if th.should_stop(token):
                    break
                delta = th.add_token_streaming(token)
                yield StreamChunk(
                    text=delta,
                    prompt_tokens=input_ids.size,
                    completion_tokens=n + 1,
                )

            final = th.finalize_delta()
            if final:
                yield StreamChunk(
                    text=final,
                    prompt_tokens=input_ids.size,
                    completion_tokens=n + 1,
                    finished=True,
                )

            mx.clear_cache()

    # ── AR decode loop (for backends with custom prefill) ─────────────

    def _run_ar_decode(
        self,
        first_token,
        prompt_cache,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prompt_token_count: int = 0,
        prompt_tps: float = 0.0,
    ) -> GenerationResult:
        """Autoregressive decode loop for backends that do their own prefill.

        Args:
            first_token: The first sampled token (mx.array scalar).
            prompt_cache: Populated KV cache from custom prefill.
            prompt_token_count: Number of prompt tokens (for metrics).
            prompt_tps: Prompt throughput (for metrics).
        """
        import mlx.core as mx

        from trio_core.generate import make_sampler

        th = self._make_token_handler()
        sampler = make_sampler(temperature, top_p)

        y = first_token
        tic = time.perf_counter()
        n_generated = 0

        for n_generated in range(max_tokens):
            tok = y.item()

            if th.should_stop(tok):
                break
            th.add_token(tok)

            outputs = self._model.language_model(y[None], cache=prompt_cache)
            logits = outputs.logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)
            mx.eval(y)

            if n_generated % 256 == 0:
                mx.clear_cache()

        n_generated = max(n_generated, 1)
        decode_time = time.perf_counter() - tic
        generation_tps = n_generated / max(decode_time, 1e-9)

        mx.clear_cache()

        return GenerationResult(
            text=th.finalize().strip(),
            prompt_tokens=prompt_token_count,
            completion_tokens=n_generated,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9 if hasattr(mx, "get_peak_memory") else 0.0,
        )

    def _run_ar_stream_decode(
        self,
        first_token,
        prompt_cache,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        prompt_token_count: int = 0,
    ) -> Generator[StreamChunk, None, None]:
        """Streaming AR decode for backends with custom prefill."""
        import mlx.core as mx

        from trio_core.generate import make_sampler

        th = self._make_token_handler()
        sampler = make_sampler(temperature, top_p)

        y = first_token
        n_generated = 0

        for n_generated in range(max_tokens):
            tok = y.item()

            if th.should_stop(tok):
                break
            delta = th.add_token_streaming(tok)
            yield StreamChunk(
                text=delta,
                prompt_tokens=prompt_token_count,
                completion_tokens=n_generated + 1,
            )

            outputs = self._model.language_model(y[None], cache=prompt_cache)
            logits = outputs.logits[:, -1, :]
            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)
            mx.eval(y)

            if n_generated % 256 == 0:
                mx.clear_cache()

        final = th.finalize_delta()
        if final:
            yield StreamChunk(
                text=final,
                prompt_tokens=prompt_token_count,
                completion_tokens=n_generated + 1,
                finished=True,
            )

        mx.clear_cache()

    # ── OOM guard ──────────────────────────────────────────────────────

    def _check_memory(self, pixel_values) -> None:
        """Estimate prefill memory and raise before Metal OOM kills process."""
        if pixel_values is None:
            return
        import mlx.core as mx

        try:
            info = mx.device_info()
        except AttributeError:
            info = mx.metal.device_info()
        budget = info.get("max_recommended_working_set_size", 0)
        if budget <= 0:
            return
        active = (
            mx.get_active_memory()
            if hasattr(mx, "get_active_memory")
            else mx.metal.get_active_memory()
        )
        available = budget - active

        # Rough estimate: prefill needs ~6x pixel_values (forward + intermediates)
        pixel_bytes = pixel_values.nbytes if hasattr(pixel_values, "nbytes") else 0
        estimated = pixel_bytes * 6
        if estimated > available:
            pixel_mb = pixel_bytes / (1024**2)
            avail_mb = available / (1024**2)
            est_mb = estimated / (1024**2)
            raise MemoryError(
                f"Visual input too large for available memory: "
                f"pixel_values={pixel_mb:.0f}MB, estimated prefill={est_mb:.0f}MB, "
                f"available={avail_mb:.0f}MB. Reduce frame count or resolution."
            )

    # ── Public generate/stream_generate (thin wrappers) ───────────────

    def generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        response_format: dict | None = None,
    ) -> GenerationResult:
        # response_format is a remote-only structured-output spec; local
        # MLX inference doesn't honor it — ignored.
        del response_format
        formatted, kwargs = self._prepare(frames, prompt)
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")
        return self._run_generate(
            input_ids,
            pixel_values,
            mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def stream_generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        response_format: dict | None = None,
    ) -> Generator[StreamChunk, None, None]:
        del response_format
        formatted, kwargs = self._prepare(frames, prompt)
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")
        return self._run_stream_generate(
            input_ids,
            pixel_values,
            mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

    def _prepare(self, frames: np.ndarray, prompt: str) -> tuple[str, dict]:
        """Route to video or image preparation based on model capability."""
        if self._is_video_model:
            result = self._prepare_video(frames, prompt)
        else:
            result = self._prepare_images(frames, prompt)
        self._check_memory(result[1].get("pixel_values"))
        return result

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

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": pil_frames},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        formatted = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Disable thinking mode (Qwen3.5 appends <think\n> by default)
        if formatted.endswith("<think\n>"):
            formatted = formatted[: -len("<think\n>")]

        inputs = self._call_processor(
            text=[formatted],
            videos=[pil_frames],
        )

        # Convert to MLX arrays for generate_step
        kwargs = {
            "input_ids": mx.array(np.asarray(inputs["input_ids"])),
            "pixel_values": mx.array(
                np.asarray(
                    inputs.get("pixel_values_videos", inputs.get("pixel_values", np.array([])))
                )
            ),
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
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except (TypeError, ValueError, KeyError, AttributeError):
            formatted = None

        # Verify prompt text is actually included in the formatted output.
        # Some processors (InternVLChatProcessor) silently drop the prompt.
        if formatted is None or prompt[:20] not in formatted:
            tokenizer = getattr(self._processor, "tokenizer", self._processor)
            image_tokens = "<image>\n" * len(images)
            text_messages = [
                {"role": "user", "content": f"{image_tokens}{prompt}"},
            ]
            try:
                formatted = tokenizer.apply_chat_template(
                    text_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                try:
                    from mlx_vlm.prompt_utils import apply_chat_template

                    formatted = apply_chat_template(
                        self._processor,
                        self._config,
                        prompt,
                        num_images=len(images),
                    )
                except ImportError:
                    # Last resort: plain text without template
                    formatted = f"{image_tokens}{prompt}"

        # Disable thinking mode (Qwen3.5 appends <think\n> by default)
        if formatted.endswith("<think\n>"):
            formatted = formatted[: -len("<think\n>")]

        inputs = self._call_processor(
            text=[formatted],
            images=images,
        )

        # Convert to MLX arrays for generate_step
        kwargs = {
            "input_ids": mx.array(np.asarray(inputs["input_ids"])),
            "mask": mx.array(np.asarray(inputs["attention_mask"])),
        }

        # Get pixel_values: from processor output, or preprocess via image_processor
        if "pixel_values" in inputs:
            kwargs["pixel_values"] = mx.array(np.asarray(inputs["pixel_values"]))
        elif hasattr(self._processor, "image_processor"):
            # LLaVA-style: image_processor returns list of (C, H, W) arrays
            pv_list = self._processor.image_processor.preprocess(images)
            kwargs["pixel_values"] = mx.array(np.stack(pv_list))
        else:
            kwargs["pixel_values"] = mx.array(np.array([]))

        for key in ("image_grid_thw",):
            if key in inputs:
                kwargs[key] = mx.array(np.asarray(inputs[key]))

        return formatted, kwargs

    def _call_processor(self, **kwargs) -> dict:
        """Call processor with numpy/PyTorch fallback."""
        try:
            inputs = self._processor(**kwargs, padding=True, return_tensors="np")
        except (ValueError, TypeError):
            inputs = self._processor(**kwargs, padding=True, return_tensors="pt")
            inputs = {k: v.numpy() if hasattr(v, "numpy") else v for k, v in inputs.items()}
        return inputs
