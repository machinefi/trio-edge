"""MLX backend with ToMe (Token Merging) in the vision encoder.

Uses the CompressedMLXBackend's generate loop (which handles token count
mismatch, position encoding, etc.) but replaces post-encoder compression
with in-encoder ToMe merging.
"""

from __future__ import annotations

import functools
import logging
import time

import numpy as np

from trio_core.backends import MLXBackend, GenerationResult, StreamChunk
from trio_core.compressed_backend import CompressedMLXBackend
from trio_core.tome_vision import (
    BaseToMeVisionWrapper,
    ToMeQwen25VisionWrapper,
    ToMeQwen3VisionWrapper,
)

logger = logging.getLogger(__name__)


class ToMeMLXBackend(MLXBackend):
    """MLX backend with Token Merging in the vision encoder.

    Auto-detects model type (Qwen2.5-VL vs Qwen3-VL) and uses
    the appropriate wrapper.
    """

    def __init__(
        self, model_name: str, tome_r: int = 8,
        metric: str = "keys", min_keep_ratio: float = 0.3,
        adaptive: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        if tome_r < 0:
            raise ValueError(f"tome_r must be >= 0, got {tome_r}")
        if metric not in ("keys", "hidden"):
            raise ValueError(f"metric must be 'keys' or 'hidden', got {metric!r}")
        if not (0.0 < min_keep_ratio <= 1.0):
            raise ValueError(
                f"min_keep_ratio must be in (0.0, 1.0], got {min_keep_ratio}"
            )

        self.tome_r = tome_r
        self.tome_metric = metric
        self.tome_min_keep_ratio = min_keep_ratio
        self.tome_adaptive = adaptive
        self._wrapper: BaseToMeVisionWrapper | None = None
        self._is_qwen3: bool = False

    @property
    def backend_name(self) -> str:
        return "mlx-tome"

    def load(self) -> None:
        super().load()

        # Auto-detect model type from vision_tower
        model_type = getattr(self._model.vision_tower, 'model_type', '')
        if model_type in ('qwen3_vl', 'qwen3_5', 'qwen3_5_moe'):
            wrapper_cls = ToMeQwen3VisionWrapper
            self._is_qwen3 = True
        else:
            wrapper_cls = ToMeQwen25VisionWrapper
            self._is_qwen3 = False

        self._wrapper = wrapper_cls(
            self._model.vision_tower,
            r=self.tome_r,
            metric=self.tome_metric,
            min_keep_ratio=self.tome_min_keep_ratio,
            adaptive=self.tome_adaptive,
        )
        self._model.vision_tower = self._wrapper

        logger.info(
            "[ToMe] Wrapped vision_tower (%s) with r=%d, metric=%s, min_keep=%.0f%%, adaptive=%s",
            wrapper_cls.__name__, self.tome_r, self.tome_metric,
            self.tome_min_keep_ratio * 100, self.tome_adaptive,
        )

    def _get_visual_token_ids(self, model):
        """Get visual token IDs, handling Qwen2.5 vs Qwen3 config differences."""
        if self._is_qwen3:
            return (
                getattr(model.config, 'video_token_index', None) or
                getattr(model.config, 'video_token_id', None),
                getattr(model.config, 'image_token_index', None) or
                getattr(model.config, 'image_token_id', None),
            )
        return model.config.video_token_id, model.config.image_token_id

    def generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> GenerationResult:
        """Run inference with ToMe-compressed vision tokens."""
        import mlx.core as mx
        from trio_core.generate import make_prompt_cache, make_sampler, maybe_quantize_kv_cache

        formatted, kwargs = self._prepare(frames, prompt)

        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

        video_grid_thw = kwargs.get("video_grid_thw", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        grid_thw = video_grid_thw if video_grid_thw is not None else image_grid_thw

        model = self._model

        # Step 1: Run vision encoder (with ToMe wrapping)
        dtype = model.vision_tower.patch_embed.proj.weight.dtype
        pv = pixel_values.astype(dtype)

        vision_output = model.vision_tower(pv, grid_thw, output_hidden_states=False)

        # Qwen3-VL returns (hidden_states, deepstack_feature_lists) tuple
        deepstack_embeds = None
        if isinstance(vision_output, tuple):
            hidden_states = vision_output[0]
            if len(vision_output) > 1:
                deepstack_embeds = vision_output[1]
        else:
            hidden_states = vision_output

        original_count = self._original_token_count(grid_thw)
        compressed_count = hidden_states.shape[0]

        logger.info(
            "ToMe compression: %d → %d tokens (%.0f%% reduction, %d layers merged)",
            original_count, compressed_count,
            (1 - compressed_count / max(original_count, 1)) * 100,
            len(self._wrapper._merge_log) if self._wrapper else 0,
        )

        # Step 2: Build compressed input sequence
        video_token_id, image_token_id = self._get_visual_token_ids(model)

        n_video = mx.sum(input_ids == video_token_id).item()
        visual_token_id = video_token_id if n_video > 0 else image_token_id

        # Trim placeholder tokens to match actual feature count
        ids_list = input_ids[0].tolist()
        new_ids = []
        vis_count = 0
        for tid in ids_list:
            if tid == visual_token_id:
                vis_count += 1
                if vis_count <= compressed_count:
                    new_ids.append(tid)
            else:
                new_ids.append(tid)

        new_input_ids = mx.array([new_ids], dtype=input_ids.dtype)
        new_mask = mx.ones(new_input_ids.shape, dtype=mx.int32)

        # Embed text tokens
        text_embeds = model.language_model.model.embed_tokens(new_input_ids)

        # Merge compressed visual tokens at placeholder positions
        if self._is_qwen3:
            # Qwen3-VL: merge_input_ids_with_image_features(features, embeds, ids, img_idx, vid_idx)
            final_embeds, image_mask = model.merge_input_ids_with_image_features(
                hidden_states, text_embeds, new_input_ids,
                image_token_id, video_token_id,
            )
        else:
            # Qwen2.5-VL: merge_input_ids_with_image_features(img_id, vid_id, features, embeds, ids)
            final_embeds = model.merge_input_ids_with_image_features(
                image_token_id, video_token_id,
                hidden_states, text_embeds, new_input_ids,
            )

        # Compute position IDs for compressed sequence
        compressed_grid = CompressedMLXBackend._compute_compressed_grid(
            grid_thw, original_count, compressed_count
        )

        # The grid may not produce exactly compressed_count tokens due to
        # rounding (e.g., compressed_count=147 but grid gives T*H*W=156).
        # Align hidden_states and input_ids to match the grid.
        grid_count = self._original_token_count(compressed_grid)
        if grid_count != compressed_count:
            logger.debug(
                "Grid alignment: compressed=%d, grid=%d, adjusting",
                compressed_count, grid_count,
            )
            if grid_count > compressed_count:
                # Pad hidden_states by repeating last token
                pad = mx.repeat(hidden_states[-1:], grid_count - compressed_count, axis=0)
                hidden_states = mx.concatenate([hidden_states, pad])
            else:
                hidden_states = hidden_states[:grid_count]

            # Rebuild new_input_ids with grid_count visual placeholders
            ids_list = input_ids[0].tolist()
            new_ids = []
            vis_count = 0
            for tid in ids_list:
                if tid == visual_token_id:
                    vis_count += 1
                    if vis_count <= grid_count:
                        new_ids.append(tid)
                else:
                    new_ids.append(tid)
            new_input_ids = mx.array([new_ids], dtype=input_ids.dtype)
            new_mask = mx.ones(new_input_ids.shape, dtype=mx.int32)

            # Re-merge embeddings with adjusted counts
            text_embeds = model.language_model.model.embed_tokens(new_input_ids)
            if self._is_qwen3:
                final_embeds, image_mask = model.merge_input_ids_with_image_features(
                    hidden_states, text_embeds, new_input_ids,
                    image_token_id, video_token_id,
                )
            else:
                final_embeds = model.merge_input_ids_with_image_features(
                    image_token_id, video_token_id,
                    hidden_states, text_embeds, new_input_ids,
                )

            compressed_count = grid_count

        kw_grid = {}
        if n_video > 0:
            kw_grid["video_grid_thw"] = compressed_grid
        else:
            kw_grid["image_grid_thw"] = compressed_grid

        position_ids, rope_deltas = model.language_model.get_rope_index(
            new_input_ids, attention_mask=new_mask, **kw_grid,
        )
        model.language_model._position_ids = position_ids
        model.language_model._rope_deltas = rope_deltas

        # Step 3: Generate
        prompt_cache = make_prompt_cache(model.language_model)
        sampler = make_sampler(temperature, top_p)
        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=5000,
            kv_group_size=64,
            kv_bits=None,
        )

        tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

        detokenizer = self._processor.detokenizer
        detokenizer.reset()

        # Prefill
        tic = time.perf_counter()

        prefill_kwargs = {}
        if self._is_qwen3 and deepstack_embeds is not None:
            visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
            prefill_kwargs["visual_pos_masks"] = visual_pos_masks
            prefill_kwargs["deepstack_visual_embeds"] = mx.eval(deepstack_embeds)

        outputs = model.language_model(
            new_input_ids,
            inputs_embeds=final_embeds,
            cache=prompt_cache,
            **prefill_kwargs,
        )
        logits = outputs.logits[:, -1, :]
        quantize_cache_fn(prompt_cache)

        logprobs = logits - mx.logsumexp(logits)
        y = sampler(logprobs)
        mx.eval(y)

        prompt_time = time.perf_counter() - tic
        prompt_token_count = new_input_ids.size
        prompt_tps = prompt_token_count / max(prompt_time, 1e-9)

        # Decode loop
        tic = time.perf_counter()
        text = ""
        n_generated = 0

        for n_generated in range(max_tokens):
            if tokenizer.stopping_criteria(y.item()):
                break

            detokenizer.add_token(y.item())
            text += detokenizer.last_segment

            outputs = model.language_model(y[None], cache=prompt_cache)
            logits = outputs.logits[:, -1, :]
            quantize_cache_fn(prompt_cache)
            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)
            mx.eval(y)

            if n_generated % 256 == 0:
                mx.clear_cache()

        detokenizer.finalize()
        text += detokenizer.last_segment

        n_generated = max(n_generated, 1)
        decode_time = time.perf_counter() - tic
        generation_tps = n_generated / max(decode_time, 1e-9)

        mx.clear_cache()

        return GenerationResult(
            text=text.strip(),
            prompt_tokens=prompt_token_count,
            completion_tokens=n_generated,
            prompt_tps=prompt_tps,
            generation_tps=generation_tps,
            peak_memory=mx.get_peak_memory() / 1e9,
        )

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ):
        result = self.generate(frames, prompt, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)
        yield StreamChunk(
            text=result.text, finished=True,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    @staticmethod
    def _original_token_count(grid_thw) -> int:
        """Compute expected token count without ToMe (after PatchMerger)."""
        spatial_merge_size = 2
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * (h // spatial_merge_size) * (w // spatial_merge_size)
        return total

    @property
    def merge_log(self) -> list[dict]:
        if self._wrapper is None:
            return []
        return self._wrapper._merge_log
