"""MLX backend with ToMe (Token Merging) in the vision encoder.

Handles custom vision encoding + token compression, then delegates
prefill/decode to generate_step for PromptCache, early stopping,
early stopping, and other features.
"""

from __future__ import annotations

import logging
from typing import Generator

import numpy as np

from trio_core.backends.base import GenerationResult
from trio_core.backends.mlx import MLXBackend, compute_compressed_grid
from trio_core.native_vision import create_tome_vision

logger = logging.getLogger(__name__)


class ToMeMLXBackend(MLXBackend):
    """MLX backend with Token Merging in the vision encoder.

    Auto-detects model type (Qwen2.5-VL vs Qwen3-VL) and uses
    the appropriate wrapper.
    """

    def __init__(
        self,
        model_name: str,
        tome_r: int = 8,
        metric: str = "keys",
        min_keep_ratio: float = 0.3,
        adaptive: bool = False,
        content_aware: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        if tome_r < 0:
            raise ValueError(f"tome_r must be >= 0, got {tome_r}")
        if metric not in ("keys", "hidden"):
            raise ValueError(f"metric must be 'keys' or 'hidden', got {metric!r}")
        if not (0.0 < min_keep_ratio <= 1.0):
            raise ValueError(f"min_keep_ratio must be in (0.0, 1.0], got {min_keep_ratio}")

        self.tome_r = tome_r
        self.tome_metric = metric
        self.tome_min_keep_ratio = min_keep_ratio
        self.tome_adaptive = adaptive
        self.tome_content_aware = content_aware
        self._native_vision = None
        self._adapter = None

    @property
    def backend_name(self) -> str:
        return "mlx-tome"

    def load(self) -> None:
        super().load()

        from trio_core.model_adapter import get_adapter

        self._adapter = get_adapter(self._model)

        if not self._adapter.supports_tome:
            logger.warning(
                "[ToMe] Model family '%s' does not support in-ViT ToMe. "
                "ToMe will be disabled. Consider using Compressed instead.",
                self._adapter.family,
            )
            self._native_vision = None
            return

        self._native_vision = create_tome_vision(
            self._model.vision_tower,
            tome_r=self.tome_r,
            metric=self.tome_metric,
            min_keep_ratio=self.tome_min_keep_ratio,
            adaptive=self.tome_adaptive,
            content_aware=self.tome_content_aware,
        )
        self._model.vision_tower = self._native_vision

    def _prepare_tome_embeds(self, input_ids, pixel_values, kwargs):
        """Run ToMe vision encoding and build compressed embeddings.

        Returns:
            (final_embeds, new_input_ids, extra_kwargs) — ready for generate_step.
        """
        import mlx.core as mx

        video_grid_thw = kwargs.get("video_grid_thw", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        grid_thw = video_grid_thw if video_grid_thw is not None else image_grid_thw

        model = self._model
        adapter = self._adapter

        # Step 1: Run vision encoder (with ToMe wrapping)
        vision_out = adapter.run_vision_encoder(pixel_values, grid_thw)
        hidden_states = vision_out.hidden_states
        deepstack_embeds = vision_out.deepstack_embeds

        if grid_thw is not None:
            original_count = adapter.original_token_count(grid_thw)
        else:
            # Standard ViT (no grid_thw) — estimate from hidden_states + merge log
            merge_log = self._native_vision._merge_log if self._native_vision else []
            total_merged = sum(e.get("merged", 0) for e in merge_log)
            original_count = hidden_states.shape[0] + total_merged
        compressed_count = hidden_states.shape[0]

        logger.info(
            "ToMe compression: %d → %d tokens (%.0f%% reduction, %d layers merged)",
            original_count,
            compressed_count,
            (1 - compressed_count / max(original_count, 1)) * 100,
            len(self._native_vision._merge_log) if self._native_vision else 0,
        )

        # Step 2: Build compressed input sequence
        video_token_id, image_token_id = adapter.get_visual_token_ids()

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
        merge_result = adapter.merge_visual_features(hidden_states, text_embeds, new_input_ids)
        final_embeds = merge_result.embeds
        image_mask = merge_result.image_mask

        # Compute position IDs for compressed sequence (MRoPE models only)
        if adapter.uses_mrope:
            compressed_grid = compute_compressed_grid(
                grid_thw,
                original_count,
                compressed_count,
                spatial_merge_size=adapter.spatial_merge_size,
            )

            # Align hidden_states and input_ids to match grid token count
            grid_count = adapter.original_token_count(compressed_grid)
            if grid_count != compressed_count:
                logger.debug(
                    "Grid alignment: compressed=%d, grid=%d, adjusting",
                    compressed_count,
                    grid_count,
                )
                if grid_count > compressed_count:
                    pad = mx.repeat(hidden_states[-1:], grid_count - compressed_count, axis=0)
                    hidden_states = mx.concatenate([hidden_states, pad])
                else:
                    hidden_states = hidden_states[:grid_count]

                # Rebuild with adjusted counts
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

                text_embeds = model.language_model.model.embed_tokens(new_input_ids)
                merge_result = adapter.merge_visual_features(
                    hidden_states, text_embeds, new_input_ids
                )
                final_embeds = merge_result.embeds
                image_mask = merge_result.image_mask
                compressed_count = grid_count

            kw_grid = {}
            if n_video > 0:
                kw_grid["video_grid_thw"] = compressed_grid
            else:
                kw_grid["image_grid_thw"] = compressed_grid

            position_ids, rope_deltas = adapter.compute_position_ids(
                new_input_ids,
                attention_mask=new_mask,
                **kw_grid,
            )
            model.language_model._position_ids = position_ids
            model.language_model._rope_deltas = rope_deltas

        # Build extra kwargs for Qwen3 deepstack
        extra_kwargs = {}
        if adapter.has_deepstack and deepstack_embeds is not None:
            visual_pos_masks = image_mask[..., 0] if image_mask is not None else None
            extra_kwargs["visual_pos_masks"] = visual_pos_masks
            mx.eval(deepstack_embeds)
            extra_kwargs["deepstack_visual_embeds"] = deepstack_embeds

        return final_embeds, new_input_ids, extra_kwargs

    def generate(
        self,
        frames: np.ndarray,
        prompt: str,
        *,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> GenerationResult:
        """Run inference with ToMe-compressed vision tokens."""
        formatted, kwargs = self._prepare(frames, prompt)

        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

        final_embeds, new_input_ids, extra_kwargs = self._prepare_tome_embeds(
            input_ids,
            pixel_values,
            kwargs,
        )
        kwargs.update(extra_kwargs)

        return self._run_generate(
            new_input_ids,
            pixel_values,
            mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            inputs_embeds=final_embeds,
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
    ) -> Generator:
        formatted, kwargs = self._prepare(frames, prompt)

        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        mask = kwargs.pop("mask")

        final_embeds, new_input_ids, extra_kwargs = self._prepare_tome_embeds(
            input_ids,
            pixel_values,
            kwargs,
        )
        kwargs.update(extra_kwargs)

        yield from self._run_stream_generate(
            new_input_ids,
            pixel_values,
            mask,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            inputs_embeds=final_embeds,
            **kwargs,
        )

    def _original_token_count(self, grid_thw) -> int:
        """Compute expected token count without ToMe (after PatchMerger)."""
        if self._adapter is not None:
            return self._adapter.original_token_count(grid_thw)
        return self._static_token_count(grid_thw)

    @staticmethod
    def _static_token_count(grid_thw, spatial_merge_size: int = 2) -> int:
        """Static helper for token count computation."""
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * (h // spatial_merge_size) * (w // spatial_merge_size)
        return total

    @property
    def merge_log(self) -> list[dict]:
        if self._native_vision is None:
            return []
        return self._native_vision._merge_log
