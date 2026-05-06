"""Compressed MLX backend — visual token compression between vision encoder and LLM.

Takes over the mlx_vlm generate loop to insert token compression:
    vision_tower → [compress] → merge_with_text → own generate loop

This is the core innovation from:
  - NeurIPS 2024: Visual Context Compression
  - ICLR 2025: Inference Optimal VLMs Need Fewer Visual Tokens
"""

from __future__ import annotations

import logging
import time
from typing import Generator

import numpy as np

from trio_core.backends.base import GenerationResult, StreamChunk
from trio_core.backends.mlx import MLXBackend, compute_compressed_grid
from trio_core.token_compression import CompressionResult, TokenCompressor

logger = logging.getLogger(__name__)


class CompressedMLXBackend(MLXBackend):
    """MLX backend with visual token compression.

    Overrides the generate path to:
    1. Run vision encoder (vision_tower)
    2. Compress visual tokens (hidden_states)
    3. Build compressed input_ids + embeddings with matching positions
    4. Run own LLM generate loop (bypassing mlx_vlm.generate)
    """

    def __init__(self, model_name: str, compressor: TokenCompressor, **kwargs):
        super().__init__(model_name, **kwargs)
        self.compressor = compressor
        self.last_compression: CompressionResult | None = None
        self._adapter = None

    def load(self) -> None:
        super().load()
        from trio_core.model_adapter import get_adapter

        self._adapter = get_adapter(self._model)

    @property
    def backend_name(self) -> str:
        return "mlx-compressed"

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
        del response_format  # remote-only spec; ignored by compressed local backend
        tic = time.perf_counter()
        y, prompt_cache, prompt_token_count = self._custom_prefill(
            frames,
            prompt,
            temperature,
            top_p,
        )
        prompt_tps = prompt_token_count / max(time.perf_counter() - tic, 1e-9)

        return self._run_ar_decode(
            y,
            prompt_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_token_count=prompt_token_count,
            prompt_tps=prompt_tps,
        )

    def _custom_prefill(self, frames, prompt, temperature, top_p):
        """Run compressed prefill, return (first_token, prompt_cache, prompt_token_count)."""
        import mlx.core as mx

        from trio_core.generate import make_prompt_cache, make_sampler

        formatted, kwargs = self._prepare(frames, prompt)
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values")
        kwargs.pop("mask")

        video_grid_thw = kwargs.get("video_grid_thw", None)
        image_grid_thw = kwargs.get("image_grid_thw", None)
        grid_thw = video_grid_thw if video_grid_thw is not None else image_grid_thw

        model = self._model
        adapter = self._adapter

        vision_out = adapter.run_vision_encoder(pixel_values, grid_thw)
        hidden_states = vision_out.hidden_states
        original_count = hidden_states.shape[0]

        comp_result = self.compressor.compress(hidden_states, grid_thw)
        self.last_compression = comp_result
        compressed_hidden = comp_result.compressed
        compressed_count = comp_result.compressed_count

        logger.info(
            "Visual token compression: %d → %d (%.0f%% reduction)",
            original_count,
            compressed_count,
            (1 - comp_result.ratio) * 100,
        )

        video_token_id, image_token_id = adapter.get_visual_token_ids()
        n_video = mx.sum(input_ids == video_token_id).item()
        visual_token_id = video_token_id if n_video > 0 else image_token_id

        # For MRoPE models, compute compressed grid first so we know the exact
        # token count the grid produces (may differ from compressed_count by ±1
        # due to integer rounding). Use grid_token_count for input_ids to ensure
        # position_ids and input_ids have matching lengths.
        effective_count = compressed_count
        compressed_grid = None
        if adapter.uses_mrope:
            compressed_grid = compute_compressed_grid(
                grid_thw,
                original_count,
                compressed_count,
                spatial_merge_size=adapter.spatial_merge_size,
            )
            # Compute actual tokens this grid produces
            grid_token_count = 0
            for i in range(compressed_grid.shape[0]):
                t_g, h_g, w_g = [x.item() for x in compressed_grid[i]]
                grid_token_count += (
                    t_g * (h_g // adapter.spatial_merge_size) * (w_g // adapter.spatial_merge_size)
                )
            if grid_token_count != compressed_count:
                logger.debug(
                    "Grid adjustment: compressed_count=%d → grid_token_count=%d",
                    compressed_count,
                    grid_token_count,
                )
                effective_count = grid_token_count
                # Also trim/pad compressed_hidden to match
                if grid_token_count < compressed_count:
                    compressed_hidden = compressed_hidden[:grid_token_count]
                elif grid_token_count > compressed_count:
                    # Pad by repeating last token
                    pad = mx.broadcast_to(
                        compressed_hidden[-1:],
                        (grid_token_count - compressed_count, compressed_hidden.shape[1]),
                    )
                    compressed_hidden = mx.concatenate([compressed_hidden, pad], axis=0)

        ids_list = input_ids[0].tolist()
        new_ids = []
        vis_count = 0
        for tid in ids_list:
            if tid == visual_token_id:
                vis_count += 1
                if vis_count <= effective_count:
                    new_ids.append(tid)
            else:
                new_ids.append(tid)

        new_input_ids = mx.array([new_ids], dtype=input_ids.dtype)
        new_mask = mx.ones(new_input_ids.shape, dtype=mx.int32)

        text_embeds = model.language_model.model.embed_tokens(new_input_ids)
        merge_result = adapter.merge_visual_features(
            compressed_hidden,
            text_embeds,
            new_input_ids,
        )
        final_embeds = merge_result.embeds

        if compressed_grid is not None:
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

        prompt_cache = make_prompt_cache(model.language_model)
        outputs = model.language_model(
            new_input_ids,
            inputs_embeds=final_embeds,
            cache=prompt_cache,
        )
        logits = outputs.logits[:, -1, :]

        sampler = make_sampler(temperature, top_p)
        logprobs = logits - mx.logsumexp(logits)
        y = sampler(logprobs)
        mx.eval(y)

        return y, prompt_cache, new_input_ids.size

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
        """Real token-by-token streaming with compressed prefill."""
        del response_format
        y, prompt_cache, prompt_token_count = self._custom_prefill(
            frames,
            prompt,
            temperature,
            top_p,
        )
        yield from self._run_ar_stream_decode(
            y,
            prompt_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            prompt_token_count=prompt_token_count,
        )
