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

from trio_core.backends import MLXBackend, GenerationResult, StreamChunk
from trio_core.token_compression import TokenCompressor, CompressionResult

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
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> GenerationResult:
        tic = time.perf_counter()
        y, prompt_cache, prompt_token_count = self._custom_prefill(
            frames, prompt, temperature, top_p,
        )
        prompt_tps = prompt_token_count / max(time.perf_counter() - tic, 1e-9)

        return self._run_ar_decode(
            y, prompt_cache,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            prompt_token_count=prompt_token_count, prompt_tps=prompt_tps,
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
            original_count, compressed_count,
            (1 - comp_result.ratio) * 100,
        )

        video_token_id, image_token_id = adapter.get_visual_token_ids()
        n_video = mx.sum(input_ids == video_token_id).item()
        visual_token_id = video_token_id if n_video > 0 else image_token_id

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

        text_embeds = model.language_model.model.embed_tokens(new_input_ids)
        merge_result = adapter.merge_visual_features(
            compressed_hidden, text_embeds, new_input_ids,
        )
        final_embeds = merge_result.embeds

        if adapter.uses_mrope:
            compressed_grid = self._compute_compressed_grid(
                grid_thw, original_count, compressed_count,
                spatial_merge_size=adapter.spatial_merge_size,
            )
            kw_grid = {}
            if n_video > 0:
                kw_grid["video_grid_thw"] = compressed_grid
            else:
                kw_grid["image_grid_thw"] = compressed_grid
            position_ids, rope_deltas = adapter.compute_position_ids(
                new_input_ids, attention_mask=new_mask, **kw_grid,
            )
            model.language_model._position_ids = position_ids
            model.language_model._rope_deltas = rope_deltas

        prompt_cache = make_prompt_cache(model.language_model)
        outputs = model.language_model(
            new_input_ids, inputs_embeds=final_embeds, cache=prompt_cache,
        )
        logits = outputs.logits[:, -1, :]

        sampler = make_sampler(temperature, top_p)
        logprobs = logits - mx.logsumexp(logits)
        y = sampler(logprobs)
        mx.eval(y)

        return y, prompt_cache, new_input_ids.size

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        """Real token-by-token streaming with compressed prefill."""
        y, prompt_cache, prompt_token_count = self._custom_prefill(
            frames, prompt, temperature, top_p,
        )
        yield from self._run_ar_stream_decode(
            y, prompt_cache,
            max_tokens=max_tokens, temperature=temperature, top_p=top_p,
            prompt_token_count=prompt_token_count,
        )

    @staticmethod
    def _compute_compressed_grid(grid_thw, original_count: int, compressed_count: int,
                                  spatial_merge_size: int = 2):
        """Compute grid_thw that produces exactly compressed_count tokens.

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
