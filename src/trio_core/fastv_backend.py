"""MLX backend with FastV-style visual token pruning in LLM layers.

Runs the vision encoder normally, merges visual+text embeddings, then runs
the first N LLM layers to compute text→visual attention importance.
Lowest-scoring visual tokens are pruned, and the full LLM runs on the
shorter sequence.

Training-free, following:
  - FastV (Chen et al., 2024): An Image is Worth 1/2 Tokens After Layer 2
  - SparseVLM (Zhang et al., 2024)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Generator

import numpy as np

from trio_core.backends import MLXBackend, GenerationResult, StreamChunk
from trio_core.compressed_backend import CompressedMLXBackend

logger = logging.getLogger(__name__)


def _bool_indices(mask):
    """Get indices where mask is True (MLX doesn't support boolean indexing)."""
    import mlx.core as mx
    n = mask.shape[0]
    indices = mx.arange(n)
    # Replace False positions with n (out of range sentinel), then sort and take
    mapped = mx.where(mask, indices, mx.array(n, dtype=indices.dtype))
    n_true = mask.astype(mx.int32).sum().item()
    return mx.sort(mapped)[:n_true]


class FastVMLXBackend(MLXBackend):
    """MLX backend with FastV-style visual token pruning in LLM layers.

    After merging visual + text embeddings, runs the first N LLM layers,
    extracts text→visual attention scores at layer N, and prunes the
    lowest-scoring visual tokens before running the full LLM.
    """

    def __init__(
        self, model_name: str,
        prune_ratio: float = 0.5,
        prune_after_layer: int = 2,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        if not (0.0 < prune_ratio < 1.0):
            raise ValueError(f"prune_ratio must be in (0.0, 1.0), got {prune_ratio}")
        if prune_after_layer < 0:
            raise ValueError(f"prune_after_layer must be >= 0, got {prune_after_layer}")

        self.prune_ratio = prune_ratio
        self.prune_after_layer = prune_after_layer
        self._is_qwen3: bool = False
        self._last_prune_log: dict | None = None

    @property
    def backend_name(self) -> str:
        return "mlx-fastv"

    def load(self) -> None:
        super().load()
        model_type = getattr(self._model.vision_tower, 'model_type', '')
        self._is_qwen3 = model_type in ('qwen3_vl', 'qwen3_5', 'qwen3_5_moe')
        logger.info(
            "[FastV] Loaded model with prune_ratio=%.2f, prune_after_layer=%d",
            self.prune_ratio, self.prune_after_layer,
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
        """Run inference with FastV visual token pruning."""
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

        # Step 1: Run vision encoder normally (no ToMe)
        dtype = model.vision_tower.patch_embed.proj.weight.dtype
        pv = pixel_values.astype(dtype)

        vision_output = model.vision_tower(pv, grid_thw, output_hidden_states=False)

        deepstack_embeds = None
        if isinstance(vision_output, tuple):
            hidden_states = vision_output[0]
            if len(vision_output) > 1:
                deepstack_embeds = vision_output[1]
        else:
            hidden_states = vision_output

        original_count = hidden_states.shape[0]

        # Step 2: Build full input sequence (all visual tokens)
        video_token_id, image_token_id = self._get_visual_token_ids(model)

        n_video = mx.sum(input_ids == video_token_id).item()
        visual_token_id = video_token_id if n_video > 0 else image_token_id

        text_embeds = model.language_model.model.embed_tokens(input_ids)

        if self._is_qwen3:
            final_embeds, image_mask = model.merge_input_ids_with_image_features(
                hidden_states, text_embeds, input_ids,
                image_token_id, video_token_id,
            )
        else:
            final_embeds = model.merge_input_ids_with_image_features(
                image_token_id, video_token_id,
                hidden_states, text_embeds, input_ids,
            )

        # Build visual_mask: True for visual token positions
        visual_mask = (input_ids[0] == visual_token_id)

        # Step 3: Compute position IDs for full sequence
        new_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        kw_grid = {}
        if n_video > 0:
            kw_grid["video_grid_thw"] = grid_thw
        else:
            kw_grid["image_grid_thw"] = grid_thw

        position_ids, rope_deltas = model.language_model.get_rope_index(
            input_ids, attention_mask=new_mask, **kw_grid,
        )

        # Step 4: Compute importance via LLM layer attention
        importance = self._compute_visual_importance(
            model.language_model, final_embeds, visual_mask, position_ids,
        )

        # Step 5: Prune - keep top (1-prune_ratio) visual tokens
        n_visual = int(visual_mask.sum().item())
        n_keep = max(4, int(n_visual * (1 - self.prune_ratio)))

        keep_visual_indices = mx.argsort(importance)[::-1][:n_keep]
        keep_visual_indices = mx.sort(keep_visual_indices)  # preserve order

        # Step 6: Build pruned sequence
        pruned_embeds, pruned_ids = self._prune_visual_tokens(
            final_embeds, input_ids, visual_mask, keep_visual_indices,
        )

        compressed_count = n_keep

        self._last_prune_log = {
            "original_visual": n_visual,
            "kept_visual": n_keep,
            "pruned_visual": n_visual - n_keep,
            "reduction_pct": (1 - n_keep / max(n_visual, 1)) * 100,
        }
        logger.info(
            "FastV prune: %d → %d visual tokens (%.0f%% reduction)",
            n_visual, n_keep, self._last_prune_log["reduction_pct"],
        )

        # Step 7: Compute position IDs for pruned sequence
        compressed_grid = CompressedMLXBackend._compute_compressed_grid(
            grid_thw, original_count, compressed_count,
        )

        # Handle grid alignment (same as ToMe backend)
        grid_count = self._original_token_count(compressed_grid)
        if grid_count != compressed_count:
            logger.debug(
                "FastV grid alignment: compressed=%d, grid=%d",
                compressed_count, grid_count,
            )

        pruned_mask = mx.ones(pruned_ids.shape, dtype=mx.int32)
        kw_grid_pruned = {}
        if n_video > 0:
            kw_grid_pruned["video_grid_thw"] = compressed_grid
        else:
            kw_grid_pruned["image_grid_thw"] = compressed_grid

        position_ids, rope_deltas = model.language_model.get_rope_index(
            pruned_ids, attention_mask=pruned_mask, **kw_grid_pruned,
        )
        model.language_model._position_ids = position_ids
        model.language_model._rope_deltas = rope_deltas

        # Step 8: Generate from pruned sequence
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

        # Deepstack for Qwen3
        prefill_kwargs = {}
        if self._is_qwen3 and deepstack_embeds is not None:
            # Recompute image_mask for pruned sequence
            pruned_visual_mask = (pruned_ids[0] == visual_token_id)
            visual_pos_masks = pruned_visual_mask[None, :, None] if pruned_visual_mask is not None else None
            prefill_kwargs["visual_pos_masks"] = visual_pos_masks
            prefill_kwargs["deepstack_visual_embeds"] = mx.eval(deepstack_embeds)

        # Prefill
        tic = time.perf_counter()
        outputs = model.language_model(
            pruned_ids,
            inputs_embeds=pruned_embeds,
            cache=prompt_cache,
            **prefill_kwargs,
        )
        logits = outputs.logits[:, -1, :]
        quantize_cache_fn(prompt_cache)

        logprobs = logits - mx.logsumexp(logits)
        y = sampler(logprobs)
        mx.eval(y)

        prompt_time = time.perf_counter() - tic
        prompt_token_count = pruned_ids.size
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

    def _compute_visual_importance(
        self, language_model, embeds, visual_mask, position_ids,
    ):
        """Run first N LLM layers, extract attention importance at layer N.

        Computes text→visual attention scores using Q, K projections at
        the target layer. Returns importance scores per visual token.
        """
        import mlx.core as mx

        h = embeds
        layers = language_model.model.layers

        # Validate layer index
        n_layers = len(layers)
        target_layer = min(self.prune_after_layer, n_layers - 1)

        # Run layers 0..target_layer
        for i in range(target_layer + 1):
            layer = layers[i]

            if i == target_layer:
                # Extract Q, K before running full attention
                normed = layer.input_layernorm(h)
                B, L, D = normed.shape

                q = layer.self_attn.q_proj(normed)
                k = layer.self_attn.k_proj(normed)

                # Attribute names differ: mlx_vlm uses n_heads/n_kv_heads,
                # some models use num_heads/num_kv_heads
                n_heads = getattr(layer.self_attn, 'n_heads',
                          getattr(layer.self_attn, 'num_heads', None))
                n_kv = getattr(layer.self_attn, 'n_kv_heads',
                       getattr(layer.self_attn, 'num_kv_heads', None))
                head_dim = layer.self_attn.head_dim

                q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)

                # Apply RoPE
                cos, sin = layer.self_attn.rotary_emb(k, position_ids)
                q, k = self._apply_mrope(q, k, cos, sin)

                # GQA: repeat k for all heads
                if n_kv < n_heads:
                    repeats = n_heads // n_kv
                    k = mx.repeat(k, repeats, axis=1)

                # Identify text and visual positions
                vis_idx = _bool_indices(visual_mask)
                text_idx = _bool_indices(~visual_mask)

                n_vis = int(visual_mask.astype(mx.int32).sum().item())
                n_text = int((~visual_mask).astype(mx.int32).sum().item())
                if n_text == 0 or n_vis == 0:
                    return mx.ones((max(n_vis, 1),))

                q_text = q[:, :, text_idx, :]     # (B, H, n_text, D)
                k_vis = k[:, :, vis_idx, :]        # (B, H, n_vis, D)

                # text→visual attention scores
                scores = (q_text @ k_vis.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
                # Mean across batch, heads, text queries
                importance = scores.mean(axis=(0, 1, 2))  # (n_vis,)
                mx.eval(importance)
                return importance
            else:
                # Run full layer (without cache — just forward pass)
                normed = layer.input_layernorm(h)
                attn_out = layer.self_attn(
                    normed, mask=None, cache=None, position_ids=position_ids,
                )
                h = h + attn_out
                normed = layer.post_attention_layernorm(h)
                h = h + layer.mlp(normed)

        # Shouldn't reach here
        n_vis = int(visual_mask.astype(mx.int32).sum().item())
        return mx.ones((max(n_vis, 1),))

    @staticmethod
    def _apply_mrope(q, k, cos, sin):
        """Apply multimodal rotary position embeddings."""
        import mlx.core as mx

        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return mx.concatenate([-x2, x1], axis=-1)

        # Broadcast cos/sin to match q/k shape
        # rotary_emb returns (cos, sin) each of shape (B, L, D)
        # We need (B, 1, L, D) for broadcasting with (B, H, L, D)
        if cos.ndim == 3:
            cos = cos[:, None, :, :]
            sin = sin[:, None, :, :]
        elif cos.ndim == 4 and cos.shape[1] != q.shape[1]:
            # MRoPE: cos is (B, 1, L, D) already
            pass

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    @staticmethod
    def _prune_visual_tokens(embeds, input_ids, visual_mask, keep_indices):
        """Remove pruned visual tokens from sequence."""
        import mlx.core as mx

        vis_positions = _bool_indices(visual_mask)
        keep_positions = vis_positions[keep_indices]

        text_positions = _bool_indices(~visual_mask)
        all_keep = mx.sort(mx.concatenate([text_positions, keep_positions]))

        pruned_embeds = embeds[:, all_keep, :]
        pruned_ids = input_ids[:, all_keep]
        return pruned_embeds, pruned_ids

    @staticmethod
    def _original_token_count(grid_thw) -> int:
        """Compute expected token count without pruning (after PatchMerger)."""
        spatial_merge_size = 2
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * (h // spatial_merge_size) * (w // spatial_merge_size)
        return total

    def stream_generate(
        self, frames: np.ndarray, prompt: str, *,
        max_tokens: int = 512, temperature: float = 0.0, top_p: float = 1.0,
    ) -> Generator[StreamChunk, None, None]:
        result = self.generate(frames, prompt, max_tokens=max_tokens,
                               temperature=temperature, top_p=top_p)
        yield StreamChunk(
            text=result.text, finished=True,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
        )

    @property
    def prune_log(self) -> dict | None:
        return self._last_prune_log
