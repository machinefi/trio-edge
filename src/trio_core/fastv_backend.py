"""MLX backend with FastV-style visual token pruning in LLM layers.

Mid-stream FastV: runs layers 0→N with KV cache, extracts text→visual
attention importance at layer N, prunes the KV cache in-place, then
continues layers N+1→end on the pruned sequence. Single-pass — no double
computation.

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
    """MLX backend with mid-stream FastV visual token pruning.

    Runs layers 0→N with KV cache, extracts text→visual attention importance
    at layer N, prunes the KV cache and hidden states in-place, then
    continues layers N+1→end. Single forward pass — zero double computation.
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
        """Run inference with mid-stream FastV visual token pruning.

        Single-pass: layers 0→N build KV cache, importance extracted at layer N,
        cache pruned in-place, layers N+1→end continue on shorter sequence.
        """
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

        # Step 3: Compute position IDs for full sequence (needed for decode offsets)
        new_mask = mx.ones(input_ids.shape, dtype=mx.int32)
        kw_grid = {}
        if n_video > 0:
            kw_grid["video_grid_thw"] = grid_thw
        else:
            kw_grid["image_grid_thw"] = grid_thw

        position_ids, rope_deltas = model.language_model.get_rope_index(
            input_ids, attention_mask=new_mask, **kw_grid,
        )

        # Warn if Qwen3 deepstack is present — mid-stream doesn't support it yet
        if self._is_qwen3 and deepstack_embeds is not None:
            logger.warning(
                "[FastV] Qwen3 deepstack_visual_embeds detected but not supported "
                "in mid-stream mode — deepstack will be ignored. Quality may degrade."
            )

        # Step 4: Create cache and run layers 0→target_layer WITH cache
        tic = time.perf_counter()
        prompt_cache = make_prompt_cache(model.language_model)
        h, importance = self._run_layers_with_cache(
            model.language_model, final_embeds, visual_mask, prompt_cache,
            position_ids,
        )

        # Step 5: Prune - keep top (1-prune_ratio) visual tokens
        n_visual = int(visual_mask.sum().item())
        n_keep = max(4, int(n_visual * (1 - self.prune_ratio)))

        keep_visual_indices = mx.argsort(importance)[::-1][:n_keep]
        keep_visual_indices = mx.sort(keep_visual_indices)  # preserve order

        # Step 6: Build pruned index (all text + kept visual positions)
        _, pruned_ids, all_keep = self._prune_visual_tokens(
            final_embeds, input_ids, visual_mask, keep_visual_indices,
        )

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

        # Step 7: Prune KV cache in-place (layers 0..target_layer)
        target_layer = min(self.prune_after_layer, len(model.language_model.model.layers) - 1)
        self._prune_kv_cache(prompt_cache, all_keep, target_layer + 1)

        # Step 8: Prune hidden state to match
        h = h[:, all_keep, :]

        # Step 9: Set position info for AR decode
        position_ids = position_ids[:, :, all_keep]
        max_pos = position_ids.max()
        rope_deltas = max_pos + 1 - pruned_ids.shape[1]
        model.language_model._position_ids = position_ids
        model.language_model._rope_deltas = rope_deltas

        # Step 10: Run remaining layers (target_layer+1 → end) with pruned cache + h
        start_layer = target_layer + 1
        normed = self._run_remaining_layers(
            model.language_model, h, prompt_cache, start_layer, position_ids,
        )

        # Step 11: Compute logits from normed output
        lm = model.language_model
        if lm.args.tie_word_embeddings:
            logits_all = lm.model.embed_tokens.as_linear(normed)
        else:
            logits_all = lm.lm_head(normed)
        logits = logits_all[:, -1, :]

        quantize_cache_fn = functools.partial(
            maybe_quantize_kv_cache,
            quantized_kv_start=5000,
            kv_group_size=64,
            kv_bits=None,
        )
        quantize_cache_fn(prompt_cache)

        sampler = make_sampler(temperature, top_p)
        logprobs = logits - mx.logsumexp(logits)
        y = sampler(logprobs)
        mx.eval(y)

        prompt_time = time.perf_counter() - tic
        prompt_token_count = pruned_ids.size
        prompt_tps = prompt_token_count / max(prompt_time, 1e-9)

        tokenizer = self._processor.tokenizer if hasattr(self._processor, "tokenizer") else self._processor
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

        detokenizer = self._processor.detokenizer
        detokenizer.reset()

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

    # ── Mid-stream helpers ──────────────────────────────────────────────────

    def _run_layers_with_cache(self, language_model, embeds, visual_mask,
                               cache, position_ids):
        """Run layers 0→target_layer with KV cache, extract importance at target.

        At the target layer, manually replicates Attention.__call__ to extract
        Q/K for importance scoring while also writing to KV cache and computing
        the real attention output. Zero-waste: the same Q/K/V serve both
        importance extraction and the actual forward pass.

        Args:
            language_model: The LLM module.
            embeds: (B, L, D) input embeddings.
            visual_mask: (L,) bool — True at visual token positions.
            cache: list of KVCache objects.
            position_ids: (3, B, L) MRoPE position IDs.

        Returns:
            (h, importance) — hidden state after layer target, and per-visual-token
            importance scores.
        """
        import mlx.core as mx
        from mlx_vlm.models.base import create_attention_mask

        h = embeds
        layers = language_model.model.layers
        n_layers = len(layers)
        target = min(self.prune_after_layer, n_layers - 1)

        mask = create_attention_mask(h, cache)

        for i in range(target + 1):
            layer = layers[i]

            if i == target:
                # ── Manual forward: extract importance + complete real attention ──
                # We call Attention.__call__ which handles MRoPE internally,
                # then separately compute importance from the pre-attention Q/K.
                normed = layer.input_layernorm(h)
                B, L, D = normed.shape
                attn = layer.self_attn

                q = attn.q_proj(normed)
                k = attn.k_proj(normed)
                v = attn.v_proj(normed)

                n_heads = getattr(attn, 'n_heads',
                          getattr(attn, 'num_heads', None))
                n_kv = getattr(attn, 'n_kv_heads',
                       getattr(attn, 'num_kv_heads', None))
                head_dim = attn.head_dim

                q = q.reshape(B, L, n_heads, head_dim).transpose(0, 2, 1, 3)
                k = k.reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)
                v = v.reshape(B, L, n_kv, head_dim).transpose(0, 2, 1, 3)

                # Apply MRoPE (same as Attention.__call__)
                cos, sin = attn.rotary_emb(v, position_ids)
                if self._is_qwen3:
                    from mlx_vlm.models.qwen3_vl.language import apply_multimodal_rotary_pos_emb
                else:
                    from mlx_vlm.models.qwen2_5_vl.language import apply_multimodal_rotary_pos_emb
                q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1)

                # Extract importance from post-RoPE Q/K (before cache update)
                importance = self._score_visual(q, k, visual_mask, n_heads, n_kv, head_dim)

                # Write to KV cache
                k, v = cache[i].update_and_fetch(k, v)

                # Apply mask
                layer_mask = mask
                if layer_mask is not None:
                    layer_mask = layer_mask[..., :k.shape[-2]]

                # Scaled dot-product attention
                output = mx.fast.scaled_dot_product_attention(
                    q, k, v, scale=attn.scale, mask=layer_mask,
                )
                output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
                attn_out = attn.o_proj(output)

                # Residual + MLP
                h = h + attn_out
                h = h + layer.mlp(layer.post_attention_layernorm(h))
            else:
                # Standard layer forward with cache + MRoPE position_ids
                h = layer(h, mask, cache[i], position_ids)

        mx.eval(h, importance)
        return h, importance

    @staticmethod
    def _score_visual(q, k, visual_mask, n_heads, n_kv, head_dim):
        """Compute text→visual attention importance from Q/K tensors.

        Args:
            q: (B, n_heads, L, head_dim) — post-RoPE queries
            k: (B, n_kv_heads, L, head_dim) — post-RoPE keys (before cache)
            visual_mask: (L,) bool — True at visual token positions
            n_heads, n_kv, head_dim: attention geometry

        Returns:
            importance: (n_visual,) — mean text→visual attention score per visual token
        """
        import mlx.core as mx

        vis_idx = _bool_indices(visual_mask)
        text_idx = _bool_indices(~visual_mask)

        n_vis = vis_idx.shape[0]
        n_text = text_idx.shape[0]
        if n_text == 0 or n_vis == 0:
            return mx.ones((max(n_vis, 1),))

        # GQA: expand K heads to match Q heads for scoring
        if n_kv < n_heads:
            k_expanded = mx.repeat(k, n_heads // n_kv, axis=1)
        else:
            k_expanded = k

        q_text = q[:, :, text_idx, :]      # (B, H, n_text, D)
        k_vis = k_expanded[:, :, vis_idx, :]  # (B, H, n_vis, D)

        # text→visual attention scores
        scores = (q_text @ k_vis.transpose(0, 1, 3, 2)) * (head_dim ** -0.5)
        # Mean across batch, heads, text queries → per visual token importance
        importance = scores.mean(axis=(0, 1, 2))  # (n_vis,)
        return importance

    @staticmethod
    def _prune_kv_cache(cache, keep_indices, n_layers):
        """Remove pruned positions from KV cache for layers 0..n_layers-1.

        Args:
            cache: list of KVCache objects (one per LLM layer)
            keep_indices: (n_keep,) int array — sequence positions to retain
            n_layers: number of cache layers to prune (0..n_layers-1)
        """
        import mlx.core as mx

        n_keep = keep_indices.shape[0]
        eval_tensors = []
        for i in range(n_layers):
            c = cache[i]
            if c.keys is not None:
                # Slice valid portion then select kept positions
                valid_k = c.keys[:, :, :c.offset, :]
                valid_v = c.values[:, :, :c.offset, :]
                c.keys = valid_k[:, :, keep_indices, :]
                c.values = valid_v[:, :, keep_indices, :]
                c.offset = n_keep
                eval_tensors.extend([c.keys, c.values])
        if eval_tensors:
            mx.eval(*eval_tensors)

    @staticmethod
    def _run_remaining_layers(language_model, h, cache, start_layer,
                               position_ids=None):
        """Run layers start_layer→end with existing KV cache.

        Layers before start_layer have already been run (cache populated).
        Layers from start_layer onward have empty caches and process the
        (pruned) hidden state as a fresh prefill.

        Args:
            language_model: The LLM module.
            h: (B, L_pruned, D) hidden state after pruning.
            cache: list of KVCache objects.
            start_layer: index of first remaining layer.
            position_ids: (3, B, L_pruned) MRoPE position IDs (pruned).

        Returns:
            Normalized hidden state (after final RMSNorm).
        """
        import mlx.core as mx
        from mlx_vlm.models.base import create_attention_mask

        layers = language_model.model.layers

        if start_layer >= len(layers):
            return language_model.model.norm(h)

        remaining_cache = cache[start_layer:]
        mask = create_attention_mask(h, remaining_cache)

        for i in range(start_layer, len(layers)):
            h = layers[i](h, mask, cache[i], position_ids)

        return language_model.model.norm(h)

    # ── Shared helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _prune_visual_tokens(embeds, input_ids, visual_mask, keep_indices):
        """Remove pruned visual tokens from sequence.

        Returns:
            (pruned_embeds, pruned_ids, all_keep_indices)
        """
        import mlx.core as mx

        vis_positions = _bool_indices(visual_mask)
        keep_positions = vis_positions[keep_indices]

        text_positions = _bool_indices(~visual_mask)
        all_keep = mx.sort(mx.concatenate([text_positions, keep_positions]))

        pruned_embeds = embeds[:, all_keep, :]
        pruned_ids = input_ids[:, all_keep]
        return pruned_embeds, pruned_ids, all_keep

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
