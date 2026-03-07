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
        tome_r: int = 0,
        tome_metric: str = "hidden",
        tome_min_keep_ratio: float = 0.3,
        tome_adaptive: bool = False,
        **kwargs,
    ):
        super().__init__(model_name, **kwargs)

        if not (0.0 < prune_ratio < 1.0):
            raise ValueError(f"prune_ratio must be in (0.0, 1.0), got {prune_ratio}")
        if prune_after_layer < 0:
            raise ValueError(f"prune_after_layer must be >= 0, got {prune_after_layer}")

        self.prune_ratio = prune_ratio
        self.prune_after_layer = prune_after_layer
        self.tome_r = tome_r
        self.tome_metric = tome_metric
        self.tome_min_keep_ratio = tome_min_keep_ratio
        self.tome_adaptive = tome_adaptive
        self._adapter = None
        self._last_prune_log: dict | None = None

    @property
    def backend_name(self) -> str:
        return "mlx-fastv"

    def load(self) -> None:
        super().load()

        from trio_core.model_adapter import get_adapter
        self._adapter = get_adapter(self._model)

        # Optionally wrap vision tower with ToMe (compound: ToMe + FastV)
        if self.tome_r > 0 and self._adapter.supports_tome:
            from trio_core.native_vision import create_tome_vision

            self._tome_wrapper = create_tome_vision(
                self._model.vision_tower,
                tome_r=self.tome_r,
                metric=self.tome_metric,
                min_keep_ratio=self.tome_min_keep_ratio,
                adaptive=self.tome_adaptive,
            )
            self._model.vision_tower = self._tome_wrapper
            logger.info(
                "[FastV+ToMe] Wrapped vision_tower with ToMe r=%d, metric=%s",
                self.tome_r, self.tome_metric,
            )
        else:
            self._tome_wrapper = None
            if self.tome_r > 0 and not self._adapter.supports_tome:
                logger.warning(
                    "[FastV] Model family '%s' does not support ToMe — "
                    "running FastV only.", self._adapter.family,
                )

        logger.info(
            "[FastV] Loaded model with prune_ratio=%.2f, prune_after_layer=%d",
            self.prune_ratio, self.prune_after_layer,
        )

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
        adapter = self._adapter

        # Step 1: Run vision encoder
        vision_out = adapter.run_vision_encoder(pixel_values, grid_thw)
        hidden_states = vision_out.hidden_states
        deepstack_embeds = vision_out.deepstack_embeds

        if grid_thw is not None:
            original_count = adapter.original_token_count(grid_thw)
        else:
            # Single-image models (InternVL, nanoLLaVA) don't produce grid_thw
            original_count = hidden_states.shape[0]
        compressed_count = hidden_states.shape[0]

        # Step 2: Build input sequence (trim if ToMe compressed visual tokens)
        video_token_id, image_token_id = adapter.get_visual_token_ids()

        n_video = mx.sum(input_ids == video_token_id).item()
        visual_token_id = video_token_id if n_video > 0 else image_token_id

        # If ToMe compressed tokens, trim input_ids to match
        original_input_ids = input_ids  # keep for grid alignment rebuild
        if self._tome_wrapper is not None and compressed_count < original_count:
            from trio_core.compressed_backend import CompressedMLXBackend

            logger.info(
                "[FastV+ToMe] Vision tokens: %d → %d (%.0f%% ToMe reduction)",
                original_count, compressed_count,
                (1 - compressed_count / max(original_count, 1)) * 100,
            )

            # Compute compressed grid for position IDs
            grid_thw = CompressedMLXBackend._compute_compressed_grid(
                grid_thw, original_count, compressed_count,
                spatial_merge_size=adapter.spatial_merge_size,
            )
            # Align to grid (rounding may change count)
            grid_count = self._original_token_count(grid_thw)
            if grid_count != compressed_count:
                if grid_count > compressed_count:
                    pad = mx.repeat(hidden_states[-1:], grid_count - compressed_count, axis=0)
                    hidden_states = mx.concatenate([hidden_states, pad])
                else:
                    hidden_states = hidden_states[:grid_count]
                compressed_count = grid_count

            # Trim visual placeholder tokens in input_ids to match final count
            # Use original_input_ids (has all visual placeholders) so we can
            # both shrink and grow relative to compressed_count
            ids_list = original_input_ids[0].tolist()
            new_ids = []
            vis_count = 0
            for tid in ids_list:
                if tid == visual_token_id:
                    vis_count += 1
                    if vis_count <= compressed_count:
                        new_ids.append(tid)
                else:
                    new_ids.append(tid)
            input_ids = mx.array([new_ids], dtype=input_ids.dtype)

        text_embeds = model.language_model.model.embed_tokens(input_ids)

        merge_result = adapter.merge_visual_features(hidden_states, text_embeds, input_ids)
        final_embeds = merge_result.embeds

        # Build visual_mask: True for visual token positions
        visual_mask = (input_ids[0] == visual_token_id)

        # Step 3: Compute position IDs for sequence (uses compressed grid if ToMe active)
        position_ids = None
        rope_deltas = None
        if adapter.uses_mrope:
            new_mask = mx.ones(input_ids.shape, dtype=mx.int32)
            kw_grid = {}
            if n_video > 0:
                kw_grid["video_grid_thw"] = grid_thw
            else:
                kw_grid["image_grid_thw"] = grid_thw

            position_ids, rope_deltas = adapter.compute_position_ids(
                input_ids, attention_mask=new_mask, **kw_grid,
            )

        # Warn if Qwen3 deepstack is present — mid-stream doesn't support it yet
        if adapter.has_deepstack and deepstack_embeds is not None:
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

        # Step 9: Set position info for AR decode (MRoPE models only)
        if adapter.uses_mrope and position_ids is not None:
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
        # mlx_vlm may return "causal" string instead of array — not supported
        # by manual attention or DeltaNet layers. During prefill with empty cache,
        # None is correct (full bidirectional attention over the prompt).
        if isinstance(mask, str):
            mask = None

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

                # Apply RoPE via adapter (handles MRoPE vs standard)
                q, k = self._adapter.apply_rope_at_layer(q, k, v, position_ids, layer, cache_offset=0)

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
                # Standard layer forward with cache (adapter handles position_ids)
                h = self._adapter.call_layer(layer, h, mask, cache[i], position_ids)

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

    def _run_remaining_layers(self, language_model, h, cache, start_layer,
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
            h = self._adapter.call_layer(layers[i], h, mask, cache[i], position_ids)

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

    def _original_token_count(self, grid_thw) -> int:
        """Compute expected token count without pruning (after PatchMerger)."""
        if grid_thw is None:
            return 0  # Single-image models; caller should use hidden_states.shape
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
