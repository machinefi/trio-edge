"""StreamMem — bounded KV cache for continuous video streams.

Manages KV cache accumulation across frames: visual tokens from each frame
are appended to the KV cache after prefill. When the total visual token count
exceeds the budget, low-saliency tokens are evicted and optionally merged
into compressed "prototype" tokens.

Saliency is computed using proxy query attention: chat template end-tokens
(e.g. <|im_end|><|im_start|>assistant) serve as a stand-in query to score
visual token importance via dot-product attention at a chosen layer.

Supports all model architectures:
- Pure KVCache (Qwen2.5-VL, Qwen3-VL, Gemma3, SmolVLM): evict all layers
- Hybrid KVCache+ArraysCache (Qwen3.5): evict KVCache layers only,
  DeltaNet layers have fixed-size recurrent state (already bounded)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class EvictionStats:
    """Statistics from a single eviction event."""

    tokens_before: int
    tokens_after: int
    n_evicted: int
    n_prototypes: int


class StreamingMemory:
    """Manages KV cache accumulation and eviction for continuous streams.

    Does NOT own the KV cache — operates on cache owned by PromptCache.
    This is a metadata manager + eviction controller.
    """

    def __init__(
        self,
        budget: int = 6000,
        prototype_ratio: float = 0.1,
        saliency_layer: int = 2,
        n_sink_tokens: int = 4,
    ):
        self.budget = budget
        self.prototype_ratio = prototype_ratio
        self.saliency_layer = saliency_layer
        self.n_sink_tokens = n_sink_tokens
        self._total_visual_tokens = 0
        self._text_prefix_len = 0
        self._frame_boundaries: List[int] = []  # cumulative visual token counts

    @property
    def over_budget(self) -> bool:
        return self._total_visual_tokens > self.budget

    def append_frame(self, n_visual_tokens: int, text_prefix_len: int = 0):
        """Register new frame's visual tokens after prefill.

        Args:
            n_visual_tokens: Number of visual tokens in this frame.
            text_prefix_len: Number of text tokens before visual tokens (first frame only).
        """
        if self._text_prefix_len == 0 and text_prefix_len > 0:
            self._text_prefix_len = text_prefix_len

        self._total_visual_tokens += n_visual_tokens
        self._frame_boundaries.append(self._total_visual_tokens)
        logger.debug(
            "StreamMem: appended %d visual tokens (total=%d, budget=%d)",
            n_visual_tokens,
            self._total_visual_tokens,
            self.budget,
        )

    def _get_visual_positions(self) -> mx.array:
        """Return indices of all visual token positions in the KV cache."""
        start = self._text_prefix_len
        end = start + self._total_visual_tokens
        return mx.arange(start, end)

    def _find_saliency_layer(self, kv_cache: List) -> int:
        """Find a KVCache layer (with keys) for saliency scoring.

        For pure KVCache models: uses self.saliency_layer directly.
        For hybrid Qwen3.5: DeltaNet layers have no keys — find first
        standard attention layer at or after self.saliency_layer.
        """
        # Try from saliency_layer onwards
        for i in range(self.saliency_layer, len(kv_cache)):
            if self._is_kvcache_layer(kv_cache[i]):
                return i
        # Fallback: search from beginning
        for i in range(len(kv_cache)):
            if self._is_kvcache_layer(kv_cache[i]):
                return i
        return -1

    @staticmethod
    def _is_kvcache_layer(cache_entry) -> bool:
        """Check if a cache entry is a standard KVCache (not ArraysCache/DeltaNet)."""
        return hasattr(cache_entry, "keys") and hasattr(cache_entry, "offset")

    def compute_saliency(
        self,
        kv_cache: List,
        language_model: nn.Module,
        proxy_query_ids: mx.array,
    ) -> mx.array:
        """Score all visual tokens using proxy query attention.

        Uses cached K values at the saliency layer. Projects proxy query
        through Q, computes dot-product attention -> saliency scores.

        Args:
            kv_cache: The KV cache list (one entry per layer).
            language_model: The LLM (model.language_model).
            proxy_query_ids: Token IDs for proxy query (e.g. last 8 tokens).

        Returns:
            Saliency scores of shape (n_visual_tokens,).
        """
        layer_idx = self._find_saliency_layer(kv_cache)
        if layer_idx < 0:
            # No KVCache layers — return uniform saliency
            return mx.ones(self._total_visual_tokens)

        layer = language_model.model.layers[layer_idx]
        cache_entry = kv_cache[layer_idx]

        # Get cached K for all positions: (B, n_kv_heads, seq_len, head_dim)
        valid_k = cache_entry.keys[:, :, : cache_entry.offset, :]

        # Embed proxy query + project to Q space
        proxy_embeds = language_model.model.embed_tokens(proxy_query_ids)
        normed = layer.input_layernorm(proxy_embeds)
        attn = layer.self_attn
        q = attn.q_proj(normed)

        # Model-agnostic attribute access (Qwen vs Gemma vs SmolVLM)
        n_heads = getattr(attn, "n_heads", getattr(attn, "num_heads", None))
        n_kv = getattr(attn, "n_kv_heads", getattr(attn, "num_kv_heads", None))
        head_dim = attn.head_dim

        if n_heads is None or n_kv is None:
            return mx.ones(self._total_visual_tokens)

        # Reshape Q -> (B, n_heads, Q_len, head_dim)
        B, Q_len, _ = q.shape
        q = q.reshape(B, Q_len, n_heads, head_dim).transpose(0, 2, 1, 3)

        # Extract visual positions' K
        vis_positions = self._get_visual_positions()
        k_vis = valid_k[:, :, vis_positions, :]

        # Expand for GQA if needed
        if n_kv < n_heads:
            k_vis = mx.repeat(k_vis, n_heads // n_kv, axis=1)

        # Attention scores -> saliency
        scores = (q @ k_vis.transpose(0, 1, 3, 2)) * (head_dim**-0.5)
        # Mean over batch, heads, and query positions
        saliency = scores.mean(axis=(0, 1, 2))  # (n_vis,)

        mx.eval(saliency)
        return saliency

    def evict_and_merge(self, kv_cache: List, saliency: mx.array) -> EvictionStats:
        """Evict low-saliency tokens, merge into prototypes.

        Only operates on KVCache layers. DeltaNet ArraysCache layers
        (Qwen3.5) are skipped — their fixed-size state is already bounded.
        """
        n_vis = self._total_visual_tokens
        tokens_before = n_vis

        # How many to keep vs evict
        n_keep = self.budget
        n_evict = n_vis - n_keep
        if n_evict <= 0:
            return EvictionStats(tokens_before, n_vis, 0, 0)

        # Protect attention sink tokens — force-keep the first N visual tokens
        # Clamp to budget so sinks don't consume the entire budget
        effective_sinks = min(self.n_sink_tokens, n_keep - 1) if n_keep > 1 else 0
        if effective_sinks > 0 and n_vis > effective_sinks:
            sink_boost = mx.zeros_like(saliency)
            sink_boost[:effective_sinks] = 1e9
            saliency = saliency + sink_boost

        # Sort by saliency — keep top-scoring
        sorted_indices = mx.argsort(saliency)  # ascending: low saliency first
        evict_indices = sorted_indices[:n_evict]
        keep_indices = mx.sort(sorted_indices[n_evict:])  # preserve order

        # Number of prototype tokens from evicted
        n_prototypes = max(1, int(n_evict * self.prototype_ratio))

        # Compute prototype indices (evenly spaced groups)
        mx.eval(evict_indices, keep_indices)

        text_start = 0
        text_end = self._text_prefix_len
        vis_start = text_end
        # Suffix tokens start after all visual tokens
        suffix_start = vis_start + n_vis

        actual_prototypes = 0
        for layer_idx in range(len(kv_cache)):
            c = kv_cache[layer_idx]
            if not self._is_kvcache_layer(c):
                continue

            keys = c.keys[:, :, : c.offset, :]  # (B, H, S, D)
            values = c.values[:, :, : c.offset, :]  # (B, H, S, D)

            # Split: text prefix | visual tokens | text suffix
            k_text = keys[:, :, text_start:text_end, :]
            v_text = values[:, :, text_start:text_end, :]

            k_vis = keys[:, :, vis_start:suffix_start, :]
            v_vis = values[:, :, vis_start:suffix_start, :]

            k_suffix = keys[:, :, suffix_start:, :]
            v_suffix = values[:, :, suffix_start:, :]

            # Kept visual tokens
            k_kept = k_vis[:, :, keep_indices, :]
            v_kept = v_vis[:, :, keep_indices, :]

            # Prototype: weighted average of evicted tokens by saliency
            k_evicted = k_vis[:, :, evict_indices, :]
            v_evicted = v_vis[:, :, evict_indices, :]

            evict_saliency = saliency[evict_indices]
            # Normalize saliency as weights
            weights = mx.softmax(evict_saliency)  # (n_evict,)

            # Compute prototypes by grouping evicted tokens
            layer_proto_keys = []
            layer_proto_values = []
            chunk_size = max(1, n_evict // n_prototypes)
            for p in range(n_prototypes):
                start_idx = p * chunk_size
                end_idx = min((p + 1) * chunk_size, n_evict)
                if start_idx >= n_evict:
                    break
                w = weights[start_idx:end_idx]
                w = w / (w.sum() + 1e-8)
                # Weighted average: (B, H, chunk, D) -> (B, H, 1, D)
                w_expanded = w[None, None, :, None]  # broadcast
                pk = (k_evicted[:, :, start_idx:end_idx, :] * w_expanded).sum(axis=2, keepdims=True)
                pv = (v_evicted[:, :, start_idx:end_idx, :] * w_expanded).sum(axis=2, keepdims=True)
                layer_proto_keys.append(pk)
                layer_proto_values.append(pv)

            if layer_proto_keys:
                k_protos = mx.concatenate(layer_proto_keys, axis=2)
                v_protos = mx.concatenate(layer_proto_values, axis=2)
                actual_prototypes = len(layer_proto_keys)
            else:
                k_protos = mx.zeros_like(k_kept[:, :, :0, :])
                v_protos = mx.zeros_like(v_kept[:, :, :0, :])

            # Reassemble: [text_prefix] + [kept visual] + [prototypes] + [suffix]
            new_keys = mx.concatenate([k_text, k_kept, k_protos, k_suffix], axis=2)
            new_values = mx.concatenate([v_text, v_kept, v_protos, v_suffix], axis=2)

            mx.eval(new_keys, new_values)

            # Update cache entry
            new_offset = new_keys.shape[2]
            c.offset = new_offset
            # Rewrite the keys/values arrays
            c.keys = new_keys
            c.values = new_values

        self._total_visual_tokens = n_keep + actual_prototypes

        stats = EvictionStats(
            tokens_before=tokens_before,
            tokens_after=self._total_visual_tokens,
            n_evicted=n_evict,
            n_prototypes=actual_prototypes,
        )
        return stats

    def maybe_evict(
        self,
        kv_cache: List,
        language_model: nn.Module,
        proxy_query_ids: mx.array,
    ) -> Optional[EvictionStats]:
        """Convenience: check budget, compute saliency, evict if needed."""
        if not self.over_budget:
            return None

        saliency = self.compute_saliency(kv_cache, language_model, proxy_query_ids)
        stats = self.evict_and_merge(kv_cache, saliency)
        return stats
