"""Visual token compression — reduce visual tokens before LLM.

Implements token pruning/merging between the vision encoder output and
the language model input, as described in:
  - NeurIPS 2024: "Efficient Large Multi-modal Models via Visual Context Compression"
  - ICLR 2025: "Inference Optimal VLMs Need Fewer Visual Tokens and More Parameters"

The key insight: visual tokens from the vision encoder have significant
spatial redundancy (sky, walls, backgrounds produce near-identical tokens).
Compressing these tokens reduces prefill latency (∝ seq_len²) and KV cache
memory (∝ seq_len) with minimal quality loss.

Compression point in the pipeline:
    pixel_values → VisionEncoder → PatchMerger → [hidden_states]
                                                        ↓
                                              ★ COMPRESS HERE ★
                                                        ↓
                                          merge_with_text_embeddings
                                                        ↓
                                                  LLM.generate()

Usage:
    compressor = TokenCompressor(strategy="similarity", ratio=0.5)
    compressed = compressor.compress(hidden_states)  # (N, D) → (N*ratio, D)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Result of visual token compression."""
    compressed: object      # mx.array (N_compressed, hidden_dim)
    original_count: int     # number of tokens before compression
    compressed_count: int   # number of tokens after compression
    ratio: float            # actual compression ratio (compressed/original)


class TokenCompressor:
    """Compress visual tokens between vision encoder and LLM.

    Strategies:
        uniform:     Keep every N-th token (simplest, fast)
        similarity:  Merge adjacent tokens above similarity threshold
        attention:   Score tokens by self-attention, drop low-attention ones
    """

    def __init__(
        self,
        strategy: str = "similarity",
        ratio: float = 0.5,
        similarity_threshold: float = 0.9,
    ):
        """
        Args:
            strategy: Compression strategy ("uniform", "similarity", "attention")
            ratio: Target compression ratio (0.5 = keep 50% of tokens)
            similarity_threshold: For similarity strategy, merge above this threshold
        """
        self.strategy = strategy
        self.ratio = ratio
        self.similarity_threshold = similarity_threshold

    def compress(self, hidden_states, grid_thw=None) -> CompressionResult:
        """Compress visual token embeddings.

        Args:
            hidden_states: (N, hidden_dim) visual token embeddings from vision encoder
            grid_thw: optional (batch, 3) grid dimensions for spatial awareness

        Returns:
            CompressionResult with compressed tokens
        """
        import mlx.core as mx

        n_tokens = hidden_states.shape[0]
        target_count = max(1, int(n_tokens * self.ratio))

        if self.strategy == "uniform":
            compressed = self._uniform(hidden_states, target_count)
        elif self.strategy == "similarity":
            compressed = self._similarity_merge(hidden_states, target_count)
        elif self.strategy == "attention":
            compressed = self._attention_score(hidden_states, target_count)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        mx.eval(compressed)
        actual_count = compressed.shape[0]

        return CompressionResult(
            compressed=compressed,
            original_count=n_tokens,
            compressed_count=actual_count,
            ratio=actual_count / n_tokens,
        )

    def _uniform(self, hidden_states, target_count):
        """Keep every N-th token uniformly."""
        import mlx.core as mx

        n = hidden_states.shape[0]
        # Select evenly spaced indices
        indices = mx.linspace(0, n - 1, target_count).astype(mx.int32)
        return hidden_states[indices]

    def _similarity_merge(self, hidden_states, target_count):
        """Merge adjacent tokens that are similar."""
        import mlx.core as mx

        n = hidden_states.shape[0]
        if n <= target_count:
            return hidden_states

        # Iteratively merge most similar pairs until we reach target
        # For efficiency, do it in rounds: find merge candidates, merge, repeat
        current = hidden_states
        current_n = n

        while current_n > target_count:
            if current_n <= 2:
                break

            # Recompute similarities for current tokens
            cur_norms = mx.linalg.norm(current, axis=1, keepdims=True)
            cur_normalized = current / mx.maximum(cur_norms, 1e-8)
            cur_sim = mx.sum(cur_normalized[:-1] * cur_normalized[1:], axis=1)

            # How many pairs to merge this round
            n_to_merge = min(
                max(1, (current_n - target_count)),
                current_n // 2,
            )

            # Find top-k most similar adjacent pairs (non-overlapping)
            # Sort by similarity descending
            sorted_indices = mx.argsort(-cur_sim)

            merge_set = set()
            merged_indices = []
            for idx in sorted_indices.tolist():
                if idx not in merge_set and (idx + 1) not in merge_set:
                    merged_indices.append(idx)
                    merge_set.add(idx)
                    merge_set.add(idx + 1)
                    if len(merged_indices) >= n_to_merge:
                        break

            if not merged_indices:
                break

            # Build new token list: merged pairs averaged, others kept
            new_tokens = []
            i = 0
            while i < current_n:
                if i in merge_set and (i + 1) < current_n and (i + 1) in merge_set:
                    # Merge pair: average embeddings
                    merged = (current[i] + current[i + 1]) / 2.0
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(current[i])
                    i += 1

            current = mx.stack(new_tokens)
            current_n = current.shape[0]

        return current

    def _attention_score(self, hidden_states, target_count):
        """Score tokens by attention energy, keep top-k."""
        import mlx.core as mx

        n, d = hidden_states.shape
        if n <= target_count:
            return hidden_states

        # Simple self-attention score: use L2 norm as importance proxy
        # High-norm tokens tend to carry more information
        norms = mx.linalg.norm(hidden_states, axis=1)  # (N,)

        # Also consider variance from mean (uniqueness)
        mean_emb = mx.mean(hidden_states, axis=0, keepdims=True)  # (1, D)
        dist_from_mean = mx.linalg.norm(hidden_states - mean_emb, axis=1)  # (N,)

        # Combined score: tokens that are both high-norm and unique
        scores = norms * dist_from_mean

        # Keep top-k by score
        top_indices = mx.argsort(-scores)[:target_count]
        # Sort indices to maintain spatial order
        top_indices = mx.sort(top_indices)

        return hidden_states[top_indices]
