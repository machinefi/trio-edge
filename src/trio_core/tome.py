"""Token Merging (ToMe) — bipartite soft matching for visual token compression.

Implements the core algorithm from:
  Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023
  https://arxiv.org/abs/2210.09461

Model-agnostic core. Model-specific wrappers live in tome_vision.py.
"""

from __future__ import annotations

import mlx.core as mx


def bipartite_soft_matching(
    metric: mx.array,
    r: int,
) -> tuple[mx.array, mx.array]:
    """Bipartite soft matching to find r token pairs to merge.

    Args:
        metric: (N, D) token features for similarity (K from attention or hidden states)
        r: number of tokens to merge (remove) this step

    Returns:
        dst_idx: (N-r,) indices to gather kept tokens
        src_dst_map: (r, 2) pairs of (src_index, dst_index) for merging
    """
    n = metric.shape[0]
    if r <= 0 or n <= 1:
        return mx.arange(n), mx.zeros((0, 2), dtype=mx.int32)

    r = min(r, n // 2)

    # Partition into sets A (even) and B (odd)
    a_idx = mx.arange(0, n, 2)
    b_idx = mx.arange(1, n, 2)

    a_metric = metric[a_idx]
    b_metric = metric[b_idx]

    # Cosine similarity between A and B
    a_norm = mx.linalg.norm(a_metric, axis=1, keepdims=True)
    b_norm = mx.linalg.norm(b_metric, axis=1, keepdims=True)
    a_normed = a_metric / mx.maximum(a_norm, 1e-8)
    b_normed = b_metric / mx.maximum(b_norm, 1e-8)

    scores = a_normed @ b_normed.T  # (Na, Nb)

    # For each A token, find most similar B token
    max_b_for_a = mx.argmax(scores, axis=1)
    max_sim_for_a = mx.max(scores, axis=1)

    # Keep top-r most similar pairs
    top_a = mx.argsort(-max_sim_for_a)[:r]

    merge_a = a_idx[top_a]
    merge_b = b_idx[max_b_for_a[top_a]]
    src_dst_map = mx.stack([merge_a, merge_b], axis=1)

    merged_src_set = set(merge_a.tolist())
    keep_indices = [i for i in range(n) if i not in merged_src_set]
    dst_idx = mx.array(keep_indices, dtype=mx.int32)

    return dst_idx, src_dst_map


def merge_tokens(
    x: mx.array,
    dst_idx: mx.array,
    src_dst_map: mx.array,
    size: mx.array | None = None,
) -> tuple[mx.array, mx.array]:
    """Merge tokens using the mapping from bipartite_soft_matching.

    Args:
        x: (N, D) token features
        dst_idx: (N-r,) indices of tokens to keep
        src_dst_map: (r, 2) pairs of (src, dst) for weighted averaging
        size: (N, 1) token sizes for weighted merge. None = uniform.

    Returns:
        merged: (N-r, D) merged token features
        new_size: (N-r, 1) updated token sizes
    """
    n = x.shape[0]

    if size is None:
        size = mx.ones((n, 1))

    if src_dst_map.shape[0] == 0:
        return x, size

    merged = x[dst_idx]
    new_size = size[dst_idx]

    for i in range(src_dst_map.shape[0]):
        src_i = src_dst_map[i, 0].item()
        dst_i = src_dst_map[i, 1].item()

        dst_pos = mx.argmin(mx.abs(dst_idx - dst_i)).item()

        s_src = size[src_i]
        s_dst = new_size[dst_pos]
        total = s_src + s_dst

        merged[dst_pos] = (merged[dst_pos] * s_dst + x[src_i] * s_src) / total
        new_size[dst_pos] = total

    return merged, new_size


def compute_k_metric(
    hidden_states: mx.array, block,
    rotary_pos_emb: mx.array | None = None,
) -> mx.array:
    """Extract K (key) matrix from a ViT block's attention weights.

    Computes K = RoPE(W_k @ LayerNorm(hidden_states)), matching what the
    actual attention mechanism uses for similarity.

    Args:
        hidden_states: (N, D) token features
        block: ViT block with .norm1 and .attn.qkv
        rotary_pos_emb: (N, rope_dim) rotary position embeddings.
            If provided, applies RoPE to K for accurate similarity matching.
    """
    normed = block.norm1(hidden_states)
    seq_len = normed.shape[0]

    # QKV projection: (seq_len, 3 * hidden_dim)
    qkv = block.attn.qkv(normed)
    num_heads = block.attn.num_heads
    head_dim = block.attn.head_dim

    # Reshape to (seq_len, 3, num_heads, head_dim), extract K
    qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
    k = qkv[:, 1, :, :]  # (seq_len, num_heads, head_dim)

    # Apply rotary position embeddings if available
    if rotary_pos_emb is not None:
        # rotary_pos_emb: (seq_len, rope_dim) where rope_dim = head_dim // 2 * 2
        # apply_rotary expects (1, seq_len, num_heads, head_dim) and (seq_len, rope_dim)
        freqs = rotary_pos_emb.reshape(seq_len, -1)
        # Inline rotary application matching mlx_vlm's apply_rotary_pos_emb_vision
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        # cos/sin: (seq_len, rope_dim) → tile to (seq_len, head_dim)
        cos = mx.tile(cos, (1, 2))[:, :head_dim]  # (seq_len, head_dim)
        sin = mx.tile(sin, (1, 2))[:, :head_dim]
        # Expand for num_heads: (seq_len, 1, head_dim)
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)
        # Rotate half
        x1 = k[..., : head_dim // 2]
        x2 = k[..., head_dim // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        k = k * cos + rotated * sin

    # Flatten heads: (seq_len, num_heads * head_dim) = (seq_len, hidden_dim)
    return k.reshape(seq_len, -1)
