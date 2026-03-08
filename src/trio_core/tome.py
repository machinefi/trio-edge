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

    src_indices = src_dst_map[:, 0]  # (r,)
    dst_values = src_dst_map[:, 1]   # (r,)

    # Find position of each dst_value in dst_idx via broadcast argmin
    # dst_values: (r,) → (r, 1), dst_idx: (N-r,) → (1, N-r)
    diffs = mx.abs(dst_values[:, None] - dst_idx[None, :])  # (r, N-r)
    dst_positions = mx.argmin(diffs, axis=1)  # (r,)

    # Check for duplicate destinations (multiple sources → same dst)
    pos_list = dst_positions.tolist()
    has_duplicates = len(set(pos_list)) < len(pos_list)

    if not has_duplicates:
        # No conflicts — fully vectorized
        s_src = size[src_indices]              # (r, 1)
        s_dst = new_size[dst_positions]        # (r, 1)
        total = s_src + s_dst                  # (r, 1)
        src_features = x[src_indices]          # (r, D)
        dst_features = merged[dst_positions]   # (r, D)
        new_values = (dst_features * s_dst + src_features * s_src) / total
        merged[dst_positions] = new_values
        new_size[dst_positions] = total
    else:
        # Rare: multiple sources → same dest, fall back to sequential
        for i in range(src_dst_map.shape[0]):
            pos = pos_list[i]
            s_src = size[src_indices[i]]
            s_dst = new_size[pos]
            total = s_src + s_dst
            merged[pos] = (merged[pos] * s_dst + x[src_indices[i]] * s_src) / total
            new_size[pos] = total

    return merged, new_size


def compute_content_diversity(hidden_states: mx.array, sample_size: int = 128) -> float:
    """Measure content diversity via inter-token cosine similarity variance.

    Low diversity (high mean similarity) → redundant tokens → safe to merge aggressively.
    High diversity (low mean similarity) → complex content → merge conservatively.

    Args:
        hidden_states: (N, D) token features from a ViT layer.
        sample_size: Max tokens to sample for efficiency (O(n²) otherwise).

    Returns:
        Diversity score in [0, 1]. 0 = all identical, 1 = maximally diverse.
    """
    n = hidden_states.shape[0]
    if n <= 1:
        return 0.0

    # Subsample for efficiency
    if n > sample_size:
        indices = mx.arange(0, n, n // sample_size)[:sample_size]
        hidden_states = hidden_states[indices]
        n = hidden_states.shape[0]

    # Normalize
    norms = mx.linalg.norm(hidden_states, axis=1, keepdims=True)
    normed = hidden_states / mx.maximum(norms, 1e-8)

    # Pairwise cosine similarity matrix
    sim_matrix = normed @ normed.T  # (N, N)

    # Mean off-diagonal similarity
    # Mask diagonal with 0, sum, divide by n*(n-1)
    mask = 1.0 - mx.eye(n)
    mean_sim = mx.sum(sim_matrix * mask).item() / max(n * (n - 1), 1)

    # Diversity = 1 - mean_similarity
    # mean_sim ∈ [-1, 1] but in practice ViT tokens are non-negative → [0, 1]
    return float(max(0.0, min(1.0, 1.0 - mean_sim)))


def compute_k_metric(
    hidden_states: mx.array, block,
    rotary_pos_emb: mx.array | None = None,
) -> mx.array:
    """Extract K (key) matrix from a ViT block's attention weights.

    Computes K = RoPE(W_k @ LayerNorm(hidden_states)), matching what the
    actual attention mechanism uses for similarity.

    Supports multiple ViT architectures:
      - Combined qkv + norm1 (Qwen ViT, InternViT)
      - Separate q_proj/k_proj/v_proj + layer_norm1 (SigLIP)

    Args:
        hidden_states: (N, D) token features
        block: ViT block with norm + attention projections
        rotary_pos_emb: (N, rope_dim) rotary position embeddings.
            If provided, applies RoPE to K for accurate similarity matching.
    """
    # Find the pre-attention norm (use `is not None` — nn.Module can be falsy)
    norm = getattr(block, 'norm1', None)
    if norm is None:
        norm = getattr(block, 'layer_norm1', None)
    if norm is None:
        return hidden_states

    normed = norm(hidden_states)
    seq_len = normed.shape[0]

    # Find the attention module
    attn = getattr(block, 'attn', None)
    if attn is None:
        attn = getattr(block, 'self_attn', None)
    if attn is None:
        return hidden_states

    # Extract K based on attention architecture
    if hasattr(attn, 'qkv'):
        # Combined QKV projection (Qwen ViT, InternViT)
        qkv = attn.qkv(normed)
        num_heads = attn.num_heads
        head_dim = getattr(attn, 'head_dim', None) or (qkv.shape[-1] // (3 * num_heads))

        qkv = qkv.reshape(seq_len, 3, num_heads, head_dim)
        k = qkv[:, 1, :, :]  # (seq_len, num_heads, head_dim)
    elif hasattr(attn, 'k_proj'):
        # Separate projections (SigLIP, standard ViT)
        k_out = attn.k_proj(normed)
        num_heads = getattr(attn, 'num_heads',
                   getattr(attn, 'n_heads', 1))
        head_dim = getattr(attn, 'head_dim', k_out.shape[-1] // num_heads)
        k = k_out.reshape(seq_len, num_heads, head_dim)
    else:
        return hidden_states

    # Apply rotary position embeddings if available
    if rotary_pos_emb is not None:
        freqs = rotary_pos_emb.reshape(seq_len, -1)
        cos = mx.cos(freqs)
        sin = mx.sin(freqs)
        cos = mx.tile(cos, (1, 2))[:, :head_dim]
        sin = mx.tile(sin, (1, 2))[:, :head_dim]
        cos = mx.expand_dims(cos, axis=1)
        sin = mx.expand_dims(sin, axis=1)
        x1 = k[..., : head_dim // 2]
        x2 = k[..., head_dim // 2 :]
        rotated = mx.concatenate([-x2, x1], axis=-1)
        k = k * cos + rotated * sin

    # Flatten heads: (seq_len, num_heads * head_dim) = (seq_len, hidden_dim)
    return k.reshape(seq_len, -1)
