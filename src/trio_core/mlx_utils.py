"""MLX utility functions — inlined from mlx-vlm/mlx-lm to avoid heavy deps.

These are small, self-contained functions that only depend on mlx.core.
Inlining them removes the need for mlx-vlm (which pulls in datasets,
pyarrow, pandas — ~200MB of unused dependencies).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx

# ── Attention Mask (from mlx_lm.models.base) ────────────────────────────────


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
    right_padding: Optional[mx.array] = None,
    left_padding: Optional[mx.array] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    if right_padding is not None:
        mask = mask & (rinds < mx.expand_dims((offset + N) - right_padding, (1, 2, 3)))
    if left_padding is not None:
        mask = mask & (mx.expand_dims(left_padding, (1, 2, 3)) <= rinds)
    return mask


def create_attention_mask(
    h,
    cache=None,
    window_size: Optional[int] = None,
    return_array: bool = False,
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


# ── Rotary Position Embedding (from mlx_vlm.models.qwen*_vl.language) ───────


def _rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return mx.concatenate([-x2, x1], axis=-1)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1):
    """Apply Rotary Position Embedding with Multimodal Sections."""
    cos = mx.expand_dims(cos, axis=unqueeze_dim)
    sin = mx.expand_dims(sin, axis=unqueeze_dim)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ── Pixel Shuffle (from mlx_vlm.models.base) ────────────────────────────────


def pixel_shuffle(input_tensor, shuffle_ratio):
    """Pixel shuffle for vision feature downsampling (InternVL)."""
    batch_size, num_patches, channels = input_tensor.shape
    patch_size = int(math.sqrt(num_patches))
    input_tensor = input_tensor.reshape(batch_size, patch_size, patch_size, -1)
    batch_size, height, width, channels = input_tensor.shape
    reshaped = input_tensor.reshape(
        batch_size, height, int(width * shuffle_ratio), int(channels / shuffle_ratio)
    )
    reshaped = reshaped.transpose(0, 2, 1, 3)
    reshaped = reshaped.reshape(
        batch_size,
        int(height * shuffle_ratio),
        int(width * shuffle_ratio),
        int(channels / (shuffle_ratio**2)),
    )
    reshaped = reshaped.transpose(0, 2, 1, 3)
    return reshaped.reshape(batch_size, -1, reshaped.shape[-1])
