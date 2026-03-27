"""Native vision encoders with built-in Token Merging.

Subclasses of mlx-vlm's VisionModel with ToMe integrated directly into
the block loop. No monkey-patching — proper OO inheritance.

Weight-compatible: inherits exact module tree from parent, so weights
loaded by mlx-vlm transfer directly via model.update().

Supports:
  - Qwen2.5-VL (windowed attention, fullatt_block_indexes)
  - Qwen3-VL / Qwen3.5 (full attention, deepstack features)
"""

from __future__ import annotations

import logging

import mlx.core as mx

from trio_core.tome import (
    bipartite_soft_matching,
    compute_content_diversity,
    compute_k_metric,
    merge_tokens,
)

logger = logging.getLogger(__name__)


# ── Shared ToMe logic ───────────────────────────────────────────────────────


class _ToMeMixin:
    """Shared ToMe configuration and helpers for native vision models."""

    tome_r: int
    tome_skip_first: int
    tome_skip_last: int
    tome_min_keep_ratio: float
    tome_metric: str
    tome_adaptive: bool
    tome_content_aware: bool
    _merge_log: list[dict]
    _initial_seq_len: int
    _content_r_factor: float  # computed per-image, scales r

    def _init_tome(
        self,
        r: int,
        skip_first: int,
        skip_last: int,
        min_keep_ratio: float,
        metric: str,
        adaptive: bool,
        content_aware: bool = False,
    ):
        self.tome_r = r
        self.tome_skip_first = skip_first
        self.tome_skip_last = skip_last
        self.tome_min_keep_ratio = min_keep_ratio
        self.tome_metric = metric
        self.tome_adaptive = adaptive
        self.tome_content_aware = content_aware
        self._merge_log = []
        self._initial_seq_len = 0
        self._content_r_factor = 1.0  # default: no scaling

    def _should_merge(self, layer_num: int, n_blocks: int) -> bool:
        return (
            self.tome_r > 0
            and layer_num >= self.tome_skip_first
            and layer_num < (n_blocks - self.tome_skip_last)
        )

    def _get_layer_r(self, layer_num: int, n_blocks: int) -> int:
        if not self._should_merge(layer_num, n_blocks):
            return 0
        base_r = self.tome_r
        if self.tome_adaptive:
            start = self.tome_skip_first
            end = n_blocks - self.tome_skip_last
            position = layer_num - start + 1
            n_mergeable = end - start
            if n_mergeable <= 0:
                return 0
            base_r = max(0, int(self.tome_r * position / n_mergeable))
        # Content-aware scaling: reduce r for complex (high diversity) images
        if self.tome_content_aware:
            base_r = max(0, int(base_r * self._content_r_factor))
        return base_r

    def _compute_content_factor(self, hidden_states: mx.array) -> None:
        """Compute content-aware r scaling factor from ViT hidden states.

        Called once at the first mergeable layer. Maps diversity to r factor:
          diversity < 0.3 (simple/redundant) → factor ~1.0 (full merge)
          diversity > 0.7 (complex/diverse)  → factor ~0.2 (minimal merge)
        """
        diversity = compute_content_diversity(hidden_states)
        # Linear mapping: diversity 0.3→1.0, diversity 0.7→0.2
        # Clamped to [0.2, 1.0]
        factor = 1.0 - 2.0 * max(0.0, diversity - 0.3)
        self._content_r_factor = max(0.2, min(1.0, factor))
        logger.info(
            "Content-aware: diversity=%.3f → r_factor=%.2f (r %d→%d)",
            diversity,
            self._content_r_factor,
            self.tome_r,
            int(self.tome_r * self._content_r_factor),
        )

    def _get_metric(self, hidden_states, block, rotary_pos_emb=None):
        if self.tome_metric == "keys":
            return compute_k_metric(hidden_states, block, rotary_pos_emb)
        return hidden_states

    def _merge_segment(self, hidden_states, rotary_pos_emb, token_size, block, orig_len, r):
        n = hidden_states.shape[0]
        min_keep = max(4, int(orig_len * self.tome_min_keep_ratio))
        max_removable = max(0, n - min_keep)
        r = min(r, n // 2, max_removable)

        if r <= 0:
            return hidden_states, rotary_pos_emb, token_size

        metric = self._get_metric(hidden_states, block, rotary_pos_emb)
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r)
        merged_hs, merged_size = merge_tokens(hidden_states, dst_idx, src_dst_map, token_size)
        merged_pe = rotary_pos_emb[dst_idx]

        return merged_hs, merged_pe, merged_size


# ── Qwen2.5-VL ──────────────────────────────────────────────────────────────


class NativeToMeQwen25Vision(_ToMeMixin):
    """Qwen2.5-VL vision encoder with built-in Token Merging.

    Subclasses mlx-vlm's VisionModel. ToMe merging happens inside the block
    loop, respecting windowed attention boundaries.

    Usage:
        from trio_core.native_vision import create_tome_vision
        model.vision_tower = create_tome_vision(model.vision_tower, tome_r=4)
    """

    def __init__(
        self,
        original_vision,
        *,
        tome_r=8,
        skip_first=2,
        skip_last=2,
        min_keep_ratio=0.3,
        metric="hidden",
        adaptive=False,
        content_aware=False,
    ):
        # Store reference to original model (keeps all weights/modules)
        self._vm = original_vision
        self._init_tome(
            tome_r,
            skip_first,
            skip_last,
            min_keep_ratio,
            metric,
            adaptive,
            content_aware=content_aware,
        )

    def _should_merge(self, layer_num: int, n_blocks: int) -> bool:
        return (
            super()._should_merge(layer_num, n_blocks)
            and layer_num not in self._vm.fullatt_block_indexes
        )

    def __call__(self, hidden_states, grid_thw, output_hidden_states=None):
        vm = self._vm
        self._merge_log = []

        hidden_states = vm.patch_embed(hidden_states)
        rotary_pos_emb = vm.rot_pos_emb(grid_thw)
        window_index, cu_window_seqlens = vm.get_window_index(grid_thw)

        # Deduplicate cu_window_seqlens
        seen = set()
        idx = []
        for i, x in enumerate(cu_window_seqlens):
            if x not in seen:
                seen.add(x)
                idx.append(i)
        cu_window_seqlens = cu_window_seqlens[mx.array(idx, dtype=mx.int32)]

        seq_len, _ = hidden_states.shape
        smu = vm.spatial_merge_unit

        # Reorder by window index
        hidden_states = hidden_states.reshape(seq_len // smu, smu, -1)[window_index, :, :].reshape(
            seq_len, -1
        )
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // smu, smu, -1)[
            window_index, :, :
        ].reshape(seq_len, -1)

        # Full-attention cu_seqlens
        cu_seqlens = []
        for i in range(grid_thw.shape[0]):
            seq_l = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_l, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        n_blocks = len(vm.blocks)
        token_size = None
        initial_window_sizes = {}
        content_factor_computed = False

        for layer_num, blk in enumerate(vm.blocks):
            cu_seqlens_now = (
                cu_seqlens if layer_num in vm.fullatt_block_indexes else cu_window_seqlens
            )

            hidden_states = blk(
                hidden_states, cu_seqlens=cu_seqlens_now, rotary_pos_emb=rotary_pos_emb
            )

            # Content-aware: compute diversity factor once at first mergeable layer
            if (
                self.tome_content_aware
                and not content_factor_computed
                and self._should_merge(layer_num, n_blocks)
            ):
                self._compute_content_factor(hidden_states)
                content_factor_computed = True

            layer_r = self._get_layer_r(layer_num, n_blocks)
            if layer_r > 0:
                before = hidden_states.shape[0]

                if not initial_window_sizes:
                    seqlens = cu_window_seqlens.tolist()
                    for w in range(len(seqlens) - 1):
                        initial_window_sizes[w] = int(seqlens[w + 1] - seqlens[w])

                hidden_states, rotary_pos_emb, cu_window_seqlens, token_size = (
                    self._merge_in_windows(
                        hidden_states,
                        rotary_pos_emb,
                        cu_window_seqlens,
                        token_size,
                        blk,
                        initial_window_sizes,
                        r=layer_r,
                    )
                )
                after = hidden_states.shape[0]

                if before != after:
                    ratio = after / before
                    cu_seqlens = mx.minimum(
                        (cu_seqlens.astype(mx.float32) * ratio).astype(mx.int32),
                        after,
                    )

                self._merge_log.append(
                    {
                        "layer": layer_num,
                        "before": before,
                        "after": after,
                        "merged": before - after,
                    }
                )

        # PatchMerger — pad to multiple of spatial_merge_unit
        current_len = hidden_states.shape[0]
        remainder = current_len % smu
        if remainder != 0:
            hidden_states = mx.pad(hidden_states, ((0, smu - remainder), (0, 0)))

        hidden_states = vm.merger(hidden_states)

        # Reverse window ordering
        n_out = hidden_states.shape[0]
        if window_index.shape[0] == n_out:
            hidden_states = hidden_states[mx.argsort(window_index, axis=0), :]

        total_merged = sum(e["merged"] for e in self._merge_log)
        if total_merged > 0:
            logger.info(
                "ToMe: merged %d tokens across %d layers", total_merged, len(self._merge_log)
            )

        return hidden_states

    def _merge_in_windows(
        self,
        hidden_states,
        rotary_pos_emb,
        cu_window_seqlens,
        token_size,
        block,
        initial_window_sizes,
        r,
    ):
        n_windows = cu_window_seqlens.shape[0] - 1
        seqlens_list = cu_window_seqlens.tolist()

        merged_hs, merged_pe, new_seqlens, all_sizes = [], [], [0], []

        for w in range(n_windows):
            start, end = int(seqlens_list[w]), int(seqlens_list[w + 1])
            window_len = end - start

            w_hs = hidden_states[start:end]
            w_pe = rotary_pos_emb[start:end]
            w_size = token_size[start:end] if token_size is not None else None

            if window_len <= 1:
                merged_hs.append(w_hs)
                merged_pe.append(w_pe)
                all_sizes.append(w_size if w_size is not None else mx.ones((window_len, 1)))
                new_seqlens.append(new_seqlens[-1] + window_len)
                continue

            orig_size = initial_window_sizes.get(w, window_len)
            min_keep = max(4, int(orig_size * self.tome_min_keep_ratio))
            max_removable = max(0, window_len - min_keep)
            r_window = min(r, window_len // 2, max_removable)

            if r_window <= 0:
                merged_hs.append(w_hs)
                merged_pe.append(w_pe)
                all_sizes.append(w_size if w_size is not None else mx.ones((window_len, 1)))
                new_seqlens.append(new_seqlens[-1] + window_len)
                continue

            metric = self._get_metric(w_hs, block, w_pe)
            dst_idx, src_dst_map = bipartite_soft_matching(metric, r_window)
            m_hs, m_size = merge_tokens(w_hs, dst_idx, src_dst_map, w_size)
            m_pe = w_pe[dst_idx]

            merged_hs.append(m_hs)
            merged_pe.append(m_pe)
            all_sizes.append(m_size)
            new_seqlens.append(new_seqlens[-1] + m_hs.shape[0])

        return (
            mx.concatenate(merged_hs, axis=0),
            mx.concatenate(merged_pe, axis=0),
            mx.array(new_seqlens, dtype=cu_window_seqlens.dtype),
            mx.concatenate(all_sizes, axis=0),
        )

    def __getattr__(self, name):
        """Delegate attribute access to original vision model."""
        if name.startswith("_") or name.startswith("tome_"):
            raise AttributeError(name)
        return getattr(self._vm, name)


# ── Qwen3-VL / Qwen3.5 ─────────────────────────────────────────────────────


class NativeToMeQwen3Vision(_ToMeMixin):
    """Qwen3-VL/Qwen3.5 vision encoder with built-in Token Merging.

    Simpler than Qwen2.5-VL: no windowed attention. Handles deepstack
    feature extraction at intermediate layers.
    """

    def __init__(
        self,
        original_vision,
        *,
        tome_r=8,
        skip_first=2,
        skip_last=2,
        min_keep_ratio=0.3,
        metric="hidden",
        adaptive=False,
        content_aware=False,
    ):
        self._vm = original_vision
        self._init_tome(
            tome_r,
            skip_first,
            skip_last,
            min_keep_ratio,
            metric,
            adaptive,
            content_aware=content_aware,
        )

    def __call__(self, hidden_states, grid_thw, **kwargs):
        vm = self._vm
        self._merge_log = []

        hidden_states = vm.patch_embed(hidden_states)
        pos_embeds = vm.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds
        rotary_pos_emb = vm.rot_pos_emb(grid_thw)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        self._initial_seq_len = seq_len

        # cu_seqlens (no windows in Qwen3-VL)
        cu_seqlens = []
        for i in range(grid_thw.shape[0]):
            seq_l = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_l, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        n_blocks = len(vm.blocks)
        token_size = None
        deepstack_feature_lists = []
        content_factor_computed = False

        for layer_num, blk in enumerate(vm.blocks):
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)

            # Deepstack: extract features at specified layers
            if layer_num in vm.deepstack_visual_indexes:
                ds_merger = vm.deepstack_merger_list[vm.deepstack_visual_indexes.index(layer_num)]
                deepstack_feature_lists.append(ds_merger(hidden_states))

            # Content-aware: compute diversity factor once at first mergeable layer
            if (
                self.tome_content_aware
                and not content_factor_computed
                and self._should_merge(layer_num, n_blocks)
            ):
                self._compute_content_factor(hidden_states)
                content_factor_computed = True

            # ToMe merge
            layer_r = self._get_layer_r(layer_num, n_blocks)
            if layer_r > 0:
                before = hidden_states.shape[0]

                hidden_states, rotary_pos_emb, token_size = self._merge_segment(
                    hidden_states,
                    rotary_pos_emb,
                    token_size,
                    blk,
                    self._initial_seq_len,
                    r=layer_r,
                )
                after = hidden_states.shape[0]

                if before != after:
                    ratio = after / before
                    cu_seqlens = mx.minimum(
                        (cu_seqlens.astype(mx.float32) * ratio).astype(mx.int32),
                        after,
                    )

                self._merge_log.append(
                    {
                        "layer": layer_num,
                        "before": before,
                        "after": after,
                        "merged": before - after,
                    }
                )

        # PatchMerger
        smu = vm.spatial_merge_size**2
        current_len = hidden_states.shape[0]
        remainder = current_len % smu
        if remainder != 0:
            hidden_states = mx.pad(hidden_states, ((0, smu - remainder), (0, 0)))

        hidden_states = vm.merger(hidden_states)

        total_merged = sum(e["merged"] for e in self._merge_log)
        if total_merged > 0:
            logger.info(
                "ToMe: merged %d tokens across %d layers", total_merged, len(self._merge_log)
            )

        return hidden_states, deepstack_feature_lists

    def __getattr__(self, name):
        if name.startswith("_") or name.startswith("tome_"):
            raise AttributeError(name)
        return getattr(self._vm, name)


# ── Factory ──────────────────────────────────────────────────────────────────


def create_tome_vision(
    vision_model,
    *,
    tome_r: int = 8,
    skip_first: int = 2,
    skip_last: int = 2,
    min_keep_ratio: float = 0.3,
    metric: str = "hidden",
    adaptive: bool = False,
    content_aware: bool = False,
):
    """Create a native ToMe vision encoder from a loaded mlx-vlm vision model.

    Auto-detects model type and returns the appropriate native implementation:
      - Qwen2.5-VL: windowed attention, fullatt_block_indexes
      - Qwen3-VL / Qwen3.5: full attention, deepstack features
      - SigLIP (nanoLLaVA): standard ViT, batched, class token
      - InternViT: NOT supported (pixel_shuffle disrupts spatial structure)
      - FastVLM: NOT supported (CNN encoder, not ViT)

    Args:
        vision_model: Loaded mlx-vlm VisionModel instance.
        tome_r: Number of tokens to merge per layer.
        skip_first: Skip first N layers (let model establish representations).
        skip_last: Skip last N layers (preserve final features).
        min_keep_ratio: Minimum fraction of tokens to keep.
        metric: Similarity metric ("keys" or "hidden").
        adaptive: Linear ramp r from 0 to tome_r across layers.
        content_aware: Dynamically scale r based on image complexity.

    Returns:
        Native vision model with built-in ToMe.

    Raises:
        ValueError: If vision model architecture doesn't support ToMe.
    """
    model_type = getattr(vision_model, "model_type", "")
    vt_type = type(vision_model).__name__.lower()

    # Reject unsupported architectures
    if "intern" in vt_type:
        raise ValueError(
            "InternViT does not support ToMe — pixel_shuffle after ViT "
            "disrupts spatial structure. Use Compressed instead."
        )
    if "fastvlm" in vt_type or "fastvithd" in vt_type:
        raise ValueError(
            "FastVLM does not support ToMe — CNN encoder is fundamentally "
            "different from ViT. Use Compressed instead."
        )

    kwargs = dict(
        tome_r=tome_r,
        skip_first=skip_first,
        skip_last=skip_last,
        min_keep_ratio=min_keep_ratio,
        metric=metric,
        adaptive=adaptive,
        content_aware=content_aware,
    )

    if model_type in ("qwen3_vl", "qwen3_5", "qwen3_5_moe"):
        cls = NativeToMeQwen3Vision
    elif model_type in ("qwen2_vl", "qwen2_5_vl", ""):
        # Check if this is a standard ViT (SigLIP, etc.)
        if "siglip" in vt_type or "clip" in vt_type:
            from trio_core.native_vision_standard import NativeToMeStandardVision

            cls = NativeToMeStandardVision
        else:
            # Default to Qwen2.5-VL (backward compatible)
            cls = NativeToMeQwen25Vision
    else:
        # Unknown model_type — try standard ViT wrapper
        if "siglip" in vt_type or "clip" in vt_type:
            from trio_core.native_vision_standard import NativeToMeStandardVision

            cls = NativeToMeStandardVision
        else:
            cls = NativeToMeQwen25Vision

    native = cls(vision_model, **kwargs)
    logger.info(
        "[NativeToMe] Created %s (r=%d, metric=%s, min_keep=%.0f%%, adaptive=%s, content_aware=%s)",
        cls.__name__,
        tome_r,
        metric,
        min_keep_ratio * 100,
        adaptive,
        content_aware,
    )
    return native
