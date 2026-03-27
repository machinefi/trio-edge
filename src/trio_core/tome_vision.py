"""ToMe wrappers for Qwen vision encoders.

Wraps VisionModel.__call__ to inject token merging between ViT blocks.
Does NOT modify mlx_vlm source — uses wrap-and-delegate approach.

Supports:
  - Qwen2.5-VL (windowed attention, fullatt_block_indexes)
  - Qwen3-VL (full attention, deepstack features)
"""

from __future__ import annotations

import logging

import mlx.core as mx

from trio_core.tome import bipartite_soft_matching, compute_k_metric, merge_tokens

logger = logging.getLogger(__name__)


class BaseToMeVisionWrapper:
    """Base class for ToMe vision encoder wrappers.

    Subclasses implement model-specific block loop and window handling.
    Core merging logic is shared.
    """

    def __init__(
        self,
        vision_model,
        r: int = 8,
        skip_first: int = 2,
        skip_last: int = 2,
        min_keep_ratio: float = 0.3,
        metric: str = "keys",  # "keys" or "hidden"
        adaptive: bool = False,
    ):
        self.vision_model = vision_model
        self.r = r
        self.skip_first = skip_first
        self.skip_last = skip_last
        self.min_keep_ratio = min_keep_ratio
        self.metric = metric
        self.adaptive = adaptive
        self._merge_log: list[dict] = []
        self._initial_seq_len: int = 0

    def _get_layer_r(self, layer_num: int, n_blocks: int) -> int:
        """Get r for this layer. Linear ramp if adaptive, else constant."""
        if not self._should_merge(layer_num, n_blocks):
            return 0
        if not self.adaptive:
            return self.r
        start = self.skip_first
        end = n_blocks - self.skip_last
        position = layer_num - start + 1
        n_mergeable = end - start
        if n_mergeable <= 0:
            return 0
        return max(0, int(self.r * position / n_mergeable))

    def _should_merge(self, layer_num: int, n_blocks: int) -> bool:
        """Whether to merge after this layer. Override for model-specific skip logic."""
        return (
            self.r > 0 and layer_num >= self.skip_first and layer_num < (n_blocks - self.skip_last)
        )

    def _get_metric(
        self,
        hidden_states: mx.array,
        block,
        rotary_pos_emb: mx.array | None = None,
    ) -> mx.array:
        """Get similarity metric for bipartite matching."""
        if self.metric == "keys":
            return compute_k_metric(hidden_states, block, rotary_pos_emb)
        return hidden_states

    def _merge_segment(
        self,
        hidden_states: mx.array,
        rotary_pos_emb: mx.array,
        token_size: mx.array | None,
        block,
        orig_len: int,
        r: int | None = None,
    ) -> tuple[mx.array, mx.array, mx.array | None]:
        """Merge tokens in a single segment (window or full sequence)."""
        n = hidden_states.shape[0]
        min_keep = max(4, int(orig_len * self.min_keep_ratio))
        max_removable = max(0, n - min_keep)
        r = min(r if r is not None else self.r, n // 2, max_removable)

        if r <= 0:
            return hidden_states, rotary_pos_emb, token_size

        metric = self._get_metric(hidden_states, block, rotary_pos_emb)
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r)

        merged_hs, merged_size = merge_tokens(hidden_states, dst_idx, src_dst_map, token_size)
        merged_pe = rotary_pos_emb[dst_idx]

        return merged_hs, merged_pe, merged_size

    def __getattr__(self, name):
        return getattr(self.vision_model, name)


class ToMeQwen25VisionWrapper(BaseToMeVisionWrapper):
    """ToMe wrapper for Qwen2.5-VL vision encoder.

    Handles windowed attention: merges within each window independently.
    Skips full-attention layers (fullatt_block_indexes).
    """

    def _should_merge(self, layer_num: int, n_blocks: int) -> bool:
        return (
            super()._should_merge(layer_num, n_blocks)
            and layer_num not in self.vision_model.fullatt_block_indexes
        )

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        output_hidden_states=None,
    ) -> mx.array:
        vm = self.vision_model
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
        idx = mx.array(idx, dtype=mx.int32)
        cu_window_seqlens = cu_window_seqlens[idx]

        seq_len, _ = hidden_states.shape
        smu = vm.spatial_merge_unit

        # Reorder by window index
        hidden_states = hidden_states.reshape(seq_len // smu, smu, -1)
        hidden_states = hidden_states[window_index, :, :]
        hidden_states = hidden_states.reshape(seq_len, -1)

        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // smu, smu, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :]
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)

        # Full-attention cu_seqlens
        batch_size = grid_thw.shape[0]
        cu_seqlens = []
        for i in range(batch_size):
            seq_l = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_l, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        n_blocks = len(vm.blocks)
        token_size = None
        initial_window_sizes = {}

        for layer_num, blk in enumerate(vm.blocks):
            if layer_num in vm.fullatt_block_indexes:
                cu_seqlens_now = cu_seqlens
            else:
                cu_seqlens_now = cu_window_seqlens

            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens_now,
                rotary_pos_emb=rotary_pos_emb,
            )

            layer_r = self._get_layer_r(layer_num, n_blocks)
            if layer_r > 0:
                before = hidden_states.shape[0]

                # Track initial window sizes
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

                # Scale full-attn cu_seqlens
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
            pad_len = smu - remainder
            hidden_states = mx.pad(hidden_states, ((0, pad_len), (0, 0)))

        hidden_states = vm.merger(hidden_states)

        # Reverse window ordering
        n_out = hidden_states.shape[0]
        if window_index.shape[0] == n_out:
            reverse_indices = mx.argsort(window_index, axis=0)
            hidden_states = hidden_states[reverse_indices, :]

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
        r: int | None = None,
    ):
        effective_r = r if r is not None else self.r
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
            min_keep = max(4, int(orig_size * self.min_keep_ratio))
            max_removable = max(0, window_len - min_keep)
            r_window = min(effective_r, window_len // 2, max_removable)

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


class ToMeQwen3VisionWrapper(BaseToMeVisionWrapper):
    """ToMe wrapper for Qwen3-VL vision encoder.

    Simpler than Qwen2.5-VL: no windowed attention, merges across full sequence.
    Handles deepstack feature extraction at intermediate layers.
    """

    def __call__(
        self,
        hidden_states: mx.array,
        grid_thw: mx.array,
        **kwargs,
    ) -> tuple[mx.array, list]:
        vm = self.vision_model
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
        batch_size = grid_thw.shape[0]
        cu_seqlens = []
        for i in range(batch_size):
            seq_l = grid_thw[i, 1] * grid_thw[i, 2]
            cu_seqlens.append(mx.repeat(seq_l, grid_thw[i, 0]))
        cu_seqlens = mx.concatenate(cu_seqlens)
        cu_seqlens = mx.cumsum(cu_seqlens.astype(mx.int32), axis=0)
        cu_seqlens = mx.pad(cu_seqlens, (1, 0), mode="constant", constant_values=0)

        n_blocks = len(vm.blocks)
        token_size = None
        deepstack_feature_lists = []

        for layer_num, blk in enumerate(vm.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                rotary_pos_emb=rotary_pos_emb,
            )

            # Deepstack: extract features at specified layers
            if layer_num in vm.deepstack_visual_indexes:
                ds_merger = vm.deepstack_merger_list[vm.deepstack_visual_indexes.index(layer_num)]
                deepstack_feature_lists.append(ds_merger(hidden_states))

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

                # Update cu_seqlens
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
            pad_len = smu - remainder
            hidden_states = mx.pad(hidden_states, ((0, pad_len), (0, 0)))

        hidden_states = vm.merger(hidden_states)

        total_merged = sum(e["merged"] for e in self._merge_log)
        if total_merged > 0:
            logger.info(
                "ToMe: merged %d tokens across %d layers", total_merged, len(self._merge_log)
            )

        return hidden_states, deepstack_feature_lists


# Convenience alias for backward compat
ToMeVisionWrapper = ToMeQwen25VisionWrapper
