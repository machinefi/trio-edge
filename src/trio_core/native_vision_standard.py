"""Native ToMe wrapper for standard ViTs (SigLIP, etc.).

Unlike Qwen's windowed/MRoPE ViT, standard ViTs like SigLIP have:
  - Batched input: (B, L, D) not unbatched (L, D)
  - Learnable position embeddings (added upfront, no RoPE tracking)
  - Class token at position 0 (must preserve during merge)
  - No cu_seqlens, no windowed attention, no PatchMerger/deepstack
  - Separate q_proj/k_proj/v_proj (not combined qkv)

Currently used for: nanoLLaVA (SigLIP-SO400M)
"""

from __future__ import annotations

import logging

import mlx.core as mx

from trio_core.native_vision import _ToMeMixin
from trio_core.tome import bipartite_soft_matching, compute_k_metric, merge_tokens

logger = logging.getLogger(__name__)


class NativeToMeStandardVision(_ToMeMixin):
    """Standard ViT (SigLIP) with built-in Token Merging.

    Wraps the vision encoder and injects ToMe between transformer blocks.
    Class token (position 0) is preserved — only patch tokens are merged.
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

    def __call__(self, hidden_states, output_hidden_states=None, **kwargs):
        """Run vision encoder with ToMe.

        Args:
            hidden_states: Pixel values or pre-embedded patches.
                SigLIP flow: patch_embed → add position_embedding → blocks → post_layernorm
        """
        vm = self._vm
        self._merge_log = []

        # SigLIP: embeddings include patch_embed + position_embedding
        # The vision_model typically has an embeddings module
        if hasattr(vm, "embeddings"):
            hidden_states = vm.embeddings(hidden_states)
        elif hasattr(vm, "patch_embed"):
            hidden_states = vm.patch_embed(hidden_states)
            if hasattr(vm, "position_embedding"):
                hidden_states = hidden_states + vm.position_embedding.weight

        # Get blocks
        blocks = None
        if hasattr(vm, "encoder") and hasattr(vm.encoder, "layers"):
            blocks = vm.encoder.layers
        elif hasattr(vm, "blocks"):
            blocks = vm.blocks
        elif hasattr(vm, "layers"):
            blocks = vm.layers

        if blocks is None:
            logger.warning("Could not find transformer blocks in vision model, returning as-is")
            return hidden_states

        n_blocks = len(blocks)

        # Detect if there's a class token (SigLIP typically has one)
        # We'll preserve it during merging
        has_cls = hasattr(vm, "class_embedding") or hasattr(vm, "cls_token")

        # hidden_states shape: (B, L, D) for standard ViTs
        is_batched = hidden_states.ndim == 3
        if not is_batched:
            hidden_states = mx.expand_dims(hidden_states, 0)

        B = hidden_states.shape[0]
        token_sizes = [None] * B
        content_factor_computed = False

        for layer_num, blk in enumerate(blocks):
            hidden_states = blk(hidden_states)

            # Content-aware: compute diversity factor once
            if (
                self.tome_content_aware
                and not content_factor_computed
                and self._should_merge(layer_num, n_blocks)
            ):
                # Use first batch element for diversity estimation
                self._compute_content_factor(hidden_states[0])
                content_factor_computed = True

            layer_r = self._get_layer_r(layer_num, n_blocks)
            if layer_r > 0:
                before = hidden_states.shape[1]

                # Merge each batch element independently
                merged_batch = []
                new_sizes = []
                for b in range(B):
                    h_b = hidden_states[b]  # (L, D)

                    if has_cls:
                        # Separate class token and patch tokens
                        cls_token = h_b[:1]  # (1, D)
                        patch_tokens = h_b[1:]  # (L-1, D)
                        patch_size = token_sizes[b]

                        metric = self._get_metric(patch_tokens, blk)
                        n_patch = patch_tokens.shape[0]
                        min_keep = max(4, int((before - 1) * self.tome_min_keep_ratio))
                        max_removable = max(0, n_patch - min_keep)
                        r_eff = min(layer_r, n_patch // 2, max_removable)

                        if r_eff > 0:
                            dst_idx, src_dst_map = bipartite_soft_matching(metric, r_eff)
                            merged_patches, merged_size = merge_tokens(
                                patch_tokens,
                                dst_idx,
                                src_dst_map,
                                patch_size,
                            )
                        else:
                            merged_patches = patch_tokens
                            merged_size = patch_size

                        # Re-attach class token
                        merged_b = mx.concatenate([cls_token, merged_patches], axis=0)
                        new_sizes.append(merged_size)
                    else:
                        # No class token — merge all tokens
                        metric = self._get_metric(h_b, blk)
                        n = h_b.shape[0]
                        min_keep = max(4, int(before * self.tome_min_keep_ratio))
                        max_removable = max(0, n - min_keep)
                        r_eff = min(layer_r, n // 2, max_removable)

                        if r_eff > 0:
                            dst_idx, src_dst_map = bipartite_soft_matching(metric, r_eff)
                            merged_b, merged_size = merge_tokens(
                                h_b,
                                dst_idx,
                                src_dst_map,
                                token_sizes[b],
                            )
                        else:
                            merged_b = h_b
                            merged_size = token_sizes[b]
                        new_sizes.append(merged_size)

                    merged_batch.append(merged_b)

                token_sizes = new_sizes

                # Pad to same length within batch (for batched processing)
                if B > 1:
                    max_len = max(m.shape[0] for m in merged_batch)
                    padded = []
                    for m in merged_batch:
                        if m.shape[0] < max_len:
                            m = mx.pad(m, ((0, max_len - m.shape[0]), (0, 0)))
                        padded.append(m)
                    hidden_states = mx.stack(padded, axis=0)
                else:
                    hidden_states = mx.expand_dims(merged_batch[0], 0)

                after = hidden_states.shape[1]
                self._merge_log.append(
                    {
                        "layer": layer_num,
                        "before": before,
                        "after": after,
                        "merged": before - after,
                    }
                )

        # Post-layernorm
        if hasattr(vm, "post_layernorm"):
            hidden_states = vm.post_layernorm(hidden_states)
        elif hasattr(vm, "layernorm") and not hasattr(vm, "encoder"):
            hidden_states = vm.layernorm(hidden_states)

        if not is_batched:
            hidden_states = hidden_states[0]

        total_merged = sum(e["merged"] for e in self._merge_log)
        if total_merged > 0:
            logger.info(
                "ToMe (standard ViT): merged %d tokens across %d layers",
                total_merged,
                len(self._merge_log),
            )

        return hidden_states

    def _get_metric(self, hidden_states, block, rotary_pos_emb=None):
        """Get similarity metric — uses generalized compute_k_metric."""
        if self.tome_metric == "keys":
            return compute_k_metric(hidden_states, block, rotary_pos_emb)
        return hidden_states

    def __getattr__(self, name):
        if name.startswith("_") or name.startswith("tome_"):
            raise AttributeError(name)
        return getattr(self._vm, name)
