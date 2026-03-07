"""ModelAdapter — abstracts model-family-specific operations for backends.

Each VLM family (Qwen2.5-VL, Qwen3-VL, InternVL3, nanoLLaVA, FastVLM) has
different token ID attributes, merge function signatures, position ID
strategies, and vision encoder interfaces. ModelAdapter encapsulates these
differences so backends can be model-agnostic.

Usage:
    from trio_core.model_adapter import get_adapter
    adapter = get_adapter(model)
    vid_id, img_id = adapter.get_visual_token_ids()
    embeds = adapter.merge_visual_features(hidden_states, text_embeds, input_ids)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class VisionOutput:
    """Result from running the vision encoder."""
    hidden_states: object  # mx.array — (N, D) visual features
    deepstack_embeds: list | None = None  # Qwen3-VL deepstack features


@dataclass
class MergeResult:
    """Result from merging visual features into text embeddings."""
    embeds: object  # mx.array — (B, L, D) merged embeddings
    image_mask: object | None = None  # mx.array — Qwen3-VL image mask


# ── Abstract adapter ─────────────────────────────────────────────────────────

class ModelAdapter(ABC):
    """Abstract adapter for model-family-specific operations."""

    def __init__(self, model):
        self._model = model

    @property
    @abstractmethod
    def family(self) -> str:
        """Model family identifier."""

    @property
    def supports_tome(self) -> bool:
        """Whether in-ViT ToMe is applicable for this model."""
        return True

    @property
    def spatial_merge_size(self) -> int:
        """Spatial merge factor used by PatchMerger (2 for Qwen, 1 for LLaVA-style)."""
        return 2

    @property
    def has_deepstack(self) -> bool:
        """Whether the model uses deepstack visual features."""
        return False

    @property
    def uses_mrope(self) -> bool:
        """Whether the model uses 3D MRoPE (Qwen) vs standard 1D RoPE."""
        return False

    @abstractmethod
    def get_visual_token_ids(self) -> tuple:
        """Return (video_token_id, image_token_id)."""

    @abstractmethod
    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        """Run the vision encoder and return processed visual features."""

    @abstractmethod
    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        """Merge visual features into text embeddings at placeholder positions."""

    def compute_position_ids(self, input_ids, attention_mask, **grid_kwargs):
        """Compute position IDs for the merged sequence.

        Returns (position_ids, rope_deltas) for MRoPE models,
        or (None, None) for standard RoPE models.
        """
        return None, None

    def apply_rope_at_layer(self, q, k, v, position_ids, layer, cache_offset=0):
        """Apply rotary position embeddings at a specific LLM layer.

        Default: standard 1D RoPE via nn.RoPE(q/k, offset=cache_offset).
        Qwen adapters override for MRoPE with multimodal RoPE.
        """
        attn = layer.self_attn
        rope = getattr(attn, 'rotary_emb', getattr(attn, 'rope', None))
        q = rope(q, offset=cache_offset)
        k = rope(k, offset=cache_offset)
        return q, k

    def call_layer(self, layer, h, mask, cache, position_ids=None):
        """Call a decoder layer. Qwen passes position_ids; others don't."""
        return layer(h, mask, cache)

    def get_vision_dtype(self):
        """Get the dtype of the vision encoder weights."""
        vt = getattr(self._model, 'vision_tower',
                     getattr(self._model, 'vision_model', None))
        # Try common paths
        if hasattr(vt, 'patch_embed'):
            if hasattr(vt.patch_embed, 'proj'):
                return vt.patch_embed.proj.weight.dtype
            if hasattr(vt.patch_embed, 'linear_patches'):
                return vt.patch_embed.linear_patches.weight.dtype
        # Fallback for CNN encoders
        if hasattr(vt, 'model') and hasattr(vt.model, 'patch_embed'):
            return vt.model.patch_embed.proj.weight.dtype
        import mlx.core as mx
        return mx.float16

    def original_token_count(self, grid_thw) -> int:
        """Compute expected visual token count after PatchMerger/projector."""
        sms = self.spatial_merge_size
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * (h // sms) * (w // sms)
        return total


# ── Qwen2.5-VL ───────────────────────────────────────────────────────────────

class Qwen25VLAdapter(ModelAdapter):
    """Adapter for Qwen2.5-VL models."""

    @property
    def family(self) -> str:
        return "qwen2.5-vl"

    @property
    def spatial_merge_size(self) -> int:
        return 2

    @property
    def uses_mrope(self) -> bool:
        return True

    def get_visual_token_ids(self) -> tuple:
        cfg = self._model.config
        return cfg.video_token_id, cfg.image_token_id

    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        dtype = self.get_vision_dtype()
        pv = pixel_values.astype(dtype)
        hs = self._model.vision_tower(pv, grid_thw, output_hidden_states=False)
        return VisionOutput(hidden_states=hs)

    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        vid_id, img_id = self.get_visual_token_ids()
        embeds = self._model.merge_input_ids_with_image_features(
            img_id, vid_id, hidden_states, text_embeds, input_ids,
        )
        return MergeResult(embeds=embeds)

    def compute_position_ids(self, input_ids, attention_mask, **grid_kwargs):
        return self._model.language_model.get_rope_index(
            input_ids, attention_mask=attention_mask, **grid_kwargs,
        )

    def apply_rope_at_layer(self, q, k, v, position_ids, layer, cache_offset=0):
        attn = layer.self_attn
        cos, sin = attn.rotary_emb(v, position_ids)
        from mlx_vlm.models.qwen2_5_vl.language import apply_multimodal_rotary_pos_emb
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1)
        return q, k

    def call_layer(self, layer, h, mask, cache, position_ids=None):
        return layer(h, mask, cache, position_ids)


# ── Qwen3-VL / Qwen3.5 ──────────────────────────────────────────────────────

class Qwen3VLAdapter(ModelAdapter):
    """Adapter for Qwen3-VL and Qwen3.5 models."""

    @property
    def family(self) -> str:
        return "qwen3-vl"

    @property
    def spatial_merge_size(self) -> int:
        return 2

    @property
    def has_deepstack(self) -> bool:
        return True

    @property
    def uses_mrope(self) -> bool:
        return True

    def get_visual_token_ids(self) -> tuple:
        cfg = self._model.config
        vid_id = (
            getattr(cfg, 'video_token_index', None) or
            getattr(cfg, 'video_token_id', None)
        )
        img_id = (
            getattr(cfg, 'image_token_index', None) or
            getattr(cfg, 'image_token_id', None)
        )
        return vid_id, img_id

    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        dtype = self.get_vision_dtype()
        pv = pixel_values.astype(dtype)
        output = self._model.vision_tower(pv, grid_thw, output_hidden_states=False)

        if isinstance(output, tuple):
            hs = output[0]
            ds = output[1] if len(output) > 1 else None
        else:
            hs = output
            ds = None
        return VisionOutput(hidden_states=hs, deepstack_embeds=ds)

    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        vid_id, img_id = self.get_visual_token_ids()
        embeds, image_mask = self._model.merge_input_ids_with_image_features(
            hidden_states, text_embeds, input_ids,
            img_id, vid_id,
        )
        return MergeResult(embeds=embeds, image_mask=image_mask)

    def compute_position_ids(self, input_ids, attention_mask, **grid_kwargs):
        return self._model.language_model.get_rope_index(
            input_ids, attention_mask=attention_mask, **grid_kwargs,
        )

    def apply_rope_at_layer(self, q, k, v, position_ids, layer, cache_offset=0):
        attn = layer.self_attn
        cos, sin = attn.rotary_emb(v, position_ids)
        from mlx_vlm.models.qwen3_vl.language import apply_multimodal_rotary_pos_emb
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, unqueeze_dim=1)
        return q, k

    def call_layer(self, layer, h, mask, cache, position_ids=None):
        return layer(h, mask, cache, position_ids)


# ── InternVL3 ────────────────────────────────────────────────────────────────

class InternVLAdapter(ModelAdapter):
    """Adapter for InternVL3 models (InternViT + pixel_shuffle + MLP + Qwen2.5 LLM).

    ToMe not supported: pixel_shuffle after ViT disrupts spatial structure.
    """

    @property
    def family(self) -> str:
        return "internvl"

    @property
    def supports_tome(self) -> bool:
        return False

    @property
    def spatial_merge_size(self) -> int:
        return 1

    def get_visual_token_ids(self) -> tuple:
        cfg = self._model.config
        img_id = getattr(cfg, 'image_token_index',
                 getattr(cfg, 'image_token_id', None))
        # InternVL uses same token for video frames
        vid_id = img_id
        return vid_id, img_id

    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        dtype = self.get_vision_dtype()
        pv = pixel_values.astype(dtype)
        if pv.ndim == 5:
            pv = pv[0]
        # Full pipeline: vision_model → CLS removal → pixel_shuffle → mlp1
        hs, _, _ = self._model.vision_model(
            pv.transpose(0, 2, 3, 1), output_hidden_states=True,
        )
        hs = hs[:, 1:, :]  # remove CLS token
        from mlx_vlm.models.base import pixel_shuffle
        hs = pixel_shuffle(hs, shuffle_ratio=self._model.downsample_ratio)
        for layer in self._model.mlp1:
            hs = layer(hs)
        # Flatten batch: (B, N, D) → (B*N, D) to match Qwen format
        hs = hs.reshape(-1, hs.shape[-1])
        return VisionOutput(hidden_states=hs)

    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        # hidden_states is (N, D) flat; need (1, N, D) for merge
        hs = hidden_states[None] if hidden_states.ndim == 2 else hidden_states
        embeds = self._model._merge_input_ids_with_image_features(
            hs, text_embeds, input_ids,
        )
        if isinstance(embeds, tuple):
            return MergeResult(embeds=embeds[0], image_mask=embeds[1] if len(embeds) > 1 else None)
        return MergeResult(embeds=embeds)

    def original_token_count(self, grid_thw) -> int:
        # InternVL: no spatial merge, tokens = T * H * W
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * h * w
        return total


# ── nanoLLaVA (SigLIP) ───────────────────────────────────────────────────────

class LLaVAAdapter(ModelAdapter):
    """Adapter for nanoLLaVA (SigLIP + MLP + Qwen1.5 LLM).

    ToMe supported: SigLIP has no pixel_shuffle post-processing.
    """

    @property
    def family(self) -> str:
        return "nanollava"

    @property
    def supports_tome(self) -> bool:
        return True

    @property
    def spatial_merge_size(self) -> int:
        return 1

    def get_visual_token_ids(self) -> tuple:
        cfg = self._model.config
        img_id = getattr(cfg, 'image_token_index',
                 getattr(cfg, 'image_token_id', None))
        vid_id = img_id
        return vid_id, img_id

    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        dtype = self.get_vision_dtype()
        pv = pixel_values.astype(dtype)
        # Full pipeline: vision_tower → hidden_state[-1] → mm_projector
        *_, hidden_state = self._model.vision_tower(
            pv.transpose(0, 2, 3, 1), output_hidden_states=True,
        )
        image_features = hidden_state[-1].astype(pv.dtype)
        image_features = self._model.mm_projector(image_features)
        # Flatten: (B, N, D) → (B*N, D)
        hs = image_features.reshape(-1, image_features.shape[-1])
        return VisionOutput(hidden_states=hs)

    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        # hidden_states is (N, D) flat; need (1, N, D) for merge
        hs = hidden_states[None] if hidden_states.ndim == 2 else hidden_states
        embeds = self._model._prepare_inputs_for_multimodal(
            hs, text_embeds, input_ids,
        )
        if isinstance(embeds, tuple):
            return MergeResult(embeds=embeds[0], image_mask=embeds[1] if len(embeds) > 1 else None)
        return MergeResult(embeds=embeds)

    def original_token_count(self, grid_thw) -> int:
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * h * w
        return total


# ── FastVLM ──────────────────────────────────────────────────────────────────

class FastVLMAdapter(ModelAdapter):
    """Adapter for Apple FastVLM (FastViTHD CNN + MLP + Qwen2 LLM).

    ToMe not supported: CNN encoder, fundamentally different from ViT.
    """

    @property
    def family(self) -> str:
        return "fastvlm"

    @property
    def supports_tome(self) -> bool:
        return False

    @property
    def spatial_merge_size(self) -> int:
        return 1

    def get_visual_token_ids(self) -> tuple:
        cfg = self._model.config
        img_id = getattr(cfg, 'image_token_index',
                 getattr(cfg, 'image_token_id', None))
        vid_id = img_id
        return vid_id, img_id

    def run_vision_encoder(self, pixel_values, grid_thw=None) -> VisionOutput:
        dtype = self.get_vision_dtype()
        pv = pixel_values.astype(dtype)
        hs = self._model.vision_tower(pv, output_hidden_states=False)
        return VisionOutput(hidden_states=hs)

    def merge_visual_features(self, hidden_states, text_embeds, input_ids) -> MergeResult:
        vid_id, img_id = self.get_visual_token_ids()
        embeds = self._model.merge_input_ids_with_image_features(
            img_id, vid_id, hidden_states, text_embeds, input_ids,
        )
        if isinstance(embeds, tuple):
            return MergeResult(embeds=embeds[0], image_mask=embeds[1] if len(embeds) > 1 else None)
        return MergeResult(embeds=embeds)

    def get_vision_dtype(self):
        """FastVLM uses CNN encoder — different weight path."""
        import mlx.core as mx
        vt = self._model.vision_tower
        if hasattr(vt, 'model') and hasattr(vt.model, 'patch_embed'):
            pe = vt.model.patch_embed
            if hasattr(pe, 'proj'):
                return pe.proj.weight.dtype
        if hasattr(vt, 'patch_embed'):
            pe = vt.patch_embed
            if hasattr(pe, 'proj'):
                return pe.proj.weight.dtype
        return mx.float16

    def original_token_count(self, grid_thw) -> int:
        total = 0
        for i in range(grid_thw.shape[0]):
            t = grid_thw[i, 0].item()
            h = grid_thw[i, 1].item()
            w = grid_thw[i, 2].item()
            total += t * h * w
        return total


# ── Factory ──────────────────────────────────────────────────────────────────

def get_adapter(model) -> ModelAdapter:
    """Auto-detect model family and return the appropriate adapter.

    Detection strategy:
    1. Check vision_tower.model_type for Qwen variants
    2. Check config attributes for LLaVA-style models
    3. Check module names for specific architectures
    """
    # Qwen family detection via vision_tower model_type
    vt = getattr(model, 'vision_tower', None)
    model_type = getattr(vt, 'model_type', '') if vt else ''

    if model_type in ('qwen3_vl', 'qwen3_5', 'qwen3_5_moe'):
        logger.debug("Detected Qwen3-VL/Qwen3.5 model")
        return Qwen3VLAdapter(model)

    if model_type in ('qwen2_vl', 'qwen2_5_vl'):
        logger.debug("Detected Qwen2.5-VL model")
        return Qwen25VLAdapter(model)

    # Check for InternVL: has vision_model (not vision_tower) or InternViT markers
    cfg = getattr(model, 'config', None)
    model_type_cfg = getattr(cfg, 'model_type', '') if cfg else ''

    if 'internvl' in model_type_cfg.lower():
        logger.debug("Detected InternVL model")
        return InternVLAdapter(model)

    # Check for architecture-specific markers
    if vt is not None:
        vt_type = type(vt).__name__.lower()

        # InternViT marker
        if 'intern' in vt_type:
            logger.debug("Detected InternVL via vision tower type: %s", type(vt).__name__)
            return InternVLAdapter(model)

        # FastVLM: CNN-based encoder
        if 'fastvlm' in vt_type or 'fastvithd' in vt_type:
            logger.debug("Detected FastVLM via vision tower type: %s", type(vt).__name__)
            return FastVLMAdapter(model)

        # SigLIP marker (nanoLLaVA, Bunny, etc.)
        if 'siglip' in vt_type:
            logger.debug("Detected SigLIP-based model (LLaVA-style)")
            return LLaVAAdapter(model)

    # Check config model_type for known patterns
    if model_type_cfg:
        mt = model_type_cfg.lower()
        if 'bunny' in mt or 'llava' in mt or 'nanollava' in mt:
            logger.debug("Detected LLaVA-style model via config.model_type")
            return LLaVAAdapter(model)
        if 'fastvlm' in mt:
            logger.debug("Detected FastVLM via config.model_type")
            return FastVLMAdapter(model)

    # Default: Qwen2.5-VL (most common, backward compatible)
    logger.debug("Defaulting to Qwen2.5-VL adapter")
    return Qwen25VLAdapter(model)
