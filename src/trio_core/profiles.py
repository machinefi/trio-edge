"""Model profiles — per-model architecture parameters for optimal inference.

Each Qwen VL model has different patch sizes, merge factors, and token budgets.
Using the wrong parameters wastes context window or produces misaligned tensors.

Usage:
    from trio_core.profiles import get_profile

    profile = get_profile("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")
    print(profile.merge_factor)   # 28
    print(profile.max_visual_tokens)  # 24576
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelProfile:
    """Architecture parameters for a specific VLM model."""

    # Model identity
    family: str              # "qwen3.5", "qwen2.5-vl"
    param_size: str          # "0.8B", "3B", "7B"

    # Vision encoder
    spatial_patch: int       # ViT patch size (14 for Qwen2.5-VL, 16 for Qwen3.5)
    temporal_patch: int      # Conv3d temporal kernel (2 for all Qwen VL)
    spatial_merge: int       # PatchMerger groups NxN patches into 1 (2 for all)

    # Derived: merge_factor = spatial_patch × spatial_merge
    # This is the IMAGE_FACTOR for smart_resize
    @property
    def merge_factor(self) -> int:
        return self.spatial_patch * self.spatial_merge

    # Context & token limits
    context_window: int      # Max total tokens (text + vision)
    max_visual_tokens: int   # Hard cap on visual tokens sent to LLM
    default_visual_ratio: float  # Fraction of context to allocate to vision (default)

    # Generation defaults
    recommended_fps: float   # Target FPS for frame extraction
    min_frames: int          # Minimum frames to extract
    max_frames: int          # Maximum frames to extract

    # Architecture flags
    has_deltanet: bool       # True if model uses Gated DeltaNet layers
    deltanet_layers: int     # Number of DeltaNet layers (0 if none)
    full_attn_layers: int    # Number of full attention layers
    kv_heads: int            # Number of KV heads for GQA

    # Memory estimates (4-bit quantization)
    model_size_gb: float     # Approximate model size on disk (4-bit)
    inference_memory_gb: float  # Approximate memory during inference

    def compute_visual_tokens(self, frames: int, height: int, width: int) -> int:
        """Compute the number of visual tokens for given video dimensions.

        Formula: (frames / temporal_patch) × (H / merge_factor) × (W / merge_factor)
        """
        t = frames // self.temporal_patch
        h = height // self.merge_factor
        w = width // self.merge_factor
        return t * h * w

    def compute_optimal_params(
        self,
        duration_sec: float,
        native_height: int,
        native_width: int,
        *,
        token_budget: int | None = None,
    ) -> tuple[int, int, int]:
        """Compute optimal (frames, height, width) to fit within token budget.

        Returns dimensions aligned to merge_factor and temporal_patch.
        """
        budget = token_budget or self.max_visual_tokens
        mf = self.merge_factor
        tp = self.temporal_patch

        # Start with ideal frames from FPS
        ideal_frames = max(self.min_frames, int(duration_sec * self.recommended_fps))
        ideal_frames = min(ideal_frames, self.max_frames)
        # Align to temporal_patch
        ideal_frames = max(tp, (ideal_frames // tp) * tp)

        # Start with native resolution aligned to merge_factor
        h = max(mf, (native_height // mf) * mf)
        w = max(mf, (native_width // mf) * mf)

        # Check if within budget
        tokens = self.compute_visual_tokens(ideal_frames, h, w)

        if tokens <= budget:
            return ideal_frames, h, w

        # Strategy 1: Reduce resolution first (cheaper than losing frames)
        scale = (budget / max(tokens, 1)) ** 0.5
        h_scaled = max(mf, round(h * scale / mf) * mf)
        w_scaled = max(mf, round(w * scale / mf) * mf)
        tokens_scaled = self.compute_visual_tokens(ideal_frames, h_scaled, w_scaled)

        if tokens_scaled <= budget:
            return ideal_frames, h_scaled, w_scaled

        # Strategy 2: Also reduce frames
        tokens_per_temporal_slot = (h_scaled // mf) * (w_scaled // mf)
        max_temporal_slots = max(1, budget // max(tokens_per_temporal_slot, 1))
        frames = max(tp, min(ideal_frames, max_temporal_slots * tp))
        # Re-align
        frames = max(tp, (frames // tp) * tp)

        return frames, h_scaled, w_scaled


# ── Profile Registry ─────────────────────────────────────────────────────────

PROFILES: dict[str, ModelProfile] = {
    "qwen3.5-0.8b": ModelProfile(
        family="qwen3.5",
        param_size="0.8B",
        spatial_patch=16,
        temporal_patch=2,
        spatial_merge=2,
        context_window=262_144,
        max_visual_tokens=8_192,
        default_visual_ratio=0.03,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=128,
        has_deltanet=True,
        deltanet_layers=18,
        full_attn_layers=6,
        kv_heads=2,
        model_size_gb=0.5,
        inference_memory_gb=0.8,
    ),
    "qwen3.5-4b": ModelProfile(
        family="qwen3.5",
        param_size="4B",
        spatial_patch=16,
        temporal_patch=2,
        spatial_merge=2,
        context_window=262_144,
        max_visual_tokens=16_384,
        default_visual_ratio=0.06,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=256,
        has_deltanet=True,
        deltanet_layers=24,
        full_attn_layers=8,
        kv_heads=4,
        model_size_gb=2.5,
        inference_memory_gb=3.5,
    ),
    "qwen3.5-9b": ModelProfile(
        family="qwen3.5",
        param_size="9B",
        spatial_patch=16,
        temporal_patch=2,
        spatial_merge=2,
        context_window=262_144,
        max_visual_tokens=24_576,
        default_visual_ratio=0.09,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=512,
        has_deltanet=True,
        deltanet_layers=24,
        full_attn_layers=8,
        kv_heads=4,
        model_size_gb=5.0,
        inference_memory_gb=7.0,
    ),
    "qwen2.5-vl-3b": ModelProfile(
        family="qwen2.5-vl",
        param_size="3B",
        spatial_patch=14,
        temporal_patch=2,
        spatial_merge=2,
        context_window=128_000,
        max_visual_tokens=24_576,
        default_visual_ratio=0.19,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=768,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=36,
        kv_heads=2,
        model_size_gb=1.8,
        inference_memory_gb=2.5,
    ),
    "qwen2.5-vl-7b": ModelProfile(
        family="qwen2.5-vl",
        param_size="7B",
        spatial_patch=14,
        temporal_patch=2,
        spatial_merge=2,
        context_window=128_000,
        max_visual_tokens=24_576,
        default_visual_ratio=0.19,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=768,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=28,
        kv_heads=4,
        model_size_gb=4.5,
        inference_memory_gb=6.0,
    ),
    "qwen3-vl-4b": ModelProfile(
        family="qwen3-vl",
        param_size="4B",
        spatial_patch=14,
        temporal_patch=2,
        spatial_merge=2,
        context_window=128_000,
        max_visual_tokens=24_576,
        default_visual_ratio=0.19,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=768,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=32,
        kv_heads=4,
        model_size_gb=2.5,
        inference_memory_gb=3.5,
    ),
}

# Aliases for common HuggingFace model IDs
_ALIASES: dict[str, str] = {
    "qwen3.5-0.8b": "qwen3.5-0.8b",
    "qwen2.5-vl-3b": "qwen2.5-vl-3b",
    "qwen2.5-vl-7b": "qwen2.5-vl-7b",
    "qwen3-vl-4b": "qwen3-vl-4b",
}

# Patterns to match HuggingFace model IDs
_PATTERNS: list[tuple[str, str]] = [
    (r"qwen3\.?5.*9b", "qwen3.5-9b"),
    (r"qwen3\.?5.*4b", "qwen3.5-4b"),
    (r"qwen3\.?5.*0\.?8b", "qwen3.5-0.8b"),
    (r"qwen3.*vl.*4b", "qwen3-vl-4b"),
    (r"qwen2\.?5.*vl.*3b", "qwen2.5-vl-3b"),
    (r"qwen2\.?5.*vl.*7b", "qwen2.5-vl-7b"),
    # Broader fallbacks
    (r"qwen3\.?5", "qwen3.5-0.8b"),
    (r"qwen3.*vl", "qwen3-vl-4b"),
    (r"qwen2\.?5.*vl", "qwen2.5-vl-3b"),
]


def get_profile(model_name: str) -> ModelProfile:
    """Get the model profile for a given model name or HuggingFace ID.

    Matches against known profiles using regex patterns.
    Falls back to Qwen2.5-VL-3B profile if no match.

    Examples:
        get_profile("mlx-community/Qwen2.5-VL-3B-Instruct-4bit")  → qwen2.5-vl-3b
        get_profile("Qwen/Qwen3.5-0.8B")                          → qwen3.5-0.8b
        get_profile("qwen2.5-vl-7b")                               → qwen2.5-vl-7b
    """
    name_lower = model_name.lower()

    # Direct lookup
    if name_lower in PROFILES:
        return PROFILES[name_lower]

    # Pattern matching
    for pattern, profile_key in _PATTERNS:
        if re.search(pattern, name_lower):
            logger.debug("Model '%s' matched profile '%s'", model_name, profile_key)
            return PROFILES[profile_key]

    # Fallback
    logger.warning(
        "No profile found for '%s', using qwen2.5-vl-3b defaults. "
        "Add a profile in profiles.py for optimal performance.",
        model_name,
    )
    return PROFILES["qwen2.5-vl-3b"]
