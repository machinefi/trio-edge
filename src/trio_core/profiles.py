"""Model profiles — per-model architecture parameters for optimal inference.

Each VLM has different patch sizes, merge factors, and token budgets.
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
from dataclasses import dataclass, field

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

    # Optimization support
    supports_tome: bool = True  # Whether in-ViT ToMe is applicable
    supports_fastv: bool = True  # Whether FastV attention pruning is applicable

    # Benchmark-proven recommended optimizations (auto-applied when auto_optimize=True).
    # Keys are EngineConfig field names, values are benchmark-optimal settings.
    # Example: {"tome_enabled": True, "tome_r": 4} for models where ToMe is best.
    recommended_optims: dict = field(default_factory=dict)

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
        supports_fastv=False,   # DeltaNet layers incompatible with FastV
        # Too small to compress: only 1.02x speedup, TextVQA -18%, GQA -6%
        recommended_optims={},
    ),
    "qwen3.5-2b": ModelProfile(
        family="qwen3.5",
        param_size="2B",
        spatial_patch=16,
        temporal_patch=2,
        spatial_merge=2,
        context_window=262_144,
        max_visual_tokens=12_288,
        default_visual_ratio=0.05,
        recommended_fps=2.0,
        min_frames=4,
        max_frames=192,
        has_deltanet=True,
        deltanet_layers=20,
        full_attn_layers=8,
        kv_heads=2,
        model_size_gb=1.5,
        inference_memory_gb=2.0,
        supports_fastv=False,   # DeltaNet layers incompatible with FastV
        # POPE: 94% baseline → 93% compressed_50 (-1%, 1.14x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.5},
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
        supports_fastv=False,   # DeltaNet layers incompatible with FastV
        # POPE: 90%→89% (-1). TextVQA: 52%→64% (+12!). ToMe cleans noisy ViT tokens.
        recommended_optims={"tome_enabled": True, "tome_r": 4, "tome_metric": "hidden"},
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
        supports_fastv=False,   # DeltaNet layers incompatible with FastV
        # ToMe gives +6% TextVQA/GQA but 0.95x E2E speed (ViT overhead).
        # Compressed 50% gives 1.21x speedup, +6% GQA, 0% TextVQA — better E2E.
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.5},
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
        supports_fastv=False,   # Over-prunes visual tokens, POPE 41%, TextVQA 14%
        # POPE: 90% baseline → 92% compressed_40 (+2%!, 1.36x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.4},
    ),
    "qwen3-vl-2b": ModelProfile(
        family="qwen3-vl",
        param_size="2B",
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
        kv_heads=2,
        model_size_gb=1.5,
        inference_memory_gb=2.0,
        supports_tome=False,    # deepstack re-adds visual embeds, breaks after ToMe merge
        supports_fastv=False,   # Produces garbage output (0% POPE, 2% MMBench, 0% GQA)
        # POPE: 92% baseline → 92% compressed_50 (0% drop, 1.23x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.5},
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
        supports_tome=False,    # deepstack re-adds visual embeds, breaks after ToMe merge
        # POPE: 91% baseline → 88% compressed_50 (-3%, 1.24x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.5},
    ),
    "qwen3-vl-8b": ModelProfile(
        family="qwen3-vl",
        param_size="8B",
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
        kv_heads=4,
        model_size_gb=5.0,
        inference_memory_gb=7.0,
        supports_tome=False,    # deepstack re-adds visual embeds, breaks after ToMe merge
        # POPE: 91% baseline → 93% compressed_50 (+2%!, 1.26x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.5},
    ),
    # ── Gemma 3n (edge-first, MatFormer nested architecture) ────────────
    "gemma3n-e2b": ModelProfile(
        family="gemma3n",
        param_size="E2B",
        spatial_patch=14,       # MobileNet-v5 vision encoder (not standard ViT)
        temporal_patch=1,       # Image model, no temporal patching
        spatial_merge=1,        # MLP projector, no PatchMerger
        context_window=32_000,
        max_visual_tokens=256,  # Fixed 256 tokens per image (256x256/512x512/768x768)
        default_visual_ratio=0.008,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=26,    # MatFormer nested layers
        kv_heads=4,
        model_size_gb=1.5,      # 5B params but 2GB memory footprint
        inference_memory_gb=2.0,
        supports_tome=False,    # MobileNet-v5, not standard ViT
    ),
    "gemma3n-e4b": ModelProfile(
        family="gemma3n",
        param_size="E4B",
        spatial_patch=14,
        temporal_patch=1,
        spatial_merge=1,
        context_window=32_000,
        max_visual_tokens=256,
        default_visual_ratio=0.008,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=34,
        kv_heads=8,
        model_size_gb=2.5,      # 8B params but 4GB memory footprint
        inference_memory_gb=3.0,
        supports_tome=False,    # MobileNet-v5, not standard ViT
    ),
    # ── Phi-4 Multimodal ─────────────────────────────────────────────
    "phi4-multimodal": ModelProfile(
        family="phi4",
        param_size="3.8B",
        spatial_patch=14,       # SigLIP ViT
        temporal_patch=1,
        spatial_merge=1,
        context_window=131_072,
        max_visual_tokens=1024, # Dynamic resolution, up to 1024 tokens
        default_visual_ratio=0.008,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=32,
        kv_heads=8,
        model_size_gb=2.0,
        inference_memory_gb=3.0,
    ),
    # ── Gemma 3 ─────────────────────────────────────────────────────────
    "gemma3-4b": ModelProfile(
        family="gemma3",
        param_size="4B",
        spatial_patch=14,       # SigLIP ViT-SO400M: patch=14
        temporal_patch=1,       # No temporal patching (image model)
        spatial_merge=1,        # No PatchMerger, uses MLP projector
        context_window=128_000,
        max_visual_tokens=256,  # Fixed 256 tokens per 896x896 crop
        default_visual_ratio=0.002,
        recommended_fps=1.0,    # Image-oriented, low FPS for video
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=26,
        kv_heads=4,
        model_size_gb=2.5,
        inference_memory_gb=3.5,
    ),
    "gemma3-12b": ModelProfile(
        family="gemma3",
        param_size="12B",
        spatial_patch=14,
        temporal_patch=1,
        spatial_merge=1,
        context_window=128_000,
        max_visual_tokens=256,
        default_visual_ratio=0.002,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=48,
        kv_heads=8,
        model_size_gb=7.0,
        inference_memory_gb=10.0,
    ),
    # ── SmolVLM ─────────────────────────────────────────────────────────
    "smolvlm-256m": ModelProfile(
        family="smolvlm",
        param_size="256M",
        spatial_patch=16,       # SigLIP-B/16
        temporal_patch=1,
        spatial_merge=1,        # Pixel shuffle + MLP projection
        context_window=8_192,
        max_visual_tokens=64,   # 64 tokens per 512x512 crop
        default_visual_ratio=0.008,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,
        kv_heads=3,
        model_size_gb=0.3,
        inference_memory_gb=0.5,
        supports_tome=False,    # Pixel shuffle disrupts spatial structure
    ),
    "smolvlm-500m": ModelProfile(
        family="smolvlm",
        param_size="500M",
        spatial_patch=16,
        temporal_patch=1,
        spatial_merge=1,
        context_window=8_192,
        max_visual_tokens=64,
        default_visual_ratio=0.008,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,
        kv_heads=3,
        model_size_gb=0.5,
        inference_memory_gb=0.8,
        supports_tome=False,    # Pixel shuffle disrupts spatial structure
    ),
    "smolvlm-2.2b": ModelProfile(
        family="smolvlm",
        param_size="2.2B",
        spatial_patch=14,       # SigLIP-SO400M for 2.2B
        temporal_patch=1,
        spatial_merge=1,
        context_window=16_384,
        max_visual_tokens=256,
        default_visual_ratio=0.016,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=32,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,
        kv_heads=4,
        model_size_gb=1.5,
        inference_memory_gb=2.0,
        supports_tome=False,    # Pixel shuffle disrupts spatial structure
    ),
    # ── InternVL3 (LLaVA-style: InternViT-300M + MLP + Qwen2.5 LLM) ──
    "internvl3-1b": ModelProfile(
        family="internvl",
        param_size="1B",
        spatial_patch=14,       # InternViT-300M, patch_size=14
        temporal_patch=1,       # Image model
        spatial_merge=1,        # Pixel unshuffle + MLP (not PatchMerger)
        context_window=32_000,
        max_visual_tokens=1024, # Dynamic tiling, 256 tokens per 448x448 tile
        default_visual_ratio=0.032,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,    # Qwen2.5-0.5B LLM
        kv_heads=2,
        supports_tome=False,    # pixel_shuffle disrupts spatial structure
        supports_fastv=False,   # Different architecture, not validated
        model_size_gb=0.6,
        inference_memory_gb=1.0,
        # Only 1.06x speedup, +21% memory overhead — not worth it for 1B model
        recommended_optims={},
    ),
    "internvl3-2b": ModelProfile(
        family="internvl",
        param_size="2B",
        spatial_patch=14,
        temporal_patch=1,
        spatial_merge=1,
        context_window=32_000,
        max_visual_tokens=1024,
        default_visual_ratio=0.032,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=28,    # Qwen2.5-1.5B LLM
        kv_heads=2,
        supports_tome=False,    # pixel_shuffle disrupts spatial structure
        supports_fastv=False,   # Different architecture, not validated
        model_size_gb=1.0,
        inference_memory_gb=1.6,
        # POPE: 95% baseline → 96% compressed_40 (+1%!, 1.31x speedup)
        recommended_optims={"compress_enabled": True, "compress_ratio": 0.4},
    ),
    # ── FastVLM (Apple: FastViTHD + MLP + Qwen2 LLM) ──────────────────
    "fastvlm-0.5b": ModelProfile(
        family="fastvlm",
        param_size="0.5B",
        spatial_patch=16,       # FastViTHD (hybrid CNN+ViT)
        temporal_patch=1,
        spatial_merge=1,        # MLP projector
        context_window=32_000,
        max_visual_tokens=576,  # Fewer tokens than standard ViT
        default_visual_ratio=0.018,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,    # Qwen2-0.5B LLM
        kv_heads=2,
        supports_tome=False,    # CNN encoder, not ViT
        model_size_gb=0.1,
        inference_memory_gb=0.5,
    ),
    "fastvlm-1.5b": ModelProfile(
        family="fastvlm",
        param_size="1.5B",
        spatial_patch=16,
        temporal_patch=1,
        spatial_merge=1,
        context_window=32_000,
        max_visual_tokens=576,
        default_visual_ratio=0.018,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=28,    # Qwen2-1.5B LLM
        kv_heads=2,
        supports_tome=False,    # CNN encoder, not ViT
        model_size_gb=0.3,
        inference_memory_gb=1.0,
    ),
    # ── nanoLLaVA (SigLIP-384 + MLP + Qwen1.5 LLM) ───────────────────
    "nanollava-1.5": ModelProfile(
        family="nanollava",
        param_size="1B",
        spatial_patch=14,       # SigLIP so400m/14-384
        temporal_patch=1,
        spatial_merge=1,        # MLP projector
        context_window=32_000,
        max_visual_tokens=729,  # 27x27 = 729 SigLIP tokens (384/14 = 27)
        default_visual_ratio=0.023,
        recommended_fps=1.0,
        min_frames=1,
        max_frames=16,
        has_deltanet=False,
        deltanet_layers=0,
        full_attn_layers=24,    # Qwen1.5-0.5B LLM
        kv_heads=2,
        model_size_gb=0.6,
        inference_memory_gb=1.0,
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
    (r"qwen3\.?5.*2b", "qwen3.5-2b"),
    (r"qwen3\.?5.*0\.?8b", "qwen3.5-0.8b"),
    (r"qwen3.*vl.*8b", "qwen3-vl-8b"),
    (r"qwen3.*vl.*4b", "qwen3-vl-4b"),
    (r"qwen3.*vl.*2b", "qwen3-vl-2b"),
    (r"qwen2\.?5.*vl.*3b", "qwen2.5-vl-3b"),
    (r"qwen2\.?5.*vl.*7b", "qwen2.5-vl-7b"),
    # Gemma 3n (edge)
    (r"gemma.*3n.*e4b", "gemma3n-e4b"),
    (r"gemma.*3n.*e2b", "gemma3n-e2b"),
    (r"gemma.*3n", "gemma3n-e2b"),
    # Gemma 3
    (r"gemma.*3.*12b", "gemma3-12b"),
    (r"gemma.*3.*4b", "gemma3-4b"),
    # Phi-4
    (r"phi.*4.*multimodal", "phi4-multimodal"),
    (r"phi.*4.*vision", "phi4-multimodal"),
    (r"phi3_v", "phi4-multimodal"),
    # SmolVLM
    (r"smolvlm.*2\.?2b", "smolvlm-2.2b"),
    (r"smolvlm.*500m", "smolvlm-500m"),
    (r"smolvlm.*256m", "smolvlm-256m"),
    # InternVL3
    (r"internvl.*3.*2b", "internvl3-2b"),
    (r"internvl.*3.*1b", "internvl3-1b"),
    (r"internvl", "internvl3-2b"),
    # FastVLM (Apple)
    (r"fastvlm.*1\.?5b", "fastvlm-1.5b"),
    (r"fastvlm.*0\.?5b", "fastvlm-0.5b"),
    (r"fastvlm", "fastvlm-0.5b"),
    # nanoLLaVA
    (r"nanollava", "nanollava-1.5"),
    (r"nano.*llava", "nanollava-1.5"),
    # Broader fallbacks
    (r"qwen3\.?5", "qwen3.5-0.8b"),
    (r"qwen3.*vl", "qwen3-vl-4b"),
    (r"qwen2\.?5.*vl", "qwen2.5-vl-3b"),
    (r"gemma.*3", "gemma3-4b"),
    (r"smolvlm", "smolvlm-2.2b"),
    (r"phi", "phi4-multimodal"),
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
