"""Engine configuration with environment variable overrides."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class EngineConfig(BaseSettings):
    """TrioCore configuration. All fields can be set via TRIO_ env vars."""

    model_config = {"env_prefix": "TRIO_"}

    # Model
    model: str = Field(
        default="mlx-community/Qwen2.5-VL-3B-Instruct-4bit",
        description="HuggingFace model ID or local path",
    )

    # Server
    host: str = Field(default="0.0.0.0", description="API bind host")
    port: int = Field(default=8000, description="API bind port")

    # Video pipeline
    video_fps: float = Field(default=2.0, description="Target FPS for frame extraction")
    video_max_frames: int = Field(default=128, description="Max frames per video")
    video_min_frames: int = Field(default=4, description="Min frames per video")

    # Temporal dedup
    dedup_enabled: bool = Field(default=True, description="Enable temporal deduplication")
    dedup_threshold: float = Field(
        default=0.95,
        description="Similarity threshold for dedup (0-1). Higher = more aggressive.",
    )

    # Motion gate
    motion_enabled: bool = Field(default=False, description="Enable motion gating")
    motion_threshold: float = Field(
        default=0.03, description="Motion threshold (fraction of changed pixels)"
    )

    # Generation
    max_tokens: int = Field(default=512, description="Max generation tokens")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    top_p: float = Field(default=1.0, description="Top-p sampling")

    # ToMe (Token Merging)
    tome_enabled: bool = Field(default=False, description="Enable Token Merging in vision encoder")
    tome_r: int = Field(default=4, description="Number of tokens merged per layer")
    tome_metric: str = Field(default="hidden", description="Similarity metric: 'keys' or 'hidden'")
    tome_min_keep_ratio: float = Field(
        default=0.3, description="Min fraction of tokens to keep (0.0-1.0]"
    )
    tome_adaptive: bool = Field(
        default=False, description="Linearly ramp r from 0 in early layers to r_max in deep layers"
    )

    # FastV (visual token pruning in LLM layers)
    fastv_enabled: bool = Field(default=False, description="Enable FastV visual token pruning")
    fastv_ratio: float = Field(
        default=0.5, description="Fraction of visual tokens to prune (0.0-1.0)"
    )
    fastv_layer: int = Field(
        default=2, description="LLM layer to compute attention importance at"
    )

    # Early stopping
    early_stop: bool = Field(default=False, description="Enable EOS-probability early stopping")
    early_stop_threshold: float = Field(
        default=0.8, description="P(EOS) threshold to trigger early stop (0.0-1.0)"
    )

    # Cache (Phase 2)
    cache_enabled: bool = Field(default=False, description="Enable video cache")
    cache_max_entries: int = Field(default=50, description="Max cache entries")
    cache_max_memory_mb: int = Field(default=2048, description="Max cache memory in MB")
