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
    adapter_path: str | None = Field(
        default=None,
        description="Path to LoRA adapter directory (contains adapters.safetensors + adapter_config.json)",
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
    tome_content_aware: bool = Field(
        default=False,
        description="Dynamically scale r based on image complexity. "
        "Simple scenes get high compression, complex scenes (dense text) get low compression.",
    )

    # FastV (visual token pruning in LLM layers)
    fastv_enabled: bool = Field(default=False, description="Enable FastV visual token pruning")
    fastv_ratio: float = Field(
        default=0.5, description="Fraction of visual tokens to prune (0.0-1.0)"
    )
    fastv_layer: int = Field(
        default=2, description="LLM layer to compute attention importance at"
    )

    # Frame-to-frame KV reuse (visual similarity gating)
    visual_similarity_threshold: float = Field(
        default=0.0,
        description="Visual embedding similarity threshold for KV reuse (0=disabled, 0.95 typical). "
        "When enabled, consecutive frames with similar visual content skip LLM prefill "
        "and reuse the KV cache from the previous frame.",
    )

    # Early stopping
    early_stop: bool = Field(default=False, description="Enable EOS-probability early stopping")
    early_stop_threshold: float = Field(
        default=0.8, description="P(EOS) threshold to trigger early stop (0.0-1.0)"
    )

    # StreamMem (streaming memory for continuous video)
    streaming_memory_enabled: bool = Field(
        default=False, description="Enable bounded KV cache accumulation for continuous streams"
    )
    streaming_memory_budget: int = Field(
        default=6000, description="Max visual tokens in KV cache before eviction"
    )
    streaming_memory_prototype_ratio: float = Field(
        default=0.1, description="Fraction of evicted tokens merged into prototypes (0.0-1.0)"
    )
    streaming_memory_sink_tokens: int = Field(
        default=4, description="Number of attention sink tokens to always preserve (StreamingLLM)"
    )

    # Visual token compression (post-encoder)
    compress_enabled: bool = Field(
        default=False, description="Enable visual token compression after vision encoder"
    )
    compress_ratio: float = Field(
        default=0.5,
        description="Fraction of visual tokens to keep (0.3-0.7). Lower = more compression.",
    )

    # Auto-optimize (apply benchmark-proven settings from model profile)
    auto_optimize: bool = Field(
        default=True,
        description="Auto-apply recommended optimizations from model profile. "
        "Set to False to use explicit config only.",
    )

    # Cache (Phase 2)
    cache_enabled: bool = Field(default=False, description="Enable video cache")
    cache_max_entries: int = Field(default=50, description="Max cache entries")
    cache_max_memory_mb: int = Field(default=2048, description="Max cache memory in MB")
