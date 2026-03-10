"""Core inference engine — orchestrates video pipeline, VLM, and caching."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncGenerator

import numpy as np

from trio_core.backends import BaseBackend, resolve_backend
from trio_core.callbacks import CallbackMixin
from trio_core.config import EngineConfig
from trio_core.profiles import ModelProfile, get_profile
from trio_core.video import (
    MotionGate,
    TemporalDeduplicator,
    load_video,
)

logger = logging.getLogger(__name__)


# ── Profiling ────────────────────────────────────────────────────────────────


class PhaseTimer:
    """Simple context manager for timing a pipeline phase."""

    def __init__(self) -> None:
        self.dt: float = 0.0  # elapsed milliseconds

    def __enter__(self) -> PhaseTimer:
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc) -> None:
        self.dt = (time.monotonic() - self._t0) * 1000


# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class InferenceMetrics:
    """Metrics from a single inference run."""

    # Frame pipeline
    frames_input: int = 0
    frames_after_dedup: int = 0
    frames_after_motion: int = 0
    dedup_removed: int = 0
    motion_skipped: bool = False
    cache_hit: str = "miss"

    # Three-phase timing (ms)
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    latency_ms: float = 0.0

    # Generation stats
    prompt_tokens: int = 0
    completion_tokens: int = 0
    prompt_tps: float = 0.0        # prefill tokens/sec
    tokens_per_sec: float = 0.0    # decode tokens/sec
    prefill_ms: float = 0.0        # prefill latency (derived)
    decode_ms: float = 0.0         # decode latency (derived)
    peak_memory_gb: float = 0.0


@dataclass
class VideoResult:
    """Result of video analysis."""

    text: str
    metrics: InferenceMetrics = field(default_factory=InferenceMetrics)


# ── Engine ───────────────────────────────────────────────────────────────────


class TrioCore(CallbackMixin):
    """Video inference engine — portable across Apple Silicon, NVIDIA, and CPU.

    Auto-detects hardware and selects the best backend:
        Apple Silicon (M1-M4) → MLX (mlx-vlm)
        NVIDIA GPU (CUDA)     → Transformers (PyTorch)
        CPU                   → Transformers (PyTorch)

    Orchestrates: video loading → dedup → motion gate → backend inference.

    Callback events fired during analyze_video():
        on_engine_load      — after model loaded
        on_frame_captured   — after raw frames extracted
        on_dedup_done       — after temporal dedup
        on_motion_check     — after motion gate evaluated
        on_vlm_start        — about to call VLM
        on_vlm_end          — VLM response received
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        callbacks: dict | None = None,
        backend: str | None = None,
    ):
        self.config = config or EngineConfig()
        self._init_callbacks(callbacks)

        self._backend: BaseBackend | None = None
        self._backend_override = backend  # "mlx", "transformers", or None for auto
        self._loaded = False
        self._lock = threading.Lock()
        self._deduplicator = TemporalDeduplicator(threshold=self.config.dedup_threshold)
        self._motion_gate = MotionGate(threshold=self.config.motion_threshold) if self.config.motion_enabled else None

        # Model profile — architecture-aware parameters
        self._profile: ModelProfile = get_profile(self.config.model)

        # Transient state for callbacks to read
        self.last_result: VideoResult | None = None
        self.last_frames: np.ndarray | None = None

    def _apply_auto_optimize(self) -> None:
        """Apply benchmark-proven optimizations from model profile.

        Only applies if auto_optimize=True and user hasn't explicitly
        enabled any optimization (compress, tome).
        """
        if not self.config.auto_optimize:
            return

        profile = self._profile
        if not profile.recommended_optims:
            return

        # Don't override if user has explicitly configured optimizations
        user_has_optims = (
            self.config.compress_enabled
            or self.config.tome_enabled
        )
        if user_has_optims:
            return

        # Apply recommended settings
        for key, value in profile.recommended_optims.items():
            setattr(self.config, key, value)

        logger.info(
            "Auto-optimize: applied %s for %s %s",
            profile.recommended_optims, profile.family, profile.param_size,
        )

    def load(self) -> None:
        """Load model — auto-detects hardware and selects best backend."""
        if self._loaded:
            logger.info("Model already loaded: %s", self.config.model)
            return

        # Apply benchmark-proven optimizations from model profile
        self._apply_auto_optimize()

        backend = resolve_backend(self.config, backend_override=self._backend_override)
        self._backend = backend
        self._backend.load()

        # Configure early stopping if enabled
        if self.config.early_stop and hasattr(self._backend, 'set_early_stop'):
            self._backend.set_early_stop(True, self.config.early_stop_threshold)

        # Configure visual similarity KV reuse if enabled
        if self.config.visual_similarity_threshold > 0 and hasattr(self._backend, 'set_visual_similarity'):
            self._backend.set_visual_similarity(self.config.visual_similarity_threshold)

        # Configure StreamMem bounded KV cache if enabled
        if self.config.streaming_memory_enabled and hasattr(self._backend, 'set_streaming_memory'):
            self._backend.set_streaming_memory(
                True,
                self.config.streaming_memory_budget,
                self.config.streaming_memory_prototype_ratio,
                self.config.streaming_memory_sink_tokens,
            )

        self._loaded = True
        self._profile = get_profile(self.config.model)
        logger.info(
            "Engine ready: backend=%s, device=%s, model=%s, "
            "merge_factor=%d, max_visual_tokens=%d",
            self._backend.backend_name,
            self._backend.device_info.device_name,
            self.config.model,
            self._profile.merge_factor,
            self._profile.max_visual_tokens,
        )
        self.run_callbacks("on_engine_load")

    def reset_context(self) -> None:
        """Reset accumulated context (KV cache, StreamMem state).

        Call between independent test cases or conversation turns to
        prevent context from one session leaking into the next.
        The model stays loaded — only inference state is cleared.
        """
        if self._backend is not None:
            self._backend._prompt_cache = None

    def analyze_video(
        self,
        video: str | Path | np.ndarray,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VideoResult:
        """Analyze a video with the loaded VLM.

        Three-phase profiled pipeline:
            1. Preprocess: load → dedup → motion gate
            2. Inference: backend.generate()
            3. Postprocess: extract metrics
        """
        self._ensure_loaded()
        t0 = time.monotonic()
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        p_pre = PhaseTimer()
        p_inf = PhaseTimer()
        p_post = PhaseTimer()

        # ── Phase 1: Preprocess ──────────────────────────────────────────
        with p_pre:
            p = self._profile
            frames = load_video(
                video,
                fps=self.config.video_fps,
                max_frames=self.config.video_max_frames,
                image_factor=p.merge_factor,
                frame_factor=p.temporal_patch,
                min_frames=p.min_frames,
                max_pixels=p.max_visual_tokens * p.merge_factor * p.merge_factor,
            )
            metrics = InferenceMetrics(frames_input=frames.shape[0])
            self.last_frames = frames
            self.run_callbacks("on_frame_captured")

            if self.config.dedup_enabled:
                result = self._deduplicator.deduplicate(frames)
                frames = result.frames
                metrics.frames_after_dedup = frames.shape[0]
                metrics.dedup_removed = result.removed_count
            else:
                metrics.frames_after_dedup = frames.shape[0]
            self.run_callbacks("on_dedup_done")

            if self._motion_gate is not None:
                has_motion = any(self._motion_gate.has_motion(frames[i]) for i in range(frames.shape[0]))
                self.run_callbacks("on_motion_check")
                if not has_motion:
                    metrics.motion_skipped = True
                    metrics.preprocess_ms = p_pre.dt
                    metrics.latency_ms = (time.monotonic() - t0) * 1000
                    vr = VideoResult(text="[NO MOTION] Scene is static, skipping analysis.", metrics=metrics)
                    self.last_result = vr
                    return vr
            metrics.frames_after_motion = frames.shape[0]

        # ── Phase 2: Inference ───────────────────────────────────────────
        with self._lock:
            with p_inf:
                self.run_callbacks("on_vlm_start")
                gen_result = self._backend.generate(
                    frames, prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=self.config.top_p,
                )

        # ── Phase 3: Postprocess ─────────────────────────────────────────
        with p_post:
            metrics.prompt_tokens = gen_result.prompt_tokens
            metrics.completion_tokens = gen_result.completion_tokens
            metrics.prompt_tps = gen_result.prompt_tps
            metrics.tokens_per_sec = gen_result.generation_tps
            metrics.peak_memory_gb = gen_result.peak_memory
            # Derive prefill/decode timing from TPS
            if gen_result.prompt_tps > 0:
                metrics.prefill_ms = (gen_result.prompt_tokens / gen_result.prompt_tps) * 1000
            if gen_result.generation_tps > 0 and gen_result.completion_tokens > 0:
                metrics.decode_ms = (gen_result.completion_tokens / gen_result.generation_tps) * 1000

        metrics.preprocess_ms = p_pre.dt
        metrics.inference_ms = p_inf.dt
        metrics.postprocess_ms = p_post.dt
        metrics.latency_ms = (time.monotonic() - t0) * 1000

        vr = VideoResult(text=gen_result.text, metrics=metrics)
        self.last_result = vr
        self.run_callbacks("on_vlm_end")
        return vr

    async def stream_analyze(
        self,
        video: str | Path | np.ndarray,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Stream video analysis token by token.

        Runs the synchronous stream_generate in a thread so the async
        event loop is not blocked during GPU inference.
        """
        import asyncio
        import queue

        self._ensure_loaded()
        t0 = time.monotonic()
        max_tokens = max_tokens or self.config.max_tokens
        temperature = temperature if temperature is not None else self.config.temperature

        p_pre = PhaseTimer()
        with p_pre:
            p = self._profile
            frames = load_video(
                video, fps=self.config.video_fps, max_frames=self.config.video_max_frames,
                image_factor=p.merge_factor, frame_factor=p.temporal_patch,
                min_frames=p.min_frames,
                max_pixels=p.max_visual_tokens * p.merge_factor * p.merge_factor,
            )
            metrics = InferenceMetrics(frames_input=frames.shape[0])

            if self.config.dedup_enabled:
                result = self._deduplicator.deduplicate(frames)
                frames = result.frames
                metrics.frames_after_dedup = frames.shape[0]
                metrics.dedup_removed = result.removed_count
            else:
                metrics.frames_after_dedup = frames.shape[0]
            metrics.frames_after_motion = frames.shape[0]
        metrics.preprocess_ms = p_pre.dt

        # Bridge sync generator → async via thread + queue
        chunk_queue: queue.Queue = queue.Queue()
        _sentinel = object()

        def _run_sync():
            try:
                with self._lock:
                    for chunk in self._backend.stream_generate(
                        frames, prompt,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=self.config.top_p,
                    ):
                        chunk_queue.put(chunk)
            except Exception as e:
                chunk_queue.put(e)
            finally:
                chunk_queue.put(_sentinel)

        loop = asyncio.get_running_loop()
        p_inf = PhaseTimer()
        with p_inf:
            task = loop.run_in_executor(None, _run_sync)
            while True:
                item = await loop.run_in_executor(None, chunk_queue.get)
                if item is _sentinel:
                    break
                if isinstance(item, Exception):
                    raise item
                yield {"text": item.text, "finished": False, "metrics": None}
            await task  # propagate any unhandled exception
        metrics.inference_ms = p_inf.dt

        metrics.latency_ms = (time.monotonic() - t0) * 1000
        yield {"text": "", "finished": True, "metrics": metrics}

    def analyze_frame(
        self,
        frame: np.ndarray,
        prompt: str,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> VideoResult:
        """Analyze a single frame (image). Convenience wrapper."""
        if frame.ndim == 3:
            if frame.shape[2] in (1, 3, 4):
                frame = frame.transpose(2, 0, 1)
            frame = frame[np.newaxis]
        return self.analyze_video(frame, prompt, max_tokens=max_tokens, temperature=temperature)

    def health(self) -> dict[str, Any]:
        """Return engine health status."""
        p = self._profile
        base = {
            "status": "ok" if self._loaded else "not_loaded",
            "model": self.config.model,
            "loaded": self._loaded,
            "profile": {
                "family": p.family,
                "param_size": p.param_size,
                "merge_factor": p.merge_factor,
                "max_visual_tokens": p.max_visual_tokens,
                "context_window": p.context_window,
                "has_deltanet": p.has_deltanet,
            },
            "config": {
                "video_fps": self.config.video_fps,
                "video_max_frames": self.config.video_max_frames,
                "dedup_enabled": self.config.dedup_enabled,
                "dedup_threshold": self.config.dedup_threshold,
                "motion_enabled": self.config.motion_enabled,
                "max_tokens": self.config.max_tokens,
            },
        }
        if self._backend:
            base["backend"] = self._backend.health()
        return base

    # ── Private ──────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call engine.load() first.")
