"""Pydantic request/response models for the TrioCore API."""

from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Video Analyze (custom endpoint) ─────────────────────────────────────────


class VideoAnalyzeRequest(BaseModel):
    """Request for /v1/video/analyze."""

    video: str = Field(..., description="Video file path or URL")
    prompt: str = Field(..., description="Question or instruction about the video")
    max_tokens: int | None = Field(None, description="Override max generation tokens")
    temperature: float | None = Field(None, description="Override sampling temperature")
    stream: bool = Field(False, description="Stream response token by token")


class InferenceMetricsResponse(BaseModel):
    """Metrics from a video inference run."""

    frames_input: int = 0
    frames_after_dedup: int = 0
    frames_after_motion: int = 0
    dedup_removed: int = 0
    motion_skipped: bool = False
    cache_hit: str = "miss"
    preprocess_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_sec: float = 0.0
    peak_memory_gb: float = 0.0


class VideoAnalyzeResponse(BaseModel):
    """Response for /v1/video/analyze."""

    id: str = Field(default_factory=lambda: f"vid-{uuid.uuid4().hex[:8]}")
    text: str
    model: str
    metrics: InferenceMetricsResponse


# ── OpenAI-Compatible Chat Completion ────────────────────────────────────────


class ContentPart(BaseModel):
    """A content part in a message (text, image, or video)."""

    type: str
    text: str | None = None
    video: str | None = None
    video_url: dict[str, str] | None = None
    image_url: dict[str, str] | None = None


class ChatMessage(BaseModel):
    """A chat message with optional multimodal content."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentPart]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="default", description="Model ID (ignored, uses loaded model)")
    messages: list[ChatMessage]
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: dict[str, str]
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[ChatCompletionChoice]
    usage: Usage = Field(default_factory=Usage)


# ── Streaming ────────────────────────────────────────────────────────────────


class DeltaContent(BaseModel):
    role: str | None = None
    content: str | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: str | None = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""

    id: str
    object: str = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str = "default"
    choices: list[StreamChoice]
    usage: Usage | None = None


# ── Health ───────────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    model: str
    loaded: bool
    config: dict[str, Any] = {}


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "trio-core"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ── TrioClaw-compatible (POST /analyze-frame) ────────────────────────────────


class AnalyzeFrameRequest(BaseModel):
    """Request for POST /analyze-frame (TrioClaw compatibility)."""

    frame_b64: str = Field(..., description="Base64-encoded JPEG image")
    question: str = Field(..., description="Question about the image")


# ── Watch API (/v1/watch) ────────────────────────────────────────────────────


class WatchCondition(BaseModel):
    """A condition to monitor for in a watch."""

    id: str = Field(..., description="Condition identifier (e.g. 'person', 'package')")
    question: str = Field(..., description="Question to ask the VLM (e.g. 'Is there a person?')")


class WatchRequest(BaseModel):
    """Request for POST /v1/watch — start watching an RTSP stream."""

    source: str = Field(..., description="RTSP URL or video source")
    conditions: list[WatchCondition] = Field(..., description="Conditions to monitor")
    fps: float = Field(1.0, description="Maximum check rate (frames per second)")
    stream: bool = Field(True, description="Return SSE stream (must be true)")


class WatchConditionResult(BaseModel):
    """Result for a single condition in a watch cycle."""

    id: str
    triggered: bool
    answer: str


class WatchMetrics(BaseModel):
    """Metrics from a single watch inference cycle."""

    latency_ms: int = 0
    tok_s: float = 0.0
    frames_analyzed: int = 0


class WatchInfo(BaseModel):
    """Info about an active watch (returned by GET /v1/watch)."""

    watch_id: str
    source: str
    state: str
    conditions: list[WatchCondition]
    uptime_s: int = 0
    checks: int = 0
    alerts: int = 0


class AnalyzeFrameResponse(BaseModel):
    """Response for POST /analyze-frame (TrioClaw compatibility)."""

    answer: str
    triggered: bool | None = None
    latency_ms: int = 0
