"""Pydantic models for Trio Console API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


# ── Events ───────────────────────────────────────────────────────────────────


class EventOut(BaseModel):
    """Event response model."""

    id: str
    timestamp: str
    camera_id: str
    camera_name: str = ""
    description: str = ""
    frame_path: str | None = None
    alert_triggered: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class EventList(BaseModel):
    """Paginated event list."""

    events: list[EventOut]
    total: int
    limit: int
    offset: int


class EventSummary(BaseModel):
    """Aggregate event statistics."""

    total_events: int
    total_alerts: int
    by_hour: dict[str, int] = Field(default_factory=dict)
    by_camera: dict[str, int] = Field(default_factory=dict)


# ── Cameras ──────────────────────────────────────────────────────────────────


class CameraIn(BaseModel):
    """Create camera request."""

    name: str
    source_url: str
    watch_condition: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class CameraOut(BaseModel):
    """Camera response model."""

    id: str
    name: str
    source_url: str
    watch_condition: str = ""
    enabled: bool = True
    created_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    has_snapshot: bool = False
    snapshot_url: str = ""


# ── Chat ─────────────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    """Chat message request for console analytics."""

    message: str
    camera_id: str | None = None
    time_range: str | None = None  # e.g. "last_hour", "today"


class ChatResponse(BaseModel):
    """Chat response with referenced events."""

    answer: str
    referenced_events: list[EventOut] = Field(default_factory=list)


# ── Alert Rules ──────────────────────────────────────────────────────────────


class AlertRuleIn(BaseModel):
    """Create alert rule request."""

    name: str
    camera_id: str | None = None
    condition: str  # natural language condition
    cooldown_seconds: int = 60
    enabled: bool = True


class AlertRuleOut(BaseModel):
    """Alert rule response model."""

    id: str
    name: str
    camera_id: str | None = None
    condition: str
    cooldown_seconds: int = 60
    enabled: bool = True
    created_at: str = ""


# ── Reports ──────────────────────────────────────────────────────────────────


class DailyReport(BaseModel):
    """Daily report response."""

    date: str
    total_events: int
    total_alerts: int
    summary: str = ""
    top_cameras: list[dict[str, Any]] = Field(default_factory=list)
    hourly_breakdown: dict[str, int] = Field(default_factory=dict)


class TrendReport(BaseModel):
    """Trend report over multiple days."""

    start_date: str
    end_date: str
    daily_counts: list[dict[str, Any]] = Field(default_factory=list)
    total_events: int = 0
    total_alerts: int = 0
    trend_direction: str = ""  # "increasing", "decreasing", "stable"
