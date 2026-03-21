"""Events API router for Trio Console."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from trio_core.api.console_models import EventList, EventOut, EventSummary

router = APIRouter(tags=["console-events"])


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


@router.get("/api/events/summary", response_model=EventSummary)
async def events_summary(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Aggregate event statistics."""
    store = _get_store(request)
    result = await store.summary(camera_id=camera_id, start=start, end=end)
    return EventSummary(**result)


@router.get("/api/events", response_model=EventList)
async def list_events(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    alert_only: bool = Query(False),
    q: str | None = Query(None, description="Search description text"),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """List events with filters and pagination."""
    store = _get_store(request)
    result = await store.list_events(
        camera_id=camera_id,
        start=start,
        end=end,
        alert_only=alert_only,
        q=q,
        limit=limit,
        offset=offset,
    )
    return EventList(events=result["events"], total=result["total"], limit=limit, offset=offset)


@router.get("/api/events/{event_id}", response_model=EventOut)
async def get_event(request: Request, event_id: str):
    """Get a single event by ID."""
    store = _get_store(request)
    event = await store.get_event(event_id)
    if event is None:
        raise HTTPException(404, f"Event {event_id} not found")
    return EventOut(**event)


@router.get("/api/events/{event_id}/frame")
async def get_event_frame(request: Request, event_id: str):
    """Serve the JPEG frame for an event."""
    store = _get_store(request)
    path = await store.get_frame_path(event_id)
    if path is None:
        raise HTTPException(404, f"No frame for event {event_id}")
    return FileResponse(path, media_type="image/jpeg")
