"""Events API router for Trio Console — with tenant isolation."""

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


async def _check_camera_access(request: Request, store, camera_id: str | None) -> str | None:
    """Validate camera_id access for tenant. Returns camera_id to use (or None for all allowed).

    If tenant has no access to the requested camera_id, returns a sentinel
    that will match nothing ('__none__').
    """
    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)
    if allowed_ids is None:
        return camera_id  # no auth = internal, allow all

    if camera_id:
        # Specific camera requested — check access
        if camera_id not in allowed_ids:
            return "__none__"  # will match nothing
        return camera_id

    # No specific camera — return first allowed (for summary) or None
    # We'll filter results post-query
    return None


def _filter_events_by_tenant(events: list[dict], allowed_ids: list[str] | None) -> list[dict]:
    """Filter event list to only include events from allowed cameras."""
    if allowed_ids is None:
        return events
    return [e for e in events if e.get("camera_id") in allowed_ids]


@router.get("/api/events/summary", response_model=EventSummary)
async def events_summary(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Aggregate event statistics with AI-generated summary."""
    store = _get_store(request)

    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)

    if allowed_ids is not None and camera_id and camera_id not in allowed_ids:
        return EventSummary(total_events=0, total_alerts=0, ai_summary="")

    result = await store.summary(camera_id=camera_id, start=start, end=end)

    # If tenant-filtered and no specific camera, recount from filtered events
    if allowed_ids is not None and not camera_id:
        events_data = await store.list_events(start=start, end=end, limit=1000)
        filtered = _filter_events_by_tenant(events_data.get("events", []), allowed_ids)
        result["total_events"] = len(filtered)
        result["total_alerts"] = sum(1 for e in filtered if e.get("alert_triggered"))

    # Generate AI summary from top insights
    ai_summary = ""
    try:
        from trio_core.insights import InsightExtractor
        events_data = await store.list_events(
            camera_id=camera_id, start=start, end=end, limit=500,
        )
        events = _filter_events_by_tenant(events_data.get("events", []), allowed_ids)
        if events:
            extractor = InsightExtractor()
            insights = extractor.extract(events)
            if insights:
                top = insights[:3]
                ai_summary = " | ".join(ins.text.split("—")[0].strip() for ins in top)
    except Exception:
        pass

    return EventSummary(**result, ai_summary=ai_summary)


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
    """List events with filters and pagination. Tenant-isolated."""
    store = _get_store(request)

    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)

    if allowed_ids is not None and camera_id and camera_id not in allowed_ids:
        return EventList(events=[], total=0, limit=limit, offset=offset)

    # Fetch more events when filtering post-query to ensure enough results
    fetch_limit = limit + offset + 200 if allowed_ids else limit
    result = await store.list_events(
        camera_id=camera_id,
        start=start,
        end=end,
        alert_only=alert_only,
        q=q,
        limit=fetch_limit,
        offset=0 if allowed_ids else offset,
    )

    events = result["events"]
    if allowed_ids is not None:
        events = _filter_events_by_tenant(events, allowed_ids)
        total = len(events)
        events = events[offset:offset + limit]
    else:
        total = result["total"]

    return EventList(events=events, total=total, limit=limit, offset=offset)


@router.get("/api/events/{event_id}", response_model=EventOut)
async def get_event(request: Request, event_id: str):
    """Get a single event by ID."""
    store = _get_store(request)
    event = await store.get_event(event_id)
    if event is None:
        raise HTTPException(404, f"Event {event_id} not found")

    # Tenant check
    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)
    if allowed_ids is not None and event.get("camera_id") not in allowed_ids:
        raise HTTPException(404, f"Event {event_id} not found")

    return EventOut(**event)


@router.get("/api/events/{event_id}/frame")
async def get_event_frame(request: Request, event_id: str):
    """Serve the JPEG frame for an event."""
    store = _get_store(request)

    # Tenant check
    event = await store.get_event(event_id)
    if event:
        from trio_core.api.routers.auth import get_tenant_camera_ids
        allowed_ids = await get_tenant_camera_ids(request, store)
        if allowed_ids is not None and event.get("camera_id") not in allowed_ids:
            raise HTTPException(404, f"No frame for event {event_id}")

    path = await store.get_frame_path(event_id)
    if path is None:
        raise HTTPException(404, f"No frame for event {event_id}")
    return FileResponse(path, media_type="image/jpeg")
