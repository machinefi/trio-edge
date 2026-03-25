"""Metrics API router for Trio Console — with tenant isolation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger("trio.console.metrics")

router = APIRouter(prefix="/api/metrics", tags=["console-metrics"])


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


async def _check_camera_access(request: Request, store, camera_id: str):
    """Verify the tenant has access to the requested camera_id."""
    from trio_core.api.routers.auth import get_tenant_camera_ids
    allowed_ids = await get_tenant_camera_ids(request, store)
    if allowed_ids is not None and camera_id not in allowed_ids:
        raise HTTPException(403, f"Access denied to camera {camera_id}")


@router.get("/")
async def query_metrics(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    metric_type: str = Query("people_in"),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
    granularity: str = Query("hour", description="minute, hour, or day"),
):
    """Query metrics time series with time-bucketed aggregation."""
    store = _get_store(request)
    await _check_camera_access(request, store, camera_id)

    if granularity not in ("minute", "hour", "day"):
        raise HTTPException(400, "granularity must be 'minute', 'hour', or 'day'")
    data = await store.query_metrics(
        camera_id=camera_id,
        metric_type=metric_type,
        start=start,
        end=end,
        granularity=granularity,
    )
    return {
        "data": data,
        "camera_id": camera_id,
        "metric_type": metric_type,
        "granularity": granularity,
    }


@router.get("/latest")
async def latest_metrics(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
):
    """Latest metric values for a camera."""
    store = _get_store(request)
    await _check_camera_access(request, store, camera_id)
    metrics = await store.latest_metrics(camera_id)
    return {"camera_id": camera_id, "metrics": metrics}


@router.get("/summary")
async def metrics_summary(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Summary statistics for a camera's metrics. Defaults to last 24 hours."""
    store = _get_store(request)
    await _check_camera_access(request, store, camera_id)

    if not start:
        from datetime import datetime, timedelta, timezone
        start = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    result = await store.metrics_summary(camera_id=camera_id, start=start, end=end)
    result["period"] = {"start": start, "end": end}
    return result


@router.post("/")
async def record_metric(request: Request):
    """Record a metric data point from the counting pipeline."""
    store = _get_store(request)
    body = await request.json()
    if not body.get("camera_id") or not body.get("metric_type"):
        raise HTTPException(400, "camera_id and metric_type are required")
    if "value" not in body:
        raise HTTPException(400, "value is required")
    # Tenant isolation on writes
    await _check_camera_access(request, store, body["camera_id"])
    metric_id = await store.insert_metric(body)
    return {"id": metric_id, "status": "recorded"}
