"""Metrics API router for Trio Console."""

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
    if granularity not in ("minute", "hour", "day"):
        raise HTTPException(400, "granularity must be 'minute', 'hour', or 'day'")
    logger.debug(
        "query_metrics camera_id=%s metric_type=%s start=%s end=%s granularity=%s",
        camera_id, metric_type, start, end, granularity,
    )
    data = await store.query_metrics(
        camera_id=camera_id,
        metric_type=metric_type,
        start=start,
        end=end,
        granularity=granularity,
    )
    logger.debug("query_metrics returned %d data points for camera_id=%s", len(data), camera_id)
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
    metrics = await store.latest_metrics(camera_id)
    return {"camera_id": camera_id, "metrics": metrics}


@router.get("/summary")
async def metrics_summary(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Summary statistics for a camera's metrics."""
    store = _get_store(request)
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
    metric_id = await store.insert_metric(body)
    return {"id": metric_id, "status": "recorded"}
