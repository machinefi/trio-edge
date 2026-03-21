"""Analytics API router for Trio Console."""

from __future__ import annotations

import statistics
from datetime import date, datetime

from fastapi import APIRouter, HTTPException, Query, Request

router = APIRouter(prefix="/api/analytics", tags=["console-analytics"])


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


@router.get("/hourly")
async def hourly_breakdown(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    date_str: str = Query(None, alias="date", description="YYYY-MM-DD, default today"),
):
    """Hourly breakdown for dashboard charts."""
    store = _get_store(request)
    if date_str is None:
        date_str = date.today().isoformat()

    day_start = f"{date_str}T00:00:00"
    day_end = f"{date_str}T23:59:59"

    # People in/out by hour from metrics
    people_in_data = await store.query_metrics(
        camera_id=camera_id, metric_type="people_in",
        start=day_start, end=day_end, granularity="hour",
    )
    people_out_data = await store.query_metrics(
        camera_id=camera_id, metric_type="people_out",
        start=day_start, end=day_end, granularity="hour",
    )

    # Events/alerts by hour from event store
    db = store._db
    assert db is not None
    cursor = await db.execute(
        """
        SELECT CAST(SUBSTR(timestamp, 12, 2) AS INTEGER) AS hour,
               COUNT(*) AS events,
               SUM(alert_triggered) AS alerts
        FROM events
        WHERE camera_id = ? AND timestamp >= ? AND timestamp <= ?
        GROUP BY hour
        """,
        (camera_id, day_start, day_end),
    )
    event_rows = {r[0]: (r[1], int(r[2] or 0)) for r in await cursor.fetchall()}

    # Build hour-keyed lookups from metrics
    def _hour_int(ts: str) -> int:
        """Extract hour integer from a truncated timestamp like '2026-03-21T14'."""
        return int(ts[-2:]) if len(ts) >= 2 else 0

    in_by_hour = {_hour_int(d["timestamp"]): d["value"] for d in people_in_data}
    out_by_hour = {_hour_int(d["timestamp"]): d["value"] for d in people_out_data}

    hours = []
    for h in range(24):
        ev, al = event_rows.get(h, (0, 0))
        hours.append({
            "hour": h,
            "people_in": in_by_hour.get(h, 0),
            "people_out": out_by_hour.get(h, 0),
            "events": ev,
            "alerts": al,
        })

    return {"date": date_str, "camera_id": camera_id, "hours": hours}


@router.get("/comparison")
async def camera_comparison(
    request: Request,
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Multi-camera comparison."""
    store = _get_store(request)
    db = store._db
    assert db is not None

    where_clauses: list[str] = []
    params: list[str] = []
    if start:
        where_clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        where_clauses.append("timestamp <= ?")
        params.append(end)

    ts_where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    # Events and alerts per camera
    cursor = await db.execute(
        f"""
        SELECT camera_id, camera_name, COUNT(*) AS events, SUM(alert_triggered) AS alerts
        FROM events {ts_where}
        GROUP BY camera_id
        """,
        params,
    )
    event_rows = {r[0]: {"camera_name": r[1] or r[0], "events": r[2], "alerts": int(r[3] or 0)}
                  for r in await cursor.fetchall()}

    # Metrics per camera
    met_where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    cursor = await db.execute(
        f"""
        SELECT camera_id, metric_type, SUM(value) AS total
        FROM metrics {met_where}
        GROUP BY camera_id, metric_type
        """,
        params,
    )
    met_rows: dict[str, dict[str, float]] = {}
    for r in await cursor.fetchall():
        met_rows.setdefault(r[0], {})[r[1]] = r[2]

    # Get camera names from cameras table
    cam_names = {}
    cam_list = await store.list_cameras()
    for c in cam_list:
        cam_names[c["id"]] = c.get("name") or c["id"]

    # Merge
    all_camera_ids = set(event_rows.keys()) | set(met_rows.keys())
    cameras = []
    for cid in sorted(all_camera_ids):
        ev = event_rows.get(cid, {"camera_name": cid, "events": 0, "alerts": 0})
        met = met_rows.get(cid, {})
        cameras.append({
            "camera_id": cid,
            "camera_name": cam_names.get(cid) or ev["camera_name"],
            "total_in": met.get("people_in", 0),
            "total_out": met.get("people_out", 0),
            "total_events": ev["events"],
            "total_alerts": ev["alerts"],
        })

    return {"cameras": cameras}


@router.get("/anomalies")
async def detect_anomalies(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    start: str | None = Query(None, description="ISO timestamp lower bound"),
    end: str | None = Query(None, description="ISO timestamp upper bound"),
):
    """Detect anomalies by comparing each hour to the average for that hour of week.

    Flags data points that deviate more than 2 standard deviations from the mean.
    """
    store = _get_store(request)
    db = store._db
    assert db is not None

    # Get all hourly metrics for this camera (people_in) to build baselines
    where_clauses = ["camera_id = ?", "metric_type = 'people_in'"]
    params: list[str] = [camera_id]
    if start:
        where_clauses.append("timestamp >= ?")
        params.append(start)
    if end:
        where_clauses.append("timestamp <= ?")
        params.append(end)
    where = "WHERE " + " AND ".join(where_clauses)

    cursor = await db.execute(
        f"""
        SELECT SUBSTR(timestamp, 1, 13) AS hour_bucket,
               SUM(value) AS total
        FROM metrics {where}
        GROUP BY hour_bucket ORDER BY hour_bucket
        """,
        params,
    )
    hourly_data = [(r[0], r[1]) for r in await cursor.fetchall()]

    if not hourly_data:
        return {"anomalies": []}

    # Group by day-of-week + hour
    # hour_bucket looks like "2026-03-21T14"
    dow_hour_values: dict[str, list[float]] = {}
    for bucket, value in hourly_data:
        try:
            dt = datetime.fromisoformat(bucket + ":00:00")
            key = f"{dt.weekday()}_{dt.hour:02d}"
            dow_hour_values.setdefault(key, []).append(value)
        except (ValueError, TypeError):
            continue

    # Compute mean/std per dow+hour
    baselines: dict[str, tuple[float, float]] = {}
    for key, values in dow_hour_values.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        baselines[key] = (mean, std)

    # Find anomalies
    anomalies = []
    for bucket, value in hourly_data:
        try:
            dt = datetime.fromisoformat(bucket + ":00:00")
            key = f"{dt.weekday()}_{dt.hour:02d}"
        except (ValueError, TypeError):
            continue

        mean, std = baselines.get(key, (0, 0))
        if std == 0 or mean == 0:
            continue

        deviation = abs(value - mean)
        if deviation > 2 * std:
            deviation_pct = round((deviation / mean) * 100, 1)
            direction = "above" if value > mean else "below"
            anomalies.append({
                "timestamp": bucket,
                "metric_type": "people_in",
                "expected_value": round(mean, 1),
                "actual_value": value,
                "deviation_pct": deviation_pct,
                "description": f"{deviation_pct}% {direction} average for this time slot",
            })

    return {"anomalies": anomalies}
