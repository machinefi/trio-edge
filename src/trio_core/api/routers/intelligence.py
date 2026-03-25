"""Intelligence API — serves KG memory data (semantic, episodic, patterns).

Reads from trio-cortex/memory_store (or trio-kg/memory_store as fallback) to serve:
- Known entities (vehicles, persons) with sighting history
- Activity timeline (episodic events)
- Detected patterns (routines, anomalies)
- Scene-specific intelligence briefs

Data is scene-aware: home shows visitors/vehicles/anomalies,
retail shows demographics/ASP, security shows incidents.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger("trio.intelligence")

router = APIRouter(prefix="/api/intelligence", tags=["intelligence"])

# KG memory store path — sibling repo to trio-core
# intelligence.py → routers/ → api/ → trio_core/ → src/ → trio-core/ → trio-enterprise/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent

# Try trio-cortex first (new name), fall back to trio-kg (legacy)
_MEMORY_STORE_CANDIDATES = [
    _PROJECT_ROOT / "trio-cortex" / "memory_store",
    _PROJECT_ROOT / "trio-kg" / "memory_store",
]


def _get_memory_store() -> Path | None:
    """Return the first existing memory_store directory, or None."""
    for candidate in _MEMORY_STORE_CANDIDATES:
        if candidate.is_dir():
            return candidate
    return None


def _load_json(filename: str) -> dict | list:
    """Load a JSON file from the memory store."""
    store_dir = _get_memory_store()
    if store_dir is None:
        logger.warning("No memory_store found in trio-cortex or trio-kg")
        return {} if filename.endswith(".json") else []
    path = store_dir / filename
    if not path.exists():
        return {} if filename.endswith(".json") else []
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {} if "semantic" in filename or "procedural" in filename else []


@router.get("/known-entities")
async def known_entities(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
):
    """Known entities (vehicles, persons) learned by semantic memory.

    Returns entities with sighting count, first/last seen, and regular status.
    """
    # Tenant check
    from trio_core.api.routers.auth import get_tenant_camera_ids
    store = getattr(request.app.state, "event_store", None)
    if store:
        allowed = await get_tenant_camera_ids(request, store)
        if allowed is not None and camera_id not in allowed:
            raise HTTPException(403, "Access denied")

    semantic = _load_json("semantic.json")
    cam_data = semantic.get(camera_id, {})

    vehicles = cam_data.get("known_vehicles", [])
    persons = cam_data.get("known_persons", [])

    return {
        "camera_id": camera_id,
        "camera_name": cam_data.get("camera_name", ""),
        "vehicles": vehicles,
        "persons": persons,
        "total_observations": cam_data.get("total_observations", 0),
        "quiet_hours": cam_data.get("quiet_hours", []),
        "avg_people_per_hour": cam_data.get("avg_people_per_hour", 0),
        "typical_actions": cam_data.get("typical_actions", []),
        "last_updated": cam_data.get("last_updated", ""),
    }


@router.get("/activity-timeline")
async def activity_timeline(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
    limit: int = Query(50, ge=1, le=200),
):
    """Recent activity events from episodic memory.

    Returns timestamped events with change details and severity.
    """
    from trio_core.api.routers.auth import get_tenant_camera_ids
    store = getattr(request.app.state, "event_store", None)
    if store:
        allowed = await get_tenant_camera_ids(request, store)
        if allowed is not None and camera_id not in allowed:
            raise HTTPException(403, "Access denied")

    episodic = _load_json("episodic.json")
    events = [e for e in episodic if e.get("camera_id") == camera_id]
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)

    return {
        "camera_id": camera_id,
        "events": events[:limit],
        "total": len(events),
    }


@router.get("/patterns")
async def detected_patterns(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
):
    """Detected patterns from procedural memory.

    Returns learned routines, trends, and anomaly clusters.
    """
    from trio_core.api.routers.auth import get_tenant_camera_ids
    store = getattr(request.app.state, "event_store", None)
    if store:
        allowed = await get_tenant_camera_ids(request, store)
        if allowed is not None and camera_id not in allowed:
            raise HTTPException(403, "Access denied")

    procedural = _load_json("procedural.json")

    # Hourly patterns
    hourly = _load_json("hourly_patterns.json")
    cam_hourly = hourly.get(camera_id, {})

    return {
        "camera_id": camera_id,
        "rules": procedural.get("rules", []),
        "hourly_patterns": cam_hourly,
    }


@router.get("/brief")
async def intelligence_brief(
    request: Request,
    camera_id: str = Query(..., description="Camera ID"),
):
    """Scene-specific intelligence brief — combines all memory layers.

    Returns a structured summary tailored to the scene type.
    """
    from trio_core.api.routers.auth import get_tenant_camera_ids
    store = getattr(request.app.state, "event_store", None)
    if store:
        allowed = await get_tenant_camera_ids(request, store)
        if allowed is not None and camera_id not in allowed:
            raise HTTPException(403, "Access denied")

    semantic = _load_json("semantic.json")
    cam_data = semantic.get(camera_id, {})
    episodic = _load_json("episodic.json")
    cam_events = [e for e in episodic if e.get("camera_id") == camera_id]

    vehicles = cam_data.get("known_vehicles", [])
    persons = cam_data.get("known_persons", [])
    total_obs = cam_data.get("total_observations", 0)

    # Build intelligence brief
    regular_vehicles = [v for v in vehicles if v.get("is_regular")]
    unknown_vehicles = [v for v in vehicles if not v.get("is_regular")]
    regular_persons = [p for p in persons if p.get("is_regular")]

    # Severity distribution
    severity_counts = Counter(e.get("severity", "info") for e in cam_events)

    # Recent anomalies
    anomalies = [e for e in cam_events if e.get("severity") in ("high", "critical")]
    recent_anomalies = sorted(anomalies, key=lambda e: e.get("timestamp", ""), reverse=True)[:5]

    # Activity by hour
    hour_counts = Counter()
    for e in cam_events:
        ts = e.get("timestamp", "")
        try:
            dt = datetime.fromisoformat(ts)
            hour_counts[dt.hour] += 1
        except Exception:
            pass

    peak_hour = hour_counts.most_common(1)[0][0] if hour_counts else None

    return {
        "camera_id": camera_id,
        "camera_name": cam_data.get("camera_name", ""),
        "total_observations": total_obs,
        "summary": {
            "known_vehicles": len(vehicles),
            "regular_vehicles": len(regular_vehicles),
            "unknown_vehicles": len(unknown_vehicles),
            "known_persons": len(persons),
            "regular_visitors": len(regular_persons),
            "total_events": len(cam_events),
            "severity_distribution": dict(severity_counts),
            "peak_activity_hour": peak_hour,
        },
        "regular_vehicles": regular_vehicles,
        "unknown_vehicles": unknown_vehicles,
        "recent_anomalies": recent_anomalies,
        "quiet_hours": cam_data.get("quiet_hours", []),
    }
