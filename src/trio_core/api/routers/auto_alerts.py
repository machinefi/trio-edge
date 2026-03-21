"""Auto-alert detection — scan events for security-relevant patterns."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from fastapi import APIRouter, Request

logger = logging.getLogger("trio.auto_alerts")

router = APIRouter(prefix="/api/auto-alerts", tags=["auto-alerts"])

# Keywords that trigger alert flagging
ALERT_KEYWORDS = {
    "high": [
        "unauthorized", "tailgating", "weapon", "gun", "knife", "fight",
        "running", "screaming", "break-in", "forced entry", "suspicious",
    ],
    "medium": [
        "loitering", "no badge", "without badge", "unidentified",
        "after hours", "overnight", "unusual", "anomaly", "tamper",
    ],
    "low": [
        "door held open", "propped", "delivery", "unknown vehicle",
        "unfamiliar", "new face",
    ],
}


def _classify_event(description: str) -> tuple[bool, str, str]:
    """Check if an event description contains alert-worthy keywords.

    Returns (is_alert, severity, matched_keyword).
    """
    desc_lower = description.lower()
    for severity in ["high", "medium", "low"]:
        for keyword in ALERT_KEYWORDS[severity]:
            if keyword in desc_lower:
                return True, severity, keyword
    return False, "", ""


@router.get("/scan")
async def scan_events(request: Request, hours: float = 24):
    """Scan recent events for alert-worthy patterns.

    Returns flagged events with severity and reason.
    """
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        return {"alerts": [], "scanned": 0}

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=hours)).isoformat()

    result = await store.list_events(start=start, limit=500)
    events = result.get("events", [])

    flagged = []
    for event in events:
        desc = event.get("description", "")
        is_alert, severity, keyword = _classify_event(desc)
        if is_alert:
            flagged.append({
                "event_id": event["id"],
                "timestamp": event["timestamp"],
                "camera_name": event.get("camera_name", ""),
                "description": desc[:200],
                "severity": severity,
                "trigger": keyword,
            })

    return {
        "alerts": flagged,
        "scanned": len(events),
        "period_hours": hours,
        "generated_at": now.isoformat(),
    }
