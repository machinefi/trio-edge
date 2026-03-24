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
    "critical": [
        "weapon", "gun", "knife", "firearm", "explosive",
        "break-in", "forced entry", "intruder", "breach",
        "fire detected", "active fire", "smoke detected", "explosion",
    ],
    "high": [
        "unauthorized", "tailgating", "piggybacking", "piggyback",
        "fight", "assault", "threat",
        "person running", "people running", "screaming",
        "suspicious", "vandalism",
        "evacuation",
    ],
    "medium": [
        "loitering", "no badge", "without badge", "unidentified",
        "after hours", "overnight", "unusual", "anomaly", "tamper",
        "door held open", "propped", "propped open",
        "unknown vehicle", "unauthorized vehicle",
        "fire alarm", "fire drill", "alarm drill",
    ],
    "low": [
        "delivery", "unfamiliar", "new face",
        "lingered", "departed without",
    ],
}


def _classify_event(description: str) -> tuple[bool, str, str]:
    """Check if an event description contains alert-worthy keywords.

    Returns (is_alert, severity, matched_keyword).
    """
    desc_lower = description.lower()
    for severity in ["critical", "high", "medium", "low"]:
        for keyword in ALERT_KEYWORDS[severity]:
            if keyword in desc_lower:
                return True, severity, keyword
    return False, "", ""


@router.get("/scan")
async def scan_events(request: Request, hours: float = 24, camera_id: str | None = None):
    """Scan recent events for alert-worthy patterns.

    Returns flagged events with severity and reason.
    Optionally filter by camera_id.
    """
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        return {"alerts": [], "scanned": 0}

    from datetime import timedelta
    now = datetime.now(timezone.utc)
    start = (now - timedelta(hours=hours)).isoformat()

    result = await store.list_events(camera_id=camera_id, start=start, limit=500)
    events = result.get("events", [])

    flagged = []
    for event in events:
        desc = event.get("description", "")
        is_alert, severity, keyword = _classify_event(desc)
        # Also flag events that have alert_triggered set in the DB
        if not is_alert and event.get("alert_triggered"):
            is_alert = True
            severity = "medium"
            keyword = "flagged by system"
        if is_alert:
            flagged.append({
                "event_id": event["id"],
                "timestamp": event["timestamp"],
                "camera_name": event.get("camera_name", ""),
                "camera_id": event.get("camera_id", ""),
                "description": desc[:200],
                "severity": severity,
                "trigger": keyword,
            })

    # Compute overall threat level from most severe alert
    severity_order = ["critical", "high", "medium", "low"]
    threat_level = "LOW"
    for s in severity_order:
        if any(a["severity"] == s for a in flagged):
            threat_level = s.upper()
            break
    if not flagged:
        threat_level = "ALL CLEAR"

    return {
        "alerts": flagged,
        "scanned": len(events),
        "period_hours": hours,
        "generated_at": now.isoformat(),
        "threat_level": threat_level,
    }
