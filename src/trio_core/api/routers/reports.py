"""Reports API router for Trio Console."""

from __future__ import annotations

import logging
import os
from datetime import date

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("trio.console.reports")

router = APIRouter(prefix="/api/reports", tags=["console-reports"])

GEMINI_MODEL = "gemini-2.0-flash"


class HourlyStat(BaseModel):
    hour: int
    events: int
    alerts: int


class CameraStat(BaseModel):
    camera_id: str
    camera_name: str
    events: int
    alerts: int


class AnomalyEvent(BaseModel):
    id: str
    timestamp: str
    camera_id: str
    camera_name: str
    description: str


class DailyReport(BaseModel):
    date: str
    total_events: int
    total_alerts: int
    hourly: list[HourlyStat] = Field(default_factory=list)
    cameras: list[CameraStat] = Field(default_factory=list)
    anomalies: list[AnomalyEvent] = Field(default_factory=list)
    summary: str = ""


class DayCount(BaseModel):
    date: str
    events: int
    alerts: int


class TrendReport(BaseModel):
    from_date: str
    to_date: str
    days: list[DayCount] = Field(default_factory=list)
    total_events: int = 0
    total_alerts: int = 0


def _get_store(request: Request):
    from fastapi import HTTPException
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


def _generate_summary_with_gemini(report: dict) -> str:
    """Call Gemini to generate a daily summary. Returns empty string on failure."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        prompt = (
            f"Summarize this security camera daily report in 2-3 sentences.\n"
            f"Date: {report['date']}\n"
            f"Total events: {report['total_events']}, Total alerts: {report['total_alerts']}\n"
            f"Cameras: {len(report['cameras'])}\n"
            f"Alert events: {len(report['anomalies'])}\n"
        )
        if report["anomalies"]:
            descs = [a["description"] for a in report["anomalies"][:10]]
            prompt += "Alert descriptions:\n" + "\n".join(f"- {d}" for d in descs) + "\n"

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.3, "max_output_tokens": 256},
        )
        return response.text or ""
    except Exception as e:
        logger.warning("Gemini summary generation failed: %s", e)
        return ""


def _stats_summary(report: dict) -> str:
    """Simple stats-based summary when Gemini is not available."""
    total = report["total_events"]
    alerts = report["total_alerts"]
    cams = len(report["cameras"])
    if total == 0:
        return f"No events recorded on {report['date']}."
    return (
        f"{total} events detected across {cams} camera(s) on {report['date']}, "
        f"with {alerts} alert(s) triggered."
    )


@router.get("/daily", response_model=DailyReport)
async def daily_report(
    request: Request,
    date_str: str = Query(None, alias="date", description="YYYY-MM-DD, default today"),
    camera_id: str | None = Query(None),
):
    """Generate a daily report with hourly breakdown and per-camera stats."""
    store = _get_store(request)
    if date_str is None:
        date_str = date.today().isoformat()

    report = await store.daily_report(date_str, camera_id)

    # Generate summary
    gemini_summary = _generate_summary_with_gemini(report)
    report["summary"] = gemini_summary if gemini_summary else _stats_summary(report)

    return DailyReport(**report)


@router.get("/export")
async def export_report(
    request: Request,
    date_str: str = Query(None, alias="date", description="YYYY-MM-DD"),
    camera_id: str | None = Query(None),
):
    """Export daily report as downloadable text file."""
    from fastapi.responses import Response

    store = _get_store(request)
    if date_str is None:
        date_str = date.today().isoformat()

    report = await store.daily_report(date_str, camera_id)
    gemini_summary = _generate_summary_with_gemini(report)
    summary = gemini_summary if gemini_summary else _stats_summary(report)

    # Build text report
    lines = [
        f"TRIO ENTERPRISE — DAILY INTELLIGENCE REPORT",
        f"Date: {date_str}",
        f"",
        f"SUMMARY",
        summary,
        f"",
        f"STATISTICS",
        f"  Total Events: {report['total_events']}",
        f"  Total Alerts: {report['total_alerts']}",
        f"  Cameras: {len(report['cameras'])}",
        f"",
        f"HOURLY BREAKDOWN",
    ]
    for h in report.get("hourly", []):
        lines.append(f"  {h['hour']:02d}:00  events={h['events']}  alerts={h['alerts']}")

    if report.get("anomalies"):
        lines.append(f"")
        lines.append(f"ANOMALIES DETECTED ({len(report['anomalies'])})")
        for a in report["anomalies"]:
            lines.append(f"  [{a['timestamp']}] {a['camera_name']}: {a['description']}")

    lines.append(f"")
    lines.append(f"Generated by Trio Enterprise")

    content = "\n".join(lines)
    return Response(
        content=content,
        media_type="text/plain",
        headers={
            "Content-Disposition": f"attachment; filename=trio_report_{date_str}.txt",
        },
    )


@router.get("/trend", response_model=TrendReport)
async def trend_report(
    request: Request,
    from_date: str = Query(..., description="Start date YYYY-MM-DD"),
    to_date: str = Query(..., description="End date YYYY-MM-DD"),
    camera_id: str | None = Query(None),
):
    """Get event/alert trend over a date range."""
    store = _get_store(request)
    result = await store.trend_report(from_date, to_date, camera_id)
    return TrendReport(**result)
