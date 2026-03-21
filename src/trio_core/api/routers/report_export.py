"""PDF/HTML/CSV report export for Trio Enterprise."""

from __future__ import annotations

import csv
import io
import logging
import os
from datetime import date

from fastapi import APIRouter, Query, Request
from fastapi.responses import HTMLResponse, Response

logger = logging.getLogger("trio.console.report_export")

router = APIRouter(prefix="/api/reports", tags=["console-reports"])


def _get_store(request: Request):
    from fastapi import HTTPException

    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


def _generate_executive_summary(report: dict) -> str:
    """Generate executive summary via Gemini or stats-based fallback."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key:
        try:
            from google import genai

            client = genai.Client(api_key=api_key)
            prompt = (
                "You are a senior intelligence analyst at a physical security firm. "
                "Write a concise executive summary (3-4 sentences) for a client-facing daily "
                "intelligence report based on the following data. Use professional, authoritative "
                "language. Do not use bullet points.\n\n"
                f"Date: {report['date']}\n"
                f"Total events: {report['total_events']}\n"
                f"Total alerts: {report['total_alerts']}\n"
                f"Active cameras: {len(report['cameras'])}\n"
            )
            if report.get("anomalies"):
                descs = [a["description"] for a in report["anomalies"][:10]]
                prompt += "Alert descriptions:\n" + "\n".join(f"- {d}" for d in descs) + "\n"
            if report.get("hourly"):
                peak = max(report["hourly"], key=lambda h: h["events"])
                prompt += f"Peak hour: {peak['hour']:02d}:00 with {peak['events']} events\n"

            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt,
                config={"temperature": 0.3, "max_output_tokens": 300},
            )
            text = (response.text or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning("Gemini summary failed: %s", e)

    # Stats-based fallback
    total = report["total_events"]
    alerts = report["total_alerts"]
    cams = len(report["cameras"])
    if total == 0:
        return f"No significant activity was recorded on {report['date']}. All monitored zones remained within normal operating parameters."

    peak_str = ""
    if report.get("hourly"):
        peak = max(report["hourly"], key=lambda h: h["events"])
        peak_str = f" Peak activity was observed at {peak['hour']:02d}:00 hours."

    alert_str = ""
    if alerts > 0:
        alert_str = f" A total of {alerts} alert(s) were triggered, requiring further review."

    return (
        f"During the reporting period of {report['date']}, the monitoring system recorded "
        f"{total} events across {cams} active camera(s).{peak_str}{alert_str}"
    )


def _build_hourly_rows(report: dict) -> str:
    """Build hourly activity table rows."""
    hourly_map: dict[int, dict] = {}
    for h in report.get("hourly", []):
        hourly_map[h["hour"]] = h

    # Find peak hour
    peak_hour = -1
    peak_count = 0
    for h, data in hourly_map.items():
        if data["events"] > peak_count:
            peak_count = data["events"]
            peak_hour = h

    rows = []
    total_events = 0
    total_alerts = 0
    for hour in range(24):
        data = hourly_map.get(hour)
        events = data["events"] if data else 0
        alerts = data["alerts"] if data else 0
        total_events += events
        total_alerts += alerts
        bold = "font-weight:700;" if hour == peak_hour else ""
        bg = "background:#f8f9fa;" if hour % 2 == 0 else ""
        rows.append(
            f'<tr style="{bg}{bold}">'
            f"<td>{hour:02d}:00&ndash;{hour:02d}:59</td>"
            f"<td>{events}</td>"
            f"<td>{alerts}</td>"
            f"</tr>"
        )
    rows.append(
        f'<tr style="font-weight:700;border-top:2px solid #1a1a2e;">'
        f"<td>TOTAL</td>"
        f"<td>{total_events}</td>"
        f"<td>{total_alerts}</td>"
        f"</tr>"
    )
    return "\n".join(rows)


def _build_detection_rows(events: list[dict]) -> str:
    """Build detection breakdown from event descriptions."""
    # Parse detection classes from descriptions
    class_counts: dict[str, int] = {}
    class_peak_hour: dict[str, dict[int, int]] = {}

    keywords = [
        "person", "people", "car", "truck", "bus", "motorcycle", "bicycle",
        "dog", "cat", "bird", "vehicle", "package", "bag",
    ]
    for evt in events:
        desc = (evt.get("description") or "").lower()
        ts = evt.get("timestamp", "")
        try:
            hour = int(ts[11:13]) if len(ts) > 13 else 0
        except (ValueError, IndexError):
            hour = 0

        matched = False
        for kw in keywords:
            if kw in desc:
                class_counts[kw] = class_counts.get(kw, 0) + 1
                if kw not in class_peak_hour:
                    class_peak_hour[kw] = {}
                class_peak_hour[kw][hour] = class_peak_hour[kw].get(hour, 0) + 1
                matched = True
        if not matched:
            class_counts["other"] = class_counts.get("other", 0) + 1

    if not class_counts:
        return '<tr><td colspan="3" style="text-align:center;color:#666;">No detections recorded</td></tr>'

    rows = []
    idx = 0
    for cls, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        peak_h = ""
        if cls in class_peak_hour:
            best_hour = max(class_peak_hour[cls], key=class_peak_hour[cls].get)  # type: ignore[arg-type]
            peak_h = f"{best_hour:02d}:00"
        bg = "background:#f8f9fa;" if idx % 2 == 0 else ""
        rows.append(
            f'<tr style="{bg}">'
            f"<td>{cls.title()}</td>"
            f"<td>{count}</td>"
            f"<td>{peak_h}</td>"
            f"</tr>"
        )
        idx += 1
    return "\n".join(rows)


def _build_event_log_rows(events: list[dict]) -> str:
    """Build event log table rows."""
    if not events:
        return '<tr><td colspan="4" style="text-align:center;color:#666;">No events recorded</td></tr>'
    rows = []
    for idx, evt in enumerate(events):
        ts = evt.get("timestamp", "")
        # Format timestamp to HH:MM:SS
        time_str = ts[11:19] if len(ts) >= 19 else ts
        camera = evt.get("camera_name") or evt.get("camera_id", "")
        desc = evt.get("description", "")
        if len(desc) > 100:
            desc = desc[:97] + "..."
        alert = "YES" if evt.get("alert_triggered") else ""
        alert_style = "color:#dc3545;font-weight:700;" if alert else ""
        bg = "background:#f8f9fa;" if idx % 2 == 0 else ""
        rows.append(
            f'<tr style="{bg}">'
            f"<td>{time_str}</td>"
            f"<td>{camera}</td>"
            f"<td>{desc}</td>"
            f'<td style="{alert_style}">{alert}</td>'
            f"</tr>"
        )
    return "\n".join(rows)


def _build_anomaly_rows(anomalies: list[dict]) -> str:
    """Build anomaly section rows."""
    if not anomalies:
        return '<tr><td colspan="4" style="text-align:center;color:#666;">No anomalies detected &mdash; all clear.</td></tr>'
    rows = []
    for idx, a in enumerate(anomalies):
        ts = a.get("timestamp", "")
        time_str = ts[11:19] if len(ts) >= 19 else ts
        camera = a.get("camera_name") or a.get("camera_id", "")
        desc = a.get("description", "")
        if len(desc) > 120:
            desc = desc[:117] + "..."
        bg = "background:#f8f9fa;" if idx % 2 == 0 else ""
        rows.append(
            f'<tr style="{bg}">'
            f"<td>HIGH</td>"
            f"<td>{time_str}</td>"
            f"<td>{camera}</td>"
            f"<td>{desc}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def _build_html_report(
    report: dict,
    events: list[dict],
    summary: str,
    facility_name: str,
) -> str:
    """Build the complete HTML report as a single f-string."""
    report_date = report.get("date", "")
    total_events = report.get("total_events", 0)
    total_alerts = report.get("total_alerts", 0)
    num_cameras = len(report.get("cameras", []))
    anomalies = report.get("anomalies", [])

    hourly_rows = _build_hourly_rows(report)
    detection_rows = _build_detection_rows(events)
    event_log_rows = _build_event_log_rows(events)
    anomaly_rows = _build_anomaly_rows(anomalies)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Trio Enterprise Intelligence Report &mdash; {report_date}</title>
<style>
@page {{
    size: A4;
    margin: 20mm 15mm 25mm 15mm;
    @top-center {{
        content: "CONFIDENTIAL — Trio Enterprise";
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #999;
    }}
    @bottom-center {{
        content: "Page " counter(page) " of " counter(pages);
        font-family: Helvetica, Arial, sans-serif;
        font-size: 8pt;
        color: #999;
    }}
}}
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{
    font-family: Helvetica, Arial, sans-serif;
    color: #1a1a2e;
    font-size: 10pt;
    line-height: 1.5;
}}

/* ── Cover Page ─────────────────────────────────────────── */
.cover {{
    page-break-after: always;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    min-height: 100vh;
    text-align: center;
    position: relative;
}}
.cover .logo {{
    font-size: 36pt;
    font-weight: 800;
    letter-spacing: 8px;
    color: #1a1a2e;
    margin-bottom: 4px;
}}
.cover .logo-accent {{
    color: #2563eb;
}}
.cover .subtitle {{
    font-size: 18pt;
    font-weight: 300;
    color: #444;
    margin-bottom: 48px;
    letter-spacing: 4px;
    text-transform: uppercase;
}}
.cover .meta {{
    font-size: 11pt;
    color: #666;
    line-height: 2;
}}
.cover .confidential {{
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%) rotate(-35deg);
    font-size: 72pt;
    font-weight: 800;
    color: rgba(0,0,0,0.03);
    letter-spacing: 16px;
    white-space: nowrap;
    pointer-events: none;
    z-index: 0;
}}
.cover .footer {{
    position: absolute;
    bottom: 40px;
    font-size: 8pt;
    color: #999;
}}

/* ── Content Pages ──────────────────────────────────────── */
.page {{
    page-break-before: always;
    padding-top: 8px;
}}
.page-header {{
    font-size: 7pt;
    color: #999;
    text-align: center;
    margin-bottom: 16px;
    letter-spacing: 2px;
    text-transform: uppercase;
}}
h2 {{
    font-size: 16pt;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 6px;
    padding-bottom: 6px;
    border-bottom: 3px solid #2563eb;
}}
h3 {{
    font-size: 11pt;
    font-weight: 600;
    color: #1a1a2e;
    margin: 16px 0 8px;
}}

/* ── Metrics Grid ───────────────────────────────────────── */
.metrics {{
    display: flex;
    gap: 16px;
    margin: 20px 0;
}}
.metric-card {{
    flex: 1;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 16px;
    text-align: center;
}}
.metric-value {{
    font-size: 28pt;
    font-weight: 700;
    color: #1a1a2e;
}}
.metric-label {{
    font-size: 8pt;
    color: #888;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}}

/* ── Tables ─────────────────────────────────────────────── */
table {{
    width: 100%;
    border-collapse: collapse;
    margin: 12px 0 20px;
    font-size: 9pt;
}}
th {{
    background: #1a1a2e;
    color: #fff;
    padding: 8px 10px;
    text-align: left;
    font-weight: 600;
    font-size: 8pt;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}}
td {{
    padding: 6px 10px;
    border-bottom: 1px solid #e8e8e8;
}}

.summary-text {{
    font-size: 10.5pt;
    line-height: 1.7;
    color: #333;
    margin: 16px 0;
    padding: 16px;
    background: #fafafa;
    border-left: 3px solid #2563eb;
}}
</style>
</head>
<body>

<!-- ══════════════════════════════════════════════════════════
     PAGE 1 — COVER
     ══════════════════════════════════════════════════════════ -->
<div class="cover">
    <div class="confidential">CONFIDENTIAL</div>
    <div class="logo"><span class="logo-accent">T</span>RIO ENTERPRISE</div>
    <div class="subtitle">Intelligence Report</div>
    <div class="meta">
        <strong>{report_date}</strong><br>
        {facility_name}<br>
        Report Period: 00:00 &ndash; 23:59
    </div>
    <div class="footer">
        Generated by Trio Enterprise &nbsp;|&nbsp; Powered by MachineFi
    </div>
</div>

<!-- ══════════════════════════════════════════════════════════
     PAGE 2 — EXECUTIVE SUMMARY
     ══════════════════════════════════════════════════════════ -->
<div class="page">
    <div class="page-header">Confidential &mdash; Trio Enterprise</div>
    <h2>Executive Summary</h2>
    <div class="summary-text">{summary}</div>

    <h3>Key Metrics</h3>
    <div class="metrics">
        <div class="metric-card">
            <div class="metric-value">{total_events}</div>
            <div class="metric-label">Total Events</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{total_alerts}</div>
            <div class="metric-label">Alerts Triggered</div>
        </div>
        <div class="metric-card">
            <div class="metric-value">{num_cameras}</div>
            <div class="metric-label">Cameras Online</div>
        </div>
    </div>

    <h3>Period Covered</h3>
    <p>{report_date} &nbsp; 00:00:00 &ndash; 23:59:59 UTC</p>
</div>

<!-- ══════════════════════════════════════════════════════════
     PAGE 3 — HOURLY ACTIVITY
     ══════════════════════════════════════════════════════════ -->
<div class="page">
    <div class="page-header">Confidential &mdash; Trio Enterprise</div>
    <h2>Hourly Activity Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Hour</th>
                <th>Events</th>
                <th>Alerts</th>
            </tr>
        </thead>
        <tbody>
            {hourly_rows}
        </tbody>
    </table>
</div>

<!-- ══════════════════════════════════════════════════════════
     PAGE 4 — DETECTION BREAKDOWN
     ══════════════════════════════════════════════════════════ -->
<div class="page">
    <div class="page-header">Confidential &mdash; Trio Enterprise</div>
    <h2>Detection Breakdown</h2>
    <table>
        <thead>
            <tr>
                <th>Class</th>
                <th>Total Count</th>
                <th>Peak Hour</th>
            </tr>
        </thead>
        <tbody>
            {detection_rows}
        </tbody>
    </table>
</div>

<!-- ══════════════════════════════════════════════════════════
     PAGE 5 — EVENT LOG
     ══════════════════════════════════════════════════════════ -->
<div class="page">
    <div class="page-header">Confidential &mdash; Trio Enterprise</div>
    <h2>Event Log</h2>
    <table>
        <thead>
            <tr>
                <th>Time</th>
                <th>Camera</th>
                <th>Description</th>
                <th>Alert</th>
            </tr>
        </thead>
        <tbody>
            {event_log_rows}
        </tbody>
    </table>
</div>

<!-- ══════════════════════════════════════════════════════════
     PAGE 6 — ANOMALIES
     ══════════════════════════════════════════════════════════ -->
<div class="page">
    <div class="page-header">Confidential &mdash; Trio Enterprise</div>
    <h2>Anomalies &amp; Alerts</h2>
    <table>
        <thead>
            <tr>
                <th>Severity</th>
                <th>Time</th>
                <th>Camera</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            {anomaly_rows}
        </tbody>
    </table>
</div>

</body>
</html>"""


@router.get("/export")
async def export_report(
    request: Request,
    date_str: str = Query(None, alias="date", description="YYYY-MM-DD, default today"),
    camera_id: str | None = Query(None),
    facility: str = Query("Facility Alpha", description="Facility name for the cover page"),
    format: str = Query("pdf", description="pdf or html"),
):
    """Export a professional intelligence report as PDF (or HTML fallback)."""
    store = _get_store(request)
    if date_str is None:
        date_str = date.today().isoformat()

    # Fetch data
    report = await store.daily_report(date_str, camera_id)

    # Fetch all events for the day (for event log and detection parsing)
    day_start = f"{date_str}T00:00:00"
    day_end = f"{date_str}T23:59:59"
    events_result = await store.list_events(
        camera_id=camera_id, start=day_start, end=day_end, limit=10000
    )
    all_events = events_result.get("events", [])

    # Generate summary
    summary = _generate_executive_summary(report)

    # Build HTML
    html = _build_html_report(report, all_events, summary, facility)
    filename = f"trio-report-{date_str}"

    # Try PDF conversion
    if format == "pdf":
        try:
            from weasyprint import HTML as WeasyHTML

            pdf_bytes = WeasyHTML(string=html).write_pdf()
            return Response(
                content=pdf_bytes,
                media_type="application/pdf",
                headers={
                    "Content-Disposition": f'attachment; filename="{filename}.pdf"',
                },
            )
        except ImportError:
            logger.info("weasyprint not installed, falling back to HTML export")
        except Exception as e:
            logger.warning("PDF generation failed: %s, falling back to HTML", e)

    # HTML fallback
    return HTMLResponse(
        content=html,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}.html"',
        },
    )


@router.get("/export-csv")
async def export_csv(
    request: Request,
    date_str: str = Query(None, alias="date", description="YYYY-MM-DD, default today"),
    camera_id: str | None = Query(None),
):
    """Export daily events as a CSV file."""
    store = _get_store(request)
    if date_str is None:
        date_str = date.today().isoformat()

    day_start = f"{date_str}T00:00:00"
    day_end = f"{date_str}T23:59:59"
    events_result = await store.list_events(
        camera_id=camera_id, start=day_start, end=day_end, limit=100000
    )
    all_events = events_result.get("events", [])

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["timestamp", "camera_name", "description", "alert_triggered"])
    for evt in all_events:
        writer.writerow([
            evt.get("timestamp", ""),
            evt.get("camera_name") or evt.get("camera_id", ""),
            evt.get("description", ""),
            "yes" if evt.get("alert_triggered") else "no",
        ])

    filename = f"trio-events-{date_str}.csv"
    return Response(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
