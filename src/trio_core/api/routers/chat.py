"""Chat API — conversational AI over camera event history."""

from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("trio.console.chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

GEMINI_MODEL = "gemini-2.0-flash"

SYSTEM_PROMPTS = {
    "security": """\
You are Trio Enterprise, an AI-powered physical security and video intelligence platform.
You serve as the virtual Security Operations Center (SOC) analyst for the facility.
You have access to a log of events detected by cameras. Each event has a timestamp,
camera name, and a description of what was observed.

Your role:
- Answer questions about access events, personnel movements, and security incidents
- Cross-reference events to identify patterns (e.g., repeat visitors, unusual timing)
- Assess threat levels and provide actionable security recommendations
- Reference specific timestamps and camera names in your answers
- Flag any compliance concerns (badge violations, after-hours access, tailgating)
- Be concise but thorough -- write like a senior SOC analyst briefing the CISO

If the event log is empty or doesn't contain relevant information, say so honestly.
Do not make up events that aren't in the log.
Always respond in the same language the user uses.""",

    "investment": """\
You are Trio Enterprise, an AI-powered video intelligence platform for investment research.
You serve as a senior research analyst providing alt-data insights from camera observations.
You have access to a log of events detected by cameras at retail locations. Each event has
a timestamp, camera name, and a description of what was observed.

Your role:
- Analyze foot traffic patterns, customer demographics, and purchasing behavior
- Calculate metrics: ASP (average selling price) from drink sizes, food attach rates
- Identify trends: peak hours, customer segments, competitive signals
- Reference specific data points and timestamps in your answers
- Provide investment-relevant observations (bullish/bearish signals)
- Be direct and data-driven -- write like a senior equity research analyst

If the event log is empty or doesn't contain relevant information, say so honestly.
Do not make up events that aren't in the log.
Always respond in the same language the user uses.""",
}

SYSTEM_PROMPT = SYSTEM_PROMPTS["security"]  # default fallback


class ChatRequest(BaseModel):
    message: str
    camera_id: str | None = None
    hours: float = 24  # how many hours back to search
    max_events: int = 50
    persona: str = "security"  # "security" or "investment"


class ReferencedEvent(BaseModel):
    id: str
    timestamp: str
    camera_name: str
    description: str
    frame_url: str


class ChatResponse(BaseModel):
    reply: str
    referenced_events: list[ReferencedEvent] = Field(default_factory=list)
    events_searched: int = 0


def _get_client():
    """Get Gemini client (lazy init)."""
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


def _parse_time_range(hours: float) -> tuple[str, str]:
    """Return (start, end) ISO strings for the given hours back from now."""
    now = datetime.now(timezone.utc)
    start = now - timedelta(hours=hours)
    return start.isoformat(), now.isoformat()


def _build_event_context(events: list[dict], camera_filter: str | None) -> str:
    """Format events as a text log for the LLM context."""
    if not events:
        return "(No events found in the specified time range)"

    lines = []
    for e in events:
        ts = e.get("timestamp", "")
        # Extract time portion for readability
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except (ValueError, AttributeError):
            time_str = ts

        cam = e.get("camera_name") or e.get("camera_id", "Unknown")
        desc = e.get("description", "")
        alert = " [ALERT]" if e.get("alert_triggered") else ""
        lines.append(f"[{time_str}] {cam}: {desc}{alert}")

    header = f"Event Log ({len(events)} events"
    if camera_filter:
        header += f", camera: {camera_filter}"
    header += ")"

    return f"--- {header} ---\n" + "\n".join(lines) + "\n---"


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Send a message and get an AI response grounded in event history."""
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        return ChatResponse(
            reply="The event store is not initialized. Please ensure the backend is properly configured.",
            referenced_events=[],
            events_searched=0,
        )

    # Query events from store
    start, end = _parse_time_range(req.hours)
    result = await store.list_events(
        camera_id=req.camera_id,
        start=start,
        end=end,
        limit=req.max_events,
    )
    events = result["events"]
    total = result["total"]

    # Build context
    event_context = _build_event_context(events, req.camera_id)

    # Enrich with InsightExtractor analysis (ASP, demographics, patterns)
    insights_context = ""
    try:
        from trio_core.insights import InsightExtractor
        extractor = InsightExtractor()
        insights = extractor.extract(events)
        if insights:
            insight_lines = [f"- [{ins.insight_type}] {ins.text}" for ins in insights]
            insights_context = (
                "\n\n--- AI-Extracted Analytics ---\n"
                + "\n".join(insight_lines)
                + "\n---"
            )
    except Exception:
        pass

    # Build prompt
    prompt = f"{event_context}{insights_context}\n\nUser: {req.message}"

    # Call Gemini
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "system_instruction": SYSTEM_PROMPTS.get(req.persona, SYSTEM_PROMPT),
                "temperature": 0.3,
                "max_output_tokens": 1024,
            },
        )
        reply = response.text or "I couldn't generate a response."
    except ValueError as e:
        reply = f"Chat not configured: {e}"
    except Exception as e:
        logger.exception("Gemini API error")
        reply = f"Error communicating with AI: {e}"

    # Find referenced events (events mentioned in the reply by timestamp)
    referenced = []
    for e in events:
        ts = e.get("timestamp", "")
        # Check if any part of the timestamp appears in the reply
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            time_str = dt.strftime("%H:%M")
            if time_str in reply or e.get("camera_name", "") in reply:
                referenced.append(ReferencedEvent(
                    id=e["id"],
                    timestamp=ts,
                    camera_name=e.get("camera_name", ""),
                    description=e.get("description", ""),
                    frame_url=f"/api/events/{e['id']}/frame",
                ))
        except (ValueError, AttributeError):
            pass

    return ChatResponse(
        reply=reply,
        referenced_events=referenced,
        events_searched=total,
    )
