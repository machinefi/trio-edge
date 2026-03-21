"""Insights API — auto-generated analytics from camera event descriptions."""

from __future__ import annotations

import logging
import os
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, HTTPException, Query, Request

logger = logging.getLogger("trio.console.insights")

router = APIRouter(prefix="/api/insights", tags=["insights"])

GEMINI_MODEL = "gemini-2.0-flash"

# ── Extraction patterns ──────────────────────────────────────────────────────

GENDER_PATTERNS = {
    "male": re.compile(r"\b(male|man|boy|gentleman|he\b|his\b)", re.I),
    "female": re.compile(r"\b(female|woman|girl|lady|she\b|her\b)", re.I),
}

AGE_PATTERNS = {
    "young": re.compile(
        r"\b(young|youth|teen|child|kid|20s|early 20|young adult|toddler|"
        r"around \d-\d years old|3-5 years|boy|girl)\b",
        re.I,
    ),
    "middle_aged": re.compile(
        r"\b(middle.?aged?|30s|40s|50s|mid-\d0s)\b", re.I
    ),
    "elderly": re.compile(
        r"\b(elderly|older|senior|aged|gray hair|grey hair|walking stick|"
        r"70s|80s|old man|old woman)\b",
        re.I,
    ),
}

ETHNICITY_PATTERNS = {
    "caucasian": re.compile(r"\bcaucasian\b", re.I),
    "asian": re.compile(r"\basian\b", re.I),
    "african_american": re.compile(r"\b(african.?american|black)\b", re.I),
    "hispanic": re.compile(r"\bhispanic\b", re.I),
}

ACTIVITY_PATTERNS = {
    "walking": re.compile(r"\b(walk(?:ing|s)?|crossing|strolling)\b", re.I),
    "standing": re.compile(r"\b(stand(?:ing|s)?|waiting|stationary)\b", re.I),
    "running": re.compile(r"\b(run(?:ning|s)?|jogging|sprinting)\b", re.I),
    "sitting": re.compile(r"\b(sit(?:ting|s)?|seated)\b", re.I),
    "carrying_bag": re.compile(r"\b(carrying a bag|bag|backpack|carrying.*bag)\b", re.I),
    "carrying_briefcase": re.compile(r"\b(briefcase|attache)\b", re.I),
    "driving": re.compile(r"\b(driv(?:ing|es?))\b", re.I),
    "parking": re.compile(r"\b(park(?:ed|ing))\b", re.I),
}

ITEM_PATTERNS = [
    "briefcase", "backpack", "bag", "umbrella", "phone", "laptop",
    "stroller", "bicycle", "skateboard", "suitcase", "paper",
    "coffee", "dog", "leash", "package",
]

VEHICLE_TYPE_PATTERNS = {
    "car": re.compile(r"\b(car|sedan|coupe|hatchback)\b", re.I),
    "suv": re.compile(r"\b(suv|crossover)\b", re.I),
    "truck": re.compile(r"\b(truck|pickup)\b", re.I),
    "bus": re.compile(r"\b(bus)\b", re.I),
    "van": re.compile(r"\b(van|minivan)\b", re.I),
    "motorcycle": re.compile(r"\b(motorcycle|motorbike|scooter)\b", re.I),
    "station_wagon": re.compile(r"\b(station wagon|wagon)\b", re.I),
}

COLOR_PATTERNS = [
    "white", "black", "silver", "gray", "grey", "red", "blue",
    "green", "yellow", "brown", "beige", "orange", "purple", "gold",
]

VEHICLE_BRANDS = [
    "toyota", "honda", "ford", "bmw", "tesla", "mercedes", "audi",
    "chevrolet", "hyundai", "kia", "volkswagen", "nissan", "subaru",
    "lexus", "mazda", "jeep", "volvo", "porsche",
    "corolla", "camry", "civic", "accord", "mustang", "f-150",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


def _time_range(
    start: str | None, end: str | None, hours: float,
) -> tuple[str, str]:
    """Compute (start, end) ISO strings."""
    now = datetime.now(timezone.utc)
    if end:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
    else:
        end_dt = now
    if start:
        start_dt = datetime.fromisoformat(start.replace("Z", "+00:00"))
    else:
        start_dt = end_dt - timedelta(hours=hours)
    return start_dt.isoformat(), end_dt.isoformat()


async def _fetch_descriptions(
    request: Request,
    camera_id: str | None,
    start: str | None,
    end: str | None,
    hours: float,
) -> list[dict]:
    """Fetch event descriptions from the store with time/camera filters."""
    store = _get_store(request)
    t_start, t_end = _time_range(start, end, hours)
    result = await store.list_events(
        camera_id=camera_id, start=t_start, end=t_end, limit=10000,
    )
    return result["events"]


def _is_person_event(desc: str) -> bool:
    """Check if description is about a person."""
    return bool(re.search(
        r"\b(person|man|woman|boy|girl|child|people|pedestrian|adult|"
        r"gentleman|lady|he |she |his |her )\b",
        desc, re.I,
    ))


def _is_vehicle_event(desc: str) -> bool:
    """Check if description is about a vehicle."""
    return bool(re.search(
        r"\b(car|truck|bus|van|suv|vehicle|sedan|pickup|motorcycle|"
        r"station wagon|driving|parked)\b",
        desc, re.I,
    ))


def _extract_hour(ts: str) -> int | None:
    """Extract hour from ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.hour
    except (ValueError, AttributeError):
        return None


# ── Demographics ──────────────────────────────────────────────────────────────

@router.get("/demographics")
async def demographics(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None, description="ISO date start"),
    end: str | None = Query(None, description="ISO date end"),
    hours: float = Query(24, description="Hours back from now (default 24)"),
):
    """Analyze event descriptions to extract demographic breakdown."""
    events = await _fetch_descriptions(request, camera_id, start, end, hours)

    person_events = [e for e in events if _is_person_event(e.get("description", ""))]
    total_people = len(person_events)

    gender: Counter = Counter()
    age_groups: Counter = Counter()
    ethnicity: Counter = Counter()
    demographics_by_hour: dict[int, dict] = defaultdict(
        lambda: {"male": 0, "female": 0, "unknown_gender": 0, "total": 0}
    )

    for e in person_events:
        desc = e.get("description", "")
        hour = _extract_hour(e.get("timestamp", ""))

        # Gender
        g = "unknown"
        for label, pat in GENDER_PATTERNS.items():
            if pat.search(desc):
                g = label
                break
        gender[g] += 1
        if hour is not None:
            key = g if g in ("male", "female") else "unknown_gender"
            demographics_by_hour[hour][key] += 1
            demographics_by_hour[hour]["total"] += 1

        # Age
        a = "unknown"
        for label, pat in AGE_PATTERNS.items():
            if pat.search(desc):
                a = label
                break
        age_groups[a] += 1

        # Ethnicity
        for label, pat in ETHNICITY_PATTERNS.items():
            if pat.search(desc):
                ethnicity[label] += 1

    # Sort by hour
    hourly_list = [
        {"hour": h, **demographics_by_hour[h]}
        for h in sorted(demographics_by_hour)
    ]

    return {
        "total_people": total_people,
        "total_events_analyzed": len(events),
        "gender": {
            "male": gender.get("male", 0),
            "female": gender.get("female", 0),
            "unknown": gender.get("unknown", 0),
        },
        "age_groups": {
            "young": age_groups.get("young", 0),
            "middle_aged": age_groups.get("middle_aged", 0),
            "elderly": age_groups.get("elderly", 0),
            "unknown": age_groups.get("unknown", 0),
        },
        "ethnicity": dict(ethnicity.most_common()),
        "demographics_by_hour": hourly_list,
    }


# ── Vehicles ──────────────────────────────────────────────────────────────────

@router.get("/vehicles")
async def vehicles(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    hours: float = Query(24),
):
    """Vehicle analytics — types, colors, brands from event descriptions."""
    events = await _fetch_descriptions(request, camera_id, start, end, hours)

    vehicle_events = [e for e in events if _is_vehicle_event(e.get("description", ""))]
    total_vehicles = len(vehicle_events)

    by_type: Counter = Counter()
    by_color: Counter = Counter()
    brands_detected: Counter = Counter()

    for e in vehicle_events:
        desc = e.get("description", "").lower()

        # Vehicle type
        for label, pat in VEHICLE_TYPE_PATTERNS.items():
            if pat.search(desc):
                by_type[label] += 1

        # Color
        for color in COLOR_PATTERNS:
            if re.search(rf"\b{color}\b", desc):
                by_color[color] += 1

        # Brand/model
        for brand in VEHICLE_BRANDS:
            if brand in desc:
                brands_detected[brand] += 1

    return {
        "total_vehicles": total_vehicles,
        "total_events_analyzed": len(events),
        "by_type": dict(by_type.most_common()),
        "by_color": dict(by_color.most_common()),
        "brands_detected": [
            {"brand": b, "count": c} for b, c in brands_detected.most_common()
        ],
    }


# ── Behavioral ────────────────────────────────────────────────────────────────

@router.get("/behavioral")
async def behavioral(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    hours: float = Query(24),
):
    """Behavioral patterns — activities, items, and notable patterns."""
    events = await _fetch_descriptions(request, camera_id, start, end, hours)

    activities: Counter = Counter()
    items: Counter = Counter()
    activity_by_hour: dict[int, Counter] = defaultdict(Counter)

    for e in events:
        desc = e.get("description", "")
        hour = _extract_hour(e.get("timestamp", ""))

        for label, pat in ACTIVITY_PATTERNS.items():
            if pat.search(desc):
                activities[label] += 1
                if hour is not None:
                    activity_by_hour[hour][label] += 1

        desc_lower = desc.lower()
        for item in ITEM_PATTERNS:
            if item in desc_lower:
                items[item] += 1

    # Generate notable patterns
    notable_patterns: list[str] = []
    total = len(events) or 1

    # Bag carrying percentage
    bag_count = activities.get("carrying_bag", 0) + activities.get("carrying_briefcase", 0)
    if bag_count > 0:
        pct = round(bag_count / total * 100)
        notable_patterns.append(f"{pct}% of observations involve people carrying bags or briefcases")

    # Walking vs standing ratio
    walking = activities.get("walking", 0)
    standing = activities.get("standing", 0)
    if walking > 0 and standing > 0:
        ratio = round(walking / (walking + standing) * 100)
        notable_patterns.append(f"{ratio}% of people are in motion (walking) vs {100 - ratio}% stationary")

    # Peak activity hour
    if activity_by_hour:
        peak_hour = max(activity_by_hour, key=lambda h: sum(activity_by_hour[h].values()))
        peak_total = sum(activity_by_hour[peak_hour].values())
        notable_patterns.append(f"Peak activity at {peak_hour:02d}:00 with {peak_total} observed behaviors")

    # Parking patterns
    parked = activities.get("parking", 0)
    if parked > 0:
        pct = round(parked / total * 100)
        notable_patterns.append(f"{pct}% of events involve parked vehicles")

    return {
        "total_events_analyzed": len(events),
        "activities": dict(activities.most_common()),
        "items_detected": dict(items.most_common()),
        "notable_patterns": notable_patterns,
        "activity_by_hour": {
            str(h): dict(c.most_common()) for h, c in sorted(activity_by_hour.items())
        },
    }


# ── Executive Summary ─────────────────────────────────────────────────────────

@router.get("/executive-summary")
async def executive_summary(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    hours: float = Query(24),
):
    """AI-generated executive summary using Gemini, grounded in extracted analytics."""
    # Collect all insights
    demo = await demographics(request, camera_id, start, end, hours)
    vehi = await vehicles(request, camera_id, start, end, hours)
    behav = await behavioral(request, camera_id, start, end, hours)

    data_points = demo["total_events_analyzed"]

    # Build structured data for Gemini
    data_summary = (
        f"DATA ANALYZED: {data_points} camera events\n\n"
        f"DEMOGRAPHICS:\n"
        f"  Total people observed: {demo['total_people']}\n"
        f"  Gender: Male={demo['gender']['male']}, Female={demo['gender']['female']}, "
        f"Unknown={demo['gender']['unknown']}\n"
        f"  Age groups: Young={demo['age_groups']['young']}, "
        f"Middle-aged={demo['age_groups']['middle_aged']}, "
        f"Elderly={demo['age_groups']['elderly']}, Unknown={demo['age_groups']['unknown']}\n"
        f"  Ethnicity mentions: {demo['ethnicity']}\n"
        f"  By hour: {demo['demographics_by_hour']}\n\n"
        f"VEHICLES:\n"
        f"  Total vehicle events: {vehi['total_vehicles']}\n"
        f"  By type: {vehi['by_type']}\n"
        f"  By color: {vehi['by_color']}\n"
        f"  Brands: {vehi['brands_detected']}\n\n"
        f"BEHAVIORAL:\n"
        f"  Activities: {behav['activities']}\n"
        f"  Items detected: {behav['items_detected']}\n"
        f"  Notable patterns: {behav['notable_patterns']}\n"
        f"  Activity by hour: {behav['activity_by_hour']}\n"
    )

    prompt = (
        "You are a senior intelligence analyst at a global security firm. "
        "Based on the following observations from security cameras, produce a "
        "5-paragraph executive briefing that would impress a C-suite executive "
        "or hedge fund partner.\n\n"
        "Include:\n"
        "1) Key Findings — the single most important insight\n"
        "2) Demographic Breakdown — who is in the area, with percentages\n"
        "3) Traffic & Movement Patterns — when and how people/vehicles move\n"
        "4) Anomalies & Security Concerns — anything unusual\n"
        "5) Recommendations — specific, actionable next steps\n\n"
        "Be specific with numbers and percentages. Reference time windows. "
        "Write in a confident, authoritative tone.\n\n"
        f"{data_summary}"
    )

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        # Fallback: return structured data without AI summary
        return {
            "summary": _generate_fallback_summary(demo, vehi, behav),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "data_points_analyzed": data_points,
            "model": "fallback-stats",
            "demographics": demo,
            "vehicles": vehi,
            "behavioral": behav,
        }

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.4, "max_output_tokens": 2048},
        )
        summary_text = response.text or "Summary generation failed."
    except Exception as e:
        logger.warning("Gemini executive summary failed: %s", e)
        summary_text = _generate_fallback_summary(demo, vehi, behav)

    return {
        "summary": summary_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_points_analyzed": data_points,
        "model": GEMINI_MODEL,
        "demographics": demo,
        "vehicles": vehi,
        "behavioral": behav,
    }


def _generate_fallback_summary(demo: dict, vehi: dict, behav: dict) -> str:
    """Stats-based summary when Gemini is not available."""
    lines = []
    total = demo["total_events_analyzed"]
    lines.append(f"Analysis of {total} camera events.")

    if demo["total_people"] > 0:
        g = demo["gender"]
        lines.append(
            f"Demographics: {demo['total_people']} people observed — "
            f"{g['male']} male, {g['female']} female, {g['unknown']} unidentified."
        )
        a = demo["age_groups"]
        parts = []
        for k, v in a.items():
            if v > 0 and k != "unknown":
                parts.append(f"{v} {k.replace('_', ' ')}")
        if parts:
            lines.append(f"Age breakdown: {', '.join(parts)}.")

    if vehi["total_vehicles"] > 0:
        lines.append(
            f"Vehicles: {vehi['total_vehicles']} vehicle events. "
            f"Types: {vehi['by_type']}. Top colors: {vehi['by_color']}."
        )

    if behav["notable_patterns"]:
        lines.append("Patterns: " + " | ".join(behav["notable_patterns"]))

    return " ".join(lines)
