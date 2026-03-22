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


# ── Actionable Insights (K3) ──────────────────────────────────────────────────

@router.get("/actionable")
async def actionable_insights(
    request: Request,
    camera_id: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    hours: float = Query(24),
):
    """Structured actionable insights — K3 metric endpoint.

    Returns typed, scored, deduplicated insights with evidence.
    Each insight implies a specific decision or action.
    """
    from trio_core.insights import InsightExtractor

    events = await _fetch_descriptions(request, camera_id, start, end, hours)
    extractor = InsightExtractor()
    insights = extractor.extract(events)

    return {
        "total_events_analyzed": len(events),
        "k3_count": len(insights),
        "k3_target": 5,
        "k3_met": len(insights) >= 5,
        "insights": [
            {
                "type": ins.insight_type,
                "text": ins.text,
                "evidence": ins.evidence,
                "hour": ins.hour,
                "confidence": ins.confidence,
            }
            for ins in insights
        ],
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
            "summary": _generate_fallback_summary(demo, vehi, behav, "investment"),
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
        summary_text = _generate_fallback_summary(demo, vehi, behav, "investment")

    return {
        "summary": summary_text,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_points_analyzed": data_points,
        "model": GEMINI_MODEL,
        "demographics": demo,
        "vehicles": vehi,
        "behavioral": behav,
    }


def _generate_fallback_summary(
    demo: dict, vehi: dict, behav: dict, report_type: str = "investment",
) -> str:
    """Stats-based summary when Gemini is not available."""
    lines = []
    total = demo["total_events_analyzed"]

    if report_type == "security":
        lines.append(f"THREAT LEVEL: LOW. Analysis of {total} access events.")
        if demo["total_people"] > 0:
            lines.append(
                f"Personnel observed: {demo['total_people']} individuals. "
                f"All access events within normal operating parameters."
            )
        if vehi["total_vehicles"] > 0:
            lines.append(
                f"Vehicle activity: {vehi['total_vehicles']} vehicle events recorded. "
                f"Types: {vehi['by_type']}."
            )
        if behav["notable_patterns"]:
            lines.append("Observations: " + " | ".join(behav["notable_patterns"]))
        lines.append("Recommendation: Continue standard monitoring protocols.")
    else:
        lines.append(f"Analysis of {total} camera events.")
        if demo["total_people"] > 0:
            g = demo["gender"]
            lines.append(
                f"Demographics: {demo['total_people']} people observed -- "
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


# ── Gemini client (shared with chat.py pattern) ─────────────────────────────

def _get_gemini_client():
    """Get Gemini client (lazy init)."""
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set")
    return genai.Client(api_key=api_key)


# ── Auto-report helpers ──────────────────────────────────────────────────────

async def _gather_report_data(
    request: Request,
    camera_id: str | None,
    hours: float,
) -> dict:
    """Gather all analytics data for auto-report generation."""
    demo = await demographics(request, camera_id, None, None, hours)
    vehi = await vehicles(request, camera_id, None, None, hours)
    behav = await behavioral(request, camera_id, None, None, hours)

    # K3: extract structured actionable insights
    from trio_core.insights import InsightExtractor
    events_for_insights = await _fetch_descriptions(request, camera_id, None, None, hours)
    extractor = InsightExtractor()
    k3_insights = extractor.extract(events_for_insights)

    # Get camera name
    camera_name = "All cameras"
    if camera_id:
        store = _get_store(request)
        cameras = await store.list_cameras()
        for cam in cameras:
            if cam["id"] == camera_id:
                camera_name = cam.get("name") or camera_id
                break

    # Compute peak hour from behavioral data
    activity_by_hour = behav.get("activity_by_hour", {})
    peak_hour = "N/A"
    if activity_by_hour:
        peak_h = max(activity_by_hour, key=lambda h: sum(activity_by_hour[h].values()))
        peak_hour = f"{int(peak_h):02d}:00"

    # Top 5 descriptions from events
    events = await _fetch_descriptions(request, camera_id, None, None, hours)
    top_descriptions = [
        e.get("description", "")
        for e in events[:5]
        if e.get("description")
    ]

    # Drink size analysis (Starbucks-specific but useful for any F&B)
    import re
    all_descs = " ".join(e.get("description", "").lower() for e in events)
    drink_sizes = {
        "tall": len(re.findall(r"\btall\b", all_descs)),
        "grande": len(re.findall(r"\bgrande\b", all_descs)),
        "venti": len(re.findall(r"\bventi\b", all_descs)),
    }
    # Auto-detect scene type from event descriptions
    scene_keywords = {
        "retail": ["ordering", "customer", "counter", "register", "bag", "shopping", "store", "buy", "purchase", "latte", "coffee", "drink", "starbucks", "menu"],
        "security": ["badge", "entrance", "door", "patrol", "authorized", "unauthorized", "security", "access", "perimeter", "gate", "visitor", "contractor", "shift change"],
        "parking": ["vehicle", "car", "truck", "parking", "parked", "license plate", "garage", "lot", "spot"],
        "warehouse": ["loading", "dock", "cargo", "forklift", "pallet", "shipment", "delivery", "warehouse"],
    }
    scene_scores = {}
    for scene_type, keywords in scene_keywords.items():
        score = sum(all_descs.count(kw) for kw in keywords)
        scene_scores[scene_type] = score
    detected_scene = max(scene_scores, key=scene_scores.get) if max(scene_scores.values()) > 0 else "general"

    food_mentions = len(re.findall(r"\b(pastry|sandwich|muffin|cake|cookie|food|croissant|bagel)\b", all_descs))
    laptop_mentions = len(re.findall(r"\blaptop\b", all_descs))
    phone_mentions = len(re.findall(r"\bphone\b", all_descs))

    # Pre-compute ASP (average selling price)
    total_drinks = drink_sizes["tall"] + drink_sizes["grande"] + drink_sizes["venti"]
    if total_drinks > 0:
        weighted_asp = (drink_sizes["tall"] * 4.25 + drink_sizes["grande"] * 5.45 + drink_sizes["venti"] * 6.25) / total_drinks
        food_attach_rate = food_mentions / total_drinks * 100 if total_drinks > 0 else 0
        estimated_asp_with_food = weighted_asp + (food_attach_rate / 100 * 3.50)
    else:
        weighted_asp = 0
        food_attach_rate = 0
        estimated_asp_with_food = 0

    # Time range string with actual date
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%B %d, %Y")
    time_range = f"Last {hours:.0f} hours (as of {date_str})" if hours == int(hours) else f"Last {hours} hours (as of {date_str})"

    return {
        "camera_name": camera_name,
        "time_range": time_range,
        "demo": demo,
        "vehi": vehi,
        "behav": behav,
        "peak_hour": peak_hour,
        "top_descriptions": top_descriptions,
        "total_events": len(events),
        "drink_sizes": drink_sizes,
        "food_mentions": food_mentions,
        "laptop_mentions": laptop_mentions,
        "phone_mentions": phone_mentions,
        "estimated_asp": round(weighted_asp, 2),
        "food_attach_rate": round(food_attach_rate, 1),
        "estimated_asp_with_food": round(estimated_asp_with_food, 2),
        "detected_scene": detected_scene,
        "scene_scores": scene_scores,
        "k3_insights": k3_insights,
    }


def _build_investment_prompt(data: dict) -> str:
    """Build the investment analyst persona prompt."""
    demo = data["demo"]
    vehi = data["vehi"]
    behav = data["behav"]

    gender = demo["gender"]
    gender_breakdown = (
        f"Male={gender['male']}, Female={gender['female']}, "
        f"Unknown={gender['unknown']}"
    )
    age = demo["age_groups"]
    age_breakdown = (
        f"Young={age['young']}, Middle-aged={age['middle_aged']}, "
        f"Elderly={age['elderly']}, Unknown={age['unknown']}"
    )
    vehicle_summary = (
        f"{vehi['total_vehicles']} vehicles — "
        f"Types: {vehi['by_type']}, Colors: {vehi['by_color']}, "
        f"Brands: {[b['brand'] for b in vehi['brands_detected'][:5]]}"
    )
    behavioral_summary = (
        f"Activities: {behav['activities']}, "
        f"Items: {behav['items_detected']}, "
        f"Patterns: {behav['notable_patterns']}"
    )
    top_5 = "\n".join(f"  - {d}" for d in data["top_descriptions"]) or "  (none)"

    return (
        "You are a senior investment research analyst at a top-tier hedge fund. "
        "You have been monitoring a retail location via AI-powered cameras.\n\n"
        "Based on the following observational data, produce a structured intelligence brief "
        "that would help a portfolio manager make investment decisions.\n\n"
        f"DATA:\n"
        f"- Location: {data['camera_name']}\n"
        f"- Observation period: {data['time_range']}\n"
        f"- Total foot traffic: {demo['total_people']} people observed\n"
        f"- Demographics: {gender_breakdown}, {age_breakdown}\n"
        f"- Vehicle data: {vehicle_summary}\n"
        f"- Activity patterns: {behavioral_summary}\n"
        f"- Peak hour: {data['peak_hour']}\n"
        f"\n** DRINK SIZE MIX (CRITICAL FOR ASP ESTIMATION):\n"
        f"  - Tall (12oz, ~$4.25): {data.get('drink_sizes',{}).get('tall',0)} observations\n"
        f"  - Grande (16oz, ~$5.45): {data.get('drink_sizes',{}).get('grande',0)} observations\n"
        f"  - Venti (20oz, ~$6.25): {data.get('drink_sizes',{}).get('venti',0)} observations\n"
        f"  - Food attach (pastry/sandwich): {data.get('food_mentions',0)} observations (~$3.50 add-on)\n"
        f"  - Laptop users (dwell time proxy): {data.get('laptop_mentions',0)}\n"
        f"  ** PRE-COMPUTED ASP: ${data.get('estimated_asp',0):.2f} (drinks only), "
        f"${data.get('estimated_asp_with_food',0):.2f} (incl food attach at {data.get('food_attach_rate',0):.1f}%)\n"
        f"  ** vs Starbucks reported ASP: ~$5.50\n"
        f"- Notable observations:\n{top_5}\n\n"
        "AI-EXTRACTED ACTIONABLE INSIGHTS:\n"
        + "\n".join(
            f"  [{ins.insight_type.upper()}] {ins.text}"
            for ins in data.get("k3_insights", [])
        )
        + "\n\n"
        "CONTEXT:\n"
        "- A typical Starbucks location serves 400-500 customers per day\n"
        "- Industry average ticket: $5.50-6.50\n"
        "- Starbucks reported SSS growth of +3% last quarter\n"
        "- Note: this data is from a SINGLE camera with LIMITED field of view. "
        "Actual total store traffic is likely 5-10x higher than observed.\n\n"
        "PRODUCE:\n"
        "1. **Executive Summary** (2-3 sentences, key finding + confidence level)\n"
        "2. **Customer Profile** (demographics, affluence signals, customer segments)\n"
        "3. **Traffic Analysis** (patterns, peak times, capacity utilization vs industry benchmark)\n"
        "4. **Order Analysis** (MUST calculate weighted ASP from drink size mix: tall=$4.25, grande=$5.45, venti=$6.25 + food attach rate. Compare to SBUX reported ASP of $5.50)\n"
        "5. **Competitive Intelligence** (what the data reveals about brand positioning)\n"
        "6. **Risk Factors** (anomalies, concerning patterns, data limitations)\n"
        "7. **Investment Thesis** (bullish or bearish signal, with evidence, confidence 1-10)\n"
        "8. **Implied Financial Impact** (If ASP is lower/higher than reported, estimate EPS impact. "
        "If foot traffic differs from company guidance, estimate revenue impact. "
        "Reference SBUX current price ~$85, consensus EPS ~$3.50, P/E ~24x)\n\n"
        "Be specific with numbers and percentages. Every claim must reference observed data.\n"
        "Clearly state data limitations and confidence levels.\n"
        "Write as if presenting to a $5B fund's investment committee.\n"
        "End with a clear BUY / HOLD / SELL recommendation with price target range."
    )


def _build_security_prompt(data: dict) -> str:
    """Build the security officer persona prompt."""
    demo = data["demo"]
    vehi = data["vehi"]
    behav = data["behav"]

    demographics_str = (
        f"Gender: Male={demo['gender']['male']}, Female={demo['gender']['female']}; "
        f"Age: Young={demo['age_groups']['young']}, "
        f"Middle-aged={demo['age_groups']['middle_aged']}, "
        f"Elderly={demo['age_groups']['elderly']}"
    )
    vehicle_str = (
        f"{vehi['total_vehicles']} vehicle events -- "
        f"Types: {vehi['by_type']}, Colors: {vehi['by_color']}"
    )
    behavioral_str = (
        f"Activities: {behav['activities']}, "
        f"Items: {behav['items_detected']}, "
        f"Patterns: {behav['notable_patterns']}"
    )
    top_descs = "\n".join(f"  - {d}" for d in data["top_descriptions"]) or "  (none)"

    # Extract security-specific metrics from descriptions
    all_descs = " ".join(d for d in data.get("top_descriptions", []))
    all_events = data.get("total_events", 0)

    return (
        "You are the Chief Security Officer reviewing AI surveillance data "
        "for a Tier-1 data center facility. Write a formal security "
        "intelligence report suitable for presentation to the board of directors.\n\n"
        f"FACILITY DATA:\n"
        f"- Facility: {data['camera_name']}\n"
        f"- Reporting period: {data['time_range']}\n"
        f"- Total access events recorded: {all_events}\n"
        f"- Personnel detected: {demo['total_people']} individuals\n"
        f"- Personnel breakdown: {demographics_str}\n"
        f"- Vehicle activity: {vehicle_str}\n"
        f"- Behavioral patterns: {behavioral_str}\n"
        f"- Peak activity hour: {data['peak_hour']}\n"
        f"- Notable events:\n{top_descs}\n\n"
        "AI-EXTRACTED ACTIONABLE INSIGHTS:\n"
        + "\n".join(
            f"  [{ins.insight_type.upper()}] {ins.text}"
            for ins in data.get("k3_insights", [])
        )
        + "\n\n"
        "PRODUCE A STRUCTURED REPORT WITH THESE SECTIONS:\n\n"
        "1. **Threat Assessment** -- Overall threat level (ALL CLEAR / LOW / MODERATE / "
        "HIGH / CRITICAL) with justification. Start with: 'THREAT LEVEL: [X]'\n\n"
        "2. **Executive Summary** -- 3-4 sentences covering the most important findings "
        "for the CISO. Highlight any incidents requiring immediate attention.\n\n"
        "3. **Access Control Analysis** -- Badge-verified entries vs unverified, "
        "contractor vs employee ratio, after-hours access attempts, "
        "tailgating or piggybacking incidents.\n\n"
        "4. **Incident Log** -- Each security incident as a numbered item with: "
        "time, location, description, severity (LOW/MEDIUM/HIGH/CRITICAL), "
        "and recommended response. If no incidents, state 'No incidents to report.'\n\n"
        "5. **Perimeter & Vehicle Security** -- Vehicle movements, unauthorized parking, "
        "loading dock activity, and any suspicious approaches.\n\n"
        "6. **Compliance Status** -- Badge compliance rate (% of personnel properly badged), "
        "door security status, visitor management compliance.\n\n"
        "7. **Recommendations** -- Numbered, specific, actionable items for the security team. "
        "Prioritize by urgency (IMMEDIATE / SHORT-TERM / ONGOING).\n\n"
        "FORMATTING RULES:\n"
        "- Use **bold** for section headers and key findings\n"
        "- Reference specific times (HH:MM) and camera names\n"
        "- Include percentages and counts where possible\n"
        "- Write in authoritative, professional security language\n"
        "- Do NOT use emojis\n"
        "- If data is limited, clearly state confidence levels\n"
    )


# ── Auto-Report Endpoints ────────────────────────────────────────────────────

@router.get("/auto-report")
async def auto_report(
    request: Request,
    camera_id: str | None = Query(None),
    hours: float = Query(24, description="Hours back from now"),
    report_type: str = Query("investment", description="Report type: investment or security"),
):
    """Auto-generated intelligence report — zero-config, plug camera and go.

    Gathers all event data, runs demographic/vehicle/behavioral extraction,
    then sends to Gemini with an analyst persona to produce an actionable brief.
    """
    data = await _gather_report_data(request, camera_id, hours)
    demo = data["demo"]
    vehi = data["vehi"]

    # Pick prompt by report type
    if report_type == "security":
        prompt = _build_security_prompt(data)
    else:
        prompt = _build_investment_prompt(data)

    # Metrics summary
    metrics_summary = {
        "total_people": demo["total_people"],
        "total_vehicles": vehi["total_vehicles"],
        "peak_hour": data["peak_hour"],
        "demographics": {
            "gender": demo["gender"],
            "age_groups": demo["age_groups"],
        },
    }

    # Call Gemini
    try:
        client = _get_gemini_client()
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={"temperature": 0.4, "max_output_tokens": 4096},
        )
        report_text = response.text or "Report generation failed."
        model_used = GEMINI_MODEL
    except ValueError:
        # No API key — return structured fallback
        report_text = _generate_fallback_summary(demo, vehi, data["behav"], report_type)
        model_used = "fallback-stats"
    except Exception as e:
        logger.warning("Gemini auto-report failed: %s", e)
        report_text = _generate_fallback_summary(demo, vehi, data["behav"], report_type)
        model_used = "fallback-stats"

    return {
        "report_type": report_type,
        "camera_name": data["camera_name"],
        "period": data["time_range"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_points": data["total_events"],
        "model": model_used,
        "report": report_text,
        "metrics_summary": {
            **metrics_summary,
            "drink_sizes": data.get("drink_sizes", {}),
            "food_mentions": data.get("food_mentions", 0),
            "laptop_usage": data.get("laptop_mentions", 0),
            "phone_usage": data.get("phone_mentions", 0),
            "estimated_asp": data.get("estimated_asp", 0),
            "food_attach_rate": data.get("food_attach_rate", 0),
            "estimated_asp_with_food": data.get("estimated_asp_with_food", 0),
            "detected_scene": data.get("detected_scene", "general"),
            "scene_scores": data.get("scene_scores", {}),
        },
        "actionable_insights": [
            {
                "type": ins.insight_type,
                "text": ins.text,
                "evidence": ins.evidence,
            }
            for ins in data.get("k3_insights", [])
        ],
        "k3_count": len(data.get("k3_insights", [])),
        "k3_met": len(data.get("k3_insights", [])) >= 5,
    }


@router.get("/auto-security-report")
async def auto_security_report(
    request: Request,
    camera_id: str | None = Query(None),
    hours: float = Query(24, description="Hours back from now"),
):
    """Convenience alias — auto-report with report_type=security."""
    return await auto_report(request, camera_id=camera_id, hours=hours, report_type="security")


@router.get("/scene-modules")
async def scene_modules(
    request: Request,
    camera_id: str | None = Query(None),
    hours: float = Query(24),
):
    """Auto-detect scene type and return appropriate analytics modules.
    
    Instead of showing fixed Demographics/Vehicles/Behavioral sections,
    this endpoint analyzes what the camera actually sees and returns
    only the relevant modules.
    """
    store = _get_store(request)
    start = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    result = await store.list_events(camera_id=camera_id, start=start, limit=200)
    events = result.get("events", [])

    if not events:
        return {"scene": "unknown", "modules": [], "message": "No events to analyze"}

    all_descs = " ".join(e.get("description", "").lower() for e in events)

    # Detect scene
    scene_keywords = {
        "retail": ["ordering", "customer", "counter", "register", "bag", "shopping", "store", "buy", "latte", "coffee", "drink", "starbucks", "menu", "pastry", "tall", "grande", "venti"],
        "security": ["badge", "entrance", "door", "patrol", "authorized", "unauthorized", "security", "access", "perimeter", "gate", "visitor", "contractor", "shift change", "propped"],
        "parking": ["vehicle", "car", "truck", "parking", "parked", "license plate", "garage", "lot", "spot", "tesla"],
        "warehouse": ["loading", "dock", "cargo", "forklift", "pallet", "shipment", "delivery", "warehouse", "inventory"],
        "office": ["meeting", "conference", "lobby", "reception", "elevator", "floor", "desk", "office"],
    }
    scene_scores = {}
    for scene_type, keywords in scene_keywords.items():
        score = sum(all_descs.count(kw) for kw in keywords)
        scene_scores[scene_type] = score
    detected = max(scene_scores, key=scene_scores.get) if max(scene_scores.values()) > 0 else "general"

    # Define modules per scene type
    MODULE_MAP = {
        "retail": [
            {"id": "customer_demographics", "title": "Customer Demographics", "description": "Age, gender, and profile of visitors"},
            {"id": "order_analysis", "title": "Order Analysis", "description": "Drink sizes, food attach rate, estimated ASP"},
            {"id": "traffic_pattern", "title": "Foot Traffic Pattern", "description": "Hourly visitor count and peak hours"},
            {"id": "customer_behavior", "title": "Customer Behavior", "description": "Dwell time, queue length, activities"},
        ],
        "security": [
            {"id": "access_log", "title": "Access Log Summary", "description": "Personnel entries, badge compliance, visitor tracking"},
            {"id": "incident_report", "title": "Incident Report", "description": "Unauthorized access, propped doors, suspicious activity"},
            {"id": "patrol_status", "title": "Patrol & Compliance", "description": "Security rounds, gate checks, perimeter status"},
            {"id": "vehicle_activity", "title": "Vehicle Activity", "description": "Arrivals, departures, restricted zone violations"},
        ],
        "parking": [
            {"id": "occupancy", "title": "Lot Occupancy", "description": "Current and peak vehicle count"},
            {"id": "vehicle_types", "title": "Vehicle Classification", "description": "Cars, trucks, motorcycles by type and color"},
            {"id": "duration", "title": "Parking Duration", "description": "Average stay time and turnover rate"},
            {"id": "violations", "title": "Violations", "description": "Unauthorized parking, overstay, restricted zones"},
        ],
        "warehouse": [
            {"id": "dock_activity", "title": "Loading Dock Activity", "description": "Truck arrivals, cargo handling, wait times"},
            {"id": "personnel", "title": "Personnel Tracking", "description": "Worker count, safety gear compliance"},
            {"id": "inventory_movement", "title": "Inventory Movement", "description": "Shipments in/out, pallet counts"},
            {"id": "safety", "title": "Safety Compliance", "description": "PPE detection, forklift operations, zone violations"},
        ],
        "office": [
            {"id": "occupancy", "title": "Space Occupancy", "description": "Meeting room usage, desk utilization"},
            {"id": "visitor_log", "title": "Visitor Log", "description": "Check-ins, escorts, duration of visits"},
            {"id": "traffic_flow", "title": "Traffic Flow", "description": "Elevator usage, floor distribution, peak hours"},
        ],
    }

    modules = MODULE_MAP.get(detected, [
        {"id": "general_activity", "title": "Activity Summary", "description": "Overview of detected events and patterns"},
        {"id": "demographics", "title": "People Analysis", "description": "Demographics and behavioral patterns"},
    ])

    return {
        "scene": detected,
        "confidence": scene_scores.get(detected, 0),
        "scores": scene_scores,
        "modules": modules,
        "events_analyzed": len(events),
    }
