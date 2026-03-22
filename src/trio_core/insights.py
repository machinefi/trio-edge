"""Structured insight extraction from camera event descriptions.

Extracts typed, scored, actionable insights from VLM event descriptions.
Each insight has: type, text, evidence, score.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass
class Insight:
    """A single actionable insight extracted from event data."""
    insight_type: str   # anomaly, trend, pattern, correlation, recommendation, comparison
    text: str           # human-readable insight
    evidence: list[str] = field(default_factory=list)  # supporting data points
    hour: int | None = None  # relevant hour if time-specific
    confidence: float = 1.0


class InsightExtractor:
    """Extract structured, actionable insights from event descriptions."""

    # Regex patterns (reused from insights.py)
    ACTIVITY_PATTERNS = {
        "walking": re.compile(r"\b(walk(?:ing|s)?|crossing|strolling|briskly)\b", re.I),
        "standing": re.compile(r"\b(stand(?:ing|s)?|waiting|stationary|idling)\b", re.I),
        "running": re.compile(r"\b(run(?:ning|s)?|jogging|sprinting)\b", re.I),
        "sitting": re.compile(r"\b(sit(?:ting|s)?|seated)\b", re.I),
        "carrying_bag": re.compile(r"\b(bag|backpack|tote|suitcase|carrying.*bag|laptop bag)\b", re.I),
        "carrying_briefcase": re.compile(r"\b(briefcase|attache)\b", re.I),
        "driving": re.compile(r"\b(driv(?:ing|es?)|pulling in)\b", re.I),
        "parking": re.compile(r"\b(park(?:ed|ing)|parked)\b", re.I),
        "on_phone": re.compile(r"\b(phone|calling|talking on)\b", re.I),
        "with_laptop": re.compile(r"\blaptop\b", re.I),
        "group": re.compile(r"\b(group|together|couple|pair|two|three|four|family)\b", re.I),
        "ordering": re.compile(r"\b(order(?:ing|s|ed)?|buying|purchasing|paying|mobile order|queue|in line|waiting in)\b", re.I),
        "browsing": re.compile(r"\b(brows(?:ing|es)?|looking at|checking out|display|menu)\b", re.I),
    }

    GENDER_PATTERNS = {
        "male": re.compile(r"\b(male|man|boy|gentleman|his\b)", re.I),
        "female": re.compile(r"\b(female|woman|girl|lady|her\b)", re.I),
    }

    AGE_PATTERNS = {
        "young": re.compile(r"\b(young|youth|teen|child|kid|20s|young adult|teenager)\b", re.I),
        "middle_aged": re.compile(r"\b(middle.?aged?|30s|40s|50s|mid-\d0s)\b", re.I),
        "elderly": re.compile(r"\b(elderly|older|senior|aged|walking stick|wheelchair)\b", re.I),
    }

    ATTIRE_PATTERNS = {
        "business": re.compile(r"\b(suit|business|blazer|tie|professional|attire|polo)\b", re.I),
        "casual": re.compile(r"\b(hoodie|jeans|t-shirt|casual|sneakers|athletic)\b", re.I),
        "uniform": re.compile(r"\b(uniform|vest|scrubs|chef|guard|security|construction|maintenance|hard hat)\b", re.I),
    }

    VEHICLE_PATTERNS = {
        "car": re.compile(r"\b(car|sedan|coupe|civic|camry|tesla|bmw|honda|ford|toyota)\b", re.I),
        "suv": re.compile(r"\b(suv|x5|rav4)\b", re.I),
        "truck": re.compile(r"\b(truck|pickup|f-150)\b", re.I),
        "delivery": re.compile(r"\b(ups|fedex|delivery|dolly|packages)\b", re.I),
    }

    DRINK_PATTERNS = {
        "tall": re.compile(r"\btall\b", re.I),
        "grande": re.compile(r"\bgrande\b", re.I),
        "venti": re.compile(r"\bventi\b", re.I),
        "coffee": re.compile(r"\b(coffee|latte|cappuccino|frappuccino|espresso|cup)\b", re.I),
    }

    SECURITY_PATTERNS = {
        "badge": re.compile(r"\b(badge|lanyard|id card|credential|badged)\b", re.I),
        "suspicious": re.compile(r"\b(suspicious|looking around|loitering|dark clothing|perimeter|lingering|unidentified|unknown)\b", re.I),
        "locked": re.compile(r"\b(lock(?:ing|ed)|securing|checking.*door)\b", re.I),
        "unauthorized": re.compile(r"\b(unauthorized|tailgat|piggyback|propped|without badge|no badge|expired|unaccompanied|unescorted)\b", re.I),
        "after_hours": re.compile(r"\b(after.?hours?|night|02:|03:|04:|05:|01:|00:|late night)\b", re.I),
        "patrol": re.compile(r"\b(patrol|rounds|guard station|shift change|escort)\b", re.I),
        "alarm": re.compile(r"\b(alarm|emergency|triggered|blind spot|unattended)\b", re.I),
    }

    def extract(self, events: list[dict], scene_type: str | None = None) -> list[Insight]:
        """Extract all insights from event descriptions.

        Args:
            events: list of event dicts with 'description' and 'timestamp'
            scene_type: optional scene hint ('retail', 'security', etc.)
                        prevents cross-contamination (e.g. badge insights on retail)
        """
        if not events:
            return []

        insights: list[Insight] = []

        # Parse all events into structured data
        parsed = self._parse_events(events)

        # Auto-detect scene if not provided
        if scene_type is None:
            scene_type = self._detect_scene(parsed)

        # Extract insights from different dimensions
        insights.extend(self._traffic_insights(parsed, events))
        insights.extend(self._demographic_insights(parsed, events))
        insights.extend(self._behavioral_insights(parsed, events))
        if scene_type in ("security", None):
            insights.extend(self._security_insights(parsed, events))
        insights.extend(self._vehicle_insights(parsed, events))
        insights.extend(self._temporal_insights(parsed, events))
        if scene_type in ("retail", "restaurant", None):
            insights.extend(self._retail_insights(parsed, events))

        # Deduplicate by checking text similarity
        insights = self._deduplicate(insights)

        return insights

    def _parse_events(self, events: list[dict]) -> dict:
        """Parse all events into structured counters."""
        data = {
            "activities": Counter(),
            "activity_by_hour": defaultdict(Counter),
            "gender": Counter(),
            "age": Counter(),
            "attire": Counter(),
            "vehicles": Counter(),
            "drinks": Counter(),
            "security": Counter(),
            "events_by_hour": Counter(),
            "person_events": 0,
            "vehicle_events": 0,
            "group_events": 0,
            "total": len(events),
        }

        for e in events:
            desc = e.get("description", "")
            hour = self._extract_hour(e.get("timestamp", ""))

            if hour is not None:
                data["events_by_hour"][hour] += 1

            # Activities
            for label, pat in self.ACTIVITY_PATTERNS.items():
                if pat.search(desc):
                    data["activities"][label] += 1
                    if hour is not None:
                        data["activity_by_hour"][hour][label] += 1

            # Demographics
            is_person = False
            for label, pat in self.GENDER_PATTERNS.items():
                if pat.search(desc):
                    data["gender"][label] += 1
                    is_person = True
            for label, pat in self.AGE_PATTERNS.items():
                if pat.search(desc):
                    data["age"][label] += 1
                    is_person = True
            if is_person:
                data["person_events"] += 1

            # Attire
            for label, pat in self.ATTIRE_PATTERNS.items():
                if pat.search(desc):
                    data["attire"][label] += 1

            # Vehicles
            for label, pat in self.VEHICLE_PATTERNS.items():
                if pat.search(desc):
                    data["vehicles"][label] += 1
                    data["vehicle_events"] += 1

            # Drinks
            for label, pat in self.DRINK_PATTERNS.items():
                if pat.search(desc):
                    data["drinks"][label] += 1

            # Security
            for label, pat in self.SECURITY_PATTERNS.items():
                if pat.search(desc):
                    data["security"][label] += 1

            # Groups
            if self.ACTIVITY_PATTERNS["group"].search(desc):
                data["group_events"] += 1

        return data

    def _extract_hour(self, ts: str) -> int | None:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00")).hour
        except (ValueError, AttributeError):
            return None

    def _detect_scene(self, p: dict) -> str:
        """Auto-detect scene type from parsed data to avoid cross-contamination."""
        sec = p["security"]
        drinks = p["drinks"]
        security_signals = sec.get("badge", 0) + sec.get("patrol", 0) + sec.get("unauthorized", 0) + sec.get("alarm", 0)
        retail_signals = drinks.get("coffee", 0) + drinks.get("tall", 0) + drinks.get("grande", 0) + drinks.get("venti", 0) + p["activities"].get("ordering", 0)

        if retail_signals > security_signals * 2:
            return "retail"
        if security_signals > retail_signals * 2:
            return "security"
        return "general"

    # ── Insight generators ────────────────────────────────────────────────

    def _traffic_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Traffic volume and flow insights."""
        insights = []
        total = p["total"]
        by_hour = p["events_by_hour"]

        if not by_hour:
            return insights

        # Peak hour
        peak_h = max(by_hour, key=by_hour.get)
        peak_count = by_hour[peak_h]
        avg_hourly = total / max(len(by_hour), 1)

        if peak_count > avg_hourly * 1.5:
            insights.append(Insight(
                insight_type="pattern",
                text=f"Peak traffic at {peak_h:02d}:00 ({peak_count} events) is {peak_count/avg_hourly:.1f}x the hourly average — consider increasing staffing during this window",
                evidence=[f"peak={peak_count}", f"avg={avg_hourly:.0f}"],
                hour=peak_h,
            ))

        # Quiet hours
        min_h = min(by_hour, key=by_hour.get)
        min_count = by_hour[min_h]
        if min_count < avg_hourly * 0.3 and len(by_hour) > 3:
            insights.append(Insight(
                insight_type="recommendation",
                text=f"Low activity at {min_h:02d}:00 ({min_count} events, {min_count/avg_hourly:.0%} of average) — opportunity to reduce staffing or schedule maintenance",
                evidence=[f"min={min_count}", f"avg={avg_hourly:.0f}"],
                hour=min_h,
            ))

        # Morning vs afternoon split
        morning = sum(by_hour.get(h, 0) for h in range(6, 12))
        afternoon = sum(by_hour.get(h, 0) for h in range(12, 18))
        if morning > 0 and afternoon > 0:
            ratio = morning / afternoon
            if ratio > 1.5:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"Morning traffic ({morning} events) is {ratio:.1f}x afternoon ({afternoon} events) — suggests business/commuter-dominant location",
                    evidence=[f"morning={morning}", f"afternoon={afternoon}"],
                ))
            elif ratio < 0.67:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"Afternoon traffic ({afternoon} events) is {1/ratio:.1f}x morning ({morning} events) — suggests retail/leisure-dominant location",
                    evidence=[f"morning={morning}", f"afternoon={afternoon}"],
                ))

        return insights

    def _demographic_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Who is visiting — demographics, attire, group composition."""
        insights = []
        total_people = p["person_events"] or 1

        # Gender split
        male = p["gender"].get("male", 0)
        female = p["gender"].get("female", 0)
        if male + female > 3:
            dominant = "male" if male > female else "female"
            pct = max(male, female) / (male + female) * 100
            if pct > 65:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"{pct:.0f}% {dominant} visitors ({max(male,female)}/{male+female}) — indicates {dominant}-skewed customer base, consider targeted marketing",
                    evidence=[f"male={male}", f"female={female}"],
                ))

        # Age distribution
        young = p["age"].get("young", 0)
        middle = p["age"].get("middle_aged", 0)
        elderly = p["age"].get("elderly", 0)
        total_age = young + middle + elderly
        if total_age > 3:
            dominant_age = max([(young, "young"), (middle, "middle-aged"), (elderly, "elderly")], key=lambda x: x[0])
            pct = dominant_age[0] / total_age * 100
            if pct > 50:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"Dominant age group: {dominant_age[1]} ({pct:.0f}% of identified visitors) — suggests tailoring product mix and pricing to this demographic",
                    evidence=[f"young={young}", f"middle_aged={middle}", f"elderly={elderly}"],
                ))

        # Attire signals (affluence/purpose)
        business = p["attire"].get("business", 0)
        casual = p["attire"].get("casual", 0)
        uniform = p["attire"].get("uniform", 0)
        total_attire = business + casual + uniform
        if total_attire > 3:
            if business > casual and business > uniform:
                pct = business / total_attire * 100
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"{pct:.0f}% business attire ({business}/{total_attire} identified) indicates professional/high-income visitor profile — opportunity for premium offerings",
                    evidence=[f"business={business}", f"casual={casual}", f"uniform={uniform}"],
                ))
            elif uniform > 3:
                pct = uniform / total_attire * 100
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"{pct:.0f}% in work uniforms ({uniform}/{total_attire}) — indicates service/construction worker traffic, consider value-oriented offerings",
                    evidence=[f"uniform={uniform}"],
                ))

        # Group behavior
        group_pct = p["group_events"] / total_people * 100 if total_people > 0 else 0
        if p["group_events"] > 2 and group_pct > 15:
            insights.append(Insight(
                insight_type="pattern",
                text=f"{group_pct:.0f}% of visitors arrive in groups ({p['group_events']} group events) — indicates social destination; optimize for group seating and multi-item orders",
                evidence=[f"groups={p['group_events']}"],
            ))

        return insights

    def _behavioral_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Activity patterns and behavioral signals."""
        insights = []
        total = p["total"] or 1

        # Motion ratio
        walking = p["activities"].get("walking", 0)
        standing = p["activities"].get("standing", 0)
        sitting = p["activities"].get("sitting", 0)
        stationary = standing + sitting

        if walking + stationary > 4:
            motion_ratio = walking / (walking + stationary) * 100
            if motion_ratio > 75:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"{motion_ratio:.0f}% of people are in transit ({walking} walking vs {stationary} stationary) — location is a pass-through, not a destination. Consider grab-and-go format",
                    evidence=[f"walking={walking}", f"stationary={stationary}"],
                ))
            elif motion_ratio < 40:
                insights.append(Insight(
                    insight_type="pattern",
                    text=f"{100-motion_ratio:.0f}% stationary ({stationary} standing/sitting vs {walking} walking) — location is a destination. Optimize for dwell time and upselling",
                    evidence=[f"walking={walking}", f"stationary={stationary}"],
                ))

        # Bag carrying (affluence signal)
        bags = p["activities"].get("carrying_bag", 0)
        if bags > 1:
            pct = bags / total * 100
            insights.append(Insight(
                insight_type="pattern",
                text=f"{pct:.0f}% carrying bags/backpacks ({bags}/{total} events) — high bag rate suggests shopping activity or commuter traffic",
                evidence=[f"bags={bags}"],
            ))

        # Phone usage
        phone = p["activities"].get("on_phone", 0)
        if phone > 1:
            pct = phone / total * 100
            insights.append(Insight(
                insight_type="pattern",
                text=f"{pct:.0f}% using phones ({phone} observations) — high mobile engagement, consider mobile ordering/app promotions",
                evidence=[f"phone={phone}"],
            ))

        return insights

    def _security_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Security-relevant insights."""
        insights = []
        sec = p["security"]

        suspicious = sec.get("suspicious", 0)
        unauthorized = sec.get("unauthorized", 0)
        badge_count = sec.get("badge", 0)
        patrol = sec.get("patrol", 0)
        alarm = sec.get("alarm", 0)
        after_hours = sec.get("after_hours", 0)

        # Suspicious activity
        if suspicious > 0:
            insights.append(Insight(
                insight_type="anomaly",
                text=f"{suspicious} suspicious activity event(s) detected — recommend reviewing footage and increasing patrol frequency during affected hours",
                evidence=[f"suspicious={suspicious}"],
                confidence=0.7,
            ))

        # Unauthorized access
        if unauthorized > 0:
            insights.append(Insight(
                insight_type="anomaly",
                text=f"{unauthorized} unauthorized access event(s) (tailgating, propped doors, expired badges, unescorted visitors) — immediate security review recommended",
                evidence=[f"unauthorized={unauthorized}"],
                confidence=0.9,
            ))

        # Badge compliance
        if badge_count > 0 and p["person_events"] > 0:
            badge_rate = badge_count / p["person_events"] * 100
            if badge_rate < 80:
                insights.append(Insight(
                    insight_type="recommendation",
                    text=f"Badge compliance rate {badge_rate:.0f}% ({badge_count}/{p['person_events']} people) — below 80% target, recommend enforcing badge-visible policy",
                    evidence=[f"badge={badge_count}", f"people={p['person_events']}"],
                ))

        # Alarm/emergency events
        if alarm > 0:
            insights.append(Insight(
                insight_type="anomaly",
                text=f"{alarm} alarm/emergency event(s) detected (triggered alarms, unattended stations, blind spots) — investigate and address security gaps",
                evidence=[f"alarms={alarm}"],
                confidence=0.8,
            ))

        # Patrol coverage
        if patrol > 0:
            insights.append(Insight(
                insight_type="pattern",
                text=f"{patrol} patrol/escort event(s) recorded — indicates active security operations; verify patrol schedule adherence and coverage gaps",
                evidence=[f"patrols={patrol}"],
            ))

        # After-hours activity
        if after_hours > 0:
            insights.append(Insight(
                insight_type="anomaly",
                text=f"{after_hours} after-hours access event(s) — review against authorized after-hours personnel list for compliance",
                evidence=[f"after_hours={after_hours}"],
                confidence=0.7,
            ))

        return insights

    def _vehicle_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Vehicle-related insights."""
        insights = []

        if p["vehicle_events"] < 2:
            return insights

        total = p["total"] or 1
        vehicle_pct = p["vehicle_events"] / total * 100

        # Vehicle traffic share
        if p["vehicle_events"] >= 3 and vehicle_pct > 15:
            insights.append(Insight(
                insight_type="pattern",
                text=f"{vehicle_pct:.0f}% vehicle traffic ({p['vehicle_events']}/{total} events) — consider optimizing parking layout and traffic flow for this volume",
                evidence=[f"vehicles={p['vehicle_events']}", f"total={total}"],
            ))

        delivery = p["vehicles"].get("delivery", 0)
        if delivery > 0:
            insights.append(Insight(
                insight_type="pattern",
                text=f"{delivery} delivery vehicle event(s) observed — consider designating delivery windows to reduce congestion during peak hours",
                evidence=[f"delivery={delivery}"],
            ))

        # Vehicle mix
        cars = p["vehicles"].get("car", 0) + p["vehicles"].get("suv", 0)
        trucks = p["vehicles"].get("truck", 0)
        if cars + trucks > 5 and trucks > cars:
            insights.append(Insight(
                insight_type="pattern",
                text=f"Truck-dominant vehicle mix ({trucks} trucks vs {cars} cars) — suggests service/construction area rather than retail",
                evidence=[f"trucks={trucks}", f"cars={cars}"],
            ))

        return insights

    def _temporal_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Time-based pattern insights."""
        insights = []
        by_hour = p["events_by_hour"]

        if len(by_hour) < 4:
            return insights

        # Detect activity gaps (no events for 2+ consecutive hours during business hours)
        business_hours = range(8, 20)
        gap_start = None
        for h in business_hours:
            if by_hour.get(h, 0) == 0:
                if gap_start is None:
                    gap_start = h
            else:
                if gap_start is not None and h - gap_start >= 2:
                    insights.append(Insight(
                        insight_type="anomaly",
                        text=f"No activity detected {gap_start:02d}:00-{h:02d}:00 ({h-gap_start} hours) during business hours — investigate if camera issue or genuine low traffic",
                        evidence=[f"gap_hours={h-gap_start}"],
                        hour=gap_start,
                    ))
                gap_start = None

        # Late-night activity
        late_events = sum(by_hour.get(h, 0) for h in range(22, 24)) + sum(by_hour.get(h, 0) for h in range(0, 6))
        if late_events > 3:
            insights.append(Insight(
                insight_type="anomaly",
                text=f"{late_events} events detected during off-hours (22:00-06:00) — review for security concerns or after-hours operations",
                evidence=[f"late_events={late_events}"],
            ))

        return insights

    def _retail_insights(self, p: dict, events: list[dict]) -> list[Insight]:
        """Retail/F&B specific insights (drink sizes, food attach, ASP)."""
        insights = []
        drinks = p["drinks"]

        tall = drinks.get("tall", 0)
        grande = drinks.get("grande", 0)
        venti = drinks.get("venti", 0)
        coffee = drinks.get("coffee", 0)
        total_sized = tall + grande + venti

        if total_sized > 3:
            asp = (tall * 4.25 + grande * 5.45 + venti * 6.25) / total_sized
            insights.append(Insight(
                insight_type="comparison",
                text=f"Observed drink ASP ${asp:.2f} (tall={tall}, grande={grande}, venti={venti}) vs SBUX reported ~$5.50 — {'bullish' if asp > 5.50 else 'bearish'} signal for average ticket",
                evidence=[f"tall={tall}", f"grande={grande}", f"venti={venti}", f"asp=${asp:.2f}"],
            ))

        if coffee > 3 and p["activities"].get("with_laptop", 0) > 0:
            laptop = p["activities"]["with_laptop"]
            insights.append(Insight(
                insight_type="correlation",
                text=f"{laptop} laptop users observed among {coffee} coffee drinkers — indicates work-from-cafe segment with longer dwell time and higher spend potential",
                evidence=[f"laptops={laptop}", f"coffee={coffee}"],
            ))

        # Queue / ordering behavior
        ordering = p["activities"].get("ordering", 0)
        browsing = p["activities"].get("browsing", 0)
        if ordering > 2:
            total = p["total"] or 1
            order_rate = ordering / total * 100
            insights.append(Insight(
                insight_type="pattern",
                text=f"{order_rate:.0f}% actively ordering ({ordering}/{total} events) — high conversion rate indicates strong purchase intent at this location",
                evidence=[f"ordering={ordering}", f"total={total}"],
            ))

        if browsing > 2 and ordering > 0 and browsing >= ordering:
            browse_to_order = min(100, ordering / browsing * 100)
            insights.append(Insight(
                insight_type="correlation",
                text=f"{browsing} browsing events → {ordering} orders ({browse_to_order:.0f}% conversion) — optimize display placement to increase browse-to-purchase conversion",
                evidence=[f"browsing={browsing}", f"ordering={ordering}"],
            ))

        return insights

    def _deduplicate(self, insights: list[Insight]) -> list[Insight]:
        """Remove near-duplicate insights based on text similarity."""
        if len(insights) <= 1:
            return insights

        unique = [insights[0]]
        for ins in insights[1:]:
            words = set(ins.text.lower().split())
            is_dup = False
            for existing in unique:
                existing_words = set(existing.text.lower().split())
                if words and existing_words:
                    overlap = len(words & existing_words) / len(words | existing_words)
                    if overlap > 0.6:
                        is_dup = True
                        break
            if not is_dup:
                unique.append(ins)

        return unique
