"""K3 Evaluator — measure actionable insights per day per camera.

Fixed eval set: 200 synthetic events over 24 hours, realistic VLM descriptions.
Metric: count of insights with score >= 0.5
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

# ── Synthetic eval data ──────────────────────────────────────────────────────

def generate_eval_events(n: int = 200) -> list[dict]:
    """Generate realistic VLM-style event descriptions for 1 day."""
    base = datetime(2026, 3, 22, 6, 0, 0, tzinfo=timezone.utc)

    # Descriptions grouped by hour-of-day pattern (busier midday)
    templates = {
        "morning_rush": [
            "Young woman in business attire walking briskly toward entrance, carrying a laptop bag and Starbucks grande cup",
            "Middle-aged man in gray suit, badge visible on lanyard, entering through main door",
            "Two young adults walking together, one carrying a backpack, the other holding a phone",
            "Woman in her 30s pushing a stroller, walking slowly along the sidewalk",
            "Man in construction vest and hard hat, standing near the loading dock, talking on phone",
            "Elderly gentleman with a walking stick, moving slowly toward the bench",
            "Young man in hoodie and jeans, jogging across the parking lot",
            "Woman in scrubs carrying a large tote bag, walking toward the east entrance",
            "Male cyclist dismounting near the bike rack, wearing a helmet and reflective vest",
            "Female security guard standing at the main entrance, checking badges",
        ],
        "midday_peak": [
            "Group of 4 young professionals walking toward the cafe, laughing and chatting",
            "Man in blue polo shirt ordering at the counter, pointing at the menu board",
            "Woman seated at outdoor table with laptop open, tall latte beside her",
            "Delivery driver in uniform carrying 3 packages toward the loading area",
            "Two middle-aged women sitting on a bench, one eating a pastry, the other checking her phone",
            "Young man in a black jacket standing near the ATM, looking at his phone",
            "Child running ahead of parent toward the playground, parent carrying a diaper bag",
            "White Tesla Model 3 pulling into handicap spot near entrance",
            "Red Honda Civic parked in visitor lot, no occupant visible",
            "Man in chef's coat taking a smoke break by the side door",
            "Woman in athletic wear jogging past the camera, AirPods visible",
            "Older man in wheelchair being pushed by a younger woman toward the ramp entrance",
            "Black SUV (BMW X5) idling in fire lane for approximately 2 minutes",
            "Young couple sharing a venti Frappuccino on the patio, both on laptops",
            "Maintenance worker in orange vest inspecting the parking lot light fixture",
        ],
        "afternoon_slow": [
            "Single person walking through empty lobby, male, mid-40s, carrying briefcase",
            "No movement detected in the main corridor for the past 15 minutes",
            "Cleaning crew (2 people in uniforms) mopping the floor near entrance",
            "Silver Toyota Camry slowly driving through the lot, appears to be looking for a spot",
            "Dog walker with 3 dogs passing along the sidewalk, heading north",
            "Teenager sitting on curb, skateboard beside him, scrolling phone",
            "UPS truck parked at loading dock, driver unloading boxes on a dolly",
            "Woman in sun hat tending to the flower bed near the main sign",
        ],
        "evening_close": [
            "Stream of 6 people exiting the building, end of shift pattern, mostly carrying bags",
            "Security guard locking the side entrance, checking the door handle twice",
            "Last car in the lot, white Ford F-150, headlights on, engine running",
            "Man in dark clothing walking along the perimeter fence, looking around",
            "Empty parking lot, overhead lights activated, no pedestrians visible",
            "Raccoon spotted near the dumpster area, no humans in frame",
        ],
    }

    # Hour distribution: morning 6-9, midday 10-14, afternoon 15-18, evening 19-22
    hour_groups = {
        range(6, 10): ("morning_rush", 8),     # 8 events/hr
        range(10, 15): ("midday_peak", 15),    # 15 events/hr
        range(15, 19): ("afternoon_slow", 5),  # 5 events/hr
        range(19, 23): ("evening_close", 3),   # 3 events/hr
    }

    events = []
    event_idx = 0
    for hours, (group, rate) in hour_groups.items():
        descs = templates[group]
        for h in hours:
            for i in range(rate):
                if event_idx >= n:
                    break
                ts = base + timedelta(hours=h, minutes=int(i * 60 / rate))
                desc = descs[event_idx % len(descs)]
                events.append({
                    "id": f"evt_eval_{event_idx:04d}",
                    "timestamp": ts.isoformat(),
                    "camera_id": "cam_eval_01",
                    "camera_name": "Eval Camera 1",
                    "description": desc,
                    "frame_path": None,
                    "alert_triggered": False,
                    "metadata": {},
                })
                event_idx += 1

    return events[:n]


# ── Current system baseline (reimplements insights.py logic) ─────────────────

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

GENDER_PATTERNS = {
    "male": re.compile(r"\b(male|man|boy|gentleman|he\b|his\b)", re.I),
    "female": re.compile(r"\b(female|woman|girl|lady|she\b|her\b)", re.I),
}

AGE_PATTERNS = {
    "young": re.compile(r"\b(young|youth|teen|child|kid|20s|early 20|young adult)\b", re.I),
    "middle_aged": re.compile(r"\b(middle.?aged?|30s|40s|50s|mid-\d0s)\b", re.I),
    "elderly": re.compile(r"\b(elderly|older|senior|aged|gray hair|grey hair|walking stick)\b", re.I),
}


def extract_baseline_insights(events: list[dict]) -> list[str]:
    """Reimplements the current behavioral notable_patterns logic."""
    from collections import Counter, defaultdict

    activities: Counter = Counter()
    activity_by_hour: dict = defaultdict(Counter)
    total = len(events) or 1

    for e in events:
        desc = e.get("description", "")
        try:
            hour = datetime.fromisoformat(e["timestamp"]).hour
        except Exception:
            hour = None

        for label, pat in ACTIVITY_PATTERNS.items():
            if pat.search(desc):
                activities[label] += 1
                if hour is not None:
                    activity_by_hour[hour][label] += 1

    patterns: list[str] = []

    # Bag carrying
    bag_count = activities.get("carrying_bag", 0) + activities.get("carrying_briefcase", 0)
    if bag_count > 0:
        pct = round(bag_count / total * 100)
        patterns.append(f"{pct}% of observations involve people carrying bags or briefcases")

    # Walk vs stand
    walking = activities.get("walking", 0)
    standing = activities.get("standing", 0)
    if walking > 0 and standing > 0:
        ratio = round(walking / (walking + standing) * 100)
        patterns.append(f"{ratio}% of people are in motion (walking) vs {100 - ratio}% stationary")

    # Peak activity
    if activity_by_hour:
        peak_hour = max(activity_by_hour, key=lambda h: sum(activity_by_hour[h].values()))
        peak_total = sum(activity_by_hour[peak_hour].values())
        patterns.append(f"Peak activity at {peak_hour:02d}:00 with {peak_total} observed behaviors")

    # Parking
    parked = activities.get("parking", 0)
    if parked > 0:
        pct = round(parked / total * 100)
        patterns.append(f"{pct}% of events involve parked vehicles")

    return patterns


# ── Insight scoring ──────────────────────────────────────────────────────────

@dataclass
class ScoredInsight:
    """An insight with quality scores."""
    text: str
    insight_type: str  # anomaly, trend, pattern, correlation, recommendation, comparison
    specificity: float = 0.0  # has concrete numbers/times?
    evidence: float = 0.0     # grounded in observed data?
    actionability: float = 0.0  # implies a decision?
    novelty: float = 1.0      # not redundant?

    @property
    def score(self) -> float:
        return (self.specificity + self.evidence + self.actionability + self.novelty) / 4

    @property
    def passes(self) -> bool:
        return self.score >= 0.5


def score_insight(text: str, insight_type: str = "pattern", all_insights: list[str] | None = None) -> ScoredInsight:
    """Auto-score an insight string."""
    s = ScoredInsight(text=text, insight_type=insight_type)

    # Specificity: has numbers, percentages, times, or dollar amounts?
    has_number = bool(re.search(r"\d+", text))
    has_percent = bool(re.search(r"\d+%", text))
    has_time = bool(re.search(r"\d{1,2}:\d{2}|\d{1,2}(am|pm|:00)", text, re.I))
    has_dollar = bool(re.search(r"\$\d+", text))
    specificity_signals = sum([has_number, has_percent, has_time, has_dollar])
    s.specificity = min(1.0, specificity_signals * 0.4)

    # Evidence: references observed data (mentions "observed", "detected", counts)?
    evidence_words = ["observ", "detect", "record", "event", "camera", "seen", "count", "total", "average", "median"]
    evidence_count = sum(1 for w in evidence_words if w in text.lower())
    s.evidence = min(1.0, evidence_count * 0.3 + (0.3 if has_number else 0))

    # Actionability: implies something should happen
    action_words = ["suggest", "recommend", "consider", "risk", "opportunity", "should", "indicates",
                     "implies", "concern", "bullish", "bearish", "alert", "investigate", "optimize",
                     "increase", "decrease", "staffing", "capacity", "target"]
    action_count = sum(1 for w in action_words if w in text.lower())
    s.actionability = min(1.0, action_count * 0.35)

    # Novelty: check overlap with other insights
    if all_insights:
        text_words = set(text.lower().split())
        max_overlap = 0
        for other in all_insights:
            if other == text:
                continue
            other_words = set(other.lower().split())
            if text_words and other_words:
                overlap = len(text_words & other_words) / len(text_words | other_words)
                max_overlap = max(max_overlap, overlap)
        s.novelty = max(0, 1.0 - max_overlap)

    return s


# ── Run eval ─────────────────────────────────────────────────────────────────

def run_eval(insight_fn, label: str = "baseline") -> dict:
    """Run K3 eval with a given insight extraction function."""
    events = generate_eval_events(200)
    insights_raw = insight_fn(events)

    scored = []
    for text in insights_raw:
        si = score_insight(text, all_insights=insights_raw)
        scored.append(si)

    passing = [s for s in scored if s.passes]
    k3 = len(passing)

    result = {
        "label": label,
        "total_events": len(events),
        "total_insights_generated": len(insights_raw),
        "insights_passing": k3,
        "k3_score": k3,
        "k3_target": 5,
        "k3_met": k3 >= 5,
        "insights": [
            {
                "text": s.text,
                "type": s.insight_type,
                "score": round(s.score, 3),
                "specificity": round(s.specificity, 3),
                "evidence": round(s.evidence, 3),
                "actionability": round(s.actionability, 3),
                "novelty": round(s.novelty, 3),
                "passes": s.passes,
            }
            for s in scored
        ],
    }

    return result


def extract_v1_insights(events: list[dict]) -> list[str]:
    """InsightExtractor v1 — structured, typed, actionable insights."""
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from trio_core.insights import InsightExtractor

    extractor = InsightExtractor()
    insights = extractor.extract(events)
    return [ins.text for ins in insights]


def print_result(result: dict) -> None:
    """Pretty-print eval results."""
    print(f"\nLabel: {result['label']}")
    print(f"Events: {result['total_events']}")
    print(f"Insights generated: {result['total_insights_generated']}")
    print(f"Insights passing (score>=0.5): {result['insights_passing']}")
    print(f"K3 score: {result['k3_score']} / {result['k3_target']}")
    print(f"K3 met: {'YES' if result['k3_met'] else 'NO'}")

    print("\nInsight details:")
    for i, ins in enumerate(result["insights"], 1):
        status = "PASS" if ins["passes"] else "FAIL"
        print(f"  {i}. [{status}] (score={ins['score']:.2f}) {ins['text'][:100]}")
        print(f"     specificity={ins['specificity']:.2f} evidence={ins['evidence']:.2f} "
              f"actionability={ins['actionability']:.2f} novelty={ins['novelty']:.2f}")


def save_result(result: dict) -> None:
    """Save result to JSON log."""
    results_path = Path(__file__).parent / "results" / "k3_eval_results.json"
    results_path.parent.mkdir(exist_ok=True)

    all_results = []
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    all_results.append(result)
    results_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {results_path}")


def main():
    print("=" * 60)
    print("K3 Evaluation — Insight Richness")
    print("=" * 60)

    # Round 1: Baseline
    print("\n--- BASELINE (current regex system) ---")
    r1 = run_eval(extract_baseline_insights, label="baseline_v0")
    print_result(r1)
    save_result(r1)

    # Round 2: InsightExtractor v1
    print("\n--- INSIGHT EXTRACTOR v1 ---")
    r2 = run_eval(extract_v1_insights, label="insight_extractor_v1")
    print_result(r2)
    save_result(r2)

    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"  Baseline:  K3={r1['k3_score']}/5 ({r1['total_insights_generated']} generated, {r1['insights_passing']} passing)")
    print(f"  V1:        K3={r2['k3_score']}/5 ({r2['total_insights_generated']} generated, {r2['insights_passing']} passing)")
    delta = r2['k3_score'] - r1['k3_score']
    print(f"  Delta:     {'+' if delta > 0 else ''}{delta}")

    return r2


if __name__ == "__main__":
    main()
