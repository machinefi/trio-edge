"""K3 Robustness eval — test InsightExtractor across different scenarios.

Scenarios:
1. Sparse data (20 events / day)
2. Dense data (500 events / day)
3. Security-only events
4. Retail-only events
5. Mixed vehicle + pedestrian
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from trio_core.insights import InsightExtractor


def _make_events(descriptions: list[str], hours_spread: int = 16) -> list[dict]:
    """Make events spread across hours."""
    base = datetime(2026, 3, 22, 6, 0, 0, tzinfo=timezone.utc)
    events = []
    for i, desc in enumerate(descriptions):
        h = (i * hours_spread) // len(descriptions)
        ts = base + timedelta(hours=h, minutes=(i % 4) * 15)
        events.append({
            "id": f"evt_{i:04d}",
            "timestamp": ts.isoformat(),
            "camera_id": "cam_test",
            "camera_name": "Test Camera",
            "description": desc,
            "frame_path": None,
            "alert_triggered": False,
            "metadata": {},
        })
    return events


# ── Scenarios ────────────────────────────────────────────────────────────────

SPARSE_DESCS = [
    "Man in suit walking toward entrance",
    "Woman carrying a bag, walking slowly",
    "Two people standing near the bench",
    "Car parked in lot, no occupant",
    "Security guard at main door",
    "Young woman jogging past camera",
    "Delivery truck at loading dock",
    "Man in hard hat near construction area",
    "Woman on phone, walking east",
    "Elderly man with walking stick near bench",
    "Group of 3 teens walking together",
    "White sedan pulling into parking spot",
    "Woman with laptop at outdoor table, grande cup",
    "Badge visible on man entering side door",
    "Man in dark clothing walking along fence, looking around",
    "Empty lobby, cleaning crew mopping floor",
    "Dog walker with 2 dogs on sidewalk",
    "UPS driver unloading packages",
    "Woman in scrubs entering east door",
    "Man seated on bench reading newspaper",
]

SECURITY_DESCS = [
    "Male employee with badge entering through main gate at normal pace",
    "Unidentified person near perimeter fence, no badge visible, dark clothing",
    "Security patrol vehicle circling building B perimeter",
    "Female contractor with visitor badge escorted through lobby",
    "Door #7 propped open, no personnel in vicinity",
    "Three employees exiting building, badges visible, end of shift",
    "Delivery driver at loading dock with proper credentials",
    "Unknown male lingering near server room entrance for 3 minutes",
    "Security guard checking badges at main entrance",
    "Female employee tailgating through secured door without badge scan",
    "Maintenance crew (2 workers in uniforms) accessing roof via service ladder",
    "Vehicle with no parking permit in restricted Zone A",
    "Night patrol officer completing rounds, flashlight visible",
    "Visitor without escort walking unaccompanied in restricted area",
    "Emergency exit alarm triggered, no personnel at door",
    "Cleaning crew accessing floor 3 with proper authorization badge",
    "Male in business suit, no badge, attempting entry at side door",
    "Security camera blind spot detected near loading area",
    "Shift change: 4 guards departing, 3 arriving at gate house",
    "Female with authorized badge accessing data hall at 02:30",
    "Suspicious package left near building entrance, no owner visible",
    "Guard station unattended for 12 minutes during patrol rotation",
    "Two contractors entering with expired visitor passes",
    "Vehicle tailgating through parking gate without scanning",
    "Motion detected in warehouse after hours, single person walking",
]

RETAIL_DESCS = [
    "Young woman ordering a grande latte at the counter",
    "Man in 30s browsing pastry display case, eventually selects a croissant",
    "Group of 4 students with laptops settling at the large table",
    "Woman in business attire ordering a venti iced coffee, seems rushed",
    "Older gentleman sitting alone, reading, tall drip coffee on table",
    "Two women in athletic wear ordering after a workout, both tall lattes",
    "Man in hoodie using mobile payment, grande Frappuccino",
    "Family of 3 ordering at counter, child pointing at cake pops",
    "Woman returning to pick up a mobile order, venti cup",
    "Barista handing out orders, 5 people waiting in queue",
    "Couple sharing a pastry at window seat, one grande one venti",
    "Man in suit quickly grabbing a tall coffee to go",
    "Three coworkers in polos having a meeting over coffee",
    "Woman on laptop, been seated for 20+ minutes, grande cup half empty",
    "Delivery partner picking up 4 mobile orders at counter",
    "Young couple splitting a venti Frappuccino on the patio",
    "Man with briefcase ordering a tall Americano",
    "Elderly woman with grande latte at corner table, reading newspaper",
    "Student with backpack and laptop, ordering a venti cold brew",
    "Woman in scrubs grabbing a quick tall coffee between shifts",
    "Barista restocking cups, 3 customers in line",
    "Man ordering a grande mocha, adds a sandwich, pays with card",
    "Two teenagers sharing food items, one tall one grande drink",
    "Woman on phone ordering a venti chai tea latte, carries shopping bags",
    "Businessman on laptop, grande espresso, tapping keyboard",
    "Group of 5 ordering multiple drinks, mix of tall and grande",
    "Young woman in gym clothes, orders tall green tea",
    "Man ordering 3 drinks (appears to be buying for coworkers), all grandes",
    "Woman with stroller, ordering a grande decaf, child has a cake pop",
    "Solo diner with venti cold brew and a muffin at outdoor table",
]

VEHICLE_MIXED_DESCS = [
    "White Tesla Model 3 entering parking structure, level 2",
    "Pedestrian crossing at marked crosswalk, male in his 40s",
    "Red Honda Civic parking in visitor section",
    "Woman walking dog along sidewalk, heading north",
    "Black BMW X5 SUV idling in fire lane for 3 minutes",
    "Cyclist on bike path, wearing helmet, heading east",
    "UPS truck double-parked at loading zone",
    "Two pedestrians walking together across parking lot",
    "Silver Toyota Camry circling lot looking for a spot",
    "Man on motorcycle parking near entrance",
    "Family of 4 loading groceries into gray minivan",
    "Uber/Lyft vehicle (sticker visible) stopping at pickup zone",
    "Jogger running through parking lot, reflective vest",
    "Ford F-150 truck backing into loading dock",
    "Woman with stroller navigating between parked cars",
    "Blue Hyundai Tucson SUV with dealer plates, likely test drive",
    "Security patrol car making rounds through lot",
    "Two construction workers walking to their trucks",
    "Electric vehicle charging at EV station, driver inside",
    "FedEx van delivering to building entrance",
]


def run_scenario(name: str, descriptions: list[str]) -> dict:
    """Run InsightExtractor on a scenario and return results."""
    events = _make_events(descriptions)
    extractor = InsightExtractor()
    insights = extractor.extract(events)

    return {
        "scenario": name,
        "events": len(events),
        "insights_count": len(insights),
        "k3_met": len(insights) >= 5,
        "insights": [
            {"type": ins.insight_type, "text": ins.text}
            for ins in insights
        ],
    }


def main():
    scenarios = [
        ("sparse_20_events", SPARSE_DESCS),
        ("security_focused", SECURITY_DESCS),
        ("retail_focused", RETAIL_DESCS),
        ("vehicle_mixed", VEHICLE_MIXED_DESCS),
        ("dense_500_events", RETAIL_DESCS * 17),  # ~510 events
    ]

    results = []
    print("=" * 70)
    print("K3 Robustness Evaluation — Multiple Scenarios")
    print("=" * 70)

    for name, descs in scenarios:
        r = run_scenario(name, descs)
        results.append(r)

        status = "PASS" if r["k3_met"] else "FAIL"
        print(f"\n[{status}] {name}: K3={r['insights_count']}/5 ({r['events']} events)")
        for ins in r["insights"]:
            print(f"  [{ins['type']:>14s}] {ins['text'][:90]}")

    # Summary
    passed = sum(1 for r in results if r["k3_met"])
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {passed}/{len(results)} scenarios met K3 target (>=5 insights)")
    print(f"{'=' * 70}")

    # Save
    results_path = Path(__file__).parent / "results" / "k3_robustness.json"
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
