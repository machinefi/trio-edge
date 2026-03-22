#!/usr/bin/env python3
"""Customer Persona Simulation — seed realistic demo data for 4 customer types.

Personas:
1. Marcus Chen — AIDC Security Chief at Equinix data center
2. Sarah Walsh — Short-seller analyst at Muddy Waters Capital
3. James Park — VP Operations at a 200-location coffee chain
4. Diana Reeves — Asset Manager at Brookfield commercial real estate

Each persona gets: camera, 24h events, counting metrics, realistic VLM descriptions.
Then we run InsightExtractor + reports to see what each customer would experience.
"""

from __future__ import annotations

import asyncio
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

random.seed(2026)


@dataclass
class Persona:
    name: str
    title: str
    company: str
    camera_name: str
    source_url: str
    scene_type: str
    report_type: str
    intent: str
    descriptions: list[str]
    hourly_traffic: dict[int, int]  # hour -> base count


# ── Persona 1: AIDC Security ────────────────────────────────────────────────

MARCUS = Persona(
    name="Marcus Chen",
    title="Chief Security Officer",
    company="Equinix SG3 Data Center",
    camera_name="SG3 Main Entrance — Gate A",
    source_url="rtsp://10.0.1.50/gate-a",
    scene_type="security",
    report_type="security",
    intent="24/7 physical access monitoring, badge compliance, unauthorized entry detection, contractor tracking",
    descriptions=[
        "Male employee with Equinix badge on lanyard, business casual, scanning badge at turnstile, access granted",
        "Female contractor with yellow visitor badge, escorted by employee in blue polo, entering through mantrap",
        "Two technicians in ESD smocks carrying server rails, badges visible, proceeding to cage area",
        "Unidentified male in dark hoodie approaching side door, no badge visible, appears to be checking door handle",
        "Security patrol officer in uniform completing hourly round, flashlight visible, checking perimeter cameras",
        "Delivery driver from DHL with shipping manifest, standing at loading dock, awaiting badge verification",
        "Female employee tailgating through secured door without badge scan, following closely behind authorized entry",
        "Maintenance crew of 3 in orange vests, all with contractor badges, accessing HVAC room via service corridor",
        "Door #7 propped open with fire extinguisher, no personnel in vicinity, alarm indicator not triggered",
        "Male in suit with Equinix executive badge, entering with guest — guest has no visible badge or escort form",
        "Night shift guard relieving day shift at Gate A, shift change handoff with clipboard exchange",
        "Forklift operator moving equipment pallet from loading dock to staging area, hard hat and safety vest worn",
        "Two visitors in business suits waiting in mantrap, one holding a laptop bag, other checking phone",
        "Cleaning crew entering via service entrance at 22:15, valid contractor badges displayed",
        "Unauthorized vehicle parked in restricted Zone C, no parking permit displayed, silver sedan license plate partially obscured",
        "Employee exiting with large cardboard box, badge scanned on exit, item appears to be personal belonging",
        "Fire alarm panel check by technician, testing LED indicators, scheduled maintenance activity",
        "Suspicious package left near transformer area, small black bag, no owner visible in 5-minute observation",
        "Server delivery from Dell: 3 pallets unloaded at Dock B, receiving staff verifying serial numbers against PO",
        "Group of 5 touring the facility, led by sales engineer, all wearing visitor badges with escort",
    ],
    hourly_traffic={
        0: 2, 1: 1, 2: 1, 3: 1, 4: 2, 5: 3,
        6: 8, 7: 15, 8: 25, 9: 20, 10: 18, 11: 15,
        12: 12, 13: 18, 14: 20, 15: 15, 16: 12, 17: 8,
        18: 25, 19: 10, 20: 5, 21: 3, 22: 4, 23: 2,
    },
)

# ── Persona 2: Short Seller ─────────────────────────────────────────────────

SARAH = Persona(
    name="Sarah Walsh",
    title="Senior Research Analyst",
    company="Muddy Waters Capital",
    camera_name="SBUX #4892 — Union Square SF",
    source_url="rtsp://fieldcam.mw.internal/sbux-4892",
    scene_type="retail",
    report_type="investment",
    intent="Track Starbucks foot traffic, order sizes, food attach rate, customer demographics for earnings prediction. Compare observed ASP vs reported $5.50",
    descriptions=[
        "Young woman in athletic wear ordering a tall green tea latte, paying with phone, no food item",
        "Man in 30s in business suit ordering a grande Americano and a croissant, card payment, rushed",
        "Group of 3 college students with laptops, ordering 2 grande Frappuccinos and 1 venti iced coffee, settling at large table",
        "Middle-aged woman ordering a venti chai tea latte, adds a breakfast sandwich, carries Nordstrom shopping bag",
        "Male in his 20s, headphones on, orders a tall drip coffee, no food, straight to go — under 90 seconds total",
        "Woman in her 40s with toddler, orders a grande decaf latte and a cake pop for the child, stays to sit",
        "Two businessmen in suits ordering grande espressos, no food, quick transaction, appear to be in between meetings",
        "Elderly couple ordering a tall pike place and a grande caramel macchiato, share a slice of lemon loaf",
        "Female barista restocking cups, 7 customers in visible queue, wait time appears ~4 minutes",
        "Delivery partner picking up 4 mobile orders at handoff counter, multiple venti cups visible",
        "Man ordering 3 grande lattes (appears buying for office), no food items, pays with company card",
        "Young couple sharing a venti Frappuccino and a blueberry muffin at outdoor patio table, both on phones",
        "Woman in scrubs ordering a tall cold brew, grabs a protein box from display, very quick transaction",
        "Solo customer with laptop, been seated 25+ minutes, grande cup nearly empty, working on spreadsheet",
        "Man in construction vest ordering a venti dark roast and 2 breakfast wraps, to-go, large order value",
        "Teenage girl ordering a grande pink drink, adds a cookie, pays cash, takes selfie with cup",
        "Woman in business attire, mobile order pickup, grande flat white, exits within 30 seconds",
        "Group of 4 ordering: 2 tall, 1 grande, 1 venti — mixed drinks, 2 food items, total ~$32",
        "Homeless person sitting outside entrance, not ordering, store manager approaches to have conversation",
        "Uber Eats driver waiting for order, checking phone, 3 bags ready on counter",
    ],
    hourly_traffic={
        5: 5, 6: 20, 7: 45, 8: 55, 9: 40, 10: 30,
        11: 35, 12: 50, 13: 40, 14: 25, 15: 20, 16: 25,
        17: 30, 18: 20, 19: 15, 20: 10, 21: 5,
    },
)

# ── Persona 3: Ops VP ────────────────────────────────────────────────────────

JAMES = Persona(
    name="James Park",
    title="VP Operations",
    company="Blue Bottle Coffee (200 locations)",
    camera_name="Blue Bottle — Hayes Valley SF",
    source_url="rtsp://ops.bluebottle.internal/hayes-valley-01",
    scene_type="retail",
    report_type="investment",
    intent="Optimize staffing based on traffic patterns, measure queue wait times, track peak hours for labor scheduling, monitor conversion rate",
    descriptions=[
        "Morning rush: line of 8 customers at counter, 2 baristas working, wait time ~6 minutes estimated",
        "Single customer ordering a pour-over, barista explaining origin story, high-touch interaction ~3 minutes",
        "Woman with stroller struggling to navigate between tables, limited floor space during peak",
        "3 customers waiting for mobile orders, checking phones, orders ready in ~2 minutes",
        "Empty store at 14:30, only 1 customer seated with laptop, 2 staff idle behind counter",
        "Group of 5 entering together, debating menu items, blocking entrance flow for ~90 seconds",
        "Regular customer (recognized by staff greeting by name), grande New Orleans iced coffee, no wait",
        "Man photographing his latte art before drinking, likely social media post, seated at window table",
        "Delivery driver picking up 6 orders simultaneously, staff juggling between walk-in and delivery queue",
        "Customer complaint: returning a drink, barista remaking without argument, ~4 minutes resolution",
        "Two people working on laptops, occupying 4-person table for 90+ minutes, not ordering additional items",
        "Queue length hit 12 during 8:30am rush, 3 customers visibly frustrated, 1 left without ordering",
        "New employee (training badge visible) shadowing experienced barista, slower preparation pace",
        "Customer trying to use outlet near window seat, outlet not working, asking staff about alternatives",
        "Afternoon lull: 1 barista sufficient, second staff member restocking and cleaning",
        "Parent with 3 children, ordering 1 adult drink + 3 pastries, children running between tables",
        "Business meeting: 4 people in suits at corner table, ordered 4 lattes, discussing documents",
        "Closing time: staff mopping floors, last customer asked to leave, chairs being stacked",
    ],
    hourly_traffic={
        6: 10, 7: 35, 8: 50, 9: 40, 10: 25, 11: 30,
        12: 35, 13: 25, 14: 15, 15: 20, 16: 25, 17: 20,
        18: 15, 19: 8, 20: 3,
    },
)

# ── Persona 4: Real Estate Asset Manager ────────────────────────────────────

DIANA = Persona(
    name="Diana Reeves",
    title="Senior Asset Manager",
    company="Brookfield Properties — Westfield Mall",
    camera_name="Westfield SF Centre — Main Atrium L1",
    source_url="rtsp://brookfield.internal/westfield-sf-atrium",
    scene_type="retail",
    report_type="investment",
    intent="Track mall foot traffic for lease renewal negotiations, identify peak hours and dead zones, monitor anchor tenant draw, compare weekday vs weekend patterns",
    descriptions=[
        "Steady stream of shoppers entering from Market Street entrance, mix of tourists and locals, moderate pace",
        "Family of 4 with shopping bags from multiple stores — Nordstrom, Zara, and H&M visible",
        "Group of teenagers browsing near the escalator, no shopping bags, window shopping behavior",
        "Woman in business attire walking briskly through atrium, appears to be using mall as pedestrian shortcut",
        "Elderly couple resting on atrium bench, 3 shopping bags beside them, appears to be taking a break",
        "Mall security guard on patrol near fountain area, radio visible, standard round",
        "Man pushing stroller with shopping bags hanging from handles, wife carrying additional bags, productive shopping trip",
        "Tourist group of 6 with cameras and maps, stopping to take photos of the atrium architecture",
        "Food court overflow: 12 people carrying food trays looking for seating, peak lunch congestion",
        "Nordstrom entrance area: steady 2-way foot traffic, appears to be highest draw among anchor tenants",
        "Empty corridor near Level 3 specialty stores, only 2 pedestrians in 5-minute observation period",
        "Flash sale signage at Forever 21: visible queue of ~15 customers extending into common area",
        "Maintenance crew cleaning spill near escalator, temporary barrier reducing foot traffic flow by 50%",
        "Young professionals in groups of 2-3 heading toward restaurant row, early evening dinner crowd",
        "Amazon return counter: consistent queue of 5-8 people with packages, high-traffic service point",
        "Street performer near main entrance drawing crowd of ~20 spectators, creating bottleneck",
        "Luxury wing (Louis Vuitton, Gucci): 3 customers visible, security guard at entrance, low volume but high value",
        "Weekend morning: families with children heading to play area, stroller density noticeably higher",
        "Store closure visible: shuttered unit between Sephora and Apple, 'Coming Soon' signage with no tenant name",
        "Late evening: thinning crowds, mostly restaurant traffic, retail storefronts beginning to close",
    ],
    hourly_traffic={
        10: 50, 11: 80, 12: 120, 13: 130, 14: 110,
        15: 100, 16: 90, 17: 85, 18: 70, 19: 50, 20: 30, 21: 10,
    },
)


ALL_PERSONAS = [MARCUS, SARAH, JAMES, DIANA]


async def seed_persona(store, persona: Persona) -> str:
    """Seed a persona's camera, events, and metrics into the store."""
    # Create camera
    cam_id = await store.create_camera({
        "name": persona.camera_name,
        "source_url": persona.source_url,
        "watch_condition": "Describe this person and their activity.",
        "intent": persona.intent,
        "intent_config": {
            "persona": "security_officer" if persona.scene_type == "security" else "investment_analyst",
            "scene_type": persona.scene_type,
            "report_type": persona.report_type,
            "customer_prompt": "Describe: age, gender, clothing, items, activity. One sentence.",
            "scene_prompt": "Describe: people count, busyness 1-10, what people are doing.",
            "correction_factor": 1.6,
            "auto_detected": True,
        },
    })

    base_ts = datetime(2026, 3, 22, 0, 0, 0, tzinfo=timezone.utc)

    # Generate events + metrics for each hour
    event_count = 0
    metric_count = 0
    for hour, base_traffic in persona.hourly_traffic.items():
        # Events: proportional to traffic
        n_events = max(1, base_traffic // 5)
        for i in range(n_events):
            minute = int(i * 60 / n_events)
            ts = base_ts + timedelta(hours=hour, minutes=minute)
            desc = persona.descriptions[event_count % len(persona.descriptions)]

            is_alert = any(w in desc.lower() for w in [
                "unauthorized", "suspicious", "propped", "tailgating",
                "no badge", "complaint", "frustrated", "left without",
            ])

            await store.insert({
                "timestamp": ts.isoformat(),
                "camera_id": cam_id,
                "camera_name": persona.camera_name,
                "description": desc,
                "alert_triggered": is_alert,
            })
            event_count += 1

        # Metrics: every 5 minutes
        for m in range(0, 60, 5):
            ts = base_ts + timedelta(hours=hour, minutes=m)
            noise = random.randint(-max(3, base_traffic // 5), max(3, base_traffic // 5))
            count = max(0, base_traffic + noise)
            await store.insert_metric({
                "timestamp": ts.isoformat(),
                "camera_id": cam_id,
                "metric_type": "people_count",
                "value": count,
                "confidence": 0.85 + random.random() * 0.15,
            })
            if count > 0:
                await store.insert_metric({
                    "timestamp": ts.isoformat(),
                    "camera_id": cam_id,
                    "metric_type": "people_in",
                    "value": max(1, count // 3 + random.randint(-2, 2)),
                })
            metric_count += 1

    return cam_id


async def evaluate_persona(store, cam_id: str, persona: Persona) -> dict:
    """Run InsightExtractor + report quality check for a persona."""
    from trio_core.insights import InsightExtractor

    # Get all events
    result = await store.list_events(camera_id=cam_id, limit=10000)
    events = result["events"]

    # Extract insights
    extractor = InsightExtractor()
    insights = extractor.extract(events)

    # Categorize insights
    insight_types = {}
    for ins in insights:
        insight_types[ins.insight_type] = insight_types.get(ins.insight_type, 0) + 1

    # Get metrics summary
    metrics = await store.metrics_summary(cam_id)

    # Get daily report
    daily = await store.daily_report("2026-03-22", camera_id=cam_id)

    return {
        "persona": persona.name,
        "title": persona.title,
        "company": persona.company,
        "camera": persona.camera_name,
        "scene_type": persona.scene_type,
        "events": len(events),
        "alerts": sum(1 for e in events if e.get("alert_triggered")),
        "k3_insights": len(insights),
        "k3_met": len(insights) >= 5,
        "insight_types": insight_types,
        "insights": [
            {"type": ins.insight_type, "text": ins.text}
            for ins in insights
        ],
        "metrics": {
            "total_in": metrics.get("total_in", 0),
            "peak_hour": metrics.get("peak_hour", ""),
            "peak_count": metrics.get("peak_count", 0),
        },
        "daily_report": {
            "events": daily["total_events"],
            "alerts": daily["total_alerts"],
            "hours_covered": len(daily["hourly"]),
        },
    }


async def main():
    from trio_core.api.store import EventStore

    print("=" * 80)
    print("CUSTOMER PERSONA SIMULATION — 4 Customer Profiles")
    print("=" * 80)

    store = EventStore(db_path="data/demo_personas.db", frames_dir="data/frames")
    await store.init()

    results = []

    for persona in ALL_PERSONAS:
        print(f"\n{'─' * 80}")
        print(f"  {persona.name} — {persona.title}")
        print(f"  {persona.company}")
        print(f"  Camera: {persona.camera_name}")
        print(f"{'─' * 80}")

        cam_id = await seed_persona(store, persona)
        print(f"  Seeded: camera={cam_id}")

        result = await evaluate_persona(store, cam_id, persona)
        results.append(result)

        print(f"  Events: {result['events']} ({result['alerts']} alerts)")
        print(f"  K3: {result['k3_insights']} insights ({'MET' if result['k3_met'] else 'NOT MET'})")
        print(f"  Insight types: {result['insight_types']}")
        print(f"  Metrics: total_in={result['metrics']['total_in']}, peak={result['metrics']['peak_hour']}")

        print(f"\n  Top insights for {persona.name}:")
        for ins in result["insights"][:8]:
            print(f"    [{ins['type']:>14s}] {ins['text'][:85]}")

    # ── Overall Summary ──
    print(f"\n{'=' * 80}")
    print("CUSTOMER EXPERIENCE SUMMARY")
    print(f"{'=' * 80}")

    all_met = True
    for r in results:
        status = "PASS" if r["k3_met"] else "FAIL"
        if not r["k3_met"]:
            all_met = False
        print(f"  [{status}] {r['persona']:20s} | {r['company']:35s} | K3={r['k3_insights']:>2d}/5 | Events={r['events']:>3d} | Alerts={r['alerts']:>2d}")

    print(f"\n  Overall: {'ALL PERSONAS MET K3' if all_met else 'SOME PERSONAS FAILED K3'}")

    # ── Customer Feedback Simulation ──
    print(f"\n{'=' * 80}")
    print("SIMULATED CUSTOMER REACTIONS")
    print(f"{'=' * 80}")

    for r in results:
        p = next(p for p in ALL_PERSONAS if p.name == r["persona"])
        print(f"\n  {r['persona']} ({r['title']}, {r['company']}):")

        if p.scene_type == "security":
            # Security persona cares about: unauthorized access, badge compliance, patrols
            has_security = any("unauthorized" in i["text"].lower() or "badge" in i["text"].lower()
                               or "suspicious" in i["text"].lower() for i in r["insights"])
            has_patrol = any("patrol" in i["text"].lower() for i in r["insights"])
            has_anomaly = any(i["type"] == "anomaly" for i in r["insights"])

            if has_security and has_patrol and has_anomaly:
                print("    💬 'This is exactly what I need. Badge compliance rate, unauthorized")
                print("       access detection, patrol verification — all automated. We're paying")
                print("       $200K/yr for Sensormatic and I still have to manually review footage.'")
                print("    VERDICT: STRONG BUY SIGNAL")
            else:
                print("    💬 'Missing some security-specific insights I need.'")
                print("    VERDICT: NEEDS WORK")

        elif p.name == "Sarah Walsh":
            # Short seller cares about: ASP, traffic vs reported, food attach
            has_asp = any("asp" in i["text"].lower() or "$" in i["text"] for i in r["insights"])
            has_traffic = any("traffic" in i["text"].lower() or "peak" in i["text"].lower() for i in r["insights"])
            has_order = any("order" in i["text"].lower() or "conversion" in i["text"].lower() for i in r["insights"])

            if has_asp and has_traffic:
                print("    💬 'The drink ASP calculation is killer. If I can get this on 50 Starbucks")
                print("       locations, I can predict SSS before the earnings call. Placer.ai gives")
                print("       me foot traffic with 24hr delay — this is real-time with evidence.'")
                print("    VERDICT: STRONG BUY SIGNAL — wants multi-location deployment")
            else:
                print("    💬 'Need more retail-specific metrics.'")
                print("    VERDICT: NEEDS MORE RETAIL DEPTH")

        elif p.name == "James Park":
            # Ops VP cares about: peak staffing, queue wait, conversion
            has_staffing = any("staff" in i["text"].lower() or "peak" in i["text"].lower() for i in r["insights"])
            has_behavior = any("stationary" in i["text"].lower() or "transit" in i["text"].lower()
                               or "destination" in i["text"].lower() for i in r["insights"])

            if has_staffing and has_behavior:
                print("    💬 'If this can tell me that 8:30am needs 3 baristas but 2pm only needs 1,")
                print("       that saves me $15K/year/location in labor. Across 200 stores = $3M.'")
                print("    VERDICT: STRONG ROI CASE — wants pilot at 5 locations")
            else:
                print("    💬 'Interesting but need queue time and staffing recommendations.'")
                print("    VERDICT: NEEDS OPERATIONAL DEPTH")

        else:  # Diana
            has_traffic = any("traffic" in i["text"].lower() or "peak" in i["text"].lower() for i in r["insights"])
            has_anchor = any("group" in i["text"].lower() or "visitor" in i["text"].lower() for i in r["insights"])

            if has_traffic and len(r["insights"]) >= 5:
                print("    💬 'If I can show tenants that their wing gets 130 people/hour at lunch")
                print("       vs 50 in the morning, I can justify premium rents for lunch-traffic")
                print("       locations. This data is worth $500K in lease negotiations.'")
                print("    VERDICT: HIGH VALUE — wants data for lease renewal season")
            else:
                print("    💬 'Need zone-level traffic data per tenant area.'")
                print("    VERDICT: NEEDS ZONE ANALYTICS")

    # Save results
    results_path = Path(__file__).parent / "results" / "persona_eval.json"
    results_path.parent.mkdir(exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {results_path}")

    await store.close()


if __name__ == "__main__":
    asyncio.run(main())
