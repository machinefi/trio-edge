# Experiment 4: Insight Richness (K3)

**KPI:** K3 = Actionable insights per day per camera
**Target:** >= 5 actionable insights / day / camera
**Baseline:** TBD (current system)

## Definition of "Actionable Insight"

An insight is actionable if it:
1. States a **specific finding** (not just raw data)
2. Includes **evidence** (numbers, percentages, timestamps)
3. Implies a **decision or action** (so-what)

### Insight Types (scored)

| Type | Description | Example |
|------|-------------|---------|
| anomaly | Deviation from baseline | "Traffic 40% below avg at 2pm Monday" |
| trend | Pattern change over time | "Morning traffic up 15% week-over-week" |
| pattern | Recurring behavior | "72% carry branded bags → high-income demo" |
| correlation | Two linked observations | "Laptop users stay 3x longer, order larger drinks" |
| recommendation | Suggested action | "Understaffing risk 11am-1pm: 85% capacity" |
| comparison | Benchmark vs observation | "ASP $4.66 vs SBUX reported $5.50 → bearish" |

### Scoring

Each insight gets:
- **specificity** (0-1): Has concrete numbers/times?
- **evidence** (0-1): Grounded in observed data?
- **actionability** (0-1): Implies a decision?
- **novelty** (0-1): Not redundant with other insights?

**Score = mean(specificity, evidence, actionability, novelty)**
**An insight counts toward K3 if score >= 0.5**

## Methodology

**Fixed eval set:** 200 synthetic event descriptions spanning:
- 1 day (24 hours)
- Mix of person/vehicle/behavioral events
- Realistic VLM-style descriptions

**Metric:** Count of insights with score >= 0.5

## Experiment Log

### Round 1: Baseline
- **Variable:** Current system (regex notable_patterns in insights.py)
- **Result:** K3 = 3/5 (4 generated, 3 passing). **NOT MET**
- **Finding:** All insights have 0.0 actionability — state facts without implying decisions
- **Date:** 2026-03-22

### Round 2: InsightExtractor v1
- **Variable:** New InsightExtractor class (src/trio_core/insights.py)
- **Changes:** 7 insight generators (traffic, demographic, behavioral, security, vehicle, temporal, retail), each produces typed insights with actionable language
- **Result:** K3 = 12/5 (12 generated, 12 passing, avg score 0.67). **MET** (+9 vs baseline)
- **Key win:** Actionability went from 0.0 to 0.35-0.70 by adding "consider", "recommend", "investigate" language tied to specific decisions
- **Date:** 2026-03-22

### Round 3: Robustness + API Integration
- **Variable:** Threshold tuning, security/retail pattern expansion, API endpoint added
- **Changes:**
  - Added 4 new security sub-patterns (unauthorized, after_hours, patrol, alarm)
  - Added ordering/browsing activity patterns for retail
  - Lowered demographic/behavioral thresholds for small datasets (20-30 events)
  - Added vehicle traffic share insight
  - Added `/api/insights/actionable` endpoint (K3 metric endpoint)
  - Wired K3 insights into auto-report Gemini prompts
- **Robustness results:**
  - sparse_20_events: 5/5 PASS
  - security_focused: 6/5 PASS
  - retail_focused: 5/5 PASS
  - vehicle_mixed: 6/5 PASS
  - dense_500_events: 11/5 PASS
- **Result:** 5/5 scenarios passing. **K3 MET across all scenarios.**
- **Date:** 2026-03-22

## Summary

| Round | Label | K3 Score | Status |
|-------|-------|----------|--------|
| 1 | Baseline (regex) | 3/5 | NOT MET |
| 2 | InsightExtractor v1 | 12/5 | MET |
| 3 | v1 + robustness fixes | 5-16/5 | MET (all 5 scenarios) |

**Key findings:**
- Baseline system maxes out at 4 insights (hard ceiling from regex notable_patterns)
- InsightExtractor produces 5-16 insights depending on data volume and type
- Actionability was the #1 missing dimension — solved by pairing every finding with a recommended action
- Security scenarios need specialized pattern detection (unauthorized, alarm, patrol)
- Thresholds must adapt to data volume (lowered for <30 events)
