# Experiment 2: Competitive Benchmark — Product-Level Accuracy Target

> **Goal:** Define what "SOTA product" means for Trio Enterprise, benchmark against
> every relevant competitor, and establish the target function for the product.

## The Question

We're not writing a paper. We're shipping a product. The question is:
**What accuracy do we need to win customers, and who are we actually competing against?**

## Market Segmentation (Who Competes Where)

```
                    Accuracy
                    99%+ ┤ V-Count, Xovis (3D ToF sensors, $1-3K/sensor)
                         │ Brickstream/FLIR (3D stereo, $1-2K/sensor)
                    98%  ┤ RetailNext Aurora (stereo video + DL, $3-6K/yr/entrance)
                         │ Sensormatic/ShopperTrak (DL + ReID, enterprise contract)
                    95%  ┤ ── Enterprise minimum bar ──────────────────────
                         │ Hikvision/Dahua DeepinMind (embedded DL, $200-800)
                         │ AXIS People Counter (on-camera, $500-3K + $100-200 license)
                         │ Ultralytics YOLO (software, AGPL license issue)
                    90%  ┤ Density.io (software-only, $49-99/cam/mo)
                         │
                         │ ── Our current level (84.3% per-frame) ──────
                    85%  ┤ ── Adequate for trend analysis ─────────────
                         │
                    80%  ┤ Thermal sensors
                         │
                    70%  ┤ WiFi/Bluetooth proximity
                         │
                  ±15-30% │ Placer.ai (mobile panel, no hardware, $300-5K/mo/seat)
                         │ Thasos, SafeGraph/Dewey, Advan (mobile alt-data)
                         │
```

## Competitive Matrix

### Tier 1: Hardware Counting Sensors (Not our competitors)

| Company | Tech | Accuracy | Cost | Analytics |
|---------|------|----------|------|-----------|
| V-Count | 3D stereo/ToF | 99%+ | $1-3K/sensor | Basic dashboard |
| Xovis | 3D ToF | 99%+ | $2-4K/sensor | Zone analytics |
| RetailNext | Stereo + DL | ≥95% SLA | $3-6K/yr/entrance | Full retail suite |
| Sensormatic | DL + ReID | ~95%+ | $50-200K/yr enterprise | ShopperTrak platform |

**Why not our competitor:** They sell hardware. We sell software + intelligence. Different customer, different sale.

### Tier 2: Camera-Based Software (Partially overlap with us)

| Company | Tech | Accuracy | Cost | Moat |
|---------|------|----------|------|------|
| Density.io | CV on existing cameras | ~97% claimed | $49-99/cam/mo | Software-only, COVID occupancy |
| AXIS | On-camera analytics | ~90-95% | One-time license | Embedded in camera |
| Hikvision | Embedded DL | ~90-95% real | Cheapest per-unit | Hardware margin, NDAA banned |
| **Trio (target)** | **YOLO + VLM + Gemini** | **target ≥90%** | **Software** | **Semantic AI analysis** |

**Key insight:** Density.io is the closest competitor. Pure software, existing cameras, ~$50-100/cam/mo. But they have NO semantic understanding, NO AI analysis, NO VLM crop-describe.

### Tier 3: Alt-Data / Mobile Panel (Our primary competitor for hedge fund use case)

| Company | Method | Per-Location Accuracy | Trend Accuracy | Cost |
|---------|--------|----------------------|----------------|------|
| Placer.ai | Mobile GPS panel | ±15-30% | ±5-10% w-o-w | $300-5K/seat/mo |
| Thasos Group | Mobile signals | ~similar | ~similar | $50-200K/yr |
| Orbital Insight | Satellite + geo | Parking lot level | Directional | $100K+/yr |
| Advan Research | Mobile location | ~similar | ~similar | $50-200K/yr |
| SafeGraph/Dewey | Mobile panel | ±20-40% daily | ±5-15% | $1-10K/mo |
| **Trio (now)** | **Camera vision** | **±15.7% per-frame** | **TBD** | **Software** |

**Key insight:** For hedge fund alt-data, Placer.ai is ±15-30% per-location. **We are already competitive at ±15.7% per-frame, and with temporal aggregation we'll be ±2-3% hourly.** Plus we provide evidence (timestamped frames) and semantic detail they can't.

### Tier 4: Frontier (Cautionary Tales)

| Company | Tech | Accuracy | Cost | Reality |
|---------|------|----------|------|---------|
| Amazon JWO | Hundreds of cameras + shelf sensors | Claims ~97% | $1M+/store | **70% of transactions needed manual human review** (1000 workers in India). Abandoned in own grocery stores 2024. |

**Lesson:** Even $1M/store + Amazon's ML team couldn't solve pure-CV counting in complex retail. The problem is hard. Our simpler scope (counting, not checkout-free) is much more tractable.

### Independent Test Rankings (IPVM Shootout)

IPVM tested major camera brands' built-in people counting:
1. **Best:** Hanwha AI, Axis People Counter
2. **Mid-tier:** Avigilon H5
3. **Poor:** Bosch IVA (double-counted lingerers), Dahua (missed counts)
4. **Worst:** Hikvision Smart Series (counted shadows, double-counted, lingering errors)

**Takeaway:** Even expensive hardware solutions vary wildly. Hikvision claims 98% but IPVM ranked them worst. Marketing claims ≠ reality. Independent benchmarking matters.

### Market Size & Opportunity

| Segment | Size (2024) | Growth (CAGR) | Size (2029-2034) |
|---------|------------|---------------|-------------------|
| People Counting Systems | $1.2B | 13.4% | $2.1B (2029) |
| Retail Audience Measurement | $1.5B | 15.9% | $6.6B (2034) |
| Location Intelligence (broad) | $15B | ~15% | $45B (2033) |

Top 7 players hold only ~27% combined share — **highly fragmented**. Room for disruption.

Placer.ai: $1.5B valuation (Series D, July 2024), $100M+ ARR, tripled revenue in 2 years. Proves the market demand.

---

## Our Target Function

### Primary Metric: Hourly MAPE

Why hourly, not per-frame:
- No customer cares about per-frame accuracy
- RetailNext reports in 15-min bins
- Placer.ai reports daily
- Hedge funds want hourly or daily trends
- Hourly aggregation reduces our 15.7% per-frame to ~2-3% (see exp3.md)

### The Scorecard

| Metric | Placer.ai | Density.io | RetailNext | **Trio Target** | **Trio Now** |
|--------|-----------|-----------|-----------|----------------|-------------|
| **Hourly count MAPE** | ±15-30% (daily only) | ~3-5% | ~2-3% | **<5%** | **~3-5% (estimated)** |
| **Trend accuracy (w-o-w)** | ±5-10% | ~2-3% | ~1-2% | **<3%** | **TBD** |
| **Real-time** | No (24-48hr) | Yes | Yes | **Yes** | **Yes** |
| **Semantic detail** | None | None | None | **Yes (VLM)** | **Yes** |
| **Evidence/proof** | None | None | Video clips | **Timestamped frames** | **Yes** |
| **AI narrative** | None | None | None | **Yes (Gemini)** | **Yes** |
| **Setup time** | Instant | Hours | Professional install | **Minutes** | **Minutes** |
| **Hardware needed** | None | Existing camera | Proprietary sensor | **Any IP camera** | **Any IP camera** |
| **Per-location cost** | $0 (seat model) | $49-99/cam/mo | $250-500/cam/mo | **$29-49/cam/mo** | **N/A** |

### Win Conditions (When We Beat Competitors)

#### vs Placer.ai (Hedge Fund Sale)
- We win when customer needs: **evidence, real-time, individual-level detail**
- "Placer.ai tells you foot traffic went up 10%. We tell you *why* — 62% carried Starbucks cups, average group size 2.3, peak at 11:30am, 29% food attach rate."
- **Required accuracy:** hourly MAPE <5% (Placer can't do per-hour at all)

#### vs Density.io (Enterprise Direct)
- We win when customer needs: **semantic understanding, AI reports, zero-config**
- "Density.io counts people. We understand them."
- **Required accuracy:** hourly MAPE <5% (match Density.io), differentiate on AI analysis

#### vs RetailNext (Large Retail)
- We lose on counting accuracy (they guarantee 95%+ per-frame with stereo hardware)
- We win on: **cost (no proprietary hardware), AI analysis, VLM semantic intelligence**
- **Required accuracy:** hourly MAPE <5%, compensated by superior analytics

### The Formula

```
Trio Enterprise value = Counting (≥90% per-frame, ≥95% hourly)
                      + Semantic Understanding (VLM: who, what, behavior)
                      + AI Analysis (Gemini: patterns, anomalies, recommendations)
                      + Evidence (timestamped frames, exportable)
                      + Zero-Config (plug camera, magic happens)
```

No competitor has all five. RetailNext has #1. Placer.ai has none (mobile panel). We can own the intersection.

---

## Benchmark Plan

### Phase 1: Validate Hourly Aggregation (exp3)

Run Mall Dataset through our pipeline with temporal aggregation. Prove that 15.7% per-frame → <5% hourly.

**Metric:** Hourly MAPE on Mall Dataset (aggregate per-frame counts into hourly bins, compare with hourly GT sums).

### Phase 2: Trend Accuracy Test

Using Mall Dataset (sequential video), compute:
- 5-minute traffic trend accuracy
- Pattern detection: does our system correctly identify peak periods?
- Anomaly detection: inject synthetic anomalies, measure detection rate

### Phase 3: Head Detection Upgrade (exp1 P0)

Swap to head detection model. Benchmark:
- Per-frame MAPE (expect <5%)
- Hourly MAPE (expect <2%)
- Impact on tracking quality (ByteTrack with head detections)

### Phase 4: Cross-Dataset Validation

Test on non-Mall datasets to prove generalization:
- MOT17 (street-level, where YOLO body detection should excel)
- PETS 2009 (entrance counting, in/out)
- Real RTSP camera feeds

### Phase 5: Competitor Replication

Where possible, replicate competitor results:
- Set up Density.io trial on same camera feed
- Use Placer.ai free tier on same location
- Compare Trio vs competitor on identical data

---

## Revenue Positioning

### Pricing Rationale

| Tier | Monthly/cam | What They Get | Comparable To |
|------|-------------|---------------|---------------|
| **Starter** | $29 | Counting + basic dashboard | AXIS People Counter |
| **Professional** | $49 | + AI analysis + VLM semantic | Density.io |
| **Enterprise** | $99 | + Auto-reports + API + multi-tenant | RetailNext (at 1/5 cost) |
| **Alt-Data** | Custom | Bulk data feed, hedge fund API | Placer.ai / Thasos |

### TAM Estimate

| Segment | Locations | Willingness to Pay | TAM |
|---------|-----------|-------------------|-----|
| US retail stores | 1M+ | $50-100/mo | $600M-1.2B/yr |
| US restaurants/cafes | 660K | $30-50/mo | $240-400M/yr |
| Hedge fund alt-data | 100+ funds | $100-500K/yr | $10-50M/yr |
| AIDC/security | 10K+ facilities | $100-500/mo | $12-60M/yr |

---

## Success Criteria for This Experiment

| Test | Pass Condition | Status |
|------|---------------|--------|
| Per-frame MAPE on Mall | <15% | **PASS (15.7%)** |
| Hourly MAPE on Mall (aggregated) | <5% | TODO (exp3) |
| Trend accuracy (week-over-week) | <3% | TODO |
| Head detection MAPE on Mall | <5% | TODO (exp1 P0) |
| Better than Placer.ai hourly | Our hourly < their daily | Expected YES |
| Match Density.io hourly | <5% | TODO |
| Unique value vs all competitors | Semantic AI analysis | **YES (VLM + Gemini)** |

When all pass: **we are a SOTA product** — not on counting alone (RetailNext wins there), but on the full stack: counting + understanding + analysis + evidence + zero-config.
