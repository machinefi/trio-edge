# Mall Dataset Benchmark — Analysis

## Results
- **MAE: 25.28** (target was <3.0) — FAILED
- **Total Accuracy: 18.9%** (target was >90%) — FAILED
- **Correlation: 0.30** (target was >0.90) — FAILED

## Root Cause
YOLOv10n is trained on **COCO dataset** (mostly eye-level photos).
Mall Dataset uses an **overhead/bird's-eye camera** mounted on the ceiling.
YOLO cannot detect people viewed from directly above — they look like circles/blobs, not the body shapes YOLO expects.

## What This Means for the Product

### Cameras that work well (eye-level/angled):
- Store entrances (side view) ✅
- Street cameras ✅
- Parking lot cameras ✅ 
- Building entrance (angled down ~30-45°) ✅
- Loading docks ✅

### Cameras that need a specialized model (overhead):
- Ceiling-mounted in malls ❌
- Directly overhead in warehouses ❌
- Bird's-eye drone views ❌

## Fix Options
1. **Use a head-detection model** for overhead cameras (e.g., FCHD, CrowdDet)
2. **Use YOLOv10 with head class** — retrain on overhead datasets
3. **Density estimation** — use CSRNet/DM-Count for dense overhead counting
4. **Tell customers honestly** — "Our system is optimized for angled cameras at entrances and corridors. For overhead ceiling cameras, we use a different detection pipeline."

## Next Steps
- Run benchmark on **MOT17** (has eye-level sequences) — expect much better results
- This proves our system needs camera angle awareness
