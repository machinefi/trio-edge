"""People counting + crop-describe pipeline — YOLO + ByteTrack + VLM.

License-safe stack:
  - YOLOv10-nano: Apache 2.0 (THU-MIG)
  - supervision: MIT (Roboflow)
  - onnxruntime: MIT (Microsoft)

Architecture:
  YOLO (30fps) → detect + track person/car/dog/etc
      ↓ new target appears
  VLM (on-demand) → crop bounding box → describe individual target
      ↓
  Event Store: "30s male, blue suit, carrying briefcase"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("trio.counter")

# COCO class IDs we care about
COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    16: "dog",
    17: "cat",
}
TRACKED_CLASS_IDS = set(COCO_CLASSES.keys())
CONFIDENCE_THRESHOLD = 0.35


@dataclass
class CropInfo:
    """A cropped detection for VLM description."""
    track_id: int
    class_name: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    crop: np.ndarray | None = None  # BGR crop of the bounding box


@dataclass
class CountResult:
    """Result from processing a single frame."""
    people_in: int = 0
    people_out: int = 0
    total_in: int = 0  # cumulative
    total_out: int = 0  # cumulative
    current_count: int = 0  # people currently in frame
    detections: int = 0  # raw detection count this frame
    by_class: dict[str, int] = field(default_factory=dict)  # {"person": 3, "car": 1, "dog": 1}
    new_track_ids: list[int] = field(default_factory=list)  # IDs that appeared this frame
    new_crops: list[CropInfo] = field(default_factory=list)  # cropped new targets for VLM


class YOLOv10Detector:
    """ONNX-based YOLOv10 person detector."""

    def __init__(self, model_path: str, confidence: float = CONFIDENCE_THRESHOLD):
        import onnxruntime as ort

        self.confidence = confidence
        self.session = ort.InferenceSession(
            model_path,
            providers=["CoreMLExecutionProvider", "CPUExecutionProvider"],
        )
        input_info = self.session.get_inputs()[0]
        self.input_name = input_info.name
        self.input_shape = input_info.shape  # [1, 3, H, W]
        self.input_h = self.input_shape[2]
        self.input_w = self.input_shape[3]
        logger.info("YOLOv10 loaded: %s (%dx%d)", model_path, self.input_w, self.input_h)

    def detect(self, frame: np.ndarray, class_filter: set[int] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Detect objects in a frame.

        Args:
            class_filter: set of COCO class IDs to keep. None = all tracked classes.

        Returns (xyxy, confidence, class_ids) arrays.
        """
        orig_h, orig_w = frame.shape[:2]

        # Preprocess: resize, normalize, transpose
        img = cv2.resize(frame, (self.input_w, self.input_h))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # add batch

        # Inference
        outputs = self.session.run(None, {self.input_name: img})
        preds = outputs[0]  # [1, N, 6] — x1, y1, x2, y2, score, class_id

        if preds.ndim == 3:
            preds = preds[0]

        # Filter: tracked classes + confidence
        ids_to_keep = class_filter or TRACKED_CLASS_IDS
        mask = (preds[:, 4] >= self.confidence) & np.isin(preds[:, 5].astype(int), list(ids_to_keep))
        preds = preds[mask]

        if len(preds) == 0:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        # Scale boxes back to original frame size
        xyxy = preds[:, :4].copy()
        xyxy[:, [0, 2]] *= orig_w / self.input_w
        xyxy[:, [1, 3]] *= orig_h / self.input_h
        confidence = preds[:, 4]
        class_ids = preds[:, 5].astype(int)

        return xyxy, confidence, class_ids


class PeopleCounter:
    """People counter with tracking, auto-calibration, and correction factor.

    The correction factor compensates for YOLO's consistent under-detection
    at certain camera angles. It's computed by comparing YOLO count with
    a VLM count (or user-provided ground truth) during calibration.

    Production flow:
        1. Camera connects → YOLO starts counting (raw)
        2. VLM counts a few frames → correction_factor = vlm_count / yolo_count
        3. All reported counts = yolo_count × correction_factor
        4. Every 30 min: VLM re-calibrates to prevent drift
    """

    def __init__(
        self,
        model_path: str = "models/yolov10n/onnx/model.onnx",
        line_start: tuple[int, int] | None = None,
        line_end: tuple[int, int] | None = None,
        confidence: float = CONFIDENCE_THRESHOLD,
        auto_calibrate: bool = True,
        calibration_frames: int = 30,
        correction_factor: float = 1.0,
    ):
        self.detector = YOLOv10Detector(model_path, confidence)
        self._line_start = line_start
        self._line_end = line_end
        self._auto_calibrate = auto_calibrate and (line_start is None)
        self._calibration_frames = calibration_frames
        self.correction_factor = correction_factor
        self._correction_samples: list[tuple[int, int]] = []  # (yolo_count, reference_count)
        self._tracker = None
        self._line_zone = None
        self._initialized = False
        self._calibrating = False
        self._calibration_positions: list[tuple[float, float, float, float]] = []  # (cx, cy, track_id, frame_idx)
        self._cal_frame_count = 0
        self._seen_ids: set[int] = set()

    def _lazy_init(self, frame_shape: tuple[int, ...]) -> None:
        """Initialize tracker. If auto_calibrate, defer line placement."""
        import supervision as sv

        h, w = frame_shape[:2]
        self._frame_h = h
        self._frame_w = w

        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            minimum_matching_threshold=0.8,
            frame_rate=10,
        )

        if self._auto_calibrate:
            # Start in calibration mode — no line yet
            self._calibrating = True
            self._cal_frame_count = 0
            self._calibration_positions = []
            # Temporary line at center (will be replaced after calibration)
            start = sv.Point(0, h // 2)
            end = sv.Point(w, h // 2)
            self._line_zone = sv.LineZone(start=start, end=end)
            self._initialized = True
            logger.info("Auto-calibrating: observing movement for %d frames...", self._calibration_frames)
        else:
            if self._line_start is None or self._line_end is None:
                self._line_start = (0, h // 2)
                self._line_end = (w, h // 2)
            start = sv.Point(*self._line_start)
            end = sv.Point(*self._line_end)
            self._line_zone = sv.LineZone(start=start, end=end)
            self._initialized = True
            logger.info("Counter initialized: line from %s to %s, frame %dx%d",
                         self._line_start, self._line_end, w, h)

    def _finish_calibration(self) -> None:
        """Analyze collected positions and place the counting line optimally."""
        import supervision as sv

        self._calibrating = False
        h, w = self._frame_h, self._frame_w

        if len(self._calibration_positions) < 4:
            # Not enough data — use center horizontal line
            self._line_start = (0, h // 2)
            self._line_end = (w, h // 2)
            logger.info("Auto-calibration: insufficient movement data. Using center horizontal line.")
        else:
            # Group positions by track_id and compute movement vectors
            from collections import defaultdict
            tracks: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for cx, cy, tid, _ in self._calibration_positions:
                tracks[int(tid)].append((cx, cy))

            # Compute average movement direction
            dx_total, dy_total = 0.0, 0.0
            vectors = 0
            for tid, positions in tracks.items():
                if len(positions) >= 2:
                    # Movement from first to last seen position
                    dx = positions[-1][0] - positions[0][0]
                    dy = positions[-1][1] - positions[0][1]
                    dx_total += dx
                    dy_total += dy
                    vectors += 1

            if vectors == 0 or (abs(dx_total) < 5 and abs(dy_total) < 5):
                # No clear direction — use center horizontal
                self._line_start = (0, h // 2)
                self._line_end = (w, h // 2)
                logger.info("Auto-calibration: no dominant direction. Using center horizontal line.")
            else:
                # Place line perpendicular to dominant movement, through center
                # If movement is mostly horizontal (left-right): place vertical line
                # If movement is mostly vertical (up-down): place horizontal line
                if abs(dx_total) > abs(dy_total):
                    # Dominant horizontal movement → vertical counting line at center
                    self._line_start = (w // 2, 0)
                    self._line_end = (w // 2, h)
                    direction = "left-right"
                else:
                    # Dominant vertical movement → horizontal counting line at center
                    self._line_start = (0, h // 2)
                    self._line_end = (w, h // 2)
                    direction = "up-down"

                logger.info("Auto-calibration complete: dominant flow is %s (dx=%.0f, dy=%.0f, %d tracks). "
                            "Line: %s to %s",
                            direction, dx_total, dy_total, vectors,
                            self._line_start, self._line_end)

        # Replace the line zone
        start = sv.Point(*self._line_start)
        end = sv.Point(*self._line_end)
        self._line_zone = sv.LineZone(start=start, end=end)

    def process(self, frame: np.ndarray) -> CountResult:
        """Process a frame: detect, track, count line crossings."""
        import supervision as sv

        if not self._initialized:
            self._lazy_init(frame.shape)

        # Detect all tracked classes
        xyxy, confidence, class_ids = self.detector.detect(frame)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )

        # Track
        tracked = self._tracker.update_with_detections(detections)

        # Auto-calibration: collect position data during calibration phase
        if self._calibrating:
            self._cal_frame_count += 1
            if tracked.tracker_id is not None:
                for i, tid in enumerate(tracked.tracker_id):
                    cx = float((tracked.xyxy[i][0] + tracked.xyxy[i][2]) / 2)
                    cy = float((tracked.xyxy[i][1] + tracked.xyxy[i][3]) / 2)
                    self._calibration_positions.append((cx, cy, float(tid), float(self._cal_frame_count)))
            if self._cal_frame_count >= self._calibration_frames:
                self._finish_calibration()

        # Count line crossings (people only)
        people_mask = tracked.class_id == 0 if tracked.class_id is not None else np.array([])
        people_dets = tracked[people_mask] if len(people_mask) > 0 and people_mask.any() else tracked
        self._line_zone.trigger(people_dets)

        # Per-class breakdown
        by_class: dict[str, int] = {}
        if tracked.class_id is not None:
            for cid in tracked.class_id:
                name = COCO_CLASSES.get(int(cid), f"class_{cid}")
                by_class[name] = by_class.get(name, 0) + 1

        # Detect new track IDs + extract crops
        new_ids = []
        new_crops = []
        if tracked.tracker_id is not None:
            for i, tid in enumerate(tracked.tracker_id):
                if tid not in self._seen_ids:
                    self._seen_ids.add(tid)
                    new_ids.append(int(tid))
                    # Extract crop with padding
                    x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
                    h, w = frame.shape[:2]
                    # Add 10% padding
                    pad_x = int((x2 - x1) * 0.1)
                    pad_y = int((y2 - y1) * 0.1)
                    x1 = max(0, x1 - pad_x)
                    y1 = max(0, y1 - pad_y)
                    x2 = min(w, x2 + pad_x)
                    y2 = min(h, y2 + pad_y)
                    crop = frame[y1:y2, x1:x2].copy()
                    cid = int(tracked.class_id[i]) if tracked.class_id is not None else 0
                    new_crops.append(CropInfo(
                        track_id=int(tid),
                        class_name=COCO_CLASSES.get(cid, f"class_{cid}"),
                        bbox=(x1, y1, x2, y2),
                        confidence=float(tracked.confidence[i]) if tracked.confidence is not None else 0.0,
                        crop=crop,
                    ))

        # Apply correction factor to person counts
        raw_person_count = by_class.get("person", 0)
        corrected_person_count = self.corrected_count(raw_person_count)
        corrected_class = dict(by_class)
        if "person" in corrected_class:
            corrected_class["person"] = corrected_person_count

        return CountResult(
            people_in=self._line_zone.in_count,
            people_out=self._line_zone.out_count,
            total_in=self._line_zone.in_count,
            total_out=self._line_zone.out_count,
            current_count=len(tracked),
            detections=len(detections),
            by_class=corrected_class,
            new_track_ids=new_ids,
            new_crops=new_crops,
        )

    def calibrate_with_reference(self, yolo_count: int, reference_count: int) -> None:
        """Add a calibration sample (yolo_count vs reference_count from VLM or user).

        Call this periodically to maintain the correction factor.
        The correction factor is the median ratio of reference/yolo across samples.
        """
        if yolo_count > 0 and reference_count > 0:
            self._correction_samples.append((yolo_count, reference_count))
            # Keep last 20 samples
            if len(self._correction_samples) > 20:
                self._correction_samples = self._correction_samples[-20:]
            # Recompute correction factor as median ratio
            ratios = [ref / yolo for yolo, ref in self._correction_samples if yolo > 0]
            if ratios:
                ratios.sort()
                mid = len(ratios) // 2
                self.correction_factor = ratios[mid]  # median
                logger.info("Correction factor updated: %.2fx (%d samples)",
                            self.correction_factor, len(self._correction_samples))

    def corrected_count(self, raw_count: int) -> int:
        """Apply correction factor to a raw YOLO count."""
        return round(raw_count * self.correction_factor)

    @property
    def calibration_info(self) -> dict:
        """Return current calibration state."""
        return {
            "correction_factor": round(self.correction_factor, 3),
            "samples": len(self._correction_samples),
            "calibrated": self.correction_factor != 1.0,
        }

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw detections, tracks, and counting line on frame."""
        import supervision as sv

        if not self._initialized:
            return frame

        # Re-detect for annotation (or cache from last process call)
        xyxy, confidence, class_ids = self.detector.detect(frame)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )
        tracked = self._tracker.update_with_detections(detections)

        # Annotate
        annotated = frame.copy()
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        line_annotator = sv.LineZoneAnnotator(thickness=2)

        labels = [f"#{t}" for t in tracked.tracker_id] if tracked.tracker_id is not None else []
        annotated = box_annotator.annotate(annotated, tracked)
        annotated = label_annotator.annotate(annotated, tracked, labels=labels)
        annotated = line_annotator.annotate(annotated, self._line_zone)

        # Draw counts
        cv2.putText(annotated, f"IN: {self._line_zone.in_count}",
                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(annotated, f"OUT: {self._line_zone.out_count}",
                     (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return annotated

    def reset(self) -> None:
        """Reset counters."""
        if self._line_zone:
            self._line_zone.in_count = 0
            self._line_zone.out_count = 0
