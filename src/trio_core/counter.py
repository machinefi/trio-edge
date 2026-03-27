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
CONFIDENCE_THRESHOLD = 0.25


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
    velocity: float = 0.0  # Kalman velocity: +increasing, -decreasing (people/frame)
    kalman_confidence: float = 1.0  # 1 - coefficient_of_variation from Kalman P matrix
    raw_count: int = 0  # raw YOLO detections before smoothing/correction


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

    def detect_tiled(
        self,
        frame: np.ndarray,
        tiles: tuple[int, int] = (2, 2),
        overlap: float = 0.3,
        class_filter: set[int] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on overlapping tiles + full frame, merge with NMS.

        Finds ~50% more small/distant objects on overhead cameras vs single-pass.
        Compute: ~4-5x single detect (still real-time at ~54ms on M2 Pro).
        """
        h, w = frame.shape[:2]
        rows, cols = tiles
        step_h, step_w = h // rows, w // cols
        all_boxes, all_confs, all_classes = [], [], []

        # Run on tiles
        for r in range(rows):
            for c in range(cols):
                y1 = max(0, r * step_h - int(step_h * overlap / 2))
                x1 = max(0, c * step_w - int(step_w * overlap / 2))
                y2 = min(h, y1 + int(step_h * (1 + overlap)))
                x2 = min(w, x1 + int(step_w * (1 + overlap)))
                tile = frame[y1:y2, x1:x2]
                xyxy, conf, cids = self.detect(tile, class_filter)
                if len(xyxy) > 0:
                    xyxy[:, [0, 2]] += x1
                    xyxy[:, [1, 3]] += y1
                    all_boxes.append(xyxy)
                    all_confs.append(conf)
                    all_classes.append(cids)

        # Also run on full frame
        xyxy, conf, cids = self.detect(frame, class_filter)
        if len(xyxy) > 0:
            all_boxes.append(xyxy)
            all_confs.append(conf)
            all_classes.append(cids)

        if not all_boxes:
            return np.empty((0, 4)), np.empty(0), np.empty(0, dtype=int)

        boxes = np.vstack(all_boxes)
        confs = np.concatenate(all_confs)
        classes = np.concatenate(all_classes)

        # NMS across tiles
        keep = self._nms(boxes, confs, iou_threshold=0.5)
        return boxes[keep], confs[keep], classes[keep]

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
        """Non-maximum suppression for merging tiled detections."""
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
            iou = inter / (area_i + area_r - inter + 1e-6)
            order = rest[iou < iou_threshold]
        return np.array(keep, dtype=int)

    def warmup(self) -> float:
        """Run a dummy inference to prime JIT compilation and CPU/GPU caches.

        Uses a small black frame as input so the first real request is not penalized
        by lazy JIT compilation and cold caches.

        Returns elapsed time in seconds.
        """
        import time as _time

        t0 = _time.perf_counter()
        # 640x480 black frame — standard YOLO input size, minimal compute
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detect(dummy)
        return _time.perf_counter() - t0


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
        correction_factor: float = 1.6,
        smoothing_window: int = 7,
        tiled: bool = True,
    ):
        self.detector = YOLOv10Detector(model_path, confidence)
        self._tiled = tiled
        self._line_start = line_start
        self._line_end = line_end
        self._auto_calibrate = auto_calibrate and (line_start is None)
        self._calibration_frames = calibration_frames
        self.correction_factor = correction_factor
        self._smoothing_window = smoothing_window
        self._recent_raw_counts: list[int] = []  # ring buffer for temporal smoothing
        self._kalman = None  # lazy-init Kalman filter
        self._use_kalman = True  # prefer Kalman over SMA
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
        if self._tiled:
            xyxy, confidence, class_ids = self.detector.detect_tiled(frame)
        else:
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

        # Per-class breakdown (from tracked detections — used for non-person classes)
        by_class: dict[str, int] = {}
        if tracked.class_id is not None:
            for cid in tracked.class_id:
                name = COCO_CLASSES.get(int(cid), f"class_{cid}")
                by_class[name] = by_class.get(name, 0) + 1

        # Raw person detection count (pre-tracker) — correction factor was
        # calibrated against raw YOLO detections, not tracked detections.
        # ByteTrack smooths across frames, so tracked ≠ raw. We must apply
        # the factor to the same signal it was calibrated on.
        raw_person_detections = int((class_ids == 0).sum()) if len(class_ids) > 0 else 0

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

        # Temporal smoothing before applying correction factor.
        if self._use_kalman:
            # Kalman filter: optimal for noisy count data.
            # State = [count, velocity]. Velocity = "traffic ramping up/down".
            # Adapts gain per frame: uncertain frames weighted less.
            if self._kalman is None:
                from filterpy.kalman import KalmanFilter as KF
                kf = KF(dim_x=2, dim_z=1)
                kf.F = np.array([[1., 1.], [0., 1.]])  # count + velocity model
                kf.H = np.array([[1., 0.]])              # observe count only
                kf.R = np.array([[9.]])                   # measurement noise σ²≈3²
                kf.Q = np.array([[0.5, 0.], [0., 0.1]])  # process noise
                kf.P *= 50.
                kf.x = np.array([[float(raw_person_detections)], [0.]])
                self._kalman = kf
            self._kalman.predict()
            self._kalman.update(np.array([[float(raw_person_detections)]]))
            smoothed_raw = max(0., self._kalman.x[0, 0])
        else:
            # Fallback: simple moving average
            self._recent_raw_counts.append(raw_person_detections)
            if len(self._recent_raw_counts) > self._smoothing_window:
                self._recent_raw_counts = self._recent_raw_counts[-self._smoothing_window:]
            smoothed_raw = sum(self._recent_raw_counts) / len(self._recent_raw_counts)

        # Apply correction factor to smoothed raw person detections
        corrected_person_count = round(smoothed_raw * self.correction_factor)
        corrected_class = dict(by_class)
        corrected_class["person"] = corrected_person_count

        # Extract Kalman velocity and confidence
        velocity = 0.0
        kalman_confidence = 1.0
        if self._use_kalman and self._kalman is not None:
            velocity = float(self._kalman.x[1, 0])  # count velocity (people/frame)
            # Confidence from Kalman covariance: lower P[0,0] = higher confidence
            count_var = float(self._kalman.P[0, 0])
            if smoothed_raw > 0:
                cv = (count_var ** 0.5) / max(smoothed_raw, 1)  # coefficient of variation
                kalman_confidence = max(0.0, min(1.0, 1.0 - cv))

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
            velocity=velocity,
            kalman_confidence=kalman_confidence,
            raw_count=raw_person_detections,
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
        """Draw detections, tracks, and counting line on frame.

        IMPORTANT: Call process() first, then annotate() on the same frame.
        This method runs detection-only (no tracker update) to avoid corrupting
        ByteTrack state with a second update_with_detections() call per frame.
        """
        import supervision as sv

        if not self._initialized:
            return frame

        # Run detection only (respecting tiled config) — do NOT update tracker.
        # Updating the shared tracker here would feed a second set of detections
        # for the same logical frame, corrupting ByteTrack's ID assignments and
        # causing phantom tracks / lost IDs.
        if self._tiled:
            xyxy, confidence, class_ids = self.detector.detect_tiled(frame)
        else:
            xyxy, confidence, class_ids = self.detector.detect(frame)
        detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids,
        )

        # Annotate using raw detections (no tracker IDs — those come from process())
        annotated = frame.copy()
        box_annotator = sv.BoxAnnotator(thickness=2)
        label_annotator = sv.LabelAnnotator(text_scale=0.5, text_thickness=1)
        line_annotator = sv.LineZoneAnnotator(thickness=2)

        annotated = box_annotator.annotate(annotated, detections)
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

