"""People counting pipeline — YOLOv10 + ByteTrack + LineZone.

License-safe stack:
  - YOLOv10-nano: Apache 2.0 (THU-MIG)
  - supervision: MIT (Roboflow)
  - onnxruntime: MIT (Microsoft)

Usage:
    counter = PeopleCounter(model_path="models/yolov10n/onnx/model.onnx")
    for frame in video_stream:
        result = counter.process(frame)
        # result.people_in, result.people_out, result.detections
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
    """People counter with tracking and line-crossing detection."""

    def __init__(
        self,
        model_path: str = "models/yolov10n/onnx/model.onnx",
        line_start: tuple[int, int] | None = None,
        line_end: tuple[int, int] | None = None,
        confidence: float = CONFIDENCE_THRESHOLD,
    ):
        self.detector = YOLOv10Detector(model_path, confidence)
        self._line_start = line_start
        self._line_end = line_end
        self._tracker = None
        self._line_zone = None
        self._initialized = False
        self._seen_ids: set[int] = set()

    def _lazy_init(self, frame_shape: tuple[int, ...]) -> None:
        """Initialize tracker and line zone on first frame."""
        import supervision as sv

        h, w = frame_shape[:2]

        # Default line: horizontal across the middle of the frame
        if self._line_start is None or self._line_end is None:
            self._line_start = (0, h // 2)
            self._line_end = (w, h // 2)

        start = sv.Point(*self._line_start)
        end = sv.Point(*self._line_end)
        self._line_zone = sv.LineZone(start=start, end=end)
        self._tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            minimum_matching_threshold=0.8,
            frame_rate=10,
        )
        self._initialized = True
        logger.info("Counter initialized: line from %s to %s, frame %dx%d",
                     self._line_start, self._line_end, w, h)

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

        # Detect new track IDs
        new_ids = []
        if tracked.tracker_id is not None:
            for tid in tracked.tracker_id:
                if tid not in self._seen_ids:
                    self._seen_ids.add(tid)
                    new_ids.append(int(tid))

        return CountResult(
            people_in=self._line_zone.in_count,
            people_out=self._line_zone.out_count,
            total_in=self._line_zone.in_count,
            total_out=self._line_zone.out_count,
            current_count=len(tracked),
            detections=len(detections),
            by_class=by_class,
            new_track_ids=new_ids,
        )

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
