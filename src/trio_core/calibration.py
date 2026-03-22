"""Cloud-assisted counting calibration.

Uses a cloud vision model (Gemini) to produce ground truth counts
on a few sample frames, then calibrates the local YOLO correction factor.

Cost: ~$0.01 per frame (Gemini Flash vision)
Usage: 5 frames on setup + 1 frame every 4-6 hours = ~$1.20/camera/month

Flow:
    1. capture_calibration_frames() — grab N frames from camera
    2. get_cloud_counts() — send to Gemini for accurate counting
    3. compute_correction() — compare cloud vs YOLO, derive factor
    4. apply to PeopleCounter.correction_factor
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger("trio.calibration")

GEMINI_MODEL = "gemini-2.0-flash"

# Prompt designed for accurate counting — tell the model to be systematic
COUNT_PROMPT = (
    "Count the EXACT number of people visible in this image. "
    "Be systematic: scan left to right, top to bottom. "
    "Count partially visible people too. "
    "Reply with ONLY a single integer number, nothing else. "
    "Example: 23"
)


@dataclass
class CalibrationResult:
    """Result of a calibration run."""
    correction_factor: float
    cloud_counts: list[int]
    yolo_counts: list[int]
    frames_used: int
    cloud_model: str
    avg_cloud: float
    avg_yolo: float
    confidence: float  # 0-1, based on consistency of ratio across frames


def get_cloud_count(frame_bgr: np.ndarray, api_key: str | None = None) -> int | None:
    """Send a frame to Gemini and get an accurate people count.

    Cost: ~$0.01 per call (Gemini Flash vision input pricing).
    Returns None if the API call fails.
    """
    api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — cannot calibrate")
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        # Convert BGR frame to JPEG bytes
        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        image_bytes = jpeg.tobytes()

        # Send to Gemini with image
        import base64
        b64_image = base64.b64encode(image_bytes).decode()

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                {
                    "parts": [
                        {"text": COUNT_PROMPT},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": b64_image,
                            }
                        },
                    ]
                }
            ],
            config={"temperature": 0.1, "max_output_tokens": 10},
        )

        text = response.text.strip() if response.text else ""
        nums = re.findall(r"\d+", text)
        if nums:
            count = int(nums[0])
            logger.info("Cloud count: %d (raw response: %r)", count, text)
            return count
        else:
            logger.warning("Cloud returned non-numeric: %r", text)
            return None

    except Exception as e:
        logger.error("Cloud counting failed: %s", e)
        return None


def calibrate_counter(
    counter,  # PeopleCounter instance
    frames: list[np.ndarray],
    api_key: str | None = None,
) -> CalibrationResult:
    """Run calibration: compare cloud counts vs YOLO counts on sample frames.

    Args:
        counter: PeopleCounter with YOLO detector
        frames: list of BGR frames to calibrate on (recommend 3-5)
        api_key: Gemini API key (or from env)

    Returns:
        CalibrationResult with the computed correction factor
    """
    cloud_counts: list[int] = []
    yolo_counts: list[int] = []

    for i, frame in enumerate(frames):
        # Get cloud count
        cloud = get_cloud_count(frame, api_key)
        if cloud is None or cloud == 0:
            continue

        # Get YOLO count on same frame
        # Reset tracker for independent per-frame count
        counter._seen_ids.clear()
        counter._tracker = None
        counter._initialized = False
        result = counter.process(frame)
        yolo = result.by_class.get("person", 0)

        if yolo == 0:
            # YOLO detected nothing — can't compute ratio
            # Still useful: cloud says there are people YOLO missed
            logger.info("Frame %d: cloud=%d, YOLO=0 (skipped for ratio)", i, cloud)
            continue

        cloud_counts.append(cloud)
        yolo_counts.append(yolo)
        logger.info("Frame %d: cloud=%d, YOLO=%d, ratio=%.2f", i, cloud, yolo, cloud / yolo)

    if len(cloud_counts) < 2:
        logger.warning("Not enough valid calibration samples (%d). Using factor 1.0", len(cloud_counts))
        return CalibrationResult(
            correction_factor=1.0,
            cloud_counts=cloud_counts,
            yolo_counts=yolo_counts,
            frames_used=len(cloud_counts),
            cloud_model=GEMINI_MODEL,
            avg_cloud=0,
            avg_yolo=0,
            confidence=0,
        )

    # Compute correction factor as median ratio
    ratios = [c / y for c, y in zip(cloud_counts, yolo_counts)]
    ratios.sort()
    mid = len(ratios) // 2
    correction_factor = ratios[mid]  # median is robust to outliers

    # Confidence: how consistent are the ratios? (lower std = higher confidence)
    std = float(np.std(ratios))
    mean = float(np.mean(ratios))
    cv = std / mean if mean > 0 else 1.0  # coefficient of variation
    confidence = max(0, min(1, 1.0 - cv))  # 1.0 = perfectly consistent

    # Apply to counter
    counter.correction_factor = correction_factor
    logger.info(
        "Calibration complete: factor=%.2fx, confidence=%.0f%%, "
        "cloud_avg=%.1f, yolo_avg=%.1f, %d frames",
        correction_factor, confidence * 100,
        np.mean(cloud_counts), np.mean(yolo_counts), len(cloud_counts),
    )

    return CalibrationResult(
        correction_factor=round(correction_factor, 3),
        cloud_counts=cloud_counts,
        yolo_counts=yolo_counts,
        frames_used=len(cloud_counts),
        cloud_model=GEMINI_MODEL,
        avg_cloud=round(float(np.mean(cloud_counts)), 1),
        avg_yolo=round(float(np.mean(yolo_counts)), 1),
        confidence=round(confidence, 3),
    )
