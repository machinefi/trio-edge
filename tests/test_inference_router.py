import base64
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

import trio_core.api.routers.inference as inference
from trio_core.engine import InferenceMetrics, VideoResult


def teardown_function():
    inference._vlm_engine = None


def test_get_vlm_does_not_cache_failed_load():
    broken_engine = MagicMock()
    broken_engine.load.side_effect = RuntimeError("load failed")

    with patch("trio_core.engine.TrioCore", return_value=broken_engine):
        with pytest.raises(RuntimeError, match="load failed"):
            inference._get_vlm()

    assert inference._vlm_engine is None


def test_align_crop_to_model_upscales_to_valid_multiple():
    crop = np.zeros((31, 56, 3), dtype=np.uint8)

    aligned = inference._align_crop_to_model(crop, image_factor=32)

    assert aligned.shape == (32, 64, 3)


def test_normalize_entities_converts_string_vehicles_to_dicts():
    entities = inference._normalize_entities({"vehicles": ["silver Toyota sedan"]})

    assert entities["vehicles"] == [
        {
            "description": "silver Toyota sedan",
            "make": "silver Toyota sedan",
            "brand": "silver Toyota sedan",
        }
    ]


def test_normalize_entities_converts_string_persons_to_dicts():
    entities = inference._normalize_entities({"persons": ["male 30s blue jacket"]})

    assert entities["persons"] == [
        {
            "description": "male 30s blue jacket",
            "attire": "male 30s blue jacket",
            "role": "unknown",
        }
    ]


def _image_b64(width: int = 120, height: int = 80) -> str:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:] = (20, 40, 60)
    ok, buf = cv2.imencode(".jpg", image)
    assert ok
    return base64.b64encode(buf.tobytes()).decode()


@pytest.mark.asyncio
async def test_crop_describe_uses_single_composite_vlm_call(monkeypatch):
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    engine.analyze_frame.return_value = VideoResult(
        text=(
            "CROPS:\n"
            "A: person: male in blue hoodie carrying backpack\n"
            "SCENE: A person is near the entrance.\n"
            "ACTIVITIES: The person is walking toward the doorway.\n"
            "NOTABLE: nothing unusual\n"
            'JSON: {"people_count":1,"vehicle_count":0,'
            '"persons":[{"appearance":"blue hoodie","action":"walking"}],'
            '"vehicles":[],"scene_type":"front_porch",'
            '"activity_level":"quiet","mood":"calm"}'
        ),
        metrics=InferenceMetrics(latency_ms=100.0),
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(
        image_b64=_image_b64(),
        crops=[
            {
                "bbox": [10, 10, 60, 70],
                "class": "person",
                "confidence": 0.91,
            }
        ],
        max_crops=1,
    )

    response = await inference._crop_describe_inner(req)

    assert engine.analyze_frame.call_count == 1
    frame_arg = engine.analyze_frame.call_args.args[0]
    assert frame_arg.shape[0] == 3
    assert frame_arg.shape[2] > 120
    assert response.crop_descriptions == ["person: male in blue hoodie carrying backpack"]
    assert response.entities["people_count"] == 1


@pytest.mark.asyncio
async def test_crop_describe_max_crops_zero_keeps_single_full_frame(monkeypatch):
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    engine.analyze_frame.return_value = VideoResult(
        text=(
            "SCENE: A quiet driveway.\n"
            "ACTIVITIES: No visible movement.\n"
            "NOTABLE: nothing unusual\n"
            'JSON: {"people_count":0,"vehicle_count":0,'
            '"persons":[],"vehicles":[],"scene_type":"driveway",'
            '"activity_level":"quiet","mood":"calm"}'
        ),
        metrics=InferenceMetrics(latency_ms=100.0),
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(
        image_b64=_image_b64(),
        crops=[
            {
                "bbox": [10, 10, 60, 70],
                "class": "person",
                "confidence": 0.91,
            }
        ],
        max_crops=0,
    )

    response = await inference._crop_describe_inner(req)

    assert engine.analyze_frame.call_count == 1
    frame_arg = engine.analyze_frame.call_args.args[0]
    assert frame_arg.shape == (3, 80, 120)
    assert response.crop_descriptions == []
    assert response.entities["scene_type"] == "driveway"
