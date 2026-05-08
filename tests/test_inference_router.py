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
async def test_crop_describe_uses_summary_field_from_scene_schema(monkeypatch):
    """SCENE_SCHEMA output (lowercase `summary`) should populate description.

    Previously the entities branch only looked at uppercase `DESCRIPTION` /
    `SCENE` keys and fell through to a "0 people, N vehicles: ..." parts
    template, throwing away the model's natural-language summary.
    """
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    engine.analyze_frame.return_value = VideoResult(
        text=(
            '{"summary":"A car wash area with vehicles in motion and parked.",'
            '"people_count":0,"vehicle_count":3,'
            '"persons":[],'
            '"vehicles":['
            '{"type":"sedan","color":"black","make":"unknown","action":"parked","is_known":false},'
            '{"type":"suv","color":"white","make":"toyota","action":"parked","is_known":true},'
            '{"type":"sedan","color":"red","make":"hyundai","action":"moving","is_known":false}'
            "],"
            '"scene_type":"car_wash_area","activity_level":"moderate"}'
        ),
        metrics=InferenceMetrics(latency_ms=100.0),
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(image_b64=_image_b64(), crops=[], max_crops=0)
    response = await inference._crop_describe_inner(req)

    assert response.description == "A car wash area with vehicles in motion and parked."
    assert response.entities["scene_type"] == "car_wash_area"
    # `summary` is read without popping so cortex's own override still sees it
    assert response.entities.get("summary")


@pytest.mark.asyncio
async def test_crop_describe_truncated_json_does_not_dump_raw_garbage(monkeypatch, caplog):
    """Truncated JSON (no closing brace) must not leak into description.

    Regression: the `else: desc = clean[:300]` fallback was storing 300-char
    raw JSON slices into observations.description, ending mid-token.
    """
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    truncated = (
        '{\n  "summary": "A scene description",\n'
        '  "scene_type": "car_wash_area",\n'
        '  "vehicles": [\n    {\n      "action": "parked",\n      "is_know'
    )
    engine.analyze_frame.return_value = VideoResult(
        text=truncated, metrics=InferenceMetrics(latency_ms=100.0)
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(image_b64=_image_b64(), crops=[], max_crops=0)
    with caplog.at_level("WARNING", logger="trio.inference"):
        response = await inference._crop_describe_inner(req)

    assert response.description == ""
    assert "{" not in response.description
    assert any("truncation suspected" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_crop_describe_malformed_json_does_not_dump_raw_garbage(monkeypatch, caplog):
    """JSON with a closing brace but invalid syntax also must not leak raw text."""
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    # Trailing comma — has both `{` and `}` so json.loads is attempted and fails.
    malformed = '{"summary": "x", "vehicles": [{"action": "parked",}]}'
    engine.analyze_frame.return_value = VideoResult(
        text=malformed, metrics=InferenceMetrics(latency_ms=100.0)
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(image_b64=_image_b64(), crops=[], max_crops=0)
    with caplog.at_level("WARNING", logger="trio.inference"):
        response = await inference._crop_describe_inner(req)

    assert response.description == ""
    assert any("JSON parse failed" in rec.message for rec in caplog.records)


@pytest.mark.asyncio
async def test_crop_describe_plain_prose_still_uses_300_char_slice(monkeypatch):
    """Plain prose (not JSON-shaped) should still fall through to clean[:300]."""
    engine = MagicMock()
    engine._profile = SimpleNamespace(merge_factor=32)
    engine.analyze_frame.return_value = VideoResult(
        text="A quiet street at dawn with no visible activity.",
        metrics=InferenceMetrics(latency_ms=100.0),
    )
    monkeypatch.setattr(inference, "_get_vlm", lambda: engine)

    req = inference.CropDescribeRequest(image_b64=_image_b64(), crops=[], max_crops=0)
    response = await inference._crop_describe_inner(req)

    assert response.description == "A quiet street at dawn with no visible activity."


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
