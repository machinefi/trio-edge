from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import trio_core.api.routers.inference as inference


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
