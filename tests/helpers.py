"""Shared test fixtures and helpers."""

from contextlib import asynccontextmanager
from unittest.mock import MagicMock


@asynccontextmanager
async def noop_lifespan(app):
    """No-op lifespan that skips model loading."""
    yield


def make_mock_engine(**overrides):
    """Create a MagicMock engine with sensible defaults.

    Pass keyword arguments to override any attribute, e.g.
    ``make_mock_engine(**{"config.model": "custom-model"})``.
    """
    engine = MagicMock()
    engine._loaded = True
    engine.config.model = "test-model"
    engine.config.max_tokens = 512
    engine.config.temperature = 0.0
    engine.config.top_p = 1.0
    engine.config.video_fps = 2.0
    engine.config.video_max_frames = 128
    engine.config.dedup_enabled = True
    engine.config.dedup_threshold = 0.95
    engine.config.motion_enabled = False
    engine.health.return_value = {
        "status": "ok",
        "model": "test-model",
        "loaded": True,
        "config": {},
    }
    for key, val in overrides.items():
        obj = engine
        parts = key.split(".")
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], val)
    return engine
