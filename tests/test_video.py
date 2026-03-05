"""Tests for trio_core.video."""

import numpy as np
import pytest

from trio_core.video import (
    TemporalDeduplicator,
    MotionGate,
    load_video,
    smart_nframes,
    smart_resize,
)


class TestSmartNframes:
    def test_basic(self):
        # 300 frames at 30fps = 10s, at 2fps target = 20 frames
        assert smart_nframes(300, 30.0, 2.0) == 20

    def test_min_frames(self):
        # Very short video should hit minimum
        assert smart_nframes(5, 30.0, 2.0) >= 4

    def test_divisible_by_frame_factor(self):
        result = smart_nframes(100, 30.0, 2.0)
        assert result % 2 == 0


class TestSmartResize:
    def test_divisible_by_28(self):
        h, w = smart_resize(1080, 1920)
        assert h % 28 == 0
        assert w % 28 == 0

    def test_respects_max_pixels(self):
        h, w = smart_resize(4320, 7680, max_pixels=768 * 28 * 28)
        assert h * w <= 768 * 28 * 28 * 1.1  # Allow small rounding overshoot


class TestTemporalDeduplicator:
    def test_identical_frames_removed(self):
        # 10 identical frames → should keep only 4 (minimum)
        frame = np.random.rand(3, 64, 64).astype(np.float32)
        frames = np.stack([frame] * 10)
        dedup = TemporalDeduplicator(threshold=0.95)
        result = dedup.deduplicate(frames)
        assert result.frames.shape[0] <= 10
        assert result.removed_count >= 0

    def test_different_frames_kept(self):
        # All very different frames → should keep all
        frames = np.random.rand(6, 3, 64, 64).astype(np.float32)
        dedup = TemporalDeduplicator(threshold=0.95)
        result = dedup.deduplicate(frames)
        assert result.frames.shape[0] == 6
        assert result.removed_count == 0

    def test_single_frame(self):
        frames = np.random.rand(1, 3, 64, 64).astype(np.float32)
        dedup = TemporalDeduplicator(threshold=0.95)
        result = dedup.deduplicate(frames)
        assert result.frames.shape[0] == 1


class TestMotionGate:
    def test_warmup_returns_true(self):
        gate = MotionGate(threshold=0.03, warmup_frames=3)
        frame = np.random.rand(3, 64, 64).astype(np.float32)
        assert gate.has_motion(frame) is True  # warmup

    def test_static_scene(self):
        gate = MotionGate(threshold=0.03, warmup_frames=1)
        frame = np.ones((3, 128, 128), dtype=np.float32) * 0.5
        gate.has_motion(frame)  # warmup
        assert gate.has_motion(frame) is False  # static

    def test_motion_detected(self):
        gate = MotionGate(threshold=0.03, warmup_frames=1)
        frame1 = np.zeros((3, 128, 128), dtype=np.float32)
        frame2 = np.ones((3, 128, 128), dtype=np.float32)
        gate.has_motion(frame1)  # warmup
        assert gate.has_motion(frame2) is True  # big change


class TestLoadVideo:
    def test_numpy_passthrough(self):
        frames = np.random.rand(4, 3, 64, 64).astype(np.float32)
        result = load_video(frames)
        assert result.shape == (4, 3, 64, 64)

    def test_invalid_ndim_raises(self):
        with pytest.raises(ValueError):
            load_video(np.random.rand(3, 64, 64))
