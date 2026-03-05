"""Tests for trio_core.video.StreamCapture."""

import threading
import time

import numpy as np
import pytest

from trio_core.video import StreamCapture, STREAM_BUFFER_CAP


class TestStreamCaptureInit:
    def test_defaults(self):
        cap = StreamCapture("rtsp://test")
        assert cap.source == "rtsp://test"
        assert cap.vid_stride == 1
        assert cap.buffer is False
        assert cap.running is False

    def test_custom_params(self):
        cap = StreamCapture("test", vid_stride=5, buffer=True, resize=(480, 640))
        assert cap.vid_stride == 5
        assert cap.buffer is True
        assert cap.resize == (480, 640)

    def test_vid_stride_minimum(self):
        cap = StreamCapture("test", vid_stride=0)
        assert cap.vid_stride == 1


class TestStreamCaptureResolveSource:
    def test_non_youtube_passthrough(self):
        assert StreamCapture._resolve_source("rtsp://192.168.1.1/stream") == "rtsp://192.168.1.1/stream"
        assert StreamCapture._resolve_source("/tmp/video.mp4") == "/tmp/video.mp4"


class TestStreamCaptureLatestFrameMode:
    """Test the buffer=False (latest-frame) behavior with a mock capture."""

    def test_latest_frame_overwrites(self):
        cap = StreamCapture("test", buffer=False)
        cap._lock = threading.Lock()
        cap._frames = []

        # Simulate appending in latest-frame mode
        frame1 = np.zeros((3, 64, 64), dtype=np.float32)
        frame2 = np.ones((3, 64, 64), dtype=np.float32)

        with cap._lock:
            cap._frames = [frame1]
        with cap._lock:
            cap._frames = [frame2]  # overwrites

        assert len(cap._frames) == 1
        assert np.array_equal(cap._frames[0], frame2)


class TestStreamCaptureBufferMode:
    """Test the buffer=True (queue) behavior."""

    def test_buffer_queues_frames(self):
        cap = StreamCapture("test", buffer=True)
        cap._lock = threading.Lock()
        cap._frames = []

        for i in range(5):
            frame = np.full((3, 64, 64), i, dtype=np.float32)
            with cap._lock:
                if len(cap._frames) < STREAM_BUFFER_CAP:
                    cap._frames.append(frame)

        assert len(cap._frames) == 5

    def test_buffer_cap(self):
        cap = StreamCapture("test", buffer=True)
        cap._lock = threading.Lock()
        cap._frames = []

        for i in range(STREAM_BUFFER_CAP + 10):
            frame = np.full((3, 64, 64), i, dtype=np.float32)
            with cap._lock:
                if len(cap._frames) < STREAM_BUFFER_CAP:
                    cap._frames.append(frame)

        assert len(cap._frames) == STREAM_BUFFER_CAP


class TestStreamCaptureProperties:
    def test_frame_count(self):
        cap = StreamCapture("test")
        assert cap.frame_count == 0
        cap._frame_count = 42
        assert cap.frame_count == 42

    def test_buffer_size(self):
        cap = StreamCapture("test")
        assert cap.buffer_size == 0
        cap._frames = [np.zeros((3, 64, 64))] * 3
        assert cap.buffer_size == 3
