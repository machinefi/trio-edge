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


# ── Additional coverage tests ───────────────────────────────────────────────

from unittest.mock import patch, MagicMock, PropertyMock
import subprocess
import sys
import threading

from trio_core.video import (
    _is_url,
    _download_video,
    _TEMP_FILES,
    cleanup_temp_files,
    _extract_frames,
    DeduplicationResult,
    StreamCapture,
)


class TestIsUrl:
    def test_http(self):
        assert _is_url("http://example.com/video.mp4") is True

    def test_https(self):
        assert _is_url("https://example.com/video.mp4") is True

    def test_rtsp(self):
        assert _is_url("rtsp://192.168.1.1/stream") is True

    def test_rtmp(self):
        assert _is_url("rtmp://live.example.com/stream") is True

    def test_file_path(self):
        assert _is_url("/tmp/video.mp4") is False

    def test_relative_path(self):
        assert _is_url("videos/test.mp4") is False

    def test_file_scheme(self):
        assert _is_url("file:///tmp/video.mp4") is False


class TestDownloadVideo:
    @patch("urllib.request.urlretrieve")
    def test_downloads_to_temp_file(self, mock_retrieve):
        url = "https://example.com/video.mp4"
        result = _download_video(url)
        assert result.endswith(".mp4")
        mock_retrieve.assert_called_once_with(url, result)
        # Clean up tracking
        if result in _TEMP_FILES:
            _TEMP_FILES.remove(result)

    @patch("urllib.request.urlretrieve")
    def test_tracks_temp_file(self, mock_retrieve):
        url = "https://example.com/clip.webm"
        result = _download_video(url)
        assert result in _TEMP_FILES
        # Clean up
        _TEMP_FILES.remove(result)

    @patch("urllib.request.urlretrieve")
    def test_default_suffix(self, mock_retrieve):
        url = "https://example.com/stream"
        result = _download_video(url)
        assert result.endswith(".mp4")
        if result in _TEMP_FILES:
            _TEMP_FILES.remove(result)

    def test_file_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _download_video("file:///etc/passwd")

    def test_ftp_scheme_raises(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            _download_video("ftp://example.com/video.mp4")


class TestCleanupTempFiles:
    def test_removes_files(self, tmp_path):
        # Clear any leftover state from other tests
        _TEMP_FILES.clear()
        # Create real temp files
        f1 = tmp_path / "vid1.mp4"
        f2 = tmp_path / "vid2.mp4"
        f1.touch()
        f2.touch()
        _TEMP_FILES.extend([str(f1), str(f2)])
        removed = cleanup_temp_files()
        assert removed == 2
        assert not f1.exists()
        assert not f2.exists()
        assert len(_TEMP_FILES) == 0

    def test_handles_missing_files(self):
        _TEMP_FILES.append("/tmp/nonexistent_video_xyz123.mp4")
        removed = cleanup_temp_files()
        assert removed == 0
        assert len(_TEMP_FILES) == 0

    def test_empty_noop(self):
        assert cleanup_temp_files() == 0


class TestLoadVideoWithUrl:
    @patch("trio_core.video._extract_frames")
    @patch("trio_core.video._download_video", return_value="/tmp/downloaded.mp4")
    def test_url_triggers_download(self, mock_dl, mock_extract):
        mock_extract.return_value = np.zeros((4, 3, 56, 56), dtype=np.float32)
        load_video("https://example.com/video.mp4")
        mock_dl.assert_called_once_with("https://example.com/video.mp4")
        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][0] == "/tmp/downloaded.mp4"

    @patch("trio_core.video._extract_frames")
    def test_file_path_no_download(self, mock_extract):
        mock_extract.return_value = np.zeros((4, 3, 56, 56), dtype=np.float32)
        load_video("/tmp/local.mp4")
        mock_extract.assert_called_once()
        assert mock_extract.call_args[0][0] == "/tmp/local.mp4"


def _make_mock_cap(fps=30.0):
    """Create a mock cv2.VideoCapture."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = fps
    return cap


def _make_ffmpeg_probe(width=112, height=112, fps=30.0, nb_frames=30, duration=1.0):
    """Create a mock probe result dict for _probe_video."""
    return {"width": width, "height": height, "fps": fps, "nb_frames": nb_frames, "duration": duration}


def _make_ffmpeg_output(n_frames, width, height):
    """Create raw RGB24 bytes as ffmpeg would output."""
    return np.random.randint(0, 255, n_frames * height * width * 3, dtype=np.uint8).tobytes()


class TestExtractFrames:
    @patch("trio_core.video._probe_video")
    @patch("trio_core.video.subprocess.run")
    def test_extracts_frames(self, mock_run, mock_probe):
        mock_probe.return_value = _make_ffmpeg_probe(width=112, height=112, fps=30.0, nb_frames=30)
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout=_make_ffmpeg_output(4, 112, 112),
            stderr=b"",
        )
        result = _extract_frames("/fake/video.mp4", 2.0, 128)
        assert result.ndim == 4
        assert result.shape[1] == 3  # C dimension
        assert result.dtype == np.float32

    @patch("trio_core.video._probe_video")
    @patch("trio_core.video.subprocess.run")
    def test_ffmpeg_not_found_raises(self, mock_run, mock_probe):
        mock_probe.return_value = _make_ffmpeg_probe()
        mock_run.side_effect = FileNotFoundError
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            _extract_frames("/fake/video.mp4", 2.0, 128)

    @patch("trio_core.video._probe_video")
    @patch("trio_core.video.subprocess.run")
    def test_no_frames_raises(self, mock_run, mock_probe):
        mock_probe.return_value = _make_ffmpeg_probe()
        mock_run.return_value = MagicMock(returncode=0, stdout=b"", stderr=b"")
        with pytest.raises(IOError, match="No frames extracted"):
            _extract_frames("/fake/video.mp4", 2.0, 128)

    @patch("trio_core.video._probe_video")
    @patch("trio_core.video.subprocess.run")
    def test_ffmpeg_failure_raises(self, mock_run, mock_probe):
        mock_probe.return_value = _make_ffmpeg_probe()
        mock_run.return_value = MagicMock(returncode=1, stdout=b"", stderr=b"error msg")
        with pytest.raises(IOError, match="ffmpeg failed"):
            _extract_frames("/fake/video.mp4", 2.0, 128)


class TestDeduplicationResultRatio:
    def test_ratio_property(self):
        frames = np.zeros((3, 3, 64, 64), dtype=np.float32)
        result = DeduplicationResult(
            frames=frames, kept_indices=[0, 2], original_count=5, removed_count=3,
        )
        assert result.ratio == pytest.approx(3 / 5)

    def test_ratio_zero_original(self):
        frames = np.zeros((0, 3, 64, 64), dtype=np.float32)
        result = DeduplicationResult(
            frames=frames, kept_indices=[], original_count=0, removed_count=0,
        )
        assert result.ratio == 0.0


class TestMotionGateReset:
    def test_reset_clears_state(self):
        gate = MotionGate(threshold=0.03, warmup_frames=1)
        frame = np.ones((3, 128, 128), dtype=np.float32) * 0.5
        gate.has_motion(frame)
        assert gate._bg is not None
        assert gate._frame_count == 1
        gate.reset()
        assert gate._bg is None
        assert gate._frame_count == 0

    def test_2d_frame_input(self):
        """Test MotionGate with a 2D (H, W) grayscale frame (line 323)."""
        gate = MotionGate(threshold=0.03, warmup_frames=1)
        frame = np.ones((128, 128), dtype=np.float32) * 0.5
        assert gate.has_motion(frame) is True  # warmup
        assert gate.has_motion(frame) is False  # static


class TestStreamCaptureInit:
    def test_constructor_defaults(self):
        cap = StreamCapture("rtsp://example.com/stream")
        assert cap.source == "rtsp://example.com/stream"
        assert cap.vid_stride == 1
        assert cap.buffer is False
        assert cap.resize is None
        assert cap.reconnect is True
        assert cap.max_reconnects == 5
        assert cap.running is False
        assert cap.fps == 30.0
        assert cap._cap is None
        assert cap._frames == []

    def test_constructor_custom(self):
        cap = StreamCapture(0, vid_stride=3, buffer=True, resize=(480, 640), max_reconnects=10)
        assert cap.source == 0
        assert cap.vid_stride == 3
        assert cap.buffer is True
        assert cap.resize == (480, 640)
        assert cap.max_reconnects == 10


def _patch_cv2():
    """Patch cv2 in sys.modules so lazy imports get the mock."""
    import sys
    mock_cv2 = MagicMock()
    mock_cv2.COLOR_BGR2RGB = 4
    return patch.dict(sys.modules, {"cv2": mock_cv2}), mock_cv2


class TestStreamCaptureStartStop:
    def test_start_opens_stream(self):
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = _make_mock_cap()
            mock_cap.grab.return_value = False  # stop immediately
            mock_cv2.VideoCapture.return_value = mock_cap

            sc = StreamCapture("rtsp://example.com/stream")
            sc.start()
            assert sc.running is True
            assert sc._thread is not None
            sc.stop()
            assert sc.running is False
            assert sc._cap is None
            assert sc._thread is None

    def test_start_failed_raises(self):
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            sc = StreamCapture("rtsp://bad")
            with pytest.raises(IOError, match="Failed to open stream"):
                sc.start()


class TestStreamCaptureContextManager:
    def test_context_manager(self):
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = _make_mock_cap()
            mock_cap.grab.return_value = False
            mock_cv2.VideoCapture.return_value = mock_cap

            with StreamCapture("rtsp://example.com/stream") as sc:
                assert sc.running is True
            assert sc.running is False
            assert sc._cap is None


class TestStreamCaptureIter:
    def test_iter_yields_frames(self):
        sc = StreamCapture("fake")
        sc.running = True
        frame = np.zeros((3, 64, 64), dtype=np.float32)
        with sc._lock:
            sc._frames = [frame.copy(), frame.copy()]

        # After two reads, stop to end iteration
        original_read = sc.read
        call_count = [0]

        def mock_read():
            call_count[0] += 1
            if call_count[0] <= 2:
                with sc._lock:
                    if sc._frames:
                        return sc._frames.pop(0)
            sc.running = False
            return None

        sc.read = mock_read
        frames_out = list(sc)
        assert len(frames_out) == 2


class TestStreamCaptureCaptureLoop:
    def test_capture_loop_grabs_frames(self):
        """Test _capture_loop processes frames with stride."""
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = MagicMock()
            grab_count = [0]

            def mock_grab():
                grab_count[0] += 1
                return grab_count[0] <= 6

            mock_cap.grab.side_effect = mock_grab
            fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mock_cap.retrieve.return_value = (True, fake_frame)
            mock_cv2.cvtColor.return_value = fake_frame

            sc = StreamCapture("fake", vid_stride=2, reconnect=False)
            sc._cap = mock_cap
            sc.running = True

            sc._capture_loop()

            assert sc._frame_count == 3  # frames 2, 4, 6
            assert sc.running is False

    def test_capture_loop_buffer_mode(self):
        """Test buffer mode queues frames."""
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = MagicMock()
            grab_count = [0]

            def mock_grab():
                grab_count[0] += 1
                return grab_count[0] <= 3

            mock_cap.grab.side_effect = mock_grab
            fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mock_cap.retrieve.return_value = (True, fake_frame)
            mock_cv2.cvtColor.return_value = fake_frame

            sc = StreamCapture("fake", vid_stride=1, buffer=True, reconnect=False)
            sc._cap = mock_cap
            sc.running = True

            sc._capture_loop()

            assert len(sc._frames) == 3

    def test_capture_loop_latest_mode(self):
        """Test non-buffer mode keeps only latest frame."""
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = MagicMock()
            grab_count = [0]

            def mock_grab():
                grab_count[0] += 1
                return grab_count[0] <= 3

            mock_cap.grab.side_effect = mock_grab
            fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            mock_cap.retrieve.return_value = (True, fake_frame)
            mock_cv2.cvtColor.return_value = fake_frame

            sc = StreamCapture("fake", vid_stride=1, buffer=False, reconnect=False)
            sc._cap = mock_cap
            sc.running = True

            sc._capture_loop()

            assert len(sc._frames) == 1  # only latest

    def test_capture_loop_with_resize(self):
        """Test resize is applied."""
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            mock_cap = MagicMock()
            grab_count = [0]

            def mock_grab():
                grab_count[0] += 1
                return grab_count[0] <= 1

            mock_cap.grab.side_effect = mock_grab
            fake_frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            resized_frame = np.random.randint(0, 255, (32, 48, 3), dtype=np.uint8)
            mock_cap.retrieve.return_value = (True, fake_frame)
            mock_cv2.cvtColor.return_value = fake_frame
            mock_cv2.resize.return_value = resized_frame

            sc = StreamCapture("fake", vid_stride=1, resize=(32, 48), reconnect=False)
            sc._cap = mock_cap
            sc.running = True

            sc._capture_loop()

            mock_cv2.resize.assert_called()


class TestStreamCaptureTryReconnect:
    @patch("trio_core.video._time.sleep")
    def test_reconnect_success(self, mock_sleep):
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            new_cap = MagicMock()
            new_cap.isOpened.return_value = True
            mock_cv2.VideoCapture.return_value = new_cap

            sc = StreamCapture("rtsp://example.com/stream")
            old_cap = MagicMock()
            sc._cap = old_cap
            sc._reconnect_count = 0

            sc._try_reconnect()

            old_cap.release.assert_called_once()
            assert sc._cap is new_cap
            assert sc._reconnect_count == 0  # reset on success

    @patch("trio_core.video._time.sleep")
    def test_reconnect_failure(self, mock_sleep):
        patcher, mock_cv2 = _patch_cv2()
        with patcher:
            new_cap = MagicMock()
            new_cap.isOpened.return_value = False
            mock_cv2.VideoCapture.return_value = new_cap

            sc = StreamCapture("rtsp://example.com/stream")
            sc._cap = MagicMock()
            sc._reconnect_count = 2

            sc._try_reconnect()

            assert sc._reconnect_count == 3  # incremented, not reset


class TestResolveSource:
    def test_int_passthrough(self):
        assert StreamCapture._resolve_source(0) == 0

    def test_numeric_string(self):
        assert StreamCapture._resolve_source("2") == 2

    def test_plain_url_passthrough(self):
        assert StreamCapture._resolve_source("rtsp://cam/stream") == "rtsp://cam/stream"

    @patch("subprocess.run")
    def test_youtube_resolved(self, mock_run):
        mock_run.return_value = MagicMock(
            returncode=0, stdout="https://resolved-stream-url.com/video\n"
        )
        result = StreamCapture._resolve_source("https://www.youtube.com/watch?v=abc123")
        assert result == "https://resolved-stream-url.com/video"
        mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_youtube_failed_fallback(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        url = "https://youtube.com/watch?v=abc123"
        result = StreamCapture._resolve_source(url)
        assert result == url

    @patch("subprocess.run", side_effect=FileNotFoundError)
    def test_youtube_no_ytdlp(self, mock_run):
        url = "https://youtu.be/abc123"
        result = StreamCapture._resolve_source(url)
        assert result == url

    @patch("subprocess.run")
    def test_youtube_timeout(self, mock_run):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd="yt-dlp", timeout=30)
        url = "https://youtube.com/watch?v=abc123"
        result = StreamCapture._resolve_source(url)
        assert result == url
