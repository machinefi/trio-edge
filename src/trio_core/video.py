"""Video pipeline: loading, temporal deduplication, motion gating, and stream capture."""

from __future__ import annotations

import logging
import tempfile
import threading
import time as _time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence
from urllib.parse import urlparse

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ── Defaults (Qwen2.5-VL compatible) ────────────────────────────────────────

DEFAULT_IMAGE_FACTOR = 28       # Qwen2.5-VL: patch=14 × merge=2
DEFAULT_FRAME_FACTOR = 2        # temporal_patch=2 for all Qwen VL
DEFAULT_FPS = 2.0
DEFAULT_MIN_FRAMES = 4
DEFAULT_MAX_FRAMES = 128
DEFAULT_MAX_PIXELS = 768 * 28 * 28

# Module-level aliases for backward compatibility
IMAGE_FACTOR = DEFAULT_IMAGE_FACTOR
FRAME_FACTOR = DEFAULT_FRAME_FACTOR
MIN_FRAMES = DEFAULT_MIN_FRAMES
MAX_FRAMES = DEFAULT_MAX_FRAMES


# ── Video Loading ────────────────────────────────────────────────────────────


def smart_nframes(
    total_frames: int,
    native_fps: float,
    target_fps: float,
    *,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_frames: int = DEFAULT_MAX_FRAMES,
    frame_factor: int = DEFAULT_FRAME_FACTOR,
) -> int:
    """Calculate number of frames to extract, respecting frame_factor divisibility."""
    duration = total_frames / max(native_fps, 1e-6)
    nframes = max(min_frames, int(duration * target_fps))
    nframes = min(nframes, total_frames, max_frames)
    # Ensure divisible by frame_factor (temporal_patch)
    nframes = max(frame_factor, (nframes // frame_factor) * frame_factor)
    return nframes


def smart_resize(
    height: int,
    width: int,
    *,
    image_factor: int = DEFAULT_IMAGE_FACTOR,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> tuple[int, int]:
    """Resize dimensions so both are divisible by image_factor within pixel budget."""
    h = max(image_factor, round(height / image_factor) * image_factor)
    w = max(image_factor, round(width / image_factor) * image_factor)
    # Scale down if exceeding pixel budget
    if h * w > max_pixels:
        scale = (max_pixels / (h * w)) ** 0.5
        h = max(image_factor, round(h * scale / image_factor) * image_factor)
        w = max(image_factor, round(w * scale / image_factor) * image_factor)
    return h, w


def load_video(
    source: str | Path | np.ndarray,
    fps: float = DEFAULT_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    *,
    image_factor: int = DEFAULT_IMAGE_FACTOR,
    frame_factor: int = DEFAULT_FRAME_FACTOR,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """Load video from file/URL/array and return (T, C, H, W) float32 tensor.

    This is the main entry point for the video pipeline.
    All alignment params default to Qwen2.5-VL values but can be overridden
    by a ModelProfile.
    """
    if isinstance(source, np.ndarray):
        if source.ndim == 4:
            return source.astype(np.float32)
        raise ValueError(f"Expected 4D array (T,C,H,W), got shape {source.shape}")

    source = str(source)

    # URL → download to temp file
    if _is_url(source):
        source = _download_video(source)

    return _extract_frames(
        source, fps, max_frames,
        image_factor=image_factor,
        frame_factor=frame_factor,
        min_frames=min_frames,
        max_pixels=max_pixels,
    )


def _is_url(path: str) -> bool:
    parsed = urlparse(path)
    return parsed.scheme in ("http", "https", "rtsp", "rtmp")


_TEMP_FILES: list[str] = []  # Track for cleanup


def _download_video(url: str) -> str:
    """Download video URL to a temp file, return path.

    Temp files are tracked and cleaned up via cleanup_temp_files().
    """
    import atexit
    import urllib.request

    suffix = Path(urlparse(url).path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    _TEMP_FILES.append(tmp.name)
    logger.info("Downloading video: %s → %s", url, tmp.name)
    urllib.request.urlretrieve(url, tmp.name)
    return tmp.name


def cleanup_temp_files() -> int:
    """Remove all temp files created by video downloads. Returns count removed."""
    import os
    removed = 0
    while _TEMP_FILES:
        path = _TEMP_FILES.pop()
        try:
            os.unlink(path)
            removed += 1
        except OSError:
            pass
    return removed


import atexit as _atexit
_atexit.register(cleanup_temp_files)


def _extract_frames(
    path: str,
    target_fps: float,
    max_frames: int,
    *,
    image_factor: int = DEFAULT_IMAGE_FACTOR,
    frame_factor: int = DEFAULT_FRAME_FACTOR,
    min_frames: int = DEFAULT_MIN_FRAMES,
    max_pixels: int = DEFAULT_MAX_PIXELS,
) -> np.ndarray:
    """Extract frames from video file using OpenCV. Returns (T, C, H, W) float32."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        nframes = smart_nframes(
            total, native_fps, target_fps,
            min_frames=min_frames, max_frames=max_frames, frame_factor=frame_factor,
        )
        nframes = min(nframes, max_frames)

        # Compute target size using model-aware image_factor
        target_h, target_w = smart_resize(
            height, width, image_factor=image_factor, max_pixels=max_pixels,
        )

        # Evenly-spaced frame indices
        indices = np.linspace(0, total - 1, nframes, dtype=int)

        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                continue
            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Resize
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h))
            # (H, W, C) → (C, H, W), normalize to [0, 1]
            frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames.append(frame)

        if not frames:
            raise IOError(f"No frames extracted from: {path}")

        logger.info(
            "Extracted %d/%d frames from %s (%.1f fps → %.1f fps, %dx%d)",
            len(frames), total, path, native_fps, target_fps, target_w, target_h,
        )
        return np.stack(frames)  # (T, C, H, W)

    finally:
        cap.release()


# ── Temporal Deduplicator ────────────────────────────────────────────────────


@dataclass
class DeduplicationResult:
    """Result of temporal deduplication."""

    frames: np.ndarray  # (T', C, H, W) — deduplicated
    kept_indices: list[int]
    original_count: int
    removed_count: int

    @property
    def ratio(self) -> float:
        """Fraction of frames removed."""
        return self.removed_count / max(self.original_count, 1)


class TemporalDeduplicator:
    """Remove near-duplicate consecutive frames from a video tensor."""

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def deduplicate(self, frames: np.ndarray) -> DeduplicationResult:
        """Deduplicate (T, C, H, W) tensor. Always keeps first frame.

        Compares consecutive frames via normalized L2 on 64x64 downscaled versions.
        """
        from trio_core.utils import frame_similarity

        t = frames.shape[0]
        if t <= 1:
            return DeduplicationResult(
                frames=frames,
                kept_indices=list(range(t)),
                original_count=t,
                removed_count=0,
            )

        kept: list[int] = [0]
        for i in range(1, t):
            sim = frame_similarity(frames[kept[-1]], frames[i])
            if sim < self.threshold:
                kept.append(i)

        kept_frames = frames[kept]
        # Ensure minimum frame count
        if len(kept) < MIN_FRAMES and t >= MIN_FRAMES:
            # Fall back to evenly-spaced selection
            indices = np.linspace(0, t - 1, MIN_FRAMES, dtype=int).tolist()
            kept = sorted(set(indices))
            kept_frames = frames[kept]

        removed = t - len(kept)
        if removed > 0:
            logger.info("Dedup: %d → %d frames (removed %d, threshold=%.2f)", t, len(kept), removed, self.threshold)

        return DeduplicationResult(
            frames=kept_frames,
            kept_indices=kept,
            original_count=t,
            removed_count=removed,
        )


# ── Motion Gate ──────────────────────────────────────────────────────────────


class MotionGate:
    """Detect whether significant motion exists between frames.

    Uses simple frame differencing on downscaled grayscale frames.
    For monitoring: skip VLM inference entirely on static scenes.
    """

    def __init__(
        self,
        threshold: float = 0.03,
        motion_fraction: float = 0.05,
        warmup_frames: int = 3,
    ):
        self.threshold = threshold          # Per-pixel intensity change threshold
        self.motion_fraction = motion_fraction  # Fraction of pixels that must change
        self.warmup_frames = warmup_frames
        self._bg: np.ndarray | None = None
        self._frame_count = 0

    def has_motion(self, frame: np.ndarray) -> bool:
        """Check if frame has significant motion vs background.

        Args:
            frame: (C, H, W) float32 array.

        Returns:
            True if motion detected or during warmup period.

        Motion is detected when more than `motion_fraction` of pixels have
        an intensity change greater than `threshold`.
        """
        self._frame_count += 1

        # Convert to 64x64 grayscale
        if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
            gray = frame.mean(axis=0)  # (H, W)
        else:
            gray = frame

        h, w = gray.shape[:2]
        bh, bw = max(1, h // 64), max(1, w // 64)
        small = gray[: bh * 64, : bw * 64].reshape(64, bh, 64, bw).mean(axis=(1, 3))

        if self._bg is None or self._frame_count <= self.warmup_frames:
            self._bg = small
            return True  # Always process during warmup

        # Frame difference
        diff = np.abs(small - self._bg)
        motion_ratio = float(np.mean(diff > self.threshold))

        # Update background with exponential moving average
        self._bg = 0.9 * self._bg + 0.1 * small

        has = motion_ratio > self.motion_fraction
        if not has:
            logger.debug("Motion gate: no motion (ratio=%.4f)", motion_ratio)
        return has

    def reset(self) -> None:
        self._bg = None
        self._frame_count = 0


# ── Stream Capture ───────────────────────────────────────────────────────────

STREAM_BUFFER_CAP = 30
STREAM_RECONNECT_DELAY = 1.0


class StreamCapture:
    """Continuous frame capture from a live stream (RTSP, YouTube, webcam).

    Inspired by ultralytics LoadStreams: daemon thread + dual-mode buffer.

    Two modes:
        buffer=False (default): Keep only the latest frame. Best for real-time
            monitoring where you want the current state, not a backlog.
        buffer=True: Queue all frames (up to STREAM_BUFFER_CAP). Best for
            digest/summary jobs where frame ordering matters.

    Usage:
        cap = StreamCapture("rtsp://...", vid_stride=5)
        cap.start()
        for frame in cap:       # yields (C, H, W) float32 frames
            process(frame)
        cap.stop()

    Or as context manager:
        with StreamCapture(source) as cap:
            for frame in cap:
                process(frame)
    """

    def __init__(
        self,
        source: str | int,
        *,
        vid_stride: int = 1,
        buffer: bool = False,
        resize: tuple[int, int] | None = None,
        reconnect: bool = True,
        max_reconnects: int = 5,
    ):
        self.source = source
        self.vid_stride = max(1, vid_stride)
        self.buffer = buffer
        self.resize = resize  # (H, W) or None for original size
        self.reconnect = reconnect
        self.max_reconnects = max_reconnects

        self.running = False
        self.fps: float = 30.0
        self._cap: cv2.VideoCapture | None = None
        self._frames: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._frame_count = 0
        self._reconnect_count = 0

    def start(self) -> StreamCapture:
        """Open the stream and start the capture thread."""
        source = self._resolve_source(self.source)
        # int source = webcam index, str source = URL/file
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            raise IOError(f"Failed to open stream: {self.source}")

        self.fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("StreamCapture started: %s (%.1f fps, stride=%d, buffer=%s)",
                     self.source, self.fps, self.vid_stride, self.buffer)
        return self

    def stop(self) -> None:
        """Stop capture and release resources."""
        self.running = False
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        with self._lock:
            self._frames.clear()
        logger.info("StreamCapture stopped: %s", self.source)

    def read(self) -> np.ndarray | None:
        """Read the next frame (blocking). Returns (C, H, W) float32 or None if stopped."""
        while self.running:
            with self._lock:
                if self._frames:
                    return self._frames.pop(0)
            _time.sleep(1 / max(self.fps, 1))
        return None

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def buffer_size(self) -> int:
        with self._lock:
            return len(self._frames)

    def __enter__(self) -> StreamCapture:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def __iter__(self) -> Iterator[np.ndarray]:
        while self.running:
            frame = self.read()
            if frame is not None:
                yield frame

    # ── Private ──────────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        """Background capture thread. Uses grab/retrieve split for efficient stride."""
        n = 0
        while self.running and self._cap is not None:
            # grab() advances the stream without decoding — cheap
            grabbed = self._cap.grab()
            if not grabbed:
                if self.reconnect and self._reconnect_count < self.max_reconnects:
                    self._try_reconnect()
                    continue
                else:
                    logger.warning("Stream ended: %s", self.source)
                    self.running = False
                    break

            n += 1
            # Only decode every vid_stride-th frame
            if n % self.vid_stride != 0:
                continue

            ret, frame = self._cap.retrieve()
            if not ret:
                continue

            # BGR → RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Resize if requested
            if self.resize is not None:
                frame = cv2.resize(frame, (self.resize[1], self.resize[0]))

            # (H, W, C) → (C, H, W), normalize
            frame = frame.transpose(2, 0, 1).astype(np.float32) / 255.0

            with self._lock:
                if self.buffer:
                    # Buffer mode: queue frames, cap at STREAM_BUFFER_CAP
                    if len(self._frames) < STREAM_BUFFER_CAP:
                        self._frames.append(frame)
                    # Backpressure: skip frame if buffer full
                else:
                    # Latest-frame mode: always overwrite
                    self._frames = [frame]

            self._frame_count += 1

    def _try_reconnect(self) -> None:
        """Attempt to reconnect to the stream."""
        self._reconnect_count += 1
        logger.warning("Reconnecting (%d/%d): %s",
                       self._reconnect_count, self.max_reconnects, self.source)
        if self._cap is not None:
            self._cap.release()
        _time.sleep(STREAM_RECONNECT_DELAY)
        source = self._resolve_source(self.source)
        self._cap = cv2.VideoCapture(source)
        if self._cap.isOpened():
            logger.info("Reconnected: %s", self.source)
            self._reconnect_count = 0  # reset on success

    @staticmethod
    def _resolve_source(source: str | int) -> str | int:
        """Resolve source to OpenCV-compatible input.

        - int or numeric string → webcam index (int)
        - YouTube URL → resolved via yt-dlp
        - other → passthrough
        """
        # Webcam index
        if isinstance(source, int):
            return source
        if source.isdigit():
            return int(source)

        parsed = urlparse(source)
        if parsed.hostname in ("www.youtube.com", "youtube.com", "youtu.be"):
            try:
                import subprocess
                result = subprocess.run(
                    ["yt-dlp", "-f", "best", "--get-url", source],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and result.stdout.strip():
                    resolved = result.stdout.strip()
                    logger.info("Resolved YouTube URL: %s → %s", source, resolved[:80])
                    return resolved
            except (FileNotFoundError, subprocess.TimeoutExpired):
                logger.warning("yt-dlp not available, using raw URL")
        return source
