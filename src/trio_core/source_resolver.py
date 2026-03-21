"""Video source resolver — auto-detect and normalize video source URLs.

Supports: RTSP, YouTube Live, HLS/m3u8, HTTP streams, USB webcams, local files.
"""

from __future__ import annotations

import logging
import re
import shutil
import subprocess

logger = logging.getLogger("trio.source")

# YouTube URL patterns
_YT_PATTERNS = [
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+"),
    re.compile(r"(?:https?://)?(?:www\.)?youtube\.com/live/[\w-]+"),
    re.compile(r"(?:https?://)?youtu\.be/[\w-]+"),
]


def detect_source_type(source: str) -> str:
    """Detect the type of video source.

    Returns: 'rtsp', 'youtube', 'hls', 'http', 'webcam', 'file'
    """
    s = source.strip()

    if s.startswith("rtsp://") or s.startswith("rtsps://"):
        return "rtsp"

    if any(p.match(s) for p in _YT_PATTERNS):
        return "youtube"

    if s.endswith(".m3u8") or "/hls/" in s.lower():
        return "hls"

    if s.startswith("http://") or s.startswith("https://"):
        return "http"

    if s.isdigit():
        return "webcam"

    return "file"


def resolve_source(source: str) -> str:
    """Resolve a source URL to a format ffmpeg/cv2 can open.

    For YouTube, extracts the actual stream URL via yt-dlp.
    For everything else, returns as-is.
    """
    source_type = detect_source_type(source)

    if source_type == "youtube":
        return _resolve_youtube(source)

    return source


def _resolve_youtube(url: str) -> str:
    """Extract direct stream URL from YouTube using yt-dlp."""
    if not shutil.which("yt-dlp"):
        raise RuntimeError(
            "yt-dlp is required for YouTube streams. Install with: brew install yt-dlp"
        )

    try:
        result = subprocess.run(
            ["yt-dlp", "--get-url", "-f", "best[height<=720]", "--no-warnings", url],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            # Try without format filter
            result = subprocess.run(
                ["yt-dlp", "--get-url", "-f", "best", "--no-warnings", url],
                capture_output=True,
                text=True,
                timeout=30,
            )

        stream_url = result.stdout.strip().split("\n")[0]
        if not stream_url:
            raise RuntimeError(f"yt-dlp returned no URL. stderr: {result.stderr[:200]}")

        logger.info("YouTube resolved: %s -> %s...", url, stream_url[:80])
        return stream_url

    except subprocess.TimeoutExpired:
        raise RuntimeError("yt-dlp timed out resolving YouTube URL")


def source_info(source: str) -> dict:
    """Return metadata about a source."""
    source_type = detect_source_type(source)
    info = {
        "original": source,
        "type": source_type,
        "resolved": source,
    }

    if source_type == "youtube":
        # Extract video ID
        for pattern in [r"v=([\w-]+)", r"live/([\w-]+)", r"youtu\.be/([\w-]+)"]:
            m = re.search(pattern, source)
            if m:
                info["video_id"] = m.group(1)
                break

    return info
