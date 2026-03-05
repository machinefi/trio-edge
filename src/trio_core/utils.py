"""Hashing, similarity, and content extraction utilities."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np


def compute_frame_hash(frame: np.ndarray) -> str:
    """SHA-256 hash of a raw frame array (first 16 hex chars)."""
    return hashlib.sha256(frame.tobytes()).hexdigest()[:16]


def compute_video_hash(frames: np.ndarray) -> str:
    """Hash a (T,C,H,W) video tensor by combining per-frame hashes."""
    h = hashlib.sha256()
    for i in range(frames.shape[0]):
        h.update(frames[i].tobytes())
    return h.hexdigest()[:16]


def compute_file_hash(path: str | Path) -> str:
    """SHA-256 hash of file contents (first 16 hex chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def frame_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Normalized L2 similarity between two frames (0=different, 1=identical).

    Downscales to 64x64 grayscale for speed.
    """
    a_small = _downscale(a)
    b_small = _downscale(b)
    a_flat = a_small.astype(np.float32).ravel()
    b_flat = b_small.astype(np.float32).ravel()
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0 or norm_b == 0:
        return 1.0 if norm_a == norm_b else 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _downscale(frame: np.ndarray, size: int = 64) -> np.ndarray:
    """Downscale a frame to size x size grayscale via averaging.

    Input: (C, H, W) or (H, W, C) or (H, W).
    Output: (size, size) float32.
    """
    if frame.ndim == 3:
        if frame.shape[0] in (1, 3, 4):  # (C, H, W)
            frame = frame.mean(axis=0)
        else:  # (H, W, C)
            frame = frame.mean(axis=2)
    h, w = frame.shape[:2]
    bh, bw = max(1, h // size), max(1, w // size)
    cropped = frame[: bh * size, : bw * size]
    return cropped.reshape(size, bh, size, bw).mean(axis=(1, 3)).astype(np.float32)


def extract_video_content(messages: list[dict[str, Any]]) -> list[str]:
    """Extract video paths/URLs from OpenAI-format messages."""
    videos: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "video":
                    video_src = part.get("video") or part.get("video_url", {}).get("url")
                    if video_src:
                        videos.append(video_src)
    return videos


def extract_text_content(messages: list[dict[str, Any]]) -> str:
    """Extract concatenated text from OpenAI-format messages."""
    texts: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    texts.append(part.get("text", ""))
    return " ".join(texts)
