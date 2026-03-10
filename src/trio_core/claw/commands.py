"""Command handlers for OpenClaw node invocations.

Bridges gateway invoke requests to trio-core engine + camera capture.
No HTTP — direct Python function calls.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import time

import cv2
import numpy as np

from .protocol import InvokeRequest, InvokeResult

logger = logging.getLogger("trio.claw")


class CommandHandler:
    """Handles OpenClaw invoke commands using trio-core engine directly."""

    def __init__(self, engine=None, camera_sources: list[str] | None = None):
        """
        Args:
            engine: TrioCore engine instance (loaded). None = lazy load.
            camera_sources: List of camera sources (RTSP URLs, device indices).
                            First entry is default.
        """
        self.engine = engine
        self.camera_sources = camera_sources or ["0"]
        self.node_id = ""

    async def handle(self, req: InvokeRequest) -> InvokeResult:
        """Dispatch invoke request to appropriate handler."""
        self.node_id = req.node_id

        handlers = {
            "camera.snap": self._camera_snap,
            "camera.list": self._camera_list,
            "vision.analyze": self._vision_analyze,
            "vision.status": self._vision_status,
        }

        handler = handlers.get(req.command)
        if handler is None:
            return self._error(req, "UNAVAILABLE", f"Unknown command: {req.command}")

        return await handler(req)

    # =========================================================================
    # camera.snap — capture a single JPEG frame
    # =========================================================================

    async def _camera_snap(self, req: InvokeRequest) -> InvokeResult:
        device_id = req.params.get("deviceId", "")
        source = self._resolve_source(device_id)
        max_width = req.params.get("maxWidth", 0)
        quality = req.params.get("quality", 0.85)

        frame = await asyncio.to_thread(self._capture_frame, source)
        if frame is None:
            return self._error(req, "CAPTURE_FAILED", f"Cannot capture from {source}")

        if max_width and frame.shape[1] > max_width:
            scale = max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=scale, fy=scale)

        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality * 100)])
        b64 = base64.b64encode(jpeg.tobytes()).decode()

        return self._ok(req, {
            "format": "jpeg",
            "base64": b64,
            "width": frame.shape[1],
            "height": frame.shape[0],
        })

    # =========================================================================
    # camera.list — enumerate available cameras
    # =========================================================================

    async def _camera_list(self, req: InvokeRequest) -> InvokeResult:
        devices = []
        for i, src in enumerate(self.camera_sources):
            if src.startswith("rtsp://"):
                name = f"RTSP Camera ({_mask_url(src)})"
            elif src.isdigit():
                name = f"Camera {src}" + (" (default)" if i == 0 else "")
            else:
                name = f"Source: {src}"
            devices.append({"id": f"cam-{i}", "name": name})

        return self._ok(req, {"devices": devices})

    # =========================================================================
    # vision.analyze — capture frame + VLM inference (direct, no HTTP)
    # =========================================================================

    async def _vision_analyze(self, req: InvokeRequest) -> InvokeResult:
        question = req.params.get("question", "")
        if not question:
            return self._error(req, "INVALID_PARAMS", "question is required")

        device_id = req.params.get("deviceId", "")
        source = self._resolve_source(device_id)
        logger.info("vision.analyze: question=%r, source=%r", question, source)

        # Capture (blocking I/O → thread)
        frame_bgr = await asyncio.to_thread(self._capture_frame, source)
        if frame_bgr is None:
            logger.error("Frame capture returned None for source=%r", source)
            return self._error(req, "CAPTURE_FAILED", f"Cannot capture from {source}")

        logger.info("Frame captured: %s, dtype=%s", frame_bgr.shape, frame_bgr.dtype)

        if self.engine is None:
            logger.error("vision.analyze failed: engine is None (model not loaded)")
            return self._error(req, "ENGINE_NOT_LOADED", "VLM engine not loaded")

        # VLM inference (blocking → thread)
        logger.info("Starting VLM inference...")

        def _infer():
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            logger.info("RGB frame: %s, min=%.2f, max=%.2f", rgb.shape, rgb.min(), rgb.max())
            return self.engine.analyze_frame(rgb, question)

        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(_infer)
        except Exception as e:
            logger.exception("VLM inference failed")
            return self._error(req, "INTERNAL_ERROR", f"VLM inference error: {e}")
        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.info("VLM result: %r (%.0fms)", result.text[:100], elapsed_ms)

        # Encode frame as JPEG for response
        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode()

        return self._ok(req, {
            "answer": result.text.strip(),
            "confidence": 0.0,  # TODO: logprobs confidence
            "latency_ms": round(elapsed_ms),
            "frame": {
                "format": "jpeg",
                "base64": frame_b64,
                "width": frame_bgr.shape[1],
                "height": frame_bgr.shape[0],
            },
        })

    # =========================================================================
    # vision.status — report engine and watch state
    # =========================================================================

    async def _vision_status(self, req: InvokeRequest) -> InvokeResult:
        status = {"watches": []}
        if self.engine:
            health = self.engine.health()
            backend = health.get("backend", {})
            status["model"] = backend.get("model", "unknown")
            status["device"] = backend.get("device", "unknown")
            status["backend"] = backend.get("backend", "unknown")
        return self._ok(req, status)

    # =========================================================================
    # Camera capture helpers
    # =========================================================================

    def _resolve_source(self, device_id: str) -> str:
        """Resolve device_id to a camera source string."""
        if not device_id:
            return self.camera_sources[0] if self.camera_sources else "0"

        # Try cam-N format
        if device_id.startswith("cam-"):
            try:
                idx = int(device_id[4:])
                if 0 <= idx < len(self.camera_sources):
                    return self.camera_sources[idx]
            except ValueError:
                pass

        # Direct source (index or URL)
        return device_id

    def _capture_frame(self, source: str) -> np.ndarray | None:
        """Capture a single frame from the given source.

        Uses OpenCV VideoCapture for both local cameras and RTSP streams.
        OpenCV's built-in ffmpeg backend handles RTSP more reliably than
        shelling out to the ffmpeg CLI (avoids Tailscale/tun routing issues).
        """
        masked = _mask_url(source) if "://" in source else source
        logger.info("Capture starting: %s", masked)

        cap_source = int(source) if source.isdigit() else source
        cap = cv2.VideoCapture(cap_source)
        if not cap.isOpened():
            logger.error("VideoCapture failed to open: %s", masked)
            return None
        try:
            # Skip a few frames for camera warmup / RTSP keyframe sync
            for _ in range(3):
                cap.read()
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.error("VideoCapture read failed: %s", masked)
                return None
            logger.info("Frame captured: %s from %s", frame.shape, masked)
            return frame
        finally:
            cap.release()

    # =========================================================================
    # Result helpers
    # =========================================================================

    def _ok(self, req: InvokeRequest, payload: dict) -> InvokeResult:
        return InvokeResult(
            id=req.id, node_id=self.node_id, ok=True, payload=payload,
        )

    def _error(self, req: InvokeRequest, code: str, message: str) -> InvokeResult:
        return InvokeResult(
            id=req.id, node_id=self.node_id, ok=False,
            error_code=code, error_message=message,
        )


def _mask_url(url: str) -> str:
    """Mask credentials in URL."""
    if "@" not in url:
        return url
    proto_end = url.find("://")
    if proto_end < 0:
        return url
    rest = url[proto_end + 3:]
    at_idx = rest.find("@")
    if at_idx < 0:
        return url
    return url[:proto_end + 3] + "***:***@" + rest[at_idx + 1:]
