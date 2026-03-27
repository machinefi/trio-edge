"""Command handlers for OpenClaw node invocations.

Bridges gateway invoke requests to trio-core engine + camera capture.
No HTTP — direct Python function calls.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .protocol import InvokeRequest, InvokeResult, generate_id, make_req

logger = logging.getLogger("trio.claw")

# RTSP failure thresholds
CAPTURE_WARN_THRESHOLD = 5  # consecutive failures before WARNING + camera.offline event
CAPTURE_STOP_THRESHOLD = 20  # consecutive failures before stopping the watch


@dataclass
class WatchTask:
    """Tracks a running vision.watch loop."""

    watch_id: str
    condition: str
    interval: float
    temporal: bool = False
    task: asyncio.Task | None = None
    stop_event: asyncio.Event = field(default_factory=asyncio.Event)
    checks: int = 0
    alerts: int = 0
    transitions: int = 0  # state transitions detected (temporal mode)
    consecutive_failures: int = 0
    last_state: str | None = None  # last detected state (temporal mode)


@dataclass
class NodeMetrics:
    """Lightweight metrics for health/Prometheus endpoint."""

    watch_checks: int = 0
    watch_alerts: int = 0
    capture_failures: int = 0
    vlm_latency_samples: list = field(default_factory=list)  # last 1000 latencies in seconds

    def record_vlm_latency(self, seconds: float) -> None:
        self.vlm_latency_samples.append(seconds)
        if len(self.vlm_latency_samples) > 1000:
            self.vlm_latency_samples = self.vlm_latency_samples[-1000:]


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
        self._watches: dict[str, WatchTask] = {}
        self._ws = None  # WebSocket ref, set by node.py before dispatch
        self._max_watches = 5
        self._metrics = NodeMetrics()

    async def handle(self, req: InvokeRequest) -> InvokeResult:
        """Dispatch invoke request to appropriate handler."""
        self.node_id = req.node_id

        handlers = {
            "camera.snap": self._camera_snap,
            "camera.list": self._camera_list,
            "vision.analyze": self._vision_analyze,
            "vision.watch": self._vision_watch,
            "vision.watch.stop": self._vision_watch_stop,
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

        return self._ok(
            req,
            {
                "format": "jpeg",
                "base64": b64,
                "width": frame.shape[1],
                "height": frame.shape[0],
            },
        )

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
        elapsed = time.monotonic() - t0
        elapsed_ms = elapsed * 1000
        self._metrics.record_vlm_latency(elapsed)
        logger.info("VLM result: %r (%.0fms)", result.text[:100], elapsed_ms)

        # Encode frame as JPEG for response
        _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        frame_b64 = base64.b64encode(jpeg.tobytes()).decode()

        return self._ok(
            req,
            {
                "answer": result.text.strip(),
                "confidence": 0.0,
                "latency_ms": round(elapsed_ms),
                "frame": {
                    "format": "jpeg",
                    "base64": frame_b64,
                    "width": frame_bgr.shape[1],
                    "height": frame_bgr.shape[0],
                },
            },
        )

    # =========================================================================
    # vision.status — report engine and watch state
    # =========================================================================

    async def _vision_status(self, req: InvokeRequest) -> InvokeResult:
        watches_list = []
        for w in self._watches.values():
            info = {
                "watchId": w.watch_id,
                "condition": w.condition,
                "interval": w.interval,
                "checks": w.checks,
                "alerts": w.alerts,
            }
            if w.temporal:
                info["temporal"] = True
                info["transitions"] = w.transitions
                info["lastState"] = w.last_state
            watches_list.append(info)
        status = {"watches": watches_list}
        if self.engine:
            health = self.engine.health()
            backend = health.get("backend", {})
            status["model"] = backend.get("model", "unknown")
            status["device"] = backend.get("device", "unknown")
            status["backend"] = backend.get("backend", "unknown")
        return self._ok(req, status)

    # =========================================================================
    # vision.watch — continuous monitoring with condition alerting
    # =========================================================================

    async def _vision_watch(self, req: InvokeRequest) -> InvokeResult:
        question = req.params.get("question", "")
        if not question:
            return self._error(req, "INVALID_PARAMS", "question is required")

        if self.engine is None:
            return self._error(req, "ENGINE_NOT_LOADED", "VLM engine not loaded")

        if self._ws is None:
            return self._error(req, "NO_WS", "No WebSocket connection for streaming results")

        if len(self._watches) >= self._max_watches:
            return self._error(req, "LIMIT_REACHED", f"Max {self._max_watches} concurrent watches")

        interval = float(req.params.get("interval", 10))
        interval = max(interval, 2.0)  # floor at 2s to avoid hammering
        temporal = bool(req.params.get("temporal", False))
        device_id = req.params.get("deviceId", "")
        source = self._resolve_source(device_id)

        # Enable StreamMem on engine for temporal mode
        if temporal:
            self._enable_streaming_memory()

        watch_id = generate_id(12)

        watch = WatchTask(
            watch_id=watch_id,
            condition=question,
            interval=interval,
            temporal=temporal,
        )
        self._watches[watch_id] = watch

        watch.task = asyncio.create_task(self._watch_loop(watch, source, req.id))

        mode_str = "temporal" if temporal else "stateless"
        logger.info(
            "vision.watch started: id=%s question=%r interval=%.0fs mode=%s",
            watch_id,
            question,
            interval,
            mode_str,
        )

        return self._ok(
            req,
            {
                "watchId": watch_id,
                "status": "started",
                "condition": question,
                "interval": interval,
                "temporal": temporal,
            },
        )

    def _enable_streaming_memory(self) -> None:
        """Enable StreamMem on the engine backend for temporal mode."""
        if self.engine is None:
            return
        backend = getattr(self.engine, "_backend", None)
        if backend is None:
            return
        if not hasattr(backend, "set_streaming_memory"):
            logger.warning("Backend does not support streaming memory — temporal mode degraded")
            return
        # Check if already enabled
        sm_config = getattr(backend, "_streaming_memory_config", None)
        if sm_config is not None:
            return  # already configured
        backend.set_streaming_memory(
            enabled=True,
            budget=6000,  # ~6K visual tokens — memory-bounded
            prototype_ratio=0.1,
            n_sink_tokens=4,
        )
        logger.info("StreamMem enabled for temporal mode (budget=6000)")

    # =========================================================================
    # vision.watch.stop — stop one or all watches
    # =========================================================================

    async def stop_all_watches(self) -> None:
        """Stop all active watches. Called by node.py on WebSocket disconnect."""
        had_temporal = any(w.temporal for w in self._watches.values())
        for w in list(self._watches.values()):
            w.stop_event.set()
            if w.task and not w.task.done():
                w.task.cancel()
        self._watches.clear()
        # Reset accumulated KV if any temporal watches were active
        if had_temporal and self.engine is not None:
            if hasattr(self.engine, "reset_context"):
                self.engine.reset_context()
        logger.info("All watches stopped (disconnect cleanup)")

    async def _vision_watch_stop(self, req: InvokeRequest) -> InvokeResult:
        watch_id = req.params.get("watchId")

        if watch_id:
            # Stop specific watch
            watch = self._watches.get(watch_id)
            if watch is None:
                return self._error(req, "NOT_FOUND", f"No watch with id {watch_id}")
            summary = await self._stop_watch(watch)
            return self._ok(req, summary)

        # Stop all watches
        summaries = []
        for w in list(self._watches.values()):
            summaries.append(await self._stop_watch(w))
        return self._ok(req, {"stopped": summaries})

    async def _stop_watch(self, watch: WatchTask) -> dict:
        """Signal a watch to stop and wait for cleanup."""
        watch.stop_event.set()
        if watch.task and not watch.task.done():
            try:
                await asyncio.wait_for(watch.task, timeout=5)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                watch.task.cancel()
        self._watches.pop(watch.watch_id, None)

        # Reset accumulated KV context only if no other temporal watches remain
        if watch.temporal and self.engine is not None:
            other_temporal = any(w.temporal for w in self._watches.values())
            if not other_temporal and hasattr(self.engine, "reset_context"):
                self.engine.reset_context()

        logger.info(
            "vision.watch stopped: id=%s checks=%d alerts=%d transitions=%d",
            watch.watch_id,
            watch.checks,
            watch.alerts,
            watch.transitions,
        )
        result = {
            "watchId": watch.watch_id,
            "status": "stopped",
            "checks": watch.checks,
            "alerts": watch.alerts,
        }
        if watch.temporal:
            result["transitions"] = watch.transitions
        return result

    # =========================================================================
    # Watch loop — periodic capture + VLM + send results via WebSocket
    # =========================================================================

    async def _watch_loop(self, watch: WatchTask, source: str, invoke_id: str) -> None:
        """Core watch loop: capture → VLM → send result → sleep → repeat.

        In temporal mode (watch.temporal=True):
          - Uses StreamMem accumulated KV to provide cross-frame context
          - Wraps the user's condition in a change-detection prompt
          - Only triggers on state transitions (not repeated detections)
        """
        try:
            while not watch.stop_event.is_set():
                t0 = time.monotonic()

                # Capture frame
                frame_bgr = await asyncio.to_thread(self._capture_frame, source)
                if frame_bgr is None:
                    watch.consecutive_failures += 1
                    self._metrics.capture_failures += 1

                    if watch.consecutive_failures >= CAPTURE_STOP_THRESHOLD:
                        logger.error(
                            "Watch %s: %d consecutive capture failures — stopping watch",
                            watch.watch_id,
                            watch.consecutive_failures,
                        )
                        await self._send_camera_offline_event(
                            source,
                            watch.watch_id,
                            watch.consecutive_failures,
                        )
                        break

                    if watch.consecutive_failures == CAPTURE_WARN_THRESHOLD:
                        logger.warning(
                            "Watch %s: %d consecutive capture failures — camera may be offline",
                            watch.watch_id,
                            watch.consecutive_failures,
                        )
                        await self._send_camera_offline_event(
                            source,
                            watch.watch_id,
                            watch.consecutive_failures,
                        )

                    logger.warning(
                        "Watch %s: capture failed (%d consecutive), retrying in %.0fs",
                        watch.watch_id,
                        watch.consecutive_failures,
                        watch.interval,
                    )
                    await self._interruptible_sleep(watch.stop_event, watch.interval)
                    continue

                # Reset failure counter on successful capture
                watch.consecutive_failures = 0

                # Build prompt — temporal mode uses change-detection wrapping
                if watch.temporal:
                    prompt = _build_temporal_prompt(watch.condition, watch.last_state)
                else:
                    prompt = watch.condition

                # VLM inference
                def _infer():
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    return self.engine.analyze_frame(rgb, prompt)

                try:
                    result = await asyncio.to_thread(_infer)
                except Exception:
                    logger.exception("Watch %s: VLM inference failed", watch.watch_id)
                    await self._interruptible_sleep(watch.stop_event, watch.interval)
                    continue

                elapsed = time.monotonic() - t0
                elapsed_ms = elapsed * 1000
                watch.checks += 1
                self._metrics.watch_checks += 1
                self._metrics.record_vlm_latency(elapsed)

                answer = result.text.strip()

                # Temporal mode: detect state transitions, not raw triggers
                if watch.temporal:
                    triggered, transition = _detect_temporal_transition(
                        answer,
                        watch.last_state,
                    )
                    if transition:
                        watch.last_state = transition["new_state"]
                    if triggered:
                        watch.transitions += 1
                        watch.alerts += 1
                        self._metrics.watch_alerts += 1
                else:
                    triggered = _detect_triggered(answer)
                    transition = None
                    if triggered:
                        watch.alerts += 1
                        self._metrics.watch_alerts += 1

                # Build result payload
                payload = {
                    "watchId": watch.watch_id,
                    "answer": answer,
                    "triggered": triggered,
                    "latency_ms": round(elapsed_ms),
                    "checks": watch.checks,
                    "alerts": watch.alerts,
                }

                if watch.temporal:
                    payload["temporal"] = True
                    payload["transitions"] = watch.transitions
                    if transition:
                        payload["transition"] = transition

                # Include frame on alerts
                if triggered:
                    _, jpeg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    payload["frame"] = base64.b64encode(jpeg.tobytes()).decode()

                # Send result back through WebSocket
                if not await self._send_watch_result(invoke_id, payload):
                    logger.info("Watch %s: stopping due to WebSocket disconnect", watch.watch_id)
                    break

                logger.info(
                    "Watch %s: check #%d triggered=%s transitions=%d (%.0fms)",
                    watch.watch_id,
                    watch.checks,
                    triggered,
                    watch.transitions,
                    elapsed_ms,
                )

                await self._interruptible_sleep(watch.stop_event, watch.interval)

        except asyncio.CancelledError:
            pass
        except Exception:
            logger.exception("Watch %s: loop crashed", watch.watch_id)
        finally:
            self._watches.pop(watch.watch_id, None)

    async def _send_camera_offline_event(
        self,
        source: str,
        watch_id: str,
        failures: int,
    ) -> None:
        """Send a camera.offline event to the Gateway."""
        if self._ws is None:
            return
        masked = _mask_url(source) if "://" in source else source
        frame = make_req(
            "node.event",
            {
                "nodeId": self.node_id,
                "event": "camera.offline",
                "payload": {
                    "camera": masked,
                    "watchId": watch_id,
                    "consecutiveFailures": failures,
                },
            },
        )
        try:
            await self._ws.send(json.dumps(frame))
            logger.info("Sent camera.offline event for %s (failures=%d)", masked, failures)
        except Exception:
            logger.warning("Failed to send camera.offline event")

    async def _send_watch_result(self, invoke_id: str, payload: dict) -> bool:
        """Send a watch check result back to the gateway via WebSocket.

        Returns False if the send failed (ws gone), signaling the loop should stop.
        """
        if self._ws is None:
            return False
        frame = make_req(
            "node.invoke.result",
            {
                "id": invoke_id,
                "nodeId": self.node_id,
                "ok": True,
                "payloadJSON": json.dumps(payload),
                "streaming": True,
            },
        )
        try:
            await self._ws.send(json.dumps(frame))
            return True
        except Exception:
            logger.warning("Watch: WebSocket send failed, stopping loop")
            return False

    @staticmethod
    async def _interruptible_sleep(event: asyncio.Event, seconds: float) -> None:
        """Sleep that can be interrupted by the stop event."""
        try:
            await asyncio.wait_for(event.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass  # Normal — event wasn't set during the sleep

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
            id=req.id,
            node_id=self.node_id,
            ok=True,
            payload=payload,
        )

    def _error(self, req: InvokeRequest, code: str, message: str) -> InvokeResult:
        return InvokeResult(
            id=req.id,
            node_id=self.node_id,
            ok=False,
            error_code=code,
            error_message=message,
        )


# =========================================================================
# Temporal mode helpers
# =========================================================================


def _build_temporal_prompt(condition: str, last_state: str | None) -> str:
    """Wrap user condition in a change-detection prompt for temporal mode.

    Instead of asking "is the door open?" every frame (stateless),
    we ask the VLM to report the current state AND whether it changed.
    The accumulated KV cache gives the model memory of previous frames.
    """
    if last_state is None:
        # First check — establish baseline state
        return (
            f'Observe this scene carefully. Regarding the condition: "{condition}"\n'
            f"Describe the current state in a few words, then answer YES or NO.\n"
            f"Format: STATE: <current state> | ANSWER: YES or NO"
        )
    # Subsequent checks — detect change from last known state
    return (
        f"You have been observing this scene over time. "
        f'The previous state was: "{last_state}"\n'
        f'Regarding: "{condition}"\n'
        f"Has the state CHANGED from the previous observation? "
        f"Describe the current state, then answer CHANGED or SAME.\n"
        f"Format: STATE: <current state> | CHANGED or SAME"
    )


def _detect_temporal_transition(
    answer: str,
    last_state: str | None,
) -> tuple[bool, dict | None]:
    """Parse temporal VLM response for state transitions.

    Returns:
        (triggered, transition_info)
        triggered: True only when state actually changed
        transition_info: {"old_state": ..., "new_state": ...} or None
    """
    lower = answer.lower().strip()

    # Extract current state from "STATE: <state> | ..."
    new_state = None
    if "state:" in lower:
        state_part = lower.split("state:")[-1]
        # Take until pipe separator or end
        if "|" in state_part:
            new_state = state_part.split("|")[0].strip()
        else:
            # Take first sentence
            new_state = state_part.split(".")[0].strip()

    # Detect change signal
    is_changed = False
    if last_state is None:
        # First observation — detect initial trigger from YES/NO
        # Check the answer portion after "ANSWER:" if present, else after "|"
        if "answer:" in lower:
            answer_part = lower.split("answer:")[-1].strip()
        elif "|" in lower:
            answer_part = lower.split("|")[-1].strip()
        else:
            answer_part = lower
        triggered = _detect_triggered(answer_part)
        if new_state:
            return (
                triggered or False,
                {
                    "old_state": None,
                    "new_state": new_state,
                },
            )
        return (triggered or False, None)

    # Look for CHANGED/SAME keywords — check the tail portion after "|"
    tail = lower.split("|")[-1].strip() if "|" in lower else lower
    negated_changed = any(
        neg in tail for neg in ("not changed", "hasn't changed", "has not changed", "no change")
    )
    if negated_changed or tail.startswith("same"):
        is_changed = False
    elif "changed" in tail:
        is_changed = True
    elif "same" in tail:
        is_changed = False
    else:
        # Fallback: compare states if we got a new one
        if new_state and new_state != last_state:
            is_changed = True

    if is_changed and new_state:
        return (
            True,
            {
                "old_state": last_state,
                "new_state": new_state,
            },
        )

    # Update state even if not changed (state description may refine)
    if new_state and new_state != last_state:
        return (False, {"old_state": last_state, "new_state": new_state})

    return (False, None)


def _detect_triggered(answer: str) -> bool | None:
    """Detect yes/no from VLM answer.

    Returns True for affirmative, False for negative, None for descriptive/ambiguous.
    """
    lower = answer.lower().strip()
    if lower.startswith(("yes", "yeah", "yep")):
        return True
    if lower.startswith(("no", "nope")):
        return False
    first_sentence = lower.split(".")[0]
    neg_patterns = (
        "there is no",
        "there are no",
        "there isn't",
        "there aren't",
        "i don't see",
        "i do not see",
        "no ",
        "not ",
        "cannot see",
        "can't see",
        "nothing",
        "nobody",
        "no one",
    )
    pos_patterns = (
        "there is a",
        "there are",
        "i see a",
        "i can see",
        "someone",
        "a person",
        "a package",
        "a delivery",
    )
    for pat in neg_patterns:
        if pat in first_sentence:
            return False
    for pat in pos_patterns:
        if pat in first_sentence:
            return True
    return None


def _mask_url(url: str) -> str:
    """Mask credentials in URL."""
    if "@" not in url:
        return url
    proto_end = url.find("://")
    if proto_end < 0:
        return url
    rest = url[proto_end + 3 :]
    at_idx = rest.find("@")
    if at_idx < 0:
        return url
    return url[: proto_end + 3] + "***:***@" + rest[at_idx + 1 :]
