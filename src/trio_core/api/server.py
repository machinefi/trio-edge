"""FastAPI server for TrioCore."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time as _time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import AsyncIterator

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware

from trio_core.api.models import (
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    DeltaContent,
    HealthResponse,
    InferenceMetricsResponse,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    Usage,
    VideoAnalyzeRequest,
    VideoAnalyzeResponse,
    WatchCondition,
    WatchConditionResult,
    WatchInfo,
    WatchMetrics,
    WatchRequest,
)
from trio_core.config import EngineConfig
from trio_core.engine import TrioCore

logger = logging.getLogger(__name__)

_engine: TrioCore | None = None
_active_requests: int = 0
_active_lock = asyncio.Lock()
_shutdown_event: asyncio.Event | None = None
_start_time: float = _time.monotonic()
_total_requests: int = 0
_total_inference_ms: float = 0.0
_total_inferences: int = 0

# Request body size limit (10 MB) — prevents OOM from oversized base64 payloads
_MAX_BODY_BYTES = 10 * 1024 * 1024


def _track_inference(latency_ms: float) -> None:
    """Record an inference for /metrics aggregation."""
    global _total_inference_ms, _total_inferences
    _total_inferences += 1
    _total_inference_ms += latency_ms


# ── Watch State ──────────────────────────────────────────────────────────────


@dataclass
class _WatchState:
    """Internal state for an active watch."""

    watch_id: str
    source: str
    conditions: list[WatchCondition]
    fps: float
    state: str = "connecting"  # connecting, running, stopped, error
    started_at: float = 0.0
    checks: int = 0
    alerts: int = 0
    stop_event: asyncio.Event | None = None
    resolution: str = "672x448"
    error: str | None = None
    camera_id: str = ""
    camera_name: str = ""
    enable_counting: bool = True


_watches: dict[str, _WatchState] = {}


def get_engine() -> TrioCore:
    if _engine is None or not _engine._loaded:
        raise HTTPException(503, "Engine not loaded")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine, _shutdown_event, _start_time
    _shutdown_event = asyncio.Event()
    _start_time = _time.monotonic()
    config = app.state.config if hasattr(app.state, "config") else EngineConfig()
    backend = getattr(app.state, "backend", None)
    _engine = TrioCore(config, backend=backend)
    logger.info("Loading model: %s", config.model)
    try:
        _engine.load()
    except Exception as e:
        logger.error("Failed to load model %s: %s", config.model, e)
        raise
    logger.info(
        "Engine ready: backend=%s", _engine._backend.backend_name if _engine._backend else "none"
    )

    # SIGHUP → reload model (same as POST /v1/admin/reload)
    import signal

    def _sighup_handler(signum, frame):
        logger.info("SIGHUP received — scheduling model reload")
        asyncio.get_event_loop().call_soon_threadsafe(
            lambda: asyncio.ensure_future(_reload_engine())
        )

    try:
        signal.signal(signal.SIGHUP, _sighup_handler)
    except (OSError, AttributeError):
        pass  # SIGHUP not available on Windows

    yield
    # Stop all active watches
    for ws in list(_watches.values()):
        if ws.stop_event:
            ws.stop_event.set()
    # Graceful shutdown: wait for in-flight requests to finish
    logger.info("Shutting down — waiting for %d active request(s)...", _active_requests)
    _shutdown_event.set()
    for _ in range(300):  # 30s max wait
        if _active_requests <= 0:
            break
        await asyncio.sleep(0.1)
    if _active_requests > 0:
        logger.warning("Shutdown with %d request(s) still active", _active_requests)
    logger.info("Shutdown complete")


class _RequestIDMiddleware(BaseHTTPMiddleware):
    """Add X-Request-ID header, enforce body size limit, and track requests."""

    async def dispatch(self, request: Request, call_next):
        global _active_requests, _total_requests

        # Reject oversized request bodies early
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={
                    "error": "payload_too_large",
                    "message": f"Request body exceeds {_MAX_BODY_BYTES // (1024 * 1024)}MB limit",
                },
            )

        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        async with _active_lock:
            _active_requests += 1
            _total_requests += 1
        try:
            logger.info("[%s] %s %s", request_id, request.method, request.url.path)
            response = await call_next(request)
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            async with _active_lock:
                _active_requests -= 1


def create_app(config: EngineConfig | None = None, backend: str | None = None) -> FastAPI:
    """Create and configure the FastAPI app."""
    app = FastAPI(
        title="TrioCore",
        description="Local VLM inference engine for video",
        version="0.3.0",
        lifespan=lifespan,
    )
    app.state.config = config or EngineConfig()
    app.state.backend = backend

    # Global exception handler — return structured JSON, never raw 500
    # Note: HTTPException is handled by FastAPI's built-in handler (correct status codes).
    # This only catches non-HTTP exceptions that would otherwise produce raw 500s.
    @app.exception_handler(Exception)
    async def _global_exception_handler(request: Request, exc: Exception):
        if isinstance(exc, HTTPException):
            raise exc  # let FastAPI handle it with correct status code
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error("[%s] Unhandled error: %s", request_id, exc, exc_info=True)
        if isinstance(exc, MemoryError):
            return JSONResponse(
                status_code=507,
                content={
                    "error": "out_of_memory",
                    "message": "Not enough memory for this request. Try fewer frames or a smaller model.",
                    "request_id": request_id,
                },
            )
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": "An internal error occurred.",
                "request_id": request_id,
            },
        )

    app.add_middleware(_RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _register_routes(app)
    return app


def _register_routes(app: FastAPI) -> None:

    @app.get("/health", response_model=HealthResponse)
    async def health():
        if _engine is None:
            return HealthResponse(status="not_loaded", model="none", loaded=False)
        return HealthResponse(**_engine.health())

    @app.get("/healthz")
    async def healthz():
        """TrioClaw-compatible health check."""
        if _engine is None or not _engine._loaded:
            raise HTTPException(503, "Engine not loaded")
        return {"status": "ok"}

    @app.post("/analyze-frame", response_model=AnalyzeFrameResponse)
    async def analyze_frame(request: AnalyzeFrameRequest):
        """TrioClaw-compatible single-frame analysis.

        Accepts base64-encoded JPEG + question, returns answer + triggered flag.
        """
        import base64
        import io
        import time as _time

        import numpy as np
        from PIL import Image

        engine = get_engine()
        t0 = _time.monotonic()

        # Decode base64 JPEG → numpy array
        jpeg_bytes = base64.b64decode(request.frame_b64)
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        frame = np.array(img, dtype=np.float32) / 255.0
        frame = frame.transpose(2, 0, 1)  # (C, H, W)
        frames = np.expand_dims(frame, axis=0)  # (1, C, H, W)

        # Run VLM inference
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.analyze_video(
                video=frames,
                prompt=request.question,
            ),
        )

        latency_ms = int((_time.monotonic() - t0) * 1000)
        _track_inference(latency_ms)
        answer = _strip_think_tags(result.text)

        # Auto-detect triggered from answer semantics
        triggered = _detect_triggered(answer)

        return AnalyzeFrameResponse(
            answer=answer,
            triggered=triggered,
            latency_ms=latency_ms,
        )

    @app.get("/v1/models", response_model=ModelListResponse)
    async def list_models():
        engine = get_engine()
        return ModelListResponse(data=[ModelInfo(id=engine.config.model)])

    @app.post("/v1/video/analyze")
    async def analyze_video(request: VideoAnalyzeRequest):
        engine = get_engine()

        if request.stream:
            return StreamingResponse(
                _stream_video_analyze(engine, request),
                media_type="text/event-stream",
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.analyze_video(
                video=request.video,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            ),
        )

        return VideoAnalyzeResponse(
            text=result.text,
            model=engine.config.model,
            metrics=InferenceMetricsResponse(**result.metrics.__dict__),
        )

    @app.post("/v1/frames/analyze")
    async def analyze_frames(
        prompt: str = Form(...),
        max_tokens: int | None = Form(None),
        temperature: float | None = Form(None),
        stream: bool = Form(False),
        frames: list[UploadFile] = File(...),
    ):
        engine = get_engine()

        import io

        import numpy as np
        from PIL import Image

        frame_arrays = []
        for f in frames:
            data = await f.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
            arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, C)
            arr = arr.transpose(2, 0, 1)  # (C, H, W)
            frame_arrays.append(arr)

        # Stack into (T, C, H, W)
        frames_array = np.stack(frame_arrays, axis=0)

        if stream:
            return StreamingResponse(
                _stream_frames_analyze(engine, frames_array, prompt, max_tokens, temperature),
                media_type="text/event-stream",
            )

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.analyze_video(
                video=frames_array,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        return VideoAnalyzeResponse(
            text=result.text,
            model=engine.config.model,
            metrics=InferenceMetricsResponse(**result.metrics.__dict__),
        )

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        engine = get_engine()

        # Extract media (video/image) and text from messages
        media, prompt = _extract_from_messages(request.messages)

        if not media:
            raise HTTPException(
                400,
                "No visual content found in messages. Use content parts with type='image_url' or 'video'.",
            )

        max_tokens = request.max_tokens or engine.config.max_tokens
        temperature = (
            request.temperature if request.temperature is not None else engine.config.temperature
        )

        # Resolve media source: base64 data URI → temp file, or path/URL as-is
        source, temp_path = _resolve_media(media[0])

        if request.stream:

            async def _stream_with_cleanup():
                try:
                    async for chunk in _stream_chat_completion(
                        engine, source, prompt, request, max_tokens, temperature
                    ):
                        yield chunk
                finally:
                    _cleanup_temp(temp_path)

            return StreamingResponse(
                _stream_with_cleanup(),
                media_type="text/event-stream",
            )

        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None,
                lambda: engine.analyze_video(
                    video=source,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                ),
            )
        finally:
            _cleanup_temp(temp_path)

        return ChatCompletionResponse(
            model=engine.config.model,
            choices=[
                ChatCompletionChoice(
                    message={"role": "assistant", "content": result.text},
                )
            ],
            usage=Usage(
                prompt_tokens=result.metrics.prompt_tokens,
                completion_tokens=result.metrics.completion_tokens,
                total_tokens=result.metrics.prompt_tokens + result.metrics.completion_tokens,
            ),
        )

    # ── Watch API (/v1/watch) ──────────────────────────────────────────────

    @app.post("/v1/watch")
    async def start_watch(request: WatchRequest, req: Request):
        """Start watching an RTSP stream. Returns SSE event stream."""
        engine = get_engine()
        watch_id = f"w_{uuid.uuid4().hex[:8]}"
        stop_event = asyncio.Event()

        ws = _WatchState(
            watch_id=watch_id,
            source=request.source,
            conditions=request.conditions,
            fps=request.fps,
            started_at=_time.time(),
            stop_event=stop_event,
            resolution=request.resolution,
            camera_id=request.camera_id,
            camera_name=request.camera_name,
            enable_counting=request.enable_counting,
        )
        _watches[watch_id] = ws

        store = getattr(req.app.state, "event_store", None)
        return StreamingResponse(
            _watch_sse_stream(engine, ws, store),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Watch-ID": watch_id,
            },
        )

    @app.get("/v1/watch")
    async def list_watches():
        """List all active watches."""
        now = _time.time()
        return [
            WatchInfo(
                watch_id=ws.watch_id,
                source=ws.source,
                state=ws.state,
                conditions=ws.conditions,
                uptime_s=int(now - ws.started_at),
                checks=ws.checks,
                alerts=ws.alerts,
            )
            for ws in _watches.values()
        ]

    @app.delete("/v1/watch/{watch_id}")
    async def stop_watch(watch_id: str):
        """Stop an active watch."""
        ws = _watches.get(watch_id)
        if ws is None:
            raise HTTPException(404, f"Watch not found: {watch_id}")
        if ws.stop_event:
            ws.stop_event.set()
        ws.state = "stopped"
        result = {
            "status": "stopped",
            "total_checks": ws.checks,
            "total_alerts": ws.alerts,
        }
        _watches.pop(watch_id, None)
        return result

    # ── Metrics & Admin ───────────────────────────────────────────────

    @app.get("/metrics")
    async def metrics():
        """Operational metrics for monitoring."""
        import os
        import resource

        now = _time.monotonic()
        uptime_s = now - _start_time

        # Process memory (RSS)
        rusage = resource.getrusage(resource.RUSAGE_SELF)
        rss_mb = rusage.ru_maxrss / (1024 * 1024)  # macOS returns bytes
        if os.uname().sysname == "Linux":
            rss_mb = rusage.ru_maxrss / 1024  # Linux returns KB

        # MLX metal memory (if available)
        metal_mb = 0.0
        try:
            import mlx.core as mx

            metal_mb = mx.metal.get_active_memory() / (1024 * 1024)
        except Exception:
            pass

        # Engine stats
        engine_info = {}
        if _engine is not None and _engine._loaded:
            engine_info = {
                "model": _engine.config.model,
                "backend": _engine._backend.backend_name if _engine._backend else "none",
                "loaded": True,
            }
        else:
            engine_info = {"model": "none", "backend": "none", "loaded": False}

        # Watch stats
        active_watches = len(_watches)
        total_watch_checks = sum(ws.checks for ws in _watches.values())
        total_watch_alerts = sum(ws.alerts for ws in _watches.values())

        return {
            "uptime_s": int(uptime_s),
            "process": {
                "rss_mb": round(rss_mb, 1),
                "metal_mb": round(metal_mb, 1),
                "pid": os.getpid(),
            },
            "requests": {
                "active": _active_requests,
                "total": _total_requests,
            },
            "inference": {
                "total": _total_inferences,
                "avg_ms": round(_total_inference_ms / max(_total_inferences, 1), 1),
            },
            "watches": {
                "active": active_watches,
                "total_checks": total_watch_checks,
                "total_alerts": total_watch_alerts,
            },
            "engine": engine_info,
        }

    @app.post("/v1/admin/reload")
    async def reload_model(
        model: str | None = None,
    ):
        """Hot-reload the engine with a new (or same) model.

        Waits for in-flight requests to drain, then reloads.
        """
        global _engine
        if _engine is None:
            raise HTTPException(503, "Engine not initialized")

        new_model = model or _engine.config.model
        logger.info("Reload requested: %s → %s", _engine.config.model, new_model)

        # Wait for in-flight inference to finish (up to 30s)
        for _ in range(300):
            if _active_requests <= 1:  # 1 = this reload request itself
                break
            await asyncio.sleep(0.1)

        config = _engine.config
        backend_name = _engine._backend_override
        config.model = new_model

        # Release old engine
        _engine._loaded = False
        old_engine = _engine

        # Create and load new engine
        _engine = TrioCore(config, backend=backend_name)
        try:
            _engine.load()
        except Exception as e:
            # Rollback on failure
            _engine = old_engine
            _engine._loaded = True
            logger.error("Reload failed, rolling back: %s", e)
            raise HTTPException(500, f"Reload failed: {e}")

        logger.info("Reload complete: %s", new_model)
        return {
            "status": "reloaded",
            "model": new_model,
            "backend": _engine._backend.backend_name if _engine._backend else "none",
        }

    # Optional console API routes (available when console extras are installed)
    try:
        from .routers import (
            alerts,
            analytics,
            auth,
            auto_alerts,
            cameras,
            chat,
            events,
            insights,
            metrics,
            report_export,
            reports,
        )

        app.include_router(auth.router)
        app.include_router(events.router)
        app.include_router(cameras.router)
        app.include_router(chat.router)
        app.include_router(alerts.router)
        app.include_router(reports.router)
        app.include_router(metrics.router)
        app.include_router(analytics.router)
        app.include_router(report_export.router)
        app.include_router(auto_alerts.router)
        app.include_router(insights.router)
    except ImportError:
        logger.debug("Console routers not available (install with trio-core[console])")


async def _reload_engine() -> None:
    """Reload the current model (used by SIGHUP handler)."""
    global _engine
    if _engine is None:
        return
    model = _engine.config.model
    logger.info("Reloading model: %s", model)
    config = _engine.config
    backend_name = _engine._backend_override
    _engine._loaded = False
    old_engine = _engine
    _engine = TrioCore(config, backend=backend_name)
    try:
        await asyncio.get_running_loop().run_in_executor(None, _engine.load)
        logger.info("Reload complete: %s", model)
    except Exception as e:
        _engine = old_engine
        _engine._loaded = True
        logger.error("Reload failed, rolled back: %s", e)


def _strip_think_tags(text: str) -> str:
    """Strip Qwen3.5 <think>...</think> reasoning blocks from VLM output."""
    # Remove <think>content</think> blocks (including multiline)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Remove any remaining orphan tags
    text = re.sub(r"</?think>", "", text)
    return text.strip()


def _detect_triggered(answer: str) -> bool | None:
    """Detect yes/no from VLM answer for trioclaw triggered flag.

    Returns True for affirmative, False for negative, None for descriptive answers.
    """
    lower = answer.lower().strip()
    # Direct yes/no start
    if lower.startswith(("yes", "yeah", "yep")):
        return True
    if lower.startswith(("no", "nope")):
        return False
    # Check for negative patterns in first sentence
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


def _extract_from_messages(messages: list[ChatMessage]) -> tuple[list[str], str]:
    """Extract video/image paths and text prompt from chat messages.

    Supports:
    - type="video" with video or video_url.url
    - type="image_url" with image_url.url (OpenAI format, file path or base64 data URI)
    - type="image" with image or image_url.url (convenience alias)
    """
    media: list[str] = []
    texts: list[str] = []

    for msg in messages:
        if isinstance(msg.content, str):
            texts.append(msg.content)
            continue
        for part in msg.content:
            if part.type == "video":
                src = part.video or (part.video_url or {}).get("url")
                if src:
                    media.append(src)
            elif part.type in ("image_url", "image"):
                src = (part.image_url or {}).get("url")
                if src:
                    media.append(src)
            elif part.type == "text" and part.text:
                texts.append(part.text)

    return media, " ".join(texts) if texts else "Describe this video."


def _resolve_media(source: str) -> tuple[str, str | None]:
    """Resolve media source to a file path.

    Handles:
    - base64 data URI (data:image/jpeg;base64,...) → temp file
    - file path or URL → returned as-is

    Returns:
        (path, temp_path) — temp_path is set if a temp file was created,
        caller should clean it up after use.
    """
    if source.startswith("data:"):
        import base64
        import tempfile

        if "," not in source:
            raise HTTPException(400, "Invalid data URI: missing comma separator")

        # Parse data URI: data:<mime>;base64,<data>
        header, data = source.split(",", 1)
        mime = header.split(":")[1].split(";")[0] if ":" in header else ""
        ext = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "video/mp4": ".mp4",
        }.get(mime, ".bin")

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(base64.b64decode(data))
        tmp.close()

        return tmp.name, tmp.name

    return source, None


def _cleanup_temp(temp_path: str | None) -> None:
    """Remove a temp file if it exists."""
    if temp_path is None:
        return
    import os

    try:
        os.unlink(temp_path)
    except OSError:
        pass


async def _stream_video_analyze(
    engine: TrioCore, request: VideoAnalyzeRequest
) -> AsyncIterator[str]:
    """Stream /v1/video/analyze response as SSE."""
    try:
        async for chunk in engine.stream_analyze(
            video=request.video,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        ):
            if chunk.get("metrics"):
                data = {"text": "", "finished": True, "metrics": chunk["metrics"].__dict__}
            else:
                data = {"text": chunk["text"], "finished": chunk["finished"]}
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as e:
        logger.error("Stream error in /v1/video/analyze: %s", e, exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_frames_analyze(
    engine: TrioCore,
    frames: "np.ndarray",
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
) -> AsyncIterator[str]:
    """Stream /v1/frames/analyze response as SSE."""
    try:
        async for chunk in engine.stream_analyze(
            video=frames,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if chunk.get("metrics"):
                data = {"text": "", "finished": True, "metrics": chunk["metrics"].__dict__}
            else:
                data = {"text": chunk["text"], "finished": chunk["finished"]}
            yield f"data: {json.dumps(data)}\n\n"
    except Exception as e:
        logger.error("Stream error in /v1/frames/analyze: %s", e, exc_info=True)
        yield f"data: {json.dumps({'error': str(e), 'finished': True})}\n\n"
    yield "data: [DONE]\n\n"


_WATCH_HEARTBEAT_INTERVAL = 30  # seconds between heartbeat events
_WATCH_MAX_RECONNECTS = 5
_WATCH_RECONNECT_DELAY = 3  # seconds


_MAX_RESOLUTION = 3840  # 4K UHD — anything larger is almost certainly a mistake


def _parse_resolution(resolution: str) -> tuple[int, int]:
    """Parse 'WxH' string into (width, height). Returns (672, 448) on error.

    Clamps each dimension to _MAX_RESOLUTION to prevent OOM from absurd values.
    """
    try:
        w, h = resolution.lower().split("x")
        w_int, h_int = int(w), int(h)
        if w_int <= 0 or h_int <= 0:
            return 672, 448
        return min(w_int, _MAX_RESOLUTION), min(h_int, _MAX_RESOLUTION)
    except (ValueError, AttributeError):
        return 672, 448


def _start_ffmpeg(effective_url: str, fps: float, frame_w: int, frame_h: int):
    """Start ffmpeg subprocess for RTSP frame reading."""
    import subprocess

    cmd = [
        "ffmpeg",
        "-v",
        "quiet",
        "-rtsp_transport",
        "tcp",
        "-i",
        effective_url,
        "-vf",
        f"fps={fps},scale={frame_w}:{frame_h}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def _kill_ffmpeg(proc) -> None:
    """Terminate an ffmpeg subprocess, close its pipes, and reap the process."""
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except Exception:
                proc.kill()
                proc.wait(timeout=2)
    except Exception:
        pass
    for pipe in (proc.stdout, proc.stderr):
        try:
            if pipe is not None:
                pipe.close()
        except Exception:
            pass


async def _watch_sse_stream(
    engine: TrioCore,
    ws: _WatchState,
    store=None,
) -> AsyncIterator[str]:
    """Core watch loop: RTSP → frames → motion gate → inference + counting → SSE events.

    Features:
    - Configurable frame resolution via WatchRequest.resolution
    - Heartbeat events every 30s so the client knows the connection is alive
    - Auto-reconnect on RTSP stream failure (up to 5 retries)
    - YOLO people counting with metrics storage (when enable_counting=True)
    """
    import base64
    import io

    from PIL import Image

    from trio_core._rtsp_proxy import ensure_rtsp_url
    from trio_core.video import MotionGate

    watch_id = ws.watch_id
    motion_gate = MotionGate()
    frame_w, frame_h = _parse_resolution(ws.resolution)
    frame_bytes = frame_w * frame_h * 3

    # Initialize people counter (YOLO + ByteTrack + Kalman)
    counter = None
    if ws.enable_counting:
        try:
            from trio_core.counter import PeopleCounter

            counter = PeopleCounter()  # uses new defaults: tiled=True, c=0.25, f=1.6
            logger.info("Watch %s: people counting enabled (tiled+Kalman)", watch_id)
        except Exception as e:
            logger.warning("Watch %s: counting unavailable: %s", watch_id, e)

    # Emit connecting status
    yield _sse_event("status", {"watch_id": watch_id, "state": "connecting"})

    # Resolve RTSP URL (handles Tailscale proxy if needed)
    loop = asyncio.get_running_loop()
    try:
        effective_url = await loop.run_in_executor(None, ensure_rtsp_url, ws.source)
    except Exception as e:
        ws.state = "error"
        ws.error = str(e)
        yield _sse_event("error", {"watch_id": watch_id, "error": f"RTSP proxy failed: {e}"})
        _watches.pop(watch_id, None)
        return

    # Start ffmpeg
    try:
        proc = await asyncio.to_thread(
            lambda: _start_ffmpeg(effective_url, ws.fps, frame_w, frame_h)
        )
    except Exception as e:
        ws.state = "error"
        ws.error = str(e)
        yield _sse_event("error", {"watch_id": watch_id, "error": f"ffmpeg failed: {e}"})
        _watches.pop(watch_id, None)
        return

    model_name = (
        engine.config.model.split("/")[-1] if "/" in engine.config.model else engine.config.model
    )

    ws.state = "running"
    ws.resolution = f"{frame_w}x{frame_h}"
    yield _sse_event(
        "status",
        {
            "watch_id": watch_id,
            "state": "running",
            "resolution": ws.resolution,
            "model": model_name,
        },
    )

    reconnect_count = 0
    last_heartbeat = _time.monotonic()

    try:
        while not ws.stop_event.is_set():
            # Read one frame from ffmpeg with timeout so heartbeats can fire
            try:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(
                        None, lambda: proc.stdout.read(frame_bytes) if proc.stdout else b""
                    ),
                    timeout=_WATCH_HEARTBEAT_INTERVAL,
                )
            except asyncio.TimeoutError:
                # Read timed out — emit heartbeat and retry
                last_heartbeat = _time.monotonic()
                yield _sse_event(
                    "heartbeat",
                    {
                        "watch_id": watch_id,
                        "ts": _iso_now(),
                        "uptime_s": int(_time.time() - ws.started_at),
                        "checks": ws.checks,
                        "alerts": ws.alerts,
                    },
                )
                continue

            # Emit heartbeat if enough time passed during normal operation
            now = _time.monotonic()
            if now - last_heartbeat >= _WATCH_HEARTBEAT_INTERVAL:
                last_heartbeat = now
                yield _sse_event(
                    "heartbeat",
                    {
                        "watch_id": watch_id,
                        "ts": _iso_now(),
                        "uptime_s": int(_time.time() - ws.started_at),
                        "checks": ws.checks,
                        "alerts": ws.alerts,
                    },
                )

            if len(raw) < frame_bytes:
                if ws.stop_event.is_set():
                    break
                # ── Auto-reconnect ──
                if reconnect_count < _WATCH_MAX_RECONNECTS:
                    reconnect_count += 1
                    logger.warning(
                        "Watch %s: RTSP stream lost, reconnecting (%d/%d)...",
                        watch_id,
                        reconnect_count,
                        _WATCH_MAX_RECONNECTS,
                    )
                    yield _sse_event(
                        "status",
                        {
                            "watch_id": watch_id,
                            "state": "reconnecting",
                            "attempt": reconnect_count,
                            "max_attempts": _WATCH_MAX_RECONNECTS,
                        },
                    )
                    # Kill old ffmpeg — terminate but don't close pipes yet,
                    # so that if reconnect fails, the retry loop reads EOF
                    # (b"") instead of raising on closed pipe.
                    old_proc = proc
                    if old_proc.poll() is None:
                        old_proc.terminate()
                        try:
                            old_proc.wait(timeout=3)
                        except Exception:
                            old_proc.kill()
                            old_proc.wait(timeout=2)
                    await asyncio.sleep(_WATCH_RECONNECT_DELAY)
                    if ws.stop_event.is_set():
                        break
                    try:
                        proc = await asyncio.to_thread(
                            lambda: _start_ffmpeg(effective_url, ws.fps, frame_w, frame_h)
                        )
                        # New proc succeeded — now safe to close old pipes
                        for pipe in (old_proc.stdout, old_proc.stderr):
                            try:
                                if pipe is not None:
                                    pipe.close()
                            except Exception:
                                pass
                        motion_gate.reset()
                        ws.state = "running"
                        yield _sse_event(
                            "status",
                            {
                                "watch_id": watch_id,
                                "state": "running",
                                "resolution": ws.resolution,
                                "reconnected": True,
                            },
                        )
                        continue
                    except Exception as e:
                        logger.error("Watch %s: reconnect failed: %s", watch_id, e)
                        continue  # will retry on next loop iteration
                else:
                    ws.state = "error"
                    ws.error = f"RTSP stream lost after {_WATCH_MAX_RECONNECTS} reconnect attempts"
                    yield _sse_event("error", {"watch_id": watch_id, "error": ws.error})
                    break

            # Successful read resets reconnect counter
            reconnect_count = 0

            # Parse frame: (H, W, 3) uint8 → (C, H, W) float32
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(frame_h, frame_w, 3)
            frame_chw = frame.transpose(2, 0, 1).astype(np.float32) / 255.0

            # Motion gate — skip inference on static scenes
            if not motion_gate.has_motion(frame_chw):
                # Still run counter on static frames (people standing still count)
                if counter is not None:
                    try:
                        frame_bgr = frame[:, :, ::-1].copy()  # RGB → BGR for OpenCV
                        count_result = await loop.run_in_executor(None, counter.process, frame_bgr)
                        if store and ws.camera_id:
                            ts_now = _iso_now()
                            await store.insert_metric(
                                {
                                    "timestamp": ts_now,
                                    "camera_id": ws.camera_id,
                                    "metric_type": "people_count",
                                    "value": count_result.by_class.get("person", 0),
                                    "confidence": count_result.kalman_confidence,
                                    "metadata": {
                                        "raw": count_result.raw_count,
                                        "velocity": count_result.velocity,
                                    },
                                }
                            )
                    except Exception as e:
                        logger.debug("Watch %s: counting error: %s", watch_id, e)
                continue

            # ── Run people counting (YOLO + ByteTrack + Kalman) ──
            count_data = None
            if counter is not None:
                try:
                    frame_bgr = frame[:, :, ::-1].copy()  # RGB → BGR for OpenCV
                    count_result = await loop.run_in_executor(None, counter.process, frame_bgr)
                    person_count = count_result.by_class.get("person", 0)
                    count_data = {
                        "people_count": person_count,
                        "people_in": int(count_result.people_in),
                        "people_out": int(count_result.people_out),
                        "velocity": round(count_result.velocity, 2),
                        "raw_detections": count_result.raw_count,
                        "confidence": round(count_result.kalman_confidence, 3),
                    }

                    # Store metrics to DB
                    if store and ws.camera_id:
                        ts_now = _iso_now()
                        await store.insert_metric(
                            {
                                "timestamp": ts_now,
                                "camera_id": ws.camera_id,
                                "metric_type": "people_count",
                                "value": person_count,
                                "confidence": count_result.kalman_confidence,
                                "metadata": {
                                    "raw": count_result.raw_count,
                                    "velocity": count_result.velocity,
                                },
                            }
                        )
                        if count_result.people_in > 0:
                            await store.insert_metric(
                                {
                                    "timestamp": ts_now,
                                    "camera_id": ws.camera_id,
                                    "metric_type": "people_in",
                                    "value": count_result.people_in,
                                }
                            )
                        if count_result.people_out > 0:
                            await store.insert_metric(
                                {
                                    "timestamp": ts_now,
                                    "camera_id": ws.camera_id,
                                    "metric_type": "people_out",
                                    "value": count_result.people_out,
                                }
                            )
                except Exception as e:
                    logger.warning("Watch %s: counting error: %s", watch_id, e)

            # ── Run VLM inference for each condition ──
            t0 = _time.monotonic()
            any_triggered = False
            condition_results = []

            last_result = None
            for cond in ws.conditions:
                last_result = await loop.run_in_executor(
                    None,
                    lambda q=cond.question: engine.analyze_frame(frame_chw, q),
                )
                answer_text = _strip_think_tags(last_result.text)
                triggered = _detect_triggered(answer_text)

                condition_results.append(
                    WatchConditionResult(
                        id=cond.id,
                        triggered=triggered is True,
                        answer=answer_text,
                    )
                )
                if triggered:
                    any_triggered = True

            latency_ms = int((_time.monotonic() - t0) * 1000)
            _track_inference(latency_ms)
            tok_s = last_result.metrics.tokens_per_sec if last_result else 0.0
            ws.checks += 1
            last_heartbeat = _time.monotonic()  # inference counts as activity

            metrics = WatchMetrics(
                latency_ms=latency_ms,
                tok_s=round(tok_s, 1),
                frames_analyzed=1,
            )

            # ── Store VLM event to DB ──
            if store and ws.camera_id and condition_results:
                description = condition_results[0].answer if condition_results else ""
                event = {
                    "timestamp": _iso_now(),
                    "camera_id": ws.camera_id,
                    "camera_name": ws.camera_name or ws.camera_id,
                    "description": description,
                    "alert_triggered": any_triggered,
                    "metadata": {"count": count_data} if count_data else {},
                }
                try:
                    event_id = await store.insert(event)
                    # Save frame for evidence
                    if any_triggered:
                        img = Image.fromarray(frame)
                        buf = io.BytesIO()
                        img.save(buf, format="JPEG", quality=80)
                        await store.save_frame(event_id, ws.camera_id, buf.getvalue())
                except Exception as e:
                    logger.debug("Watch %s: event store error: %s", watch_id, e)

            # ── Emit SSE events ──
            event_payload = {
                "watch_id": watch_id,
                "ts": _iso_now(),
                "conditions": [c.model_dump() for c in condition_results],
                "metrics": metrics.model_dump(),
            }
            if count_data:
                event_payload["counting"] = count_data

            if any_triggered:
                ws.alerts += 1
                # Encode frame as base64 JPEG for the alert
                img = Image.fromarray(frame)
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=80)
                event_payload["frame_b64"] = base64.b64encode(buf.getvalue()).decode()
                yield _sse_event("alert", event_payload)
            else:
                yield _sse_event("result", event_payload)

    except asyncio.CancelledError:
        pass
    except Exception as e:
        logger.error("Watch %s error: %s", watch_id, e, exc_info=True)
        yield _sse_event("error", {"watch_id": watch_id, "error": str(e)})
    finally:
        # Cleanup
        _kill_ffmpeg(proc)
        ws.state = "stopped"
        _watches.pop(watch_id, None)


def _sse_event(event_type: str, data: dict) -> str:
    """Format an SSE event."""
    return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"


def _iso_now() -> str:
    """Return current time in ISO 8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _stream_chat_completion(
    engine: TrioCore,
    video: str,
    prompt: str,
    request: ChatCompletionRequest,
    max_tokens: int,
    temperature: float,
) -> AsyncIterator[str]:
    """Stream OpenAI-compatible SSE for /v1/chat/completions."""
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    model = engine.config.model

    # First chunk: role
    first = ChatCompletionChunk(
        id=response_id,
        model=model,
        choices=[StreamChoice(delta=DeltaContent(role="assistant"))],
    )
    yield f"data: {first.model_dump_json(exclude_none=True)}\n\n"

    # Content chunks
    try:
        async for chunk in engine.stream_analyze(
            video=video,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if chunk.get("finished") and chunk.get("metrics"):
                # Final chunk with finish_reason
                final = ChatCompletionChunk(
                    id=response_id,
                    model=model,
                    choices=[
                        StreamChoice(
                            delta=DeltaContent(content=""),
                            finish_reason="stop",
                        )
                    ],
                )
                yield f"data: {final.model_dump_json(exclude_none=True)}\n\n"
                break

            if chunk.get("text"):
                content_chunk = ChatCompletionChunk(
                    id=response_id,
                    model=model,
                    choices=[StreamChoice(delta=DeltaContent(content=chunk["text"]))],
                )
                yield f"data: {content_chunk.model_dump_json(exclude_none=True)}\n\n"
    except Exception as e:
        logger.error("Stream error in /v1/chat/completions: %s", e, exc_info=True)
        error_chunk = ChatCompletionChunk(
            id=response_id,
            model=model,
            choices=[
                StreamChoice(
                    delta=DeltaContent(content=""),
                    finish_reason="error",
                )
            ],
        )
        yield f"data: {error_chunk.model_dump_json(exclude_none=True)}\n\n"

    yield "data: [DONE]\n\n"
