"""FastAPI server for TrioCore."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from trio_core.api.models import (
    AnalyzeFrameRequest,
    AnalyzeFrameResponse,
    ChatCompletionChunk,
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ContentPart,
    DeltaContent,
    HealthResponse,
    InferenceMetricsResponse,
    ModelInfo,
    ModelListResponse,
    StreamChoice,
    Usage,
    VideoAnalyzeRequest,
    VideoAnalyzeResponse,
)
from trio_core.config import EngineConfig
from trio_core.engine import TrioCore

logger = logging.getLogger(__name__)

_engine: TrioCore | None = None


def get_engine() -> TrioCore:
    if _engine is None or not _engine._loaded:
        raise HTTPException(503, "Engine not loaded")
    return _engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    config = app.state.config if hasattr(app.state, "config") else EngineConfig()
    backend = getattr(app.state, "backend", None)
    _engine = TrioCore(config, backend=backend)
    logger.info("Loading model: %s", config.model)
    _engine.load()
    logger.info("Engine ready: backend=%s", _engine._backend.backend_name if _engine._backend else "none")
    yield
    logger.info("Shutting down")


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
        import tempfile
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
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.analyze_video(
                video=frames,
                prompt=request.question,
            ),
        )

        latency_ms = int((_time.monotonic() - t0) * 1000)
        answer = result.text.strip()

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
        return ModelListResponse(
            data=[ModelInfo(id=engine.config.model)]
        )

    @app.post("/v1/video/analyze")
    async def analyze_video(request: VideoAnalyzeRequest):
        engine = get_engine()

        if request.stream:
            return StreamingResponse(
                _stream_video_analyze(engine, request),
                media_type="text/event-stream",
            )

        loop = asyncio.get_event_loop()
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

        loop = asyncio.get_event_loop()
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
            raise HTTPException(400, "No visual content found in messages. Use content parts with type='image_url' or 'video'.")

        max_tokens = request.max_tokens or engine.config.max_tokens
        temperature = request.temperature if request.temperature is not None else engine.config.temperature

        # Resolve media source: base64 data URI → temp file, or path/URL as-is
        source = _resolve_media(media[0])

        if request.stream:
            return StreamingResponse(
                _stream_chat_completion(engine, source, prompt, request, max_tokens, temperature),
                media_type="text/event-stream",
            )

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: engine.analyze_video(
                video=source,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
        )

        return ChatCompletionResponse(
            model=engine.config.model,
            choices=[ChatCompletionChoice(
                message={"role": "assistant", "content": result.text},
            )],
            usage=Usage(
                prompt_tokens=result.metrics.prompt_tokens,
                completion_tokens=result.metrics.completion_tokens,
                total_tokens=result.metrics.prompt_tokens + result.metrics.completion_tokens,
            ),
        )


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
    neg_patterns = ("there is no", "there are no", "there isn't", "there aren't",
                    "i don't see", "i do not see", "no ", "not ", "cannot see",
                    "can't see", "nothing", "nobody", "no one")
    pos_patterns = ("there is a", "there are", "i see a", "i can see",
                    "someone", "a person", "a package", "a delivery")
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


def _resolve_media(source: str) -> str:
    """Resolve media source to a file path.

    Handles:
    - base64 data URI (data:image/jpeg;base64,...) → temp file
    - file path or URL → returned as-is
    """
    if source.startswith("data:"):
        import base64
        import tempfile

        if "," not in source:
            raise HTTPException(400, "Invalid data URI: missing comma separator")

        # Parse data URI: data:<mime>;base64,<data>
        header, data = source.split(",", 1)
        mime = header.split(":")[1].split(";")[0] if ":" in header else ""
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "video/mp4": ".mp4"}.get(mime, ".bin")

        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        tmp.write(base64.b64decode(data))
        tmp.close()

        # Register for cleanup
        from trio_core.video import _TEMP_FILES
        _TEMP_FILES.append(tmp.name)

        return tmp.name

    return source


async def _stream_video_analyze(
    engine: TrioCore, request: VideoAnalyzeRequest
) -> AsyncIterator[str]:
    """Stream /v1/video/analyze response as SSE."""
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
    yield "data: [DONE]\n\n"


async def _stream_frames_analyze(
    engine: TrioCore,
    frames: "np.ndarray",
    prompt: str,
    max_tokens: int | None,
    temperature: float | None,
) -> AsyncIterator[str]:
    """Stream /v1/frames/analyze response as SSE."""
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
    yield "data: [DONE]\n\n"


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
                choices=[StreamChoice(
                    delta=DeltaContent(content=""),
                    finish_reason="stop",
                )],
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

    yield "data: [DONE]\n\n"
