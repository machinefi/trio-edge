"""Clean inference-only server for trio-core.

trio serve -> starts this
Port 8100 (inference only)
Optional TRIO_API_KEY env var for authentication.

Endpoints:
    GET  /health
    POST /api/inference/detect
    POST /api/inference/crop-describe
    POST /api/inference/describe
    GET  /api/inference/status
"""

from __future__ import annotations

import hmac
import logging
import os
import time as _time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trio_core.api.routers.inference import router as inference_router

if TYPE_CHECKING:
    import trio_core.engine

logger = logging.getLogger("trio.inference_server")

# ── Auth ────────────────────────────────────────────────────────────────────

_API_KEY: str | None = os.environ.get("TRIO_API_KEY")

# ── Request size limit (10 MB) — prevents OOM from oversized payloads ──────
_MAX_BODY_BYTES = 10 * 1024 * 1024

# Global engine reference — loaded once in lifespan startup
_vlm_engine: "trio_core.engine.TrioCore | None" = None


def _load_engine() -> "trio_core.engine.TrioCore":
    """Load (or return cached) VLM engine synchronously during lifespan startup."""
    global _vlm_engine
    if _vlm_engine is None:
        from trio_core.engine import TrioCore
        from trio_core.config import EngineConfig

        config = EngineConfig()
        _vlm_engine = TrioCore(config)
        _vlm_engine.load()
    # Also set on the inference router's global so _get_vlm() reuses this instance
    try:
        from trio_core.api.routers import inference

        inference._vlm_engine = _vlm_engine
    except Exception:
        pass  # Router not yet imported — endpoints will load lazily
    return _vlm_engine


@asynccontextmanager
async def _inference_lifespan(app: FastAPI):
    """Startup/shutdown lifespan for the inference server.

    On startup: load the model and optionally run warm-up inference.
    On shutdown: log uptime (engine cleanup is automatic via GC).
    """
    global _start_time
    _start_time = _time.monotonic()

    warmup_enabled = getattr(app.state, "warmup", True)
    logger.info("Loading model...")
    engine = _load_engine()
    logger.info("Engine ready: backend=%s", engine._backend.backend_name if engine._backend else "none")

    if warmup_enabled:
        import time as _wall

        _t0 = _wall.perf_counter()
        warmup_parts = []
        # 1. YOLO warm-up (always runs — YOLO is required)
        if hasattr(engine, "_counter") and hasattr(engine._counter, "warmup"):
            _elapsed = engine._counter.warmup()
            warmup_parts.append(f"YOLO={_elapsed:.3f}s")
            logger.info("YOLO warm-up complete (%.3fs)", _elapsed)
        # 2. VLM warm-up (only if VLM is loaded — graceful no-op otherwise)
        if hasattr(engine, "warmup"):
            _elapsed = engine.warmup()
            if _elapsed is not None:
                warmup_parts.append(f"VLM={_elapsed:.3f}s")
                logger.info("VLM warm-up complete (%.3fs)", _elapsed)
        _total = _wall.perf_counter() - _t0
        if warmup_parts:
            logger.info("Model warm-up complete (%s) total=%.3fs", ", ".join(warmup_parts), _total)
        else:
            logger.info("Model warm-up complete (total=%.3fs)", _total)

    yield  # server runs here

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("Shutting down — uptime %.0fs", _time.monotonic() - _start_time)


def create_app(warmup: bool = True) -> FastAPI:
    """Create the inference-only FastAPI application."""
    app = FastAPI(
        title="trio-core inference",
        description="YOLO + VLM inference engine",
        version="1.0.0",
        lifespan=_inference_lifespan,
    )
    app.state.warmup = warmup

    # CORS — allow any local dev tools
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Body size limit middleware — reject oversized requests early
    @app.middleware("http")
    async def _check_body_size(request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > _MAX_BODY_BYTES:
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "message": f"Request body exceeds {_MAX_BODY_BYTES // (1024*1024)}MB limit"},
            )
        return await call_next(request)

    # Optional API key middleware
    if _API_KEY:
        _api_key_bytes = _API_KEY.encode()

        @app.middleware("http")
        async def _check_api_key(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            expected = f"Bearer {_API_KEY}"
            # Use constant-time comparison to prevent timing attacks
            if hmac.compare_digest(auth.encode(), expected.encode()):
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        logger.info("API key auth enabled (TRIO_API_KEY is set)")
    else:
        logger.warning("No API key auth (TRIO_API_KEY not set) — server is UNPROTECTED")

    # ── Routes ──

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "trio-core-inference",
            "uptime_s": int(_time.monotonic() - _start_time),
        }

    app.include_router(inference_router)

    return app


_start_time = _time.monotonic()


def main(host: str = "0.0.0.0", port: int = 8100, log_level: str = "info", warmup: bool = True):
    """Run the inference server directly."""
    import uvicorn

    app = create_app(warmup=warmup)
    uvicorn.run(app, host=host, port=port, log_level=log_level)
