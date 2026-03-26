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
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from trio_core.api.routers.inference import router as inference_router

logger = logging.getLogger("trio.inference_server")

# ── Auth ────────────────────────────────────────────────────────────────────

_API_KEY: str | None = os.environ.get("TRIO_API_KEY")

# ── Request size limit (10 MB) — prevents OOM from oversized payloads ──────
_MAX_BODY_BYTES = 10 * 1024 * 1024


def create_app() -> FastAPI:
    """Create the inference-only FastAPI application."""

    app = FastAPI(
        title="trio-core inference",
        description="YOLO + VLM inference engine",
        version="1.0.0",
    )

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
            "uptime_s": int(time.monotonic() - _start_time),
        }

    app.include_router(inference_router)

    return app


_start_time = time.monotonic()


def main(host: str = "0.0.0.0", port: int = 8100, log_level: str = "info"):
    """Run the inference server directly."""
    import uvicorn

    app = create_app()
    uvicorn.run(app, host=host, port=port, log_level=log_level)


if __name__ == "__main__":
    main()
