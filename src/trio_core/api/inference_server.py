"""Clean inference-only server for trio-core.

trio serve -> starts this
Port 8100 (inference only, separate from cortex on 8000)
Optional TRIO_API_KEY env var for authentication.

Endpoints:
    GET  /health
    POST /api/inference/detect
    POST /api/inference/crop-describe
    POST /api/inference/describe
    GET  /api/inference/status
"""

from __future__ import annotations

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


def create_app() -> FastAPI:
    """Create the inference-only FastAPI application."""

    app = FastAPI(
        title="trio-core inference",
        description="YOLO + VLM inference engine",
        version="1.0.0",
    )

    # CORS — allow cortex and any local dev tools
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Optional API key middleware
    if _API_KEY:
        @app.middleware("http")
        async def _check_api_key(request: Request, call_next):
            if request.url.path == "/health":
                return await call_next(request)
            auth = request.headers.get("Authorization", "")
            if auth == f"Bearer {_API_KEY}":
                return await call_next(request)
            return JSONResponse(status_code=401, content={"detail": "Invalid API key"})
        logger.info("API key auth enabled (TRIO_API_KEY is set)")
    else:
        logger.info("No API key auth (TRIO_API_KEY not set)")

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
