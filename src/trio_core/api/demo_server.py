"""Lightweight demo server — serves console API endpoints without VLM model.

Usage: python -m trio_core.api.demo_server
Starts on port 8000 with all console endpoints backed by demo data.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from trio_core.api.store import EventStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trio.demo")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = EventStore()
    await store.init()
    app.state.event_store = store
    logger.info("Demo server ready — event store loaded")
    yield
    await store.close()


def create_demo_app() -> FastAPI:
    app = FastAPI(title="Trio Enterprise Demo", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register all console routers
    from trio_core.api.routers import (
        alerts,
        analytics,
        auth,
        auto_alerts,
        cameras,
        chat,
        events,
        insights,
        intelligence,
        metrics,
        reports,
    )

    app.include_router(cameras.router)
    app.include_router(events.router)
    app.include_router(metrics.router)
    app.include_router(analytics.router)
    app.include_router(insights.router)
    app.include_router(intelligence.router)
    app.include_router(reports.router)
    app.include_router(alerts.router)
    app.include_router(auto_alerts.router)
    app.include_router(auth.router)
    app.include_router(chat.router)

    @app.get("/health")
    async def health():
        return {"status": "ok", "mode": "demo"}

    return app


if __name__ == "__main__":
    app = create_demo_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
