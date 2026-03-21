"""Cameras API router for Trio Console."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from fastapi.responses import Response

from trio_core.api.console_models import CameraIn, CameraOut, IntentConfigOut, IntentRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["console-cameras"])

# ── In-memory caches ─────────────────────────────────────────────────────────

# Snapshot cache: camera_id -> (jpeg_bytes, timestamp)
_snapshot_cache: dict[str, tuple[bytes, float]] = {}
SNAPSHOT_TTL = 30  # seconds

# Resolved source URL cache: raw_url -> (resolved_url, timestamp)
_resolved_url_cache: dict[str, tuple[str, float]] = {}
RESOLVED_URL_TTL = 300  # 5 minutes (YouTube URLs expire)


def _get_cached_snapshot(camera_id: str) -> bytes | None:
    """Return cached snapshot bytes if fresh, else None."""
    entry = _snapshot_cache.get(camera_id)
    if entry is None:
        return None
    jpeg_bytes, ts = entry
    if time.monotonic() - ts > SNAPSHOT_TTL:
        return None
    return jpeg_bytes


def _put_cached_snapshot(camera_id: str, jpeg_bytes: bytes) -> None:
    """Store snapshot in the in-memory cache."""
    _snapshot_cache[camera_id] = (jpeg_bytes, time.monotonic())


def _get_cached_resolved_url(source_url: str) -> str | None:
    """Return cached resolved URL if fresh, else None."""
    entry = _resolved_url_cache.get(source_url)
    if entry is None:
        return None
    resolved, ts = entry
    if time.monotonic() - ts > RESOLVED_URL_TTL:
        return None
    return resolved


def _put_cached_resolved_url(source_url: str, resolved: str) -> None:
    """Store resolved URL in cache."""
    _resolved_url_cache[source_url] = (resolved, time.monotonic())


# ── Helpers ──────────────────────────────────────────────────────────────────


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


async def _resolve_source_cached(source_url: str) -> str:
    """Resolve source URL with caching for YouTube/HLS."""
    cached = _get_cached_resolved_url(source_url)
    if cached is not None:
        return cached

    from trio_core.source_resolver import resolve_source

    resolved = await asyncio.to_thread(resolve_source, source_url)
    _put_cached_resolved_url(source_url, resolved)
    return resolved


async def _capture_snapshot(source_url: str) -> bytes:
    """Capture a single JPEG frame from a source URL via ffmpeg."""
    from trio_core.source_resolver import detect_source_type

    resolved = await _resolve_source_cached(source_url)
    source_type = detect_source_type(source_url)

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        ffmpeg_args = ["ffmpeg", "-y"]
        if source_type == "rtsp":
            ffmpeg_args += ["-rtsp_transport", "tcp"]
        ffmpeg_args += ["-i", resolved, "-frames:v", "1", "-q:v", "2", tmp_path]

        proc = await asyncio.create_subprocess_exec(
            *ffmpeg_args,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=15.0)

        if proc.returncode != 0:
            msg = f"ffmpeg failed (exit {proc.returncode})"
            raise RuntimeError(msg)

        with open(tmp_path, "rb") as f:
            frame_bytes = f.read()

        if not frame_bytes:
            msg = "ffmpeg produced empty output"
            raise RuntimeError(msg)

        return frame_bytes
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── Endpoints ────────────────────────────────────────────────────────────────


@router.get("/api/cameras", response_model=list[CameraOut])
async def list_cameras(request: Request):
    """List all cameras with snapshot availability info."""
    store = _get_store(request)
    rows = await store.list_cameras()
    result = []
    for r in rows:
        cam = CameraOut(**r)
        cam_id = cam.id
        # Check in-memory cache first, then disk
        has = _get_cached_snapshot(cam_id) is not None
        if not has:
            disk_bytes = await store.get_camera_snapshot(cam_id)
            if disk_bytes is not None:
                has = True
                # Warm the in-memory cache from disk
                _put_cached_snapshot(cam_id, disk_bytes)
        cam.has_snapshot = has
        cam.snapshot_url = f"/api/cameras/{cam_id}/snapshot" if has else ""
        result.append(cam)
    return result


@router.post("/api/cameras", response_model=CameraOut, status_code=201)
async def create_camera(request: Request, body: CameraIn):
    """Register a new camera."""
    store = _get_store(request)
    cam_id = await store.create_camera(body.model_dump())
    # Fetch back the created camera
    cameras = await store.list_cameras()
    cam = next((c for c in cameras if c["id"] == cam_id), None)
    if cam is None:
        raise HTTPException(500, "Failed to create camera")
    return CameraOut(**cam)


@router.delete("/api/cameras/{camera_id}", status_code=204)
async def delete_camera(request: Request, camera_id: str):
    """Delete a camera."""
    store = _get_store(request)
    deleted = await store.delete_camera(camera_id)
    if not deleted:
        raise HTTPException(404, f"Camera {camera_id} not found")
    # Clean up cache
    _snapshot_cache.pop(camera_id, None)


DEFAULT_INTENT_CONFIG: dict = {
    "persona": "general",
    "customer_prompt": (
        "Describe this person: age, gender, clothing, items carried, activity. One sentence."
    ),
    "scene_prompt": (
        "Describe: number of people, busyness 1-10, what people are doing. Be specific."
    ),
    "report_type": "investment",
    "key_metrics": ["foot_traffic", "demographics", "behavioral"],
    "report_focus": "General activity patterns and demographics",
    "comparison_target": "",
}

_GEMINI_SYSTEM_PROMPT = """You are configuring an AI video surveillance system.
The user described their monitoring goal:
"{user_intent}"

Translate this into a structured JSON config with these fields:
{{
  "persona": "investment_analyst" or "security_officer" or "operations_manager" or "general",
  "customer_prompt": "VLM prompt for describing each person detected",
  "scene_prompt": "VLM prompt for periodic scene analysis",
  "report_type": "investment" or "security" or "operations",
  "key_metrics": ["foot_traffic", "demographics", "order_analysis", "dwell_time", etc],
  "report_focus": "what the daily report should emphasize",
  "comparison_target": "what to compare against (e.g. company reported numbers)"
}}
Return ONLY valid JSON, no explanation."""


async def _call_gemini(intent: str) -> dict:
    """Call Gemini API to translate natural language intent to structured config."""
    import httpx

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set, returning default intent config")
        return dict(DEFAULT_INTENT_CONFIG)

    prompt = _GEMINI_SYSTEM_PROMPT.replace("{user_intent}", intent)
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.0-flash:generateContent?key={api_key}"
    )
    payload = {"contents": [{"parts": [{"text": prompt}]}]}

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    # Strip markdown code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[: text.rfind("```")]
    return json.loads(text.strip())


@router.post("/api/cameras/{camera_id}/configure-intent", response_model=IntentConfigOut)
async def configure_intent(request: Request, camera_id: str, body: IntentRequest):
    """Translate a natural language monitoring intent into structured VLM config via Gemini."""
    store = _get_store(request)
    cameras = await store.list_cameras()
    cam = next((c for c in cameras if c["id"] == camera_id), None)
    if cam is None:
        raise HTTPException(404, f"Camera {camera_id} not found")

    intent_text = body.intent.strip()
    if not intent_text:
        config = dict(DEFAULT_INTENT_CONFIG)
    else:
        try:
            config = await _call_gemini(intent_text)
        except Exception as exc:
            logger.error("Gemini call failed for camera %s: %s", camera_id, exc)
            config = dict(DEFAULT_INTENT_CONFIG)

    await store.update_camera_intent(camera_id, intent_text, config)
    return IntentConfigOut(**config)


@router.get("/api/cameras/{camera_id}/snapshot")
async def camera_snapshot(request: Request, camera_id: str):
    """Capture a live snapshot from a camera, with in-memory + disk caching."""
    store = _get_store(request)
    cameras = await store.list_cameras()
    cam = next((c for c in cameras if c["id"] == camera_id), None)
    if cam is None:
        raise HTTPException(404, f"Camera {camera_id} not found")

    # 1. Check in-memory cache
    cached = _get_cached_snapshot(camera_id)
    if cached is not None:
        return Response(content=cached, media_type="image/jpeg")

    # 2. Check disk cache (survives restarts)
    disk_bytes = await store.get_camera_snapshot(camera_id)
    if disk_bytes is not None:
        _put_cached_snapshot(camera_id, disk_bytes)
        return Response(content=disk_bytes, media_type="image/jpeg")

    # 3. Capture fresh snapshot
    try:
        frame_bytes = await _capture_snapshot(cam["source_url"])
    except asyncio.TimeoutError:
        raise HTTPException(504, "Snapshot capture timed out")
    except Exception as e:
        logger.error("Snapshot error for camera %s: %s", camera_id, e)
        raise HTTPException(502, f"Snapshot failed: {e}")

    # Cache in memory and persist to disk
    _put_cached_snapshot(camera_id, frame_bytes)
    await store.save_camera_snapshot(camera_id, frame_bytes)

    return Response(content=frame_bytes, media_type="image/jpeg")


@router.post("/api/cameras/refresh-snapshots", status_code=202)
async def refresh_snapshots(request: Request, background_tasks: BackgroundTasks):
    """Trigger background snapshot refresh for all cameras.

    Returns 202 immediately. Snapshots are captured in the background
    and cached for subsequent requests.
    """
    store = _get_store(request)
    cameras = await store.list_cameras()
    if not cameras:
        return {"status": "no cameras"}

    async def _refresh_all(camera_list: list[dict], st) -> None:
        """Capture snapshots for all cameras concurrently."""
        async def _refresh_one(cam: dict) -> None:
            cam_id = cam["id"]
            try:
                frame_bytes = await _capture_snapshot(cam["source_url"])
                _put_cached_snapshot(cam_id, frame_bytes)
                await st.save_camera_snapshot(cam_id, frame_bytes)
                logger.info("Refreshed snapshot for camera %s", cam_id)
            except Exception:
                logger.warning("Failed to refresh snapshot for camera %s", cam_id, exc_info=True)

        await asyncio.gather(*[_refresh_one(c) for c in camera_list])

    background_tasks.add_task(_refresh_all, cameras, store)

    return {"status": "refreshing", "camera_count": len(cameras)}
