"""Cameras API router for Trio Console."""

from __future__ import annotations

import asyncio
import logging
import tempfile

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response

from trio_core.api.console_models import CameraIn, CameraOut

logger = logging.getLogger(__name__)

router = APIRouter(tags=["console-cameras"])


def _get_store(request: Request):
    store = getattr(request.app.state, "event_store", None)
    if store is None:
        raise HTTPException(503, "Event store not initialized")
    return store


@router.get("/api/cameras", response_model=list[CameraOut])
async def list_cameras(request: Request):
    """List all cameras."""
    store = _get_store(request)
    rows = await store.list_cameras()
    return [CameraOut(**r) for r in rows]


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


@router.get("/api/cameras/{camera_id}/snapshot")
async def camera_snapshot(request: Request, camera_id: str):
    """Capture a live snapshot from a camera using ffmpeg."""
    store = _get_store(request)
    cameras = await store.list_cameras()
    cam = next((c for c in cameras if c["id"] == camera_id), None)
    if cam is None:
        raise HTTPException(404, f"Camera {camera_id} not found")

    source_url = cam["source_url"]
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-y", "-rtsp_transport", "tcp",
            "-i", source_url, "-frames:v", "1", "-q:v", "2", tmp_path,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await asyncio.wait_for(proc.wait(), timeout=10.0)

        if proc.returncode != 0:
            raise HTTPException(502, f"ffmpeg failed (exit {proc.returncode})")

        with open(tmp_path, "rb") as f:
            frame_bytes = f.read()

        if not frame_bytes:
            raise HTTPException(502, "ffmpeg produced empty output")

        return Response(content=frame_bytes, media_type="image/jpeg")
    except asyncio.TimeoutError:
        raise HTTPException(504, "Snapshot capture timed out")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Snapshot error for camera %s: %s", camera_id, e)
        raise HTTPException(502, f"Snapshot failed: {e}")
    finally:
        import os
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
