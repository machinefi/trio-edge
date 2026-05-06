"""Shared inference API — VLM + YOLO as a service for all pipelines.

Loads models ONCE in the server process. Pipelines call these endpoints
instead of loading their own copies.

Memory savings: 3GB VLM + 0.3GB YOLO loaded once vs per-pipeline.
"""

from __future__ import annotations

import base64
import functools
import logging
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("trio.inference")

router = APIRouter(prefix="/api/inference", tags=["inference"])

# Semaphore caps concurrent VLM requests at the FastAPI handler. Capacity is
# read lazily from EngineConfig.vlm_api_concurrency the first time the engine
# is loaded — defaults to 1 (safe for local GPU backends), and is typically
# set to 8-16 in deployments using remote_vlm_url where the remote service
# handles its own scheduling and the per-backend lock is a nullcontext.
import asyncio as _asyncio

_vlm_semaphore: _asyncio.Semaphore | None = None


def _get_vlm_semaphore() -> _asyncio.Semaphore:
    global _vlm_semaphore
    if _vlm_semaphore is not None:
        return _vlm_semaphore
    # Force engine load so config is materialized; cheap after first call.
    engine = _get_vlm()
    capacity = max(1, int(getattr(engine.config, "vlm_api_concurrency", 1)))
    _vlm_semaphore = _asyncio.Semaphore(capacity)
    logger.info("VLM API semaphore initialized: capacity=%d", capacity)
    return _vlm_semaphore

import re as _re

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)
_PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_PANEL_COLORS = (
    (0, 255, 255),
    (255, 128, 0),
    (80, 220, 80),
    (255, 0, 255),
    (0, 160, 255),
)


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from VLM output (Qwen3.5 thinking mode)."""
    text = _THINK_RE.sub("", text).strip()
    # Also handle unclosed <think> tag
    if "<think>" in text:
        text = (
            text.split("</think>")[-1].strip()
            if "</think>" in text
            else text.split("<think>")[0].strip()
        )
    return text


def _vlm_http_error(e: BaseException) -> HTTPException:
    """Translate a VLM upstream exception into a clean HTTPException.

    Upstream HTTP 4xx (e.g. DashScope `data_inspection_failed` content moderation)
    is a per-request problem; surface as 422 so callers can drop the frame
    without treating the VLM service as down. Anything else (5xx, connection,
    timeout, decode bugs) maps to 503.
    """
    from openai import APIStatusError

    if isinstance(e, APIStatusError):
        sc = int(getattr(e, "status_code", 0) or 0)
        if 400 <= sc < 500:
            body = getattr(getattr(e, "response", None), "text", "") or str(e)
            moderated = (
                "data_inspection_failed" in body or "inappropriate" in body.lower()
            )
            logger.warning(
                "VLM upstream rejected request (HTTP %d, moderated=%s): %s",
                sc, moderated, str(e)[:200],
            )
            return HTTPException(
                status_code=422,
                detail={
                    "kind": "vlm_client_error",
                    "upstream_status": sc,
                    "moderated": moderated,
                    "message": str(e)[:300],
                },
            )
    logger.error("VLM inference failed: %s", e)
    return HTTPException(status_code=503, detail=f"VLM inference error: {e}")


# ── Lazy-loaded shared models ──

_vlm_engine = None
_yolo_counter = None
_vlm_init_lock = threading.Lock()


def _get_vlm():
    global _vlm_engine
    if _vlm_engine is not None:
        return _vlm_engine

    with _vlm_init_lock:
        if _vlm_engine is not None:
            return _vlm_engine

        from trio_core.config import EngineConfig
        from trio_core.engine import TrioCore

        config = EngineConfig.from_env_file()
        engine = TrioCore(config)
        engine.load()
        _vlm_engine = engine
        logger.info("VLM loaded (shared): %s", config.model)
    return _vlm_engine


def _get_yolo():
    global _yolo_counter
    if _yolo_counter is None:
        import os
        from pathlib import Path

        from trio_core.counter import PeopleCounter

        # Find model: check TRIO_YOLO_MODEL env, then relative to trio-core package
        model_path = os.environ.get("TRIO_YOLO_MODEL")
        if not model_path:
            # Resolve relative to the trio-core repo root (file is at src/trio_core/api/routers/)
            pkg_root = Path(__file__).resolve().parent.parent.parent.parent.parent
            model_path = str(pkg_root / "models" / "yolov10n" / "onnx" / "model.onnx")
        _yolo_counter = PeopleCounter(model_path=model_path)
        logger.info(
            "YOLO loaded (shared): tiled=%s, confidence=%s",
            _yolo_counter._tiled,
            _yolo_counter.detector.confidence,
        )
    return _yolo_counter


# ── Request/Response models ──


class DescribeRequest(BaseModel):
    image_b64: str = Field(
        ..., max_length=14_000_000, description="Base64-encoded JPEG image (max ~10MB)"
    )
    prompt: str = Field(
        default="Describe everything you see: people (age, gender, clothing), vehicles (color, make, type), animals.",
        max_length=4096,
        description="VLM prompt",
    )
    response_format: dict | None = Field(
        default=None,
        description=(
            "OpenAI-compatible structured-output spec, forwarded to remote "
            "VLM backends. Ignored by local backends."
        ),
    )


class DescribeResponse(BaseModel):
    description: str
    elapsed_ms: int


class CropDescribeRequest(BaseModel):
    image_b64: str = Field(
        ..., max_length=14_000_000, description="Base64-encoded JPEG of full frame (max ~10MB)"
    )
    crops: list[dict] = Field(
        default_factory=list, max_length=50, description="List of crop bboxes from YOLO detect"
    )
    max_crops: int = Field(
        default=3,
        ge=0,
        le=20,
        description=(
            "Number of YOLO crops to include as labeled zoom panels in the "
            "single full-scene VLM pass."
        ),
    )
    scene_prompt: str = Field(
        default="",
        max_length=4096,
        description="Custom scene prompt (optional, uses default if empty)",
    )
    response_format: dict | None = Field(
        default=None,
        description=(
            "OpenAI-compatible structured-output spec, forwarded to remote "
            "VLM backends. Ignored by local backends."
        ),
    )


class CropDescribeResponse(BaseModel):
    description: str
    entities: dict | None = None
    crop_descriptions: list[str] = Field(default_factory=list)
    elapsed_ms: int


class DetectRequest(BaseModel):
    image_b64: str = Field(
        ..., max_length=14_000_000, description="Base64-encoded JPEG image (max ~10MB)"
    )
    pad_ratio: float = Field(default=0.15, ge=0.0, le=1.0)


class DetectResponse(BaseModel):
    people_count: int
    vehicle_count: int
    by_class: dict
    crops_b64: list[dict] = Field(default_factory=list, description="Crops with base64 images")
    elapsed_ms: int


# ── Helpers ──


def _decode_image(b64: str) -> np.ndarray:
    """Decode base64 JPEG to BGR numpy array."""
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Invalid image data")
    return frame


def _frame_to_chw(frame_bgr: np.ndarray) -> np.ndarray:
    """Convert BGR frame to CHW float32 for VLM."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return rgb.transpose(2, 0, 1).astype(np.float32) / 255.0


def _align_crop_to_model(crop_bgr: np.ndarray, image_factor: int) -> np.ndarray:
    """Upscale small crops to a valid multiple for model processors."""
    if crop_bgr.size == 0:
        return crop_bgr

    h, w = crop_bgr.shape[:2]
    th = max(image_factor, round(h / image_factor) * image_factor)
    tw = max(image_factor, round(w / image_factor) * image_factor)

    return cv2.resize(crop_bgr, (tw, th)) if (th != h or tw != w) else crop_bgr


def _bbox_from_crop_info(crop_info: dict) -> tuple[int, int, int, int] | None:
    bbox = crop_info.get("bbox") or crop_info.get("xyxy")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (int(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _crop_with_padding(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
    *,
    pad_ratio: float = 0.15,
) -> np.ndarray | None:
    x1, y1, x2, y2 = bbox
    h, w = frame.shape[:2]
    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    if cx2 <= cx1 or cy2 <= cy1:
        return None
    crop = frame[cy1:cy2, cx1:cx2]
    if crop.shape[0] < 8 or crop.shape[1] < 8:
        return None
    return crop


def _prepare_zoom_panels(
    frame: np.ndarray,
    crops: list[dict],
    *,
    max_crops: int,
    image_factor: int,
) -> list[dict]:
    if max_crops <= 0:
        return []

    panels: list[dict] = []
    valid = [crop for crop in crops if isinstance(crop, dict)]
    valid.sort(key=lambda c: c.get("confidence", 0), reverse=True)

    for crop_info in valid:
        bbox = _bbox_from_crop_info(crop_info)
        if bbox is None:
            continue
        crop_bgr = _crop_with_padding(frame, bbox)
        if crop_bgr is None:
            continue

        label = _PANEL_LABELS[len(panels)]
        panels.append(
            {
                "label": label,
                "class": crop_info.get("class") or crop_info.get("label") or "object",
                "bbox": bbox,
                "confidence": crop_info.get("confidence"),
                "crop_bgr": _align_crop_to_model(crop_bgr, image_factor),
            }
        )
        if len(panels) >= min(max_crops, len(_PANEL_LABELS)):
            break

    return panels


def _resize_letterboxed(
    image: np.ndarray,
    *,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    if image.size == 0 or target_w <= 0 or target_h <= 0:
        return canvas

    h, w = image.shape[:2]
    scale = min(target_w / max(w, 1), target_h / max(h, 1))
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = cv2.resize(image, (new_w, new_h))
    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _draw_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(
        image,
        (x, max(0, y - th - baseline - 6)),
        (x + tw + 8, y + baseline + 4),
        color,
        -1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y),
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _compose_zoom_context_image(frame: np.ndarray, panels: list[dict]) -> np.ndarray:
    if not panels:
        return frame

    full = frame.copy()
    h, w = full.shape[:2]
    panel_w = max(160, min(384, w // 3 if w >= 480 else w // 2))
    slot_h = max(120, h // max(len(panels), 1))
    canvas_h = max(h, slot_h * len(panels))
    canvas = np.zeros((canvas_h, w + panel_w, 3), dtype=np.uint8)

    for index, panel in enumerate(panels):
        color = _PANEL_COLORS[index % len(_PANEL_COLORS)]
        x1, y1, x2, y2 = panel["bbox"]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        cv2.rectangle(full, (x1, y1), (x2, y2), color, 3)
        _draw_label(full, panel["label"], (x1 + 4, max(y1 + 24, 24)), color)

    canvas[:h, :w] = full

    for index, panel in enumerate(panels):
        color = _PANEL_COLORS[index % len(_PANEL_COLORS)]
        y0 = index * slot_h
        y1 = canvas_h if index == len(panels) - 1 else (index + 1) * slot_h
        slot = canvas[y0:y1, w : w + panel_w]
        slot[:] = (18, 18, 18)

        confidence = panel.get("confidence")
        conf_text = ""
        if isinstance(confidence, (int, float)):
            conf_text = f" {float(confidence):.2f}"
        label_text = f"{panel['label']} {panel['class']}{conf_text}"
        cv2.rectangle(slot, (0, 0), (panel_w - 1, y1 - y0 - 1), color, 2)
        _draw_label(slot, label_text, (8, 24), color)

        image_h = max(1, y1 - y0 - 38)
        image_w = max(1, panel_w - 12)
        thumb = _resize_letterboxed(
            panel["crop_bgr"],
            target_w=image_w,
            target_h=image_h,
        )
        slot[32 : 32 + image_h, 6 : 6 + image_w] = thumb

    return canvas


def _format_yolo_detection_context(crops: list[dict], *, limit: int = 8) -> str:
    valid = [crop for crop in crops if isinstance(crop, dict)]
    valid.sort(key=lambda c: c.get("confidence", 0), reverse=True)

    lines = []
    for crop in valid[:limit]:
        bbox = crop.get("bbox") or crop.get("xyxy")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        label = crop.get("class") or crop.get("label") or "object"
        parts = [f"class={label}", f"bbox={[int(v) for v in bbox]}"]
        track_id = crop.get("track_id")
        if track_id is not None:
            parts.insert(0, f"track_id={track_id}")
        confidence = crop.get("confidence")
        if isinstance(confidence, (int, float)):
            parts.append(f"confidence={float(confidence):.2f}")
        lines.append("- " + ", ".join(parts))

    if not lines:
        return ""
    return (
        "YOLO detections to use as visual hints; verify against the image:\n"
        + "\n".join(lines)
    )


def _format_zoom_panel_context(panels: list[dict]) -> str:
    if not panels:
        return ""

    lines = []
    for panel in panels:
        confidence = panel.get("confidence")
        conf_text = ""
        if isinstance(confidence, (int, float)):
            conf_text = f", confidence={float(confidence):.2f}"
        lines.append(
            f"- {panel['label']}: class={panel['class']}, "
            f"bbox={list(panel['bbox'])}{conf_text}"
        )
    return (
        "The input image is a composite: the main full camera frame is on the "
        "left, and labeled zoom panels are on the right. Box labels in the "
        "full frame match the zoom panels. Use zoom panels for appearance "
        "details, but ground location and activity in the full frame.\n"
        "Zoom panels:\n"
        + "\n".join(lines)
        + "\n\nBefore SCENE, include crop details exactly as:\n"
        "CROPS:\n"
        + "\n".join(
            f"{panel['label']}: {panel['class']}: [short phrase]"
            for panel in panels
        )
    )


def _extract_crop_descriptions(text: str, panels: list[dict]) -> list[str]:
    descriptions = []
    lines = text.splitlines()
    for panel in panels:
        label = str(panel["label"])
        obj_class = str(panel["class"])
        found = ""
        pattern = _re.compile(
            rf"^\s*(?:[-*]\s*)?{_re.escape(label)}\s*[:.)-]\s*(.+)$"
        )
        for line in lines:
            match = pattern.match(line.strip())
            if not match:
                continue
            found = match.group(1).strip()
            break
        if not found:
            continue
        if not found.lower().startswith(f"{obj_class.lower()}:"):
            found = f"{obj_class}: {found}"
        descriptions.append(found)
    return descriptions


def _normalize_entity_item(kind: str, item) -> dict:
    """Normalize model-emitted entity items (strings or dicts) to a standard dict."""
    if isinstance(item, dict):
        norm = dict(item)
    else:
        text = str(item).strip()
        norm = {"description": text}
        if kind == "vehicles":
            norm["make"] = norm["brand"] = text
        elif kind == "persons":
            norm["attire"] = text
            norm["role"] = "unknown"
        elif kind == "animals":
            norm["breed"] = text

    if kind == "vehicles":
        label = norm.get("brand") or norm.get("make") or norm.get("description", "")
        norm.setdefault("brand", label)
        norm.setdefault("make", label)
    return norm


def _normalize_entities(entities: dict | None) -> dict:
    """Normalize parsed entities payload into the shape downstream code expects."""
    if not isinstance(entities, dict):
        return {}

    res = dict(entities)
    for k in ("persons", "vehicles", "animals"):
        items = res.get(k) or []
        if not isinstance(items, list):
            items = [items]
        res[k] = [_normalize_entity_item(k, i) for i in items]
    return res


# ── Endpoints ──


@router.post("/describe", response_model=DescribeResponse)
async def describe(req: DescribeRequest):
    """Run VLM on a single image. Returns natural language description."""
    import asyncio

    async with _get_vlm_semaphore():
        loop = asyncio.get_event_loop()
        t0 = time.time()

        def _sync_describe():
            frame = _decode_image(req.image_b64)
            frame_chw = _frame_to_chw(frame)
            engine = _get_vlm()
            return engine.analyze_frame(
                frame_chw, req.prompt, response_format=req.response_format
            )

        try:
            result = await loop.run_in_executor(None, _sync_describe)
        except Exception as e:
            raise _vlm_http_error(e) from e
        elapsed = int((time.time() - t0) * 1000)

    return DescribeResponse(
        description=_strip_thinking(result.text or ""),
        elapsed_ms=elapsed,
    )


@router.post("/crop-describe", response_model=CropDescribeResponse)
async def crop_describe(req: CropDescribeRequest):
    """Describe a frame, optionally adding crop zoom panels.

    This endpoint always performs one VLM generation. ``max_crops`` controls
    how many YOLO boxes become labeled zoom panels in the composite image.
    """

    async with _get_vlm_semaphore():
        try:
            return await _crop_describe_inner(req)
        except HTTPException:
            raise
        except Exception as e:
            raise _vlm_http_error(e) from e


async def _crop_describe_inner(req: CropDescribeRequest):
    import asyncio
    import json

    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _decode_image, req.image_b64)
    engine = _get_vlm()
    image_factor = getattr(getattr(engine, "_profile", None), "merge_factor", 32)

    t0 = time.time()
    loop = asyncio.get_event_loop()

    zoom_panels = _prepare_zoom_panels(
        frame,
        req.crops,
        max_crops=req.max_crops,
        image_factor=image_factor,
    )

    scene_prompt = req.scene_prompt or (
        "You are an expert scene analyst for a video intelligence system. "
        "Your job is to UNDERSTAND what is happening, not just list objects.\n\n"
        "Analyze this camera frame and provide:\n"
        "1. SCENE: What is the overall situation? (e.g. 'busy morning commute', 'quiet residential street at night', 'loading dock with delivery in progress')\n"
        "2. ACTIVITIES: What are people/vehicles DOING? Describe actions, movements, interactions. (e.g. 'two people having a conversation near the bench', 'delivery truck unloading packages', 'person walking a dog heading east')\n"
        "3. RELATIONSHIPS: How do entities relate to each other? (e.g. 'group of 3 walking together', 'car waiting for pedestrian to cross', 'person appears to be watching the building entrance')\n"
        "4. NOTABLE: Anything unusual, security-relevant, or worth remembering? (e.g. 'vehicle parked in no-parking zone', 'person lingering near entrance without entering', 'unusually empty for this type of location')\n\n"
        "Be SPECIFIC about appearances: vehicle make/model/color, person clothing/build, animal breed.\n\n"
        "Format your response as:\n"
        "SCENE: [one sentence scene summary]\n"
        "ACTIVITIES: [2-3 sentences describing what's happening]\n"
        "NOTABLE: [anything unusual or worth tracking, or 'nothing unusual']\n"
        'JSON: {"people_count":N,"vehicle_count":N,"persons":[{"appearance":"...","action":"...","location":"..."}],"vehicles":[{"type":"...","color":"...","make":"...","action":"..."}],"scene_type":"...","activity_level":"quiet|moderate|busy","mood":"calm|tense|active"}'
    )

    if zoom_panels:
        scene_prompt = _format_zoom_panel_context(zoom_panels) + "\n\n" + scene_prompt
    elif req.crops and "YOLO detections" not in scene_prompt:
        yolo_context = _format_yolo_detection_context(req.crops)
        if yolo_context:
            scene_prompt = yolo_context + "\n\n" + scene_prompt

    context_frame = _compose_zoom_context_image(frame, zoom_panels)
    frame_chw = _frame_to_chw(context_frame)
    result = await loop.run_in_executor(
        None,
        functools.partial(
            engine.analyze_frame,
            frame_chw,
            scene_prompt,
            response_format=req.response_format,
        ),
    )
    text = _strip_thinking(result.text or "")
    crop_descriptions = _extract_crop_descriptions(text, zoom_panels)

    # Parse response — handle multiple VLM output formats:
    # Format 1: DESCRIPTION: ... JSON: {...}
    # Format 2: ```json {...} ``` (markdown code block)
    # Format 3: raw JSON {...}
    # Format 4: plain text only
    desc = ""
    entities = None

    # Strip markdown code fences
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
    if clean.endswith("```"):
        clean = clean[: clean.rfind("```")]
    clean = clean.strip()

    # Try to extract JSON (from any format)
    s, e = clean.find("{"), clean.rfind("}")
    if s >= 0 and e > s:
        try:
            entities = json.loads(clean[s : e + 1])
        except json.JSONDecodeError:
            pass
    entities = _normalize_entities(entities)

    # Extract description — combine SCENE + ACTIVITIES + NOTABLE into rich description
    scene_line = ""
    activities_line = ""
    notable_line = ""

    for prefix in ["SCENE:", "ACTIVITIES:", "NOTABLE:", "DESCRIPTION:"]:
        if prefix in text:
            after = text.split(prefix, 1)[1]
            # Take until next section header or JSON
            for end_marker in ["SCENE:", "ACTIVITIES:", "NOTABLE:", "JSON:", "```"]:
                if end_marker != prefix and end_marker in after:
                    after = after.split(end_marker, 1)[0]
                    break
            line = after.strip().split("\n")[0].strip() if after.strip() else ""
            if prefix == "SCENE:" or prefix == "DESCRIPTION:":
                scene_line = line
            elif prefix == "ACTIVITIES:":
                activities_line = line
            elif prefix == "NOTABLE:":
                notable_line = line

    if scene_line:
        desc = scene_line
        if activities_line:
            desc += " " + activities_line
        if notable_line and notable_line.lower() not in (
            "nothing unusual",
            "none",
            "n/a",
            "nothing unusual.",
        ):
            desc += " " + notable_line
    elif entities:
        if "DESCRIPTION" in entities:
            desc = entities.pop("DESCRIPTION")
        elif "SCENE" in entities:
            desc = entities.pop("SCENE")
        else:
            parts = []
            for p in entities.get("persons") or []:
                action = p.get("action", p.get("attire", "person"))
                parts.append(action)
            for v in entities.get("vehicles") or []:
                parts.append(
                    f"{v.get('color', '')} {v.get('make', '')} {v.get('action', v.get('type', ''))}".strip()
                )
            desc = (
                f"{entities.get('people_count', 0)} people, "
                f"{entities.get('vehicle_count', len(entities.get('vehicles', [])))} vehicles"
            )
            if parts:
                desc += ": " + ", ".join(parts[:5])
    else:
        desc = clean[:300] if clean else ""

    # Fallback: if no structured entities parsed, construct from description
    if not entities:
        entities = {}
        # Infer scene metadata from description text
        dl = desc.lower()
        if any(w in dl for w in ["bustling", "crowded", "busy", "packed", "dense"]):
            entities["activity_level"] = "busy"
        elif any(w in dl for w in ["moderate", "steady", "normal"]):
            entities["activity_level"] = "moderate"
        elif any(w in dl for w in ["quiet", "empty", "calm", "sparse", "deserted"]):
            entities["activity_level"] = "quiet"
        if any(w in dl for w in ["times square", "new york", "nyc"]):
            entities["scene_type"] = "Times Square plaza"
        elif any(w in dl for w in ["parking", "lot"]):
            entities["scene_type"] = "parking lot"
        elif any(w in dl for w in ["intersection", "crosswalk", "street"]):
            entities["scene_type"] = "urban street"
        elif any(w in dl for w in ["plaza", "square", "park"]):
            entities["scene_type"] = "public plaza"
        if any(w in dl for w in ["tense", "aggressive", "alarming"]):
            entities["mood"] = "tense"
        elif any(w in dl for w in ["active", "energetic", "bustling"]):
            entities["mood"] = "active"
        else:
            entities["mood"] = "calm"

    elapsed = int((time.time() - t0) * 1000)

    return CropDescribeResponse(
        description=desc,
        entities=entities,
        crop_descriptions=crop_descriptions,
        elapsed_ms=elapsed,
    )


@router.post("/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    """Run YOLO detection on an image. Returns counts + crop bounding boxes."""
    import asyncio

    frame = _decode_image(req.image_b64)
    counter = _get_yolo()

    t0 = time.time()
    loop = asyncio.get_event_loop()

    # Run YOLO
    det = counter.detector
    if counter._tiled:
        xyxy, confs, cids = await loop.run_in_executor(None, det.detect_tiled, frame)
    else:
        xyxy, confs, cids = await loop.run_in_executor(None, det.detect, frame)

    h, w = frame.shape[:2]
    COCO = {
        0: "person",
        1: "bicycle",
        2: "car",
        3: "motorcycle",
        5: "bus",
        7: "truck",
        16: "dog",
        17: "cat",
    }
    by_class = {}
    crops_b64 = []

    for i in range(len(xyxy)):
        cid = int(cids[i])
        name = COCO.get(cid, f"class_{cid}")
        by_class[name] = by_class.get(name, 0) + 1

        x1, y1, x2, y2 = xyxy[i].astype(int)
        pw = int((x2 - x1) * req.pad_ratio)
        ph_pad = int((y2 - y1) * req.pad_ratio)
        _cx1, _cy1 = max(0, x1 - pw), max(0, y1 - ph_pad)
        _cx2, _cy2 = min(w, x2 + pw), min(h, y2 + ph_pad)

        crops_b64.append(
            {
                "class": name,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(float(confs[i]), 3),
            }
        )

    people = by_class.get("person", 0)
    vehicles = sum(by_class.get(k, 0) for k in ["car", "bus", "truck", "motorcycle"])
    elapsed = int((time.time() - t0) * 1000)

    return DetectResponse(
        people_count=people,
        vehicle_count=vehicles,
        by_class=by_class,
        crops_b64=crops_b64,
        elapsed_ms=elapsed,
    )


@router.get("/status")
async def inference_status():
    """Check which models are loaded."""
    return {
        "vlm_loaded": _vlm_engine is not None,
        "vlm_model": _vlm_engine.config.model if _vlm_engine else None,
        "yolo_loaded": _yolo_counter is not None,
    }
