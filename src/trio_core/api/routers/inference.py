"""Shared inference API — VLM + YOLO as a service for all pipelines.

Loads models ONCE in the server process. Pipelines call these endpoints
instead of loading their own copies.

Memory savings: 3GB VLM + 0.3GB YOLO loaded once vs per-pipeline.
"""

from __future__ import annotations

import base64
import logging
import threading
import time

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("trio.inference")

router = APIRouter(prefix="/api/inference", tags=["inference"])

# Semaphore to serialize VLM requests — prevents thread exhaustion under concurrent load
import asyncio as _asyncio

_vlm_semaphore = _asyncio.Semaphore(1)

import re as _re

_THINK_RE = _re.compile(r"<think>.*?</think>", _re.DOTALL)


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

        config = EngineConfig()
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
    max_crops: int = Field(default=3, le=20)
    scene_prompt: str = Field(
        default="",
        max_length=4096,
        description="Custom scene prompt (optional, uses default if empty)",
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

    async with _vlm_semaphore:
        loop = asyncio.get_event_loop()
        t0 = time.time()

        def _sync_describe():
            frame = _decode_image(req.image_b64)
            frame_chw = _frame_to_chw(frame)
            engine = _get_vlm()
            return engine.analyze_frame(frame_chw, req.prompt)

        try:
            result = await loop.run_in_executor(None, _sync_describe)
        except Exception as e:
            logger.error("VLM describe failed: %s", e)
            raise HTTPException(status_code=503, detail=f"VLM inference error: {e}")
        elapsed = int((time.time() - t0) * 1000)

    return DescribeResponse(
        description=_strip_thinking(result.text or ""),
        elapsed_ms=elapsed,
    )


@router.post("/crop-describe", response_model=CropDescribeResponse)
async def crop_describe(req: CropDescribeRequest):
    """Crop-then-describe: describe individual crops + full scene.

    YOLO detects objects and crops them, then VLM describes each
    entity individually before generating a full scene description.
    """

    async with _vlm_semaphore:
        return await _crop_describe_inner(req)


async def _crop_describe_inner(req: CropDescribeRequest):
    import asyncio
    import json

    loop = asyncio.get_event_loop()
    frame = await loop.run_in_executor(None, _decode_image, req.image_b64)
    engine = _get_vlm()
    image_factor = getattr(getattr(engine, "_profile", None), "merge_factor", 32)

    t0 = time.time()
    loop = asyncio.get_event_loop()

    # Phase 1: Describe individual crops
    crop_descriptions = []
    crops_to_describe = sorted(req.crops, key=lambda c: c.get("confidence", 0), reverse=True)[
        : req.max_crops
    ]

    for crop_info in crops_to_describe:
        bbox = crop_info.get("bbox", [])
        obj_class = crop_info.get("class", "object")
        if len(bbox) != 4:
            continue

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        # Add padding
        pad = 0.15
        pw, ph = int((x2 - x1) * pad), int((y2 - y1) * pad)
        cx1, cy1 = max(0, x1 - pw), max(0, y1 - ph)
        cx2, cy2 = min(w, x2 + pw), min(h, y2 + ph)
        crop_bgr = frame[cy1:cy2, cx1:cx2]

        if crop_bgr.shape[0] < 8 or crop_bgr.shape[1] < 8:
            continue
        crop_bgr = _align_crop_to_model(crop_bgr, image_factor)

        # Build crop prompt
        if obj_class in ("car", "truck", "bus", "motorcycle"):
            prompt = (
                "Identify this vehicle in one short phrase. Include: "
                "color, make, model if visible, type (sedan/SUV/truck/van/pickup). "
                "Example: 'silver Toyota Camry sedan'. Answer ONLY the phrase."
            )
        elif obj_class == "person":
            prompt = (
                "Describe this person in one short phrase. Include: "
                "age, gender, clothing, items carried. "
                "Example: 'male 30s, blue polo, carrying laptop bag'. Answer ONLY the phrase."
            )
        else:
            prompt = f"Identify this {obj_class} in one short phrase."

        crop_chw = _frame_to_chw(crop_bgr)
        result = await loop.run_in_executor(None, engine.analyze_frame, crop_chw, prompt)
        text = _strip_thinking((result.text or "").strip())
        if "<" in text:
            text = text.split(">")[-1].strip()
        cd = f"{obj_class}: {text}" if text else f"{obj_class}: unidentified"
        crop_descriptions.append(cd)

    # Phase 2: Full scene understanding with crop context
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

    if crop_descriptions:
        scene_prompt = (
            "Detailed object identification:\n"
            + "\n".join(f"- {cd}" for cd in crop_descriptions)
            + "\n\nIncorporate these details.\n\n"
            + scene_prompt
        )

    frame_chw = _frame_to_chw(frame)
    result = await loop.run_in_executor(None, engine.analyze_frame, frame_chw, scene_prompt)
    text = _strip_thinking(result.text or "")

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
        if notable_line and notable_line.lower() not in ("nothing unusual", "none", "n/a", "nothing unusual."):
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
