"""Torch-free Qwen VL processor — pure numpy + PIL replacement.

Handles both Qwen2.5-VL (patch_size=14, CLIP mean/std) and
Qwen3-VL / Qwen3.5 (patch_size=16, 0.5/0.5 mean/std).

Usage:
    from trio_core.processors.qwen_vl import load_processor
    processor = load_processor("/path/to/model")
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Smart Resize ────────────────────────────────────────────────────────────


def smart_resize_image(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 28 * 28 * 1280,
) -> tuple[int, int]:
    """Resize dimensions maintaining aspect ratio within pixel budget."""
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme: {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def smart_resize_video(
    num_frames: int,
    height: int,
    width: int,
    temporal_factor: int = 2,
    factor: int = 32,
    min_pixels: int = 128 * 128,
    max_pixels: int = 16 * 16 * 2 * 2 * 2 * 6144,
) -> tuple[int, int]:
    """Resize video dimensions considering temporal dimension."""
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be >= factor:{factor}")
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"Aspect ratio too extreme: {max(height, width) / min(height, width)}")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = math.ceil(num_frames / temporal_factor) * temporal_factor
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


# ── Pixel Processing ────────────────────────────────────────────────────────


def _process_pixels_to_patches(
    images: list[np.ndarray],
    resized_height: int,
    resized_width: int,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    """Convert list of (C, H, W) float32 arrays into flattened patches.

    Returns (flatten_patches, (grid_t, grid_h, grid_w)).
    """
    patches = np.array(images)  # (T, C, H, W)

    # Temporal padding: repeat last frame to make T divisible by temporal_patch_size
    T = patches.shape[0]
    pad = (-T) % temporal_patch_size
    if pad > 0:
        repeats = np.repeat(patches[-1:], pad, axis=0)
        patches = np.concatenate([patches, repeats], axis=0)

    channel = patches.shape[1]
    grid_t = patches.shape[0] // temporal_patch_size
    grid_h = resized_height // patch_size
    grid_w = resized_width // patch_size

    patches = patches.reshape(
        grid_t,
        temporal_patch_size,
        channel,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    # (grid_t, gh//ms, gw//ms, ms, ms, C, tp, ps, ps)
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channel * temporal_patch_size * patch_size * patch_size,
    )
    return flatten_patches, (grid_t, grid_h, grid_w)


def _pil_to_processed(
    img: Image.Image,
    target_h: int,
    target_w: int,
    mean: list[float],
    std: list[float],
    rescale_factor: float = 1 / 255.0,
) -> np.ndarray:
    """PIL Image → resized, rescaled, normalized (C, H, W) float32 array."""
    img = img.convert("RGB")
    img = img.resize((target_w, target_h), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32)  # (H, W, 3)
    arr = arr * rescale_factor  # 0-1
    mean_arr = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
    std_arr = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    arr = (arr - mean_arr) / std_arr
    arr = arr.transpose(2, 0, 1)  # (C, H, W)
    return arr


# ── Qwen VL Processor ───────────────────────────────────────────────────────


class QwenVLProcessor:
    """Torch-free processor for Qwen2.5-VL / Qwen3-VL / Qwen3.5 models.

    Drop-in replacement for transformers AutoProcessor. Supports the same
    __call__ interface used by trio_core.backends.MLXBackend.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        # Image processor config
        image_patch_size: int = 14,
        image_temporal_patch_size: int = 2,
        image_merge_size: int = 2,
        image_min_pixels: int = 56 * 56,
        image_max_pixels: int = 28 * 28 * 1280,
        image_mean: list[float] | None = None,
        image_std: list[float] | None = None,
        # Video processor config
        video_patch_size: int | None = None,
        video_temporal_patch_size: int | None = None,
        video_merge_size: int | None = None,
        video_min_pixels: int | None = None,
        video_max_pixels: int | None = None,
        video_mean: list[float] | None = None,
        video_std: list[float] | None = None,
        video_fps: float = 2.0,
        video_min_frames: int = 4,
        video_max_frames: int = 768,
        # Model type hint
        is_qwen3: bool = False,
    ):
        self.tokenizer = tokenizer

        # Image config
        self.image_patch_size = image_patch_size
        self.image_temporal_patch_size = image_temporal_patch_size
        self.image_merge_size = image_merge_size
        self.image_min_pixels = image_min_pixels
        self.image_max_pixels = image_max_pixels
        self.image_mean = image_mean or [0.48145466, 0.4578275, 0.40821073]
        self.image_std = image_std or [0.26862954, 0.26130258, 0.27577711]

        # Video config (defaults to image config if not specified)
        self.video_patch_size = video_patch_size or image_patch_size
        self.video_temporal_patch_size = video_temporal_patch_size or image_temporal_patch_size
        self.video_merge_size = video_merge_size or image_merge_size
        self.video_min_pixels = video_min_pixels or (128 * 128)
        self.video_max_pixels = video_max_pixels or (16 * 16 * 2 * 2 * 2 * 6144)
        self.video_mean = video_mean or self.image_mean
        self.video_std = video_std or self.image_std
        self.video_fps = video_fps
        self.video_min_frames = video_min_frames
        self.video_max_frames = video_max_frames

        self.is_qwen3 = is_qwen3

        # Compatibility attributes used by backends.py
        self.merge_size = image_merge_size

        # Special tokens
        self.image_token = "<|image_pad|>"
        self.video_token = "<|video_pad|>"
        self.vision_start_token = "<|vision_start|>"
        self.vision_end_token = "<|vision_end|>"

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
        self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)

    # ── Chat template ────────────────────────────────────────────────────

    def apply_chat_template(self, messages, **kwargs):
        """Delegate to tokenizer's chat template."""
        return self.tokenizer.apply_chat_template(messages, **kwargs)

    # ── Image Processing ─────────────────────────────────────────────────

    def _process_images(self, images: list[Image.Image]) -> dict[str, np.ndarray]:
        """Process a list of PIL images → pixel_values + image_grid_thw."""
        ps = self.image_patch_size
        tp = self.image_temporal_patch_size
        ms = self.image_merge_size
        factor = ps * ms

        all_patches = []
        all_grids = []

        for img in images:
            img = img.convert("RGB")
            w, h = img.size

            target_h, target_w = smart_resize_image(
                h, w,
                factor=factor,
                min_pixels=self.image_min_pixels,
                max_pixels=self.image_max_pixels,
            )

            processed = _pil_to_processed(
                img, target_h, target_w,
                self.image_mean, self.image_std,
            )
            flat, grid = _process_pixels_to_patches(
                [processed], target_h, target_w, ps, tp, ms,
            )
            all_patches.append(flat)
            all_grids.append(list(grid))

        pixel_values = np.concatenate(all_patches, axis=0).astype(np.float32)
        image_grid_thw = np.array(all_grids, dtype=np.int64)
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw}

    # ── Video Processing ─────────────────────────────────────────────────

    def _process_videos(self, videos: list[list[Image.Image]]) -> dict[str, Any]:
        """Process video frames → pixel_values_videos + video_grid_thw."""
        ps = self.video_patch_size
        tp = self.video_temporal_patch_size
        ms = self.video_merge_size
        factor = ps * ms

        all_patches = []
        all_grids = []
        all_metadata = []

        for frames in videos:
            pil_frames = [f.convert("RGB") if isinstance(f, Image.Image) else f for f in frames]
            num_frames = len(pil_frames)
            w, h = pil_frames[0].size

            target_h, target_w = smart_resize_video(
                num_frames=num_frames,
                height=h,
                width=w,
                temporal_factor=tp,
                factor=factor,
                min_pixels=self.video_min_pixels,
                max_pixels=self.video_max_pixels,
            )

            processed = [
                _pil_to_processed(f, target_h, target_w, self.video_mean, self.video_std)
                for f in pil_frames
            ]

            flat, grid = _process_pixels_to_patches(
                processed, target_h, target_w, ps, tp, ms,
            )
            all_patches.append(flat)
            all_grids.append(list(grid))

            # Metadata for Qwen3 timestamps
            all_metadata.append({
                "num_frames": num_frames,
                "fps": 24.0,  # default assumption for PIL frame lists
                "frames_indices": list(range(num_frames)),
            })

        pixel_values_videos = np.concatenate(all_patches, axis=0).astype(np.float32)
        video_grid_thw = np.array(all_grids, dtype=np.int64)

        return {
            "pixel_values_videos": pixel_values_videos,
            "video_grid_thw": video_grid_thw,
            "_video_metadata": all_metadata,
        }

    # ── Token Replacement ────────────────────────────────────────────────

    def _replace_image_tokens(self, text: str, image_grid_thw: np.ndarray) -> str:
        """Replace <|image_pad|> with correct count based on grid_thw."""
        merge_length = self.image_merge_size ** 2
        idx = 0
        while self.image_token in text and idx < len(image_grid_thw):
            num_tokens = int(np.prod(image_grid_thw[idx])) // merge_length
            text = text.replace(
                self.image_token,
                "<|placeholder|>" * num_tokens,
                1,
            )
            idx += 1
        text = text.replace("<|placeholder|>", self.image_token)
        return text

    def _replace_video_tokens(
        self, text: str, video_grid_thw: np.ndarray, metadata: list[dict],
    ) -> str:
        """Replace <|video_pad|> with correct count, adding timestamps for Qwen3."""
        merge_length = self.video_merge_size ** 2
        idx = 0
        full_token = f"{self.vision_start_token}{self.video_token}{self.vision_end_token}"

        while self.video_token in text and idx < len(video_grid_thw):
            grid = video_grid_thw[idx]
            grid_t = int(grid[0])

            if self.is_qwen3 and full_token in text:
                # Qwen3 style: per-frame timestamps + vision tokens
                meta = metadata[idx]
                timestamps = self._calculate_timestamps(
                    meta["frames_indices"], meta["fps"], self.video_temporal_patch_size,
                )
                frame_seqlen = int(np.prod(grid[1:])) // merge_length
                video_placeholder = ""
                for frame_idx in range(grid_t):
                    t = timestamps[frame_idx] if frame_idx < len(timestamps) else 0.0
                    video_placeholder += f"<{t:.1f} seconds>"
                    video_placeholder += (
                        self.vision_start_token
                        + "<|placeholder|>" * frame_seqlen
                        + self.vision_end_token
                    )
                text = text.replace(full_token, video_placeholder, 1)
            else:
                # Qwen2.5 style: simple replacement
                num_tokens = int(np.prod(grid)) // merge_length
                text = text.replace(
                    self.video_token,
                    "<|placeholder|>" * num_tokens,
                    1,
                )
            idx += 1

        text = text.replace("<|placeholder|>", self.video_token)
        return text

    @staticmethod
    def _calculate_timestamps(
        indices: list[int], fps: float, temporal_patch_size: int = 2,
    ) -> list[float]:
        """Average timestamps within temporal patch groups (Qwen3 style)."""
        indices = list(indices)
        pad = (-len(indices)) % temporal_patch_size
        if pad > 0:
            indices.extend([indices[-1]] * pad)
        timestamps = [idx / fps for idx in indices]
        return [
            (timestamps[i] + timestamps[i + temporal_patch_size - 1]) / 2
            for i in range(0, len(timestamps), temporal_patch_size)
        ]

    # ── Main __call__ ────────────────────────────────────────────────────

    def __call__(
        self,
        text: list[str] | str | None = None,
        images: list[Image.Image] | None = None,
        videos: list[list[Image.Image]] | None = None,
        padding: bool = False,
        return_tensors: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Process text + images/videos → model inputs (numpy arrays).

        Args:
            text: List of pre-formatted text strings (after apply_chat_template).
            images: List of PIL images.
            videos: List of video frame lists (each video = list of PIL images).
            padding: Whether to pad sequences.
            return_tensors: "np" for numpy (default), ignored otherwise.

        Returns:
            Dict with input_ids, attention_mask, pixel_values/pixel_values_videos,
            image_grid_thw/video_grid_thw.
        """
        result = {}

        # Process images
        image_grid_thw = None
        if images is not None:
            img_out = self._process_images(images)
            result.update(img_out)
            image_grid_thw = img_out["image_grid_thw"]

        # Process videos
        video_grid_thw = None
        video_metadata = None
        if videos is not None:
            vid_out = self._process_videos(videos)
            video_grid_thw = vid_out["video_grid_thw"]
            video_metadata = vid_out.pop("_video_metadata")
            result.update(vid_out)

        # Token replacement in text
        if text is not None:
            if isinstance(text, str):
                text = [text]
            text = list(text)  # copy

            for i in range(len(text)):
                if image_grid_thw is not None:
                    text[i] = self._replace_image_tokens(text[i], image_grid_thw)
                if video_grid_thw is not None:
                    text[i] = self._replace_video_tokens(
                        text[i], video_grid_thw, video_metadata or [],
                    )

            # Tokenize
            tok_out = self.tokenizer(
                text, padding=padding, return_attention_mask=True,
            )
            result["input_ids"] = np.array(tok_out["input_ids"], dtype=np.int64)
            result["attention_mask"] = np.array(tok_out["attention_mask"], dtype=np.int64)

        return result


# ── Loader ───────────────────────────────────────────────────────────────────


def load_processor(model_path: str | Path) -> QwenVLProcessor:
    """Load a QwenVLProcessor from a local model directory.

    Reads preprocessor_config.json and processor_config.json to configure
    the processor. Uses AutoTokenizer for text (torch-free).

    Args:
        model_path: Path to the model directory (already downloaded).

    Returns:
        QwenVLProcessor instance.
    """
    from transformers import AutoTokenizer

    model_path = Path(model_path)

    # Load tokenizer (torch-free)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))

    # Read configs
    preprocessor_config = {}
    processor_config = {}
    video_config = {}

    pp_path = model_path / "preprocessor_config.json"
    if pp_path.exists():
        with open(pp_path) as f:
            preprocessor_config = json.load(f)

    pc_path = model_path / "processor_config.json"
    if pc_path.exists():
        with open(pc_path) as f:
            processor_config = json.load(f)

    vp_path = model_path / "video_preprocessor_config.json"
    if vp_path.exists():
        with open(vp_path) as f:
            video_config = json.load(f)

    # Determine model type
    proc_class = preprocessor_config.get("processor_class", "")
    is_qwen3 = "Qwen3" in proc_class

    # Image config from preprocessor_config.json or processor_config.json
    img_cfg = processor_config.get("image_processor", preprocessor_config)
    vid_cfg = processor_config.get("video_processor", video_config)

    # Extract image settings
    img_size = img_cfg.get("size", {})
    image_min_pixels = img_size.get("shortest_edge", img_cfg.get("min_pixels", 56 * 56))
    image_max_pixels = img_size.get("longest_edge", img_cfg.get("max_pixels", 28 * 28 * 1280))

    # Extract video settings
    vid_size = vid_cfg.get("size", {})
    video_min_pixels = vid_size.get("shortest_edge", vid_cfg.get("min_pixels", None))
    video_max_pixels = vid_size.get("longest_edge", vid_cfg.get("max_pixels", None))

    processor = QwenVLProcessor(
        tokenizer=tokenizer,
        image_patch_size=img_cfg.get("patch_size", 14),
        image_temporal_patch_size=img_cfg.get("temporal_patch_size", 2),
        image_merge_size=img_cfg.get("merge_size", 2),
        image_min_pixels=image_min_pixels,
        image_max_pixels=image_max_pixels,
        image_mean=img_cfg.get("image_mean"),
        image_std=img_cfg.get("image_std"),
        video_patch_size=vid_cfg.get("patch_size"),
        video_temporal_patch_size=vid_cfg.get("temporal_patch_size"),
        video_merge_size=vid_cfg.get("merge_size"),
        video_min_pixels=video_min_pixels,
        video_max_pixels=video_max_pixels,
        video_mean=vid_cfg.get("image_mean"),
        video_std=vid_cfg.get("image_std"),
        video_fps=vid_cfg.get("fps", 2.0),
        video_min_frames=vid_cfg.get("min_frames", 4),
        video_max_frames=vid_cfg.get("max_frames", 768),
        is_qwen3=is_qwen3,
    )

    logger.info("Loaded torch-free QwenVLProcessor (is_qwen3=%s)", is_qwen3)
    return processor
