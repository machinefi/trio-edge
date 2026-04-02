"""Verify custom QwenVLProcessor matches transformers AutoProcessor output."""

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from PIL import Image


class _FakeTokenizer:
    def __init__(self):
        self._ids = {
            "<|image_pad|>": 11,
            "<|video_pad|>": 12,
            "<|vision_start|>": 13,
            "<|vision_end|>": 14,
            "<|im_start|>": 15,
            "<|im_end|>": 16,
        }
        self._next_id = 100

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._ids.get(token)

    def _encode_one(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        specials = sorted(self._ids, key=len, reverse=True)
        while i < len(text):
            for token in specials:
                if text.startswith(token, i):
                    ids.append(self._ids[token])
                    i += len(token)
                    break
            else:
                if not text[i].isspace():
                    ids.append(self._next_id)
                    self._next_id += 1
                i += 1
        return ids

    def __call__(self, text, padding=False, return_attention_mask=False):
        texts = [text] if isinstance(text, str) else list(text)
        all_ids = [self._encode_one(item) for item in texts]
        if padding:
            max_len = max(len(ids) for ids in all_ids)
            padded = []
            masks = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded.append(ids + [0] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
        else:
            padded = all_ids
            masks = [[1] * len(ids) for ids in all_ids]
        result = {"input_ids": padded}
        if return_attention_mask:
            result["attention_mask"] = masks
        return result


def make_processor():
    from trio_core.processors.qwen_vl import QwenVLProcessor

    return QwenVLProcessor(
        tokenizer=_FakeTokenizer(),
        image_patch_size=16,
        image_temporal_patch_size=2,
        image_merge_size=2,
        image_min_pixels=65536,
        image_max_pixels=16777216,
        image_mean=[0.5, 0.5, 0.5],
        image_std=[0.5, 0.5, 0.5],
        video_patch_size=16,
        video_temporal_patch_size=2,
        video_merge_size=2,
        video_min_pixels=4096,
        video_max_pixels=25165824,
        video_mean=[0.5, 0.5, 0.5],
        video_std=[0.5, 0.5, 0.5],
        is_qwen3=True,
    )


# Test with a simple synthetic image
def make_test_image(h=480, w=640):
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_test_frames(n=4, h=480, w=640):
    return [make_test_image(h, w) for _ in range(n)]


class TestQwenVLProcessor:
    """Test custom processor against known expected behavior."""

    def test_smart_resize_image_basic(self):
        from trio_core.processors.qwen_vl import smart_resize_image

        # 640x480 with factor=28 (Qwen2.5)
        h, w = smart_resize_image(480, 640, factor=28, min_pixels=3136, max_pixels=12845056)
        assert h % 28 == 0
        assert w % 28 == 0
        assert h * w >= 3136
        assert h * w <= 12845056

    def test_smart_resize_image_qwen35(self):
        from trio_core.processors.qwen_vl import smart_resize_image

        # 640x480 with factor=32 (Qwen3.5)
        h, w = smart_resize_image(480, 640, factor=32, min_pixels=65536, max_pixels=16777216)
        assert h % 32 == 0
        assert w % 32 == 0

    def test_smart_resize_video(self):
        from trio_core.processors.qwen_vl import smart_resize_video

        h, w = smart_resize_video(
            4, 480, 640, temporal_factor=2, factor=32, min_pixels=4096, max_pixels=25165824
        )
        assert h % 32 == 0
        assert w % 32 == 0

    def test_image_processing_shapes(self):
        processor = make_processor()

        img = make_test_image(480, 640)
        text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe this image.<|im_end|>\n<|im_start|>assistant\n"

        result = processor(text=[text], images=[img], padding=True)

        assert "input_ids" in result
        assert "attention_mask" in result
        assert "pixel_values" in result
        assert "image_grid_thw" in result
        assert result["pixel_values"].dtype == np.float32
        assert result["image_grid_thw"].shape == (1, 3)
        # Verify patch embedding dim = 3 * temporal_patch_size * patch_size^2
        embed_dim = 3 * 2 * 16 * 16  # = 1536
        assert result["pixel_values"].shape[1] == embed_dim

    def test_video_processing_shapes(self):
        processor = make_processor()

        frames = make_test_frames(4, 480, 640)
        text = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Describe this video.<|im_end|>\n<|im_start|>assistant\n"

        result = processor(text=[text], videos=[frames], padding=True)

        assert "input_ids" in result
        assert "pixel_values_videos" in result
        assert "video_grid_thw" in result
        assert result["pixel_values_videos"].dtype == np.float32
        assert result["video_grid_thw"].shape == (1, 3)
        embed_dim = 3 * 2 * 16 * 16  # = 1536
        assert result["pixel_values_videos"].shape[1] == embed_dim

    def test_token_count_matches_grid(self):
        """Verify the number of pad tokens in input_ids matches grid_thw."""
        processor = make_processor()

        img = make_test_image(480, 640)
        text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"

        result = processor(text=[text], images=[img], padding=True)

        # Count image_pad tokens in input_ids
        image_pad_id = processor.image_token_id
        n_pads = np.sum(result["input_ids"] == image_pad_id)

        # Expected from grid
        grid = result["image_grid_thw"][0]
        expected = int(np.prod(grid)) // (processor.image_merge_size**2)

        assert n_pads == expected, f"Token count {n_pads} != expected {expected} from grid {grid}"


@pytest.mark.skipif(True, reason="Run manually: compares custom vs transformers processor")
class TestProcessorComparison:
    """Compare custom processor output with transformers AutoProcessor."""

    def test_pixel_values_match(self):
        """Verify pixel values are numerically close to transformers output."""
        from pathlib import Path

        import huggingface_hub
        from transformers import AutoProcessor

        from trio_core.processors.qwen_vl import load_processor

        model_path = Path(huggingface_hub.snapshot_download("mlx-community/Qwen3.5-2B-MLX-4bit"))

        # Load both processors
        custom = load_processor(model_path)
        reference = AutoProcessor.from_pretrained(str(model_path))

        img = make_test_image(480, 640)
        make_test_frames(4, 480, 640)

        # Compare image processing
        text = "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe.<|im_end|>\n<|im_start|>assistant\n"

        custom_out = custom(text=[text], images=[img], padding=True)
        ref_out = reference(text=[text], images=[img], padding=True, return_tensors="np")

        assert custom_out["pixel_values"].shape == ref_out["pixel_values"].shape, (
            f"Shape mismatch: {custom_out['pixel_values'].shape} vs {ref_out['pixel_values'].shape}"
        )
        np.testing.assert_allclose(
            custom_out["pixel_values"],
            ref_out["pixel_values"],
            rtol=1e-5,
            atol=1e-5,
        )
