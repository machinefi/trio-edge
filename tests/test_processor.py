"""Verify custom QwenVLProcessor matches transformers AutoProcessor output."""

import numpy as np
import pytest

pytest.importorskip("tokenizers")

from PIL import Image


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
        from pathlib import Path

        import huggingface_hub

        from trio_core.processors.qwen_vl import load_processor

        model_path = Path(huggingface_hub.snapshot_download("mlx-community/Qwen3.5-2B-MLX-4bit"))
        processor = load_processor(model_path)

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
        from pathlib import Path

        import huggingface_hub

        from trio_core.processors.qwen_vl import load_processor

        model_path = Path(huggingface_hub.snapshot_download("mlx-community/Qwen3.5-2B-MLX-4bit"))
        processor = load_processor(model_path)

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
        from pathlib import Path

        import huggingface_hub

        from trio_core.processors.qwen_vl import load_processor

        model_path = Path(huggingface_hub.snapshot_download("mlx-community/Qwen3.5-2B-MLX-4bit"))
        processor = load_processor(model_path)

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
