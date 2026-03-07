"""Tests for trio_core.model_adapter — ModelAdapter abstraction."""

from unittest.mock import MagicMock, PropertyMock
import pytest

from trio_core.model_adapter import (
    ModelAdapter, Qwen25VLAdapter, Qwen3VLAdapter,
    InternVLAdapter, LLaVAAdapter, FastVLMAdapter,
    get_adapter, VisionOutput, MergeResult,
)


# ---------------------------------------------------------------------------
# Adapter detection via get_adapter()
# ---------------------------------------------------------------------------

class TestGetAdapter:
    """Test auto-detection of model family."""

    def _mock_model(self, vt_model_type="", vt_class_name="VisionModel",
                    config_model_type=""):
        model = MagicMock()
        vt = MagicMock()
        type(vt).__name__ = vt_class_name
        vt.model_type = vt_model_type
        model.vision_tower = vt
        model.config.model_type = config_model_type
        return model

    def test_qwen25_vl(self):
        model = self._mock_model(vt_model_type="qwen2_5_vl")
        adapter = get_adapter(model)
        assert isinstance(adapter, Qwen25VLAdapter)
        assert adapter.family == "qwen2.5-vl"

    def test_qwen3_vl(self):
        model = self._mock_model(vt_model_type="qwen3_vl")
        adapter = get_adapter(model)
        assert isinstance(adapter, Qwen3VLAdapter)
        assert adapter.family == "qwen3-vl"

    def test_qwen35(self):
        model = self._mock_model(vt_model_type="qwen3_5")
        adapter = get_adapter(model)
        assert isinstance(adapter, Qwen3VLAdapter)

    def test_qwen35_moe(self):
        model = self._mock_model(vt_model_type="qwen3_5_moe")
        adapter = get_adapter(model)
        assert isinstance(adapter, Qwen3VLAdapter)

    def test_internvl_by_config(self):
        model = self._mock_model(config_model_type="internvl_chat")
        adapter = get_adapter(model)
        assert isinstance(adapter, InternVLAdapter)
        assert adapter.family == "internvl"

    def test_internvl_by_vt_type(self):
        model = self._mock_model(vt_class_name="InternVisionModel")
        adapter = get_adapter(model)
        assert isinstance(adapter, InternVLAdapter)

    def test_fastvlm_by_vt_type(self):
        model = self._mock_model(vt_class_name="FastVLMVisionEncoder")
        adapter = get_adapter(model)
        assert isinstance(adapter, FastVLMAdapter)
        assert adapter.family == "fastvlm"

    def test_siglip_by_vt_type(self):
        model = self._mock_model(vt_class_name="SiglipVisionModel")
        adapter = get_adapter(model)
        assert isinstance(adapter, LLaVAAdapter)
        assert adapter.family == "nanollava"

    def test_llava_by_config(self):
        model = self._mock_model(config_model_type="bunny-llava")
        adapter = get_adapter(model)
        assert isinstance(adapter, LLaVAAdapter)

    def test_default_fallback(self):
        model = self._mock_model()
        adapter = get_adapter(model)
        assert isinstance(adapter, Qwen25VLAdapter)


# ---------------------------------------------------------------------------
# Adapter properties
# ---------------------------------------------------------------------------

class TestAdapterProperties:

    def _make_adapter(self, cls):
        model = MagicMock()
        return cls(model)

    def test_qwen25_properties(self):
        a = self._make_adapter(Qwen25VLAdapter)
        assert a.spatial_merge_size == 2
        assert a.uses_mrope is True
        assert a.supports_tome is True
        assert a.has_deepstack is False

    def test_qwen3_properties(self):
        a = self._make_adapter(Qwen3VLAdapter)
        assert a.spatial_merge_size == 2
        assert a.uses_mrope is True
        assert a.supports_tome is True
        assert a.has_deepstack is True

    def test_internvl_properties(self):
        a = self._make_adapter(InternVLAdapter)
        assert a.spatial_merge_size == 1
        assert a.uses_mrope is False
        assert a.supports_tome is False
        assert a.has_deepstack is False

    def test_llava_properties(self):
        a = self._make_adapter(LLaVAAdapter)
        assert a.spatial_merge_size == 1
        assert a.uses_mrope is False
        assert a.supports_tome is True
        assert a.has_deepstack is False

    def test_fastvlm_properties(self):
        a = self._make_adapter(FastVLMAdapter)
        assert a.spatial_merge_size == 1
        assert a.uses_mrope is False
        assert a.supports_tome is False
        assert a.has_deepstack is False


# ---------------------------------------------------------------------------
# Token count computation
# ---------------------------------------------------------------------------

class TestOriginalTokenCount:
    """Test original_token_count with different spatial_merge_size values."""

    def test_qwen_spatial_merge_2(self):
        import mlx.core as mx
        model = MagicMock()
        a = Qwen25VLAdapter(model)
        # t=1, h=28, w=28 → 1 * (28//2) * (28//2) = 196
        grid = mx.array([[1, 28, 28]], dtype=mx.int32)
        assert a.original_token_count(grid) == 196

    def test_internvl_spatial_merge_1(self):
        import mlx.core as mx
        model = MagicMock()
        a = InternVLAdapter(model)
        # t=1, h=28, w=28 → 1 * 28 * 28 = 784 (no spatial merge)
        grid = mx.array([[1, 28, 28]], dtype=mx.int32)
        assert a.original_token_count(grid) == 784

    def test_llava_spatial_merge_1(self):
        import mlx.core as mx
        model = MagicMock()
        a = LLaVAAdapter(model)
        grid = mx.array([[1, 14, 14]], dtype=mx.int32)
        assert a.original_token_count(grid) == 196


# ---------------------------------------------------------------------------
# Visual token IDs
# ---------------------------------------------------------------------------

class TestVisualTokenIds:

    def test_qwen25_token_ids(self):
        model = MagicMock()
        model.config.video_token_id = 151656
        model.config.image_token_id = 151655
        a = Qwen25VLAdapter(model)
        vid, img = a.get_visual_token_ids()
        assert vid == 151656
        assert img == 151655

    def test_qwen3_token_index_fallback(self):
        """Qwen3 uses video_token_index/image_token_index."""
        model = MagicMock()
        model.config.video_token_index = 42
        model.config.image_token_index = 43
        # Make video_token_id not exist
        del model.config.video_token_id
        del model.config.image_token_id
        a = Qwen3VLAdapter(model)
        vid, img = a.get_visual_token_ids()
        assert vid == 42
        assert img == 43

    def test_internvl_same_token(self):
        """InternVL uses same token for video and image."""
        model = MagicMock()
        model.config.image_token_index = 100
        a = InternVLAdapter(model)
        vid, img = a.get_visual_token_ids()
        assert vid == img == 100


# ---------------------------------------------------------------------------
# Profiles supports_tome field
# ---------------------------------------------------------------------------

class TestProfilesSupportsTome:
    """Verify supports_tome is correctly set for each model family."""

    def test_qwen_supports_tome(self):
        from trio_core.profiles import PROFILES
        assert PROFILES["qwen2.5-vl-3b"].supports_tome is True
        assert PROFILES["qwen3.5-0.8b"].supports_tome is True
        assert PROFILES["qwen3-vl-4b"].supports_tome is True

    def test_internvl_no_tome(self):
        from trio_core.profiles import PROFILES
        assert PROFILES["internvl3-1b"].supports_tome is False
        assert PROFILES["internvl3-2b"].supports_tome is False

    def test_fastvlm_no_tome(self):
        from trio_core.profiles import PROFILES
        assert PROFILES["fastvlm-0.5b"].supports_tome is False
        assert PROFILES["fastvlm-1.5b"].supports_tome is False

    def test_nanollava_supports_tome(self):
        from trio_core.profiles import PROFILES
        assert PROFILES["nanollava-1.5"].supports_tome is True

    def test_default_supports_tome(self):
        """Models without explicit supports_tome should default to True."""
        from trio_core.profiles import PROFILES
        assert PROFILES["gemma3-4b"].supports_tome is True
        assert PROFILES["phi4-multimodal"].supports_tome is True


# ---------------------------------------------------------------------------
# VisionOutput and MergeResult dataclasses
# ---------------------------------------------------------------------------

class TestDataClasses:

    def test_vision_output_defaults(self):
        import mlx.core as mx
        hs = mx.zeros((10, 64))
        vo = VisionOutput(hidden_states=hs)
        assert vo.deepstack_embeds is None

    def test_vision_output_with_deepstack(self):
        import mlx.core as mx
        hs = mx.zeros((10, 64))
        ds = [mx.zeros((5, 64))]
        vo = VisionOutput(hidden_states=hs, deepstack_embeds=ds)
        assert len(vo.deepstack_embeds) == 1

    def test_merge_result_defaults(self):
        import mlx.core as mx
        embeds = mx.zeros((1, 20, 64))
        mr = MergeResult(embeds=embeds)
        assert mr.image_mask is None


# ---------------------------------------------------------------------------
# create_tome_vision rejects unsupported architectures
# ---------------------------------------------------------------------------

class TestCreateTomeVisionRejects:

    def test_rejects_internvit(self):
        from trio_core.native_vision import create_tome_vision

        class FakeInternViT:
            model_type = ""
        vt = FakeInternViT()

        with pytest.raises(ValueError, match="InternViT"):
            create_tome_vision(vt, tome_r=4)

    def test_rejects_fastvlm(self):
        from trio_core.native_vision import create_tome_vision

        class FakeFastVLMEncoder:
            model_type = ""
        vt = FakeFastVLMEncoder()

        with pytest.raises(ValueError, match="FastVLM"):
            create_tome_vision(vt, tome_r=4)
