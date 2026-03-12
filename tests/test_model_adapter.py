"""Tests for trio_core.model_adapter — ModelAdapter abstraction."""

from unittest.mock import MagicMock, PropertyMock
import pytest

pytest.importorskip("mlx.core", reason="MLX optional dependency not installed")

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
        assert PROFILES["qwen3-vl-4b"].supports_tome is False  # deepstack breaks ToMe

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


# ---------------------------------------------------------------------------
# get_adapter: additional detection paths
# ---------------------------------------------------------------------------

class TestGetAdapterExtra:

    def test_fastvlm_via_config_model_type(self):
        """Line 505-507: detect FastVLM via config.model_type."""
        model = MagicMock()
        # No vision_tower model_type match, no vt class name match
        vt = MagicMock()
        type(vt).__name__ = "SomeEncoder"
        vt.model_type = ""
        model.vision_tower = vt
        model.config.model_type = "fastvlm"
        adapter = get_adapter(model)
        assert isinstance(adapter, FastVLMAdapter)


# ---------------------------------------------------------------------------
# Base ModelAdapter methods
# ---------------------------------------------------------------------------

class TestBaseAdapterMethods:

    def test_compute_position_ids_returns_none(self):
        """Line 91: base compute_position_ids returns (None, None)."""
        model = MagicMock()
        # Use a concrete subclass that doesn't override compute_position_ids
        a = InternVLAdapter(model)
        pos, delta = a.compute_position_ids(
            input_ids=MagicMock(), attention_mask=MagicMock()
        )
        assert pos is None
        assert delta is None

    def test_apply_rope_at_layer_base(self):
        """Lines 99-103: base apply_rope_at_layer uses attn.rotary_emb."""
        model = MagicMock()
        a = InternVLAdapter(model)

        layer = MagicMock()
        rope_mock = MagicMock(return_value="rotated")
        layer.self_attn.rotary_emb = rope_mock
        q, k = MagicMock(), MagicMock()

        q_out, k_out = a.apply_rope_at_layer(q, k, None, None, layer, cache_offset=5)
        assert rope_mock.call_count == 2
        rope_mock.assert_any_call(q, offset=5)
        rope_mock.assert_any_call(k, offset=5)

    def test_apply_rope_at_layer_falls_back_to_rope_attr(self):
        """Lines 99-103: fallback to attn.rope when rotary_emb missing."""
        model = MagicMock()
        a = InternVLAdapter(model)

        layer = MagicMock()
        del layer.self_attn.rotary_emb  # remove rotary_emb
        rope_mock = MagicMock(return_value="rotated")
        layer.self_attn.rope = rope_mock

        q_out, k_out = a.apply_rope_at_layer(MagicMock(), MagicMock(), None, None, layer)
        assert rope_mock.call_count == 2

    def test_call_layer_base(self):
        """Line 107: base call_layer ignores position_ids."""
        model = MagicMock()
        a = InternVLAdapter(model)
        layer = MagicMock(return_value="output")
        result = a.call_layer(layer, "h", "mask", "cache", position_ids="pos")
        layer.assert_called_once_with("h", "mask", "cache")

    def test_get_vision_dtype_patch_embed_proj(self):
        """Lines 114-116: vision_tower.patch_embed.proj.weight.dtype."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float32
        del model.vision_model  # ensure fallback to vision_tower
        # Remove linear_patches so it takes proj path
        del model.vision_tower.patch_embed.linear_patches
        a = InternVLAdapter(model)
        assert a.get_vision_dtype() == mx.float32

    def test_get_vision_dtype_linear_patches(self):
        """Lines 117-118: vision_tower.patch_embed.linear_patches.weight.dtype."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower.patch_embed.proj  # no proj
        model.vision_tower.patch_embed.linear_patches.weight.dtype = mx.bfloat16
        del model.vision_model
        a = InternVLAdapter(model)
        assert a.get_vision_dtype() == mx.bfloat16

    def test_get_vision_dtype_model_patch_embed(self):
        """Lines 120-121: vision_tower.model.patch_embed.proj.weight.dtype."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower.patch_embed  # no direct patch_embed
        model.vision_tower.model.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_model
        a = InternVLAdapter(model)
        assert a.get_vision_dtype() == mx.float16

    def test_get_vision_dtype_fallback(self):
        """Lines 122-123: fallback returns mx.float16."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower.patch_embed
        del model.vision_tower.model
        del model.vision_model
        a = InternVLAdapter(model)
        assert a.get_vision_dtype() == mx.float16

    def test_get_vision_dtype_via_vision_model(self):
        """Lines 111-112: fallback to model.vision_model when no vision_tower."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower  # no vision_tower
        model.vision_model.patch_embed.proj.weight.dtype = mx.float32
        del model.vision_model.patch_embed.linear_patches
        a = InternVLAdapter(model)
        assert a.get_vision_dtype() == mx.float32


# ---------------------------------------------------------------------------
# Qwen25VLAdapter method tests
# ---------------------------------------------------------------------------

class TestQwen25Methods:

    def test_run_vision_encoder(self):
        """Lines 159-162: calls vision_tower with correct args."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model
        expected_hs = mx.zeros((10, 64))
        model.vision_tower.return_value = expected_hs

        a = Qwen25VLAdapter(model)
        pv = mx.ones((1, 3, 224, 224))
        grid = mx.array([[1, 28, 28]], dtype=mx.int32)
        result = a.run_vision_encoder(pv, grid_thw=grid)

        assert isinstance(result, VisionOutput)
        model.vision_tower.assert_called_once()
        call_args = model.vision_tower.call_args
        assert call_args[1]['output_hidden_states'] is False

    def test_merge_visual_features(self):
        """Lines 165-169: calls merge_input_ids_with_image_features."""
        import mlx.core as mx
        model = MagicMock()
        model.config.video_token_id = 100
        model.config.image_token_id = 101
        expected = mx.zeros((1, 20, 64))
        model.merge_input_ids_with_image_features.return_value = expected

        a = Qwen25VLAdapter(model)
        hs = mx.zeros((10, 64))
        text_embeds = mx.zeros((1, 15, 64))
        input_ids = mx.zeros((1, 15), dtype=mx.int32)
        result = a.merge_visual_features(hs, text_embeds, input_ids)

        assert isinstance(result, MergeResult)
        model.merge_input_ids_with_image_features.assert_called_once_with(
            101, 100, hs, text_embeds, input_ids,
        )

    def test_compute_position_ids(self):
        """Line 172: delegates to language_model.get_rope_index."""
        model = MagicMock()
        model.language_model.get_rope_index.return_value = ("pos", "delta")
        a = Qwen25VLAdapter(model)
        pos, delta = a.compute_position_ids(
            input_ids="ids", attention_mask="mask", image_grid_thw="grid"
        )
        model.language_model.get_rope_index.assert_called_once_with(
            "ids", attention_mask="mask", image_grid_thw="grid",
        )
        assert pos == "pos"
        assert delta == "delta"

    def test_apply_rope_at_layer(self):
        """Lines 177-181: uses rotary_emb + apply_multimodal_rotary_pos_emb."""
        import mlx.core as mx
        from unittest.mock import patch

        model = MagicMock()
        a = Qwen25VLAdapter(model)

        layer = MagicMock()
        cos_sin = (mx.ones((1, 4)), mx.ones((1, 4)))
        layer.self_attn.rotary_emb.return_value = cos_sin

        q = mx.zeros((1, 4, 2, 64))
        k = mx.zeros((1, 4, 2, 64))
        v = mx.zeros((1, 4, 2, 64))
        pos_ids = mx.zeros((3, 1, 4))

        with patch(
            "trio_core.mlx_utils.apply_multimodal_rotary_pos_emb",
            return_value=(q, k),
        ) as mock_rope:
            q_out, k_out = a.apply_rope_at_layer(q, k, v, pos_ids, layer)
            layer.self_attn.rotary_emb.assert_called_once_with(v, pos_ids)
            mock_rope.assert_called_once()

    def test_call_layer(self):
        """Line 184: passes position_ids to layer."""
        model = MagicMock()
        a = Qwen25VLAdapter(model)
        layer = MagicMock(return_value="out")
        result = a.call_layer(layer, "h", "mask", "cache", position_ids="pos")
        layer.assert_called_once_with("h", "mask", "cache", "pos")


# ---------------------------------------------------------------------------
# Qwen3VLAdapter method tests
# ---------------------------------------------------------------------------

class TestQwen3Methods:

    def test_run_vision_encoder_tuple(self):
        """Lines 221-231: vision_tower returns tuple (hs, deepstack)."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model

        hs = mx.zeros((10, 64))
        ds = [mx.zeros((5, 64))]
        model.vision_tower.return_value = (hs, ds)

        a = Qwen3VLAdapter(model)
        pv = mx.ones((1, 3, 224, 224))
        result = a.run_vision_encoder(pv, grid_thw=None)

        assert isinstance(result, VisionOutput)
        assert result.deepstack_embeds is ds

    def test_run_vision_encoder_non_tuple(self):
        """Lines 228-230: vision_tower returns plain tensor."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model

        hs = mx.zeros((10, 64))
        model.vision_tower.return_value = hs

        a = Qwen3VLAdapter(model)
        pv = mx.ones((1, 3, 224, 224))
        result = a.run_vision_encoder(pv, grid_thw=None)

        assert result.deepstack_embeds is None

    def test_run_vision_encoder_tuple_single(self):
        """Line 227: tuple with only one element."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model

        hs = mx.zeros((10, 64))
        model.vision_tower.return_value = (hs,)

        a = Qwen3VLAdapter(model)
        pv = mx.ones((1, 3, 224, 224))
        result = a.run_vision_encoder(pv, grid_thw=None)

        assert result.deepstack_embeds is None

    def test_merge_visual_features(self):
        """Lines 234-239: calls merge with (hs, text, ids, img_id, vid_id)."""
        import mlx.core as mx
        model = MagicMock()
        model.config.video_token_index = 42
        model.config.image_token_index = 43
        del model.config.video_token_id
        del model.config.image_token_id

        embeds = mx.zeros((1, 20, 64))
        mask = mx.ones((1, 20))
        model.merge_input_ids_with_image_features.return_value = (embeds, mask)

        a = Qwen3VLAdapter(model)
        hs = mx.zeros((10, 64))
        text_embeds = mx.zeros((1, 15, 64))
        input_ids = mx.zeros((1, 15), dtype=mx.int32)
        result = a.merge_visual_features(hs, text_embeds, input_ids)

        assert isinstance(result, MergeResult)
        assert result.image_mask is mask
        model.merge_input_ids_with_image_features.assert_called_once_with(
            hs, text_embeds, input_ids, 43, 42,
        )

    def test_compute_position_ids(self):
        """Line 242: delegates to language_model.get_rope_index."""
        model = MagicMock()
        model.language_model.get_rope_index.return_value = ("pos", "delta")
        a = Qwen3VLAdapter(model)
        pos, delta = a.compute_position_ids(
            input_ids="ids", attention_mask="mask",
        )
        assert pos == "pos"

    def test_apply_rope_at_layer(self):
        """Lines 247-251: uses rotary_emb + qwen3_vl apply_multimodal_rotary_pos_emb."""
        import mlx.core as mx
        from unittest.mock import patch

        model = MagicMock()
        a = Qwen3VLAdapter(model)

        layer = MagicMock()
        cos_sin = (mx.ones((1, 4)), mx.ones((1, 4)))
        layer.self_attn.rotary_emb.return_value = cos_sin

        q = mx.zeros((1, 4, 2, 64))
        k = mx.zeros((1, 4, 2, 64))
        v = mx.zeros((1, 4, 2, 64))
        pos_ids = mx.zeros((3, 1, 4))

        with patch(
            "trio_core.mlx_utils.apply_multimodal_rotary_pos_emb",
            return_value=(q, k),
        ) as mock_rope:
            q_out, k_out = a.apply_rope_at_layer(q, k, v, pos_ids, layer)
            mock_rope.assert_called_once()

    def test_call_layer(self):
        """Line 254: passes position_ids to layer."""
        model = MagicMock()
        a = Qwen3VLAdapter(model)
        layer = MagicMock(return_value="out")
        a.call_layer(layer, "h", "mask", "cache", position_ids="pos")
        layer.assert_called_once_with("h", "mask", "cache", "pos")


# ---------------------------------------------------------------------------
# InternVLAdapter method tests
# ---------------------------------------------------------------------------

class TestInternVLMethods:

    def test_run_vision_encoder(self):
        """Lines 286-301: full InternVL vision pipeline."""
        import mlx.core as mx
        from unittest.mock import patch

        model = MagicMock()
        # Setup get_vision_dtype path
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model

        # vision_model returns (hs, _, _)
        B, N_plus_cls, D = 2, 197, 64
        hs_with_cls = mx.ones((B, N_plus_cls, D))
        model.vision_model = MagicMock()
        model.vision_model.return_value = (hs_with_cls, None, None)
        model.downsample_ratio = 2

        # mlp1 layers
        mlp_layer = MagicMock(side_effect=lambda x: x)
        model.mlp1 = [mlp_layer]

        with patch(
            "trio_core.mlx_utils.pixel_shuffle",
            side_effect=lambda x, shuffle_ratio: x,
        ) as mock_ps:
            a = InternVLAdapter(model)
            pv = mx.ones((1, 2, 3, 224, 224))  # 5D input
            result = a.run_vision_encoder(pv)

            assert isinstance(result, VisionOutput)
            model.vision_model.assert_called_once()
            mock_ps.assert_called_once()
            mlp_layer.assert_called_once()

    def test_merge_visual_features_plain(self):
        """Lines 305-311: merge returns plain tensor."""
        import mlx.core as mx
        model = MagicMock()
        expected = mx.zeros((1, 20, 64))
        model._merge_input_ids_with_image_features.return_value = expected

        a = InternVLAdapter(model)
        hs = mx.zeros((10, 64))  # 2D input, gets [None] added
        result = a.merge_visual_features(hs, mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32))

        assert isinstance(result, MergeResult)
        assert result.image_mask is None

    def test_merge_visual_features_tuple(self):
        """Lines 309-310: merge returns tuple (embeds, mask)."""
        import mlx.core as mx
        model = MagicMock()
        embeds = mx.zeros((1, 20, 64))
        mask = mx.ones((1, 20))
        model._merge_input_ids_with_image_features.return_value = (embeds, mask)

        a = InternVLAdapter(model)
        hs = mx.zeros((1, 10, 64))  # 3D, no [None] needed
        result = a.merge_visual_features(hs, mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32))

        assert result.image_mask is mask


# ---------------------------------------------------------------------------
# LLaVAAdapter method tests
# ---------------------------------------------------------------------------

class TestLLaVAMethods:

    def test_get_visual_token_ids(self):
        """Lines 345-349."""
        model = MagicMock()
        model.config.image_token_index = 200
        del model.config.image_token_id
        a = LLaVAAdapter(model)
        vid, img = a.get_visual_token_ids()
        assert vid == img == 200

    def test_run_vision_encoder(self):
        """Lines 352-362: vision_tower → hidden_state[-1] → mm_projector."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed.linear_patches
        del model.vision_model

        hidden_state = [mx.zeros((2, 196, 64)), mx.ones((2, 196, 64))]
        # vision_tower returns (*_, hidden_state) via unpacking
        model.vision_tower.return_value = ("pooler", "last_hs", hidden_state)
        model.mm_projector.return_value = mx.ones((2, 196, 64))

        a = LLaVAAdapter(model)
        pv = mx.ones((2, 3, 224, 224))
        result = a.run_vision_encoder(pv)

        assert isinstance(result, VisionOutput)
        model.vision_tower.assert_called_once()
        model.mm_projector.assert_called_once()

    def test_merge_visual_features_plain(self):
        """Lines 366-372: merge returns plain tensor."""
        import mlx.core as mx
        model = MagicMock()
        expected = mx.zeros((1, 20, 64))
        model._prepare_inputs_for_multimodal.return_value = expected

        a = LLaVAAdapter(model)
        hs = mx.zeros((10, 64))
        result = a.merge_visual_features(hs, mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32))

        assert result.image_mask is None
        model._prepare_inputs_for_multimodal.assert_called_once()

    def test_merge_visual_features_tuple(self):
        """Lines 370-371: merge returns tuple."""
        import mlx.core as mx
        model = MagicMock()
        embeds = mx.zeros((1, 20, 64))
        mask = mx.ones((1, 20))
        model._prepare_inputs_for_multimodal.return_value = (embeds, mask)

        a = LLaVAAdapter(model)
        hs = mx.zeros((10, 64))
        result = a.merge_visual_features(hs, mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32))
        assert result.image_mask is mask


# ---------------------------------------------------------------------------
# FastVLMAdapter method tests
# ---------------------------------------------------------------------------

class TestFastVLMMethods:

    def test_get_visual_token_ids(self):
        """Lines 405-409."""
        model = MagicMock()
        model.config.image_token_index = 300
        del model.config.image_token_id
        a = FastVLMAdapter(model)
        vid, img = a.get_visual_token_ids()
        assert vid == img == 300

    def test_run_vision_encoder(self):
        """Lines 412-415: calls vision_tower without grid_thw."""
        import mlx.core as mx
        model = MagicMock()
        # FastVLM get_vision_dtype path
        model.vision_tower.model.patch_embed.proj.weight.dtype = mx.float16
        del model.vision_tower.patch_embed

        expected = mx.zeros((10, 64))
        model.vision_tower.return_value = expected

        a = FastVLMAdapter(model)
        pv = mx.ones((1, 3, 224, 224))
        result = a.run_vision_encoder(pv)

        assert isinstance(result, VisionOutput)
        model.vision_tower.assert_called_once()
        call_kwargs = model.vision_tower.call_args[1]
        assert call_kwargs['output_hidden_states'] is False

    def test_merge_visual_features_plain(self):
        """Lines 418-424: merge returns plain tensor."""
        import mlx.core as mx
        model = MagicMock()
        model.config.image_token_index = 300
        del model.config.image_token_id
        expected = mx.zeros((1, 20, 64))
        model.merge_input_ids_with_image_features.return_value = expected

        a = FastVLMAdapter(model)
        result = a.merge_visual_features(
            mx.zeros((10, 64)), mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32)
        )
        assert result.image_mask is None

    def test_merge_visual_features_tuple(self):
        """Lines 422-423: merge returns tuple."""
        import mlx.core as mx
        model = MagicMock()
        model.config.image_token_index = 300
        del model.config.image_token_id
        embeds = mx.zeros((1, 20, 64))
        mask = mx.ones((1, 20))
        model.merge_input_ids_with_image_features.return_value = (embeds, mask)

        a = FastVLMAdapter(model)
        result = a.merge_visual_features(
            mx.zeros((10, 64)), mx.zeros((1, 15, 64)), mx.zeros((1, 15), dtype=mx.int32)
        )
        assert result.image_mask is mask

    def test_get_vision_dtype_model_patch_embed(self):
        """Lines 430-433: vt.model.patch_embed.proj path."""
        import mlx.core as mx
        model = MagicMock()
        model.vision_tower.model.patch_embed.proj.weight.dtype = mx.float32
        del model.vision_tower.patch_embed

        a = FastVLMAdapter(model)
        assert a.get_vision_dtype() == mx.float32

    def test_get_vision_dtype_direct_patch_embed(self):
        """Lines 434-436: vt.patch_embed.proj path."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower.model  # no model sub-attr
        model.vision_tower.patch_embed.proj.weight.dtype = mx.bfloat16

        a = FastVLMAdapter(model)
        assert a.get_vision_dtype() == mx.bfloat16

    def test_get_vision_dtype_fallback(self):
        """Line 438: fallback to mx.float16."""
        import mlx.core as mx
        model = MagicMock()
        del model.vision_tower.model
        del model.vision_tower.patch_embed

        a = FastVLMAdapter(model)
        assert a.get_vision_dtype() == mx.float16

    def test_original_token_count(self):
        """Lines 441-447: no spatial merge."""
        import mlx.core as mx
        model = MagicMock()
        a = FastVLMAdapter(model)
        grid = mx.array([[2, 14, 14]], dtype=mx.int32)
        assert a.original_token_count(grid) == 2 * 14 * 14
