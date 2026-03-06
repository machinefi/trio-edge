"""Tests for trio_core.tome_backend and trio_core.tome_vision — integration layer."""

import mlx.core as mx
import pytest

from trio_core.tome_vision import BaseToMeVisionWrapper


# ---------------------------------------------------------------------------
# ToMeMLXBackend.__init__ validation
# ---------------------------------------------------------------------------

class TestToMeMLXBackendValidation:
    """Test __init__ parameter validation without loading a real model.

    We patch super().__init__ to avoid needing a real model_name on disk.
    """

    def _make_backend(self, **kwargs):
        """Create a ToMeMLXBackend with a patched parent __init__."""
        from unittest.mock import patch
        from trio_core.tome_backend import ToMeMLXBackend

        with patch.object(ToMeMLXBackend.__bases__[0], "__init__", return_value=None):
            return ToMeMLXBackend("fake-model", **kwargs)

    def test_negative_tome_r_raises(self):
        with pytest.raises(ValueError, match="tome_r must be >= 0"):
            self._make_backend(tome_r=-1)

    def test_zero_tome_r_ok(self):
        backend = self._make_backend(tome_r=0)
        assert backend.tome_r == 0

    def test_bad_metric_raises(self):
        with pytest.raises(ValueError, match="metric must be"):
            self._make_backend(metric="cosine")

    def test_valid_metrics_accepted(self):
        for m in ("keys", "hidden"):
            backend = self._make_backend(metric=m)
            assert backend.tome_metric == m

    def test_min_keep_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="min_keep_ratio"):
            self._make_backend(min_keep_ratio=0.0)

    def test_min_keep_ratio_negative_raises(self):
        with pytest.raises(ValueError, match="min_keep_ratio"):
            self._make_backend(min_keep_ratio=-0.5)

    def test_min_keep_ratio_above_one_raises(self):
        with pytest.raises(ValueError, match="min_keep_ratio"):
            self._make_backend(min_keep_ratio=1.5)

    def test_min_keep_ratio_one_ok(self):
        backend = self._make_backend(min_keep_ratio=1.0)
        assert backend.tome_min_keep_ratio == 1.0

    def test_min_keep_ratio_normal_ok(self):
        backend = self._make_backend(min_keep_ratio=0.3)
        assert backend.tome_min_keep_ratio == 0.3


# ---------------------------------------------------------------------------
# ToMeMLXBackend._original_token_count
# ---------------------------------------------------------------------------

class TestOriginalTokenCount:
    def _count(self, grid_thw_list):
        from trio_core.tome_backend import ToMeMLXBackend
        grid_thw = mx.array(grid_thw_list, dtype=mx.int32)
        return ToMeMLXBackend._original_token_count(grid_thw)

    def test_single_image(self):
        # t=1, h=28, w=28 → spatial_merge=2 → 1 * 14 * 14 = 196
        assert self._count([[1, 28, 28]]) == 196

    def test_video_multiple_frames(self):
        # t=4, h=14, w=14 → 4 * 7 * 7 = 196
        assert self._count([[4, 14, 14]]) == 196

    def test_multiple_batches(self):
        # Two entries: (1, 28, 28) → 196, (2, 14, 14) → 2*7*7=98, total=294
        assert self._count([[1, 28, 28], [2, 14, 14]]) == 294

    def test_small_grid(self):
        # t=1, h=4, w=4 → 1 * 2 * 2 = 4
        assert self._count([[1, 4, 4]]) == 4

    def test_rectangular_grid(self):
        # t=2, h=28, w=14 → 2 * 14 * 7 = 196
        assert self._count([[2, 28, 14]]) == 196


# ---------------------------------------------------------------------------
# ToMeMLXBackend._get_visual_token_ids
# ---------------------------------------------------------------------------

class TestGetVisualTokenIds:
    """Test Qwen2.5 vs Qwen3 config attribute differences."""

    def _make_backend_with_qwen_flag(self, is_qwen3):
        from unittest.mock import patch
        from trio_core.tome_backend import ToMeMLXBackend

        with patch.object(ToMeMLXBackend.__bases__[0], "__init__", return_value=None):
            backend = ToMeMLXBackend("fake-model")
        backend._is_qwen3 = is_qwen3
        return backend

    def test_qwen25_uses_direct_token_id(self):
        """Qwen2.5 reads config.video_token_id and config.image_token_id directly."""
        from unittest.mock import MagicMock

        backend = self._make_backend_with_qwen_flag(is_qwen3=False)
        model = MagicMock()
        model.config.video_token_id = 151652
        model.config.image_token_id = 151655

        vid_id, img_id = backend._get_visual_token_ids(model)
        assert vid_id == 151652
        assert img_id == 151655

    def test_qwen3_uses_token_index_fallback(self):
        """Qwen3 tries video_token_index first, falls back to video_token_id."""
        from unittest.mock import MagicMock

        backend = self._make_backend_with_qwen_flag(is_qwen3=True)
        model = MagicMock()
        # Qwen3 has token_index attributes
        model.config.video_token_index = 151900
        model.config.image_token_index = 151901
        model.config.video_token_id = None
        model.config.image_token_id = None

        vid_id, img_id = backend._get_visual_token_ids(model)
        assert vid_id == 151900
        assert img_id == 151901

    def test_qwen3_fallback_to_token_id(self):
        """Qwen3 falls back to video_token_id when video_token_index is missing."""
        from unittest.mock import MagicMock

        backend = self._make_backend_with_qwen_flag(is_qwen3=True)
        model = MagicMock()
        # No token_index attributes
        model.config.video_token_index = None
        model.config.image_token_index = None
        model.config.video_token_id = 151652
        model.config.image_token_id = 151655

        vid_id, img_id = backend._get_visual_token_ids(model)
        assert vid_id == 151652
        assert img_id == 151655


# ---------------------------------------------------------------------------
# ToMeMLXBackend.backend_name
# ---------------------------------------------------------------------------

class TestBackendName:
    def test_backend_name_returns_mlx_tome(self):
        from unittest.mock import patch
        from trio_core.tome_backend import ToMeMLXBackend

        with patch.object(ToMeMLXBackend.__bases__[0], "__init__", return_value=None):
            backend = ToMeMLXBackend("fake-model")
        assert backend.backend_name == "mlx-tome"


# ---------------------------------------------------------------------------
# BaseToMeVisionWrapper._should_merge
# ---------------------------------------------------------------------------

class TestShouldMerge:
    def _make_wrapper(self, skip_first=2, skip_last=2, r=8):
        """Create a BaseToMeVisionWrapper with a dummy vision_model."""
        from unittest.mock import MagicMock
        vm = MagicMock()
        return BaseToMeVisionWrapper(
            vm, r=r, skip_first=skip_first, skip_last=skip_last,
        )

    def test_skip_first_layers(self):
        wrapper = self._make_wrapper(skip_first=2, skip_last=2)
        n_blocks = 10
        assert not wrapper._should_merge(0, n_blocks)
        assert not wrapper._should_merge(1, n_blocks)

    def test_skip_last_layers(self):
        wrapper = self._make_wrapper(skip_first=2, skip_last=2)
        n_blocks = 10
        assert not wrapper._should_merge(8, n_blocks)
        assert not wrapper._should_merge(9, n_blocks)

    def test_middle_layers_allowed(self):
        wrapper = self._make_wrapper(skip_first=2, skip_last=2)
        n_blocks = 10
        for layer in range(2, 8):
            assert wrapper._should_merge(layer, n_blocks), f"Layer {layer} should merge"

    def test_r_zero_disables_all(self):
        wrapper = self._make_wrapper(skip_first=2, skip_last=2, r=0)
        n_blocks = 10
        for layer in range(n_blocks):
            assert not wrapper._should_merge(layer, n_blocks), (
                f"Layer {layer} should NOT merge when r=0"
            )

    def test_all_skipped_when_too_few_blocks(self):
        """With skip_first=2, skip_last=2, n_blocks=4 → layers 2,3 are in skip_last zone."""
        wrapper = self._make_wrapper(skip_first=2, skip_last=2)
        n_blocks = 4
        for layer in range(n_blocks):
            assert not wrapper._should_merge(layer, n_blocks)

    def test_boundary_layer_first_mergeable(self):
        """Layer skip_first should be the first mergeable layer."""
        wrapper = self._make_wrapper(skip_first=3, skip_last=1)
        n_blocks = 10
        assert not wrapper._should_merge(2, n_blocks)
        assert wrapper._should_merge(3, n_blocks)

    def test_boundary_layer_last_mergeable(self):
        """Layer n_blocks - skip_last - 1 should be the last mergeable layer."""
        wrapper = self._make_wrapper(skip_first=1, skip_last=3)
        n_blocks = 10
        assert wrapper._should_merge(6, n_blocks)  # 10 - 3 - 1 = 6
        assert not wrapper._should_merge(7, n_blocks)


# ---------------------------------------------------------------------------
# BaseToMeVisionWrapper._get_metric
# ---------------------------------------------------------------------------

class TestGetMetric:
    def test_hidden_metric_returns_hidden_states(self):
        """When metric='hidden', _get_metric returns hidden_states directly."""
        import mlx.nn as nn
        from unittest.mock import MagicMock

        vm = MagicMock()
        wrapper = BaseToMeVisionWrapper(vm, metric="hidden")

        hidden = mx.random.normal((8, 32))
        block = MagicMock()
        result = wrapper._get_metric(hidden, block, rotary_pos_emb=None)

        assert mx.array_equal(result, hidden)

    def test_keys_metric_calls_compute_k(self):
        """When metric='keys', _get_metric calls compute_k_metric."""
        import mlx.nn as nn
        from unittest.mock import MagicMock

        vm = MagicMock()
        wrapper = BaseToMeVisionWrapper(vm, metric="keys")

        hidden_dim, num_heads, N = 64, 4, 6
        # Build a real block since compute_k_metric needs real layers
        block = MagicMock()
        block.norm1 = nn.Identity()
        block.attn.num_heads = num_heads
        block.attn.head_dim = hidden_dim // num_heads
        block.attn.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)

        hidden = mx.random.normal((N, hidden_dim))
        result = wrapper._get_metric(hidden, block, rotary_pos_emb=None)

        assert result.shape == (N, hidden_dim)
        # Should NOT be identical to hidden_states (it's a projection)
        assert not mx.array_equal(result, hidden)


# ---------------------------------------------------------------------------
# Profile matching for Qwen3-VL
# ---------------------------------------------------------------------------

class TestQwen3VLProfile:
    def test_qwen3_vl_4b_direct(self):
        from trio_core.profiles import get_profile
        p = get_profile("qwen3-vl-4b")
        assert p.family == "qwen3-vl"
        assert p.param_size == "4B"

    def test_qwen3_vl_4b_huggingface_id(self):
        from trio_core.profiles import get_profile
        p = get_profile("mlx-community/Qwen3-VL-4B-Instruct-4bit")
        assert p.family == "qwen3-vl"
        assert p.param_size == "4B"

    def test_qwen3_vl_generic_fallback(self):
        from trio_core.profiles import get_profile
        p = get_profile("Qwen/Qwen3-VL-8B")
        assert p.family == "qwen3-vl"
