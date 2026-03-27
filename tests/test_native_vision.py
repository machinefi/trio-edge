"""Tests for trio_core.native_vision — native ViT with built-in ToMe."""

import pytest

mx = pytest.importorskip("mlx.core")


# ---------------------------------------------------------------------------
# Factory auto-detection
# ---------------------------------------------------------------------------


class TestCreateTomeVision:
    def test_qwen25_detected(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen25Vision, create_tome_vision

        vm = MagicMock()
        vm.model_type = "qwen2_5_vl"
        result = create_tome_vision(vm, tome_r=4)
        assert isinstance(result, NativeToMeQwen25Vision)
        assert result.tome_r == 4

    def test_qwen3_detected(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen3Vision, create_tome_vision

        vm = MagicMock()
        vm.model_type = "qwen3_vl"
        result = create_tome_vision(vm, tome_r=8, metric="hidden")
        assert isinstance(result, NativeToMeQwen3Vision)
        assert result.tome_metric == "hidden"

    def test_qwen35_detected(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen3Vision, create_tome_vision

        vm = MagicMock()
        vm.model_type = "qwen3_5"
        result = create_tome_vision(vm, tome_r=4)
        assert isinstance(result, NativeToMeQwen3Vision)

    def test_unknown_defaults_to_qwen25(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen25Vision, create_tome_vision

        vm = MagicMock()
        vm.model_type = "unknown"
        result = create_tome_vision(vm, tome_r=4)
        assert isinstance(result, NativeToMeQwen25Vision)


# ---------------------------------------------------------------------------
# ToMe mixin logic
# ---------------------------------------------------------------------------


class TestToMeMixin:
    def _make(self, model_type="qwen2_5_vl", **kwargs):
        from unittest.mock import MagicMock

        from trio_core.native_vision import create_tome_vision

        vm = MagicMock()
        vm.model_type = model_type
        vm.fullatt_block_indexes = []
        return create_tome_vision(vm, **kwargs)

    def test_should_merge_skips_first(self):
        v = self._make(tome_r=4, skip_first=2, skip_last=2)
        assert not v._should_merge(0, 10)
        assert not v._should_merge(1, 10)
        assert v._should_merge(2, 10)

    def test_should_merge_skips_last(self):
        v = self._make(tome_r=4, skip_first=2, skip_last=2)
        assert v._should_merge(7, 10)
        assert not v._should_merge(8, 10)
        assert not v._should_merge(9, 10)

    def test_should_merge_r_zero(self):
        v = self._make(tome_r=0)
        assert not v._should_merge(5, 10)

    def test_get_layer_r_constant(self):
        v = self._make(tome_r=4, skip_first=2, skip_last=2, adaptive=False)
        assert v._get_layer_r(3, 10) == 4
        assert v._get_layer_r(0, 10) == 0  # skipped

    def test_get_layer_r_adaptive(self):
        v = self._make(tome_r=8, skip_first=2, skip_last=2, adaptive=True)
        # 10 blocks, skip 2+2 = 6 mergeable layers (2,3,4,5,6,7)
        r2 = v._get_layer_r(2, 10)  # position 1/6 → r=8*1/6≈1
        r7 = v._get_layer_r(7, 10)  # position 6/6 → r=8*6/6=8
        assert r2 < r7
        assert r7 == 8


# ---------------------------------------------------------------------------
# Qwen2.5-VL fullatt skip
# ---------------------------------------------------------------------------


class TestQwen25FullattSkip:
    def test_fullatt_layers_skipped(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen25Vision

        vm = MagicMock()
        vm.fullatt_block_indexes = [7, 15, 23, 31]
        v = NativeToMeQwen25Vision(vm, tome_r=4, skip_first=2, skip_last=2)

        # Layer 7 is in fullatt → skip merge
        assert not v._should_merge(7, 32)
        # Layer 5 is not in fullatt and within range → merge
        assert v._should_merge(5, 32)


# ---------------------------------------------------------------------------
# Attribute delegation
# ---------------------------------------------------------------------------


class TestDelegation:
    def test_delegates_to_original(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen25Vision

        vm = MagicMock()
        vm.fullatt_block_indexes = []
        vm.patch_size = 14
        vm.spatial_merge_size = 2

        v = NativeToMeQwen25Vision(vm, tome_r=4)
        assert v.patch_size == 14
        assert v.spatial_merge_size == 2

    def test_tome_attrs_not_delegated(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen25Vision

        vm = MagicMock()
        vm.fullatt_block_indexes = []
        v = NativeToMeQwen25Vision(vm, tome_r=4)

        # tome_ attributes should come from the native vision, not be delegated
        assert v.tome_r == 4

    def test_qwen3_delegates(self):
        from unittest.mock import MagicMock

        from trio_core.native_vision import NativeToMeQwen3Vision

        vm = MagicMock()
        vm.spatial_merge_size = 2
        v = NativeToMeQwen3Vision(vm, tome_r=4)
        assert v.spatial_merge_size == 2


# ---------------------------------------------------------------------------
# Content diversity (compute_content_diversity)
# ---------------------------------------------------------------------------


class TestContentDiversity:
    def test_identical_tokens_zero_diversity(self):
        from trio_core.tome import compute_content_diversity

        # All tokens identical → mean cosine similarity = 1.0 → diversity = 0.0
        tokens = mx.ones((100, 64))
        d = compute_content_diversity(tokens)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_random_tokens_high_diversity(self):
        from trio_core.tome import compute_content_diversity

        # Random tokens → low mean similarity → high diversity
        mx.random.seed(42)
        tokens = mx.random.normal((100, 64))
        d = compute_content_diversity(tokens)
        assert d > 0.5  # should be well above 0.5

    def test_single_token_zero(self):
        from trio_core.tome import compute_content_diversity

        d = compute_content_diversity(mx.ones((1, 64)))
        assert d == 0.0

    def test_two_identical_tokens(self):
        from trio_core.tome import compute_content_diversity

        tokens = mx.stack([mx.ones(64), mx.ones(64)])
        d = compute_content_diversity(tokens)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_two_orthogonal_tokens(self):
        from trio_core.tome import compute_content_diversity

        a = mx.zeros(64)
        a = a.at[0].add(1.0)
        b = mx.zeros(64)
        b = b.at[1].add(1.0)
        tokens = mx.stack([a, b])
        d = compute_content_diversity(tokens)
        assert d == pytest.approx(1.0, abs=0.01)

    def test_subsampling(self):
        from trio_core.tome import compute_content_diversity

        # Large input triggers subsampling — should still work
        tokens = mx.ones((1000, 64))
        d = compute_content_diversity(tokens, sample_size=64)
        assert d == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Content-aware r scaling
# ---------------------------------------------------------------------------


class TestContentAwareR:
    def _make(self, **kwargs):
        from unittest.mock import MagicMock

        from trio_core.native_vision import create_tome_vision

        vm = MagicMock()
        vm.model_type = "qwen2_5_vl"
        vm.fullatt_block_indexes = []
        return create_tome_vision(vm, **kwargs)

    def test_content_aware_default_off(self):
        v = self._make(tome_r=4)
        assert v.tome_content_aware is False
        assert v._content_r_factor == 1.0

    def test_content_aware_enabled(self):
        v = self._make(tome_r=4, content_aware=True)
        assert v.tome_content_aware is True

    def test_content_factor_simple_image(self):
        """Simple (low diversity) image should keep r_factor near 1.0."""
        v = self._make(tome_r=8, content_aware=True)
        # All-ones tokens = zero diversity → factor should be 1.0
        tokens = mx.ones((100, 64))
        v._compute_content_factor(tokens)
        assert v._content_r_factor == pytest.approx(1.0, abs=0.05)
        assert v._get_layer_r(3, 10) == 8  # full r

    def test_content_factor_complex_image(self):
        """Complex (high diversity) image should reduce r significantly."""
        v = self._make(tome_r=8, content_aware=True)
        # Random tokens = high diversity → factor should be low
        mx.random.seed(42)
        tokens = mx.random.normal((100, 64))
        v._compute_content_factor(tokens)
        assert v._content_r_factor < 0.5
        assert v._get_layer_r(3, 10) < 8  # reduced r

    def test_content_factor_minimum_clamp(self):
        """Factor should never go below 0.2."""
        v = self._make(tome_r=8, content_aware=True)
        # Force extreme diversity
        v._content_r_factor = 0.0  # shouldn't happen via _compute, but test clamp
        v._compute_content_factor(mx.random.normal((200, 64)))
        assert v._content_r_factor >= 0.2

    def test_content_aware_with_adaptive(self):
        """Content-aware + adaptive should stack — both scale r."""
        v = self._make(tome_r=8, skip_first=2, skip_last=2, adaptive=True, content_aware=True)
        # Simple image → factor ~1.0
        v._compute_content_factor(mx.ones((100, 64)))
        r_simple = v._get_layer_r(4, 10)

        # Complex image → factor < 1.0
        mx.random.seed(99)
        v._compute_content_factor(mx.random.normal((100, 64)))
        r_complex = v._get_layer_r(4, 10)

        assert r_simple >= r_complex

    def test_config_content_aware(self):
        from trio_core.config import EngineConfig

        config = EngineConfig(tome_content_aware=True)
        assert config.tome_content_aware is True

    def test_config_default_off(self):
        from trio_core.config import EngineConfig

        config = EngineConfig()
        assert config.tome_content_aware is False
