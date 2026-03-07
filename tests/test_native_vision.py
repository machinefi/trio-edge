"""Tests for trio_core.native_vision — native ViT with built-in ToMe."""

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Factory auto-detection
# ---------------------------------------------------------------------------

class TestCreateTomeVision:

    def test_qwen25_detected(self):
        from unittest.mock import MagicMock
        from trio_core.native_vision import create_tome_vision, NativeToMeQwen25Vision

        vm = MagicMock()
        vm.model_type = "qwen2_5_vl"
        result = create_tome_vision(vm, tome_r=4)
        assert isinstance(result, NativeToMeQwen25Vision)
        assert result.tome_r == 4

    def test_qwen3_detected(self):
        from unittest.mock import MagicMock
        from trio_core.native_vision import create_tome_vision, NativeToMeQwen3Vision

        vm = MagicMock()
        vm.model_type = "qwen3_vl"
        result = create_tome_vision(vm, tome_r=8, metric="hidden")
        assert isinstance(result, NativeToMeQwen3Vision)
        assert result.tome_metric == "hidden"

    def test_qwen35_detected(self):
        from unittest.mock import MagicMock
        from trio_core.native_vision import create_tome_vision, NativeToMeQwen3Vision

        vm = MagicMock()
        vm.model_type = "qwen3_5"
        result = create_tome_vision(vm, tome_r=4)
        assert isinstance(result, NativeToMeQwen3Vision)

    def test_unknown_defaults_to_qwen25(self):
        from unittest.mock import MagicMock
        from trio_core.native_vision import create_tome_vision, NativeToMeQwen25Vision

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
