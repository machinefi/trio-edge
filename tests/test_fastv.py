"""Tests for trio_core.fastv_backend — FastV visual token pruning."""

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# FastVMLXBackend.__init__ validation
# ---------------------------------------------------------------------------

class TestFastVMLXBackendValidation:
    """Test __init__ parameter validation without loading a real model."""

    def _make_backend(self, **kwargs):
        from unittest.mock import patch
        from trio_core.fastv_backend import FastVMLXBackend

        with patch.object(FastVMLXBackend.__bases__[0], "__init__", return_value=None):
            return FastVMLXBackend("fake-model", **kwargs)

    def test_default_params(self):
        backend = self._make_backend()
        assert backend.prune_ratio == 0.5
        assert backend.prune_after_layer == 2

    def test_custom_params(self):
        backend = self._make_backend(prune_ratio=0.3, prune_after_layer=4)
        assert backend.prune_ratio == 0.3
        assert backend.prune_after_layer == 4

    def test_prune_ratio_zero_raises(self):
        with pytest.raises(ValueError, match="prune_ratio"):
            self._make_backend(prune_ratio=0.0)

    def test_prune_ratio_one_raises(self):
        with pytest.raises(ValueError, match="prune_ratio"):
            self._make_backend(prune_ratio=1.0)

    def test_prune_ratio_negative_raises(self):
        with pytest.raises(ValueError, match="prune_ratio"):
            self._make_backend(prune_ratio=-0.1)

    def test_prune_after_layer_negative_raises(self):
        with pytest.raises(ValueError, match="prune_after_layer"):
            self._make_backend(prune_after_layer=-1)

    def test_prune_after_layer_zero_ok(self):
        backend = self._make_backend(prune_after_layer=0)
        assert backend.prune_after_layer == 0


# ---------------------------------------------------------------------------
# FastVMLXBackend.backend_name
# ---------------------------------------------------------------------------

class TestFastVBackendName:
    def test_backend_name(self):
        from unittest.mock import patch
        from trio_core.fastv_backend import FastVMLXBackend

        with patch.object(FastVMLXBackend.__bases__[0], "__init__", return_value=None):
            backend = FastVMLXBackend("fake-model")
        assert backend.backend_name == "mlx-fastv"


# ---------------------------------------------------------------------------
# FastVMLXBackend._prune_visual_tokens
# ---------------------------------------------------------------------------

class TestPruneVisualTokens:
    def test_prune_keeps_correct_positions(self):
        from trio_core.fastv_backend import FastVMLXBackend

        # Sequence: [text, vis, vis, vis, vis, text]
        # IDs:       [1,    2,   2,   2,   2,   3  ]
        B, L, D = 1, 6, 4
        embeds = mx.arange(B * L * D).reshape(B, L, D).astype(mx.float32)
        input_ids = mx.array([[1, 2, 2, 2, 2, 3]], dtype=mx.int32)
        # 1D mask (as created by input_ids[0] == visual_token_id)
        visual_mask = (input_ids[0] == 2)

        # Keep visual tokens at indices 0 and 2 (positions 1 and 3 in sequence)
        keep_indices = mx.array([0, 2], dtype=mx.int32)

        pruned_embeds, pruned_ids = FastVMLXBackend._prune_visual_tokens(
            embeds, input_ids, visual_mask, keep_indices,
        )

        # Should keep: pos 0 (text), pos 1 (vis[0]), pos 3 (vis[2]), pos 5 (text)
        assert pruned_ids.shape == (1, 4)
        assert pruned_ids[0].tolist() == [1, 2, 2, 3]
        assert pruned_embeds.shape == (1, 4, D)

    def test_prune_all_visual_tokens_kept(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, L, D = 1, 5, 4
        embeds = mx.ones((B, L, D))
        input_ids = mx.array([[1, 2, 2, 2, 3]], dtype=mx.int32)
        visual_mask = (input_ids[0] == 2)

        # Keep all 3 visual tokens
        keep_indices = mx.array([0, 1, 2], dtype=mx.int32)

        pruned_embeds, pruned_ids = FastVMLXBackend._prune_visual_tokens(
            embeds, input_ids, visual_mask, keep_indices,
        )

        assert pruned_ids.shape == (1, 5)
        assert pruned_embeds.shape == (1, 5, D)


# ---------------------------------------------------------------------------
# FastVMLXBackend._apply_mrope
# ---------------------------------------------------------------------------

class TestApplyMRoPE:
    def test_output_shape_matches_input(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, H, L, D = 1, 4, 10, 16
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        cos = mx.ones((B, L, D))
        sin = mx.zeros((B, L, D))

        q_out, k_out = FastVMLXBackend._apply_mrope(q, k, cos, sin)
        assert q_out.shape == (B, H, L, D)
        assert k_out.shape == (B, H, L, D)

    def test_identity_rotation(self):
        """cos=1, sin=0 should preserve input."""
        from trio_core.fastv_backend import FastVMLXBackend

        B, H, L, D = 1, 2, 4, 8
        q = mx.random.normal((B, H, L, D))
        k = mx.random.normal((B, H, L, D))
        cos = mx.ones((B, L, D))
        sin = mx.zeros((B, L, D))

        q_out, k_out = FastVMLXBackend._apply_mrope(q, k, cos, sin)
        assert mx.allclose(q_out, q, atol=1e-6)
        assert mx.allclose(k_out, k, atol=1e-6)


# ---------------------------------------------------------------------------
# FastVMLXBackend._original_token_count
# ---------------------------------------------------------------------------

class TestFastVOriginalTokenCount:
    def test_single_image(self):
        from trio_core.fastv_backend import FastVMLXBackend
        grid_thw = mx.array([[1, 28, 28]], dtype=mx.int32)
        assert FastVMLXBackend._original_token_count(grid_thw) == 196

    def test_video(self):
        from trio_core.fastv_backend import FastVMLXBackend
        grid_thw = mx.array([[4, 14, 14]], dtype=mx.int32)
        assert FastVMLXBackend._original_token_count(grid_thw) == 196


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestFastVConfig:
    def test_config_defaults(self):
        from trio_core.config import EngineConfig
        config = EngineConfig()
        assert config.fastv_enabled is False
        assert config.fastv_ratio == 0.5
        assert config.fastv_layer == 2

    def test_config_custom(self):
        from trio_core.config import EngineConfig
        config = EngineConfig(fastv_enabled=True, fastv_ratio=0.3, fastv_layer=4)
        assert config.fastv_enabled is True
        assert config.fastv_ratio == 0.3
        assert config.fastv_layer == 4
