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

        pruned_embeds, pruned_ids, all_keep = FastVMLXBackend._prune_visual_tokens(
            embeds, input_ids, visual_mask, keep_indices,
        )

        # Should keep: pos 0 (text), pos 1 (vis[0]), pos 3 (vis[2]), pos 5 (text)
        assert pruned_ids.shape == (1, 4)
        assert pruned_ids[0].tolist() == [1, 2, 2, 3]
        assert pruned_embeds.shape == (1, 4, D)
        assert all_keep.shape == (4,)

    def test_prune_all_visual_tokens_kept(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, L, D = 1, 5, 4
        embeds = mx.ones((B, L, D))
        input_ids = mx.array([[1, 2, 2, 2, 3]], dtype=mx.int32)
        visual_mask = (input_ids[0] == 2)

        # Keep all 3 visual tokens
        keep_indices = mx.array([0, 1, 2], dtype=mx.int32)

        pruned_embeds, pruned_ids, all_keep = FastVMLXBackend._prune_visual_tokens(
            embeds, input_ids, visual_mask, keep_indices,
        )

        assert pruned_ids.shape == (1, 5)
        assert pruned_embeds.shape == (1, 5, D)
        assert all_keep.shape == (5,)


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
# FastVMLXBackend._prune_kv_cache
# ---------------------------------------------------------------------------

class TestPruneKVCache:
    """Test in-place KV cache pruning for mid-stream FastV."""

    def _make_cache(self, B, n_kv, seq_len, head_dim):
        """Create a KVCache populated with sequential data."""
        from mlx_lm.models.cache import KVCache
        c = KVCache()
        k = mx.arange(B * n_kv * seq_len * head_dim).reshape(
            B, n_kv, seq_len, head_dim,
        ).astype(mx.float32)
        v = mx.arange(B * n_kv * seq_len * head_dim).reshape(
            B, n_kv, seq_len, head_dim,
        ).astype(mx.float32) + 1000
        c.update_and_fetch(k, v)
        return c

    def test_prune_reduces_offset(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_kv, L, D = 1, 2, 8, 4
        cache = [self._make_cache(B, n_kv, L, D) for _ in range(3)]
        assert cache[0].offset == L

        # Keep positions 0, 2, 5
        keep = mx.array([0, 2, 5], dtype=mx.int32)
        FastVMLXBackend._prune_kv_cache(cache, keep, n_layers=2)

        # Pruned layers 0, 1
        assert cache[0].offset == 3
        assert cache[1].offset == 3
        # Layer 2 untouched
        assert cache[2].offset == L

    def test_prune_preserves_correct_positions(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_kv, L, D = 1, 1, 6, 2
        cache = [self._make_cache(B, n_kv, L, D)]

        # Original keys at positions 0..5
        original_keys = cache[0].keys[:, :, :L, :].tolist()

        keep = mx.array([1, 3, 4], dtype=mx.int32)
        FastVMLXBackend._prune_kv_cache(cache, keep, n_layers=1)

        # After pruning: should contain positions 1, 3, 4 from original
        pruned_keys = cache[0].keys[:, :, :cache[0].offset, :].tolist()
        assert pruned_keys == [[
            [original_keys[0][0][1],
             original_keys[0][0][3],
             original_keys[0][0][4]],
        ]]

    def test_prune_empty_cache_is_noop(self):
        from mlx_lm.models.cache import KVCache
        from trio_core.fastv_backend import FastVMLXBackend

        cache = [KVCache()]  # keys/values are None
        keep = mx.array([0, 1], dtype=mx.int32)
        # Should not raise
        FastVMLXBackend._prune_kv_cache(cache, keep, n_layers=1)
        assert cache[0].offset == 0

    def test_prune_keeps_all_positions(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_kv, L, D = 1, 2, 4, 4
        cache = [self._make_cache(B, n_kv, L, D)]

        keep = mx.arange(L, dtype=mx.int32)
        FastVMLXBackend._prune_kv_cache(cache, keep, n_layers=1)

        assert cache[0].offset == L


# ---------------------------------------------------------------------------
# FastVMLXBackend._score_visual
# ---------------------------------------------------------------------------

class TestScoreVisual:
    """Test the visual importance scoring helper."""

    def test_basic_importance_shape(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_heads, n_kv, L, D = 1, 4, 4, 10, 8
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_kv, L, D))
        # 6 visual tokens, 4 text tokens
        visual_mask = mx.array(
            [False, False, True, True, True, True, True, True, False, False]
        )

        importance = FastVMLXBackend._score_visual(
            q, k, visual_mask, n_heads, n_kv, D,
        )
        assert importance.shape == (6,)

    def test_no_visual_tokens_returns_ones(self):
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_heads, n_kv, L, D = 1, 2, 2, 4, 4
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_kv, L, D))
        visual_mask = mx.zeros((L,), dtype=mx.bool_)

        importance = FastVMLXBackend._score_visual(
            q, k, visual_mask, n_heads, n_kv, D,
        )
        assert importance.shape == (1,)
        assert importance[0].item() == 1.0

    def test_gqa_expansion(self):
        """With n_kv < n_heads, K should be expanded for scoring."""
        from trio_core.fastv_backend import FastVMLXBackend

        B, n_heads, n_kv, L, D = 1, 8, 2, 6, 4
        q = mx.random.normal((B, n_heads, L, D))
        k = mx.random.normal((B, n_kv, L, D))
        visual_mask = mx.array([False, True, True, True, False, False])

        importance = FastVMLXBackend._score_visual(
            q, k, visual_mask, n_heads, n_kv, D,
        )
        assert importance.shape == (3,)


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
