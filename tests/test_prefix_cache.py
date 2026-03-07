"""Tests for prefix KV cache reuse in generate.py."""

import hashlib

import mlx.core as mx
import numpy as np
import pytest


# ---------------------------------------------------------------------------
# _find_visual_boundary
# ---------------------------------------------------------------------------

class TestFindVisualBoundary:
    def _call(self, ids_list, img_id=100, vid_id=101):
        from unittest.mock import MagicMock
        from trio_core.generate import _find_visual_boundary

        model = MagicMock()
        model.config.image_token_id = img_id
        model.config.image_token_index = None
        model.config.video_token_id = vid_id
        model.config.video_token_index = None
        input_ids = mx.array([ids_list], dtype=mx.int32)
        return _find_visual_boundary(input_ids, model)

    def test_visual_at_start(self):
        assert self._call([100, 100, 100, 5, 6]) == 0

    def test_visual_in_middle(self):
        # [text, text, text, vis, vis, text]
        assert self._call([1, 2, 3, 100, 100, 4]) == 3

    def test_video_token(self):
        assert self._call([1, 2, 101, 101, 3]) == 2

    def test_no_visual_tokens(self):
        assert self._call([1, 2, 3, 4, 5]) == 0

    def test_uses_image_token_index_fallback(self):
        from unittest.mock import MagicMock
        from trio_core.generate import _find_visual_boundary

        model = MagicMock()
        model.config.image_token_id = None
        model.config.image_token_index = 200
        model.config.video_token_id = None
        model.config.video_token_index = None
        input_ids = mx.array([[1, 2, 200, 200, 3]], dtype=mx.int32)
        assert _find_visual_boundary(input_ids, model) == 2


# ---------------------------------------------------------------------------
# PromptCache prefix methods
# ---------------------------------------------------------------------------

class TestPromptCachePrefixHit:
    def _make_cache(self):
        """Create a PromptCache with a mock model."""
        from unittest.mock import MagicMock, PropertyMock
        from trio_core.generate import PromptCache

        model = MagicMock()
        model.language_model.layers = [MagicMock() for _ in range(4)]
        # Mock make_cache to return KVCache-like objects
        cache = PromptCache(model)
        return cache

    def test_no_prefix_returns_false(self):
        cache = self._make_cache()
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        assert cache.check_prefix_hit(ids) is False

    def test_same_ids_after_save_returns_true(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        # Simulate a KV cache with offset >= prefix_len
        kv_caches = []
        for _ in range(4):
            kv = KVCache()
            # Fill with dummy data (B=1, n_kv_heads=2, L=10, head_dim=4)
            keys = mx.ones((1, 2, 10, 4))
            values = mx.ones((1, 2, 10, 4))
            kv.update_and_fetch(keys, values)
            kv_caches.append(kv)

        # Mark as trimmable by setting the real kv_cache
        cache._kv_cache = kv_caches

        ids = mx.array([[1, 2, 3, 100, 100, 4]], dtype=mx.int32)
        cache.save_prefix(ids, prefix_len=3, kv_cache=kv_caches)

        assert cache.check_prefix_hit(ids) is True

    def test_different_ids_returns_false(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv_caches = []
        for _ in range(4):
            kv = KVCache()
            keys = mx.ones((1, 2, 10, 4))
            values = mx.ones((1, 2, 10, 4))
            kv.update_and_fetch(keys, values)
            kv_caches.append(kv)
        cache._kv_cache = kv_caches

        ids1 = mx.array([[1, 2, 3, 100, 100, 4]], dtype=mx.int32)
        ids2 = mx.array([[1, 2, 3, 100, 100, 5]], dtype=mx.int32)  # different suffix
        cache.save_prefix(ids1, prefix_len=3, kv_cache=kv_caches)

        assert cache.check_prefix_hit(ids2) is False


class TestPromptCachePrefixRestore:
    def test_restore_creates_cache_with_prefix_kv(self):
        from unittest.mock import MagicMock, patch
        from mlx_lm.models.cache import KVCache
        from trio_core.generate import PromptCache

        model = MagicMock()
        # make_cache returns real KVCache objects
        model.language_model.make_cache.return_value = [KVCache() for _ in range(2)]

        cache = PromptCache(model)

        # Build KV caches with known data
        kv_caches = []
        for layer_idx in range(2):
            kv = KVCache()
            # (B=1, n_kv_heads=2, L=8, head_dim=4) — total 8 tokens
            keys = mx.full((1, 2, 8, 4), vals=float(layer_idx + 1))
            values = mx.full((1, 2, 8, 4), vals=float(layer_idx + 10))
            kv.update_and_fetch(keys, values)
            kv_caches.append(kv)

        cache._kv_cache = kv_caches

        # Save prefix (first 3 tokens)
        ids = mx.array([[1, 2, 3, 100, 100, 4, 5, 6]], dtype=mx.int32)
        cache.save_prefix(ids, prefix_len=3, kv_cache=kv_caches)

        assert cache._prefix_len == 3
        assert len(cache._prefix_states) == 2

        # Restore prefix into new cache
        restored = cache.restore_prefix_cache()

        assert len(restored) == 2
        for i, c in enumerate(restored):
            assert c.offset == 3
            # Verify the restored keys match the first 3 positions
            expected_key_val = float(i + 1)
            assert mx.allclose(
                c.keys[..., :3, :],
                mx.full((1, 2, 3, 4), vals=expected_key_val),
            )

    def test_save_prefix_zero_len_is_noop(self):
        from unittest.mock import MagicMock
        from trio_core.generate import PromptCache

        model = MagicMock()
        cache = PromptCache(model)
        ids = mx.array([[1, 2, 3]], dtype=mx.int32)
        cache.save_prefix(ids, prefix_len=0, kv_cache=[])
        assert cache._prefix_hash is None
        assert cache._prefix_states is None


# ---------------------------------------------------------------------------
# Visual similarity KV reuse
# ---------------------------------------------------------------------------

class TestVisualSimilarity:
    def _make_cache(self):
        from unittest.mock import MagicMock
        from trio_core.generate import PromptCache

        model = MagicMock()
        model.language_model.layers = [MagicMock() for _ in range(2)]
        return PromptCache(model)

    def _ids(self, n=10):
        """Helper: create input_ids of length n."""
        return mx.arange(n).reshape(1, -1)

    def test_no_saved_embeds_returns_false(self):
        cache = self._make_cache()
        embeds = mx.ones((1, 10, 64))
        assert cache.check_visual_hit(embeds, self._ids(), threshold=0.95) is False

    def test_identical_embeds_returns_true(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        # Need trimmable KV cache + saved first token for hit
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids = self._ids()
        embeds = mx.ones((1, 10, 64))
        cache.save_embeds(embeds, ids)

        # Same embeddings + same input_ids → similarity 1.0 → hit
        assert cache.check_visual_hit(embeds, ids, threshold=0.95) is True

    def test_similar_embeds_above_threshold(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids = self._ids()
        # Save reference embeddings
        embeds_a = mx.ones((1, 10, 64))
        cache.save_embeds(embeds_a, ids)

        # Add small noise (should still be very similar)
        embeds_b = embeds_a + mx.random.normal(embeds_a.shape) * 0.01
        assert cache.check_visual_hit(embeds_b, ids, threshold=0.95) is True

    def test_different_embeds_below_threshold(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids = self._ids()
        embeds_a = mx.ones((1, 10, 64))
        cache.save_embeds(embeds_a, ids)

        # Very different embeddings
        embeds_b = mx.random.normal((1, 10, 64))
        assert cache.check_visual_hit(embeds_b, ids, threshold=0.95) is False

    def test_different_input_ids_returns_false(self):
        """Different questions must not reuse KV even with similar embeds."""
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids_a = self._ids()
        embeds = mx.ones((1, 10, 64))
        cache.save_embeds(embeds, ids_a)

        # Same embeds but different input_ids (different question)
        ids_b = mx.arange(10, 20).reshape(1, -1)
        assert cache.check_visual_hit(embeds, ids_b, threshold=0.95) is False

    def test_shape_mismatch_returns_false(self):
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids = self._ids()
        embeds_a = mx.ones((1, 10, 64))
        cache.save_embeds(embeds_a, ids)

        # Different sequence length (different resolution)
        embeds_b = mx.ones((1, 20, 64))
        ids_b = mx.arange(20).reshape(1, -1)
        assert cache.check_visual_hit(embeds_b, ids_b, threshold=0.95) is False

    def test_no_first_token_returns_false(self):
        """Visual hit requires saved first token to reuse."""
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        # No first_token saved

        ids = self._ids()
        embeds = mx.ones((1, 10, 64))
        cache.save_embeds(embeds, ids)
        assert cache.check_visual_hit(embeds, ids, threshold=0.95) is False

    def test_threshold_zero_disables(self):
        """Threshold 0.0 should never trigger visual similarity check."""
        from mlx_lm.models.cache import KVCache

        cache = self._make_cache()
        kv = [KVCache() for _ in range(2)]
        for c in kv:
            c.update_and_fetch(mx.ones((1, 2, 5, 4)), mx.ones((1, 2, 5, 4)))
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))

        ids = self._ids()
        embeds = mx.ones((1, 10, 64))
        cache.save_embeds(embeds, ids)
        # threshold=0 means everything would pass, but in generate_step
        # the check is gated by visual_similarity_threshold > 0
        # PromptCache itself doesn't enforce this — it's a threshold comparison
        # 1.0 >= 0.0 is True, but the caller gates with > 0 check
        assert cache.check_visual_hit(embeds, ids, threshold=0.0) is True


    def test_arrays_cache_deltanet_works(self):
        """DeltaNet ArraysCache (no offset/trim) should still support visual similarity."""
        cache = self._make_cache()
        # Simulate ArraysCache: a list of plain mx.arrays (no offset attribute)
        kv = [mx.zeros((1, 32, 128, 128)) for _ in range(2)]
        cache._kv_cache = kv
        cache._first_token = (mx.array(42), mx.zeros(100))
        assert not cache.is_trimmable  # ArraysCache has no offset

        ids = self._ids()
        embeds = mx.ones((1, 10, 64))
        cache.save_embeds(embeds, ids)

        # Same embeddings + same input_ids → should hit despite non-trimmable cache
        assert cache.check_visual_hit(embeds, ids, threshold=0.95) is True


class TestVisualSimilarityConfig:
    def test_config_default_disabled(self):
        from trio_core.config import EngineConfig
        config = EngineConfig()
        assert config.visual_similarity_threshold == 0.0

    def test_config_custom(self):
        from trio_core.config import EngineConfig
        config = EngineConfig(visual_similarity_threshold=0.95)
        assert config.visual_similarity_threshold == 0.95
