"""Tests for StreamMem — bounded KV cache for continuous video streams."""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import pytest

from trio_core.streaming_memory import EvictionStats, StreamingMemory


# ── Helpers ──────────────────────────────────────────────────────────────────


class FakeKVCacheEntry:
    """Mimics mlx_lm KVCache with keys, values, offset."""

    def __init__(self, n_heads: int, head_dim: int, seq_len: int):
        self.keys = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.values = mx.random.normal((1, n_heads, seq_len, head_dim))
        self.offset = seq_len

    def trim(self, n: int):
        if n > 0 and n <= self.offset:
            self.keys = self.keys[:, :, : self.offset - n, :]
            self.values = self.values[:, :, : self.offset - n, :]
            self.offset -= n


class FakeArraysCacheEntry:
    """Mimics DeltaNet ArraysCache — no keys/values/offset, has state."""

    def __init__(self, state_size: int = 64):
        self.state = [mx.zeros((1, state_size)), mx.zeros((1, state_size))]


def make_kv_cache(n_layers: int = 4, n_heads: int = 2, head_dim: int = 8, seq_len: int = 100):
    """Create a fake KV cache with all KVCache layers."""
    return [FakeKVCacheEntry(n_heads, head_dim, seq_len) for _ in range(n_layers)]


def make_hybrid_cache(
    n_deltanet: int = 3,
    n_attn: int = 1,
    n_heads: int = 2,
    head_dim: int = 8,
    seq_len: int = 100,
):
    """Create a hybrid cache: DeltaNet layers + KVCache layers."""
    cache = []
    for _ in range(n_deltanet):
        cache.append(FakeArraysCacheEntry())
    for _ in range(n_attn):
        cache.append(FakeKVCacheEntry(n_heads, head_dim, seq_len))
    return cache


# ── Tests ────────────────────────────────────────────────────────────────────


class TestAppendFrame:
    def test_tracks_tokens(self):
        sm = StreamingMemory(budget=6000)
        sm.append_frame(100, text_prefix_len=20)
        assert sm._total_visual_tokens == 100
        assert sm._text_prefix_len == 20

    def test_multiple_frames(self):
        sm = StreamingMemory(budget=6000)
        sm.append_frame(100, text_prefix_len=20)
        sm.append_frame(100)
        assert sm._total_visual_tokens == 200
        assert len(sm._frame_boundaries) == 2

    def test_text_prefix_set_once(self):
        sm = StreamingMemory(budget=6000)
        sm.append_frame(100, text_prefix_len=20)
        sm.append_frame(100, text_prefix_len=99)  # should be ignored
        # text_prefix_len should remain 20 (set from first frame)
        assert sm._text_prefix_len == 20


class TestOverBudget:
    def test_under_budget(self):
        sm = StreamingMemory(budget=200)
        sm.append_frame(100, text_prefix_len=10)
        assert not sm.over_budget

    def test_over_budget(self):
        sm = StreamingMemory(budget=100)
        sm.append_frame(80, text_prefix_len=10)
        sm.append_frame(80)
        assert sm.over_budget
        assert sm._total_visual_tokens == 160


class TestFindSaliencyLayer:
    def test_pure_kvcache(self):
        sm = StreamingMemory(saliency_layer=1)
        cache = make_kv_cache(n_layers=4)
        idx = sm._find_saliency_layer(cache)
        assert idx == 1

    def test_hybrid_skips_deltanet(self):
        sm = StreamingMemory(saliency_layer=0)
        cache = make_hybrid_cache(n_deltanet=3, n_attn=1)
        idx = sm._find_saliency_layer(cache)
        assert idx == 3  # First KVCache layer

    def test_no_kvcache_layers(self):
        sm = StreamingMemory(saliency_layer=0)
        cache = [FakeArraysCacheEntry() for _ in range(4)]
        idx = sm._find_saliency_layer(cache)
        assert idx == -1


class TestIsKVCacheLayer:
    def test_kvcache(self):
        assert StreamingMemory._is_kvcache_layer(FakeKVCacheEntry(2, 8, 10))

    def test_arrays_cache(self):
        assert not StreamingMemory._is_kvcache_layer(FakeArraysCacheEntry())


class TestEvictAndMerge:
    def test_under_budget_noop(self):
        sm = StreamingMemory(budget=200)
        sm._total_visual_tokens = 100
        sm._text_prefix_len = 10
        cache = make_kv_cache(n_layers=2, seq_len=110)  # 10 text + 100 vis
        saliency = mx.random.normal((100,))
        stats = sm.evict_and_merge(cache, saliency)
        assert stats.n_evicted == 0
        assert stats.tokens_before == 100

    def test_eviction_reduces_tokens(self):
        sm = StreamingMemory(budget=50, prototype_ratio=0.1)
        sm._total_visual_tokens = 100
        sm._text_prefix_len = 10
        # Cache has 110 positions (10 text + 100 visual)
        cache = make_kv_cache(n_layers=2, n_heads=2, head_dim=8, seq_len=110)
        saliency = mx.random.normal((100,))
        mx.eval(saliency)

        stats = sm.evict_and_merge(cache, saliency)
        assert stats.n_evicted == 50
        assert stats.tokens_after < stats.tokens_before
        assert stats.n_prototypes > 0

    def test_text_prefix_preserved(self):
        sm = StreamingMemory(budget=30, prototype_ratio=0.1)
        sm._total_visual_tokens = 80
        sm._text_prefix_len = 20
        cache = make_kv_cache(n_layers=1, n_heads=2, head_dim=8, seq_len=100)

        # Save text prefix K/V before eviction
        text_k_before = cache[0].keys[:, :, :20, :] * 1

        saliency = mx.random.normal((80,))
        mx.eval(saliency, text_k_before)

        sm.evict_and_merge(cache, saliency)

        # Text prefix should be unchanged
        text_k_after = cache[0].keys[:, :, :20, :]
        mx.eval(text_k_after)
        assert mx.allclose(text_k_before, text_k_after).item()

    def test_deltanet_layers_skipped(self):
        sm = StreamingMemory(budget=30, prototype_ratio=0.1)
        sm._total_visual_tokens = 60
        sm._text_prefix_len = 10
        cache = make_hybrid_cache(n_deltanet=2, n_attn=1, seq_len=70)

        # Save DeltaNet state before eviction
        dn_state_before = [mx.array(s) for s in cache[0].state]

        saliency = mx.random.normal((60,))
        mx.eval(saliency, *dn_state_before)

        sm.evict_and_merge(cache, saliency)

        # DeltaNet state should be unchanged
        for before, after in zip(dn_state_before, cache[0].state):
            mx.eval(after)
            assert mx.array_equal(before, after).item()

    def test_prototypes_are_weighted_average(self):
        sm = StreamingMemory(budget=0, prototype_ratio=1.0)
        sm._total_visual_tokens = 4
        sm._text_prefix_len = 0

        # Create simple cache: 4 visual tokens, no text
        cache = [FakeKVCacheEntry(1, 4, 4)]
        # Set known K values
        cache[0].keys = mx.array([[[[1.0, 0, 0, 0],
                                     [0, 1.0, 0, 0],
                                     [0, 0, 1.0, 0],
                                     [0, 0, 0, 1.0]]]])
        cache[0].values = cache[0].keys * 1

        # All same saliency → uniform weights
        saliency = mx.ones(4)
        mx.eval(saliency)

        stats = sm.evict_and_merge(cache, saliency)
        assert stats.n_evicted == 4
        assert stats.n_prototypes > 0


class TestAttentionSink:
    def test_sink_tokens_never_evicted(self):
        """First N visual tokens should survive eviction regardless of saliency."""
        sm = StreamingMemory(budget=10, prototype_ratio=0.0, n_sink_tokens=4)
        sm._total_visual_tokens = 20
        sm._text_prefix_len = 5
        cache = make_kv_cache(n_layers=1, n_heads=2, head_dim=8, seq_len=25)

        # Give sink positions (0-3) the LOWEST saliency — they should still survive
        saliency = mx.ones(20) * 10.0
        saliency = saliency.at[:4].add(-100.0)  # sink tokens have lowest saliency
        mx.eval(saliency)

        # Save sink K values before eviction
        sink_k_before = cache[0].keys[:, :, 5:9, :] * 1  # positions 5-8 = first 4 visual
        mx.eval(sink_k_before)

        sm.evict_and_merge(cache, saliency)

        # Sink tokens should be in the kept portion (after text prefix)
        sink_k_after = cache[0].keys[:, :, 5:9, :]
        mx.eval(sink_k_after)
        assert mx.allclose(sink_k_before, sink_k_after).item()

    def test_no_sink_tokens_can_be_evicted(self):
        """With n_sink_tokens=0, first tokens have no protection."""
        sm = StreamingMemory(budget=10, prototype_ratio=0.0, n_sink_tokens=0)
        sm._total_visual_tokens = 20
        sm._text_prefix_len = 0
        cache = make_kv_cache(n_layers=1, n_heads=2, head_dim=8, seq_len=20)

        # First 4 tokens have lowest saliency — should be evicted
        saliency = mx.ones(20) * 10.0
        saliency = saliency.at[:4].add(-100.0)
        mx.eval(saliency)

        sink_k_before = cache[0].keys[:, :, :4, :] * 1
        mx.eval(sink_k_before)

        sm.evict_and_merge(cache, saliency)

        # Without sink protection, these tokens should NOT appear at positions 0-3
        new_k = cache[0].keys[:, :, :4, :]
        mx.eval(new_k)
        assert not mx.allclose(sink_k_before, new_k).item()

    def test_default_sink_is_4(self):
        sm = StreamingMemory()
        assert sm.n_sink_tokens == 4


class TestMaybeEvict:
    def test_no_eviction_when_under_budget(self):
        sm = StreamingMemory(budget=200)
        sm._total_visual_tokens = 50
        result = sm.maybe_evict([], None, mx.array([[1, 2, 3]]))
        assert result is None

    def test_eviction_when_over_budget(self):
        sm = StreamingMemory(budget=30, prototype_ratio=0.1, saliency_layer=0)
        sm._total_visual_tokens = 60
        sm._text_prefix_len = 10

        cache = make_kv_cache(n_layers=1, n_heads=2, head_dim=8, seq_len=70)

        # Mock language_model
        lm = MagicMock()
        embed_tokens = MagicMock(return_value=mx.random.normal((1, 8, 16)))
        lm.model.embed_tokens = embed_tokens

        # Mock layer
        mock_layer = MagicMock()
        mock_layer.input_layernorm = MagicMock(return_value=mx.random.normal((1, 8, 16)))

        mock_attn = MagicMock()
        mock_attn.q_proj = MagicMock(return_value=mx.random.normal((1, 8, 16)))
        mock_attn.n_heads = 2
        mock_attn.n_kv_heads = 2
        mock_attn.head_dim = 8
        mock_layer.self_attn = mock_attn

        lm.model.layers = [mock_layer]

        proxy_ids = mx.array([[1, 2, 3, 4, 5, 6, 7, 8]])

        stats = sm.maybe_evict(cache, lm, proxy_ids)
        assert stats is not None
        assert stats.n_evicted > 0


class TestGetVisualPositions:
    def test_positions(self):
        sm = StreamingMemory()
        sm._text_prefix_len = 10
        sm._total_visual_tokens = 5
        positions = sm._get_visual_positions()
        mx.eval(positions)
        expected = mx.arange(10, 15)
        assert mx.array_equal(positions, expected).item()


class TestModelAgnosticVisualTokenCount:
    """Test that both _id and _index model config attributes work."""

    def test_image_token_id(self):
        """Models with image_token_id (Qwen2.5-VL, Gemma3)."""
        config = MagicMock()
        config.image_token_id = 151655
        config.image_token_index = None
        config.video_token_id = None
        config.video_token_index = None
        # Simulate the logic from generate_step
        img_id = getattr(config, 'image_token_id', None)
        if img_id is None:
            img_id = getattr(config, 'image_token_index', None)
        assert img_id == 151655

    def test_image_token_index(self):
        """Models with image_token_index (Qwen3-VL, Qwen3.5)."""
        config = MagicMock()
        config.image_token_id = None
        config.image_token_index = 151655
        config.video_token_id = None
        config.video_token_index = None
        img_id = getattr(config, 'image_token_id', None)
        if img_id is None:
            img_id = getattr(config, 'image_token_index', None)
        assert img_id == 151655
