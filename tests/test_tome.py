"""Tests for trio_core.tome — core ToMe algorithm (bipartite matching + merging)."""

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from trio_core.tome import bipartite_soft_matching, merge_tokens, compute_k_metric


# ---------------------------------------------------------------------------
# bipartite_soft_matching
# ---------------------------------------------------------------------------

class TestBipartiteSoftMatching:
    def test_basic_n16_r4(self):
        """N=16, r=4 → dst_idx has 12 elements, src_dst_map has 4 pairs."""
        mx.random.seed(42)
        metric = mx.random.normal((16, 32))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=4)

        assert dst_idx.shape == (12,), f"Expected (12,), got {dst_idx.shape}"
        assert src_dst_map.shape == (4, 2), f"Expected (4,2), got {src_dst_map.shape}"

        # dst_idx values should be in [0, 15]
        assert mx.min(dst_idx).item() >= 0
        assert mx.max(dst_idx).item() <= 15

        # All dst_idx values should be unique
        dst_list = dst_idx.tolist()
        assert len(set(dst_list)) == len(dst_list), "dst_idx should have unique values"

    def test_r_zero_no_merging(self):
        """r=0 → no merging, dst_idx = arange(N)."""
        N = 10
        metric = mx.random.normal((N, 16))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=0)

        assert mx.array_equal(dst_idx, mx.arange(N))
        assert src_dst_map.shape[0] == 0

    def test_r_greater_than_half_n_clamped(self):
        """r > N//2 → clamped to N//2."""
        N = 8
        metric = mx.random.normal((N, 16))
        # r=10 > N//2=4, should clamp to 4
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=10)

        assert dst_idx.shape[0] == N - N // 2  # 8 - 4 = 4
        assert src_dst_map.shape[0] == N // 2  # 4

    def test_n1_no_merging(self):
        """N=1 → no merging possible."""
        metric = mx.random.normal((1, 8))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=1)

        assert dst_idx.shape == (1,)
        assert mx.array_equal(dst_idx, mx.arange(1))
        assert src_dst_map.shape[0] == 0

    def test_n2_r1_one_pair(self):
        """N=2, r=1 → one pair merged, one token kept."""
        metric = mx.random.normal((2, 8))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=1)

        assert dst_idx.shape == (1,)
        assert src_dst_map.shape == (1, 2)

        # The src and dst should be different indices
        src, dst = src_dst_map[0, 0].item(), src_dst_map[0, 1].item()
        assert src != dst

    def test_negative_r_no_merging(self):
        """Negative r should behave like r=0."""
        N = 6
        metric = mx.random.normal((N, 8))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=-5)

        assert mx.array_equal(dst_idx, mx.arange(N))
        assert src_dst_map.shape[0] == 0

    def test_src_indices_are_even(self):
        """Source (merged-away) tokens come from the even-index set A."""
        mx.random.seed(7)
        metric = mx.random.normal((12, 16))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=3)

        src_indices = src_dst_map[:, 0].tolist()
        for s in src_indices:
            assert s % 2 == 0, f"Source index {s} should be even (set A)"

    def test_dst_indices_are_odd(self):
        """Destination (merge-into) tokens come from the odd-index set B."""
        mx.random.seed(7)
        metric = mx.random.normal((12, 16))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=3)

        dst_indices = src_dst_map[:, 1].tolist()
        for d in dst_indices:
            assert d % 2 == 1, f"Dest index {d} should be odd (set B)"


# ---------------------------------------------------------------------------
# merge_tokens
# ---------------------------------------------------------------------------

class TestMergeTokens:
    def test_shape_preservation(self):
        """Basic merge: (N,D) → (N-r, D)."""
        N, D, r = 16, 32, 4
        mx.random.seed(42)
        x = mx.random.normal((N, D))
        metric = mx.random.normal((N, D))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=r)

        merged, new_size = merge_tokens(x, dst_idx, src_dst_map)

        assert merged.shape == (N - r, D)
        assert new_size.shape == (N - r, 1)

    def test_uniform_size_defaults_to_ones(self):
        """With size=None, defaults to ones and merged tokens get size=2."""
        N, D = 4, 8
        x = mx.ones((N, D))
        metric = mx.ones((N, D))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=1)

        merged, new_size = merge_tokens(x, dst_idx, src_dst_map, size=None)

        assert merged.shape == (N - 1, D)
        # One token absorbed another, so one size entry should be 2
        sizes = new_size.tolist()
        flat_sizes = [s[0] for s in sizes]
        assert 2.0 in flat_sizes, f"Expected a size=2 entry, got {flat_sizes}"

    def test_custom_size_weighted_average(self):
        """With custom sizes, merge computes weighted average."""
        # Create a controlled 2-token case
        x = mx.array([[1.0, 0.0], [0.0, 1.0]])
        # Force them to be very similar so they merge
        metric = mx.array([[1.0, 0.0], [1.0, 0.0]])
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=1)

        # src=0 (even), dst=1 (odd), sizes 3:1 → weighted avg = (3*[1,0] + 1*[0,1])/4
        size = mx.array([[3.0], [1.0]])
        merged, new_size = merge_tokens(x, dst_idx, src_dst_map, size=size)

        assert merged.shape == (1, 2)
        assert new_size.item() == pytest.approx(4.0)

        # The single remaining token is at dst_idx (which is index 1 = odd token)
        # It should be weighted average: (size_dst * x_dst + size_src * x_src) / total
        # The dst in dst_idx is whichever survived — verify values sum correctly
        vals = merged[0].tolist()
        assert vals[0] + vals[1] == pytest.approx(1.0, abs=1e-5)

    def test_empty_src_dst_map_no_change(self):
        """Empty src_dst_map → returns original x and size."""
        N, D = 8, 16
        x = mx.random.normal((N, D))
        dst_idx = mx.arange(N)
        src_dst_map = mx.zeros((0, 2), dtype=mx.int32)

        merged, new_size = merge_tokens(x, dst_idx, src_dst_map)

        assert mx.array_equal(merged, x)
        assert new_size.shape == (N, 1)

    def test_merge_preserves_non_merged_tokens(self):
        """Tokens not involved in merging should remain unchanged."""
        N, D = 6, 4
        mx.random.seed(99)
        x = mx.random.normal((N, D))
        metric = mx.random.normal((N, D))
        dst_idx, src_dst_map = bipartite_soft_matching(metric, r=1)

        merged, _ = merge_tokens(x, dst_idx, src_dst_map)

        # dst_idx has the kept indices; those not in src_dst_map should be unchanged
        src_set = set(src_dst_map[:, 0].tolist())
        dst_set = set(src_dst_map[:, 1].tolist())
        for pos, orig_idx in enumerate(dst_idx.tolist()):
            if orig_idx not in dst_set:
                assert mx.allclose(merged[pos], x[orig_idx], atol=1e-6), (
                    f"Token at dst_idx={orig_idx} should be unchanged"
                )


# ---------------------------------------------------------------------------
# compute_k_metric
# ---------------------------------------------------------------------------

class _MockAttn:
    """Minimal mock of a ViT block's attention sub-module."""

    def __init__(self, hidden_dim: int, num_heads: int):
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        # QKV projects to 3 * hidden_dim
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim, bias=False)


class _MockBlock:
    """Minimal mock of a ViT block with norm1 and attn."""

    def __init__(self, hidden_dim: int, num_heads: int, use_identity_norm: bool = True):
        if use_identity_norm:
            self.norm1 = nn.Identity()
        else:
            self.norm1 = nn.RMSNorm(hidden_dim)
        self.attn = _MockAttn(hidden_dim, num_heads)


class TestComputeKMetric:
    def test_output_shape_no_rotary(self):
        """Without rotary → returns (N, hidden_dim)."""
        hidden_dim, num_heads, N = 64, 4, 10
        block = _MockBlock(hidden_dim, num_heads)
        hidden_states = mx.random.normal((N, hidden_dim))

        k = compute_k_metric(hidden_states, block, rotary_pos_emb=None)

        assert k.shape == (N, hidden_dim)

    def test_output_shape_with_rotary(self):
        """With rotary → still returns (N, hidden_dim)."""
        hidden_dim, num_heads, N = 64, 4, 10
        head_dim = hidden_dim // num_heads
        rope_dim = head_dim // 2 * 2  # = head_dim for even head_dim
        block = _MockBlock(hidden_dim, num_heads)
        hidden_states = mx.random.normal((N, hidden_dim))
        rotary_pos_emb = mx.random.normal((N, rope_dim))

        k = compute_k_metric(hidden_states, block, rotary_pos_emb=rotary_pos_emb)

        assert k.shape == (N, hidden_dim)

    def test_rotary_changes_values(self):
        """With rotary → returns different values than without."""
        hidden_dim, num_heads, N = 64, 4, 10
        head_dim = hidden_dim // num_heads
        rope_dim = head_dim // 2 * 2
        block = _MockBlock(hidden_dim, num_heads)
        hidden_states = mx.random.normal((N, hidden_dim))
        rotary_pos_emb = mx.random.normal((N, rope_dim))

        k_no_rope = compute_k_metric(hidden_states, block, rotary_pos_emb=None)
        k_with_rope = compute_k_metric(hidden_states, block, rotary_pos_emb=rotary_pos_emb)

        # They should differ (unless rotary is zero, which is astronomically unlikely)
        assert not mx.allclose(k_no_rope, k_with_rope, atol=1e-5), (
            "K with and without RoPE should differ"
        )

    def test_with_rmsnorm(self):
        """Block with RMSNorm (non-identity) still works."""
        hidden_dim, num_heads, N = 64, 4, 8
        block = _MockBlock(hidden_dim, num_heads, use_identity_norm=False)
        hidden_states = mx.random.normal((N, hidden_dim))

        k = compute_k_metric(hidden_states, block, rotary_pos_emb=None)

        assert k.shape == (N, hidden_dim)
        # Should be finite
        assert mx.all(mx.isfinite(k))

    def test_deterministic(self):
        """Same input → same output (no randomness in K computation)."""
        hidden_dim, num_heads, N = 32, 2, 5
        block = _MockBlock(hidden_dim, num_heads)
        hidden_states = mx.random.normal((N, hidden_dim))

        k1 = compute_k_metric(hidden_states, block)
        k2 = compute_k_metric(hidden_states, block)

        assert mx.array_equal(k1, k2)
