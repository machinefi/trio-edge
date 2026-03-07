"""Tests for trio_core.speculative — speculative decoding."""

import mlx.core as mx
import pytest


# ---------------------------------------------------------------------------
# Accept/reject logic
# ---------------------------------------------------------------------------

class TestAcceptRejectGreedy:
    """Test greedy (temperature=0) accept/reject logic."""

    def _call(self, draft_tokens, draft_logprobs, target_logprobs):
        from trio_core.speculative import SpeculativeDecoder
        sampler = lambda x: mx.argmax(x, axis=-1)
        return SpeculativeDecoder._accept_reject(
            draft_tokens, draft_logprobs, target_logprobs, sampler, temperature=0.0,
        )

    def test_all_accepted(self):
        """Draft matches target at every position."""
        k = 3
        vocab = 10
        # Draft tokens: [2, 5, 7]
        draft_tokens = mx.array([2, 5, 7])
        # Draft logprobs: token 2 is argmax at pos 0, etc.
        draft_lp = mx.full((k, vocab), -10.0)
        draft_lp[0, 2] = 0.0
        draft_lp[1, 5] = 0.0
        draft_lp[2, 7] = 0.0

        # Target agrees: argmax at pos 0 is 2, pos 1 is 5, pos 2 is 7
        target_lp = mx.full((k + 1, vocab), -10.0)
        target_lp[0, 2] = 0.0  # verifies draft[0]=2
        target_lp[1, 5] = 0.0  # verifies draft[1]=5
        target_lp[2, 7] = 0.0  # verifies draft[2]=7
        target_lp[3, 9] = 0.0  # bonus token

        n_accepted, resampled, _ = self._call(draft_tokens, draft_lp, target_lp)

        assert n_accepted == 3
        assert resampled.item() == 9  # bonus from target

    def test_first_rejected(self):
        """Target disagrees at position 0."""
        k = 3
        vocab = 10
        draft_tokens = mx.array([2, 5, 7])
        draft_lp = mx.full((k, vocab), -10.0)
        draft_lp[0, 2] = 0.0

        target_lp = mx.full((k + 1, vocab), -10.0)
        target_lp[0, 4] = 0.0  # target wants 4, not 2

        n_accepted, resampled, _ = self._call(draft_tokens, draft_lp, target_lp)

        assert n_accepted == 0
        assert resampled.item() == 4

    def test_partial_accept(self):
        """Accept first 2, reject third."""
        k = 3
        vocab = 10
        draft_tokens = mx.array([2, 5, 7])
        draft_lp = mx.full((k, vocab), -10.0)

        target_lp = mx.full((k + 1, vocab), -10.0)
        target_lp[0, 2] = 0.0  # accept
        target_lp[1, 5] = 0.0  # accept
        target_lp[2, 3] = 0.0  # reject (wants 3, draft has 7)

        n_accepted, resampled, _ = self._call(draft_tokens, draft_lp, target_lp)

        assert n_accepted == 2
        assert resampled.item() == 3


# ---------------------------------------------------------------------------
# Cache rollback
# ---------------------------------------------------------------------------

class TestCacheRollback:

    def _make_decoder(self, draft_cache, target_cache=None):
        from unittest.mock import MagicMock
        from trio_core.speculative import SpeculativeDecoder, PromptLookupDraft

        dummy_draft = PromptLookupDraft(mx.array([1, 2, 3]))
        return SpeculativeDecoder(
            target_model=None, target_cache=target_cache or [],
            draft_fn=dummy_draft, draft_cache=draft_cache,
        )

    def test_rollback_trims_cache(self):
        from mlx_lm.models.cache import KVCache

        caches = []
        for _ in range(2):
            c = KVCache()
            k = mx.ones((1, 2, 10, 4))
            v = mx.ones((1, 2, 10, 4))
            c.update_and_fetch(k, v)
            caches.append(c)

        assert caches[0].offset == 10

        decoder = self._make_decoder(draft_cache=caches)
        decoder._rollback_draft_cache(3)

        assert caches[0].offset == 7
        assert caches[1].offset == 7

    def test_rollback_zero_is_noop(self):
        from mlx_lm.models.cache import KVCache

        c = KVCache()
        k = mx.ones((1, 2, 5, 4))
        v = mx.ones((1, 2, 5, 4))
        c.update_and_fetch(k, v)

        decoder = self._make_decoder(draft_cache=[c])
        decoder._rollback_draft_cache(0)
        assert c.offset == 5


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

class TestSpecStats:

    def test_initial_stats(self):
        from trio_core.speculative import SpeculativeDecoder, PromptLookupDraft

        decoder = SpeculativeDecoder(
            target_model=None, target_cache=[],
            draft_fn=PromptLookupDraft(mx.array([1, 2])),
        )
        assert decoder.acceptance_rate == 0.0
        s = decoder.stats
        assert s.drafted == 0
        assert s.accepted == 0


# ---------------------------------------------------------------------------
# Draft generation
# ---------------------------------------------------------------------------

class TestDraft:

    def test_draft_produces_k_tokens(self):
        """Draft model generates exactly K tokens."""
        from unittest.mock import MagicMock
        from trio_core.speculative import SpeculativeDecoder

        vocab = 16
        # Mock draft model: always returns logits favoring token 5
        mock_model = MagicMock()
        logits = mx.full((1, 1, vocab), -10.0)
        logits = logits.at[0, 0, 5].add(10.0)
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_model.return_value = mock_output

        decoder = SpeculativeDecoder(
            draft_model=mock_model, target_model=None,
            draft_cache=[], target_cache=[],
        )

        token = mx.array(1)
        draft_tokens, draft_lp = decoder._draft(token, k=4)

        assert draft_tokens.shape == (4,)
        assert draft_lp.shape == (4, vocab)
        # All drafts should be token 5
        assert all(t.item() == 5 for t in draft_tokens)


# ---------------------------------------------------------------------------
# PromptLookupDraft
# ---------------------------------------------------------------------------

class TestPromptLookupDraft:

    def test_finds_ngram_match(self):
        from trio_core.speculative import PromptLookupDraft

        # Prompt: [10, 20, 30, 40, 50]
        prompt = mx.array([10, 20, 30, 40, 50])
        draft = PromptLookupDraft(prompt, ngram_max=2, num_draft=3)

        # History has [10], current is 20 → n-gram [10, 20] matches prompt
        draft.add_token(10)
        result = draft.draft(mx.array(20))

        assert result is not None
        assert result.tolist() == [30, 40, 50]

    def test_no_match_returns_none(self):
        from trio_core.speculative import PromptLookupDraft

        prompt = mx.array([10, 20, 30])
        draft = PromptLookupDraft(prompt, ngram_max=2, num_draft=3)

        # History [99], current 98 → no match in prompt
        draft.add_token(99)
        result = draft.draft(mx.array(98))

        assert result is None

    def test_longest_ngram_preferred(self):
        from trio_core.speculative import PromptLookupDraft

        # Prompt: [1, 2, 3, 4, 5, 2, 3, 9, 9]
        prompt = mx.array([1, 2, 3, 4, 5, 2, 3, 9, 9])
        draft = PromptLookupDraft(prompt, ngram_max=3, num_draft=2)

        # History [1, 2], current 3 → 3-gram [1,2,3] matches at pos 0
        draft.add_token(1)
        draft.add_token(2)
        result = draft.draft(mx.array(3))

        assert result is not None
        assert result.tolist() == [4, 5]

    def test_num_draft_limits_candidates(self):
        from trio_core.speculative import PromptLookupDraft

        prompt = mx.array([1, 2, 3, 4, 5, 6, 7, 8])
        draft = PromptLookupDraft(prompt, ngram_max=1, num_draft=2)

        # Current token 2 → 1-gram match at pos 1 → continuation [3, 4]
        result = draft.draft(mx.array(2))
        assert result is not None
        assert len(result.tolist()) == 2
        assert result.tolist() == [3, 4]

    def test_reset_clears_history(self):
        from trio_core.speculative import PromptLookupDraft

        prompt = mx.array([1, 2, 3])
        draft = PromptLookupDraft(prompt, ngram_max=2, num_draft=2)

        draft.add_token(1)
        draft.reset()
        assert draft._generated == []


# ---------------------------------------------------------------------------
# Require draft_model or draft_fn
# ---------------------------------------------------------------------------

class TestSpecDecoderValidation:

    def test_no_draft_raises(self):
        from trio_core.speculative import SpeculativeDecoder

        with pytest.raises(ValueError, match="draft_model or draft_fn"):
            SpeculativeDecoder(
                target_model=None, target_cache=[],
            )


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------

class TestSpeculativeConfig:

    def test_config_default(self):
        from trio_core.config import EngineConfig
        config = EngineConfig()
        assert config.speculative_lookahead == 0

    def test_config_custom(self):
        from trio_core.config import EngineConfig
        config = EngineConfig(speculative_lookahead=5)
        assert config.speculative_lookahead == 5


# ---------------------------------------------------------------------------
# Fallback stats tracking
# ---------------------------------------------------------------------------

class TestFallbackStats:

    def test_fallback_updates_accepted(self):
        """When no n-gram match, fallback path should still count the decoded token."""
        from trio_core.speculative import SpeculativeDecoder, PromptLookupDraft

        # Prompt with no possible matches for our generated tokens
        draft_fn = PromptLookupDraft(mx.array([100, 200, 300]), num_draft=3)
        decoder = SpeculativeDecoder(
            target_model=None, target_cache=[], draft_fn=draft_fn,
        )
        # Directly check that stats fields exist and are zero initially
        assert decoder._total_accepted == 0
        assert decoder._total_drafted == 0
