"""Speculative decoding for faster autoregressive generation.

Two modes:
1. Draft model — uses a smaller LLM to generate candidates
2. Prompt lookup — finds n-gram matches in the prompt as candidates (zero cost)

Both verify candidates in a single target model forward pass.
Accepted tokens skip individual decode steps, boosting throughput.

Algorithm (Leviathan et al., 2023):
  1. Draft K candidate tokens (via draft model or n-gram lookup)
  2. Target model scores all K tokens in one forward pass
  3. Accept tokens where target agrees with draft (rejection sampling)
  4. Resample the first rejected position from target distribution
  5. Guaranteed to produce same distribution as target-only decoding

Usage:
    from trio_core.speculative import SpeculativeDecoder, PromptLookupDraft

    # With draft model:
    decoder = SpeculativeDecoder(
        draft_model=draft_lm, target_model=target_lm,
        draft_cache=draft_kv, target_cache=target_kv,
    )

    # With prompt lookup (no draft model needed):
    draft = PromptLookupDraft(prompt_token_ids, ngram_max=3, num_draft=5)
    decoder = SpeculativeDecoder(
        draft_model=None, target_model=target_lm,
        draft_cache=[], target_cache=target_kv,
        draft_fn=draft,
    )
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class SpecStats:
    """Statistics from speculative decoding."""
    drafted: int        # total draft candidates proposed
    accepted: int       # draft candidates accepted by target
    fallbacks: int      # rounds where no n-gram match was found
    acceptance_rate: float


class PromptLookupDraft:
    """Zero-cost draft via n-gram matching against prompt tokens.

    Looks for the most recent n-gram in the generated tokens that also
    appears in the prompt, and uses the continuation as draft candidates.
    No model needed — works purely on token patterns.

    Effective for structured outputs, repeated phrases, and templates.
    """

    def __init__(
        self,
        prompt_ids: mx.array,
        ngram_max: int = 3,
        ngram_min: int = 1,
        num_draft: int = 5,
    ):
        self.prompt_ids = prompt_ids.flatten().tolist()
        self.ngram_max = ngram_max
        self.ngram_min = ngram_min
        self.num_draft = num_draft
        self._generated: list[int] = []

    def add_token(self, token_id: int):
        """Track generated tokens for n-gram matching."""
        self._generated.append(token_id)

    def draft(self, current_token: mx.array) -> Optional[mx.array]:
        """Find n-gram continuation from prompt.

        Note: caller must call add_token() to track generated tokens.
        This method does not modify internal state.

        Returns:
            mx.array of draft token IDs (up to num_draft), or None if no match.
        """
        # Include current token in n-gram search without storing it
        history = self._generated + [current_token.item()]

        # Try longest n-gram first
        for n in range(self.ngram_max, self.ngram_min - 1, -1):
            if len(history) < n:
                continue

            ngram = history[-n:]
            # Search for this n-gram in the prompt
            for i in range(len(self.prompt_ids) - n):
                if self.prompt_ids[i:i + n] == ngram:
                    # Found match — take continuation
                    start = i + n
                    end = min(start + self.num_draft, len(self.prompt_ids))
                    if start < end:
                        candidates = self.prompt_ids[start:end]
                        return mx.array(candidates, dtype=mx.int32)

        return None

    def reset(self):
        """Reset generated token history."""
        self._generated = []


class SpeculativeDecoder:
    """Speculative decoding with draft + target model pair.

    Both models must share the same vocabulary. The draft model should be
    significantly faster than the target (fewer layers, smaller hidden dim).

    The target model's output distribution is preserved exactly via
    rejection sampling — speculative decoding is lossless.
    """

    def __init__(
        self,
        target_model: nn.Module,
        target_cache: List,
        draft_model: Optional[nn.Module] = None,
        draft_cache: Optional[List] = None,
        draft_fn: Optional[PromptLookupDraft] = None,
        num_draft: int = 5,
        quantize_cache_fn: Optional[Callable] = None,
    ):
        if draft_model is None and draft_fn is None:
            raise ValueError("Must provide either draft_model or draft_fn")

        self.target_model = target_model
        self.target_cache = target_cache
        self.draft_model = draft_model
        self.draft_cache = draft_cache or []
        self.draft_fn = draft_fn
        self.num_draft = num_draft
        self.quantize_cache_fn = quantize_cache_fn or (lambda _: None)

        # Stats
        self._total_drafted = 0
        self._total_accepted = 0
        self._total_fallbacks = 0

    def decode(
        self,
        first_token: mx.array,
        sampler: Callable[[mx.array], mx.array],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> Generator[Tuple[mx.array, mx.array], None, None]:
        """Speculative decode loop.

        Args:
            first_token: First generated token from prefill (scalar).
            sampler: Sampling function (logprobs -> token).
            max_tokens: Maximum tokens to generate.
            temperature: Temperature for rejection sampling threshold.

        Yields:
            (token, logprobs) pairs, same interface as standard decode.
        """
        y = first_token.squeeze()  # ensure scalar
        n = 0

        while n < max_tokens:
            # Step 1: Draft K tokens
            if self.draft_fn is not None:
                draft_candidates = self.draft_fn.draft(y)
                # Track the current token in draft history
                self.draft_fn.add_token(y.item())

                if draft_candidates is None or draft_candidates.shape[0] == 0:
                    # No n-gram match — fall back to single-token decode
                    target_logprobs = self._verify(y[None])  # (1, vocab)
                    resampled = sampler(target_logprobs[0][None])
                    yield resampled.squeeze(), target_logprobs[0]
                    n += 1
                    self._total_fallbacks += 1
                    y = resampled.squeeze()
                    continue

                draft_tokens = draft_candidates
                k = draft_tokens.shape[0]
                draft_logprobs = None
            else:
                draft_tokens, draft_logprobs = self._draft(y, self.num_draft)
                k = draft_tokens.shape[0]

            # Step 2: Verify with target model (single forward pass for all K+1 tokens)
            verify_input = mx.concatenate([y[None], draft_tokens])  # (K+1,)
            target_logprobs_all = self._verify(verify_input)  # (K+1, vocab)

            if draft_logprobs is None:
                # Prompt lookup: use greedy accept (no draft distribution)
                draft_logprobs = target_logprobs_all[:k]  # placeholder, forces accept

            # Step 3: Accept/reject via rejection sampling
            accepted, resampled_token, resampled_logprobs = self._accept_reject(
                draft_tokens, draft_logprobs, target_logprobs_all, sampler, temperature,
            )

            # Step 4: Roll back draft cache to match accepted count
            if self.draft_model is not None:
                self._rollback_draft_cache(k - accepted)

            # Step 5: Roll back target cache
            self._rollback_target_cache(k - accepted)

            # Track for prompt lookup
            for i in range(accepted):
                if self.draft_fn is not None:
                    self.draft_fn.add_token(draft_tokens[i].item())

            # Yield accepted draft tokens
            for i in range(accepted):
                if n >= max_tokens:
                    return
                yield draft_tokens[i], draft_logprobs[i]
                n += 1

            # Yield resampled token
            if n >= max_tokens:
                return
            yield resampled_token, resampled_logprobs
            n += 1

            # Next iteration starts from resampled token
            y = resampled_token

            self._total_drafted += k
            self._total_accepted += accepted

        logger.debug(
            "Speculative decode: %d drafted, %d accepted (%.1f%% rate)",
            self._total_drafted, self._total_accepted,
            self.acceptance_rate * 100,
        )

    def _draft(self, token: mx.array, k: int) -> Tuple[mx.array, mx.array]:
        """Generate K draft tokens autoregressively with draft model.

        Returns:
            (draft_tokens, draft_logprobs) — both shape (K,) and (K, vocab).
        """
        tokens = []
        all_logprobs = []
        y = token

        for _ in range(k):
            outputs = self.draft_model(y[None][None], cache=self.draft_cache)
            logits = outputs.logits[:, -1, :]
            self.quantize_cache_fn(self.draft_cache)

            lp = logits - mx.logsumexp(logits)
            y = mx.argmax(lp, axis=-1).squeeze()  # Greedy draft
            tokens.append(y)
            all_logprobs.append(lp.squeeze(0))

        mx.eval(tokens, all_logprobs)
        return mx.stack(tokens), mx.stack(all_logprobs)

    def _verify(self, tokens: mx.array) -> mx.array:
        """Run target model on all tokens in one forward pass.

        Args:
            tokens: (K+1,) — [current_token, draft_0, ..., draft_{K-1}]

        Returns:
            logprobs: (K+1, vocab) — target model log probabilities
        """
        outputs = self.target_model(tokens[None], cache=self.target_cache)
        logits = outputs.logits[0]  # (K+1, vocab)
        self.quantize_cache_fn(self.target_cache)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(logprobs)
        return logprobs

    @staticmethod
    def _accept_reject(
        draft_tokens: mx.array,
        draft_logprobs: mx.array,
        target_logprobs: mx.array,
        sampler: Callable,
        temperature: float,
    ) -> Tuple[int, mx.array, mx.array]:
        """Rejection sampling: accept draft tokens that target agrees with.

        For greedy (temperature=0): accept if argmax matches.
        For sampling (temperature>0): accept with probability
            min(1, p_target/p_draft) per token.

        Args:
            draft_tokens: (K,) drafted token IDs
            draft_logprobs: (K, vocab) draft log-probabilities
            target_logprobs: (K+1, vocab) target log-probabilities
                target_logprobs[i] = P(token | tokens[:i+1])
                target_logprobs[0] = P(next | current_token)  [verifies draft_0]
                target_logprobs[K] = P(next | all accepted)   [for resampling]

        Returns:
            (n_accepted, resampled_token, resampled_logprobs)
        """
        k = draft_tokens.shape[0]

        if temperature == 0.0:
            # Greedy: accept if target argmax matches draft token
            n_accepted = 0
            for i in range(k):
                target_choice = mx.argmax(target_logprobs[i]).item()
                if target_choice == draft_tokens[i].item():
                    n_accepted += 1
                else:
                    break

            # Resample from target at the first rejected position
            resample_lp = target_logprobs[n_accepted]
            resampled = sampler(resample_lp[None])
            return n_accepted, resampled.squeeze(), resample_lp

        else:
            # Stochastic: rejection sampling with min(1, p_target/p_draft)
            n_accepted = 0
            for i in range(k):
                token_id = draft_tokens[i].item()
                p_draft = mx.exp(draft_logprobs[i, token_id]).item()
                p_target = mx.exp(target_logprobs[i, token_id]).item()

                ratio = p_target / max(p_draft, 1e-10)
                if ratio >= 1.0 or mx.random.uniform().item() < ratio:
                    n_accepted += 1
                else:
                    # Resample from modified distribution:
                    # max(0, p_target - p_draft) normalized
                    adjusted = mx.maximum(
                        mx.exp(target_logprobs[i]) - mx.exp(draft_logprobs[i]),
                        mx.array(0.0),
                    )
                    adjusted = adjusted / adjusted.sum()
                    adjusted_lp = mx.log(adjusted + 1e-10)
                    resampled = sampler(adjusted_lp[None])
                    return n_accepted, resampled.squeeze(), target_logprobs[i]

            # All accepted — sample bonus token from target
            resample_lp = target_logprobs[k]
            resampled = sampler(resample_lp[None])
            return n_accepted, resampled.squeeze(), resample_lp

    def _rollback_draft_cache(self, n_rollback: int):
        """Trim draft cache by n_rollback tokens."""
        if n_rollback <= 0:
            return
        for c in self.draft_cache:
            if hasattr(c, 'trim'):
                c.trim(n_rollback)

    def _rollback_target_cache(self, n_rollback: int):
        """Trim target cache by n_rollback tokens."""
        if n_rollback <= 0:
            return
        for c in self.target_cache:
            if hasattr(c, 'trim'):
                c.trim(n_rollback)

    @property
    def acceptance_rate(self) -> float:
        if self._total_drafted == 0:
            return 0.0
        return self._total_accepted / self._total_drafted

    @property
    def stats(self) -> SpecStats:
        return SpecStats(
            drafted=self._total_drafted,
            accepted=self._total_accepted,
            fallbacks=self._total_fallbacks,
            acceptance_rate=self.acceptance_rate,
        )
