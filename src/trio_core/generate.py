"""Custom generate loop — drop-in replacement for mlx-vlm's generate_step.

Phase 1: Custom generate with persistent KV cache and early stopping.

Usage:
    from trio_core.generate import generate_step, PromptCache

    # Basic (no cache reuse):
    for token, logprobs in generate_step(input_ids, model, pixel_values, mask):
        ...

    # With persistent cache:
    pcache = PromptCache(model)
    for token, logprobs in generate_step(input_ids, model, pixel_values, mask,
                                          prompt_cache_manager=pcache):
        ...
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce

logger = logging.getLogger(__name__)


# ── Internalized utilities (from mlx-lm) ────────────────────────────────────
# These were previously imported from mlx_lm.sample_utils and mlx_lm.generate.
# Internalized to remove the mlx-lm runtime dependency.


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _categorical_sampling(logits, temp):
    return mx.random.categorical(logits * (1 / temp))


@partial(mx.compile, inputs=mx.random.state, outputs=mx.random.state)
def _apply_top_p(logprobs: mx.array, top_p: float) -> mx.array:
    probs = mx.exp(logprobs)
    sorted_indices = mx.argsort(logprobs, axis=-1)
    sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)
    cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
    inverse_indices = mx.put_along_axis(
        mx.zeros_like(sorted_indices),
        sorted_indices,
        mx.arange(sorted_indices.shape[-1], dtype=sorted_indices.dtype),
        axis=-1,
    )
    cumulative_probs = mx.take_along_axis(cumulative_probs, inverse_indices, axis=-1)
    return mx.where(cumulative_probs > 1 - top_p, logprobs, -float("inf"))


def _make_repetition_penalty(penalty: float, context_size: int = 20):
    if penalty < 0 or not isinstance(penalty, (int, float)):
        raise ValueError(f"penalty must be a non-negative float, got {penalty}")

    def repetition_penalty_processor(tokens, logits):
        if len(tokens) > 0:
            tokens = tokens[-context_size:]
            selected_logits = logits[:, tokens]
            selected_logits = mx.where(
                selected_logits < 0,
                selected_logits * penalty,
                selected_logits / penalty,
            )
            logits[:, tokens] = selected_logits
        return logits

    return repetition_penalty_processor


def make_sampler(
    temp: float = 0.0,
    top_p: float = 0.0,
) -> Callable[[mx.array], mx.array]:
    """Create a sampler function for token generation."""
    if temp == 0:
        return lambda x: mx.argmax(x, axis=-1)

    sampling_methods = []
    if top_p > 0 and top_p < 1.0:
        sampling_methods.append(lambda x: _apply_top_p(x, top_p))

    def sampler(logprobs):
        for method in sampling_methods:
            logprobs = method(logprobs)
        return _categorical_sampling(logprobs, temp)

    return sampler


def make_logits_processors(
    logit_bias: Optional[Dict[int, float]] = None,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
) -> List[Callable]:
    """Create logits processors for bias and repetition penalty."""
    processors = []
    if logit_bias:
        indices = mx.array(list(logit_bias.keys()))
        values = mx.array(list(logit_bias.values()))

        def logit_bias_processor(_, logits):
            logits[:, indices] += values
            return logits

        processors.append(logit_bias_processor)

    if repetition_penalty and repetition_penalty != 0.0:
        processors.append(
            _make_repetition_penalty(repetition_penalty, repetition_context_size)
        )
    return processors


def maybe_quantize_kv_cache(prompt_cache, quantized_kv_start, kv_group_size, kv_bits):
    """Quantize KV cache entries if kv_bits is set."""
    if kv_bits is None:
        return
    for e, c in enumerate(prompt_cache):
        if hasattr(c, "to_quantized") and c.offset >= quantized_kv_start:
            prompt_cache[e] = c.to_quantized(group_size=kv_group_size, bits=kv_bits)


def make_prompt_cache(
    model: nn.Module,
    max_kv_size: Optional[int] = None,
) -> List[Any]:
    """Construct KV cache for the model's language model.

    Defers to model.make_cache() if available (e.g. DeltaNet's ArraysCache),
    otherwise creates standard KVCache per layer.
    """
    if hasattr(model, "make_cache"):
        return model.make_cache()

    from mlx_lm.models.cache import KVCache, RotatingKVCache

    num_layers = len(model.layers)
    if max_kv_size is not None:
        return [RotatingKVCache(max_size=max_kv_size, keep=4) for _ in range(num_layers)]
    return [KVCache() for _ in range(num_layers)]


@dataclass
class EarlyStopConfig:
    """EOS-probability-based early stopping for decode.

    Checks P(EOS) after generating min_tokens. If the model is confident
    the response is complete, we stop early instead of generating padding
    or repetitive tokens.

    Only affects decode — prefill is untouched. Accuracy should be identical
    to no-early-stop when threshold is set correctly (the model was going to
    emit EOS anyway).
    """

    eos_threshold: float = 0.8       # Stop if P(EOS) > this after min_tokens
    min_tokens: int = 1              # Don't stop before this many tokens
    eos_token_ids: list[int] = field(default_factory=list)  # From model config

    def should_stop(self, logprobs: mx.array, n_generated: int) -> bool:
        """Check if we should early-stop based on EOS probability."""
        if n_generated < self.min_tokens:
            return False
        if not self.eos_token_ids:
            return False
        # logprobs is (vocab_size,) — convert to probabilities for EOS tokens
        for eos_id in self.eos_token_ids:
            p_eos = mx.exp(logprobs[eos_id]).item()
            if p_eos > self.eos_threshold:
                logger.debug(
                    "Early stop: P(EOS=%d)=%.3f > %.3f after %d tokens",
                    eos_id, p_eos, self.eos_threshold, n_generated,
                )
                return True
        return False


class PromptCache:
    """Persistent KV cache manager for cross-request reuse.

    Benefits:
    1. Buffer reuse: avoids GPU buffer re-allocation between requests
    2. Exact-match: identical input_ids → skip entire prefill (ViT + LLM)

    Usage:
        pcache = PromptCache(model)
        # Pass to generate_step — it handles cache lifecycle
        for token, logprobs in generate_step(..., prompt_cache_manager=pcache):
            ...
    """

    def __init__(self, model: nn.Module, *, max_kv_size: Optional[int] = None):
        self._kv_cache: Optional[List[Any]] = None
        self._input_hash: Optional[str] = None
        self._first_token: Optional[Tuple[mx.array, mx.array]] = None
        self._cached_kwargs: Dict[str, Any] = {}  # cross_attention_states etc.
        self._prefill_offset: int = 0  # KV offset after prefill (before decode)
        self._model = model
        self._max_kv_size = max_kv_size

    @property
    def kv_cache(self) -> Optional[List[Any]]:
        return self._kv_cache

    @property
    def is_trimmable(self) -> bool:
        """Check if cache supports trim/offset (KVCache vs ArraysCache)."""
        return (self._kv_cache is not None
                and len(self._kv_cache) > 0
                and hasattr(self._kv_cache[0], 'offset'))

    def get_or_create_cache(self) -> List[Any]:
        """Get existing cache (trimmed to 0) or create new one.

        For KVCache (standard attention): reuse buffers, trim to 0.
        For ArraysCache (DeltaNet): must recreate — no trim support.
        """
        if self._kv_cache is not None:
            if self.is_trimmable:
                # KVCache: reuse buffers — trim all content but keep allocated memory
                for c in self._kv_cache:
                    c.trim(c.offset)
                return self._kv_cache
            else:
                # ArraysCache (DeltaNet): can't trim, must recreate
                pass

        self._kv_cache = make_prompt_cache(
            self._model.language_model,
            max_kv_size=self._max_kv_size,
        )
        return self._kv_cache

    def check_hit(self, input_ids: mx.array, pixel_values: mx.array = None) -> bool:
        """Check if current input exactly matches cached state.

        Only works with trimmable caches (KVCache). ArraysCache (DeltaNet)
        cannot be trimmed, so exact-match reuse is not supported.
        """
        if not self.is_trimmable:
            return False
        h = self._hash_input(input_ids, pixel_values)
        return self._input_hash is not None and self._input_hash == h

    def save_state(self, input_ids: mx.array, first_token: mx.array, first_logprobs: mx.array,
                   kv_cache: List[Any], pixel_values: mx.array = None,
                   kwargs: Optional[Dict[str, Any]] = None):
        """Save the input hash, first token, and prefill offset for reuse."""
        self._input_hash = self._hash_input(input_ids, pixel_values)
        self._first_token = (first_token, first_logprobs)
        self._cached_kwargs = kwargs or {}
        # Record KV offset right after prefill (before decode starts)
        # ArraysCache (DeltaNet) has no offset — cache reuse not supported
        if kv_cache and hasattr(kv_cache[0], 'offset'):
            self._prefill_offset = kv_cache[0].offset
        else:
            self._prefill_offset = 0

    def get_cached_first_token(self) -> Optional[Tuple[mx.array, mx.array]]:
        """Get cached first token from exact-match hit."""
        return self._first_token

    def invalidate(self):
        """Clear cached state (keeps buffers for reuse)."""
        self._input_hash = None
        self._first_token = None

    @staticmethod
    def _hash_input(input_ids: mx.array, pixel_values: mx.array = None) -> str:
        """Fast hash of input_ids + pixel_values for exact-match detection."""
        import numpy as np
        h = hashlib.md5(np.array(input_ids.flatten(), copy=False).tobytes())
        if pixel_values is not None and pixel_values.size > 0:
            h.update(np.array(pixel_values.flatten(), copy=False).tobytes())
        return h.hexdigest()

# Dedicated stream for generation (matches mlx-vlm)
_generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def _wired_limit(model: nn.Module):
    """Temporarily set Metal wired memory limit based on model size.

    Copied from mlx-vlm to ensure identical memory behavior.
    """
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        logger.warning(
            "Model requires %d MB, close to max recommended %d MB. May be slow.",
            model_mb, max_rec_mb,
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        mx.synchronize(_generation_stream)
        mx.synchronize()
        mx.set_wired_limit(old_limit)


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prompt_cache_manager: Optional[PromptCache] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = 2048,
    early_stop: Optional[EarlyStopConfig] = None,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Generate tokens with optional persistent cache and early stopping.

    When prompt_cache_manager is provided:
    - Reuses GPU buffers across requests (avoids re-allocation)
    - Detects exact-match inputs → skips entire prefill (ViT + LLM)

    When early_stop is provided:
    - Checks P(EOS) after each decode step
    - Stops early if model is confident response is complete

    Yields:
        (token_id, logprobs) tuples.
    """
    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(temperature, top_p)

    processors = make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    original_input_ids = input_ids  # Save before chunked prefill modifies it
    tokens = mx.array([], dtype=input_ids.dtype)

    # Check for exact-match cache hit
    cache_hit = False
    if prompt_cache_manager is not None:
        cache_hit = prompt_cache_manager.check_hit(original_input_ids, pixel_values)
        if cache_hit:
            prompt_cache = prompt_cache_manager.kv_cache
            # Trim decode tokens, keep only prefill state
            for c in prompt_cache:
                if hasattr(c, 'trim'):
                    decode_tokens = c.offset - prompt_cache_manager._prefill_offset
                    if decode_tokens > 0:
                        c.trim(decode_tokens)
            logger.debug("Cache HIT — skipping prefill (%d tokens)", input_ids.size)
        else:
            prompt_cache = prompt_cache_manager.get_or_create_cache()
            logger.debug("Cache MISS — full prefill (%d tokens)", input_ids.size)
    elif prompt_cache is None:
        prompt_cache = make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs

        with mx.stream(_generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])
                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)

            # Propagate cross-attention / encoder states (for encoder-decoder models)
            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0)

    if cache_hit:
        # Exact match — skip ViT + prefill entirely, use cached first token
        cached = prompt_cache_manager.get_cached_first_token()
        y, logprobs = cached
        # Restore kwargs (cross_attention_states etc.) for encoder-decoder models
        kwargs = dict(prompt_cache_manager._cached_kwargs)
    else:
        with mx.stream(_generation_stream):
            # Get input embeddings (ViT forward + embedding projection)
            embedding_output = model.get_input_embeddings(
                input_ids, pixel_values, mask=mask, **kwargs
            )

            inputs_embeds = embedding_output.inputs_embeds

            kwargs.update(
                {
                    k: v
                    for k, v in embedding_output.to_dict().items()
                    if k != "inputs_embeds" and v is not None
                }
            )

            # Chunked prefill for long sequences
            if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()

                input_ids = input_ids[:, -1:]

            # First step: prefill remaining + get first token
            y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

        # Save state for future exact-match detection
        if prompt_cache_manager is not None:
            prompt_cache_manager.save_state(
                original_input_ids, y, logprobs, prompt_cache, pixel_values, kwargs=kwargs
            )

    mx.async_eval(y)

    # Decode loop
    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs

        # Early stopping: check NEXT step's P(EOS) — already computed at loop top
        if early_stop is not None and early_stop.should_stop(next_logprobs, n + 1):
            break

        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1
