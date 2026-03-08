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

    Four-tier cache hierarchy (checked in order):
    1. Exact-match: identical input_ids + pixel_values → skip entire prefill
    2. Visual-similarity: similar embeddings (cosine > threshold) → reuse full KV
    3. Prefix-match: same prompt template + resolution → skip text prefix prefill
    4. Full miss: complete prefill from scratch

    Visual similarity (tier 2) enables frame-to-frame KV reuse for video:
    consecutive frames with similar content skip LLM prefill entirely,
    reusing the KV cache built from the previous frame. This is an
    approximation — the model "sees" the old frame's visual context.
    Quality vs speed tradeoff controlled by threshold (0.95 typical).

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

        # Prefix caching: reuse text prefix KV across frames
        self._prefix_hash: Optional[str] = None
        self._prefix_states: Optional[List[Any]] = None  # [(keys, values), ...]
        self._prefix_len: int = 0
        self._prefix_position_ids = None
        self._prefix_rope_deltas = None

        # Visual similarity: reuse full KV for similar frames
        self._last_embeds: Optional[mx.array] = None
        self._last_input_ids: Optional[mx.array] = None

        # StreamMem: bounded KV cache accumulation for continuous streams
        self._streaming_memory = None

    @property
    def kv_cache(self) -> Optional[List[Any]]:
        return self._kv_cache

    @property
    def streaming_memory(self):
        return self._streaming_memory

    def set_streaming_memory(self, memory):
        self._streaming_memory = memory

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
        In streaming mode: keep accumulated visual KV, only trim decode tokens.
        """
        if self._kv_cache is not None:
            if self.is_trimmable:
                if self._streaming_memory is not None:
                    # Streaming mode: keep accumulated KV, only trim decode tokens
                    for c in self._kv_cache:
                        if hasattr(c, 'trim'):
                            decode_tokens = c.offset - self._prefill_offset
                            if decode_tokens > 0:
                                c.trim(decode_tokens)
                    return self._kv_cache
                # Normal mode: trim all content but keep allocated memory
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
        if kv_cache and hasattr(kv_cache[0], 'offset'):
            self._prefill_offset = kv_cache[0].offset
        else:
            self._prefill_offset = 0
        # Snapshot ArraysCache (DeltaNet) state right after prefill.
        # DeltaNet recurrent state is modified in-place during decode and
        # cannot be rolled back, so we deep-copy the state arrays here.
        self._arrays_cache_snapshot = None
        if kv_cache and not self.is_trimmable:
            self._snapshot_arrays_cache(kv_cache)

    def _snapshot_arrays_cache(self, kv_cache: List[Any]):
        """Deep-copy ArraysCache states for later restore on visual_hit."""
        snapshot = []
        for c in kv_cache:
            if hasattr(c, 'state'):
                # ArraysCache: copy each state array
                snapshot.append([mx.array(a) if a is not None else None for a in c.state])
            else:
                snapshot.append(None)
        self._arrays_cache_snapshot = snapshot

    def restore_arrays_cache(self):
        """Restore ArraysCache state to post-prefill snapshot."""
        if self._arrays_cache_snapshot is None or self._kv_cache is None:
            return
        for c, snap in zip(self._kv_cache, self._arrays_cache_snapshot):
            if snap is not None and hasattr(c, 'state'):
                c.state = [mx.array(a) if a is not None else None for a in snap]

    def get_cached_first_token(self) -> Optional[Tuple[mx.array, mx.array]]:
        """Get cached first token from exact-match hit."""
        return self._first_token

    def invalidate(self):
        """Clear cached state (keeps buffers for reuse)."""
        self._input_hash = None
        self._first_token = None

    # ── Visual similarity ──────────────────────────────────────────

    def check_visual_hit(
        self, inputs_embeds: mx.array, input_ids: mx.array, threshold: float,
    ) -> bool:
        """Check if embeddings are similar enough to reuse full KV cache.

        This is an approximation: on hit, the model uses the OLD frame's
        visual KV cache, not the new frame's. Acceptable when frames are
        visually similar (surveillance, slow-moving video).

        Requires:
        - Same input_ids (same prompt/question). Different questions MUST
          NOT reuse KV because the answer depends on the question.
        - Similar visual embeddings (cosine similarity above threshold).

        Args:
            inputs_embeds: (1, seq_len, hidden_dim) from ViT + embedding merge.
            input_ids: (1, seq_len) token IDs for the current request.
            threshold: Cosine similarity threshold (0.95 typical).

        Returns:
            True if prompt matches AND similarity >= threshold.
        """
        if self._last_embeds is None:
            return False
        if self._kv_cache is None:
            return False
        if self._first_token is None:
            return False
        # Must be same prompt (same question) — different questions need
        # different answers even with the same image.
        if self._last_input_ids is None:
            return False
        if input_ids.shape != self._last_input_ids.shape:
            return False
        if not mx.array_equal(input_ids, self._last_input_ids):
            return False
        if inputs_embeds.shape != self._last_embeds.shape:
            return False

        # Per-token cosine similarity, 10th percentile (robust to text
        # tokens being identical — captures visual token differences).
        a = inputs_embeds.squeeze(0)  # (seq_len, hidden_dim)
        b = self._last_embeds.squeeze(0)
        dot = mx.sum(a * b, axis=-1)  # (seq_len,)
        norm_a = mx.sqrt(mx.sum(a * a, axis=-1) + 1e-8)
        norm_b = mx.sqrt(mx.sum(b * b, axis=-1) + 1e-8)
        cos_sim = dot / (norm_a * norm_b)
        # Sort and take 10th percentile — captures visual token similarity
        sorted_sim = mx.sort(cos_sim)
        p10_idx = max(0, int(cos_sim.size * 0.1))
        mean_sim = sorted_sim[p10_idx].item()

        if mean_sim >= threshold:
            logger.debug(
                "Visual similarity %.4f >= %.4f — KV reuse",
                mean_sim, threshold,
            )
            return True
        else:
            logger.debug(
                "Visual similarity %.4f < %.4f — re-prefill",
                mean_sim, threshold,
            )
            return False

    def save_embeds(self, inputs_embeds: mx.array, input_ids: mx.array = None):
        """Save input embeddings and IDs for future visual similarity comparison."""
        self._last_embeds = inputs_embeds
        if input_ids is not None:
            self._last_input_ids = input_ids

    # ── Prefix caching ──────────────────────────────────────────────

    def check_prefix_hit(self, input_ids: mx.array) -> bool:
        """Check if text prefix matches (same prompt template, different pixels).

        Hashes the full input_ids (not pixels). Same input_ids means same
        prompt + same visual token count (same resolution), so prefix KV
        and position_ids are valid for reuse.
        """
        if self._prefix_hash is None or self._prefix_states is None:
            return False
        # Prefix states must contain real KV data (not all None)
        if not any(s is not None for s in self._prefix_states):
            return False
        import numpy as np
        h = hashlib.md5(np.array(input_ids.flatten(), copy=False).tobytes(), usedforsecurity=False).hexdigest()
        return h == self._prefix_hash

    def save_prefix(
        self, input_ids: mx.array, prefix_len: int, kv_cache: List[Any],
        position_ids=None, rope_deltas=None,
    ):
        """Save text prefix KV state for reuse across frames.

        Args:
            input_ids: Full input_ids (hashed for matching).
            prefix_len: Number of text prefix tokens (before first visual token).
            kv_cache: KV cache after full prefill (contains all tokens' KV).
            position_ids: MRoPE position_ids to restore on prefix hit.
            rope_deltas: MRoPE rope_deltas to restore after decode.
        """
        import numpy as np
        if prefix_len <= 0:
            return
        self._prefix_hash = hashlib.md5(
            np.array(input_ids.flatten(), copy=False).tobytes(), usedforsecurity=False
        ).hexdigest()
        self._prefix_len = prefix_len
        self._prefix_position_ids = position_ids
        self._prefix_rope_deltas = rope_deltas
        self._prefix_states = []
        for c in kv_cache:
            if hasattr(c, 'keys') and c.keys is not None and c.offset >= prefix_len:
                k = c.keys[..., :prefix_len, :]
                v = c.values[..., :prefix_len, :]
                mx.eval(k, v)
                self._prefix_states.append((k, v))
            else:
                self._prefix_states.append(None)
        logger.debug("Prefix saved: %d tokens, %d layers", prefix_len, len(self._prefix_states))

    def restore_prefix_cache(self) -> List[Any]:
        """Create new KV cache initialized with saved prefix state."""
        cache = make_prompt_cache(
            self._model.language_model, max_kv_size=self._max_kv_size,
        )
        for c, state in zip(cache, self._prefix_states):
            if state is not None:
                k, v = state
                c.state = (k, v)
        self._kv_cache = cache
        return cache

    # ── Private ──────────────────────────────────────────────────────

    @staticmethod
    def _hash_input(input_ids: mx.array, pixel_values: mx.array = None) -> str:
        """Fast hash of input_ids + pixel_values for exact-match detection.

        Uses strided sampling for large pixel_values to avoid hashing
        multi-MB tensors. Shape + first/last/strided slices provide
        collision resistance without full-data cost.
        """
        import numpy as np
        h = hashlib.md5(np.array(input_ids.flatten(), copy=False).tobytes(), usedforsecurity=False)
        if pixel_values is not None and pixel_values.size > 0:
            flat = np.array(pixel_values.reshape(-1), copy=False)
            n = flat.shape[0]
            # Encode shape for collision resistance
            h.update(np.array(pixel_values.shape, dtype=np.int64).tobytes())
            if n <= 65536:
                h.update(flat.tobytes())
            else:
                # Sample first 8K + last 8K + 16K strided elements
                stride = max(1, n // 16384)
                h.update(flat[:8192].tobytes())
                h.update(flat[-8192:].tobytes())
                h.update(flat[::stride][:16384].tobytes())
        return h.hexdigest()

def _find_visual_boundary(input_ids: mx.array, model: nn.Module) -> int:
    """Find position of first visual placeholder token in input_ids.

    Returns the index where visual tokens start, or 0 if none found.
    The tokens before this position are the "text prefix" that can be cached.
    """
    import numpy as np
    ids = np.array(input_ids[0])
    img_id = getattr(model.config, 'image_token_id', None)
    if img_id is None:
        img_id = getattr(model.config, 'image_token_index', None)
    vid_id = getattr(model.config, 'video_token_id', None)
    if vid_id is None:
        vid_id = getattr(model.config, 'video_token_index', None)
    for i, tid in enumerate(ids):
        if (img_id is not None and tid == img_id) or (vid_id is not None and tid == vid_id):
            return int(i)
    return 0


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
    inputs_embeds: Optional[mx.array] = None,
    visual_similarity_threshold: float = 0.0,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """Generate tokens with optional persistent cache and early stopping.

    When prompt_cache_manager is provided:
    - Reuses GPU buffers across requests (avoids re-allocation)
    - Detects exact-match inputs → skips entire prefill (ViT + LLM)

    When inputs_embeds is provided:
    - Skips model.get_input_embeddings() (ViT + embedding)
    - Uses pre-computed embeddings directly (e.g., ToMe path)
    - Caller must set position_ids on model.language_model before calling

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

    # Check for exact-match cache hit, then prefix hit
    # Note: visual similarity check happens later (after ViT) — we must NOT
    # trim/reset the KV cache here if visual similarity might apply.
    cache_hit = False
    prefix_hit = False
    may_visual_hit = (
        visual_similarity_threshold > 0
        and prompt_cache_manager is not None
        and prompt_cache_manager._last_embeds is not None
    )
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
            if may_visual_hit:
                # Don't trim/restore KV cache yet — visual similarity check
                # needs the full previous-frame KV cache intact. Defer prefix_hit
                # until after ViT forward + visual similarity check.
                prefix_hit = prompt_cache_manager.check_prefix_hit(original_input_ids)
                prompt_cache = prompt_cache_manager.kv_cache or prompt_cache_manager.get_or_create_cache()
                logger.debug("Cache MISS — deferring for visual similarity check (prefix_match=%s)", prefix_hit)
            else:
                prefix_hit = prompt_cache_manager.check_prefix_hit(original_input_ids)
                if prefix_hit:
                    prompt_cache = prompt_cache_manager.restore_prefix_cache()
                    logger.debug(
                        "Prefix HIT — reusing %d prefix tokens, re-prefilling suffix",
                        prompt_cache_manager._prefix_len,
                    )
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

    visual_hit = False

    if cache_hit:
        # Exact match — skip ViT + prefill entirely, use cached first token
        cached = prompt_cache_manager.get_cached_first_token()
        y, logprobs = cached
        # Restore kwargs (cross_attention_states etc.) for encoder-decoder models
        kwargs = dict(prompt_cache_manager._cached_kwargs)
    else:
        with mx.stream(_generation_stream):
            if inputs_embeds is not None:
                # Pre-computed embeddings (e.g., ToMe path)
                # Caller already set position_ids and kwargs (deepstack etc.)
                pass
            else:
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

            # Save full embeddings before chunked prefill may slice them
            full_inputs_embeds = inputs_embeds

            # Visual similarity check (tier 2: between exact hit and prefix hit)
            # When may_visual_hit, we deferred prefix_hit restoration so the
            # full previous-frame KV cache is still intact for reuse.
            visual_hit = (
                may_visual_hit
                and prompt_cache_manager.check_visual_hit(
                    inputs_embeds, original_input_ids, visual_similarity_threshold,
                )
            )

            if visual_hit:
                # Reuse full KV cache from previous frame (approximate)
                prompt_cache = prompt_cache_manager.kv_cache
                # KVCache: trim decode tokens to restore prefill state
                for c in prompt_cache:
                    if hasattr(c, 'trim'):
                        decode_tokens = c.offset - prompt_cache_manager._prefill_offset
                        if decode_tokens > 0:
                            c.trim(decode_tokens)
                # ArraysCache (DeltaNet): restore from snapshot (state is
                # modified in-place during decode, can't roll back)
                prompt_cache_manager.restore_arrays_cache()
                cached = prompt_cache_manager.get_cached_first_token()
                y, logprobs = cached
                kwargs = dict(prompt_cache_manager._cached_kwargs)
                logger.debug(
                    "Visual similarity HIT — reusing KV cache (skipping LLM prefill)"
                )
            elif prefix_hit:
                # Prefix cache hit: skip text prefix prefill, only prefill
                # visual + suffix tokens. Restore prefix cache state now
                # (deferred earlier when may_visual_hit was true).
                if may_visual_hit:
                    prompt_cache = prompt_cache_manager.restore_prefix_cache()
                prefix_len = prompt_cache_manager._prefix_len
                lm = model.language_model

                # Restore full position_ids so model can slice for suffix
                # (only Qwen models with MRoPE store _position_ids/_rope_deltas)
                if hasattr(lm, '_position_ids'):
                    lm._position_ids = prompt_cache_manager._prefix_position_ids
                    saved_rope_deltas = prompt_cache_manager._prefix_rope_deltas
                    lm._rope_deltas = None
                else:
                    saved_rope_deltas = None

                suffix_embeds = inputs_embeds[:, prefix_len:]
                suffix_ids = input_ids[:, prefix_len:]

                y, logprobs = _step(suffix_ids, inputs_embeds=suffix_embeds)
                quantize_cache_fn(prompt_cache)

                # Restore rope_deltas for sequential decode phase
                if hasattr(lm, '_rope_deltas'):
                    lm._rope_deltas = saved_rope_deltas

                logger.debug(
                    "Prefix prefill skipped %d tokens, prefilled %d suffix tokens",
                    prefix_len, suffix_ids.shape[1],
                )
            else:
                # Visual similarity was deferred but failed — reset cache now
                if may_visual_hit and prompt_cache_manager is not None:
                    prompt_cache = prompt_cache_manager.get_or_create_cache()

                # Full prefill (possibly chunked)
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

                y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

                # Save text prefix KV for future prefix-match reuse
                if prompt_cache_manager is not None:
                    vis_boundary = _find_visual_boundary(original_input_ids, model)
                    if vis_boundary > 0:
                        lm = model.language_model
                        prompt_cache_manager.save_prefix(
                            original_input_ids, vis_boundary, prompt_cache,
                            position_ids=getattr(lm, '_position_ids', None),
                            rope_deltas=getattr(lm, '_rope_deltas', None),
                        )

        # Save state for future exact-match and visual-similarity detection
        # On visual_hit: keep existing saved state (KV is from original prefill)
        if prompt_cache_manager is not None and not visual_hit:
            prompt_cache_manager.save_state(
                original_input_ids, y, logprobs, prompt_cache, pixel_values, kwargs=kwargs
            )
            prompt_cache_manager.save_embeds(full_inputs_embeds, original_input_ids)

        # StreamMem: register frame + evict if over budget
        if (
            prompt_cache_manager is not None
            and prompt_cache_manager.streaming_memory is not None
            and not cache_hit
            and not visual_hit
        ):
            import numpy as _np_sm

            sm = prompt_cache_manager.streaming_memory

            # Count visual tokens (model-agnostic: check both _id and _index)
            img_id = getattr(model.config, 'image_token_id', None)
            if img_id is None:
                img_id = getattr(model.config, 'image_token_index', None)
            vid_id = getattr(model.config, 'video_token_id', None)
            if vid_id is None:
                vid_id = getattr(model.config, 'video_token_index', None)

            ids_np = _np_sm.array(original_input_ids[0])
            n_visual = 0
            if img_id is not None:
                n_visual += int((ids_np == img_id).sum())
            if vid_id is not None:
                n_visual += int((ids_np == vid_id).sum())

            # Text prefix length = position of first visual token
            vis_boundary = _find_visual_boundary(original_input_ids, model)

            sm.append_frame(n_visual, text_prefix_len=vis_boundary)

            # Proxy query: last 8 tokens (chat template end tokens)
            proxy_ids = original_input_ids[:, -8:]

            stats = sm.maybe_evict(prompt_cache, model.language_model, proxy_ids)
            if stats:
                logger.info(
                    "StreamMem: %d→%d tokens (%d prototypes)",
                    stats.tokens_before, stats.tokens_after, stats.n_prototypes,
                )
                # Update prefill_offset for KVCache layers
                for c in prompt_cache:
                    if hasattr(c, 'offset'):
                        prompt_cache_manager._prefill_offset = c.offset
                        break

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
