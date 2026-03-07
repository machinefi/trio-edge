"""Base classes and utilities for native model definitions.

Replaces mlx_vlm.models.base and mlx_vlm.models.cache imports.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn


# ── Config base ──────────────────────────────────────────────────────────────


@dataclass
class BaseModelConfig:
    @classmethod
    def from_dict(cls, params):
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ── Model output types ──────────────────────────────────────────────────────


@dataclass
class LanguageModelOutput:
    logits: mx.array
    hidden_states: Optional[List[mx.array]] = None
    cross_attention_states: Optional[List[mx.array]] = None
    encoder_outputs: Optional[List[mx.array]] = None


@dataclass
class InputEmbeddingsFeatures:
    inputs_embeds: mx.array
    attention_mask_4d: Optional[mx.array] = None
    visual_pos_masks: Optional[mx.array] = None
    deepstack_visual_embeds: Optional[mx.array] = None
    per_layer_inputs: Optional[mx.array] = None
    cross_attention_states: Optional[mx.array] = None
    cross_attention_mask: Optional[mx.array] = None
    full_text_row_masked_out_mask: Optional[mx.array] = None
    decoder_inputs_embeds: Optional[mx.array] = None
    attention_mask: Optional[mx.array] = None

    def to_dict(self):
        return {
            "inputs_embeds": self.inputs_embeds,
            "attention_mask_4d": self.attention_mask_4d,
            "visual_pos_masks": self.visual_pos_masks,
            "deepstack_visual_embeds": self.deepstack_visual_embeds,
            "per_layer_inputs": self.per_layer_inputs,
            "cross_attention_states": self.cross_attention_states,
            "cross_attention_mask": self.cross_attention_mask,
            "full_text_row_masked_out_mask": self.full_text_row_masked_out_mask,
            "decoder_inputs_embeds": self.decoder_inputs_embeds,
            "attention_mask": self.attention_mask,
        }


# ── Attention utilities ──────────────────────────────────────────────────────


def create_causal_mask(
    N: int,
    offset: int = 0,
    window_size: Optional[int] = None,
):
    rinds = mx.arange(offset + N)
    linds = mx.arange(offset, offset + N) if offset else rinds
    linds = linds[:, None]
    rinds = rinds[None]
    mask = linds >= rinds
    if window_size is not None:
        mask = mask & (linds < rinds + window_size)
    return mask


def create_attention_mask(
    h, cache=None, window_size: Optional[int] = None, return_array: bool = False
):
    N = h.shape[1]
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(N, return_array=return_array, window_size=window_size)
    if N == 1:
        return None
    if return_array or (window_size and N > window_size):
        return create_causal_mask(N, window_size=window_size)
    return "causal"


def create_ssm_mask(h, cache=None):
    if cache and hasattr(cache, "make_mask"):
        return cache.make_mask(h.shape[1])
    return None


def scaled_dot_product_attention(
    queries,
    keys,
    values,
    cache,
    scale: float,
    mask: Optional[mx.array],
    sinks: Optional[mx.array] = None,
) -> mx.array:
    if hasattr(cache, "bits"):
        return nn.quantized_scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            group_size=cache.group_size,
            bits=cache.bits,
        )
    else:
        return mx.fast.scaled_dot_product_attention(
            queries,
            keys,
            values,
            scale=scale,
            mask=mask,
            sinks=sinks,
        )


# ── KV Cache ─────────────────────────────────────────────────────────────────


class KVCache:
    step = 256

    def __init__(self):
        self.keys = None
        self.values = None
        self.offset = 0

    def update_and_fetch(self, keys, values):
        prev = self.offset
        if self.keys is None or (prev + keys.shape[2]) > self.keys.shape[2]:
            B, n_kv_heads, _, k_head_dim = keys.shape
            v_head_dim = values.shape[3]
            n_steps = (self.step + keys.shape[2] - 1) // self.step
            k_shape = (B, n_kv_heads, n_steps * self.step, k_head_dim)
            v_shape = (B, n_kv_heads, n_steps * self.step, v_head_dim)
            new_k = mx.zeros(k_shape, keys.dtype)
            new_v = mx.zeros(v_shape, values.dtype)
            if self.keys is not None:
                if prev % self.step != 0:
                    self.keys = self.keys[..., :prev, :]
                    self.values = self.values[..., :prev, :]
                self.keys = mx.concatenate([self.keys, new_k], axis=2)
                self.values = mx.concatenate([self.values, new_v], axis=2)
            else:
                self.keys, self.values = new_k, new_v

        self.offset += keys.shape[2]
        self.keys[..., prev : self.offset, :] = keys
        self.values[..., prev : self.offset, :] = values
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]

    def size(self):
        return self.offset

    @property
    def state(self):
        if self.offset == self.keys.shape[2]:
            return self.keys, self.values
        else:
            return (
                self.keys[..., : self.offset, :],
                self.values[..., : self.offset, :],
            )

    @state.setter
    def state(self, v):
        self.keys, self.values = v
        self.offset = self.keys.shape[2]


# ── Arrays Cache (for SSM/DeltaNet layers) ─────────────────────────────────


class ArraysCache:
    def __init__(self, size, left_padding: Optional[List[int]] = None):
        self.cache = [None] * size
        self.left_padding = mx.array(left_padding) if left_padding else None
        self.lengths = None

    def __setitem__(self, idx, value):
        self.cache[idx] = value

    def __getitem__(self, idx):
        return self.cache[idx]

    @property
    def state(self):
        return self.cache

    @state.setter
    def state(self, v):
        self.cache = v

    def make_mask(self, N: int):
        if self.left_padding is not None:
            pos = mx.arange(N)
            return pos >= self.left_padding[:, None]
        elif self.lengths is not None:
            pos = mx.arange(N)
            return pos < self.lengths[:, None]
        else:
            return None

    def empty(self):
        return self.cache[0] is None
