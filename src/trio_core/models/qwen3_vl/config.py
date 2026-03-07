"""Qwen3-VL model configuration."""

import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..base import BaseModelConfig


@dataclass
class VisionConfig(BaseModelConfig):
    model_type: str = "qwen3_vl"
    depth: int = 32
    hidden_size: int = 1280
    intermediate_size: int = 3420
    out_hidden_size: int = 1536
    num_heads: int = 16
    image_size: int = 384
    patch_size: int = 14
    vocab_size: int = 32000
    mlp_ratio: float = 4.0
    in_channels: int = 3
    layer_norm_eps: float = 1e-6
    spatial_patch_size: int = 14
    spatial_merge_size: int = 2
    tokens_per_second: int = 2
    temporal_patch_size: int = 2
    num_position_embeddings: int = 5184
    deepstack_visual_indexes: list[int] = field(default_factory=list)


@dataclass
class TextConfig(BaseModelConfig):
    model_type: str = ""
    hidden_size: int = 2048
    num_hidden_layers: int = 36
    intermediate_size: int = 11008
    num_attention_heads: int = 16
    head_dim: Optional[int] = None
    rms_norm_eps: float = 1e-6
    vocab_size: int = 151936
    num_key_value_heads: Optional[int] = None
    max_position_embeddings: Optional[int] = 128000
    rope_theta: float = 1000000.0
    rope_traditional: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = True
    attention_bias: bool = False

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads

        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads

        if self.rope_scaling:
            if "mrope_section" not in self.rope_scaling:
                raise ValueError("rope_scaling must contain 'mrope_section'")
            # Normalize: accept both 'type' and 'rope_type'
            if "rope_type" in self.rope_scaling and "type" not in self.rope_scaling:
                self.rope_scaling["type"] = self.rope_scaling["rope_type"]


@dataclass
class ModelConfig(BaseModelConfig):
    text_config: TextConfig = None
    vision_config: VisionConfig = None
    model_type: str = "qwen3_vl"
    ignore_index: int = -100
    image_token_id: int = 151655
    video_token_id: int = 151656
    image_token_index: int = 151655
    video_token_index: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    vision_token_id: int = 151654
    vision_feature_select_strategy: str = "default"
    vision_feature_layer: int = -2
    vocab_size: int = 32000
    eos_token_id: Optional[List[int]] = None

    def __post_init__(self):
        # Alias: image_token_index -> image_token_id
        if hasattr(self, "image_token_index") and self.image_token_index is not None:
            self.image_token_id = self.image_token_index
        if hasattr(self, "video_token_index") and self.video_token_index is not None:
            self.video_token_id = self.video_token_index

    @classmethod
    def from_dict(cls, params):
        # Don't mutate the original dict
        params = dict(params)
        excluded_keys = {"vision_config", "text_config"}
        params["text_config"] = dict(
            filter(lambda x: x[0] not in excluded_keys, params.items())
        )
        return cls(
            **{
                k: v
                for k, v in params.items()
                if k in inspect.signature(cls).parameters
            }
        )
