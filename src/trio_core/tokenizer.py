"""Lightweight tokenizer — replaces transformers.AutoTokenizer.

Uses the `tokenizers` library (Rust-based, 8MB) + jinja2 for chat templates,
avoiding the full `transformers` package (47MB + torch transitive deps).

Supports:
- encode / decode / __call__
- convert_tokens_to_ids
- apply_chat_template (Jinja2-based)
- eos_token / eos_token_id / pad_token_id
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Union

from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Default ChatML template (used by Qwen2.5/3/3.5 when not in tokenizer_config.json)
_DEFAULT_CHATML_TEMPLATE = """\
{%- for message in messages %}
{{- '<|im_start|>' + message['role'] + '\n' }}
{%- if message['content'] is string %}
{{- message['content'] }}
{%- else %}
{%- for block in message['content'] %}
{%- if block['type'] == 'text' %}
{{- block['text'] }}
{%- endif %}
{%- endfor %}
{%- endif %}
{{- '<|im_end|>\n' }}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\n' }}
{%- endif %}"""


class ChatTokenizer:
    """Lightweight tokenizer wrapping HuggingFace tokenizers + Jinja2 chat template."""

    def __init__(self, model_path: Union[str, Path]):
        model_path = Path(model_path)

        # Load the fast tokenizer
        tokenizer_file = model_path / "tokenizer.json"
        if not tokenizer_file.exists():
            raise FileNotFoundError(f"tokenizer.json not found in {model_path}")
        self._tokenizer = Tokenizer.from_file(str(tokenizer_file))

        # Load tokenizer config
        config_file = model_path / "tokenizer_config.json"
        self._config: dict[str, Any] = {}
        if config_file.exists():
            with open(config_file) as f:
                self._config = json.load(f)

        # Build token -> id map for special tokens
        self._special_token_map: dict[str, int] = {}
        for id_str, info in self._config.get("added_tokens_decoder", {}).items():
            self._special_token_map[info["content"]] = int(id_str)

        # EOS / PAD tokens
        self.eos_token = self._config.get("eos_token")
        self.eos_token_id = self._special_token_map.get(self.eos_token) if self.eos_token else None
        pad_token = self._config.get("pad_token")
        self.pad_token_id = (
            self._special_token_map.get(pad_token) if pad_token else self.eos_token_id
        )

        # Chat template (Jinja2) — fallback to default ChatML for Qwen models
        self._chat_template_str = self._config.get("chat_template") or _DEFAULT_CHATML_TEMPLATE
        self._chat_template = None

    def _get_chat_template(self):
        if self._chat_template is None and self._chat_template_str:
            import jinja2

            env = jinja2.Environment(
                undefined=jinja2.StrictUndefined,
                keep_trailing_newline=True,
            )
            # Add tojson filter
            env.filters["tojson"] = lambda x, **kw: json.dumps(x, ensure_ascii=False)
            self._chat_template = env.from_string(self._chat_template_str)
        return self._chat_template

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        encoding = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
        return encoding.ids

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        # Check special tokens first
        if token in self._special_token_map:
            return self._special_token_map[token]
        # Fall back to vocab lookup
        token_id = self._tokenizer.token_to_id(token)
        return token_id

    def __call__(
        self,
        text: str | list[str],
        padding: bool | str = False,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> dict[str, list]:
        """Tokenize text, returning dict with input_ids and optional attention_mask."""
        if isinstance(text, str):
            text = [text]

        all_ids = []
        for t in text:
            enc = self._tokenizer.encode(t, add_special_tokens=True)
            all_ids.append(enc.ids)

        result: dict[str, Any] = {"input_ids": all_ids if len(all_ids) > 1 else all_ids}

        if padding and len(all_ids) > 1:
            max_len = max(len(ids) for ids in all_ids)
            pad_id = self.pad_token_id or 0
            padded_ids = []
            masks = []
            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded_ids.append(ids + [pad_id] * pad_len)
                masks.append([1] * len(ids) + [0] * pad_len)
            result["input_ids"] = padded_ids
            if return_attention_mask:
                result["attention_mask"] = masks
        elif return_attention_mask:
            result["attention_mask"] = [[1] * len(ids) for ids in all_ids]

        return result

    def apply_chat_template(
        self,
        messages: list[dict],
        tokenize: bool = True,
        add_generation_prompt: bool = True,
        **kwargs,
    ) -> str | list[int]:
        """Render chat template using Jinja2."""
        template = self._get_chat_template()
        if template is None:
            raise ValueError("No chat_template found in tokenizer_config.json")

        # Render with standard variables that Jinja templates expect
        context = {
            "messages": messages,
            "add_generation_prompt": add_generation_prompt,
            "bos_token": self._config.get("bos_token", ""),
            "eos_token": self.eos_token or "",
            "tools": None,
            **kwargs,
        }
        rendered = template.render(**context)
        # Jinja2 may add an extra trailing newline beyond what the template produces;
        # normalize to exactly one trailing newline (matching transformers behavior)
        rendered = rendered.rstrip("\n") + "\n"

        if tokenize:
            return self.encode(rendered, add_special_tokens=False)
        return rendered


def load_tokenizer(model_path: str | Path) -> ChatTokenizer:
    """Load a ChatTokenizer from a local model directory."""
    return ChatTokenizer(model_path)
