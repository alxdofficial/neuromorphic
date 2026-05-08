"""Llama-3.2 tokenizer — single source of truth for the project."""

from __future__ import annotations

from functools import lru_cache

from transformers import AutoTokenizer, PreTrainedTokenizerBase


DEFAULT_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"


# Llama-3 chat template — base 3.2 models ship without a chat template,
# but the Instruct variants use this exact format. Set it here so plain
# 3.2 base tokenizer can do chat formatting consistent with the Instruct
# convention.
_LLAMA3_CHAT_TEMPLATE = (
    "{% set loop_messages = messages %}"
    "{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n' + message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.first %}{% set content = bos_token + content %}{% endif %}"
    "{{ content }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}"
    "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
    "{% endif %}"
)


@lru_cache(maxsize=4)
def get_tokenizer(
    name: str = DEFAULT_TOKENIZER_NAME,
    *,
    add_bos: bool = True,
) -> PreTrainedTokenizerBase:
    """Load and cache the Llama-3.2 tokenizer.

    Args:
        name: HF model id. Default Llama-3.2-1B (uses the same tokenizer
              as 3B/8B/70B in the Llama-3 family).
        add_bos: whether to add a BOS token by default (consistent with
                 Llama's standard chat template behavior).

    Returns:
        PreTrainedTokenizerBase. Use `.encode(text)` for plain tokenization,
        or `.apply_chat_template(messages)` for chat formatting (template
        is auto-set if missing — Llama-3 base models ship without one).
    """
    tok = AutoTokenizer.from_pretrained(name)
    # Llama tokenizer doesn't have a pad token by default — assign EOS.
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.add_bos_token = add_bos
    # Set the standard Llama-3 chat template if missing (base 3.2-1B
    # doesn't have one; Instruct variants do).
    if tok.chat_template is None:
        tok.chat_template = _LLAMA3_CHAT_TEMPLATE
    return tok


def vocab_size() -> int:
    """Return the vocab size of the project tokenizer."""
    return len(get_tokenizer())
