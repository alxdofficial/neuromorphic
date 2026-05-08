"""Chat template + assistant-token masking.

Multi-turn data uses Llama's chat template tokens (`<|start_header_id|>`,
`<|end_header_id|>`, `<|eot_id|>`). For training, we want to compute NTP
loss / surprise only on assistant-generated tokens (per plan §3.3).

`build_assistant_mask` returns a per-token bool tensor: True where the
token is part of an assistant turn's text (excluding the role-header
boilerplate), False otherwise.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from src.trajectory_memory.data.tokenizer import get_tokenizer


def apply_chat_template(
    messages: list[dict],
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    add_generation_prompt: bool = False,
) -> list[int]:
    """Apply the Llama chat template and return token IDs.

    Args:
        messages: list of {"role": "system"|"user"|"assistant", "content": str}.
        tokenizer: tokenizer instance; defaults to project tokenizer.
        add_generation_prompt: if True, append the assistant header so the
                               model sees `<|start_header_id|>assistant<|end_header_id|>`
                               and is positioned to generate.

    Returns:
        List[int] of token IDs.
    """
    tok = tokenizer or get_tokenizer()
    text = tok.apply_chat_template(
        messages,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )
    return tok.encode(text, add_special_tokens=False)


def build_assistant_mask(
    input_ids: list[int] | Tensor,
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
) -> Tensor:
    """Build a [T] bool tensor: True where the token is part of an
    assistant message's content (NOT the role-header bracket tokens).

    Algorithm: scan for the assistant-header pattern
    `<|start_header_id|> assistant <|end_header_id|>\\n\\n`, then mark
    tokens until the next `<|eot_id|>` as True.

    This is robust to Llama-3 chat template variants — the role marker
    tokens are stable across versions.
    """
    tok = tokenizer or get_tokenizer()
    if isinstance(input_ids, Tensor):
        ids = input_ids.tolist()
    else:
        ids = list(input_ids)

    # Special token IDs (Llama-3 / 3.1 / 3.2 family).
    start_header = tok.convert_tokens_to_ids("<|start_header_id|>")
    end_header = tok.convert_tokens_to_ids("<|end_header_id|>")
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    assistant_id = tok.convert_tokens_to_ids("assistant")

    mask = [False] * len(ids)
    in_assistant = False
    i = 0
    while i < len(ids):
        if not in_assistant:
            # Look for "<|start_header_id|> assistant <|end_header_id|>"
            if (
                i + 2 < len(ids)
                and ids[i] == start_header
                and ids[i + 1] == assistant_id
                and ids[i + 2] == end_header
            ):
                # Skip header (3 tokens) + the "\n\n" newlines after.
                # Llama-3 chat template typically has 1-2 newline tokens
                # after the end_header. Find the first non-newline token
                # to start the assistant span.
                j = i + 3
                # Skip forward through whitespace tokens until we hit content
                # or eot_id. The assistant span starts at the first content token.
                while j < len(ids) and ids[j] not in (eot, start_header):
                    # Anything that's not EOT or another header start is content
                    mask[j] = True
                    j += 1
                in_assistant = False  # we've already filled this assistant span
                i = j
                continue
        i += 1
    return torch.tensor(mask, dtype=torch.bool)
