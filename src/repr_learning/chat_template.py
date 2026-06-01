"""Backbone-agnostic chat-template scaffold for memory-injection LMs.

Pre-tokenizes the chat structure once at model init time, so forward
and decode can splice memory embeddings + question/answer without
re-tokenizing per step.

Works with any HuggingFace tokenizer that implements `apply_chat_template`
(Llama-3, Qwen-2.5, Mistral-v3+, Phi-3, Gemma, etc.). To swap backbones,
change `cfg.llama_model` — this module figures out the right scaffold
tokens automatically.

Slot layout (the model sees this concatenated embedding sequence):

  [pre_memory_ids]   ← <|begin|><|sys_header|>{system_intro}
  [memory embeds]    ← M memory tokens (model-specific d_hidden)
  [post_memory_ids]  ← <|eot|><|user_header|>
  [question tokens]
  [post_question_ids]← <|eot|><|asst_header|>
  [answer tokens + eot_id at end]

For decode-time generation, drop the trailing answer part and use the
prefix [pre_memory; memory; post_memory; question; post_question] as
inputs_embeds.

Reference: Llama-3 chat template renders as
  <|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n
  {system_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n
  {user_msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n
  {asst_msg}<|eot_id|>
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

DEFAULT_SYSTEM_INTRO = (
    "You are a helpful assistant. The following text contains memories "
    "from a long document. Use only those memories to answer the user's question."
)

# Pinned date for chat-template rendering. Llama-3 (and similar) chat
# templates inject `Today Date: {date}` via the tokenizer; without pinning,
# the scaffold tokens drift day-to-day, invalidating train/eval consistency.
# Set to the date this protocol was finalized (tranche-4 launch: 2026-05-28).
# To rebuild against a different date for a new tranche, override at call site.
PINNED_DATE_STRING = "28 May 2026"


@dataclass
class ChatTemplate:
    """Pre-tokenized chat scaffold around a memory + Q + A triple.

    All tensors are 1D `[L]` token-id sequences, dtype=long, on CPU.
    """
    pre_memory_ids:    torch.Tensor   # <|begin|>{sys_header}{system_intro}
    post_memory_ids:   torch.Tensor   # <|eot|>{user_header}
    post_question_ids: torch.Tensor   # <|eot|>{asst_header}
    eot_id:            int            # appended to answer; used as stop token at decode
    system_intro:      str            # for logging/debugging
    date_string:       str = ""       # date pinned into the scaffold (Llama-3+ templates)

    @property
    def scaffold_token_count(self) -> int:
        return (len(self.pre_memory_ids) + len(self.post_memory_ids)
                + len(self.post_question_ids))

    def summary(self) -> str:
        return (
            f"ChatTemplate(pre={len(self.pre_memory_ids)}, "
            f"post_mem={len(self.post_memory_ids)}, "
            f"post_q={len(self.post_question_ids)}, eot={self.eot_id}, "
            f"scaffold_tokens={self.scaffold_token_count}, "
            f"date_string={self.date_string!r})"
        )


# Placeholders chosen to NOT collide with any natural language and to be
# stable under all HF tokenizers — pure ASCII, no special-token characters.
_MEM_PLACEHOLDER = "MMMMMMMEMORYMMMMMMM"
_Q_PLACEHOLDER   = "QQQQQQQUESTIONQQQQQQQ"
_A_PLACEHOLDER   = "AAAAAAAANSWERAAAAAAAA"


def build_chat_template(
    tokenizer,
    system_intro: str = DEFAULT_SYSTEM_INTRO,
    date_string: Optional[str] = None,
) -> ChatTemplate:
    """Render the tokenizer's chat template with placeholders, then split
    by placeholder substrings to extract the scaffold token spans.

    `date_string` is pinned (default PINNED_DATE_STRING) so Llama-3-class
    templates don't inject today's auto-date — without pinning, scaffold
    tokens drift day-to-day and train/eval consistency breaks.

    Robust to per-tokenizer template variations because we delegate
    rendering to `apply_chat_template` and only assume the placeholders
    survive as substrings.
    """
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            f"Tokenizer {type(tokenizer).__name__} has no apply_chat_template "
            "method — this backbone is not chat-formatted."
        )

    if date_string is None:
        date_string = PINNED_DATE_STRING

    # Render with placeholders embedded in each role's content.
    # date_string kwarg is honored by Llama-3+ templates that inject
    # `Today Date:` — silently ignored by templates that don't reference it.
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "system",    "content": f"{system_intro} {_MEM_PLACEHOLDER}"},
            {"role": "user",      "content": _Q_PLACEHOLDER},
            {"role": "assistant", "content": _A_PLACEHOLDER},
        ],
        tokenize=False,
        add_generation_prompt=False,
        date_string=date_string,
    )

    # Split at placeholders → 7 parts: [pre_mem, MEM, post_mem, Q, post_q, A, post_a]
    parts = re.split(
        f"({re.escape(_MEM_PLACEHOLDER)}|{re.escape(_Q_PLACEHOLDER)}|{re.escape(_A_PLACEHOLDER)})",
        full_text,
    )
    if len(parts) != 7:
        raise RuntimeError(
            f"chat-template split produced {len(parts)} parts (expected 7). "
            f"Tokenizer {type(tokenizer).__name__} may have a non-standard "
            f"template. Rendered text starts: {full_text[:200]!r}"
        )
    (pre_memory_text, mem_marker, post_memory_text,
     q_marker, post_question_text, a_marker, post_answer_text) = parts
    assert mem_marker == _MEM_PLACEHOLDER
    assert q_marker == _Q_PLACEHOLDER
    assert a_marker == _A_PLACEHOLDER

    def _encode(text: str) -> torch.Tensor:
        if not text:
            return torch.empty(0, dtype=torch.long)
        # add_special_tokens=False because apply_chat_template ALREADY
        # injected special tokens (<|begin|>, headers, <|eot|>); we just
        # need to re-tokenize them as-is. HF tokenizers recognize special-
        # token literals like "<|begin_of_text|>" as the corresponding id.
        ids = tokenizer(text, add_special_tokens=False,
                        return_tensors="pt")["input_ids"][0]
        return ids

    pre_memory_ids    = _encode(pre_memory_text)
    post_memory_ids   = _encode(post_memory_text)
    post_question_ids = _encode(post_question_text)
    post_answer_ids   = _encode(post_answer_text)

    # eot = the assistant-turn-end token, which is the last token after the
    # placeholder-A. For Llama-3 that's <|eot_id|>=128009; for Qwen that's
    # <|im_end|>; etc. Fall back to tokenizer.eos_token_id only if the
    # template doesn't append anything after the assistant content.
    if len(post_answer_ids) > 0:
        eot_id = int(post_answer_ids[-1].item())
    else:
        eot_id = int(tokenizer.eos_token_id)

    return ChatTemplate(
        pre_memory_ids=pre_memory_ids,
        post_memory_ids=post_memory_ids,
        post_question_ids=post_question_ids,
        eot_id=eot_id,
        system_intro=system_intro,
        date_string=date_string,
    )


def maybe_append_eot(answer_ids: torch.Tensor, eot_id: int) -> torch.Tensor:
    """Append eot_id to a 1D answer-token tensor if not already present.
    Returns a NEW tensor (input unchanged)."""
    if answer_ids.numel() > 0 and int(answer_ids[-1].item()) == eot_id:
        return answer_ids
    return torch.cat([answer_ids, torch.tensor([eot_id], dtype=answer_ids.dtype)])
