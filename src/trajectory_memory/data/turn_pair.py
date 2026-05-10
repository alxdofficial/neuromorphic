"""TurnPair extraction — convert multi-turn sessions into flat (prior, response) pairs.

Per plan §4.8, we flatten chat sessions of N assistant turns into N
training examples. The k-th example has `prior` = everything up to and
including the user/tool turn before assistant turn k, and `response` =
the k-th assistant turn's tokens.

This sidesteps two issues with native multi-turn rollout: irregular
batching shapes and turn-to-turn dependency in sampling. Each TurnPair
samples a single response from a clean ground-truth prior.

The session reset semantic (concept_states reset per TurnPair example)
is the trade-off: cross-TurnPair memory accumulation isn't trained
directly, only within a TurnPair. See plan §4.8 "TurnPair flattening —
rationale and tradeoff."
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import PreTrainedTokenizerBase

from src.trajectory_memory.data.tokenizer import get_tokenizer


@dataclass
class TurnPair:
    """One training example: a fixed `prior` + a sampleable `response`."""

    prior_ids: list[int]      # All tokens up through and including the last
                              # user/system/tool turn before the response.
    response_ids: list[int]   # The assistant turn's tokens. The model
                              # samples (Wave 3/4) or teacher-forces
                              # (Wave 2) over these.
    source: str               # "wildchat" / "ultrachat" / "agentinstruct" / etc.
    session_id: str | None = None    # original session identifier (for telemetry)
    turn_index: int | None = None    # 0-based assistant turn index in session


def session_to_turn_pairs(
    messages: list[dict],
    *,
    tokenizer: PreTrainedTokenizerBase | None = None,
    source: str = "unknown",
    session_id: str | None = None,
    min_prior_tokens: int = 0,
    max_response_tokens: int | None = None,
) -> list[TurnPair]:
    """Flatten a multi-turn session into TurnPairs.

    Args:
        messages: list of {"role": str, "content": str} in chronological order.
        tokenizer: project tokenizer; defaults to Llama-3.2.
        source: data-source tag for the resulting TurnPairs.
        session_id: optional identifier carried through to TurnPair.
        min_prior_tokens: drop TurnPairs whose prior is shorter than this
                          (used in Wave 2/4 to filter for memory-stress
                          examples; pass 4096 to match plan §4.5).
        max_response_tokens: cap response length (truncate longer turns).

    Returns:
        List of TurnPair, one per assistant turn whose prior meets the
        length filter.
    """
    tok = tokenizer or get_tokenizer()

    pairs: list[TurnPair] = []
    asst_count = 0

    # We re-run apply_chat_template on growing prefixes; this is O(N^2)
    # in turns but correctness > speed for one-time preprocessing.
    for k, msg in enumerate(messages):
        if msg["role"] != "assistant":
            continue

        # Prior = messages[0..k] up to (but NOT including) the assistant
        # turn k. Then we add the assistant header so the response is
        # what follows the header.
        prior_msgs = messages[:k]
        # S8 fix — skip assistant-first sessions (k=0 with assistant role)
        # which would call apply_chat_template([]) → IndexError. The chat
        # template needs at least one prior message to bracket the
        # assistant header.
        if not prior_msgs:
            continue
        # Use tokenize=False then encode — apply_chat_template's tokenize=True
        # path returns a BatchEncoding (not a flat list) in some transformers
        # versions, which complicates downstream typing.
        prior_text = tok.apply_chat_template(
            prior_msgs,
            add_generation_prompt=True,   # adds "<|start_header_id|>assistant<|end_header_id|>"
            tokenize=False,
        )
        prior_ids = tok.encode(prior_text, add_special_tokens=False)

        # Response = just the assistant content + the eot_id.
        response_text = msg["content"]
        response_ids = tok.encode(response_text, add_special_tokens=False)
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        response_ids = response_ids + [eot]

        if max_response_tokens is not None:
            response_ids = response_ids[:max_response_tokens]

        if len(prior_ids) < min_prior_tokens:
            asst_count += 1
            continue

        pairs.append(
            TurnPair(
                prior_ids=prior_ids,
                response_ids=response_ids,
                source=source,
                session_id=session_id,
                turn_index=asst_count,
            )
        )
        asst_count += 1

    return pairs
