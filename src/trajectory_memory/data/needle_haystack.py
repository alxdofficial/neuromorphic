"""Synthetic needle-in-haystack documents for Wave 1 memory pretraining
(plan §4.5: "Plant fact at position X, query at Y > X+2K. Forces measurable
memory contribution to NTP loss.").

Generates documents of the form:
    [filler_1] ... [needle: "The secret code is XYZ123."] ... [filler_K]
    ... [query: "What is the secret code?"] [answer: "XYZ123"]

Where:
  - The needle is planted at position X.
  - The query/answer are at position Y > X + 2K (beyond Llama's 2K cap).
  - The filler is real-world text drawn from a base corpus.

Training NTP loss on the answer-completion segment is reducible only if
memory has carried the needle past the LM's effective context cap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator

from src.trajectory_memory.data.tokenizer import get_tokenizer


# Templates for needles. Each is (statement, query, answer).
_NEEDLE_TEMPLATES = [
    (
        "The secret access code is {answer}. Remember this; it will be tested.",
        "What is the secret access code?",
        "{answer}",
    ),
    (
        "The customer's user ID is {answer}.",
        "What was the customer's user ID?",
        "{answer}",
    ),
    (
        "Note: the magic phrase is {answer}.",
        "What is the magic phrase?",
        "{answer}",
    ),
    (
        "The reference number for this order is {answer}.",
        "What is the order's reference number?",
        "{answer}",
    ),
    (
        "Important: the quarterly target value is {answer}.",
        "What is the quarterly target value?",
        "{answer}",
    ),
]


def _random_answer(rng: random.Random, kind: str = "alphanum") -> str:
    """Generate a unique tokenize-safe answer string."""
    if kind == "alphanum":
        # 6-char uppercase + digit combo: e.g., "K7T9XB"
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(rng.choice(chars) for _ in range(6))
    if kind == "phrase":
        words = ["raven", "crimson", "lattice", "tundra", "echo", "vector",
                 "harvest", "pivot", "marble", "kindle"]
        return f"{rng.choice(words)}-{rng.choice(words)}-{rng.choice(words)}"
    raise ValueError(f"unknown kind: {kind}")


@dataclass
class NeedleDoc:
    """A synthetic document with a planted needle + answer at the end."""

    text: str            # the full document text
    needle_pos_chars: int   # character offset of the needle in text
    query_pos_chars: int    # character offset of the query (= answer site)
    answer: str             # the gold answer string
    target_distance: int    # tokens between needle and query (approximate)


def _split_filler(text: str, n_chunks: int) -> list[str]:
    """Split filler text into roughly equal chunks (by character)."""
    L = len(text)
    chunk_size = max(L // n_chunks, 1)
    return [text[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]


def make_needle_doc(
    filler_text: str,
    *,
    target_distance_tokens: int,
    rng: random.Random | None = None,
    answer_kind: str = "alphanum",
) -> NeedleDoc:
    """Build one needle-in-haystack document.

    The needle is planted at the start; filler fills until the desired
    distance is reached; then the query+answer ends the document.

    Args:
        filler_text: a body of natural text to use as filler (a long doc
                     from FineWeb or PG19 works).
        target_distance_tokens: approximate token distance between needle
                                and query. >2K forces memory.
        rng: random source (default: fresh).
        answer_kind: "alphanum" or "phrase".
    """
    rng = rng or random.Random()
    tok = get_tokenizer()

    # Pick a template
    statement_tpl, query_tpl, answer_tpl = rng.choice(_NEEDLE_TEMPLATES)
    answer = _random_answer(rng, kind=answer_kind)
    statement = statement_tpl.format(answer=answer)
    query = query_tpl
    answer_text = answer_tpl.format(answer=answer)

    # Estimate filler length needed (chars) to hit target token distance.
    # Llama tokenizer has ~4 chars/token avg on English text.
    chars_per_token = 4
    filler_chars_needed = target_distance_tokens * chars_per_token
    if len(filler_text) < filler_chars_needed:
        # Repeat the filler if too short.
        repeats = (filler_chars_needed // max(len(filler_text), 1)) + 1
        filler_text = filler_text * repeats
    filler = filler_text[:filler_chars_needed]

    # Compose: [statement] [filler] [query+answer]
    document = (
        statement + "\n\n"
        + filler + "\n\n"
        + query + " " + answer_text + "\n"
    )
    needle_pos = 0
    query_pos = len(statement) + 2 + len(filler) + 2

    # Verify approximate token distance (just for the metadata).
    needle_token_len = len(tok.encode(statement, add_special_tokens=False))
    filler_token_len = len(tok.encode(filler, add_special_tokens=False))
    actual_distance = needle_token_len + filler_token_len

    return NeedleDoc(
        text=document,
        needle_pos_chars=needle_pos,
        query_pos_chars=query_pos,
        answer=answer,
        target_distance=actual_distance,
    )


def generate_needle_docs(
    fillers: Iterable[str],
    *,
    target_distances: list[int] = (3000, 8000, 16000, 32000),
    seed: int = 0,
    answer_kind: str = "alphanum",
) -> Iterator[NeedleDoc]:
    """Generate needle docs from an iterable of filler bodies.

    Cycles through `target_distances` so the dataset has a curriculum of
    increasing memory pressure.
    """
    rng = random.Random(seed)
    for i, filler in enumerate(fillers):
        target = target_distances[i % len(target_distances)]
        yield make_needle_doc(
            filler, target_distance_tokens=target, rng=rng,
            answer_kind=answer_kind,
        )
