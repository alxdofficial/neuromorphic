#!/usr/bin/env python3
"""Generate multi-needle passphrase-retrieval training data as jsonl.

Each line of the output jsonl has:
    {
        "prompt_ids":    [int, ...],   # tokens the model sees
        "prompt_text":   str,          # human-readable (lossy if tokenizer
                                        # reinserts whitespace differently)
        "answer_ids":    [int, ...],   # ground-truth continuation tokens
        "answer_text":   str,          # "<p1>. <p2>. ... <pN>."
        "passphrases":   [str, ...],   # ordered phrases — used by reward_fn
        "n_phrases":     int,
        "target_length": int,          # requested prompt_ids length
        "actual_length": int,          # measured prompt_ids length after tokenize
    }

`prompt_ids` is built by tokenizing the full assembled prompt once, after
sizing filler per slot to hit `target_length` with the fewest-token filler
slice that does so. Measured `actual_length` is usually within ±2 tokens
of `target_length` due to BPE boundary effects.

Usage:
    python scripts/build_passphrase_data.py \\
        --tokenizer meta-llama/Llama-3.2-3B \\
        --num-samples 2000 \\
        --n-phrases 2,3,5 \\
        --target-lengths 512,1024 \\
        --out data/passphrase/train.jsonl

Determinism: `--seed` fixes phrase sampling and (n_phrases, target_length)
draw for each example.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from src.pretrained.passphrase_data import (
    FILLER_PARAGRAPH,
    PHRASE_POOL,
    build_answer,
    build_prompt_parts,
    sample_phrases,
)


def _tokenize_piece(tokenizer, text: str) -> list[int]:
    """Raw BPE token ids for a piece, no bos/eos added. Pieces are
    composed at generate-time so we must not inject specials per-piece."""
    return tokenizer.encode(text, add_special_tokens=False)


def _build_filler_token_bank(tokenizer, min_tokens: int) -> list[int]:
    """Tokenize enough repetitions of FILLER_PARAGRAPH to cover `min_tokens`
    of filler budget. Returned once per example — we slice from it per slot."""
    bank: list[int] = []
    reps = 0
    while len(bank) < min_tokens + 64:  # +64 slack to survive boundary rounding
        reps += 1
        bank = _tokenize_piece(tokenizer, (FILLER_PARAGRAPH + " ") * reps)
        if reps > 200:
            raise RuntimeError(
                f"FILLER_PARAGRAPH too short to fill {min_tokens} tokens; "
                f"increase paragraph length or lower target_length.")
    return bank


def build_example(
    tokenizer,
    rng: random.Random,
    *,
    n_phrases: int,
    target_length: int,
) -> dict:
    """Build one (prompt_ids, answer_ids, passphrases) example.

    Fills N+1 filler slots with roughly equal token budgets to hit
    target_length. If instruction + phrase-prefixes + phrases alone exceed
    target_length, filler budgets collapse to zero and actual_length
    overshoots.
    """
    phrases = sample_phrases(rng, n_phrases, pool=PHRASE_POOL)
    # Assemble with empty filler first to measure the structural token cost.
    parts_empty = build_prompt_parts(phrases, filler_text_per_gap=[""] * (n_phrases + 1))
    structural_ids = _tokenize_piece(tokenizer, parts_empty.assemble())
    structural_len = len(structural_ids)

    filler_budget = max(0, target_length - structural_len)
    n_slots = n_phrases + 1
    per_slot = filler_budget // n_slots
    remainder = filler_budget - per_slot * n_slots  # distribute leftover tokens

    bank = _build_filler_token_bank(tokenizer, min_tokens=filler_budget) if filler_budget > 0 else []

    # Slice `bank` into N+1 chunks of per_slot tokens (plus leftover on first
    # chunks). Decode each chunk to text and plug into the prompt.
    chunk_sizes = [per_slot + (1 if i < remainder else 0) for i in range(n_slots)]
    filler_texts: list[str] = []
    cursor = 0
    for size in chunk_sizes:
        chunk_ids = bank[cursor:cursor + size]
        cursor += size
        # `skip_special_tokens=True` is a no-op here (filler has none) but
        # defensive. clean_up_tokenization_spaces keeps whitespace honest.
        filler_texts.append(
            tokenizer.decode(chunk_ids, skip_special_tokens=True).strip())

    parts = build_prompt_parts(phrases, filler_text_per_gap=filler_texts)
    prompt_text = parts.assemble()
    prompt_ids = _tokenize_piece(tokenizer, prompt_text)

    answer_text = build_answer(phrases)
    answer_ids = _tokenize_piece(tokenizer, answer_text)

    return {
        "prompt_ids": prompt_ids,
        "prompt_text": prompt_text,
        "answer_ids": answer_ids,
        "answer_text": answer_text,
        "passphrases": phrases,
        "n_phrases": n_phrases,
        "target_length": target_length,
        "actual_length": len(prompt_ids),
    }


def _parse_int_list(s: str) -> list[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.2-3B",
                        help="HF tokenizer id or local path")
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--n-phrases", type=str, default="3",
                        help="Comma-separated list of phrase counts (e.g., '2,3,5')")
    parser.add_argument("--target-lengths", type=str, default="512",
                        help="Comma-separated list of target prompt token lengths")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, required=True,
                        help="Output jsonl path")
    args = parser.parse_args()

    n_phrase_options = _parse_int_list(args.n_phrases)
    length_options = _parse_int_list(args.target_lengths)
    assert n_phrase_options, "--n-phrases must list at least one count"
    assert length_options, "--target-lengths must list at least one length"
    assert max(n_phrase_options) <= len(PHRASE_POOL), (
        f"PHRASE_POOL has {len(PHRASE_POOL)} phrases; "
        f"cannot generate n_phrases={max(n_phrase_options)} without repetition")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"  vocab size: {len(tokenizer)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    actual_lengths: list[int] = []
    n_counts: dict[int, int] = {n: 0 for n in n_phrase_options}

    with args.out.open("w") as f:
        for i in range(args.num_samples):
            n = rng.choice(n_phrase_options)
            target = rng.choice(length_options)
            ex = build_example(tokenizer, rng, n_phrases=n, target_length=target)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            actual_lengths.append(ex["actual_length"])
            n_counts[n] += 1
            if (i + 1) % 200 == 0:
                print(f"  [{i+1:>6}/{args.num_samples}] "
                      f"actual_len mean={sum(actual_lengths)/len(actual_lengths):.1f}")

    al = actual_lengths
    print(f"\nDone: {args.num_samples} samples -> {args.out}")
    print(f"  actual_length: min={min(al)} mean={sum(al)/len(al):.1f} max={max(al)}")
    print(f"  n_phrases distribution: {n_counts}")


if __name__ == "__main__":
    main()
