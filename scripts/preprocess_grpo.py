"""Preprocess Wave 3 GRPO datasets (GSM8K, NumninaMath, HumanEval, NarrativeQA)
into a unified prompt/response parquet.

Output: parquet with columns:
    - prompt_ids:    List[int]   tokenized prompt
    - gold_ids:      List[int]   tokenized gold answer (for reward + reference)
    - num_prompt:    int
    - num_gold:      int
    - source:        str         "gsm8k" / "narrativeqa" / etc.
    - reward_kind:   str         "exact_match" / "bert_cosine" / "rule_based"
    - meta_json:     str         per-example metadata (e.g., gold_number for math)

The trainer reads this file, samples J responses per prompt via the
trajectory-memory model, scores against gold (rule-based or
exact_match + BERT cosine per the project's no-LLM-judge policy), and
runs GRPO advantage policy gradient.

Usage:
    python scripts/preprocess_grpo.py gsm8k \\
        --output data/wave3/gsm8k.parquet --max-examples 1000
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from src.trajectory_memory.data.tokenizer import get_tokenizer


# source → preprocessor function (returns (prompt_text, gold_text, reward_kind, meta))
def _gsm8k_extract(ex):
    # gsm8k format: {"question", "answer": "...####N"}
    # Verified 100% of train (7473) + test (1319) answers contain `####`.
    # Final answers are integers per GSM8K spec, but the regex tolerates
    # commas and decimals defensively.
    answer_text = ex["answer"]
    match = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", answer_text)
    gold_num = match.group(1).replace(",", "") if match else None
    prompt = (
        "Solve this math problem step by step, then give the final answer.\n\n"
        f"Problem: {ex['question']}\n\nSolution:"
    )
    return prompt, answer_text, "exact_match", {"gold_number": gold_num}


def _narrativeqa_extract(
    ex,
    *,
    tokenizer=None,
    max_prompt_tokens: int = 8192,
    use_summary: bool = False,
):
    """Extract a NarrativeQA prompt + gold answers.

    NarrativeQA's full `document.text` is huge (~700K chars / ~150K tokens).
    `document.summary.text` is much shorter (~5K chars / ~1K tokens) and
    fits in the LM's 2K context cap — defeats the memory-stress purpose.

    We slice the FULL document and truncate **at token level** so the
    final tokenized prompt fits inside `max_prompt_tokens` while preserving
    the trailing `Question: ... Answer:` markers. Earlier versions used a
    fixed `passage_chars=32000` and let the caller truncate via
    `prompt_ids[:max_prompt_tokens]`, which dropped the question on
    ~26% of examples (the truncation cuts from the END). This version
    fixes that.

    Pass `use_summary=True` to fall back to the short summary (faster but
    less memory-stress).
    """
    doc = ex["document"]
    full_passage = (
        doc.get("summary", {}).get("text", "")
        if use_summary else doc.get("text", "")
    )
    if not full_passage:
        return None
    question = ex["question"]["text"]
    gold_answers = [a["text"] for a in ex.get("answers", []) if a.get("text")]
    if not gold_answers:
        return None
    gold = gold_answers[0]

    intro = "Read the following passage and answer the question.\n\nPassage:\n"
    suffix = f"\n\nQuestion: {question}\n\nAnswer:"

    if tokenizer is None:
        # Fallback path (no tokenizer available — use char approx).
        # ~3 chars/token for English narrative; leave headroom.
        passage_chars = (max_prompt_tokens - 200) * 3
        passage = full_passage[:passage_chars]
    else:
        # Token-aware truncation: ensure the full prompt fits, with the
        # passage absorbing all the truncation (intro + question + answer
        # marker stay intact). Pre-trim at char-level first so we don't
        # spend O(150K tokens) tokenizing each NarrativeQA passage when
        # we'll only keep the first ~8K.
        intro_ids = tokenizer.encode(intro, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        margin = 16  # safety for tokenizer joining quirks at boundaries
        passage_budget = max_prompt_tokens - len(intro_ids) - len(suffix_ids) - margin
        if passage_budget < 256:
            return None  # question too long; skip this example
        # Conservative char→token estimate: 5 chars/token (loose upper bound
        # — actual narrative text is ~3.5-4 chars/token, but tokenizers
        # sometimes split short words into more tokens than chars/avg
        # suggests). We tokenize a slice ~25% over budget then truncate
        # precisely.
        passage_chars_safe = passage_budget * 5 + 200
        passage_ids = tokenizer.encode(
            full_passage[:passage_chars_safe], add_special_tokens=False,
        )
        if len(passage_ids) > passage_budget:
            passage_ids = passage_ids[:passage_budget]
            passage = tokenizer.decode(passage_ids, skip_special_tokens=True)
        else:
            passage = full_passage[:passage_chars_safe]

    prompt = f"{intro}{passage}{suffix}"
    return prompt, gold, "exact_match_or_bert_cosine", {"all_answers": gold_answers}


def _humaneval_extract(ex):
    prompt = ex["prompt"]
    gold = ex["canonical_solution"]
    return prompt, gold, "rule_based_exec", {
        "test": ex.get("test", ""),
        "entry_point": ex.get("entry_point", ""),
        "task_id": ex.get("task_id", ""),
    }


def _numinamath_extract(ex):
    # NuminaMath-TIR: {"problem", "solution"}
    prompt = (
        "Solve this math problem step by step, including code if needed.\n\n"
        f"Problem: {ex['problem']}\n\nSolution:"
    )
    gold = ex["solution"]
    # B3 fix — brace-balanced extractor for nested-brace answers like
    # `\boxed{\frac{1}{2}}` (the prior `[^}]+` regex stopped at first `}`).
    from src.trajectory_memory.training.rewards import extract_last_boxed
    gold_boxed = extract_last_boxed(gold)
    return prompt, gold, "exact_match", {"gold_boxed": gold_boxed}


_SOURCES = {
    "gsm8k": {
        "id": "openai/gsm8k", "config": "main", "split": "train",
        "extract": _gsm8k_extract,
    },
    "narrativeqa": {
        "id": "deepmind/narrativeqa", "config": None, "split": "train",
        "extract": _narrativeqa_extract,
    },
    "humaneval": {
        "id": "openai_humaneval", "config": None, "split": "test",
        "extract": _humaneval_extract,
    },
    "numinamath": {
        "id": "AI-MO/NuminaMath-TIR", "config": None, "split": "train",
        "extract": _numinamath_extract,
    },
}


def iterate_examples(
    source: str, *, streaming: bool, max_examples: int | None,
) -> Iterable[dict]:
    info = _SOURCES[source]
    kwargs = {"path": info["id"], "split": info["split"], "streaming": streaming}
    if info["config"]:
        kwargs["name"] = info["config"]
    ds = load_dataset(**kwargs)
    for i, ex in enumerate(ds):
        if max_examples is not None and i >= max_examples:
            return
        yield ex


def preprocess(
    source: str,
    *,
    output: Path,
    max_examples: int | None,
    max_prompt_tokens: int,
    max_gold_tokens: int,
    streaming: bool,
) -> None:
    info = _SOURCES[source]
    tok = get_tokenizer()
    extract = info["extract"]

    rows_prompt = []
    rows_gold = []
    rows_num_p = []
    rows_num_g = []
    rows_source = []
    rows_reward = []
    rows_meta = []

    # NarrativeQA needs the tokenizer to do passage-budget truncation so the
    # `Question: .../Answer:` suffix isn't cut off. Other extracts ignore it.
    extract_kwargs = {}
    if source == "narrativeqa":
        extract_kwargs = {"tokenizer": tok, "max_prompt_tokens": max_prompt_tokens}

    n_seen = 0
    n_kept = 0
    for ex in iterate_examples(
        source, streaming=streaming, max_examples=max_examples,
    ):
        n_seen += 1
        try:
            result = extract(ex, **extract_kwargs)
        except Exception as e:
            print(f"  [warn] extract failed at {n_seen}: {e}")
            continue
        if result is None:
            continue
        prompt, gold, reward_kind, meta = result

        prompt_ids = tok.encode(prompt, add_special_tokens=True)[:max_prompt_tokens]
        gold_ids = tok.encode(gold, add_special_tokens=False)[:max_gold_tokens]

        rows_prompt.append(prompt_ids)
        rows_gold.append(gold_ids)
        rows_num_p.append(len(prompt_ids))
        rows_num_g.append(len(gold_ids))
        rows_source.append(source)
        rows_reward.append(reward_kind)
        rows_meta.append(json.dumps(meta))
        n_kept += 1

        if n_kept % 200 == 0:
            print(f"  [{source}] seen={n_seen} kept={n_kept}")

    print(f"  [{source}] total: seen={n_seen} kept={n_kept}")
    if n_kept == 0:
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "prompt_ids": rows_prompt,
        "gold_ids": rows_gold,
        "num_prompt": rows_num_p,
        "num_gold": rows_num_g,
        "source": rows_source,
        "reward_kind": rows_reward,
        "meta_json": rows_meta,
    })
    pq.write_table(table, output)
    print(f"  [{source}] wrote {output} ({output.stat().st_size / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", choices=list(_SOURCES.keys()))
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    ap.add_argument("--max-gold-tokens", type=int, default=2048)
    ap.add_argument("--streaming", action="store_true")
    args = ap.parse_args()

    preprocess(
        source=args.source,
        output=args.output,
        max_examples=args.max_examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_gold_tokens=args.max_gold_tokens,
        streaming=args.streaming,
    )


if __name__ == "__main__":
    main()
