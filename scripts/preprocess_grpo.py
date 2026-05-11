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


def _musique_extract(ex, *, tokenizer=None, max_prompt_tokens=16384,
                     pool=None, n_distractors=0, rng=None):
    """MuSiQue-Ans: multi-hop QA over Wikipedia paragraphs with distractors.

    Format: {"paragraphs": [{"idx", "title", "paragraph_text", "is_supporting"}],
             "question", "answer", "answer_aliases": [str]}.

    Build prompt by concatenating ALL paragraphs (supporting + distractors)
    in given order — that's the standard packing the LongBench mixup uses
    to get ~14K-token contexts that exercise memory.
    """
    paragraphs = ex.get("paragraphs", [])
    question = ex.get("question") or ""
    answer = ex.get("answer") or ""
    aliases = ex.get("answer_aliases") or []
    if not paragraphs or not question or not answer:
        return None

    # Build context with one paragraph per line, title-prefixed.
    parts = []
    for p in paragraphs:
        title = p.get("title", "").strip()
        body = p.get("paragraph_text", "").strip()
        if not body:
            continue
        parts.append(f"Title: {title}\n{body}" if title else body)

    if pool is not None and n_distractors > 0 and rng is not None:
        context = _inject_distractors(parts, pool, n_distractors, rng)
    else:
        context = "\n\n".join(parts)
    intro = "Read the following passages and answer the question.\n\n"
    suffix = f"\n\nQuestion: {question}\n\nAnswer:"

    # Token-aware truncation (mirrors narrativeqa pattern).
    if tokenizer is not None:
        intro_ids = tokenizer.encode(intro, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        margin = 16
        budget = max_prompt_tokens - len(intro_ids) - len(suffix_ids) - margin
        if budget < 512:
            return None
        passage_chars_safe = budget * 5 + 200
        passage_ids = tokenizer.encode(
            context[:passage_chars_safe], add_special_tokens=False,
        )
        if len(passage_ids) > budget:
            passage_ids = passage_ids[:budget]
            context = tokenizer.decode(passage_ids, skip_special_tokens=True)

    prompt = f"{intro}{context}{suffix}"
    all_answers = [answer] + [a for a in aliases if a]
    return prompt, answer, "f1_qa", {"all_answers": all_answers}


def _hotpotqa_extract(ex, *, tokenizer=None, max_prompt_tokens=16384,
                      pool=None, n_distractors=0, rng=None):
    """HotpotQA distractor split: 10 paragraphs (some supporting, some
    distractor) and a multi-hop question.

    HF schema (`hotpotqa/hotpot_qa`, config `distractor`):
      context: {"title": [...], "sentences": [[s, s, ...], ...]}
      question, answer
      type: 'bridge' | 'comparison'
      supporting_facts: {"title": [...], "sent_id": [...]}  (for eval, not used here)
    """
    ctx = ex.get("context") or {}
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    question = ex.get("question") or ""
    answer = ex.get("answer") or ""
    if not titles or not sentences or not question or not answer:
        return None

    parts = []
    for title, sents in zip(titles, sentences):
        body = " ".join(s.strip() for s in sents if s).strip()
        if not body:
            continue
        parts.append(f"Title: {title}\n{body}")

    if pool is not None and n_distractors > 0 and rng is not None:
        context = _inject_distractors(parts, pool, n_distractors, rng)
    else:
        context = "\n\n".join(parts)
    intro = "Read the following passages and answer the question.\n\n"
    suffix = f"\n\nQuestion: {question}\n\nAnswer:"

    if tokenizer is not None:
        intro_ids = tokenizer.encode(intro, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        margin = 16
        budget = max_prompt_tokens - len(intro_ids) - len(suffix_ids) - margin
        if budget < 512:
            return None
        passage_chars_safe = budget * 5 + 200
        passage_ids = tokenizer.encode(
            context[:passage_chars_safe], add_special_tokens=False,
        )
        if len(passage_ids) > budget:
            passage_ids = passage_ids[:budget]
            context = tokenizer.decode(passage_ids, skip_special_tokens=True)

    prompt = f"{intro}{context}{suffix}"
    return prompt, answer, "f1_qa", {"all_answers": [answer]}


def _2wikimultihop_extract(ex, *, tokenizer=None, max_prompt_tokens=8192,
                            pool=None, n_distractors=0, rng=None):
    """2WikiMultiHopQA: multi-hop chains across Wikipedia paragraphs.

    HF schema (`framolfese/2WikiMultihopQA` — used because the canonical
    `xanhho/2WikiMultihopQA` requires a Python loading script which HF
    datasets ≥3.0 no longer supports):
      context: {"title": [...], "sentences": [[s, s, ...], ...]}
        — same shape as HotpotQA, identical extractor logic.
      question, answer
      supporting_facts (for eval)
    """
    ctx = ex.get("context") or {}
    titles = ctx.get("title") or []
    sentences = ctx.get("sentences") or []
    question = ex.get("question") or ""
    answer = ex.get("answer") or ""
    if not titles or not sentences or not question or not answer:
        return None

    parts = []
    for title, sents in zip(titles, sentences):
        body = " ".join(s.strip() for s in sents if isinstance(s, str)).strip()
        if not body:
            continue
        parts.append(f"Title: {title}\n{body}")

    if pool is not None and n_distractors > 0 and rng is not None:
        context = _inject_distractors(parts, pool, n_distractors, rng)
    else:
        context = "\n\n".join(parts)
    if not context:
        return None
    intro = "Read the following passages and answer the question.\n\n"
    suffix = f"\n\nQuestion: {question}\n\nAnswer:"

    if tokenizer is not None:
        intro_ids = tokenizer.encode(intro, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        margin = 16
        budget = max_prompt_tokens - len(intro_ids) - len(suffix_ids) - margin
        if budget < 512:
            return None
        passage_chars_safe = budget * 5 + 200
        passage_ids = tokenizer.encode(
            context[:passage_chars_safe], add_special_tokens=False,
        )
        if len(passage_ids) > budget:
            passage_ids = passage_ids[:budget]
            context = tokenizer.decode(passage_ids, skip_special_tokens=True)

    prompt = f"{intro}{context}{suffix}"
    return prompt, answer, "f1_qa", {"all_answers": [answer]}


# ── LongBench-style distractor mixup ──────────────────────────────────
#
# Native HotpotQA-distractor / 2WikiMultiHopQA prompts are only ~1.3K /
# ~0.9K tokens — they fit comfortably in Llama's 2K KV cache and don't
# exercise the memory module. The LongBench benchmark "packs" multiple
# examples' contexts together to extend each example to 6K-12K tokens.
# We replicate that: for each example we sample K random distractor
# paragraphs from OTHER examples in the same source and interleave them
# with the example's own passages. The model still has to find the
# relevant 10-ish supporting paragraphs among the larger context — that's
# exactly the long-range retrieval pattern we want to train.


def _build_paragraph_pool_hotpotqa(examples: list[dict]) -> list[tuple[str, str]]:
    """Flatten all hotpotqa contexts into a (title, body) pool."""
    pool: list[tuple[str, str]] = []
    for ex in examples:
        ctx = ex.get("context") or {}
        titles = ctx.get("title") or []
        sentences = ctx.get("sentences") or []
        for title, sents in zip(titles, sentences):
            body = " ".join(s.strip() for s in sents if isinstance(s, str)).strip()
            if body:
                pool.append((title, body))
    return pool


def _build_paragraph_pool_musique(examples: list[dict]) -> list[tuple[str, str]]:
    """Flatten all musique paragraphs into a (title, body) pool. Excludes
    `is_supporting=True` paragraphs to avoid sampling someone else's
    *answer* as a distractor in our context."""
    pool: list[tuple[str, str]] = []
    for ex in examples:
        for p in ex.get("paragraphs", []):
            if p.get("is_supporting"):
                continue
            title = (p.get("title") or "").strip()
            body = (p.get("paragraph_text") or "").strip()
            if body:
                pool.append((title, body))
    return pool


def _inject_distractors(
    own_parts: list[str],
    pool: list[tuple[str, str]],
    n_distractors: int,
    rng,
) -> str:
    """Interleave the example's own context paragraphs with N random
    distractor paragraphs sampled from `pool`. Order is randomized so
    supporting paragraphs aren't always grouped at the start."""
    distractor_idxs = rng.sample(range(len(pool)), min(n_distractors, len(pool)))
    extra_parts = [f"Title: {t}\n{b}" if t else b for t, b in (pool[i] for i in distractor_idxs)]
    combined = list(own_parts) + extra_parts
    rng.shuffle(combined)
    return "\n\n".join(combined)


def _quality_extract(ex, *, tokenizer=None, max_prompt_tokens=8192):
    """QuALITY: multiple-choice QA over ~5K-token articles.

    HF schema (`emozilla/quality`):
      article (str)
      question (str)
      options (list[str], 4 options)
      gold_label (int, 1-indexed) OR answer (int 0-indexed)

    Reward: mc_letter — string-match the candidate's first A/B/C/D letter
    against the gold letter. Cheapest verifiable reward in our suite.
    """
    article = ex.get("article") or ""
    question = ex.get("question") or ""
    options = ex.get("options") or []
    # HF version stores gold as 1-indexed `gold_label`; some mirrors use
    # 0-indexed `answer`. Accept either.
    gold_idx = ex.get("gold_label")
    if gold_idx is not None:
        gold_idx = int(gold_idx) - 1   # to 0-indexed
    else:
        gold_idx = ex.get("answer")
        gold_idx = int(gold_idx) if gold_idx is not None else None

    if not article or not question or len(options) != 4 or gold_idx is None:
        return None
    if not (0 <= gold_idx < 4):
        return None
    gold_letter = "ABCD"[gold_idx]
    gold_text = options[gold_idx]

    # Build prompt.
    intro = "Read the following passage and answer the multiple-choice question.\n\nPassage:\n"
    options_block = "\n".join(
        f"{letter}. {opt}" for letter, opt in zip("ABCD", options)
    )
    suffix = f"\n\nQuestion: {question}\n\n{options_block}\n\nAnswer:"

    if tokenizer is not None:
        intro_ids = tokenizer.encode(intro, add_special_tokens=True)
        suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        margin = 16
        budget = max_prompt_tokens - len(intro_ids) - len(suffix_ids) - margin
        if budget < 512:
            return None
        passage_chars_safe = budget * 5 + 200
        passage_ids = tokenizer.encode(
            article[:passage_chars_safe], add_special_tokens=False,
        )
        if len(passage_ids) > budget:
            passage_ids = passage_ids[:budget]
            article = tokenizer.decode(passage_ids, skip_special_tokens=True)

    prompt = f"{intro}{article}{suffix}"
    # gold_text is the option content; gold_letter is the MC label.
    # mc_letter reward keys on `gold_letter`; we keep `gold_text` for
    # debug / alternative scoring.
    return prompt, gold_letter, "mc_letter", {
        "gold_letter": gold_letter, "gold_text": gold_text,
    }


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
        # `openai/openai_humaneval` is the canonical HF id; bare
        # `openai_humaneval` resolves via legacy alias but the org-prefixed
        # form is documented and avoids future rename breakage.
        "id": "openai/openai_humaneval", "config": None, "split": "test",
        "extract": _humaneval_extract,
    },
    "numinamath": {
        "id": "AI-MO/NuminaMath-TIR", "config": None, "split": "train",
        "extract": _numinamath_extract,
    },
    # Long-context multi-hop QA additions (memory-stress training data).
    # All use F1 reward, packing all distractor + supporting paragraphs
    # into the prompt to force memory engagement.
    "musique": {
        "id": "dgslibisey/MuSiQue", "config": None, "split": "train",
        "extract": _musique_extract,
    },
    "hotpotqa": {
        "id": "hotpotqa/hotpot_qa", "config": "distractor", "split": "train",
        "extract": _hotpotqa_extract,
    },
    "2wikimultihop": {
        # framolfese mirror — canonical xanhho/2WikiMultihopQA requires a
        # Python loading script (blocked by HF datasets ≥3.0). framolfese
        # uses HotpotQA-style context dict.
        "id": "framolfese/2WikiMultihopQA", "config": None, "split": "train",
        "extract": _2wikimultihop_extract,
    },
    "quality": {
        "id": "emozilla/quality", "config": None, "split": "train",
        "extract": _quality_extract,
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
    n_distractors: int = 0,
    seed: int = 0,
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

    # Long-context sources need the tokenizer to do passage-budget
    # truncation so the question / answer markers aren't cut off when
    # the document is longer than `max_prompt_tokens`.
    extract_kwargs = {}
    if source in ("narrativeqa", "musique", "hotpotqa", "2wikimultihop", "quality"):
        extract_kwargs = {"tokenizer": tok, "max_prompt_tokens": max_prompt_tokens}

    # LongBench-style distractor padding for short multi-hop QA sources.
    # Requires non-streaming mode because we need to build the
    # paragraph pool first. Only meaningful for {musique, hotpotqa,
    # 2wikimultihop} — the others are either already long or have a
    # different prompt structure.
    pool = None
    rng = None
    if n_distractors > 0:
        if source not in ("musique", "hotpotqa", "2wikimultihop"):
            print(f"  [warn] --pad-with-distractors only supported for "
                  f"musique/hotpotqa/2wikimultihop, skipping for {source}")
        elif streaming:
            raise ValueError(
                "--pad-with-distractors requires non-streaming mode "
                "(need to materialize the example pool first). Run "
                "without --streaming."
            )
        else:
            print(f"  [{source}] building distractor pool ({max_examples or 'all'} examples)...")
            import random as _random
            rng = _random.Random(seed)
            all_examples = list(iterate_examples(
                source, streaming=False, max_examples=max_examples,
            ))
            if source == "musique":
                pool = _build_paragraph_pool_musique(all_examples)
            else:
                pool = _build_paragraph_pool_hotpotqa(all_examples)
            print(f"  [{source}] pool: {len(pool)} paragraphs from {len(all_examples)} examples")
            extract_kwargs["pool"] = pool
            extract_kwargs["n_distractors"] = n_distractors
            extract_kwargs["rng"] = rng
            # Use the pre-loaded examples for iteration.
            iter_source = iter(all_examples)
        # Sentinel for the loop below.
        use_preloaded = pool is not None
    else:
        use_preloaded = False
        iter_source = None

    n_seen = 0
    n_kept = 0
    if use_preloaded:
        iterator = iter_source
    else:
        iterator = iterate_examples(
            source, streaming=streaming, max_examples=max_examples,
        )
    for ex in iterator:
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
    ap.add_argument(
        "--pad-with-distractors", type=int, default=0,
        help="LongBench-style distractor mixup: append N random paragraphs "
             "from other examples' contexts to extend the prompt. Only "
             "supported for {musique, hotpotqa, 2wikimultihop}. Requires "
             "non-streaming mode (builds pool first). "
             "Recommended: 40-60 for hotpotqa/2wikimultihop to hit ~6-8K tokens.",
    )
    args = ap.parse_args()

    preprocess(
        source=args.source,
        output=args.output,
        max_examples=args.max_examples,
        max_prompt_tokens=args.max_prompt_tokens,
        max_gold_tokens=args.max_gold_tokens,
        streaming=args.streaming,
        n_distractors=args.pad_with_distractors,
    )


if __name__ == "__main__":
    main()
