"""Wave 3 passphrase data loader for `phase1_ar_pretrained_step`.

Yields ``Phase1ARBatch(prefix_ids, continuation_ids)`` where:
- ``prefix_ids``: [BS, T_pre] = filler_pre + fact_paraphrase + filler_mid + question
- ``continuation_ids``: [BS, T_cont] = reference_answer (teacher-force target)

The walker writes the fact during the prefix forward; under AR
continuation the LM only sees one token at a time, so the walker is
the only carrier that lets the prediction succeed. See
`docs/wave3_passphrase_plan.md` for the full design.

Key knobs (all controlled by caller):
- ``T_pre``: total prefix length per example. Curriculum-controlled.
- ``T_cont``: continuation (answer) length. Fixed per run, e.g. 48 tokens.
- ``filler_mid_min/max``: token length of filler BETWEEN the fact and
  the question. This is what stresses the walker — it must retain the
  fact across this many tokens. Curriculum ramps this up.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from src.graph_walker.pretrained.train_phase1_ar import Phase1ARBatch


@dataclass
class _ExpandedFact:
    id: int
    topic: str
    fact: str
    paraphrases: list[str]
    questions: list[str]
    reference_answers: list[str]


def _load_facts(path: str | Path) -> list[_ExpandedFact]:
    """Load and validate `expanded.json` produced by build_user_facts.py."""
    with Path(path).open() as f:
        raw = json.load(f)
    facts: list[_ExpandedFact] = []
    for entry in raw:
        facts.append(_ExpandedFact(
            id=entry["id"],
            topic=entry.get("topic", "misc"),
            fact=entry["fact"],
            paraphrases=entry["paraphrases"],
            questions=entry["questions"],
            reference_answers=entry["reference_answers"],
        ))
    return facts


def _split_train_heldout(
    facts: list[_ExpandedFact], n_heldout: int, seed: int = 42,
) -> tuple[list[_ExpandedFact], list[_ExpandedFact]]:
    """Deterministic split. `n_heldout` random facts are held out for
    evaluation; everything else goes to training."""
    rng = random.Random(seed)
    ids = sorted([f.id for f in facts])
    rng.shuffle(ids)
    heldout_ids = set(ids[:n_heldout])
    train_facts: list[_ExpandedFact] = []
    heldout_facts: list[_ExpandedFact] = []
    for f in facts:
        (heldout_facts if f.id in heldout_ids else train_facts).append(f)
    return train_facts, heldout_facts


class _FillerPool:
    """Pre-tokenized FineWeb-edu buffer for cheap filler sampling.

    Reads up to `target_tokens` of FineWeb-edu text at construction,
    tokenizes once with the Llama-3.2 tokenizer, and stores the flat
    token stream. At sample time, slices a random contiguous window.

    Memory cost: target_tokens * 4 bytes (int32) ≈ 40 MB for 10M tokens.
    Fast (no per-batch tokenization).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        tokenizer,
        target_tokens: int = 5_000_000,
        seed: int = 0,
    ) -> None:
        path = Path(parquet_path)
        if not path.exists():
            raise FileNotFoundError(f"FineWeb-edu parquet not found at {path}")
        rng = np.random.default_rng(seed)
        eos_id = tokenizer.eos_token_id
        if eos_id is None:
            raise ValueError("tokenizer.eos_token_id is None")

        # Read parquet in shuffled row order; tokenize until we hit target.
        table = pq.read_table(path, columns=["text"])
        order = rng.permutation(len(table))
        text_col = table.column("text")

        buf: list[int] = []
        for idx in order:
            text = text_col[int(idx)].as_py()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            buf.extend(ids)
            buf.append(eos_id)
            if len(buf) >= target_tokens:
                break
        self.tokens = np.asarray(buf, dtype=np.int32)
        self._rng = np.random.default_rng(seed + 1)
        print(f"[FillerPool] tokenized {len(self.tokens):,} FineWeb-edu tokens from {path.name}")

    def sample(self, n_tokens: int) -> list[int]:
        """Return a random contiguous slice of n_tokens. Wraps if needed."""
        if n_tokens <= 0:
            return []
        if n_tokens > len(self.tokens):
            raise ValueError(f"requested {n_tokens} > pool size {len(self.tokens)}")
        start = int(self._rng.integers(0, len(self.tokens) - n_tokens))
        return self.tokens[start:start + n_tokens].tolist()


def _build_one_example(
    fact: _ExpandedFact,
    rng: random.Random,
    *,
    tokenizer,
    filler_pool: _FillerPool,
    T_pre: int,
    T_cont: int,
    filler_mid_tokens: int,
) -> tuple[list[int], list[int]] | None:
    """Construct one (prefix_ids, continuation_ids) example.

    Returns None if the example doesn't fit T_pre after tokenization (skip).

    Layout:
        [BOS]
        [filler_pre tokens — fills the rest of T_pre]
        [fact_paraphrase tokens]
        [filler_mid_tokens]
        [question tokens]

    continuation = [reference_answer tokens, padded/truncated to T_cont]
    """
    eos_id = tokenizer.eos_token_id
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else eos_id

    fact_text = rng.choice(fact.paraphrases)
    question_text = rng.choice(fact.questions)
    answer_text = rng.choice(fact.reference_answers)

    # Tokenize each component.
    fact_ids = tokenizer.encode(" " + fact_text, add_special_tokens=False)
    q_ids = tokenizer.encode("\n\nQuestion: " + question_text + "\n\nAnswer:",
                              add_special_tokens=False)
    a_ids = tokenizer.encode(" " + answer_text, add_special_tokens=False)

    # Continuation: pad / truncate to T_cont with EOS.
    if len(a_ids) < T_cont:
        a_ids = a_ids + [eos_id] * (T_cont - len(a_ids))
    else:
        a_ids = a_ids[:T_cont]

    # Compute how much filler_pre we have room for.
    # T_pre = 1 (BOS) + filler_pre + fact + filler_mid + q
    fixed_tokens = 1 + len(fact_ids) + filler_mid_tokens + len(q_ids)
    filler_pre_len = T_pre - fixed_tokens
    if filler_pre_len < 0:
        return None  # Doesn't fit; caller should reduce filler_mid or shorten q

    filler_pre = filler_pool.sample(filler_pre_len) if filler_pre_len > 0 else []
    filler_mid = filler_pool.sample(filler_mid_tokens) if filler_mid_tokens > 0 else []

    prefix_ids = [bos_id] + filler_pre + fact_ids + filler_mid + q_ids
    if len(prefix_ids) != T_pre:
        # Shouldn't happen given the math above, but guard against off-by-one.
        if len(prefix_ids) > T_pre:
            prefix_ids = prefix_ids[:T_pre]
        else:
            prefix_ids = prefix_ids + [eos_id] * (T_pre - len(prefix_ids))
    return prefix_ids, a_ids


def passphrase_phase1ar_iter(
    expanded_path: str | Path,
    tokenizer,
    filler_parquet: str | Path,
    *,
    bs: int,
    T_pre: int,
    T_cont: int = 48,
    filler_mid_schedule: list[tuple[int, int]] | None = None,
    n_heldout: int = 20,
    device: torch.device | str = "cuda",
    seed: int = 0,
    max_batches: int | None = None,
    use_heldout_split: bool = False,
    filler_pool_size: int = 5_000_000,
) -> Iterator[Phase1ARBatch]:
    """Yield Phase1ARBatch from passphrase data.

    Args:
        expanded_path: path to expanded.json (from build_user_facts.py).
        tokenizer: HF tokenizer (same one used by Llama-3.2).
        filler_parquet: path to FineWeb-edu parquet.
        bs: batch size.
        T_pre: total prefix length per example. Constant within a run.
        T_cont: continuation (answer) length. Constant within a run.
        filler_mid_schedule: list of (step_threshold, filler_mid_tokens)
            tuples. At step S, the largest threshold T <= S determines
            filler_mid. Default: [(0, 100), (1000, 300), (3000, 800), (6000, 1500)].
        n_heldout: # facts held out for evaluation (no overlap with training).
        use_heldout_split: if True, sample only from the held-out set; else
            from the training set.

    The curriculum schedule is checked at every batch via an internal
    step counter that advances with each yielded batch.
    """
    if filler_mid_schedule is None:
        filler_mid_schedule = [(0, 100), (1000, 300), (3000, 800), (6000, 1500)]

    facts = _load_facts(expanded_path)
    train_facts, heldout_facts = _split_train_heldout(facts, n_heldout=n_heldout, seed=seed)
    pool_facts = heldout_facts if use_heldout_split else train_facts
    if not pool_facts:
        raise ValueError(f"no facts in {'heldout' if use_heldout_split else 'train'} split")

    filler_pool = _FillerPool(filler_parquet, tokenizer, target_tokens=filler_pool_size, seed=seed)
    rng = random.Random(seed + 1)
    device_t = torch.device(device)

    def _filler_mid_for_step(step: int) -> int:
        """Pick filler_mid length per current schedule entry."""
        chosen = filler_mid_schedule[0][1]
        for threshold, length in filler_mid_schedule:
            if step >= threshold:
                chosen = length
        return chosen

    step = 0
    yielded = 0
    while True:
        filler_mid_tokens = _filler_mid_for_step(step)
        prefixes: list[list[int]] = []
        continuations: list[list[int]] = []
        attempts = 0
        while len(prefixes) < bs:
            attempts += 1
            if attempts > bs * 20:
                raise RuntimeError(
                    f"Couldn't construct {bs} examples with T_pre={T_pre}, "
                    f"filler_mid={filler_mid_tokens}. T_pre too small for "
                    "the answer/question/fact lengths."
                )
            fact = rng.choice(pool_facts)
            ex = _build_one_example(
                fact, rng, tokenizer=tokenizer, filler_pool=filler_pool,
                T_pre=T_pre, T_cont=T_cont, filler_mid_tokens=filler_mid_tokens,
            )
            if ex is None:
                continue
            prefixes.append(ex[0])
            continuations.append(ex[1])

        prefix_arr = np.asarray(prefixes, dtype=np.int64)
        cont_arr = np.asarray(continuations, dtype=np.int64)
        prefix_ids = torch.from_numpy(prefix_arr).to(device_t, non_blocking=True)
        continuation_ids = torch.from_numpy(cont_arr).to(device_t, non_blocking=True)
        yield Phase1ARBatch(prefix_ids=prefix_ids, continuation_ids=continuation_ids)

        step += 1
        yielded += 1
        if max_batches is not None and yielded >= max_batches:
            return
