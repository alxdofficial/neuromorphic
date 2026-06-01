"""v1h QA dataset for representation learning.

Three sources, all sharing the same `(context_ids, question_ids,
answer_ids, ...)` interface:

- composite_v1 (synthetic, 9 task families) — `QADataset`
- HotpotQA distractor split (multi-hop) — `HotpotQADataset`
- NarrativeQA (long-form, narrative) — `NarrativeQADataset`

A `MixedQADataset` combines them with configurable weights and emits
exactly one example per `__next__`. Each example is a 4096-token context,
a question, and an answer (with optional content-mask positions for
load-bearing answer tokens).

For HotpotQA/NarrativeQA we treat the entire answer span as content
(no filler distinction). For composite_v1 we honor the explicit
`answer_content_token_positions`.
"""
from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data.wave1.common.sampler import CompositeSampler  # noqa: E402

from .config import ReprConfig


@dataclass
class QABatch:
    """A batched v1h example for QA training."""
    context_ids: Tensor          # [B, T_ctx] long — packed passages
    context_mask: Tensor         # [B, T_ctx] bool — True at valid context positions
    question_ids: Tensor         # [B, T_q] long — right-padded
    question_mask: Tensor        # [B, T_q] bool
    answer_ids: Tensor           # [B, T_a] long — right-padded
    answer_mask: Tensor          # [B, T_a] bool — True at valid (non-pad) answer positions
    answer_content_mask: Tensor  # [B, T_a] bool — True at load-bearing content positions
    task_family: list[str]       # length B — per-example task family for telemetry
    question_type: list[str]     # length B — per-example question type for telemetry


def _pack_context(
    passage_token_ids_list: list[list[int]],
    chunk_size: int,
    sep_token_id: int,
    pad_token_id: int,
) -> tuple[list[int], int]:
    """Concat passages separated by sep_token_id. Truncate or pad to chunk_size.
    Returns (token_ids, valid_length)."""
    out: list[int] = []
    for i, p in enumerate(passage_token_ids_list):
        if i > 0:
            out.append(sep_token_id)
        out.extend(p)
        if len(out) >= chunk_size:
            out = out[:chunk_size]
            return out, chunk_size
    valid = len(out)
    if valid < chunk_size:
        out = out + [pad_token_id] * (chunk_size - valid)
    return out, valid


class QADataset(IterableDataset):
    """Iterable dataset yielding packed-context QA examples for v1h."""

    def __init__(
        self,
        passages_path: Path,
        questions_path: Path,
        chunk_size: int = 4096,
        passages_per_chunk: int = 80,
        sep_token_id: int = 198,           # newline (matches sentence-pack default)
        pad_token_id: int = 128_001,       # Llama-3.2 EOS used as pad
        task_weights: Optional[dict[str, float]] = None,
        seed: int = 0,
    ):
        super().__init__()
        self.passages_path = Path(passages_path)
        self.questions_path = Path(questions_path)
        self.chunk_size = chunk_size
        self.passages_per_chunk = passages_per_chunk
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.task_weights = task_weights
        self.seed = seed

        print(f"[data v1h] loading composite_v1 from {self.passages_path.parent}")
        self.sampler = CompositeSampler(
            self.passages_path, self.questions_path,
            task_weights=task_weights, seed=seed,
        )
        n_passages = sum(len(v) for v in self.sampler.passages_by_family.values())
        n_questions = sum(len(v) for v in self.sampler.questions_by_family.values())
        print(f"[data v1h]   {n_passages:,} passages, {n_questions:,} questions "
              f"across {len(self.sampler._families)} task families")

    def __iter__(self) -> Iterator[dict]:
        # Worker-aware RNG for evidence/distractor interleaving
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid * 100_003 + 31)
        # CRITICAL fix (2026-05-28): re-seed the CompositeSampler's RNG per
        # worker too — forked workers inherit the same self.sampler.rng state
        # set at __init__, so without this every worker produces identical
        # (family, question_id) sequences and effective composite diversity
        # is halved at num_workers=2. The +7 offset keeps it distinct from
        # the evidence-shuffling RNG above.
        self.sampler.rng = random.Random(self.seed + wid * 100_003 + 7)

        while True:
            chunk = self.sampler.sample_chunk(self.passages_per_chunk)
            passages = chunk["passages"]
            target_q = chunk["target_question"]
            evidence_idxs = set(chunk["evidence_idxs"])

            # Split into evidence vs distractors. Evidence MUST end up in
            # the 4096-token context, otherwise the example is unsolvable
            # by the encoder and the loss signal is misleading.
            evidence = [passages[i] for i in evidence_idxs]
            distractors = [passages[i] for i, _ in enumerate(passages) if i not in evidence_idxs]

            # Pack evidence first; if it can't fit alone, skip this example.
            ev_tokens = [p["passage_token_ids"] for p in evidence]
            ev_total = sum(len(t) for t in ev_tokens) + max(0, len(ev_tokens) - 1)
            if ev_total > self.chunk_size:
                continue  # evidence-too-big; resample

            # Greedily insert distractors at random positions until budget exhausted
            rng.shuffle(distractors)
            packed = list(ev_tokens)
            cur_total = ev_total
            for d in distractors:
                d_tokens = d["passage_token_ids"]
                add_cost = len(d_tokens) + 1  # plus separator
                if cur_total + add_cost > self.chunk_size:
                    continue  # this distractor too big; try next (might be smaller)
                insert_pos = rng.randint(0, len(packed))
                packed.insert(insert_pos, d_tokens)
                cur_total += add_cost

            # Final shuffle of evidence positions within the packed list:
            # at this point evidence is interleaved with distractors via
            # the random insertions above, so positional fairness holds.
            ctx_tokens, valid_len = _pack_context(
                packed,
                chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
            )

            q_ids = list(target_q["question_token_ids"])
            a_ids = list(target_q["answer_token_ids"])
            content_positions = list(
                target_q.get("answer_content_token_positions") or []
            )
            if not content_positions:
                content_positions = list(range(len(a_ids)))

            content_mask = [False] * len(a_ids)
            for pos in content_positions:
                if 0 <= pos < len(a_ids):
                    content_mask[pos] = True

            # For composite_v1, refs = [target_value] (load-bearing content
            # only) so EM/F1 isn't fooled by templated prefixes. Fall back
            # to the full answer string when target_value is absent.
            answer_full = target_q.get("answer") or ""
            target_val = target_q.get("target_value") or ""
            # Span-only when target_value exists (drop the echoing sentence —
            # else F1 pays out for parroting the question prefix). Fall back to
            # the full answer only when target_value is absent.
            refs = [target_val] if target_val else ([answer_full] if answer_full else [])
            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": chunk["task_family"],
                "question_type": target_q["question_type"],
                "answer_refs": refs,
            }


def collate_qa(samples: list[dict], pad_token_id: int = 128_001) -> QABatch:
    """Stack per-sample dicts into a QABatch, padding question and answer
    to the max length in the batch."""
    B = len(samples)
    T_q = max(int(s["question_ids"].shape[0]) for s in samples)
    T_a = max(int(s["answer_ids"].shape[0]) for s in samples)

    context_ids = torch.stack([s["context_ids"] for s in samples])
    context_mask = torch.stack([s["context_mask"] for s in samples])

    question_ids = torch.full((B, T_q), pad_token_id, dtype=torch.long)
    question_mask = torch.zeros((B, T_q), dtype=torch.bool)
    answer_ids = torch.full((B, T_a), pad_token_id, dtype=torch.long)
    answer_mask = torch.zeros((B, T_a), dtype=torch.bool)
    answer_content_mask = torch.zeros((B, T_a), dtype=torch.bool)

    for i, s in enumerate(samples):
        tq = int(s["question_ids"].shape[0])
        ta = int(s["answer_ids"].shape[0])
        question_ids[i, :tq] = s["question_ids"]
        question_mask[i, :tq] = True
        answer_ids[i, :ta] = s["answer_ids"]
        answer_mask[i, :ta] = True
        cm = s["answer_content_mask_list"]
        for j, v in enumerate(cm):
            if j < T_a and v:
                answer_content_mask[i, j] = True

    return QABatch(
        context_ids=context_ids,
        context_mask=context_mask,
        question_ids=question_ids,
        question_mask=question_mask,
        answer_ids=answer_ids,
        answer_mask=answer_mask,
        answer_content_mask=answer_content_mask,
        task_family=[s["task_family"] for s in samples],
        question_type=[s["question_type"] for s in samples],
    )


def make_qa_dataloader(
    cfg: ReprConfig,
    passages_path: Path | str,
    questions_path: Path | str,
    chunk_size: int = 4096,
    passages_per_chunk: int = 80,
    num_workers: int = 0,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> DataLoader:
    ds = QADataset(
        Path(passages_path), Path(questions_path),
        chunk_size=chunk_size,
        passages_per_chunk=passages_per_chunk,
        sep_token_id=cfg.sep_token_id,
        pad_token_id=cfg.pad_token_id,
        seed=seed,
    )
    pad_id = cfg.pad_token_id
    return DataLoader(
        ds,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        num_workers=num_workers,
        collate_fn=lambda s: collate_qa(s, pad_token_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# HotpotQA — distractor split, multi-hop questions over Wikipedia paragraphs
# ═══════════════════════════════════════════════════════════════════════════


class HotpotQADataset(IterableDataset):
    """HotpotQA distractor config. Each example is a question + answer + 10
    Wikipedia paragraphs (mix of gold supporting paragraphs and distractors).
    We concatenate the 10 paragraphs, tokenize, and pack to `chunk_size`.

    Multi-hop questions force memory to combine information from multiple
    paragraphs — a meaningfully harder memory test than single-fact recall.

    First call downloads ~570MB from HuggingFace.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        sep_token_id: int = 198,             # newline
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        print(f"[data v1h] loading HotpotQA distractor split={split} "
              f"(downloads ~570MB on first call)...")
        self.data = load_dataset("hotpot_qa", "distractor", split=split,
                                  trust_remote_code=True)
        print(f"[data v1h]   {len(self.data):,} HotpotQA {split} examples")

    def _tokenize_paragraph(self, title: str, sents: list) -> list:
        text = title.strip() + " " + " ".join(s.strip() for s in sents)
        return self.tokenizer(text, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _sample_distractor_paragraph(self, rng, avoid_idx: int) -> list:
        """Pick a random paragraph from a random example (not avoid_idx)."""
        n = len(self.data)
        for _ in range(8):  # bounded retries to dodge degenerate cases
            j = rng.randint(0, n - 1)
            if j == avoid_idx:
                continue
            ex_j = self.data[j]
            ctx_j = ex_j["context"]
            titles = ctx_j["title"]
            sents_list = ctx_j["sentences"]
            if not titles:
                continue
            k = rng.randint(0, len(titles) - 1)
            return self._tokenize_paragraph(titles[k], sents_list[k])
        return []  # gave up

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        # Top-up loop targets ≥95% fill. Each candidate paragraph is
        # ~150-200 tokens; the loop keeps trying smaller paragraphs once
        # the budget tightens. MAX_TOPUP_TRIES bounds worst case.
        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            ctx = ex["context"]
            titles = ctx["title"]
            sents_list = ctx["sentences"]

            # Identify supporting (gold) paragraphs — these MUST stay in context.
            # supporting_facts has parallel lists `title` and `sent_id`. We mark
            # any paragraph whose title appears in supporting_facts as supporting.
            sf = ex.get("supporting_facts", {})
            support_titles = set(sf.get("title", []))
            is_support = [t in support_titles for t in titles]

            # Tokenize own paragraphs
            own_tok = [self._tokenize_paragraph(t, s)
                       for t, s in zip(titles, sents_list)]

            # Pack supporting first (guaranteed inclusion). If ANY support
            # paragraph won't fit chunk_size, this example is unsolvable —
            # skip it (resample) rather than yield an example whose answer
            # is not derivable from the visible context.
            packed = []
            cur_total = 0
            sep_cost = 1  # one sep_token between paragraphs
            order = sorted(range(len(own_tok)),
                           key=lambda i: not is_support[i])  # support first
            example_ok = True
            for i in order:
                tk = own_tok[i]
                add = len(tk) + (sep_cost if packed else 0)
                if cur_total + add > self.chunk_size:
                    if is_support[i]:
                        example_ok = False
                        break  # bail; this example is unsolvable
                    continue   # non-supporting can be skipped
                packed.append(tk)
                cur_total += add
            if not example_ok:
                continue  # outer while loop — resample a different example

            # Top-up: add random distractors from OTHER examples until full
            tries = 0
            while (cur_total < FILL_THRESHOLD * self.chunk_size
                   and tries < MAX_TOPUP_TRIES):
                tries += 1
                tk = self._sample_distractor_paragraph(rng, avoid_idx=idx)
                if not tk:
                    continue
                add = len(tk) + sep_cost
                if cur_total + add > self.chunk_size:
                    continue
                # Insert at random position to avoid front-loading
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, tk)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed,
                chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
            )

            q_ids = tok(ex["question"], add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_text = ex["answer"]
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]

            content_mask = [True] * len(a_ids)

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "hotpot_qa",
                "question_type": ex.get("type", "comparison_or_bridge"),
                "answer_refs": [a_text],
            }


# ═══════════════════════════════════════════════════════════════════════════
# NarrativeQA — long-form QA over books and movie scripts
# ═══════════════════════════════════════════════════════════════════════════


class NarrativeQADataset(IterableDataset):
    """NarrativeQA in the **summaries-only** setting (Kočiský et al., 2018, TACL).

    NarrativeQA ships two official tasks: 'summaries only' (the ~659-token
    human-written plot summary) and 'stories only' (the full book/script,
    ~60k tokens average, up to 400k+). The QA pairs were authored by
    annotators who read the SUMMARY, so the answer-bearing facts live in the
    summary — not at an arbitrary offset in the full story. We therefore use
    the summary as the gold context and pack OTHER stories' summaries as
    distractors to fill the chunk (mirrors HotpotQA/MuSiQue packing), so the
    supporting facts are guaranteed present and the example is a genuine
    retrieve-among-distractors read.

    (Earlier versions took a random `chunk_size` window of the full ~60k-token
    story — that is NEITHER official setting and leaves the answer facts
    absent ≈100% of the time; verified by data-sample audit 2026-06-01.)

    Answers are free-form abstractive (mean ~4.7 tokens, not document spans),
    so EM/containment under-credit them; the headline metric is the LLM judge
    with EM/containment as quick secondary signals (max-over-refs).

    First call downloads ~700MB from HuggingFace.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        print(f"[data v1h] loading NarrativeQA (summaries-only) split={split} "
              f"(downloads ~700MB on first call)...")
        self.data = load_dataset("narrativeqa", split=split, trust_remote_code=True)
        print(f"[data v1h]   {len(self.data):,} NarrativeQA {split} examples")

    def _summary_ids(self, ex) -> list:
        """Tokenize a story's human-written summary. Raises a clear error if
        the expected schema field is missing."""
        try:
            txt = ex["document"]["summary"]["text"]
        except (KeyError, TypeError) as e:  # pragma: no cover — schema guard
            raise KeyError(
                "NarrativeQA example missing document.summary.text — the HF "
                "`narrativeqa` schema changed. Summaries-only setting requires "
                "the summary field."
            ) from e
        return self.tokenizer(txt, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _doc_id(self, ex):
        return ex["document"]["id"]

    def _sample_distractor_summary(self, rng, avoid_doc_id: str) -> list:
        """A random OTHER story's summary, for distractor packing."""
        n = len(self.data)
        for _ in range(8):
            j = rng.randint(0, n - 1)
            ex_j = self.data[j]
            if self._doc_id(ex_j) == avoid_doc_id:
                continue
            toks = self._summary_ids(ex_j)
            if toks:
                return toks
        return []

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 17 + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            q_text = ex["question"]["text"]
            # Multiple reference answers; sample one for the training tensor,
            # surface ALL for multi-ref eval (scoring is max over references).
            ans_list = ex["answers"]
            if not ans_list:
                continue
            answer_refs = [a["text"] for a in ans_list if a.get("text")]
            if not answer_refs:
                continue
            a_text = rng.choice(answer_refs)

            gold = self._summary_ids(ex)
            if not gold:
                continue
            if len(gold) > self.chunk_size:        # ~659 tok; guard anyway
                gold = gold[:self.chunk_size]

            # Gold summary guaranteed in; top-up with other-story summaries as
            # distractors until ≥95% fill, inserting at random positions so the
            # gold summary isn't always first (positional fairness).
            doc_id = self._doc_id(ex)
            packed = [gold]
            cur_total = len(gold)
            tries = 0
            while (cur_total < FILL_THRESHOLD * self.chunk_size
                   and tries < MAX_TOPUP_TRIES):
                tries += 1
                d = self._sample_distractor_summary(rng, doc_id)
                if not d:
                    continue
                add = len(d) + 1  # plus separator
                if cur_total + add > self.chunk_size:
                    continue
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, d)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed, chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id, pad_token_id=self.pad_token_id,
            )

            q_ids = tok(q_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            content_mask = [True] * len(a_ids)

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "narrative_qa",
                "question_type": "narrative",
                "answer_refs": answer_refs,  # multi-ref for eval; ignored by collate
            }


# ═══════════════════════════════════════════════════════════════════════════
# MuSiQue-Ans — 2-4 hop QA with contamination-controlled chains
# ═══════════════════════════════════════════════════════════════════════════


class MuSiQueDataset(IterableDataset):
    """MuSiQue-Ans (Trivedi et al., 2022). 2-4 hop questions constructed by
    chaining single-hop questions with answerability filters — wider
    human-machine gap than HotpotQA's shortcut-friendly setup. Same
    Wikipedia substrate, so the contrast diagnoses shortcut learning.

    Each example: 20 paragraphs (mix of supporting + distractor) + a
    multi-hop question + answer (+ answer_aliases for liberal scoring).
    Packing mirrors HotpotQA: supporting first (guaranteed inclusion),
    own distractors fill, cross-example top-up to ≥95% fill.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        print(f"[data v1h] loading MuSiQue split={split} "
              f"(downloads ~120MB on first call)...")
        # MuSiQue-Ans is hosted on HF as `dgslibisey/MuSiQue` (answerable
        # only) and provides 'train' / 'validation' splits.
        self.data = load_dataset("dgslibisey/MuSiQue", split=split)
        # Filter to answerable rows only (some splits include unanswerable
        # examples — those are useful but a different framing). Many
        # downstream papers report on the answerable subset.
        if "answerable" in self.data.column_names:
            self.data = self.data.filter(lambda ex: ex["answerable"])
        print(f"[data v1h]   {len(self.data):,} MuSiQue {split} examples (answerable)")

    def _tokenize_paragraph(self, title: str, text: str) -> list:
        full = (title.strip() + " " + text.strip()).strip()
        return self.tokenizer(full, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _sample_distractor_paragraph(self, rng, avoid_idx: int) -> list:
        n = len(self.data)
        for _ in range(8):
            j = rng.randint(0, n - 1)
            if j == avoid_idx:
                continue
            ex_j = self.data[j]
            ps = ex_j["paragraphs"]
            if not ps:
                continue
            p = rng.choice(ps)
            return self._tokenize_paragraph(p["title"], p["paragraph_text"])
        return []

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 23 + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        FILL_THRESHOLD = 0.95
        MAX_TOPUP_TRIES = 200

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            paragraphs = ex["paragraphs"]
            # is_supporting flags supporting paragraphs (guaranteed include).
            own_tok = [self._tokenize_paragraph(p["title"], p["paragraph_text"])
                       for p in paragraphs]
            is_support = [bool(p["is_supporting"]) for p in paragraphs]

            # Pack supports first; skip example if any support won't fit
            # (would yield an unsolvable QA pair otherwise).
            packed = []
            cur_total = 0
            sep_cost = 1
            order = sorted(range(len(own_tok)),
                           key=lambda i: not is_support[i])
            example_ok = True
            for i in order:
                tk = own_tok[i]
                add = len(tk) + (sep_cost if packed else 0)
                if cur_total + add > self.chunk_size:
                    if is_support[i]:
                        example_ok = False
                        break
                    continue
                packed.append(tk)
                cur_total += add
            if not example_ok:
                continue  # outer while — resample

            tries = 0
            while (cur_total < FILL_THRESHOLD * self.chunk_size
                   and tries < MAX_TOPUP_TRIES):
                tries += 1
                tk = self._sample_distractor_paragraph(rng, avoid_idx=idx)
                if not tk:
                    continue
                add = len(tk) + sep_cost
                if cur_total + add > self.chunk_size:
                    continue
                insert_at = rng.randint(0, len(packed))
                packed.insert(insert_at, tk)
                cur_total += add

            ctx_tokens, valid_len = _pack_context(
                packed, chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id, pad_token_id=self.pad_token_id,
            )

            q_ids = tok(ex["question"], add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_text = ex["answer"] or ""
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            content_mask = [True] * len(a_ids)
            aliases = ex.get("answer_aliases") or []
            answer_refs = [a_text] + [a for a in aliases if a and a != a_text]

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "musique",
                "question_type": "multi_hop_ans",
                "answer_refs": answer_refs,
            }


# ═══════════════════════════════════════════════════════════════════════════
# BABILong — synthetic long-context state-tracking (bAbI tasks + filler)
# ═══════════════════════════════════════════════════════════════════════════


class BABILongDataset(IterableDataset):
    """BABILong (Kuratov et al., 2024). bAbI tasks (qa1-qa20) with the
    relevant fact sentences scattered through a long filler context
    (PG-19 book excerpts). Synthetic = contamination-free; tests pure
    state-tracking / multi-fact composition. Length is parameterized
    ('4k', '8k', '16k', etc.) — naturally matches our tranches.

    Each example: `input` (already-formatted long context), `question`,
    `target` (short answer). No packing needed — pre-formatted length.
    """

    # 20 bAbI tasks. qa1=single-supporting-fact, qa2/qa3=two/three-supporting,
    # qa4=two-arg relations, etc. Mix of all 20 gives full coverage.
    DEFAULT_TASKS = [f"qa{i}" for i in range(1, 21)]

    def __init__(
        self,
        split: str,                          # "train" or "validation" (mapped below)
        tokenizer,
        chunk_size: int = 4096,
        config_name: str = "4k",             # "0k", "1k", "2k", "4k", "8k", "16k"
        tasks: Optional[list] = None,        # None = use all 20
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.config_name = config_name
        self.tasks = list(tasks) if tasks is not None else list(self.DEFAULT_TASKS)
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        # BABILong on HF is `RMT-team/babilong` with configs per length and
        # one split per task (qa1-qa20). We pool tasks for diversity and
        # partition each task by deterministic index hash into train/val
        # so train and val don't sample the same examples (audit fix #1).
        # Convention: examples whose index % 5 == 0 are val, else train.
        # 80/20 split, deterministic, no shuffle dependence.
        print(f"[data v1h] loading BABILong config={config_name}, {len(self.tasks)} tasks "
              f"(downloads ~per-task on first call)...")
        VAL_MOD = 5
        is_val = split not in ("train",)
        self.task_data = {}
        for task in self.tasks:
            try:
                ds_full = load_dataset("RMT-team/babilong", config_name, split=task)
            except Exception as e:
                print(f"[data v1h]   skipping {task} ({type(e).__name__}: {e})")
                continue
            # Partition by index for clean train/val separation.
            if is_val:
                keep_idx = [i for i in range(len(ds_full)) if i % VAL_MOD == 0]
            else:
                keep_idx = [i for i in range(len(ds_full)) if i % VAL_MOD != 0]
            if not keep_idx:
                print(f"[data v1h]   {task}: 0 rows after {split} partition — skipping")
                continue
            ds = ds_full.select(keep_idx)
            if len(ds) == 0:
                print(f"[data v1h]   {task}: empty after select — skipping (audit #6)")
                continue
            self.task_data[task] = ds
        total = sum(len(d) for d in self.task_data.values())
        print(f"[data v1h]   {total:,} total BABILong examples across "
              f"{len(self.task_data)} tasks at {config_name} ({split} partition)")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 31 + wid * 100_003)
        tok = self.tokenizer
        task_list = list(self.task_data.keys())
        if not task_list:
            raise RuntimeError("BABILong: no tasks loaded — check config_name")

        while True:
            task = rng.choice(task_list)
            ds = self.task_data[task]
            idx = rng.randint(0, len(ds) - 1)
            ex = ds[idx]
            ctx_text = ex["input"]
            q_text = ex["question"]
            a_text = (ex["target"] or "").strip()

            # Tokenize and truncate context to chunk_size (BABILong is
            # already approximately our chunk length per its config name)
            ctx_ids = tok(ctx_text, add_special_tokens=False,
                          return_attention_mask=False)["input_ids"]
            if len(ctx_ids) > self.chunk_size:
                ctx_ids = ctx_ids[:self.chunk_size]
            valid_len = len(ctx_ids)
            if valid_len < self.chunk_size:
                ctx_ids = ctx_ids + [self.pad_token_id] * (self.chunk_size - valid_len)

            q_ids = tok(q_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            content_mask = [True] * len(a_ids)

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "answer_refs": [a_text],   # for eval scoring (EM/F1)
                "task_family": f"babilong_{task}",
                "question_type": task,
            }


# ═══════════════════════════════════════════════════════════════════════════
# RULER — synthetic multi-key needle-in-a-haystack (EVAL-ONLY, OOD)
# ═══════════════════════════════════════════════════════════════════════════


class RULERNIAHDataset(IterableDataset):
    """RULER-style multi-key needle-in-a-haystack (Hsieh et al., 2024, NVIDIA).

    EVAL-ONLY out-of-distribution probe — never trained on. K distinct "magic
    number" needles are hidden in benign filler; given one key, the model must
    retrieve its value. Fully synthetic → contamination-free (Llama cannot know
    the random values); tests pure associative recall at long context. The
    answer is the exact value, so EM/F1 are clean (no echo/abstraction).
    """

    _FILLER = [
        "The afternoon light moved slowly across the wooden floor.",
        "A train passed somewhere far away, and then the quiet returned.",
        "He counted the steps out of habit, not because it mattered.",
        "Rain had been promised all week, but the clouds never broke.",
        "The market was busy with the ordinary errands of a Tuesday.",
        "She folded the letter twice and left it on the table.",
        "Outside, the wind turned the pages of a forgotten newspaper.",
        "They spoke of small things, the weather and the price of bread.",
        "A door closed somewhere upstairs and the house settled again.",
        "The river was low that summer and the stones showed through.",
        "He sharpened the pencil to a fine point and set it down.",
        "The cat watched the birds without any real intention.",
        "Lamps came on one by one along the length of the street.",
        "Someone was practicing scales on a piano two floors below.",
        "The kettle clicked off and the steam thinned into nothing.",
        "Leaves gathered in the corner of the yard near the fence.",
        "The bus was late, as it usually was on rainy mornings.",
        "She traced the rim of the cup with one absent finger.",
        "The clock in the hall had been five minutes slow for years.",
        "A gull wheeled once over the harbor and was gone.",
        "The bookshelf sagged a little under its uneven weight.",
        "He left the radio on low, more for company than for news.",
        "The path curved away under the bare and patient trees.",
        "Footsteps crossed the ceiling and then went quiet.",
    ]
    _ADJ = ["silver", "quiet", "ancient", "golden", "hollow", "crimson",
            "northern", "restless", "frozen", "gentle", "distant", "amber",
            "velvet", "iron", "scarlet", "winding"]
    _NOUN = ["harbor", "lantern", "meadow", "compass", "willow", "cipher",
             "falcon", "garden", "ferry", "beacon", "orchard", "summit",
             "quartz", "river", "thicket", "anchor"]

    def __init__(
        self,
        split: str,                 # accepted for interface parity; ignored (synthetic)
        tokenizer,
        chunk_size: int = 8192,
        n_needles: int = 1,         # single needle: 4-needle/64:1 is unwinnable for compressive memory
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.n_needles = n_needles
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

    def _key(self, rng) -> str:
        return f"{rng.choice(self._ADJ)}-{rng.choice(self._NOUN)}"

    def __iter__(self) -> Iterator[dict]:
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        rng = random.Random(self.seed + 911 + wid * 100_003)
        tok = self.tokenizer

        def _tok(s: str) -> list:
            return tok(s, add_special_tokens=False,
                       return_attention_mask=False)["input_ids"]

        while True:
            # Ordered list (not a set) so key order is deterministic regardless
            # of PYTHONHASHSEED.
            keys, _seen = [], set()
            while len(keys) < self.n_needles:
                k = self._key(rng)
                if k not in _seen:
                    _seen.add(k)
                    keys.append(k)
            vals = [rng.randint(1_000_000, 9_999_999) for _ in keys]
            needles = [_tok(f"The special magic number for {k} is {v}.")
                       for k, v in zip(keys, vals)]
            needle_total = sum(len(n) for n in needles) + len(needles)

            # Fill benign filler up to ~chunk_size minus the needle budget, so
            # the needles fit without truncation. Guard against OVERSHOOT: a
            # final filler unit (~10-15 tok) could push the total past chunk_size
            # and clip the tail — which, if a needle landed last, would drop the
            # answer from the context. Stop before exceeding budget so the
            # post-insertion total stays ≤ chunk_size and no needle is truncated.
            budget = self.chunk_size - needle_total - 8
            filler_units, acc = [], 0
            while acc < budget:
                u = _tok(rng.choice(self._FILLER))
                if acc + len(u) + 1 > budget:
                    break
                filler_units.append(u)
                acc += len(u) + 1

            # Insert the needles at distinct random positions among the filler.
            n_slots = len(filler_units) + 1
            positions = sorted(rng.sample(range(n_slots),
                                          min(self.n_needles, n_slots)))
            units, ni = [], 0
            for i in range(n_slots):
                while ni < len(positions) and positions[ni] == i:
                    units.append(needles[ni])
                    ni += 1
                if i < len(filler_units):
                    units.append(filler_units[i])

            ctx_ids: list = []
            for u in units:
                ctx_ids.extend(u)
                ctx_ids.append(self.sep_token_id)
            ctx_ids = ctx_ids[:self.chunk_size]
            valid_len = len(ctx_ids)
            if valid_len < self.chunk_size:
                ctx_ids = ctx_ids + [self.pad_token_id] * (self.chunk_size - valid_len)

            # Target only from needles actually inserted (small chunks may fit
            # fewer than n_needles), so the answer is always present in context.
            target = rng.randrange(len(positions))
            # Explicit instruction: without it the Instruct model refuses the
            # unusual "magic number" framing ("I'm not aware of any...") and the
            # whole task scores 0 regardless of memory (verified 2026-05-30).
            q_ids = _tok(f"What is the special magic number for {keys[target]}? "
                         f"Answer with only the number.")
            a_text = str(vals[target])
            a_ids = _tok(a_text)

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (self.chunk_size - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": [a_text],
                "task_family": "ruler_niah",
                "question_type": "niah_multikey",
            }


# ═══════════════════════════════════════════════════════════════════════════
# LoCoMo — very-long-term conversational memory (EVAL-ONLY, OOD, long-context)
# ═══════════════════════════════════════════════════════════════════════════


class LoCoMoQADataset(IterableDataset):
    """LoCoMo (Maharana et al., 2024) — very-long-term conversational memory.

    EVAL-ONLY, OOD: never trained on. 10 multi-session human-machine dialogues
    (~300 turns, ~9k–26k tokens each) with ~200 QA pairs per conversation
    across 5 categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
    5=adversarial (answer = "not mentioned" — tests knowing what you DON'T know).

    The FULL rendered conversation is the context. It exceeds 8k, so run this
    family at a larger chunk_size (e.g. 32768): the streaming encoder windows
    the whole dialogue into the fixed-footprint O(1) memory. This is a
    length-generalization probe — trained at 8k, evaluated to ~26k, exactly the
    regime where a fixed-footprint compressor should hold up while context
    grows. Answers are short (dates/names/phrases); headline metric is the
    LLM judge (abstractive + adversarial-negative answers).

    Source: snap-research/locomo `data/locomo10.json` (cached locally).
    """

    _URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
            "main/data/locomo10.json")
    _CACHE = REPO_ROOT / "data/eval/locomo10.json"
    _CAT_NAME = {1: "multihop", 2: "temporal", 3: "open_domain",
                 4: "single_hop", 5: "adversarial"}

    def __init__(
        self,
        split: str,                          # ignored (eval-only, all 10 convs)
        tokenizer,
        chunk_size: int = 24576,             # max LoCoMo conv ≈ 24.4k tokens
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        import json
        if not self._CACHE.exists():
            import urllib.request
            self._CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"[data v1h] downloading LoCoMo → {self._CACHE} ...")
            urllib.request.urlretrieve(self._URL, self._CACHE)
        with open(self._CACHE) as f:
            raw = json.load(f)

        # Pre-render + pre-tokenize each conversation ONCE (10 of them). Build a
        # flat (conv_idx, qa) work-list so each QA pair shares its conversation's
        # cached token list.
        tok = tokenizer
        self._conv_ids: list[list[int]] = []
        self._qa: list[tuple[int, dict]] = []
        n_trunc = 0
        for ci, sample in enumerate(raw):
            text = self._render_conversation(sample["conversation"])
            ids = tok(text, add_special_tokens=False,
                      return_attention_mask=False)["input_ids"]
            if len(ids) > chunk_size:
                n_trunc += 1
            self._conv_ids.append(ids)
            for qa in sample.get("qa", []):
                if qa.get("question") is None or qa.get("answer") is None:
                    continue
                self._qa.append((ci, qa))
        lens = [len(x) for x in self._conv_ids]
        print(f"[data v1h]   LoCoMo: {len(raw)} conversations, {len(self._qa)} QA; "
              f"conv tokens min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}; "
              f"{n_trunc} conv(s) exceed chunk_size={chunk_size} (truncated)")
        if n_trunc:
            print(f"[data v1h]   WARN: {n_trunc} conversation(s) > {chunk_size} "
                  f"tokens — run LoCoMo with a larger --chunk-size to avoid "
                  f"dropping late-session evidence.")

    def _render_conversation(self, conv: dict) -> str:
        """Render all sessions in numeric order as dated speaker turns."""
        sess_ids = sorted(
            (int(k.split("_")[1]) for k in conv
             if k.startswith("session_") and not k.endswith("date_time")),
        )
        lines: list[str] = []
        for sid in sess_ids:
            turns = conv.get(f"session_{sid}")
            if not turns:
                continue
            date = conv.get(f"session_{sid}_date_time", "")
            lines.append(f"[Session {sid} — {date}]")
            for t in turns:
                spk = t.get("speaker", "")
                txt = (t.get("text") or "").strip()
                cap = t.get("blip_caption")
                if cap:
                    txt = f"{txt} [shares an image: {cap}]".strip()
                lines.append(f"{spk}: {txt}")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 4127 + wid * 100_003)
        tok = self.tokenizer
        order = list(range(len(self._qa)))
        rng.shuffle(order)

        cs = self.chunk_size
        pos = 0
        while True:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            ci, qa = self._qa[order[pos]]
            pos += 1

            ctx_ids = list(self._conv_ids[ci])
            if len(ctx_ids) > cs:
                ctx_ids = ctx_ids[:cs]
            valid_len = len(ctx_ids)
            if valid_len < cs:
                ctx_ids = ctx_ids + [self.pad_token_id] * (cs - valid_len)

            q_text = str(qa["question"])
            a_text = str(qa["answer"]).strip()
            cat = qa.get("category")
            q_ids = tok(q_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (cs - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": [a_text],
                "task_family": "locomo",
                "question_type": self._CAT_NAME.get(cat, f"cat{cat}"),
            }


# ═══════════════════════════════════════════════════════════════════════════
# Mixed sampler — composite_v1 + HotpotQA + NarrativeQA + MuSiQue + BABILong
# ═══════════════════════════════════════════════════════════════════════════


class MixedQADataset(IterableDataset):
    """Weighted-sample from multiple QA sources. Each `__next__` picks a
    source according to `weights` and yields one example from it."""

    def __init__(
        self,
        sources: list[IterableDataset],
        weights: list[float],
        seed: int = 0,
    ):
        super().__init__()
        assert len(sources) == len(weights), "sources/weights must align"
        self.sources = sources
        self.weights = list(weights)
        self.seed = seed

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 7919 + wid * 100_003)
        iters = [iter(s) for s in self.sources]
        while True:
            idx = rng.choices(range(len(self.sources)), weights=self.weights, k=1)[0]
            try:
                yield next(iters[idx])
            except StopIteration:
                iters[idx] = iter(self.sources[idx])
                yield next(iters[idx])


def make_mixed_qa_dataloader(
    cfg: ReprConfig,
    tokenizer,
    *,
    composite_passages_path: Optional[Path | str],
    composite_questions_path: Optional[Path | str],
    use_hotpot: bool,
    use_narrative: bool,
    use_musique: bool = False,
    use_babilong: bool = False,
    babilong_config: str = "4k",            # BABILong length-config to use
    split: str = "train",                    # used for HotpotQA/NarrativeQA/MuSiQue
    chunk_size: int = 4096,
    passages_per_chunk: int = 300,           # composite_v1 only
    weights: tuple = (0.5, 0.25, 0.25, 0.0, 0.0),  # (composite, hotpot, narrative, musique, babilong)
    composite_task_weights: Optional[dict[str, float]] = None,  # per-family inside composite_v1
    num_workers: int = 0,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Build a mixed QA dataloader over up to 5 sources: composite_v1,
    HotpotQA, NarrativeQA, MuSiQue, BABILong. Any source with flag=False
    or weight≤0 is skipped (saves data-load cost).

    weights is a 5-tuple. Older 3-tuple callers are accepted with the
    missing entries defaulted to 0.
    """
    # Back-compat: pad weights tuple to length 5
    weights = tuple(weights) + (0.0,) * (5 - len(weights))

    sources, src_weights, names = [], [], []

    if (composite_passages_path is not None
            and composite_questions_path is not None
            and weights[0] > 0):
        sources.append(QADataset(
            Path(composite_passages_path), Path(composite_questions_path),
            chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            task_weights=composite_task_weights,
            seed=seed,
        ))
        src_weights.append(weights[0]); names.append("composite_v1")

    if use_hotpot and weights[1] > 0:
        hp_split = "train" if split == "train" else "validation"
        sources.append(HotpotQADataset(
            split=hp_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 1,
        ))
        src_weights.append(weights[1]); names.append("hotpot_qa")

    if use_narrative and weights[2] > 0:
        nq_split = "train" if split == "train" else "validation"
        sources.append(NarrativeQADataset(
            split=nq_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 2,
        ))
        src_weights.append(weights[2]); names.append("narrative_qa")

    if use_musique and weights[3] > 0:
        mq_split = "train" if split == "train" else "validation"
        sources.append(MuSiQueDataset(
            split=mq_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 3,
        ))
        src_weights.append(weights[3]); names.append("musique")

    if use_babilong and weights[4] > 0:
        sources.append(BABILongDataset(
            split=split, tokenizer=tokenizer, chunk_size=chunk_size,
            config_name=babilong_config,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 4,
        ))
        src_weights.append(weights[4]); names.append(f"babilong_{babilong_config}")

    if not sources:
        raise ValueError("MixedQA needs at least one source enabled")

    print(f"[data v1h] mixed sources: {list(zip(names, src_weights))}")
    ds = MixedQADataset(sources, src_weights, seed=seed)
    pad_id = cfg.pad_token_id
    return DataLoader(
        ds,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        num_workers=num_workers,
        collate_fn=lambda s: collate_qa(s, pad_token_id=pad_id),
        pin_memory=torch.cuda.is_available(),
    )
