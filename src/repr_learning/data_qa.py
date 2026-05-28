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
            }


# ═══════════════════════════════════════════════════════════════════════════
# NarrativeQA — long-form QA over books and movie scripts
# ═══════════════════════════════════════════════════════════════════════════


class NarrativeQADataset(IterableDataset):
    """NarrativeQA. Each example is a question + reference answer(s) +
    a long document (a book or movie-script summary). Documents are typically
    longer than 4096 tokens, so we truncate.

    Truncation strategy: tokenize the full document, then take a 4096-token
    window. If the answer text is found in the document tokens, we center
    the window on the answer location (so the model has a chance). If not,
    we take the start of the document.

    Tradeoff: this is a "favorable" truncation that biases toward easier
    examples. We accept it because the alternative (random/uniform truncation)
    means most examples don't contain the answer at all, which makes the
    training signal mostly noise.

    First call downloads ~700MB from HuggingFace.
    """

    def __init__(
        self,
        split: str,                          # "train" or "validation"
        tokenizer,
        chunk_size: int = 4096,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        self.split = split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.seed = seed

        from datasets import load_dataset
        print(f"[data v1h] loading NarrativeQA split={split} "
              f"(downloads ~700MB on first call)...")
        self.data = load_dataset("narrativeqa", split=split, trust_remote_code=True)
        print(f"[data v1h]   {len(self.data):,} NarrativeQA {split} examples")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 17 + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer
        cs = self.chunk_size

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            doc_text = ex["document"]["text"]
            q_text = ex["question"]["text"]
            # NarrativeQA has multiple reference answers; sample one
            ans_list = ex["answers"]
            if not ans_list:
                continue
            a_text = rng.choice(ans_list)["text"]

            # Tokenize document fully (this can be slow for very long docs)
            doc_ids = tok(doc_text, add_special_tokens=False,
                           return_attention_mask=False)["input_ids"]
            a_ids_full = tok(a_text, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

            # Random window. (Prior version centered the window on a
            # substring-match of the answer in the document — that leaks
            # the answer location at training time, which is an oracle the
            # model wouldn't have at deploy time. Random window is the
            # honest baseline; pair with summary-as-context mode if a
            # "retrieval-free ceiling" is wanted.)
            start = rng.randint(0, max(0, len(doc_ids) - cs))

            ctx_tokens = doc_ids[start:start + cs]
            valid_len = len(ctx_tokens)
            if valid_len < cs:
                ctx_tokens = ctx_tokens + [self.pad_token_id] * (cs - valid_len)

            q_ids = tok(q_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_ids = a_ids_full
            content_mask = [True] * len(a_ids)

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (cs - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": content_mask,
                "task_family": "narrative_qa",
                "question_type": "narrative",
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
                "task_family": f"babilong_{task}",
                "question_type": task,
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
            pad_token_id=cfg.pad_token_id, seed=seed + 2,
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
