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
        # CompositeSampler is already seeded; just keep emitting chunks
        while True:
            chunk = self.sampler.sample_chunk(self.passages_per_chunk)
            passages = chunk["passages"]
            target_q = chunk["target_question"]

            # Pack passages into a single context up to chunk_size tokens.
            # The evidence may or may not all fit if chunk_size is tight.
            # Order: keep the sampler's shuffle (positional fairness already
            # handled inside sample_chunk).
            ctx_tokens, valid_len = _pack_context(
                [p["passage_token_ids"] for p in passages],
                chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
            )

            q_ids = list(target_q["question_token_ids"])
            a_ids = list(target_q["answer_token_ids"])
            content_positions = list(
                target_q.get("answer_content_token_positions") or []
            )
            # If no content positions specified, fall back to the full
            # answer span (the trainer's default behavior in v4/v5).
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

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid * 100_003)
        n = len(self.data)
        tok = self.tokenizer

        while True:
            idx = rng.randint(0, n - 1)
            ex = self.data[idx]
            # ex["context"] has parallel lists `title` and `sentences` (one
            # sentence-list per paragraph). Concat title + sentences for each.
            ctx = ex["context"]
            paragraphs = []
            for title, sents in zip(ctx["title"], ctx["sentences"]):
                paragraphs.append(title.strip() + " " + " ".join(s.strip() for s in sents))

            # Tokenize each paragraph; concat with sep_token_id.
            tok_paragraphs = [
                tok(p, add_special_tokens=False, return_attention_mask=False)["input_ids"]
                for p in paragraphs
            ]
            ctx_tokens, valid_len = _pack_context(
                tok_paragraphs,
                chunk_size=self.chunk_size,
                sep_token_id=self.sep_token_id,
                pad_token_id=self.pad_token_id,
            )

            q_ids = tok(ex["question"], add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]
            a_text = ex["answer"]
            a_ids = tok(a_text, add_special_tokens=False,
                         return_attention_mask=False)["input_ids"]

            # Whole answer is content (no filler distinction for HotpotQA)
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

            # Try to find the answer span in the document tokens
            a_ids_full = tok(a_text, add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]
            anchor_pos = -1
            if len(a_ids_full) > 0 and len(a_ids_full) <= 40:
                # Naive substring search on token list
                target = a_ids_full
                for i in range(0, len(doc_ids) - len(target) + 1, 1):
                    if doc_ids[i:i + len(target)] == target:
                        anchor_pos = i
                        break

            if anchor_pos >= 0:
                # Center the window on the answer (with some randomness)
                half = cs // 2
                jitter = rng.randint(-half // 4, half // 4)
                start = max(0, anchor_pos + len(a_ids_full) // 2 - half + jitter)
                start = min(start, max(0, len(doc_ids) - cs))
            else:
                # Answer not found in tokens (paraphrase or oov tokenization).
                # Random window — model still tries to answer from partial context.
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
# Mixed sampler — composite_v1 + HotpotQA + NarrativeQA at weighted ratios
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
    split: str = "train",                    # used for HotpotQA/NarrativeQA
    chunk_size: int = 4096,
    passages_per_chunk: int = 300,           # composite_v1 only
    weights: tuple = (0.5, 0.25, 0.25),      # (composite, hotpot, narrative)
    num_workers: int = 0,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Build a mixed QA dataloader. Any of (composite, hotpot, narrative)
    can be disabled by passing `None` paths or `False` flags; the weights
    for disabled sources are dropped and the remaining weights renormalize."""
    sources, src_weights, names = [], [], []

    if composite_passages_path is not None and composite_questions_path is not None:
        sources.append(QADataset(
            Path(composite_passages_path), Path(composite_questions_path),
            chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed,
        ))
        src_weights.append(weights[0]); names.append("composite_v1")

    if use_hotpot:
        hp_split = "train" if split == "train" else "validation"
        sources.append(HotpotQADataset(
            split=hp_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 1,
        ))
        src_weights.append(weights[1]); names.append("hotpot_qa")

    if use_narrative:
        nq_split = "train" if split == "train" else "validation"
        sources.append(NarrativeQADataset(
            split=nq_split, tokenizer=tokenizer, chunk_size=chunk_size,
            pad_token_id=cfg.pad_token_id, seed=seed + 2,
        ))
        src_weights.append(weights[2]); names.append("narrative_qa")

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
