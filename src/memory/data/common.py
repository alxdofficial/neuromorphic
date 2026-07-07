"""Shared QA-reader core: the `(context_ids, question_ids, answer_ids, ...)`
contract used by every reader in `src.memory.data`.

Holds the base pieces that all per-dataset readers import:
- `QABatch` — the batched-example dataclass every reader/collate produces.
- `_pack_context` — concat/truncate/pad passages into a fixed-width context.
- `QADataset` — the composite `bio` reader (synthetic, 9 task families).
- `collate_qa` — stack per-sample dicts into a `QABatch`.
- `make_qa_dataloader` — DataLoader wrapper for `QADataset`.

The per-dataset readers (hotpot, narrativeqa, musique, babilong, ruler,
locomo, mixed, babi, bio, mae, continuation) live in sibling modules and
import from here. Generator/build layer for the composite `bio` data:
`scripts/data_build/generate/bio/` → stored in `data/bio/`.
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

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.data_build.common.sampler import CompositeSampler  # noqa: E402

from ..config import ReprConfig


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
    answer_refs: list = None     # length B — per-example reference answer strings (for generated
                                 # EM/F1 eval on abstractive QA); optional, dropped by loss/CE paths


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
        tokenizer,                         # LLM-agnostic: re-tokenize the TEXT fields
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
        self.tokenizer = tokenizer
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

    def _tok(self, text: str) -> list[int]:
        return self.tokenizer(text or "", add_special_tokens=False,
                              return_attention_mask=False)["input_ids"]

    def _tok_answer(self, answer_text: str, target_value: str) -> tuple[list[int], list[bool]]:
        """Tokenize the answer (LLM-agnostically) and mark the load-bearing content
        tokens = those overlapping the `target_value` char span (via offset mapping).
        Falls back to whole-answer-is-content when target_value is absent / not found /
        the tokenizer is slow (no offsets)."""
        answer_text = answer_text or ""
        if target_value and getattr(self.tokenizer, "is_fast", False):
            enc = self.tokenizer(answer_text, add_special_tokens=False,
                                 return_attention_mask=False, return_offsets_mapping=True)
            ids, offs = enc["input_ids"], enc["offset_mapping"]
            lo = answer_text.find(target_value)
            if lo >= 0:
                hi = lo + len(target_value)
                content = [(e > lo and s < hi) for (s, e) in offs]
                if any(content):
                    return ids, content
            return ids, [True] * len(ids)
        ids = self._tok(answer_text)
        return ids, [True] * len(ids)

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
            # LLM-agnostic: re-tokenize the passage TEXT with the active tokenizer
            # (NOT the baked-in Llama `passage_token_ids`).
            ev_tokens = [self._tok(p["passage"]) for p in evidence]
            ev_total = sum(len(t) for t in ev_tokens) + max(0, len(ev_tokens) - 1)
            if ev_total > self.chunk_size:
                continue  # evidence-too-big; resample

            # Greedily insert distractors at random positions until budget exhausted
            rng.shuffle(distractors)
            packed = list(ev_tokens)
            cur_total = ev_total
            for d in distractors:
                d_tokens = self._tok(d["passage"])
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

            # LLM-agnostic: re-tokenize question/answer TEXT with the active tokenizer;
            # content mask = the target_value span (via offsets), re-derived (the stored
            # answer_content_token_positions index the OLD Llama tokenization).
            answer_full = target_q.get("answer") or ""
            target_val = target_q.get("target_value") or ""
            q_ids = self._tok(target_q["question"])
            a_ids, content_mask = self._tok_answer(answer_full, target_val)
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
        assert len(cm) == ta, (                              # fail loud on a malformed sample
            f"answer_content_mask_list length {len(cm)} != answer_ids length {ta} "
            f"(task={s.get('task_family')!r}); a length-mismatched mask silently mis-scores loss.")
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
        answer_refs=[s.get("answer_refs", []) for s in samples],
    )


def make_qa_dataloader(
    cfg: ReprConfig,
    tokenizer,
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
        tokenizer=tokenizer,
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
