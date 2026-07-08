"""LongMemEval — the headline long-term chat-memory benchmark (EVAL-ONLY).

EVAL reader. HF-auto-downloaded on first use (`xiaowu0162/longmemeval-cleaned`,
falling back to the deprecated `xiaowu0162/longmemeval` if the cleaned repo is
unreachable); cached by `huggingface_hub` under `~/.cache/huggingface`. No
build/ingest step and no `data/longmemeval/` on disk. See DATASETS.md.

500 questions, each keyed to its OWN multi-session chat history (haystack) —
unlike LoCoMo (10 shared conversations x ~200 questions), every LongMemEval
question has a private haystack, so there is no cross-question context reuse.
Three variants (`variant=`):
  - "s"      (default) — ~40-60 sessions/example, ~115k tokens rendered.
  - "m"      — ~500 sessions/example, ~1.5M tokens rendered (needs a very
               large `chunk_size`, or heavy truncation — mostly useful for
               streaming/windowed evaluation, not single-shot prepend).
  - "oracle" — only the answer-bearing (+ a couple of adjacent) sessions per
               example; a cheap sanity/ceiling check, NOT the real haystack task.

Question types map onto the paper's 5-way taxonomy (`question_type`):
single-session (merges the raw single-session-user/-assistant/-preference
splits), multi-session, temporal (raw: temporal-reasoning), knowledge-update,
and abstention — the last is NOT its own raw `question_type` value; ~30/500
questions carry a `_abs` suffix on `question_id` and an "insufficient
information" gold answer, so we detect and relabel those explicitly (an
abstention question that gets bucketed under e.g. "single-session" would
silently vanish from that slice's numbers).
"""
from __future__ import annotations

import json
import random
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa

_REPO_CLEANED = "xiaowu0162/longmemeval-cleaned"
_REPO_ORIGINAL = "xiaowu0162/longmemeval"
_FILES_CLEANED = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
_FILES_ORIGINAL = {
    # the original (deprecated) repo hosts these with NO .json suffix.
    "s": "longmemeval_s",
    "m": "longmemeval_m",
    "oracle": "longmemeval_oracle",
}

# raw `question_type` -> paper's 5-way canonical bucket.
_CANON_TYPE = {
    "single-session-user": "single-session",
    "single-session-assistant": "single-session",
    "single-session-preference": "single-session",
    "multi-session": "multi-session",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge-update",
}


def _load_raw(variant: str) -> list:
    """Download + parse one LongMemEval split. Tries the maintained
    `-cleaned` repo first, falls back to the original (deprecated but
    still-hosted) repo, and raises a clear error if neither is reachable."""
    from huggingface_hub import hf_hub_download

    errors = []
    for repo, files in ((_REPO_CLEANED, _FILES_CLEANED), (_REPO_ORIGINAL, _FILES_ORIGINAL)):
        fname = files[variant]
        try:
            print(f"[data v1h] LongMemEval: fetching {repo}/{fname} ...")
            path = hf_hub_download(repo, filename=fname, repo_type="dataset")
            with open(path) as f:
                return json.load(f)
        except Exception as e:  # noqa: BLE001 — try next source, report all at the end
            errors.append(f"{repo}/{fname}: {type(e).__name__}: {e}")
            print(f"[data v1h]   unavailable ({type(e).__name__}: {e}); trying fallback...")
    raise RuntimeError(
        "LongMemEval: could not load variant "
        f"{variant!r} from either HF repo. Tried:\n  " + "\n  ".join(errors) +
        "\nCheck network access / HF Hub reachability."
    )


class LongMemEvalDataset(IterableDataset):
    """LongMemEval (Wu et al., 2024). Multi-session synthetic chat histories
    (haystack_sessions) built by interleaving ShareGPT/UltraChat sessions with
    the answer-bearing session(s); question + short gold answer follow.

    The FULL rendered chat history is the context (all sessions, in order,
    each tagged with its session id + timestamp so temporal-reasoning and
    knowledge-update questions — which hinge on WHEN something was said, or
    which of several conflicting statements is the LATEST — stay answerable).
    """

    def __init__(
        self,
        split: str,                          # ignored (eval-only, fixed 500-question set)
        tokenizer,
        chunk_size: int = 131_072,           # _S renders to ~115k tok; _M needs far more
        variant: str = "s",                  # "s" (default) | "m" | "oracle"
        max_examples: Optional[int] = None,  # bound-load: sample only the first N (None = all 500)
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        if variant not in ("s", "m", "oracle"):
            raise ValueError(f"LongMemEval: variant must be 's'/'m'/'oracle', got {variant!r}")
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.variant = variant
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        raw = _load_raw(variant)
        if max_examples is not None:
            raw = raw[:max_examples]

        tok = tokenizer
        self._ctx_ids: list[list[int]] = []
        self._meta: list[dict] = []
        n_trunc = 0
        for ex in raw:
            text = self._render(ex)
            ids = tok(text, add_special_tokens=False,
                      return_attention_mask=False)["input_ids"]
            if len(ids) > chunk_size:
                n_trunc += 1
            self._ctx_ids.append(ids)

            qid = str(ex.get("question_id", ""))
            raw_type = ex.get("question_type", "")
            if qid.endswith("_abs"):
                qtype = "abstention"
            else:
                qtype = _CANON_TYPE.get(raw_type, raw_type or "unknown")
            self._meta.append({
                "question": str(ex.get("question", "")),
                "answer": str(ex.get("answer", "")),
                "question_type": qtype,
            })

        lens = [len(x) for x in self._ctx_ids]
        n = len(self._ctx_ids)
        print(f"[data v1h]   LongMemEval[{variant}]: {n} examples; "
              f"rendered-context tokens min/mean/max = "
              f"{min(lens) if lens else 0}/{(sum(lens)//n) if n else 0}/{max(lens) if lens else 0}; "
              f"{n_trunc} example(s) exceed chunk_size={chunk_size} (truncated)")
        if n_trunc:
            print(f"[data v1h]   WARN: {n_trunc} example(s) > {chunk_size} tokens — "
                  f"a larger --chunk-size may be needed (esp. for variant='m').")
        if n == 0:
            raise RuntimeError(f"LongMemEval[{variant}]: 0 examples loaded — check max_examples/data source.")

    @staticmethod
    def _render(ex: dict) -> str:
        """Render every haystack session in order, dated + speaker-tagged, so
        the model can (in principle) resolve temporal / knowledge-update
        questions from the rendered text alone."""
        sessions = ex.get("haystack_sessions") or []
        dates = ex.get("haystack_dates") or []
        sids = ex.get("haystack_session_ids") or []
        lines: list[str] = []
        for i, turns in enumerate(sessions):
            sid = sids[i] if i < len(sids) else i
            date = dates[i] if i < len(dates) else ""
            lines.append(f"[Session {sid} — {date}]")
            for t in (turns or []):
                role = t.get("role", "user")
                content = (t.get("content") or "").strip()
                speaker = "User" if role == "user" else "Assistant"
                if content:
                    lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 6151 + wid * 100_003)
        tok = self.tokenizer
        n = len(self._ctx_ids)
        cs = self.chunk_size
        order = list(range(n))
        rng.shuffle(order)

        pos = 0
        while True:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            idx = order[pos]
            pos += 1

            ctx_ids = list(self._ctx_ids[idx])
            if len(ctx_ids) > cs:
                ctx_ids = ctx_ids[:cs]
            valid_len = len(ctx_ids)
            if valid_len < cs:
                ctx_ids = ctx_ids + [self.pad_token_id] * (cs - valid_len)

            meta = self._meta[idx]
            q_text = meta["question"]
            a_text = meta["answer"]
            q_ids = tok(q_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            if not a_ids:  # guard: empty gold answer would zero-length the target
                a_ids = tok(" ", add_special_tokens=False,
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
                "task_family": "longmemeval",
                "question_type": meta["question_type"],
            }


def make_longmemeval_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 131_072,
    variant: str = "s",
    max_examples: Optional[int] = None,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone LongMemEval loader (EVAL-ONLY). The headline memory
    benchmark — multi-session chat history as context, short QA pairs across
    5 question types (see module docstring)."""
    ds = LongMemEvalDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                            variant=variant, max_examples=max_examples,
                            sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
