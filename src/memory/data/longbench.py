"""LongBench — 21-subtask long-context understanding benchmark (EVAL-ONLY),
with an optional LongBench-v2 (503 long-context MCQs) add-on.

EVAL reader. HF-hosted (`THUDM/LongBench`) as a single `data.zip` of
per-subtask JSONL files behind a now-unsupported `datasets` loading script
(`trust_remote_code` dataset scripts were removed in `datasets>=3`), so we
fetch `data.zip` directly via `huggingface_hub.hf_hub_download` (cached) and
parse the JSONL ourselves. `THUDM/LongBench-v2` (`data.json`, 503 MCQs, no
loading script) is fetched the same way and folded in as one more "subtask"
when requested. No build/ingest step and no `data/longbench/` on disk. See
DATASETS.md.
"""
from __future__ import annotations

import json
import random
import zipfile
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import _pack_context, collate_qa

_REPO_V1 = "THUDM/LongBench"
_REPO_V2 = "THUDM/LongBench-v2"

# The 21 primary LongBench-v1 subtasks (excludes the "_e" length-balanced
# resamples of the same tasks, which are an appendix-only setting).
ALL_TASKS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader",
    "gov_report", "qmsum", "multi_news", "vcsum",
    "trec", "triviaqa", "samsum", "lsht",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh",
    "lcc", "repobench-p",
]
# Default = the English QA subtasks (single- + multi-hop + long-doc), the
# most directly comparable to our other QA readers (hotpot/musique/narrativeqa).
DEFAULT_SUBSET = ["hotpotqa", "2wikimqa", "musique", "narrativeqa", "qasper", "multifieldqa_en"]
_V2_TAG = "longbench_v2"


def _load_v1_task(task: str) -> list[dict]:
    from huggingface_hub import hf_hub_download
    zpath = hf_hub_download(_REPO_V1, filename="data.zip", repo_type="dataset")
    with zipfile.ZipFile(zpath) as z:
        fn = f"data/{task}.jsonl"
        try:
            raw = z.read(fn).decode("utf-8")
        except KeyError as e:
            raise KeyError(
                f"LongBench: subtask {task!r} not found in data.zip "
                f"(valid tasks: {ALL_TASKS})"
            ) from e
    rows = []
    for line in raw.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _load_v2() -> list[dict]:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(_REPO_V2, filename="data.json", repo_type="dataset")
    with open(path) as f:
        return json.load(f)


class LongBenchDataset(IterableDataset):
    """LongBench (Bai et al., 2023) — bilingual long-context understanding
    across 21 subtasks (single/multi-doc QA, summarization, few-shot,
    synthetic retrieval, code completion). Each row already carries its own
    `context` + `input` (question) + `answers` (list of acceptable refs); we
    truncate `context` to `chunk_size` (front-truncate, matching every other
    reader in this package) and pass `input`/`answers` through unchanged.

    Optionally folds in LongBench-v2 (Bai et al., 2025) — 503 harder, longer
    (avg ~150k-word) multiple-choice questions across 6 domains — as an extra
    "subtask" named `longbench_v2` when requested via `subset`. MCQs are
    rendered as the question + 4 lettered choices; gold = the letter.
    """

    def __init__(
        self,
        split: str,                          # accepted for interface parity; LongBench ships one "test" split
        tokenizer,
        chunk_size: int = 8192,
        subset: Optional[list] = None,       # None -> DEFAULT_SUBSET; pass ALL_TASKS for everything;
                                              # include "longbench_v2" to add the v2 MCQ set
        max_examples_per_task: Optional[int] = None,  # bound-load per subtask (None = all)
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
        subset = list(subset) if subset is not None else list(DEFAULT_SUBSET)

        # rows: list of (task_name, question_text, context_text, answer_refs)
        self._rows: list[tuple[str, str, str, list]] = []
        for task in subset:
            if task == _V2_TAG:
                print(f"[data v1h] LongBench: fetching LongBench-v2 (data.json, 503 MCQs)...")
                try:
                    v2_rows = _load_v2()
                except Exception as e:
                    print(f"[data v1h]   LongBench-v2 unavailable ({type(e).__name__}: {e}) — skipping")
                    continue
                if max_examples_per_task is not None:
                    v2_rows = v2_rows[:max_examples_per_task]
                for ex in v2_rows:
                    letters = "ABCD"
                    choice_lines = "\n".join(
                        f"{L}) {ex.get('choice_' + L, '')}" for L in letters
                    )
                    q = (f"{ex['question']}\n{choice_lines}\n"
                         f"Answer with only the letter (A, B, C, or D).")
                    ans = str(ex.get("answer", "")).strip()
                    self._rows.append((_V2_TAG, q, ex.get("context", ""), [ans]))
                print(f"[data v1h]   LongBench-v2: {len(v2_rows)} rows")
                continue

            if task not in ALL_TASKS:
                raise ValueError(f"LongBench: unknown subtask {task!r} (valid: {ALL_TASKS + [_V2_TAG]})")
            print(f"[data v1h] LongBench: fetching data.zip (subtask={task}, ~114MB shared archive, "
                  f"cached after first subtask)...")
            try:
                task_rows = _load_v1_task(task)
            except Exception as e:
                raise RuntimeError(
                    f"LongBench: failed to load subtask {task!r} from {_REPO_V1} "
                    f"(check network / HF Hub reachability): {type(e).__name__}: {e}"
                ) from e
            if max_examples_per_task is not None:
                task_rows = task_rows[:max_examples_per_task]
            for ex in task_rows:
                answers = list(ex.get("answers") or [])
                self._rows.append((task, ex["input"], ex["context"], answers))
            print(f"[data v1h]   LongBench/{task}: {len(task_rows)} rows")

        if not self._rows:
            raise RuntimeError("LongBench: 0 rows loaded — check subset / network access.")
        print(f"[data v1h]   LongBench total: {len(self._rows)} rows across "
              f"{len(subset)} subtask(s)")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 733 + wid * 100_003)
        tok = self.tokenizer
        cs = self.chunk_size
        order = list(range(len(self._rows)))
        rng.shuffle(order)

        pos = 0
        while True:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            task, q_text, ctx_text, answer_refs = self._rows[order[pos]]
            pos += 1

            ctx_ids = tok(ctx_text, add_special_tokens=False,
                          return_attention_mask=False)["input_ids"]
            ctx_tokens, valid_len = _pack_context(
                [ctx_ids], chunk_size=cs,
                sep_token_id=self.sep_token_id, pad_token_id=self.pad_token_id,
            )

            q_ids = tok(q_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            a_text = answer_refs[0] if answer_refs else ""
            a_ids = tok(a_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            if not a_ids:
                a_ids = tok(" ", add_special_tokens=False,
                            return_attention_mask=False)["input_ids"]

            yield {
                "context_ids": torch.tensor(ctx_tokens, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (cs - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": answer_refs if answer_refs else [a_text],
                "task_family": "longbench",
                "question_type": task,
            }


def make_longbench_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 8192,
    subset: Optional[list] = None,
    max_examples_per_task: Optional[int] = None,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone LongBench loader (EVAL-ONLY). Default subset = the English
    QA subtasks (hotpotqa/2wikimqa/musique/narrativeqa/qasper/multifieldqa_en);
    pass `subset=ALL_TASKS` for all 21, or include "longbench_v2" to add the
    LongBench-v2 MCQ set."""
    ds = LongBenchDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                          subset=subset, max_examples_per_task=max_examples_per_task,
                          sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
