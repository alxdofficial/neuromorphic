"""∞Bench (InfiniteBench) — ultra-long-context synthetic recall + real QA
(EVAL-ONLY).

EVAL reader. HF-auto-downloaded (`xinrongzhang2022/InfiniteBench`, one JSONL
file per subtask) on first use via `huggingface_hub.hf_hub_download` (cached);
no build/ingest step and no `data/infinitebench/` on disk. See DATASETS.md.

Default subset = the `Retrieve.*` synthetic recall tasks (PassKey/Number/KV —
pure key->value lookup buried in a long filler context, the cleanest,
least-confounded long-context probe) plus `En.QA` (real long-book QA, the
"does it generalize past synthetic recall" check). Average context length
across these subtasks is >100k tokens (well past most models' native window),
so `chunk_size` truncation is expected and reported.
"""
from __future__ import annotations

import json
import random
from typing import Iterator, Optional, Union

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa

_REPO = "xinrongzhang2022/InfiniteBench"

# canonical InfiniteBench task name -> jsonl filename in the HF repo.
_TASK_FILES = {
    "Retrieve.PassKey": "passkey.jsonl",
    "Retrieve.Number": "number_string.jsonl",
    "Retrieve.KV": "kv_retrieval.jsonl",
    "En.QA": "longbook_qa_eng.jsonl",
    "En.MC": "longbook_choice_eng.jsonl",
    "En.Sum": "longbook_sum_eng.jsonl",
    "En.Dia": "longdialogue_qa_eng.jsonl",
    "Zh.QA": "longbook_qa_chn.jsonl",
    "Code.Debug": "code_debug.jsonl",
    "Code.Run": "code_run.jsonl",
    "Math.Calc": "math_calc.jsonl",
    "Math.Find": "math_find.jsonl",
}
# the cleanest, least-confounded probes: pure synthetic key->value recall,
# plus one real-document QA task for a non-synthetic check.
DEFAULT_TASKS = ["Retrieve.PassKey", "Retrieve.Number", "Retrieve.KV", "En.QA"]


def _load_task(task: str) -> list[dict]:
    if task not in _TASK_FILES:
        raise ValueError(f"InfiniteBench: unknown task {task!r} (valid: {sorted(_TASK_FILES)})")
    from huggingface_hub import hf_hub_download
    fname = _TASK_FILES[task]
    try:
        path = hf_hub_download(_REPO, filename=fname, repo_type="dataset")
    except Exception as e:
        raise RuntimeError(
            f"InfiniteBench: failed to fetch {_REPO}/{fname} for task {task!r} "
            f"(check network / HF Hub reachability): {type(e).__name__}: {e}"
        ) from e
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class InfiniteBenchDataset(IterableDataset):
    """InfiniteBench (Zhang et al., 2024). Ultra-long-context (avg >100k
    token) synthetic recall (`Retrieve.*`) and real-document tasks (`En.*`
    / `Zh.*` / `Code.*` / `Math.*`). Every subtask shares one flat schema:
    `context` (the haystack), `input` (question/instruction), `answer`
    (list of acceptable references), `options` (non-empty only for MC tasks).
    """

    def __init__(
        self,
        split: str,                          # accepted for interface parity; InfiniteBench ships one split
        tokenizer,
        chunk_size: int = 131_072,
        task: Union[str, list, None] = None,  # None -> DEFAULT_TASKS; str -> single task; list -> several
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

        if task is None:
            tasks = list(DEFAULT_TASKS)
        elif isinstance(task, str):
            tasks = [task]
        else:
            tasks = list(task)

        # rows: (task_name, question_text, context_text, answer_refs)
        self._rows: list[tuple[str, str, str, list]] = []
        for t in tasks:
            print(f"[data v1h] InfiniteBench: fetching {t} ({_TASK_FILES.get(t, '?')})...")
            task_rows = _load_task(t)
            if max_examples_per_task is not None:
                task_rows = task_rows[:max_examples_per_task]
            for ex in task_rows:
                q_text = str(ex.get("input", ""))
                options = ex.get("options") or []
                if options:
                    letters = [chr(ord("A") + i) for i in range(len(options))]
                    choice_lines = "\n".join(f"{L}) {c}" for L, c in zip(letters, options))
                    q_text = f"{q_text}\n{choice_lines}\nAnswer with only the letter."
                answers = ex.get("answer")
                if answers is None:
                    answer_refs = []
                elif isinstance(answers, list):
                    answer_refs = [str(a) for a in answers]
                else:
                    answer_refs = [str(answers)]
                self._rows.append((t, q_text, str(ex.get("context", "")), answer_refs))
            print(f"[data v1h]   InfiniteBench/{t}: {len(task_rows)} rows")

        if not self._rows:
            raise RuntimeError("InfiniteBench: 0 rows loaded — check task / network access.")
        ctx_char_lens = [len(r[2]) for r in self._rows]
        print(f"[data v1h]   InfiniteBench total: {len(self._rows)} rows across {len(tasks)} task(s); "
              f"context chars min/mean/max = {min(ctx_char_lens)}/"
              f"{sum(ctx_char_lens)//len(ctx_char_lens)}/{max(ctx_char_lens)}")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 4703 + wid * 100_003)
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
            if len(ctx_ids) > cs:
                ctx_ids = ctx_ids[:cs]
            valid_len = len(ctx_ids)
            if valid_len < cs:
                ctx_ids = ctx_ids + [self.pad_token_id] * (cs - valid_len)

            q_ids = tok(q_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            a_text = answer_refs[0] if answer_refs else ""
            a_ids = tok(a_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            if not a_ids:
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
                "answer_refs": answer_refs if answer_refs else [a_text],
                "task_family": "infinitebench",
                "question_type": task,
            }


def make_infinitebench_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 131_072,
    task: Union[str, list, None] = None,
    max_examples_per_task: Optional[int] = None,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone InfiniteBench loader (EVAL-ONLY). Default task set =
    Retrieve.PassKey/Retrieve.Number/Retrieve.KV (pure synthetic recall) +
    En.QA (real long-book QA); pass `task=` a string or list to narrow/widen."""
    ds = InfiniteBenchDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                              task=task, max_examples_per_task=max_examples_per_task,
                              sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
