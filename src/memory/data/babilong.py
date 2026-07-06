"""BABILong — synthetic long-context state-tracking (bAbI tasks + filler).

EVAL reader. HF-auto-downloaded (`RMT-team/babilong`) on first use; no
build/ingest step and no `data/babilong/` on disk. See DATASETS.md.
Gotcha: only qa1-qa10 exist at configs ≥1k, so requesting qa11-qa20 there
silently loads fewer tasks (now warned — see __init__).
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import collate_qa


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
        # SAFE FIX: never silently drop requested tasks. Only qa1-qa10 exist at
        # configs ≥1k, so a request for qa11-qa20 there loads fewer than asked.
        missing = set(self.tasks) - set(self.task_data)
        if missing:
            print(f"[WARN] BABILong: requested tasks not loaded (only qa1-10 exist "
                  f"at configs >=1k): {sorted(missing)}", flush=True)
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


def make_babilong_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 4096,
    config_name: str = "4k",
    tasks: Optional[list] = None,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone BABILong loader (EVAL). Mirrors the mixed-sampler construction."""
    ds = BABILongDataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                         config_name=config_name, tasks=tasks,
                         sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
