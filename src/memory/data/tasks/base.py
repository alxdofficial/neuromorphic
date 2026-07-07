"""Task base — shape a Source's items into the on-the-wire sample dict, per an EpisodeSpec.

A Task knows *what we present and ask* (reconstruction / qa / continuation / mae). It draws items
from a Source, applies the EpisodeSpec's difficulty knobs (length, distractors, query lag), and
emits the exact per-sample dict ``common.collate_qa`` consumes — so ``compute_loss`` + the
REAL/SHUF/OFF gate are unchanged. See ``docs/data_arch_plan.md`` (Layer L2).

``TaskDataset`` is the generic (source × task × spec) IterableDataset that replaces the per-reader
datasets; ``make_task_dataloader`` wires it to ``collate_qa``.
"""
from __future__ import annotations

import random
from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..common import collate_qa
from ..schedule import EpisodeSpec


class Task(ABC):
    """Builds one training example from source items + an EpisodeSpec.

    ``build`` returns the sample dict (keys: context_ids/context_mask, question_ids, answer_ids,
    answer_content_mask_list, task_family, question_type, answer_refs [+ optional k_slots/n_tokens]),
    or ``None`` to signal "unlucky draw, resample" (TaskDataset retries)."""

    accepts: tuple = ()               # source kinds this task handles, e.g. ("qa",) or ("corpus",)

    @abstractmethod
    def build(self, source, spec: EpisodeSpec, tok, rng, pad_token_id: int) -> dict | None:
        raise NotImplementedError


class TaskDataset(IterableDataset):
    """Infinite stream of (source × task × spec) examples. Worker-seeded; retries on None draws."""

    _MAX_RETRY = 50

    def __init__(self, source, task: Task, spec: EpisodeSpec, tokenizer,
                 pad_token_id: int, seed: int = 0, n_items: int = 1_000_000):
        super().__init__()
        if task.accepts and source.kind not in task.accepts:
            raise ValueError(f"task {task.__class__.__name__} accepts {task.accepts}, "
                             f"but source kind is {source.kind!r}")
        self.source = source
        self.task = task
        self.spec = spec
        self.tok = tokenizer
        self.pad_token_id = pad_token_id
        self.seed = seed
        self.n_items = n_items

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            ex = None
            for _ in range(self._MAX_RETRY):
                ex = self.task.build(self.source, self.spec, self.tok, rng, self.pad_token_id)
                if ex is not None:
                    break
            if ex is None:
                raise ValueError(
                    f"task {self.task.__class__.__name__} on source {self.source.kind!r} "
                    f"returned None {self._MAX_RETRY}× — spec too tight: {self.spec}")
            yield ex


def make_task_dataloader(source, task: Task, spec: EpisodeSpec, tokenizer, *,
                         batch_size: int, pad_token_id: int, seed: int = 0,
                         num_workers: int = 2) -> DataLoader:
    ds = TaskDataset(source, task, spec, tokenizer, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))
