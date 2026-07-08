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
        wid = wi.id if wi is not None else 0
        rng = random.Random(self.seed + wid)
        # telemetry: builds/episode surfaces a too-tight spec (many None-resamples) — both a data-health
        # signal AND a perf signal (wasted tokenization). Readable externally when num_workers=0.
        self.n_episodes = 0
        self.n_builds = 0
        for _ in range(self.n_items):
            ex, tries = None, 0
            for _ in range(self._MAX_RETRY):
                tries += 1
                ex = self.task.build(self.source, self.spec, self.tok, rng, self.pad_token_id)
                if ex is not None:
                    break
            if ex is None:
                raise ValueError(
                    f"task {self.task.__class__.__name__} on source {self.source.kind!r} "
                    f"returned None {self._MAX_RETRY}× — spec too tight: {self.spec}")
            self.n_episodes += 1
            self.n_builds += tries
            if tries >= 15 and self.n_episodes % 2000 == 1:          # throttled near-cap warning
                print(f"[data] {self.task.__class__.__name__}/{self.source.kind} w{wid}: an episode "
                      f"needed {tries}/{self._MAX_RETRY} resamples — spec may be too tight "
                      f"(avg {self.n_builds / self.n_episodes:.1f} builds/episode)", flush=True)
            yield ex


def _collate(samples, pad_token_id):
    """collate_qa + the capacity-relative k_slots/n_tokens stamping the MAE loss path reads
    (harmless no-op for tasks that don't emit k_slots)."""
    batch = collate_qa(samples, pad_token_id)
    if "k_slots" in samples[0]:
        batch.k_slots = max(int(s["k_slots"]) for s in samples)
        batch.n_tokens = [int(s["n_tokens"]) for s in samples]
    if "mask_ratio" in samples[0]:                     # mae curriculum mask-% (else loss uses cfg default)
        batch.mask_ratio = float(samples[0]["mask_ratio"])
    if "n_horizons" in samples[0]:                     # continuation: cap scored window boundaries (else all)
        batch.n_horizons = int(samples[0]["n_horizons"])
    return batch


def make_task_dataloader(source, task: Task, spec: EpisodeSpec, tokenizer, *,
                         batch_size: int, pad_token_id: int, seed: int = 0,
                         num_workers: int = 2) -> DataLoader:
    ds = TaskDataset(source, task, spec, tokenizer, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: _collate(s, pad_token_id))
