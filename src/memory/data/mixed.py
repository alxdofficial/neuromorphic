"""Mixed QA sampler — HotpotQA + NarrativeQA + MuSiQue + BABILong.

Weighted union of several external eval-QA readers, emitting exactly one example
per `__next__`. Registered as the `mixed` eval reader; the composite synthetic
`bio` source that this once also unioned in has been retired.

Composes the readers in this package (hotpot/narrativeqa/musique/babilong); see
each reader module and DATASETS.md for their data.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from ..config import ReprConfig
from .common import collate_qa
from .babilong import BABILongDataset
from .hotpot import HotpotQADataset
from .musique import MuSiQueDataset
from .narrativeqa import NarrativeQADataset


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
    use_hotpot: bool = True,
    use_narrative: bool = True,
    use_musique: bool = False,
    use_babilong: bool = False,
    babilong_config: str = "4k",            # BABILong length-config to use
    split: str = "train",                    # used for HotpotQA/NarrativeQA/MuSiQue
    chunk_size: int = 4096,
    weights: tuple = (0.5, 0.25, 0.25, 0.0),  # (hotpot, narrative, musique, babilong)
    num_workers: int = 0,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> DataLoader:
    """Build a mixed QA dataloader over up to 4 eval sources: HotpotQA,
    NarrativeQA, MuSiQue, BABILong. Any source with flag=False or weight≤0 is
    skipped (saves data-load cost).

    weights is a 4-tuple (hotpot, narrative, musique, babilong).
    """
    # Back-compat: pad weights tuple to length 4
    weights = tuple(weights) + (0.0,) * (4 - len(weights))

    sources, src_weights, names = [], [], []

    if use_hotpot and weights[0] > 0:
        hp_split = "train" if split == "train" else "validation"
        sources.append(HotpotQADataset(
            split=hp_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 1,
        ))
        src_weights.append(weights[0]); names.append("hotpot_qa")

    if use_narrative and weights[1] > 0:
        nq_split = "train" if split == "train" else "validation"
        sources.append(NarrativeQADataset(
            split=nq_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 2,
        ))
        src_weights.append(weights[1]); names.append("narrative_qa")

    if use_musique and weights[2] > 0:
        mq_split = "train" if split == "train" else "validation"
        sources.append(MuSiQueDataset(
            split=mq_split, tokenizer=tokenizer, chunk_size=chunk_size,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 3,
        ))
        src_weights.append(weights[2]); names.append("musique")

    if use_babilong and weights[3] > 0:
        sources.append(BABILongDataset(
            split=split, tokenizer=tokenizer, chunk_size=chunk_size,
            config_name=babilong_config,
            sep_token_id=cfg.sep_token_id, pad_token_id=cfg.pad_token_id,
            seed=seed + 4,
        ))
        src_weights.append(weights[3]); names.append(f"babilong_{babilong_config}")

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
