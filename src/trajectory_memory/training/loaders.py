"""Parquet dataset readers + collators for each wave's preprocessed format.

- `LongDocDataset`        — Wave 1 (long-doc TF). Reads {input_ids, num_tokens}
                            parquet, packs into D*T_window chunks.
- `TurnPairDataset`       — Wave 2 / Wave 4 (chat TurnPairs). Reads
                            {prior_ids, response_ids} parquet,
                            length-buckets for batching.
- `PromptResponseDataset` — Wave 3 (GRPO). Reads {prompt_ids, gold_ids}
                            parquet, yields one example per call.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import IterableDataset


# ── Wave 1 long-doc ────────────────────────────────────────────────────


class LongDocDataset(IterableDataset):
    """Streams long-doc parquet, packs each doc's tokens into chunks of
    `chunk_tokens` tokens (= D * T_window). Documents are NOT mixed
    within a chunk (per plan §4.5). Trailing partial chunks are padded
    with `pad_id` if `drop_short=False`, else dropped.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        chunk_tokens: int,
        pad_id: int,
        drop_short: bool = False,
        shuffle: bool = True,
        seed: int = 0,
    ):
        super().__init__()
        self.paths = [Path(p) for p in parquet_paths]
        self.chunk_tokens = chunk_tokens
        self.pad_id = pad_id
        self.drop_short = drop_short
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[Tensor]:
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)

        paths = list(self.paths)
        if self.shuffle:
            rng.shuffle(paths)

        for path in paths:
            tbl = pq.read_table(path, columns=["input_ids"])
            rows = tbl.column("input_ids").to_pylist()
            if self.shuffle:
                rng.shuffle(rows)
            for ids in rows:
                if not ids:
                    continue
                for start in range(0, len(ids), self.chunk_tokens):
                    chunk = ids[start : start + self.chunk_tokens]
                    if len(chunk) < self.chunk_tokens:
                        if self.drop_short:
                            continue
                        chunk = chunk + [self.pad_id] * (self.chunk_tokens - len(chunk))
                    yield torch.tensor(chunk, dtype=torch.int64)


# ── Wave 2 / Wave 4 TurnPair ───────────────────────────────────────────


@dataclass
class TurnPairBatch:
    """A length-bucketed batch of TurnPairs."""
    prior_ids: Tensor          # [BS, T_prior_max] padded
    response_ids: Tensor       # [BS, T_response_max] padded
    prior_mask: Tensor         # [BS, T_prior_max] bool
    response_mask: Tensor      # [BS, T_response_max] bool
    sources: list[str]


class TurnPairDataset:
    """Loads a TurnPair parquet and yields length-bucketed batches.

    Implements graph_walker's W4 batching pattern (plan §4.8): sort by
    prior length, sample windows of B near-uniform-length neighbors,
    truncate to min length within batch.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        batch_size: int,
        pad_id: int,
        bucket_window: int = 32,    # B near-uniform-length neighbors per bucket
        seed: int = 0,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.bucket_window = bucket_window
        self.seed = seed

        # Load all rows into memory (TurnPair datasets are small enough).
        rows = []
        for path in self.paths:
            tbl = pq.read_table(path)
            for prior, response, source in zip(
                tbl.column("prior_ids").to_pylist(),
                tbl.column("response_ids").to_pylist(),
                tbl.column("source").to_pylist(),
            ):
                rows.append({
                    "prior_ids": prior,
                    "response_ids": response,
                    "source": source,
                })
        # Sort by prior length for bucketing.
        rows.sort(key=lambda r: len(r["prior_ids"]))
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows) // self.batch_size

    def __iter__(self) -> Iterator[TurnPairBatch]:
        rng = random.Random(self.seed)
        # Walk in bucket-sized windows. Sample B from each bucket.
        n = len(self._rows)
        bucket = self.bucket_window
        windows = list(range(0, n - bucket, bucket))
        rng.shuffle(windows)

        for start in windows:
            window = self._rows[start : start + bucket]
            rng.shuffle(window)
            picks = window[: self.batch_size]
            if len(picks) < self.batch_size:
                continue

            # Truncate to the min length within this batch (per plan).
            t_prior_min = min(len(r["prior_ids"]) for r in picks)
            t_resp_max = max(len(r["response_ids"]) for r in picks)

            BS = len(picks)
            prior_ids = torch.full((BS, t_prior_min), self.pad_id, dtype=torch.int64)
            response_ids = torch.full((BS, t_resp_max), self.pad_id, dtype=torch.int64)
            prior_mask = torch.zeros((BS, t_prior_min), dtype=torch.bool)
            response_mask = torch.zeros((BS, t_resp_max), dtype=torch.bool)

            for i, r in enumerate(picks):
                # Truncate prior from the LEFT (keep the most recent context).
                p = r["prior_ids"][-t_prior_min:]
                prior_ids[i, :len(p)] = torch.tensor(p, dtype=torch.int64)
                prior_mask[i, :len(p)] = True
                resp = r["response_ids"][:t_resp_max]
                response_ids[i, :len(resp)] = torch.tensor(resp, dtype=torch.int64)
                response_mask[i, :len(resp)] = True

            yield TurnPairBatch(
                prior_ids=prior_ids,
                response_ids=response_ids,
                prior_mask=prior_mask,
                response_mask=response_mask,
                sources=[r["source"] for r in picks],
            )


# ── Wave 3 prompt-response (GRPO) ──────────────────────────────────────


@dataclass
class PromptResponseBatch:
    prompt_ids: list[list[int]]      # variable length per example
    gold_ids: list[list[int]]
    sources: list[str]
    reward_kinds: list[str]
    meta: list[dict]                 # parsed from meta_json


class PromptResponseDataset:
    """Loads a Wave 3 prompt-response parquet for GRPO.

    Each example is one prompt; the trainer rolls out J responses and
    computes group-relative reward against `gold_ids`.

    Doesn't pad / batch — GRPO rollout is typically per-example (or
    small fixed batches with same-length prompts). We yield one example
    at a time; trainer can batch if it wants.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        shuffle: bool = True,
        seed: int = 0,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        self.shuffle = shuffle
        self.seed = seed

        import json as _json
        rows = []
        for path in self.paths:
            tbl = pq.read_table(path)
            cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
            for i in range(len(cols["prompt_ids"])):
                rows.append({
                    "prompt_ids": cols["prompt_ids"][i],
                    "gold_ids": cols["gold_ids"][i],
                    "source": cols["source"][i],
                    "reward_kind": cols["reward_kind"][i],
                    "meta": _json.loads(cols["meta_json"][i]) if cols["meta_json"][i] else {},
                })
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self):
        rng = random.Random(self.seed)
        rows = list(self._rows)
        if self.shuffle:
            rng.shuffle(rows)
        for r in rows:
            yield r
