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


@dataclass
class LongDocChunk:
    """A single Wave 1 chunk + the metadata the trainer needs to thread
    state across consecutive chunks of the same document.

    - `input_ids`:   [chunk_tokens] token IDs
    - `is_doc_start`: True iff this is the first chunk of a new document.
                      Trainer resets `prev_states / hiddens / lm_context`
                      when this fires.
    - `valid_mask`:   [chunk_tokens] bool — True for real tokens, False for
                      pad tokens added at the end of partial chunks. Used
                      by `step_wave1` as `target_mask` so pad positions
                      don't contribute to NTP CE.
    """
    input_ids: Tensor
    is_doc_start: bool
    valid_mask: Tensor


class LongDocDataset(IterableDataset):
    """Streams long-doc parquet, packs each doc's tokens into chunks of
    `chunk_tokens` tokens (= D * T_window). Documents are NOT mixed
    within a chunk (per plan §4.5). Trailing partial chunks are padded
    with `pad_id` if `drop_short=False`, else dropped.

    Yields `LongDocChunk` (not bare tensors). The `is_doc_start` flag
    lets the trainer reset manifold/hidden/LM state at document
    boundaries while threading state through consecutive chunks of the
    same document — critical for memory to be load-bearing on long docs
    (otherwise each chunk fits in the LM cap and direct attention can
    solve the loss without using memory).
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
        # B2 fix — track epoch count so each iter gets a different shuffle.
        # The earlier `random.Random(seed)` per __iter__ produced identical
        # batch order every epoch → multi-epoch training degenerated into
        # deterministic memorization passes (gradient noise ≪ what should
        # be expected; momentum/Adam stats drift along a fixed trajectory).
        self._epoch = 0

    def __iter__(self) -> Iterator[LongDocChunk]:
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + self._epoch + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        self._epoch += 1

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
                for chunk_idx, start in enumerate(
                    range(0, len(ids), self.chunk_tokens),
                ):
                    chunk = ids[start : start + self.chunk_tokens]
                    n_real = len(chunk)
                    if n_real < self.chunk_tokens:
                        if self.drop_short:
                            continue
                        chunk = chunk + [self.pad_id] * (self.chunk_tokens - n_real)
                    valid_mask = torch.zeros(self.chunk_tokens, dtype=torch.bool)
                    valid_mask[:n_real] = True
                    yield LongDocChunk(
                        input_ids=torch.tensor(chunk, dtype=torch.int64),
                        is_doc_start=(chunk_idx == 0),
                        valid_mask=valid_mask,
                    )


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
        self._epoch = 0  # B2 fix — see LongDocDataset

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
        # Number of batches per epoch — how many windows × how many batches
        # per window. We drop incomplete tails on both axes.
        n = len(self._rows)
        bucket = self.bucket_window
        n_windows = max(n - bucket + 1, 1) // bucket
        batches_per_window = bucket // self.batch_size
        return n_windows * batches_per_window

    def __iter__(self) -> Iterator[TurnPairBatch]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        # Walk in bucket-sized windows; within each bucket emit ALL examples
        # in batches of size `batch_size` (length-similar within a batch).
        # Earlier behavior emitted only ONE batch per bucket and discarded
        # the remaining `bucket - batch_size` rows — at default
        # bucket_window=32 / batch_size=2 that dropped 94% of the dataset
        # per epoch. WildChat 4978 rows → 310 examples; UltraChat 27463
        # rows → 1716 examples. Now we get full coverage.
        n = len(self._rows)
        bucket = self.bucket_window
        windows = list(range(0, max(n - bucket + 1, 1), bucket))
        rng.shuffle(windows)

        for start in windows:
            window = list(self._rows[start : start + bucket])
            if len(window) < self.batch_size:
                continue
            rng.shuffle(window)
            # Emit floor(len(window) / batch_size) batches from this bucket.
            for b_start in range(0, len(window) - self.batch_size + 1, self.batch_size):
                picks = window[b_start : b_start + self.batch_size]
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
        self._epoch = 0  # B2 fix — see LongDocDataset

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
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        rows = list(self._rows)
        if self.shuffle:
            rng.shuffle(rows)
        for r in rows:
            yield r
