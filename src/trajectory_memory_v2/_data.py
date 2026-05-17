"""Dataset readers for v2 trainers.

Inlined from `src/trajectory_memory/training/{phase1_retrieval,loaders}.py`
so the v2 package is self-contained. Originals preserved on
`abandoned/trajectory-memory-v1`.

Public surface:
- `RetrievalSampler` — composite-v1 fact-pool sampler (Wave 1)
- `TurnPairBatch`, `TurnPairDataset` — chat parquet reader (Wave 2)
- `PromptResponseDataset` — Wave 3 GRPO prompt/gold parquet reader
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Iterable

import pyarrow.parquet as pq
import torch
from torch import Tensor


# ── Wave 1 retrieval sampler ──────────────────────────────────────────────


class RetrievalSampler:
    """Samples 8-fact chunks from a JSONL fact pool.

    Each fact row must have: entity_class, entity_key, attribute,
    passage_token_ids (list[int]), question_token_ids, answer_token_ids.

    Constraint: in any sampled chunk of 8, all 8 must have distinct
    (entity_class, attribute) keys. Each step also picks a uniform-random
    target index 0..7.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        seed: int = 0,
        chunk_size: int = 8,
    ):
        self.facts = [
            json.loads(line) for line in Path(jsonl_path).read_text().splitlines()
            if line.strip()
        ]
        if not self.facts:
            raise ValueError(f"no facts in {jsonl_path}")
        self.chunk_size = chunk_size
        self.rng = random.Random(seed)
        self.by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, f in enumerate(self.facts):
            self.by_key[(f["entity_class"], f["attribute"])].append(i)
        self.keys = list(self.by_key.keys())
        if len(self.keys) < chunk_size:
            raise ValueError(
                f"only {len(self.keys)} distinct (class,attr) keys; need >= "
                f"{chunk_size}"
            )

    def sample_chunk(self) -> dict:
        keys = self.rng.sample(self.keys, self.chunk_size)
        fact_indices = [self.rng.choice(self.by_key[k]) for k in keys]
        facts = [self.facts[i] for i in fact_indices]
        target_idx = self.rng.randrange(self.chunk_size)
        target = facts[target_idx]
        return {
            "fact_passages_token_ids": [f["passage_token_ids"] for f in facts],
            "question_token_ids": target["question_token_ids"],
            "answer_token_ids": target["answer_token_ids"],
            "target_idx": target_idx,
            "target_fact_id": target["fact_id"],
            "metadata": {
                "target_attribute": target["attribute"],
                "target_entity_class": target["entity_class"],
            },
        }

    def sample_batch(self, batch_size: int) -> list[dict]:
        return [self.sample_chunk() for _ in range(batch_size)]


# ── Wave 2 chat TurnPair reader ───────────────────────────────────────────


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

    Sorts by prior length, samples windows of B near-uniform-length
    neighbors, truncates priors from the LEFT (preserves recent context
    + the BOS token).
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        batch_size: int,
        pad_id: int,
        bucket_window: int = 32,
        seed: int = 0,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        self.batch_size = batch_size
        self.pad_id = pad_id
        self.bucket_window = bucket_window
        self.seed = seed
        self._epoch = 0

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
        rows.sort(key=lambda r: len(r["prior_ids"]))
        self._rows = rows

    def __len__(self) -> int:
        n = len(self._rows)
        bucket = self.bucket_window
        total = 0
        for start in range(0, max(n - bucket + 1, 1), bucket):
            window_len = min(bucket, max(n - start, 0))
            total += window_len // self.batch_size
        return total

    def __iter__(self) -> Iterator[TurnPairBatch]:
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        n = len(self._rows)
        bucket = self.bucket_window
        windows = list(range(0, max(n - bucket + 1, 1), bucket))
        rng.shuffle(windows)

        for start in windows:
            window = list(self._rows[start : start + bucket])
            if len(window) < self.batch_size:
                continue
            rng.shuffle(window)
            for b_start in range(0, len(window) - self.batch_size + 1, self.batch_size):
                picks = window[b_start : b_start + self.batch_size]
                if len(picks) < self.batch_size:
                    continue

                t_prior_min = min(len(r["prior_ids"]) for r in picks)
                t_resp_max = max(len(r["response_ids"]) for r in picks)

                BS = len(picks)
                prior_ids = torch.full((BS, t_prior_min), self.pad_id, dtype=torch.int64)
                response_ids = torch.full((BS, t_resp_max), self.pad_id, dtype=torch.int64)
                prior_mask = torch.zeros((BS, t_prior_min), dtype=torch.bool)
                response_mask = torch.zeros((BS, t_resp_max), dtype=torch.bool)

                for i, r in enumerate(picks):
                    full_prior = r["prior_ids"]
                    if len(full_prior) <= t_prior_min:
                        p = full_prior
                    elif t_prior_min == 1:
                        p = [full_prior[0]]
                    else:
                        bos = full_prior[0]
                        p = [bos] + list(full_prior[-(t_prior_min - 1):])
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


# ── Wave 3 GRPO prompt/response reader ────────────────────────────────────


class PromptResponseDataset:
    """Loads a Wave 3 prompt-response parquet for GRPO.

    Each example is one prompt; the trainer rolls out J responses and
    computes group-relative reward against `gold_ids` (or gold text via
    `meta`). Yields one example at a time; trainer batches if it wants.

    Expected parquet columns (per `scripts/data/preprocess_grpo.py`):
        prompt_ids: List[int]
        gold_ids:   List[int]
        source:     str         "gsm8k" / "narrativeqa" / etc.
        reward_kind: str        "exact_match" / "f1_qa" / etc.
        meta_json:  str         JSON-encoded per-example metadata

    `source_weights` (optional) — per-source multiplier on replication
    frequency before shuffling. Use to upweight long-context sources
    (e.g. {"narrativeqa": 3.0, "musique": 3.0, "gsm8k": 1.0}). Rounded
    to nearest int ≥ 1.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        shuffle: bool = True,
        seed: int = 0,
        source_weights: dict[str, float] | None = None,
    ):
        self.paths = [Path(p) for p in parquet_paths]
        self.shuffle = shuffle
        self.seed = seed
        self._epoch = 0

        rows = []
        per_source_counts: dict[str, int] = {}
        for path in self.paths:
            tbl = pq.read_table(path)
            cols = {c: tbl.column(c).to_pylist() for c in tbl.column_names}
            for i in range(len(cols["prompt_ids"])):
                src = cols["source"][i]
                rows.append({
                    "prompt_ids": cols["prompt_ids"][i],
                    "gold_ids": cols["gold_ids"][i],
                    "source": src,
                    "reward_kind": cols["reward_kind"][i],
                    "meta": json.loads(cols["meta_json"][i]) if cols["meta_json"][i] else {},
                })
                per_source_counts[src] = per_source_counts.get(src, 0) + 1

        self.source_weights = source_weights or {}
        if self.source_weights:
            weighted_rows = []
            for r in rows:
                w = float(self.source_weights.get(r["source"], 1.0))
                n_copies = max(1, int(round(w)))
                weighted_rows.extend([r] * n_copies)
            rows = weighted_rows

        self._rows = rows
        self._per_source_counts = per_source_counts

    def __len__(self) -> int:
        return len(self._rows)

    def source_breakdown(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self._rows:
            out[r["source"]] = out.get(r["source"], 0) + 1
        return out

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        rows = list(self._rows)
        if self.shuffle:
            rng.shuffle(rows)
        for r in rows:
            yield r
