"""Parquet dataset readers + collators for each wave's preprocessed format.

- `LongDocDataset`        — Wave 1 (long-doc TF). Reads {input_ids, num_tokens}
                            parquet, packs into D*T_window chunks. Round-robin
                            interleaves across sources so a small needle source
                            isn't starved by a larger fineweb source. Also
                            reads needle answer metadata (when present) and
                            applies an `answer_span_weight` boost so the
                            5-token answer gradient isn't buried under 30K
                            filler tokens.
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
    - `valid_mask`:   [chunk_tokens] FLOAT32 — per-token weight for NTP CE.
                      0.0 for pad tokens (excluded from loss).
                      1.0 for normal real tokens.
                      `answer_span_weight` (default 100.0) for tokens that
                      fall in the needle-answer span. Lets the few critical
                      answer tokens drive a meaningful gradient signal
                      against the overwhelming filler background.
    - `source`:       which preprocessor parquet this chunk came from
                      (e.g. "fineweb_edu", "needle"). Used by the trainer
                      for per-source train-loss telemetry.
    """
    input_ids: Tensor
    is_doc_start: bool
    valid_mask: Tensor
    source: str = ""


class LongDocDataset(IterableDataset):
    """Streams long-doc parquet, packs each doc's tokens into chunks of
    `chunk_tokens` tokens (= D * T_window). Documents are NOT mixed
    within a chunk (per plan §4.5). Trailing partial chunks are padded
    with `pad_id` if `drop_short=False`, else dropped.

    Source-mix interleaving (Tier 1 fix): instead of iterating ALL of
    source A's docs before any of source B (the prior bug that caused
    a 10k-step run to never reach `needle` because `slimpajama` had
    33M tokens to chew through first), we round-robin sample across
    sources document-by-document. Each source's doc queue is
    independently shuffled; the per-doc source choice is random
    weighted by remaining queue length so each source ends in step
    with the others (proportional to its size).

    Answer-span weighting (Tier 1 fix): if the parquet contains the
    needle-haystack metadata columns (`answer`, `query_pos_token`),
    we tokenize the answer string at __init__ time, locate it in
    each doc's token sequence, and emit a float32 `valid_mask` that
    weights answer-span tokens by `answer_span_weight` (default 100×)
    relative to filler. Without this, the 5-token answer gradient is
    invisible against 30K filler tokens — even a perfect-retrieval
    model only moves the average doc loss by ~0.001 nats.

    Yields `LongDocChunk` (with float `valid_mask` post-fix). The
    `is_doc_start` flag lets the trainer reset manifold/hidden/LM
    state at document boundaries while threading state through
    consecutive chunks of the same document.
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
        answer_span_weight: float = 100.0,
    ):
        super().__init__()
        self.paths = [Path(p) for p in parquet_paths]
        self.chunk_tokens = chunk_tokens
        self.pad_id = pad_id
        self.drop_short = drop_short
        self.shuffle = shuffle
        self.seed = seed
        self.answer_span_weight = answer_span_weight
        # B2 fix — track epoch count so each iter gets a different shuffle.
        self._epoch = 0
        # Pre-load all sources at init time. This trades a one-time RAM
        # cost (~few hundred MB for our W1 dataset) for clean source-mix
        # interleaving and zero per-doc parquet I/O during iteration.
        self._sources = self._load_sources()

    def _load_sources(self) -> list[dict]:
        """Read each parquet, extract input_ids + needle metadata when
        present. Returns a list of {name, rows: [{input_ids, ans_start,
        ans_end}, ...]} dicts, one per source path.

        Two paths for needle answer-span:
          (a) PREFERRED — `answer_start_token` + `answer_end_token`
              columns precomputed at synthesizer time (exact, fast).
          (b) FALLBACK — `answer` + `query_pos_token` columns (older
              parquet schema). Re-tokenize answer at runtime and locate
              via subseq match. Less reliable when query_pos_token's
              chars/4 estimate is way off.
        """
        tok = None
        sources: list[dict] = []
        for path in self.paths:
            tbl = pq.read_table(path)
            cols = set(tbl.column_names)
            has_exact_span = (
                "answer_start_token" in cols and "answer_end_token" in cols
            )
            has_needle_legacy = "answer" in cols and "query_pos_token" in cols
            ids_col = tbl.column("input_ids").to_pylist()

            if has_exact_span:
                # Path (a): use precomputed exact token positions.
                start_col = tbl.column("answer_start_token").to_pylist()
                end_col = tbl.column("answer_end_token").to_pylist()
                rows = []
                for ids, s, e in zip(ids_col, start_col, end_col):
                    ans_start = s if (s is not None and s >= 0) else None
                    ans_end = e if (e is not None and e >= 0) else None
                    rows.append({
                        "input_ids": ids,
                        "ans_start": ans_start,
                        "ans_end": ans_end,
                    })
            elif has_needle_legacy:
                # Path (b): legacy parquet — locate at runtime.
                if tok is None:
                    from src.trajectory_memory.data.tokenizer import get_tokenizer
                    tok = get_tokenizer()
                answer_col = tbl.column("answer").to_pylist()
                query_pos_col = tbl.column("query_pos_token").to_pylist()
                rows = []
                for ids, answer, qp in zip(ids_col, answer_col, query_pos_col):
                    ans_start, ans_end = self._locate_answer_span(
                        ids, answer, qp, tok,
                    )
                    rows.append({
                        "input_ids": ids,
                        "ans_start": ans_start,
                        "ans_end": ans_end,
                    })
            else:
                rows = [
                    {"input_ids": ids, "ans_start": None, "ans_end": None}
                    for ids in ids_col
                ]

            sources.append({
                "name": path.stem.split(".")[0],   # "fineweb_edu" from "fineweb_edu.train.parquet"
                "rows": rows,
            })
        return sources

    @staticmethod
    def _locate_answer_span(
        input_ids: list[int],
        answer_text: str,
        query_pos_token: int | None,
        tokenizer,
    ) -> tuple[int | None, int | None]:
        """Find the answer-token span inside `input_ids` so the trainer can
        weight it. The answer immediately follows the query in the synth
        template, so we search starting from `query_pos_token`.

        BPE quirk: `tokenizer.encode("K7T9XB")` may differ from the answer
        as it appears IN context (with leading space). We try both
        leading-space and no-leading-space tokenizations and take whichever
        matches first.

        Returns (start, end) absolute token positions, or (None, None) if
        the answer can't be located (e.g. multi-token answer that
        retokenizes differently).
        """
        if not answer_text or not input_ids:
            return None, None
        # Search lower bound — start near the query if we have it, else 0.
        lo = max(0, (query_pos_token or 0) - 4)

        # Try multiple encodings to handle the BPE leading-space quirk.
        candidates = [
            tokenizer.encode(answer_text, add_special_tokens=False),
            tokenizer.encode(" " + answer_text, add_special_tokens=False),
        ]
        for cand in candidates:
            if not cand:
                continue
            n = len(cand)
            # Linear scan from lo onward.
            for i in range(lo, len(input_ids) - n + 1):
                if input_ids[i:i + n] == cand:
                    return i, i + n
        return None, None

    def __iter__(self) -> Iterator[LongDocChunk]:
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + self._epoch + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        self._epoch += 1

        # Per-source independently shuffled doc queues. Iterate via .pop()
        # from end (O(1)).
        per_source_queues: list[list[int]] = []
        for src in self._sources:
            indices = list(range(len(src["rows"])))
            if self.shuffle:
                rng.shuffle(indices)
            per_source_queues.append(indices)

        # Active-source rotation: each iteration picks a source weighted by
        # its remaining queue length, pops one doc, yields its chunks. When
        # a source's queue empties, it drops out of the rotation. Result:
        # each source is fully consumed exactly once per epoch, with docs
        # interleaved across sources rather than concatenated.
        while True:
            active = [i for i, q in enumerate(per_source_queues) if q]
            if not active:
                break
            weights = [len(per_source_queues[i]) for i in active]
            chosen = rng.choices(active, weights=weights, k=1)[0]
            doc_idx = per_source_queues[chosen].pop()
            doc = self._sources[chosen]["rows"][doc_idx]
            ids = doc["input_ids"]
            if not ids:
                continue
            ans_start = doc["ans_start"]
            ans_end = doc["ans_end"]
            source_name = self._sources[chosen]["name"]

            for chunk_idx, start in enumerate(
                range(0, len(ids), self.chunk_tokens),
            ):
                chunk = ids[start : start + self.chunk_tokens]
                n_real = len(chunk)
                if n_real < self.chunk_tokens:
                    if self.drop_short:
                        continue
                    chunk = chunk + [self.pad_id] * (self.chunk_tokens - n_real)

                # Float mask: 0 for pad, 1 for filler real tokens, boost
                # for answer-span tokens within this chunk.
                valid_mask = torch.zeros(self.chunk_tokens, dtype=torch.float32)
                valid_mask[:n_real] = 1.0
                if ans_start is not None and ans_end is not None:
                    chunk_lo = start
                    chunk_hi = start + n_real    # exclusive
                    overlap_lo = max(ans_start, chunk_lo)
                    overlap_hi = min(ans_end, chunk_hi)
                    if overlap_lo < overlap_hi:
                        local_lo = overlap_lo - chunk_lo
                        local_hi = overlap_hi - chunk_lo
                        valid_mask[local_lo:local_hi] = self.answer_span_weight

                yield LongDocChunk(
                    input_ids=torch.tensor(chunk, dtype=torch.int64),
                    is_doc_start=(chunk_idx == 0),
                    valid_mask=valid_mask,
                    source=source_name,
                )


# ── Wave 1 BATCHED multi-stream dataset (Tier 4 #13) ──────────────────


@dataclass
class BatchedLongDocChunk:
    """A batched W1 chunk where each batch slot may be at a different
    point in a different document.

    - `input_ids`:               [BS, chunk_tokens]
    - `is_doc_start_per_slot`:   [BS] bool — True for slots starting a new doc
    - `valid_mask`:              [BS, chunk_tokens] float — per-slot loss weights
    - `sources`:                 list[str] of length BS — per-slot source name
    """
    input_ids: Tensor
    is_doc_start_per_slot: Tensor
    valid_mask: Tensor
    sources: list[str]


class BatchedLongDocDataset(IterableDataset):
    """Multi-stream W1 dataset: maintains BS parallel "lanes," each with its
    own current-doc + chunk position. Per training step yields ONE chunk
    per slot (batched). When a slot's doc finishes, that slot independently
    pulls the next doc from a SHARED source pool so source-mix interleaving
    is preserved across slots.

    Use when --batch-size > 1 in `train_wave1.py`. The trainer is
    responsible for per-slot state reset using `is_doc_start_per_slot`:

        if batch.is_doc_start_per_slot.any():
            fresh_states = manifold.reset_states(BS)
            mask = batch.is_doc_start_per_slot[:, None, None]
            prev_states = torch.where(mask, fresh_states, prev_states)
            ... (same for hiddens, lm_buffer)

    Wraps `LongDocDataset`'s source loader; reuses the same parquet
    parsing + answer-span metadata extraction.
    """

    def __init__(
        self,
        parquet_paths: list[Path],
        *,
        batch_size: int,
        chunk_tokens: int,
        pad_id: int,
        drop_short: bool = False,
        shuffle: bool = True,
        seed: int = 0,
        answer_span_weight: float = 100.0,
    ):
        super().__init__()
        assert batch_size >= 1, f"batch_size must be >= 1, got {batch_size}"
        # Reuse single-stream dataset for source loading + metadata.
        self._base = LongDocDataset(
            parquet_paths,
            chunk_tokens=chunk_tokens,
            pad_id=pad_id,
            drop_short=drop_short,
            shuffle=shuffle,
            seed=seed,
            answer_span_weight=answer_span_weight,
        )
        self.batch_size = batch_size
        self.chunk_tokens = chunk_tokens
        self.pad_id = pad_id
        self.answer_span_weight = answer_span_weight
        self.shuffle = shuffle
        self.seed = seed
        # Track epoch separately from base so multi-stream and single-stream
        # don't fight over shuffle seeds.
        self._epoch = 0

    @property
    def _sources(self):
        return self._base._sources

    def __iter__(self) -> Iterator[BatchedLongDocChunk]:
        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed + self._epoch + (worker_info.id if worker_info else 0)
        rng = random.Random(seed)
        self._epoch += 1

        # Shared source pool: per-source shuffled doc indices, popped by
        # whichever slot needs a doc next. Keeps source-mix interleaving
        # across slots (the small needle source isn't starved by the larger
        # fineweb source — same proportional draw as single-stream).
        per_source_queues: list[list[int]] = []
        for src in self._sources:
            indices = list(range(len(src["rows"])))
            if self.shuffle:
                rng.shuffle(indices)
            per_source_queues.append(indices)

        def pop_doc() -> tuple[int, int] | None:
            """Pull (source_idx, doc_idx) from the shared pool, weighted by
            remaining queue lengths. Returns None when all sources empty."""
            active = [i for i, q in enumerate(per_source_queues) if q]
            if not active:
                return None
            weights = [len(per_source_queues[i]) for i in active]
            chosen = rng.choices(active, weights=weights, k=1)[0]
            doc_idx = per_source_queues[chosen].pop()
            return chosen, doc_idx

        # Per-slot state: which doc, which chunk in that doc, source name.
        slots: list[dict] = [
            {"doc": None, "chunk_idx": 0, "source_name": "", "ans_start": None,
             "ans_end": None}
            for _ in range(self.batch_size)
        ]

        while True:
            # Refill empty slots from the shared pool. If the pool is dry,
            # this slot will stay empty and we end the epoch this iter.
            for slot in slots:
                if slot["doc"] is None:
                    pop = pop_doc()
                    if pop is None:
                        continue
                    src_idx, doc_idx = pop
                    row = self._sources[src_idx]["rows"][doc_idx]
                    slot["doc"] = row["input_ids"]
                    slot["ans_start"] = row["ans_start"]
                    slot["ans_end"] = row["ans_end"]
                    slot["chunk_idx"] = 0
                    slot["source_name"] = self._sources[src_idx]["name"]

            # End-of-epoch: any slot that still has no doc → epoch done.
            if any(slot["doc"] is None for slot in slots):
                return

            # Build one batched chunk: pull current chunk from each slot.
            batch_input_ids: list[Tensor] = []
            batch_is_start: list[bool] = []
            batch_valid_mask: list[Tensor] = []
            batch_sources: list[str] = []

            for slot in slots:
                doc_ids = slot["doc"]
                ci = slot["chunk_idx"]
                start = ci * self.chunk_tokens
                ids_chunk = doc_ids[start : start + self.chunk_tokens]
                n_real = len(ids_chunk)
                if n_real < self.chunk_tokens:
                    ids_chunk = list(ids_chunk) + [self.pad_id] * (
                        self.chunk_tokens - n_real
                    )

                # Per-token mask: 0 for pad, 1 for filler real token,
                # answer_span_weight for needle answer-span tokens (when
                # the metadata is present and overlaps this chunk).
                valid_mask = torch.zeros(self.chunk_tokens, dtype=torch.float32)
                valid_mask[:n_real] = 1.0
                if slot["ans_start"] is not None and slot["ans_end"] is not None:
                    chunk_lo = start
                    chunk_hi = start + n_real
                    overlap_lo = max(slot["ans_start"], chunk_lo)
                    overlap_hi = min(slot["ans_end"], chunk_hi)
                    if overlap_lo < overlap_hi:
                        local_lo = overlap_lo - chunk_lo
                        local_hi = overlap_hi - chunk_lo
                        valid_mask[local_lo:local_hi] = self.answer_span_weight

                batch_input_ids.append(
                    torch.tensor(ids_chunk, dtype=torch.int64),
                )
                batch_is_start.append(ci == 0)
                batch_valid_mask.append(valid_mask)
                batch_sources.append(slot["source_name"])

                # Advance slot. If we just emitted the last chunk of this
                # doc, mark slot for refill on next iteration.
                slot["chunk_idx"] += 1
                if start + self.chunk_tokens >= len(doc_ids):
                    slot["doc"] = None

            yield BatchedLongDocChunk(
                input_ids=torch.stack(batch_input_ids),
                is_doc_start_per_slot=torch.tensor(batch_is_start, dtype=torch.bool),
                valid_mask=torch.stack(batch_valid_mask),
                sources=batch_sources,
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
        total = 0
        for start in range(0, max(n - bucket + 1, 1), bucket):
            window_len = min(bucket, max(n - start, 0))
            total += window_len // self.batch_size
        return total

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
                    # B4 fix — Truncate prior from LEFT (keep recent context)
                    # but preserve BOS at position 0. Llama-3 expects
                    # `<|begin_of_text|>` as token 0; without it, attention
                    # produces different distributions than training.
                    full_prior = r["prior_ids"]
                    if len(full_prior) <= t_prior_min:
                        p = full_prior  # no truncation needed
                    elif t_prior_min == 1:
                        # Edge case: degenerate batch with a 1-token prior.
                        # `full_prior[-(t_prior_min - 1):]` = `full_prior[0:]`
                        # = entire list, then `[bos] + entire_list` would
                        # overflow the prior_ids[i, :len(p)] slice.
                        # Just keep BOS only.
                        p = [full_prior[0]]
                    else:
                        bos = full_prior[0]  # preserve original BOS
                        # Take last (t_prior_min - 1) tokens, prepend BOS
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

    `source_weights` (optional) — per-source multiplier on appearance
    frequency. Rows from each source are replicated `weight` times before
    shuffling. Use to upweight long-context / memory-relevant sources
    (e.g. {"narrativeqa": 3.0, "musique": 3.0, "gsm8k": 1.0}). Default 1.0
    for any unmentioned source. Fractional weights are supported via
    rounding to nearest int (≥1) — so 1.5× replication ≈ 1 or 2 copies.
    For finer-grained mixing, also use `target_long_frac` (below).
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
        self._epoch = 0  # B2 fix — see LongDocDataset

        import json as _json
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
                    "meta": _json.loads(cols["meta_json"][i]) if cols["meta_json"][i] else {},
                })
                per_source_counts[src] = per_source_counts.get(src, 0) + 1

        # Apply per-source weights via replication.
        self.source_weights = source_weights or {}
        if self.source_weights:
            weighted_rows = []
            for r in rows:
                w = float(self.source_weights.get(r["source"], 1.0))
                # Replicate (rounded to nearest int, ≥1).
                n_copies = max(1, int(round(w)))
                weighted_rows.extend([r] * n_copies)
            rows = weighted_rows

        self._rows = rows
        self._per_source_counts = per_source_counts

    def __len__(self) -> int:
        return len(self._rows)

    def source_breakdown(self) -> dict[str, int]:
        """Return effective per-source counts after weighting — for logging."""
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

    def iter_batched(
        self,
        batch_size: int,
        *,
        min_prompt_len: int | None = None,
    ):
        """Yield groups of `batch_size` rows with similar prompt lengths
        (length-bucketed sampling for Phase 2 BS_outer > 1).

        Algorithm: sort rows by num_prompt, partition into consecutive
        chunks of `batch_size`, shuffle chunk order so we don't always
        train short→long within an epoch. Each chunk's rows are
        sorted-similar in length, so batched prefill pads at most by
        the longest-vs-shortest ratio within the chunk.

        `min_prompt_len`: optional filter — drop rows whose prompt is
        shorter than this. Useful to ensure all prompts in a batch
        comfortably exceed `effective_lm_context=2048` so that after
        prefill all rows' KV caches are at the same (trimmed) length,
        eliminating the need for per-row cache padding.
        """
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        rows = list(self._rows)
        if min_prompt_len is not None:
            rows = [r for r in rows if len(r["prompt_ids"]) >= min_prompt_len]
        if not rows:
            return
        # Sort by prompt length ascending; group into batch-size chunks.
        rows.sort(key=lambda r: len(r["prompt_ids"]))
        n_full = len(rows) // batch_size
        chunks = [rows[i * batch_size:(i + 1) * batch_size] for i in range(n_full)]
        # Tail rows that don't fill a batch are dropped (consistent batching
        # is more important than seeing every example each epoch — they'll
        # come back next epoch with a different shuffle).
        if self.shuffle:
            rng.shuffle(chunks)
        for chunk in chunks:
            yield chunk
