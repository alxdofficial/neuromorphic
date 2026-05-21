"""Real-data dataset for representation learning.

Mixes FineWeb-edu (long pretokenized documents) and composite_v1
(short pretokenized passages) at a fixed ratio. Yields 256-token windows
with span masking ready for ReprLearningModel.

The encoder sees the full unmasked window; the Llama decoder gets the
masked variant and reconstructs.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.utils.data import IterableDataset, DataLoader

from .config import ReprConfig
from .data import WindowBatch, make_batch_mask_positions


def _load_fineweb_token_chunks(
    path: Path, window_size: int,
) -> list[list[int]]:
    """Load FineWeb parquet and slice each long document into window_size chunks.

    Returns a list of token-id lists, each of length exactly window_size.
    Drops the tail of each document if it doesn't fill a full window.
    """
    table = pq.read_table(path, columns=["input_ids"])
    chunks: list[list[int]] = []
    for ids in table["input_ids"].to_pylist():
        n = len(ids)
        n_chunks = n // window_size
        for i in range(n_chunks):
            chunks.append(ids[i * window_size : (i + 1) * window_size])
    return chunks


def _load_fineweb_documents(
    path: Path, min_len: int,
) -> list[list[int]]:
    """Load full FineWeb-edu documents (no pre-chunking).

    Returns one list per document, filtered to those with at least `min_len`
    tokens. v1b uses this to enable document-aware sampling: pick a random
    window from inside a single document rather than slicing arbitrarily
    across boundaries.
    """
    table = pq.read_table(path, columns=["input_ids"])
    docs: list[list[int]] = []
    for ids in table["input_ids"].to_pylist():
        if len(ids) >= min_len:
            docs.append(list(ids))
    return docs


@dataclass
class ChunkPair:
    """A pair of consecutive same-document chunks for hidden-state matching."""
    chunk_1: Tensor   # [chunk_size] long — first window
    chunk_2: Tensor   # [chunk_size] long — next window in same doc


class ChunkPairDataset(IterableDataset):
    """v1e dataset: yields (chunk_1, chunk_2) pairs of consecutive chunks
    drawn from the same FineWeb-edu document.

    Each chunk is exactly `chunk_size` tokens. For a doc of length N, we
    enumerate non-overlapping pairs (offset[k], offset[k+1]) where
    offset[k] = k · chunk_size for k = 0 .. ⌊N/chunk_size⌋ − 2. Random
    pair sampling across docs.

    Composite passages are intentionally excluded — they're single-passage
    units with no natural "next chunk." v1e tests representation quality
    on coherent continuous text only.
    """

    def __init__(
        self,
        fineweb_path: Path,
        chunk_size: int = 256,
        seed: int = 0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.seed = seed

        print(f"[data] loading FineWeb-edu docs from {fineweb_path}...")
        # Need ≥ 2 × chunk_size tokens to yield at least one pair
        self.docs = _load_fineweb_documents(fineweb_path, min_len=2 * chunk_size)
        print(f"[data]   {len(self.docs):,} docs with ≥ {2 * chunk_size} tokens")

        # Pre-enumerate pair starting offsets so sampling is O(1)
        self.pair_index: list[tuple[int, int]] = []  # (doc_idx, start_token_offset)
        for di, doc in enumerate(self.docs):
            n_pairs = (len(doc) // chunk_size) - 1   # k=0..n_pairs-1, both chunks fit
            for k in range(n_pairs):
                self.pair_index.append((di, k * chunk_size))
        print(f"[data]   {len(self.pair_index):,} consecutive chunk pairs available")

        if not self.pair_index:
            raise ValueError(
                f"No chunk pairs available — every doc is shorter than "
                f"{2 * chunk_size} tokens. Check fineweb_path."
            )

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid)
        cs = self.chunk_size

        while True:
            di, off = rng.choice(self.pair_index)
            doc = self.docs[di]
            c1 = doc[off : off + cs]
            c2 = doc[off + cs : off + 2 * cs]
            yield {
                "chunk_1": torch.tensor(c1, dtype=torch.long),
                "chunk_2": torch.tensor(c2, dtype=torch.long),
            }


def collate_chunkpair(samples: list[dict]) -> ChunkPair:
    """Stack a list of per-sample chunk-pair dicts into a ChunkPair batch."""
    chunk_1 = torch.stack([s["chunk_1"] for s in samples])
    chunk_2 = torch.stack([s["chunk_2"] for s in samples])
    return ChunkPair(chunk_1=chunk_1, chunk_2=chunk_2)


def make_chunkpair_dataloader(
    cfg: ReprConfig,
    fineweb_path: Path | str,
    chunk_size: int = 256,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    ds = ChunkPairDataset(Path(fineweb_path), chunk_size=chunk_size, seed=seed)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        collate_fn=collate_chunkpair,
        pin_memory=torch.cuda.is_available(),
    )


def _load_composite_passages(path: Path) -> list[list[int]]:
    """Load composite passages JSONL. Returns list of token-id lists
    (variable length, typically 30-100 tokens)."""
    passages: list[list[int]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            ids = row.get("passage_token_ids")
            if ids:
                passages.append(list(ids))
    return passages


class MixedReprDataset(IterableDataset):
    """Iterable dataset mixing FineWeb-edu + composite at fixed ratio.

    Yields dicts with input_ids, attention_mask, mask_positions for one
    sample (no batching — DataLoader batches).

    Composite: concat random passages until >= window_size tokens, truncate.
    FineWeb-edu: pick a random pre-chunked 256-token window.
    """

    def __init__(
        self,
        cfg: ReprConfig,
        fineweb_path: Path,
        composite_passages_path: Path,
        fineweb_ratio: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.fineweb_ratio = fineweb_ratio
        self.seed = seed

        window_size = cfg.fixed_window_size

        print(f"[data] loading FineWeb-edu from {fineweb_path}...")
        self.fineweb_chunks = _load_fineweb_token_chunks(
            fineweb_path, window_size,
        )
        print(f"[data]   {len(self.fineweb_chunks):,} chunks of {window_size} tokens")

        print(f"[data] loading composite passages from {composite_passages_path}...")
        self.composite_passages = _load_composite_passages(composite_passages_path)
        print(f"[data]   {len(self.composite_passages):,} passages "
              f"(avg {sum(len(p) for p in self.composite_passages) / max(len(self.composite_passages),1):.0f} tokens)")

        if not self.fineweb_chunks:
            raise ValueError(f"No FineWeb chunks loaded from {fineweb_path}")
        if not self.composite_passages:
            raise ValueError(f"No composite passages loaded from {composite_passages_path}")

    def _sample_fineweb_window(self, rng: random.Random) -> list[int]:
        return rng.choice(self.fineweb_chunks)

    def _sample_composite_window(self, rng: random.Random) -> list[int]:
        W = self.cfg.fixed_window_size
        out: list[int] = []
        while len(out) < W:
            out.extend(rng.choice(self.composite_passages))
        return out[:W]

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid)
        cfg = self.cfg

        while True:
            if rng.random() < self.fineweb_ratio:
                ids = self._sample_fineweb_window(rng)
                source = "fineweb"
            else:
                ids = self._sample_composite_window(rng)
                source = "composite"

            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.ones(cfg.fixed_window_size, dtype=torch.bool)

            # Per-sample span mask, capped at target_n so we don't exceed
            # mask_ratio_max (each new span only counts unique additions).
            ratio = rng.uniform(cfg.mask_ratio_min, cfg.mask_ratio_max)
            mask_positions = torch.zeros(cfg.fixed_window_size, dtype=torch.bool)
            target_n = int(cfg.fixed_window_size * ratio)
            masked: set[int] = set()
            attempts = 0
            done = False
            while not done and attempts < cfg.fixed_window_size * 4:
                span_len = rng.randint(cfg.mask_span_min, cfg.mask_span_max)
                start = rng.randint(0, max(cfg.fixed_window_size - span_len, 0))
                for i in range(start, min(start + span_len, cfg.fixed_window_size)):
                    if i not in masked:
                        masked.add(i)
                        if len(masked) >= target_n:
                            done = True
                            break
                attempts += 1
            for p in masked:
                mask_positions[p] = True

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mask_positions": mask_positions,
                "source": source,
            }


class SentencePackedDataset(IterableDataset):
    """v1b variable-length dataset preserving document/passage boundaries.

    Differences from MixedReprDataset:
      - FineWeb: each sample is a contiguous window drawn from inside a
        single document (no cross-document chunking). Window length is
        sampled uniformly in [min_window_size, max_window_size].
      - Composite: pack consecutive passages with a separator token until
        the running length is ≥ min_window_size; stop before exceeding
        max_window_size. Each window is then a sequence of complete
        passages — no mid-passage truncation.
      - Yields variable-length samples; collate_variable() pads to the
        longest in batch using `cfg.pad_token_id`.
    """

    def __init__(
        self,
        cfg: ReprConfig,
        fineweb_path: Path,
        composite_passages_path: Path,
        fineweb_ratio: float = 0.5,
        seed: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.fineweb_ratio = fineweb_ratio
        self.seed = seed

        self.min_len = cfg.min_window_size
        self.max_len = cfg.max_window_size

        print(f"[data] loading FineWeb-edu documents from {fineweb_path}...")
        self.fineweb_docs = _load_fineweb_documents(fineweb_path, min_len=self.min_len)
        avg = sum(len(d) for d in self.fineweb_docs) / max(len(self.fineweb_docs), 1)
        print(f"[data]   {len(self.fineweb_docs):,} docs (avg {avg:.0f} tokens)")

        print(f"[data] loading composite passages from {composite_passages_path}...")
        self.composite_passages = _load_composite_passages(composite_passages_path)
        avg_c = sum(len(p) for p in self.composite_passages) / max(len(self.composite_passages), 1)
        print(f"[data]   {len(self.composite_passages):,} passages (avg {avg_c:.0f} tokens)")

        if not self.fineweb_docs:
            raise ValueError(
                f"No FineWeb documents ≥ {self.min_len} tokens loaded from {fineweb_path}"
            )
        if not self.composite_passages:
            raise ValueError(f"No composite passages loaded from {composite_passages_path}")

    def _sample_fineweb_window(self, rng: random.Random) -> list[int]:
        """Pick a doc, take a random-length window from a random offset within it."""
        doc = rng.choice(self.fineweb_docs)
        n = len(doc)
        win_len = rng.randint(self.min_len, min(self.max_len, n))
        start = rng.randint(0, n - win_len)
        return doc[start : start + win_len]

    def _sample_composite_window(self, rng: random.Random) -> list[int]:
        """Pack passages with separator token until ≥ min_len; cap at max_len.

        Each window contains complete passages — no mid-passage truncation.
        If a single passage exceeds max_len, truncate it.
        """
        out: list[int] = []
        sep = self.cfg.sep_token_id
        while len(out) < self.min_len:
            p = rng.choice(self.composite_passages)
            # If adding this passage would exceed max_len, stop unless we
            # haven't reached min_len yet (then truncate to fit).
            need_sep = bool(out)
            extra = (1 if need_sep else 0) + len(p)
            if len(out) + extra > self.max_len:
                if len(out) < self.min_len:
                    # Truncate this passage to fit, then break
                    space = self.max_len - len(out) - (1 if need_sep else 0)
                    if need_sep and space > 0:
                        out.append(sep)
                    out.extend(p[:max(space, 0)])
                break
            if need_sep:
                out.append(sep)
            out.extend(p)
        return out[: self.max_len]

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid)
        cfg = self.cfg

        while True:
            if rng.random() < self.fineweb_ratio:
                ids = self._sample_fineweb_window(rng)
                source = "fineweb"
            else:
                ids = self._sample_composite_window(rng)
                source = "composite"

            T = len(ids)
            input_ids = torch.tensor(ids, dtype=torch.long)
            attention_mask = torch.ones(T, dtype=torch.bool)

            # Per-sample span mask, with target_n derived from this sample's
            # actual length (not a padded length).
            ratio = rng.uniform(cfg.mask_ratio_min, cfg.mask_ratio_max)
            mask_positions = torch.zeros(T, dtype=torch.bool)
            target_n = int(T * ratio)
            masked: set[int] = set()
            attempts = 0
            done = False
            while not done and attempts < T * 4:
                span_len = rng.randint(cfg.mask_span_min, cfg.mask_span_max)
                start = rng.randint(0, max(T - span_len, 0))
                for i in range(start, min(start + span_len, T)):
                    if i not in masked:
                        masked.add(i)
                        if len(masked) >= target_n:
                            done = True
                            break
                attempts += 1
            for p in masked:
                mask_positions[p] = True

            yield {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "mask_positions": mask_positions,
                "source": source,
            }


def collate(samples: list[dict]) -> WindowBatch:
    """Stack a list of per-sample dicts into a WindowBatch."""
    input_ids = torch.stack([s["input_ids"] for s in samples])
    attention_mask = torch.stack([s["attention_mask"] for s in samples])
    mask_positions = torch.stack([s["mask_positions"] for s in samples])
    return WindowBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_positions=mask_positions,
    )


def collate_variable(samples: list[dict], pad_token_id: int) -> WindowBatch:
    """Pad variable-length samples to the longest in batch.

    Pad token id should be the model's EOS or a dedicated pad id; attention
    mask is False at padded positions so they don't contribute to recon CE
    and the encoder ignores them.
    """
    T = max(int(s["input_ids"].shape[0]) for s in samples)
    B = len(samples)

    input_ids = torch.full((B, T), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((B, T), dtype=torch.bool)
    mask_positions = torch.zeros((B, T), dtype=torch.bool)
    for i, s in enumerate(samples):
        t = int(s["input_ids"].shape[0])
        input_ids[i, :t] = s["input_ids"]
        attention_mask[i, :t] = s["attention_mask"]
        mask_positions[i, :t] = s["mask_positions"]
    return WindowBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        mask_positions=mask_positions,
    )


def make_dataloader(
    cfg: ReprConfig,
    fineweb_path: Path | str,
    composite_passages_path: Path | str,
    fineweb_ratio: float = 0.5,
    num_workers: int = 0,
    seed: int = 0,
) -> DataLoader:
    if cfg.use_variable_length:
        ds = SentencePackedDataset(
            cfg,
            Path(fineweb_path),
            Path(composite_passages_path),
            fineweb_ratio=fineweb_ratio,
            seed=seed,
        )
        pad_id = cfg.pad_token_id
        collate_fn = lambda samples: collate_variable(samples, pad_token_id=pad_id)
    else:
        ds = MixedReprDataset(
            cfg,
            Path(fineweb_path),
            Path(composite_passages_path),
            fineweb_ratio=fineweb_ratio,
            seed=seed,
        )
        collate_fn = collate
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )
