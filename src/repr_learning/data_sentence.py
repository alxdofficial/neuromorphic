"""v1g sentence-level shuffled-retrieval dataset.

Each example is a single FineWeb-edu document chunk of `chunk_size` tokens
(default 4096), split into sentences at terminator-token boundaries (".",
"!", "?"). At training time we randomly pick K sentences, mask ~80% of
their tokens, and expose a random fraction r ∈ [0, 1) of the masked
positions for the MaskGIT-style "revealed = previously predicted" channel.

The encoder ingests the full chunk; for each queried sentence the decoder
runs a SEPARATE forward pass that reconstructs the still-masked positions
given (memory, unmasked positions, revealed positions).

Yielded fields per example:
    input_ids       : [T_chunk] long — the full chunk (encoder input)
    attention_mask  : [T_chunk] bool — True at valid positions
    n_sentences     : int            — actual sentence count in this chunk
    query_idx       : [K] long       — which sentence indices we query
    query_starts    : [K] long       — start token offset of each queried sentence
    query_lengths   : [K] long       — length in tokens of each queried sentence
    query_input_ids : [K, T_sent_max] long — sentence token ids, right-padded
    mask_positions  : [K, T_sent_max] bool — True where token is masked (80%)
    reveal_positions: [K, T_sent_max] bool — True where masked AND revealed via GT
                                             (still-masked positions = mask & ~reveal)

The "still-masked" positions are the CE targets. Revealed positions go in
as GT embeddings (treated as previously-predicted) and are not part of
the loss. Per the user's spec: prediction at a still-masked position
sees unmasked + revealed + itself, never other still-masked positions.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

from .config import ReprConfig
from .data_real import _load_fineweb_documents


def _build_sentence_terminator_ids(tokenizer) -> set[int]:
    """Scan the tokenizer vocab once and return the set of token ids whose
    decoded text marks a sentence boundary.

    Criterion: decoded text contains `.`, `!`, or `?` and is short (≤4
    chars after strip). This catches tokens like ".", "?", "!", ". ", ".\n"
    while avoiding false positives like ".com" or numbers ("3.14").

    Misses some edge cases (e.g., terminator inside a longer token like
    "sentence."), but for FineWeb-edu prose the recall is high enough for
    v1g.0. We can swap in a tokenizer-aware sentence splitter later.
    """
    terminators: set[int] = set()
    # Llama-3.2 vocab is 128_256, but most special tokens live above 128_000.
    # Scan the natural-text portion only.
    vocab_size = min(getattr(tokenizer, "vocab_size", 128_000), 128_000)
    for tid in range(vocab_size):
        text = tokenizer.decode([tid])
        stripped = text.strip()
        if not stripped:
            continue
        last = stripped[-1]
        if last in ".!?" and len(stripped) <= 4:
            terminators.add(tid)
    return terminators


def _split_chunk_into_sentences(
    tokens: list[int],
    terminator_ids: set[int],
    min_sent_len: int,
    max_sent_len: int,
) -> list[tuple[int, int]]:
    """Find sentence offsets in a token-id chunk.

    Returns a list of (start, end_exclusive) tuples covering the chunk.
    Each tuple's length is in [min_sent_len, max_sent_len]:
      - Sentences shorter than min_sent_len are merged into the next sentence.
      - Sentences longer than max_sent_len are split at max_sent_len boundaries.

    The final fragment after the last terminator is included as a sentence
    only if it satisfies the length constraint.
    """
    if not tokens:
        return []

    raw_boundaries: list[int] = []  # token-id index AFTER each terminator
    for i, tid in enumerate(tokens):
        if tid in terminator_ids:
            raw_boundaries.append(i + 1)
    # Always treat the tail as a candidate boundary; the merge/drop step
    # below handles the case where the tail is too short.
    if not raw_boundaries or raw_boundaries[-1] != len(tokens):
        raw_boundaries.append(len(tokens))

    sentences: list[tuple[int, int]] = []
    start = 0
    for end in raw_boundaries:
        length = end - start
        if length < min_sent_len:
            # Merge into the next sentence by NOT advancing `start`.
            continue
        # Split long runs into max_sent_len chunks.
        while end - start > max_sent_len:
            sentences.append((start, start + max_sent_len))
            start += max_sent_len
        if end - start >= min_sent_len:
            sentences.append((start, end))
        start = end
    return sentences


@dataclass
class SentenceChunkBatch:
    """A batched v1g example, ready for the encoder + per-sentence decoder."""
    input_ids: Tensor          # [B, T_chunk] long — encoder input
    attention_mask: Tensor     # [B, T_chunk] bool
    # Per-example: K queried sentences
    query_starts: Tensor       # [B, K] long — sentence start offset in chunk
    query_lengths: Tensor      # [B, K] long — actual sentence length
    query_input_ids: Tensor    # [B, K, T_sent_max] long — sentence tokens, right-padded
    mask_positions: Tensor     # [B, K, T_sent_max] bool — True at masked tokens (the 80%)
    reveal_positions: Tensor   # [B, K, T_sent_max] bool — True at revealed (= previously predicted) tokens
    # Useful for diagnostics
    n_sentences: Tensor        # [B] long — total sentence count per chunk


def _sample_query_and_masks(
    sentences: list[tuple[int, int]],
    tokens: list[int],
    n_queries: int,
    mask_ratio: float,
    reveal_lo: float,
    reveal_hi: float,
    sentence_max_len: int,
    rng: random.Random,
) -> dict:
    """Sample K sentences, build per-sentence mask + reveal patterns.

    Returns dict shaped as in `SentenceChunkBatch` (single example, no batch dim).
    """
    K = n_queries
    # Indices of queryable sentences (need at least 5 tokens to mask meaningfully)
    candidates = [i for i, (s, e) in enumerate(sentences) if (e - s) >= 5]
    if len(candidates) < K:
        # Fall back to sampling with replacement if document is short on sentences
        chosen = [rng.choice(candidates) for _ in range(K)] if candidates else [0] * K
    else:
        chosen = rng.sample(candidates, K)
    # Shuffle to remove positional ordering signal across the K
    rng.shuffle(chosen)

    query_starts = torch.zeros(K, dtype=torch.long)
    query_lengths = torch.zeros(K, dtype=torch.long)
    query_input_ids = torch.zeros(K, sentence_max_len, dtype=torch.long)
    mask_positions = torch.zeros(K, sentence_max_len, dtype=torch.bool)
    reveal_positions = torch.zeros(K, sentence_max_len, dtype=torch.bool)

    for k, sent_idx in enumerate(chosen):
        if sent_idx < len(sentences):
            s, e = sentences[sent_idx]
        else:
            s, e = 0, 0
        L = min(e - s, sentence_max_len)
        query_starts[k] = s
        query_lengths[k] = L
        if L == 0:
            continue
        query_input_ids[k, :L] = torch.tensor(tokens[s : s + L], dtype=torch.long)

        # Random-position mask: pick floor(L * mask_ratio) unique positions.
        # Allow predicting from at least 1 unmasked position by clamping.
        n_mask = max(1, min(L - 1, int(round(L * mask_ratio))))
        mask_idx = rng.sample(range(L), n_mask)
        for p in mask_idx:
            mask_positions[k, p] = True

        # Reveal a uniform-random fraction r ∈ [reveal_lo, reveal_hi] of the
        # masked positions. These positions go to the decoder with GT embedding
        # AND mask_positions[k] still True (so they're identified as previously
        # predicted, not as unmasked input). The CE loss is computed on
        # `mask & ~reveal` positions only.
        r = rng.uniform(reveal_lo, reveal_hi)
        n_reveal = int(round(n_mask * r))
        if n_reveal > 0:
            reveal_idx = rng.sample(mask_idx, n_reveal)
            for p in reveal_idx:
                reveal_positions[k, p] = True

    return {
        "query_starts": query_starts,
        "query_lengths": query_lengths,
        "query_input_ids": query_input_ids,
        "mask_positions": mask_positions,
        "reveal_positions": reveal_positions,
    }


class SentenceChunkDataset(IterableDataset):
    """Iterable dataset yielding 4096-token chunks with sentence metadata
    and per-sentence mask/reveal patterns for v1g training.

    The tokenizer is loaded once at init and the sentence-terminator id set
    is computed up front (one-time vocab scan). Sentence boundaries are
    detected on the fly per chunk.
    """

    def __init__(
        self,
        fineweb_path: Path,
        tokenizer,
        chunk_size: int = 4096,
        n_queries: int = 3,
        mask_ratio: float = 0.8,
        reveal_lo: float = 0.0,
        reveal_hi: float = 0.9,
        sentence_min_len: int = 8,
        sentence_max_len: int = 80,
        seed: int = 0,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.n_queries = n_queries
        self.mask_ratio = mask_ratio
        self.reveal_lo = reveal_lo
        self.reveal_hi = reveal_hi
        self.sentence_min_len = sentence_min_len
        self.sentence_max_len = sentence_max_len
        self.seed = seed

        print(f"[data v1g] loading FineWeb-edu docs from {fineweb_path}...")
        # Need at least one full chunk; preferring 2× so we can pick random start offsets.
        self.docs = _load_fineweb_documents(fineweb_path, min_len=chunk_size)
        print(f"[data v1g]   {len(self.docs):,} docs with ≥{chunk_size} tokens")
        if not self.docs:
            raise ValueError(
                f"No FineWeb documents ≥{chunk_size} tokens in {fineweb_path}"
            )

        print(f"[data v1g] scanning tokenizer for sentence-terminator ids...")
        self.terminator_ids = _build_sentence_terminator_ids(tokenizer)
        print(f"[data v1g]   {len(self.terminator_ids):,} terminator token ids found")

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + wid * 100_003)
        cs = self.chunk_size

        while True:
            doc = rng.choice(self.docs)
            # Random start offset within the doc (for chunk diversity); fall
            # back to start=0 if doc is exactly chunk-sized.
            max_start = max(len(doc) - cs, 0)
            start = rng.randint(0, max_start) if max_start > 0 else 0
            tokens = doc[start : start + cs]

            sentences = _split_chunk_into_sentences(
                tokens,
                terminator_ids=self.terminator_ids,
                min_sent_len=self.sentence_min_len,
                max_sent_len=self.sentence_max_len,
            )
            if len(sentences) < self.n_queries:
                # Chunk doesn't have enough sentences (rare); skip + resample.
                continue

            qmasks = _sample_query_and_masks(
                sentences,
                tokens,
                n_queries=self.n_queries,
                mask_ratio=self.mask_ratio,
                reveal_lo=self.reveal_lo,
                reveal_hi=self.reveal_hi,
                sentence_max_len=self.sentence_max_len,
                rng=rng,
            )

            yield {
                "input_ids": torch.tensor(tokens, dtype=torch.long),
                "attention_mask": torch.ones(cs, dtype=torch.bool),
                "n_sentences": torch.tensor(len(sentences), dtype=torch.long),
                **qmasks,
            }


def collate_sentence_chunk(samples: list[dict]) -> SentenceChunkBatch:
    """Stack per-example dicts into a SentenceChunkBatch."""
    return SentenceChunkBatch(
        input_ids=torch.stack([s["input_ids"] for s in samples]),
        attention_mask=torch.stack([s["attention_mask"] for s in samples]),
        query_starts=torch.stack([s["query_starts"] for s in samples]),
        query_lengths=torch.stack([s["query_lengths"] for s in samples]),
        query_input_ids=torch.stack([s["query_input_ids"] for s in samples]),
        mask_positions=torch.stack([s["mask_positions"] for s in samples]),
        reveal_positions=torch.stack([s["reveal_positions"] for s in samples]),
        n_sentences=torch.stack([s["n_sentences"] for s in samples]),
    )


def make_sentence_chunk_dataloader(
    cfg: ReprConfig,
    fineweb_path: Path | str,
    tokenizer,
    chunk_size: int = 4096,
    n_queries: int = 3,
    mask_ratio: float = 0.8,
    reveal_lo: float = 0.0,
    reveal_hi: float = 0.9,
    sentence_min_len: int = 8,
    sentence_max_len: int = 80,
    num_workers: int = 0,
    seed: int = 0,
    batch_size: Optional[int] = None,
) -> DataLoader:
    ds = SentenceChunkDataset(
        Path(fineweb_path),
        tokenizer=tokenizer,
        chunk_size=chunk_size,
        n_queries=n_queries,
        mask_ratio=mask_ratio,
        reveal_lo=reveal_lo,
        reveal_hi=reveal_hi,
        sentence_min_len=sentence_min_len,
        sentence_max_len=sentence_max_len,
        seed=seed,
    )
    return DataLoader(
        ds,
        batch_size=batch_size if batch_size is not None else cfg.batch_size,
        num_workers=num_workers,
        collate_fn=collate_sentence_chunk,
        pin_memory=torch.cuda.is_available(),
    )
