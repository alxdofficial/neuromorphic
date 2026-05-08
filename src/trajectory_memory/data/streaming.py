"""Streaming long-document tokenizer + TBPTT chunker.

Wave 1 (long-doc TF NTP) processes documents that are much longer than
Llama's effective context. We:
  1. Tokenize each document.
  2. Concatenate documents with EOS separators (avoid mixing into one
     mega-document; keep per-document boundaries clean).
  3. Yield fixed-size sequences of length `chunk_tokens = D * T_window`,
     each ready to drop into TBPTT's `windows` tensor.

Document boundaries inside a chunk: align sequences so each chunk is
strictly within one document where possible. Docs shorter than chunk_tokens
get padded; docs longer get split across sequences but never re-mixed
mid-chunk (per plan §4.5: "align sequence boundaries with documents").
"""

from __future__ import annotations

from typing import Iterable, Iterator

import torch
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from src.trajectory_memory.data.tokenizer import get_tokenizer


def pack_documents(
    docs: Iterable[str],
    *,
    chunk_tokens: int,
    tokenizer: PreTrainedTokenizerBase | None = None,
    pad_id: int | None = None,
    drop_short: bool = False,
) -> Iterator[Tensor]:
    """Tokenize and pack documents into fixed-size [chunk_tokens] sequences.

    Each document is tokenized independently, then split into chunks of
    `chunk_tokens` tokens. Trailing partial chunks are either dropped
    (drop_short=True) or padded (drop_short=False, with `pad_id`).

    Documents are NOT mixed within a chunk — each chunk comes from a
    single document. This preserves "memory resets per training sequence
    aligned to document boundary" semantic (plan §4.5).

    Args:
        docs: iterable of document text strings.
        chunk_tokens: total sequence length per chunk (= D * T_window).
        tokenizer: project tokenizer.
        pad_id: token to pad with (default tokenizer.pad_token_id).
        drop_short: drop the trailing partial chunk of each document.

    Yields:
        [chunk_tokens] int64 tensors.
    """
    tok = tokenizer or get_tokenizer()
    pad = pad_id if pad_id is not None else tok.pad_token_id

    for doc in docs:
        if not doc:
            continue
        ids = tok.encode(doc, add_special_tokens=True)  # adds BOS
        for start in range(0, len(ids), chunk_tokens):
            chunk = ids[start : start + chunk_tokens]
            if len(chunk) < chunk_tokens:
                if drop_short:
                    continue
                chunk = chunk + [pad] * (chunk_tokens - len(chunk))
            yield torch.tensor(chunk, dtype=torch.int64)


class StreamingTokenChunker:
    """Wraps `pack_documents` for use as a torch IterableDataset.

    Given a generator of documents, yields chunks of [chunk_tokens] suitable
    for TBPTT. Stateless — just wraps `pack_documents`.
    """

    def __init__(
        self,
        chunk_tokens: int,
        *,
        tokenizer: PreTrainedTokenizerBase | None = None,
        pad_id: int | None = None,
        drop_short: bool = False,
    ):
        self.chunk_tokens = chunk_tokens
        self.tokenizer = tokenizer or get_tokenizer()
        self.pad_id = pad_id if pad_id is not None else self.tokenizer.pad_token_id
        self.drop_short = drop_short

    def chunk(self, docs: Iterable[str]) -> Iterator[Tensor]:
        return pack_documents(
            docs,
            chunk_tokens=self.chunk_tokens,
            tokenizer=self.tokenizer,
            pad_id=self.pad_id,
            drop_short=self.drop_short,
        )
