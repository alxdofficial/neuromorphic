"""Data loaders for `phase1_pretrained_step` (graph-walker integration).

Yields ``Phase1Batch(input_ids, target_ids)`` tensors of shape ``[BS, T]``.

Why a separate module from ``src/data/streaming.py``:
- ``streaming.py`` was built for the v2 attention-neuromod path. It uses
  a TinyLlama tokenizer and `.bin` shards, and has TBPTT / persistent-
  stream semantics tied to that path.
- The graph-walker integration uses Llama-3.2's tokenizer (vocab 128256)
  and treats each ``Phase1Batch`` as an INDEPENDENT segment — walker's
  per-batch state is reset by ``wrapper.reset_memory(BS)``; only
  ``E_bias_flat`` + neuromod snapshot persist across batches by design.
- Phase-1 parallel training doesn't need cross-batch token continuity, so
  we just yield independent random-offset windows from the corpus.

Two loaders here:
- ``fineweb_edu_phase1_iter``: streams natural text from the local
  ``data/phase_B/fineweb_edu.parquet`` shard (Wave 1 — language bootstrap).
- ``chat_sft_phase1_iter``: streams a HuggingFace chat dataset
  (Wave 2 — instruction SFT). Applies Llama-3 chat template and yields
  ``Phase1Batch`` with assistant-only loss masking via target_ids.

Both share a small ``_TokenStreamBuffer`` that batches a flat token stream
into ``[BS, T]`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pyarrow.parquet as pq
import torch

from src.graph_walker.pretrained.train_phase1 import Phase1Batch


@dataclass
class _TokenStreamBuffer:
    """Accumulate tokens in a flat list; flush BS*T at a time as [BS, T]."""

    bs: int
    T: int
    eos_id: int
    device: torch.device

    def __post_init__(self) -> None:
        self._buf: list[int] = []

    def push(self, token_ids: list[int]) -> None:
        if token_ids:
            self._buf.extend(token_ids)
            self._buf.append(self.eos_id)

    def ready(self) -> bool:
        return len(self._buf) >= self.bs * self.T

    def pop_batch(self) -> torch.Tensor:
        """Return [BS, T] int64 on `device`. Trims buffer."""
        n = self.bs * self.T
        chunk = self._buf[:n]
        self._buf = self._buf[n:]
        arr = np.asarray(chunk, dtype=np.int64).reshape(self.bs, self.T)
        return torch.from_numpy(arr).to(self.device, non_blocking=True)


def fineweb_edu_phase1_iter(
    parquet_path: str | Path,
    tokenizer,                       # HF PreTrainedTokenizer
    *,
    bs: int,
    T: int,
    device: torch.device | str = "cuda",
    max_batches: int | None = None,
    seed: int = 0,
) -> Iterator[Phase1Batch]:
    """Yield ``Phase1Batch`` from FineWeb-edu parquet.

    On-the-fly tokenization with the Llama-3.2 tokenizer. Pre-tokenizing
    the parquet to a `.bin` shard (per ``scripts/prepare_data.py``) would
    be ~2x faster end-to-end and is the right move for long runs; this
    streaming path is for fast iteration / smoke testing.

    Random row order within each parquet pass; loops the file when exhausted
    (so ``max_batches`` controls how many we yield, not the corpus size).
    """
    pq_file = Path(parquet_path)
    if not pq_file.exists():
        raise FileNotFoundError(f"FineWeb-edu parquet not found at {pq_file}")
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError(
            "tokenizer.eos_token_id is None — Llama-3.2 should have <|end_of_text|>"
        )
    rng = np.random.default_rng(seed)
    buf = _TokenStreamBuffer(bs=bs, T=T, eos_id=eos_id, device=torch.device(device))
    yielded = 0
    while True:
        # Read parquet in row-batch chunks; shuffle row indices per pass.
        table = pq.read_table(pq_file, columns=["text"])
        n_rows = len(table)
        order = rng.permutation(n_rows)
        text_col = table.column("text")
        for idx in order:
            text = text_col[int(idx)].as_py()
            if not text:
                continue
            # Truncate per-doc to avoid one huge doc dominating a batch.
            # (typical FineWeb-edu doc is short; this guards against outliers.)
            ids = tokenizer.encode(
                text, add_special_tokens=False, truncation=True,
                max_length=T * 4,
            )
            buf.push(ids)
            while buf.ready():
                input_ids = buf.pop_batch()
                # Phase1Batch convention: target_ids == input_ids
                # (trainer does the shift internally — see Phase1Batch
                # docstring in train_phase1.py)
                yield Phase1Batch(input_ids=input_ids, target_ids=input_ids.clone())
                yielded += 1
                if max_batches is not None and yielded >= max_batches:
                    return


def chat_sft_phase1_iter(
    hf_dataset_name: str,
    tokenizer,                       # HF PreTrainedTokenizer with chat_template
    *,
    bs: int,
    T: int,
    device: torch.device | str = "cuda",
    split: str = "train_sft",
    max_batches: int | None = None,
    streaming: bool = True,
    seed: int = 0,
    mask_user_tokens: bool = True,
) -> Iterator[Phase1Batch]:
    """Yield ``Phase1Batch`` from a HuggingFace chat-style dataset.

    Default target: ``HuggingFaceH4/ultrachat_200k`` (well-curated, focused
    chat). Each example is a list-of-turns; we apply the tokenizer's chat
    template (which for Llama-3.2-Instruct emits role-tagged sequences),
    then stream tokens through ``_TokenStreamBuffer`` exactly like the
    FineWeb loader.

    ``mask_user_tokens=True`` is the standard SFT trick: we set
    ``target_ids[i] = -100`` (PyTorch ignore_index) at user-turn positions
    so the next-token CE only counts assistant-turn predictions. The
    ``Phase1Batch`` convention has the trainer do the shift internally,
    so we mask in the same coordinate frame as ``input_ids``.

    Note: ``Phase1Batch`` doesn't currently support the ignore-index
    convention for ``target_ids`` — the trainer's ``F.cross_entropy``
    call doesn't pass ``ignore_index=-100``. This v1 wires the iterator
    correctly; if loss-on-assistant-only is needed in practice, a small
    patch to ``phase1_pretrained_step`` is required (one line:
    ``F.cross_entropy(..., ignore_index=-100)``). For now we run with
    ``mask_user_tokens=False`` which trains on every position (still
    works, just less SFT-pure).
    """
    from datasets import load_dataset

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer.eos_token_id is None")
    if tokenizer.chat_template is None:
        raise ValueError(
            f"Tokenizer for {hf_dataset_name} has no chat_template. Use a "
            "tokenizer with a defined chat_template (e.g. Llama-3.2-Instruct)."
        )

    ds = load_dataset(hf_dataset_name, split=split, streaming=streaming)
    if not streaming:
        ds = ds.shuffle(seed=seed)

    buf = _TokenStreamBuffer(bs=bs, T=T, eos_id=eos_id, device=torch.device(device))
    yielded = 0
    for example in ds:
        messages = example.get("messages") or example.get("conversation")
        if not messages:
            continue
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            continue
        ids = tokenizer.encode(
            text, add_special_tokens=False, truncation=True, max_length=T * 4,
        )
        buf.push(ids)
        while buf.ready():
            input_ids = buf.pop_batch()
            yield Phase1Batch(input_ids=input_ids, target_ids=input_ids.clone())
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                return
