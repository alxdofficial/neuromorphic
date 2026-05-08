"""Data loaders for `phase1_pretrained_step` (graph-walker integration).

Yields ``Phase1Batch(input_ids, target_ids)`` tensors of shape ``[BS, T]``.

Why a separate module from ``src/data/streaming.py``:
- ``streaming.py`` was built for the v2 attention-neuromod path. It uses
  a TinyLlama tokenizer and `.bin` shards, and has TBPTT / persistent-
  stream semantics tied to that path.
- The graph-walker integration uses Llama-3.2's tokenizer (vocab 128256)
  and treats each ``Phase1Batch`` as an INDEPENDENT segment — walker's
  per-batch state is reset by ``model.begin_segment(BS)``; only
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


def pretokenized_phase1_iter(
    bin_path: str | Path,
    *,
    bs: int,
    T: int,
    device: torch.device | str = "cuda",
    max_batches: int | None = None,
    seed: int = 0,
) -> "Iterator[Phase1Batch]":
    """Yield Phase1Batch from a memory-mapped flat int32 token file.

    The companion preprocess scripts (``scripts/preprocess_fineweb_edu_llama32.py``,
    ``scripts/preprocess_ultrachat_llama32.py``) write a ``.bin`` file of
    flat int32 tokens with EOS separators. This iterator memory-maps that
    file and yields random T-token windows as ``[bs, T]`` int64 tensors,
    one batch at a time.

    If a sibling ``{bin_path}.mask.bin`` exists (uint8, aligned 1:1 with
    ``.bin``), it's used as an assistant-only loss mask: ``target_ids``
    is set to -100 (PyTorch ``ignore_index``) at positions where the
    mask byte is 0. Without it ``target_ids = input_ids`` (whole-text
    LM, correct for FineWeb but WRONG for chat data — chat datasets MUST
    be re-preprocessed with the latest preprocess script to ship the
    mask sidecar).

    Memory ceiling: just the memmap (which the OS pages in/out) + one
    ``[bs, T]`` int64 batch (~bs*T*8 bytes). The full corpus never sits
    in process RAM.

    Random offsets per batch (uniform). Each batch is iid — there's no
    cross-batch token continuity, which matches phase-1's design (walker
    state is reset per batch via ``model.begin_segment(BS)``; only
    ``E_bias_flat`` + neuromod snapshot persist by design).
    """
    bin_p = Path(bin_path)
    if not bin_p.exists():
        raise FileNotFoundError(
            f"pretokenized bin not found at {bin_p} — run "
            f"scripts/preprocess_*.py to produce it"
        )
    arr = np.memmap(bin_p, dtype=np.int32, mode="r")
    n_tokens = arr.shape[0]
    span = bs * T
    if n_tokens < span:
        raise ValueError(
            f"pretokenized file has {n_tokens} tokens, < bs*T={span}"
        )
    mask_p = bin_p.parent / (bin_p.name + ".mask.bin")
    if mask_p.exists():
        mask_arr = np.memmap(mask_p, dtype=np.uint8, mode="r")
        if mask_arr.shape[0] != n_tokens:
            raise ValueError(
                f"mask sidecar {mask_p} has {mask_arr.shape[0]} bytes; "
                f"expected {n_tokens} (1:1 aligned with .bin)"
            )
        print(f"[pretokenized_phase1_iter] using assistant-mask sidecar "
              f"{mask_p} ({100 * mask_arr.sum() / max(n_tokens, 1):.1f}% "
              "supervised)")
    else:
        mask_arr = None
    rng = np.random.default_rng(seed)
    yielded = 0
    dev = torch.device(device)
    while True:
        # Sample bs starting offsets, slice T tokens each, stack.
        offsets = rng.integers(0, n_tokens - T + 1, size=bs)
        rows = np.empty((bs, T), dtype=np.int64)
        if mask_arr is not None:
            mrows = np.empty((bs, T), dtype=np.bool_)
            for r, off in enumerate(offsets):
                rows[r] = arr[int(off):int(off) + T]
                mrows[r] = mask_arr[int(off):int(off) + T].astype(np.bool_)
            input_ids = torch.from_numpy(rows).to(dev, non_blocking=True)
            mask_t = torch.from_numpy(mrows).to(dev, non_blocking=True)
            target_ids = torch.where(
                mask_t, input_ids, torch.full_like(input_ids, -100),
            )
        else:
            for r, off in enumerate(offsets):
                rows[r] = arr[int(off):int(off) + T]
            input_ids = torch.from_numpy(rows).to(dev, non_blocking=True)
            target_ids = input_ids.clone()
        yield Phase1Batch(input_ids=input_ids, target_ids=target_ids)
        yielded += 1
        if max_batches is not None and yielded >= max_batches:
            return


@dataclass
class _TokenStreamBuffer:
    """Accumulate tokens in a flat list; flush BS*T at a time as [BS, T].

    Also maintains a parallel "supervised-target" mask stream. By
    default every token is supervised (mask=True); the chat-SFT loader
    sets mask=False for non-assistant tokens (user, system, role
    headers, padding) so they're excluded from CE via ignore_index.
    """

    bs: int
    T: int
    eos_id: int
    device: torch.device

    def __post_init__(self) -> None:
        self._buf: list[int] = []
        self._mask_buf: list[bool] = []

    def push(
        self,
        token_ids: list[int],
        target_mask: list[bool] | None = None,
    ) -> None:
        if not token_ids:
            return
        if target_mask is None:
            # Default: every pushed token is a valid supervised target.
            target_mask = [True] * len(token_ids)
        if len(target_mask) != len(token_ids):
            raise ValueError(
                f"target_mask len {len(target_mask)} != "
                f"token_ids len {len(token_ids)}"
            )
        self._buf.extend(token_ids)
        self._mask_buf.extend(target_mask)
        # The trailing eos_id is a document boundary, not assistant
        # content — keep it in inputs but don't supervise.
        self._buf.append(self.eos_id)
        self._mask_buf.append(False)

    def ready(self) -> bool:
        return len(self._buf) >= self.bs * self.T

    def pop_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(input_ids[BS, T], target_ids[BS, T])`` on ``device``.

        ``target_ids`` is ``input_ids`` with -100 (PyTorch
        ``ignore_index``) wherever the target_mask was False — that is,
        wherever the corresponding token is NOT a supervised assistant
        position. ``F.cross_entropy(..., ignore_index=-100)`` then
        skips those positions exactly. The trainer is responsible for
        passing ``ignore_index=-100``; without it the -100 values would
        be treated as out-of-vocab token ids and crash.
        """
        n = self.bs * self.T
        chunk = self._buf[:n]
        mask_chunk = self._mask_buf[:n]
        self._buf = self._buf[n:]
        self._mask_buf = self._mask_buf[n:]
        arr = np.asarray(chunk, dtype=np.int64).reshape(self.bs, self.T)
        mask_arr = np.asarray(mask_chunk, dtype=bool).reshape(self.bs, self.T)
        target_arr = np.where(mask_arr, arr, np.int64(-100))
        input_t = torch.from_numpy(arr).to(self.device, non_blocking=True)
        target_t = torch.from_numpy(target_arr).to(self.device, non_blocking=True)
        return input_t, target_t


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
                input_ids, target_ids = buf.pop_batch()
                # FineWeb-edu has no role structure — every token is a
                # valid supervised target, target_ids == input_ids
                # (Phase1Batch convention: trainer does the shift
                # internally, see Phase1Batch docstring).
                yield Phase1Batch(input_ids=input_ids, target_ids=target_ids)
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
        # Build the (token_ids, target_mask) stream by templating each
        # message INDIVIDUALLY and using the cumulative-length boundary
        # detection trick (same approach as preprocess_wildchat_llama32):
        # apply_chat_template(messages[:i+1]) - apply_chat_template(messages[:i])
        # gives the token range owned by message i. We mask everything
        # whose owning role != "assistant" — that drops user, system,
        # AND the chat-template injected role headers / EOT markers
        # within those non-assistant turns. Assistant role headers and
        # EOT markers also fall in the "assistant" range and DO get
        # supervised, which matches standard chat-SFT practice (the
        # assistant turn IS the model's output, headers and all).
        try:
            full_ids = tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            )
        except Exception:
            continue
        if not isinstance(full_ids, list):
            full_ids = list(full_ids)
        # Drop sessions that are too long to fit a useful chunk; the
        # streaming buffer would shred them across batch boundaries
        # which would scramble the role-mask alignment.
        if len(full_ids) > T * 4:
            continue
        target_mask: list[bool] = [False] * len(full_ids)
        prev_end = 0
        ok = True
        for i, m in enumerate(messages):
            role = m.get("role") or m.get("from")
            try:
                cum_ids = tokenizer.apply_chat_template(
                    messages[:i + 1], tokenize=True,
                    add_generation_prompt=False,
                )
            except Exception:
                ok = False
                break
            cur_end = len(cum_ids)
            if role == "assistant":
                for k in range(prev_end, min(cur_end, len(full_ids))):
                    target_mask[k] = True
            prev_end = cur_end
        if not ok:
            continue
        # Sanity: per-turn templating may drift from full-sequence
        # templating; if so the role-mask alignment is unreliable, skip.
        if prev_end != len(full_ids):
            continue
        buf.push(full_ids, target_mask=target_mask)
        while buf.ready():
            input_ids, target_ids = buf.pop_batch()
            # Assistant-only mask: target_ids has -100 (PyTorch
            # ignore_index) at user/system/template positions, so CE
            # only counts assistant-turn predictions. The trainer
            # (phase1_pretrained_step) passes ignore_index=-100 to
            # F.cross_entropy. Without that the -100 values would be
            # treated as out-of-vocab token ids and the call would
            # crash — so the assistant mask and the trainer's
            # ignore_index must stay in sync.
            yield Phase1Batch(input_ids=input_ids, target_ids=target_ids)
            yielded += 1
            if max_batches is not None and yielded >= max_batches:
                return
