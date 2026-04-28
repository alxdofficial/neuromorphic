"""
Persistent parallel stream data loading for TBPTT training.

Key design:
- BS independent persistent streams (not independent sequences)
- Each stream is a continuous flow of documents separated by <|endoftext|>
- Model state (LM scan carries + memory graph W/decay/hebbian) persists
  across TBPTT chunks; memory graph state also persists across document
  boundaries (lifelong memory).
- Different streams hit document boundaries at different positions.

Two dataset backends:
- TokenShardDataset: pre-tokenized .bin shards (fast, recommended)
- PersistentStreamDataset: on-the-fly tokenization from text (fallback)

This is fundamentally different from transformer data loading where
each batch item is an independent sequence.
"""

import logging
import os
import threading
import time
from queue import Queue

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from typing import Callable, Optional, Iterator, List, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

from .config import DatasetConfig, DATASET_CONFIGS
from .tokenizer import get_tokenizer


def _effective_streaming(config: DatasetConfig) -> bool:
    """Return whether to use streaming mode for this dataset.

    Datasets with ``download_first=True`` are loaded as map-style datasets
    so that HuggingFace downloads (and caches) them locally.  This avoids
    opening hundreds of concurrent HTTP connections when many parallel
    streams are created.
    """
    if config.download_first:
        return False
    return config.streaming


def _load_dataset_from_config(config: DatasetConfig):
    """Load a dataset from either a local file or the HuggingFace hub.

    If ``config.hf_path`` points to an existing local file (parquet, jsonl,
    csv, etc.), the file is loaded directly via ``load_dataset``.  Otherwise
    the path is treated as a HuggingFace hub identifier.
    """
    path = config.hf_path
    if os.path.exists(path):
        ext = os.path.splitext(path)[1].lstrip(".")
        logger.info("Loading local dataset: %s (format=%s)", path, ext)
        return load_dataset(ext, data_files=path, split="train"), False
    # Remote HuggingFace dataset
    use_streaming = _effective_streaming(config)
    ds = load_dataset(
        path, config.hf_name, split=config.split, streaming=use_streaming,
    )
    return ds, use_streaming


@dataclass
class StreamBatch:
    """A batch of tokens from persistent streams.

    .. warning::
        ``target_ids`` is **pre-shifted** here (already ``input_ids[:, 1:]``
        padded). The graph-walker pretrained training path
        (``Phase1Batch`` in ``src/graph_walker/pretrained/train_phase1.py``)
        expects ``target_ids`` to be **unshifted** — it does the shift
        internally (logits[:, :-1] vs target_ids[:, 1:]). Passing this
        StreamBatch's ``target_ids`` directly into Phase1Batch would train
        against ``t+2`` targets. When wiring this dataloader to graph-walker,
        either re-pass ``input_ids`` as ``Phase1Batch.target_ids`` (the
        standard NTP convention) OR shift this batch's ``target_ids`` back.
    """
    input_ids: torch.Tensor      # [BS, T] - input tokens
    target_ids: torch.Tensor     # [BS, T] - target tokens (pre-shifted by 1)
    prev_token: torch.Tensor     # [BS] - last token from previous chunk (for reset detection)


class _PrefetchIter:
    """Background thread that pre-fills a queue of batches for async overlap."""

    def __init__(self, gen: Iterator, prefetch: int = 2):
        self._queue: Queue = Queue(maxsize=prefetch)
        self._gen = gen
        self._thread = threading.Thread(target=self._fill, daemon=True)
        self._thread.start()

    def _fill(self):
        try:
            for item in self._gen:
                self._queue.put(item)
        except Exception as e:
            self._queue.put(e)
        finally:
            self._queue.put(None)  # sentinel

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is None:
            raise StopIteration
        if isinstance(item, Exception):
            raise item
        return item


class TokenShardDataset(IterableDataset):
    """Pre-tokenized shard dataset with pinned memory and prefetch.

    Reads from a memory-mapped .bin file of uint16 token IDs (produced by
    prepare_data.py). BS streams read from evenly-spaced offsets with
    wrap-around. Eliminates tokenization overhead and uses pinned memory
    for non-blocking CPU→GPU transfer.

    Usage:
        dataset = TokenShardDataset(
            shard_path="data/pile/pile_train.bin",
            eos_token_id=2,
            batch_size=16,
            seq_length=1024,
        )
        for batch in dataset:
            batch.input_ids  # [BS, T], pinned memory
    """

    def __init__(
        self,
        shard_path: str,
        eos_token_id: int,
        batch_size: int = 16,
        seq_length: int = 256,
        seed: int = 42,
        max_steps: Optional[int] = None,
        prefetch: int = 2,
    ):
        self.shard_path = shard_path
        self.eos_token_id = eos_token_id
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed
        self.max_steps = max_steps
        self.prefetch = prefetch

        # Memory-map the shard (read-only, OS page cache handles prefetch)
        self.tokens = np.memmap(shard_path, dtype=np.uint16, mode="r")
        self.shard_len = len(self.tokens)

        if self.shard_len < seq_length + 1:
            raise ValueError(
                f"Shard has {self.shard_len} tokens, need at least {seq_length + 1}"
            )

        self.step_count = 0
        self.consumed_step: Optional[int] = None  # set by mark_consumed()
        self._consumed_prev_tokens: Optional[torch.Tensor] = None
        self.prev_tokens: Optional[torch.Tensor] = None
        self._resume_step: int = 0  # CUMULATIVE step offset for resume
        self._resume_base: int = 0  # set in _generate; init here for early state_dict
        self._resume_prev_tokens: Optional[torch.Tensor] = None
        # Compat with PersistentStreamDataset monitoring
        self.stream_restarts_total = 0
        self.stream_restarts_last_batch = 0
        self.streams_exhausted_last_batch = 0

    def state_dict(self) -> dict:
        """Return serialisable snapshot for checkpointing.

        Saves the CUMULATIVE step count (resume_base + consumed_this_session)
        so that a subsequent resume lands at the correct absolute stream
        position. Also saves the consumer-side ``prev_tokens`` (not the
        producer's prefetch-ahead version) so reset_mask on resume is correct.

        Callers MUST invoke ``mark_consumed(n, prev_tokens)`` before
        checkpointing.
        """
        # Cumulative = what we resumed from + what we consumed this session.
        consumed = (self.consumed_step if self.consumed_step is not None
                    else self.step_count)
        cumulative = self._resume_base + consumed
        # Consumer-side prev_tokens if available, else producer's (best effort).
        prev = (self._consumed_prev_tokens if self._consumed_prev_tokens is not None
                else self.prev_tokens)
        return {
            "step_count": cumulative,
            "prev_tokens": (prev.clone() if prev is not None else None),
        }

    def mark_consumed(self, consumed: int,
                      prev_tokens: torch.Tensor | None = None) -> None:
        """Record how many batches + the last prev_token the consumer saw.

        Decouples checkpointed state from the prefetch-ahead internal
        counter AND the prefetch-ahead prev_tokens, both of which can
        run ahead of what the training loop actually processed.
        """
        self.consumed_step = consumed
        if prev_tokens is not None:
            self._consumed_prev_tokens = prev_tokens.clone()

    def load_state_dict(self, state: dict) -> None:
        """Restore dataset position from a checkpoint.

        ``state["step_count"]`` is CUMULATIVE (across all prior sessions).
        ``_generate`` fast-forwards positions by this many T-token strides
        from the seed-derived starting positions. ``prev_tokens`` is the
        consumer-side value from the checkpoint so the first resumed chunk
        gets the correct reset_mask.
        """
        self._resume_step = state.get("step_count", 0)
        saved_prev = state.get("prev_tokens")
        if saved_prev is not None:
            self._resume_prev_tokens = saved_prev.clone()

    def _generate(self) -> Iterator[StreamBatch]:
        """Core generation loop (wrapped by prefetch thread).

        Each step reads T+1 tokens from the shard (for the input/target
        shift) but advances the cursor by T (not T+1) so the last target
        of chunk k overlaps with the first input of chunk k+1.  Previous
        versions advanced by T+1, silently skipping one token per chunk.
        """
        BS = self.batch_size
        T = self.seq_length
        shard_len = self.shard_len
        read_size = T + 1  # need +1 for target shift

        # Distribute streams evenly across shard with seed-based jitter
        rng = np.random.RandomState(self.seed)
        stride = shard_len // BS
        positions = np.array(
            [(i * stride + rng.randint(0, max(stride, 1))) % shard_len for i in range(BS)],
            dtype=np.int64,
        )

        # Fast-forward positions if resuming from a checkpoint. The cursor
        # stored in the checkpoint is CUMULATIVE (total batches consumed
        # across all prior sessions). Each batch advances by T tokens.
        resume_step = self._resume_step
        self._resume_base = resume_step  # for state_dict's cumulative calc
        if resume_step > 0:
            positions = (positions + resume_step * T) % shard_len
            self._resume_step = 0  # consumed

        # Initialize prev_tokens. On resume, use the saved tensor so the
        # first resumed batch doesn't see a spurious EOS carry. Otherwise
        # start with EOS which (correctly) triggers a reset on chunk 0.
        if self._resume_prev_tokens is not None and self._resume_prev_tokens.shape == (BS,):
            self.prev_tokens = self._resume_prev_tokens.clone()
            self._resume_prev_tokens = None
        else:
            self.prev_tokens = torch.full((BS,), self.eos_token_id, dtype=torch.long)
        # step_count tracks batches yielded in this session (starts at 0,
        # compared against max_steps which is the remaining step count).
        self.step_count = 0

        # Pre-allocate pinned buffer for async CPU→GPU transfer
        use_pin = torch.cuda.is_available()
        buf = torch.empty(BS, read_size, dtype=torch.long, pin_memory=use_pin)

        while True:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            # Fill buffer from memory-mapped array. Read T+1 tokens at the
            # current position, then advance the cursor by T (not T+1) so
            # the next chunk's input[0] == this chunk's target[-1]. This is
            # the "overlapping read window" pattern that makes the stream
            # contiguous.
            for i in range(BS):
                pos = int(positions[i])
                end = pos + read_size
                if end <= shard_len:
                    chunk = self.tokens[pos:end]
                else:
                    # Wrap around shard boundary
                    chunk = np.concatenate([
                        self.tokens[pos:],
                        self.tokens[: end - shard_len],
                    ])
                buf[i] = torch.from_numpy(chunk.astype(np.int64))
                positions[i] = (pos + T) % shard_len  # advance by T, not T+1

            batch = StreamBatch(
                input_ids=buf[:, :-1].clone(),
                target_ids=buf[:, 1:].clone(),
                prev_token=self.prev_tokens.clone(),
            )

            # Update prev_tokens for next chunk (last input token).
            # buf[:, -2] == input_ids[:, -1] since input_ids = buf[:, :-1].
            self.prev_tokens = buf[:, -2].clone()
            self.step_count += 1

            yield batch

    def __iter__(self) -> Iterator[StreamBatch]:
        if self.prefetch > 0:
            return _PrefetchIter(self._generate(), prefetch=self.prefetch)
        return self._generate()

    def reset(self):
        """Reset dataset state."""
        self.step_count = 0
        self.prev_tokens = None

    def reset_streams(self):
        """Alias for reset() — compatibility with PersistentStreamDataset interface."""
        self.reset()

    def monitor_stats(self) -> dict:
        return {
            "stream_restarts_total": 0,
            "stream_restarts_last_batch": 0,
            "streams_exhausted_last_batch": 0,
        }


_MAX_RETRIES = 10
_BASE_WAIT = 5  # seconds

_NETWORK_ERROR_FRAGMENTS = (
    "Cannot send a request",
    "client has been closed",
    "ConnectionError",
    "ReadTimeout",
    "RemoteDisconnected",
    "Server disconnected",
    "IncompleteRead",
    "ConnectionReset",
    "ChunkedEncodingError",
)


def _is_network_error(exc: BaseException) -> bool:
    """Return True if *exc* looks like a transient HTTP/network failure."""
    msg = str(exc)
    return any(frag in msg for frag in _NETWORK_ERROR_FRAGMENTS)


class DocumentStream:
    """
    Iterator over a single document stream.

    Continuously yields tokens from documents, with EOT separators.
    Used internally by PersistentStreamDataset.
    """

    def __init__(
        self,
        dataset_iter: Iterator[Dict[str, Any]],
        tokenizer: PreTrainedTokenizerFast,
        text_column: str = "text",
        buffer_size: int = 8192,
        iter_factory: Optional[Callable[..., Iterator[Dict[str, Any]]]] = None,
    ):
        """
        Args:
            dataset_iter: Iterator over dataset examples
            tokenizer: Tokenizer for encoding text
            text_column: Column name containing text data
            buffer_size: Number of tokens to buffer before yielding
            iter_factory: Callable that builds a fresh iterator. Accepts an
                optional ``retry_count`` kwarg so each retry gets a different
                shuffle seed (prevents restarting from the same prefix on
                transient network errors).
        """
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.buffer_size = buffer_size
        self._iter_factory = iter_factory
        self._retry_count = 0

        self.token_buffer: List[int] = []
        self.exhausted = False
        self.eos_token_id = tokenizer.eos_token_id

    def _fill_buffer(self):
        """Fill token buffer from dataset, retrying on transient network errors."""
        while len(self.token_buffer) < self.buffer_size and not self.exhausted:
            try:
                example = next(self.dataset_iter)
                text = example[self.text_column]
                if text and text.strip():
                    tokens = self.tokenizer.encode(text, add_special_tokens=False)
                    self.token_buffer.extend(tokens)
                    self.token_buffer.append(self.eos_token_id)
            except StopIteration:
                self.exhausted = True
                break
            except Exception as exc:
                if not _is_network_error(exc) or self._iter_factory is None:
                    raise
                # Transient network error — rebuild the iterator and retry.
                # Each retry increments _retry_count so the factory can use
                # a different shuffle seed, avoiding restarting from the same
                # prefix (which would silently duplicate data).
                for attempt in range(1, _MAX_RETRIES + 1):
                    wait = _BASE_WAIT * (2 ** (attempt - 1))
                    logger.warning(
                        "Network error in DocumentStream: %s — "
                        "retrying in %ds (attempt %d/%d)",
                        exc, wait, attempt, _MAX_RETRIES,
                    )
                    time.sleep(wait)
                    try:
                        self._retry_count += 1
                        self.dataset_iter = self._iter_factory(
                            retry_count=self._retry_count)
                        # Iterator rebuilt successfully, continue filling
                        break
                    except Exception as rebuild_exc:
                        if attempt == _MAX_RETRIES:
                            raise RuntimeError(
                                f"Failed to rebuild dataset iterator after "
                                f"{_MAX_RETRIES} attempts"
                            ) from rebuild_exc
                        logger.warning(
                            "Rebuild attempt %d failed: %s",
                            attempt, rebuild_exc,
                        )

    def get_tokens(self, count: int) -> List[int]:
        """
        Get exactly `count` tokens from the stream.

        If stream is exhausted, returns fewer tokens.
        """
        self._fill_buffer()

        tokens = self.token_buffer[:count]
        self.token_buffer = self.token_buffer[count:]
        return tokens

    def is_exhausted(self) -> bool:
        """Check if stream has no more tokens."""
        self._fill_buffer()
        return self.exhausted and len(self.token_buffer) == 0


class PersistentStreamDataset(IterableDataset):
    """
    Dataset providing persistent parallel streams for TBPTT training.

    Each worker maintains BS independent streams. Streams persist across
    __iter__ calls - state is maintained for the lifetime of the dataset.

    Usage:
        dataset = PersistentStreamDataset(
            dataset_config=DATASET_CONFIGS["fineweb-edu"],
            tokenizer=tokenizer,
            batch_size=16,
            seq_length=256,
        )
        for batch in dataset:
            # batch.input_ids: [BS, T]
            # batch.target_ids: [BS, T]
            # batch.prev_token: [BS]
            ...
    """

    def __init__(
        self,
        dataset_config: DatasetConfig,
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        seq_length: int = 256,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ):
        """
        Args:
            dataset_config: Configuration for the dataset to load
            tokenizer: Tokenizer for encoding text
            batch_size: Number of parallel streams (BS)
            seq_length: TBPTT chunk length (T)
            seed: Random seed for shuffling
            max_steps: Optional maximum number of batches to yield
        """
        self.config = dataset_config
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed
        self.max_steps = max_steps
        self.eos_token_id = tokenizer.eos_token_id

        # Will be initialized on first iteration
        self._base_dataset = None
        self._stream_restarts: Optional[List[int]] = None
        self.streams: Optional[List[DocumentStream]] = None
        self.prev_tokens: Optional[torch.Tensor] = None
        self.step_count = 0
        self._resume_step = 0
        self.stream_restarts_total = 0
        self.stream_restarts_last_batch = 0
        self.streams_exhausted_last_batch = 0

    def _make_stream(self, stream_idx: int) -> DocumentStream:
        """Create one stream iterator (supports exhausted-stream recycling)."""
        assert self._base_dataset is not None
        restart_count = self._stream_restarts[stream_idx] if self._stream_restarts is not None else 0
        base_seed = self.seed + stream_idx + 9973 * restart_count

        ds = self._base_dataset
        is_streaming = self._is_streaming

        def _make_iter(retry_count: int = 0):
            # Each retry uses a different seed so a network hiccup doesn't
            # restart from the same shuffled prefix (which would duplicate data).
            seed = base_seed + 7919 * retry_count
            if is_streaming:
                return iter(ds.shuffle(seed=seed, buffer_size=10000))
            return iter(ds.shuffle(seed=seed))

        return DocumentStream(
            dataset_iter=_make_iter(),
            tokenizer=self.tokenizer,
            text_column=self.config.text_column,
            iter_factory=_make_iter,
        )

    def _init_streams(self):
        """Initialize BS independent document streams."""
        self.streams = []
        self._stream_restarts = [0 for _ in range(self.batch_size)]

        # Load dataset once, create BS shuffled iterators from it.
        self._base_dataset, self._is_streaming = _load_dataset_from_config(
            self.config
        )

        for i in range(self.batch_size):
            self.streams.append(self._make_stream(i))

        # Per-stream carry: the last token of the previous chunk's T+1 window.
        # On the next step this token becomes the first token of the new T+1
        # window, so the stream is contiguous (no skipped tokens). Initialized
        # to None (first chunk has no carry — reads a full T+1 from the stream).
        self._carry: List[Optional[int]] = [None] * self.batch_size

        # Initialize prev_tokens to EOS (triggers reset on first chunk)
        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def reset_streams(self):
        """Re-initialize streams from the already-loaded base dataset."""
        if self._base_dataset is None:
            return  # will be initialized on first __iter__
        self._stream_restarts = [0 for _ in range(self.batch_size)]
        self.streams = [self._make_stream(i) for i in range(self.batch_size)]
        self._carry = [None] * self.batch_size
        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def state_dict(self) -> dict:
        """Return serialisable snapshot for checkpointing.

        For streaming HF datasets exact iterator positions cannot be saved.
        We store a CUMULATIVE step count so on resume we fast-forward to the
        right absolute position. Same pattern as TokenShardDataset. Also
        saves consumer-side prev_tokens so the first resumed batch gets the
        correct reset_mask.
        """
        consumed = (self._consumed_step if hasattr(self, '_consumed_step')
                    and self._consumed_step is not None else self.step_count)
        cumulative = getattr(self, '_resume_base', 0) + consumed
        prev = (self._consumed_prev_tokens
                if hasattr(self, '_consumed_prev_tokens')
                and self._consumed_prev_tokens is not None
                else self.prev_tokens)
        return {
            "step_count": cumulative,
            "prev_tokens": (prev.clone() if prev is not None else None),
        }

    def mark_consumed(self, consumed: int,
                      prev_tokens: torch.Tensor | None = None) -> None:
        """Record consumed batches + consumer-side prev_tokens for
        checkpoint (same API as TokenShardDataset)."""
        self._consumed_step = consumed
        if prev_tokens is not None:
            self._consumed_prev_tokens = prev_tokens.clone()

    def load_state_dict(self, state: dict) -> None:
        """Restore dataset position from a checkpoint.

        ``step_count`` is CUMULATIVE across all prior sessions.
        ``prev_tokens`` is the consumer-side value so the first resumed
        chunk gets the correct reset_mask.
        """
        self._resume_step = state.get("step_count", 0)
        saved_prev = state.get("prev_tokens")
        if saved_prev is not None:
            self._resume_prev_tokens = saved_prev.clone()

    def __iter__(self) -> Iterator[StreamBatch]:
        """
        Iterate over batches of tokens.

        Yields StreamBatch with input_ids, target_ids, and prev_token.
        Streams persist across calls - each call continues where it left off.
        """
        if self.streams is None:
            self._init_streams()

        # Restore prev_tokens from checkpoint if available (must happen
        # after _init_streams which sets prev_tokens to all-EOS).
        resume_prev = getattr(self, "_resume_prev_tokens", None)
        if (resume_prev is not None
                and resume_prev.shape == self.prev_tokens.shape):
            self.prev_tokens = resume_prev.clone()
            self._resume_prev_tokens = None

        # Fast-forward past already-consumed batches on resume.
        resume_step = getattr(self, "_resume_step", 0)
        self._resume_base = resume_step  # for cumulative state_dict
        if resume_step > 0:
            logger.info(
                "Fast-forwarding PersistentStreamDataset: consuming %d batches",
                resume_step,
            )
            _ff_count = 0
            while _ff_count < resume_step:
                # Consume one batch worth of tokens from each stream,
                # using the same carry-overlap pattern as the main loop.
                for i, stream in enumerate(self.streams):
                    carry = self._carry[i]
                    fresh_needed = self.seq_length if carry is not None else self.seq_length + 1
                    tokens = stream.get_tokens(fresh_needed)
                    if len(tokens) < fresh_needed and stream.is_exhausted():
                        self._stream_restarts[i] += 1
                        self.stream_restarts_total += 1
                        self.streams[i] = self._make_stream(i)
                        refill = self.streams[i].get_tokens(fresh_needed - len(tokens))
                        tokens.extend(refill)
                    if carry is not None:
                        tokens = [carry] + tokens
                    if len(tokens) >= self.seq_length + 1:
                        self._carry[i] = tokens[-1]
                    # Update prev_tokens so first resumed batch has correct value
                    if len(tokens) >= self.seq_length + 1:
                        self.prev_tokens[i] = tokens[-2]  # input_ids[:, -1]
                _ff_count += 1
            self._resume_step = 0
            # step_count tracks batches yielded this session (compared
            # against max_steps = remaining steps). state_dict adds
            # _resume_base to get the cumulative count.
            self.step_count = 0

        while True:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            # Collect T+1 tokens from each stream (need +1 for target shift)
            batch_tokens = []
            all_exhausted = True
            batch_restarts = 0
            batch_exhausted = 0

            for i, stream in enumerate(self.streams):
                # If we have a carry token from the previous chunk, prepend it
                # and read T fresh tokens so total = T+1. Otherwise read T+1.
                # This mirrors TokenShardDataset's "advance by T, not T+1"
                # pattern: the last target of chunk k overlaps with the first
                # input of chunk k+1, keeping the stream contiguous.
                carry = self._carry[i]
                fresh_needed = self.seq_length if carry is not None else self.seq_length + 1
                tokens = stream.get_tokens(fresh_needed)
                produced_any = len(tokens) > 0

                # Recycle exhausted streams so batch capacity does not decay.
                if len(tokens) < fresh_needed and stream.is_exhausted():
                    batch_exhausted += 1
                    self._stream_restarts[i] += 1
                    self.stream_restarts_total += 1
                    self.streams[i] = self._make_stream(i)
                    refill = self.streams[i].get_tokens(fresh_needed - len(tokens))
                    tokens.extend(refill)
                    produced_any = produced_any or (len(refill) > 0)
                    batch_restarts += 1

                # Prepend carry if present
                if carry is not None:
                    tokens = [carry] + tokens

                needed = self.seq_length + 1
                if len(tokens) < needed:
                    raise RuntimeError(
                        f"PersistentStreamDataset: lane {i} could not produce "
                        f"{needed} tokens even after recycling an exhausted "
                        f"stream (got {len(tokens)}). The backing dataset is "
                        f"likely smaller than the requested sequence length, "
                        f"or stream recycling is broken. Refusing to pad "
                        f"with EOS (would inject false document boundaries)."
                    )

                # Save last token as carry for next chunk (overlap).
                self._carry[i] = tokens[-1]

                if produced_any:
                    all_exhausted = False
                batch_tokens.append(tokens)

            if all_exhausted:
                break

            self.stream_restarts_last_batch = batch_restarts
            self.streams_exhausted_last_batch = batch_exhausted

            # Convert to tensors
            tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)  # [BS, T+1]

            input_ids = tokens_tensor[:, :-1]   # [BS, T]
            target_ids = tokens_tensor[:, 1:]   # [BS, T]

            batch = StreamBatch(
                input_ids=input_ids,
                target_ids=target_ids,
                prev_token=self.prev_tokens.clone(),
            )

            # Update prev_tokens for next chunk
            self.prev_tokens = input_ids[:, -1].clone()
            self.step_count += 1

            yield batch

    def reset(self):
        """Reset streams to start from beginning."""
        self._base_dataset = None
        self._stream_restarts = None
        self.streams = None
        self.prev_tokens = None
        self.step_count = 0
        self.stream_restarts_total = 0
        self.stream_restarts_last_batch = 0
        self.streams_exhausted_last_batch = 0

    def monitor_stats(self) -> dict:
        """Latest stream-health counters for monitoring/logging."""
        return {
            "stream_restarts_total": self.stream_restarts_total,
            "stream_restarts_last_batch": self.stream_restarts_last_batch,
            "streams_exhausted_last_batch": self.streams_exhausted_last_batch,
        }


class MixedStreamDataset(IterableDataset):
    """
    Dataset mixing multiple sources with specified weights.

    For Phase B training with 70% FineWeb-Edu + 30% SlimPajama:
        dataset = MixedStreamDataset(
            configs=[FINEWEB_EDU, SLIMPAJAMA],
            weights=[0.7, 0.3],
            tokenizer=tokenizer,
            batch_size=16,
        )
    """

    def __init__(
        self,
        configs: List[DatasetConfig],
        weights: List[float],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int = 16,
        seq_length: int = 256,
        seed: int = 42,
        max_steps: Optional[int] = None,
    ):
        """
        Args:
            configs: List of dataset configurations
            weights: Mixing weights (should sum to 1.0)
            tokenizer: Tokenizer for encoding
            batch_size: Number of parallel streams
            seq_length: TBPTT chunk length
            seed: Random seed
            max_steps: Maximum steps to yield
        """
        assert len(configs) == len(weights), "configs and weights must have same length"
        assert abs(sum(weights) - 1.0) < 1e-6, f"weights must sum to 1.0, got {sum(weights)}"

        self.configs = configs
        self.weights = weights
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.seed = seed
        self.max_steps = max_steps
        self.eos_token_id = tokenizer.eos_token_id

        # Assign streams to datasets based on weights
        self.stream_assignments = self._compute_assignments()

        self._ds_cache: Optional[Dict[Any, Any]] = None
        self._stream_restarts: Optional[List[int]] = None
        self.streams: Optional[List[DocumentStream]] = None
        self.prev_tokens: Optional[torch.Tensor] = None
        self.step_count = 0
        self.stream_restarts_total = 0
        self.stream_restarts_last_batch = 0
        self.streams_exhausted_last_batch = 0

    def _compute_assignments(self) -> List[int]:
        """Assign each stream to a dataset based on weights."""
        assignments = []
        cumulative = 0.0

        for stream_idx in range(self.batch_size):
            # Determine which dataset this stream belongs to
            position = stream_idx / self.batch_size
            cumulative = 0.0
            for ds_idx, weight in enumerate(self.weights):
                cumulative += weight
                if position < cumulative:
                    assignments.append(ds_idx)
                    break
            else:
                assignments.append(len(self.configs) - 1)

        return assignments

    def _init_streams(self):
        """Initialize streams with mixed sources."""
        self.streams = []
        self._stream_restarts = [0 for _ in range(self.batch_size)]

        # Load each unique dataset once, keyed by hf_path.
        self._ds_cache = {}
        self._ds_streaming = {}
        for config in self.configs:
            key = (config.hf_path, config.hf_name, config.split)
            if key not in self._ds_cache:
                ds, is_streaming = _load_dataset_from_config(config)
                self._ds_cache[key] = ds
                self._ds_streaming[key] = is_streaming
            self._ds_streaming[id(config)] = self._ds_streaming[key]

        for i in range(self.batch_size):
            self.streams.append(self._make_stream(i))

        self._carry: List[Optional[int]] = [None] * self.batch_size
        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def reset_streams(self):
        """Re-initialize streams from already-loaded datasets."""
        if self._ds_cache is None:
            return
        self._stream_restarts = [0 for _ in range(self.batch_size)]
        self.streams = [self._make_stream(i) for i in range(self.batch_size)]
        self._carry = [None] * self.batch_size
        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def state_dict(self) -> dict:
        """Checkpoint state — same pattern as PersistentStreamDataset."""
        consumed = (self._consumed_step if hasattr(self, '_consumed_step')
                    and self._consumed_step is not None else self.step_count)
        cumulative = getattr(self, '_resume_base', 0) + consumed
        prev = (self._consumed_prev_tokens
                if hasattr(self, '_consumed_prev_tokens')
                and self._consumed_prev_tokens is not None
                else self.prev_tokens)
        return {
            "step_count": cumulative,
            "prev_tokens": (prev.clone() if prev is not None else None),
        }

    def mark_consumed(self, consumed: int,
                      prev_tokens: torch.Tensor | None = None) -> None:
        self._consumed_step = consumed
        if prev_tokens is not None:
            self._consumed_prev_tokens = prev_tokens.clone()

    def load_state_dict(self, state: dict) -> None:
        self._resume_step = state.get("step_count", 0)
        saved_prev = state.get("prev_tokens")
        if saved_prev is not None:
            self._resume_prev_tokens = saved_prev.clone()

    def __iter__(self) -> Iterator[StreamBatch]:
        if self.streams is None:
            self._init_streams()

        # Restore prev_tokens from checkpoint
        resume_prev = getattr(self, "_resume_prev_tokens", None)
        if (resume_prev is not None and self.prev_tokens is not None
                and resume_prev.shape == self.prev_tokens.shape):
            self.prev_tokens = resume_prev.clone()
            self._resume_prev_tokens = None

        # Fast-forward on resume (same as PersistentStreamDataset)
        resume_step = getattr(self, "_resume_step", 0)
        self._resume_base = resume_step
        if resume_step > 0:
            logger.info(
                "Fast-forwarding MixedStreamDataset: consuming %d batches",
                resume_step,
            )
            for _ in range(resume_step):
                for i, stream in enumerate(self.streams):
                    carry = self._carry[i]
                    fresh_needed = self.seq_length if carry is not None else self.seq_length + 1
                    tokens = stream.get_tokens(fresh_needed)
                    if len(tokens) < fresh_needed and stream.is_exhausted():
                        self._stream_restarts[i] += 1
                        self.stream_restarts_total += 1
                        self.streams[i] = self._make_stream(i)
                        refill = self.streams[i].get_tokens(fresh_needed - len(tokens))
                        tokens.extend(refill)
                    if carry is not None:
                        tokens = [carry] + tokens
                    if len(tokens) >= self.seq_length + 1:
                        self._carry[i] = tokens[-1]
                        self.prev_tokens[i] = tokens[-2]
            self._resume_step = 0
            self.step_count = 0

        while True:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            batch_tokens = []
            all_exhausted = True
            batch_restarts = 0
            batch_exhausted = 0

            for i, stream in enumerate(self.streams):
                # Same carry-overlap pattern as PersistentStreamDataset:
                # prepend the carry from the previous chunk so the stream
                # is contiguous (advance by T, not T+1).
                carry = self._carry[i]
                fresh_needed = self.seq_length if carry is not None else self.seq_length + 1
                tokens = stream.get_tokens(fresh_needed)
                produced_any = len(tokens) > 0

                if len(tokens) < fresh_needed and stream.is_exhausted():
                    batch_exhausted += 1
                    self._stream_restarts[i] += 1
                    self.stream_restarts_total += 1
                    self.streams[i] = self._make_stream(i)
                    refill = self.streams[i].get_tokens(fresh_needed - len(tokens))
                    tokens.extend(refill)
                    produced_any = produced_any or (len(refill) > 0)
                    batch_restarts += 1

                if carry is not None:
                    tokens = [carry] + tokens

                needed = self.seq_length + 1
                if len(tokens) < needed:
                    raise RuntimeError(
                        f"MixedStreamDataset: lane {i} could not produce "
                        f"{needed} tokens even after recycling (got "
                        f"{len(tokens)}). Backing dataset too small."
                    )

                self._carry[i] = tokens[-1]

                if produced_any:
                    all_exhausted = False
                batch_tokens.append(tokens)

            if all_exhausted:
                break

            self.stream_restarts_last_batch = batch_restarts
            self.streams_exhausted_last_batch = batch_exhausted

            tokens_tensor = torch.tensor(batch_tokens, dtype=torch.long)
            input_ids = tokens_tensor[:, :-1]
            target_ids = tokens_tensor[:, 1:]

            batch = StreamBatch(
                input_ids=input_ids,
                target_ids=target_ids,
                prev_token=self.prev_tokens.clone(),
            )

            self.prev_tokens = input_ids[:, -1].clone()
            self.step_count += 1

            yield batch

    def reset(self):
        self._ds_cache = None
        self._stream_restarts = None
        self.streams = None
        self.prev_tokens = None
        self.step_count = 0
        self.stream_restarts_total = 0
        self.stream_restarts_last_batch = 0
        self.streams_exhausted_last_batch = 0

    def monitor_stats(self) -> dict:
        return {
            "stream_restarts_total": self.stream_restarts_total,
            "stream_restarts_last_batch": self.stream_restarts_last_batch,
            "streams_exhausted_last_batch": self.streams_exhausted_last_batch,
        }

    def _make_stream(self, stream_idx: int) -> DocumentStream:
        """Create one stream for its assigned dataset (supports recycling)."""
        assert self._ds_cache is not None
        ds_idx = self.stream_assignments[stream_idx]
        config = self.configs[ds_idx]
        is_streaming = self._ds_streaming[id(config)]
        key = (config.hf_path, config.hf_name, config.split)
        ds = self._ds_cache[key]

        restart_count = self._stream_restarts[stream_idx] if self._stream_restarts is not None else 0
        base_seed = self.seed + stream_idx + 9973 * restart_count

        def _make_iter(retry_count: int = 0):
            seed = base_seed + 7919 * retry_count
            if is_streaming:
                return iter(ds.shuffle(seed=seed, buffer_size=10000))
            return iter(ds.shuffle(seed=seed))

        return DocumentStream(
            dataset_iter=_make_iter(),
            tokenizer=self.tokenizer,
            text_column=config.text_column,
            iter_factory=_make_iter,
        )


class _DatasetIterator:
    """Iterator wrapper that exposes underlying dataset monitor stats.

    Iteration is deferred until the first ``__next__`` call so that
    ``load_state_dict`` can be called after construction but before any
    data is consumed.
    """

    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self._it: Optional[Iterator] = None  # deferred

    def _ensure_iter(self):
        if self._it is None:
            self._it = iter(self.dataset)

    def __iter__(self):
        return self

    def __next__(self):
        self._ensure_iter()
        return next(self._it)

    def monitor_stats(self) -> dict:
        if hasattr(self.dataset, "monitor_stats"):
            return self.dataset.monitor_stats()
        return {}

    def mark_consumed(self, consumed: int,
                      prev_tokens: torch.Tensor | None = None) -> None:
        """Proxy to underlying dataset's mark_consumed() for accurate
        checkpointing of the consumer position (vs prefetch-ahead counter)."""
        if hasattr(self.dataset, "mark_consumed"):
            self.dataset.mark_consumed(consumed, prev_tokens=prev_tokens)

    def state_dict(self) -> dict:
        """Proxy to underlying dataset's state_dict for checkpointing."""
        if hasattr(self.dataset, "state_dict"):
            return self.dataset.state_dict()
        return {}

    def load_state_dict(self, state: dict) -> None:
        """Proxy to underlying dataset's load_state_dict for resume.

        Must be called **before** the first ``__next__`` so that the
        dataset can fast-forward positions when iteration starts.
        """
        if hasattr(self.dataset, "load_state_dict"):
            self.dataset.load_state_dict(state)


def _find_shard(config) -> Optional[str]:
    """Check if a pre-tokenized .bin shard exists alongside the parquet file."""
    path = config.hf_path
    if not os.path.exists(path):
        return None
    base, ext = os.path.splitext(path)
    if ext.lower() in (".parquet", ".jsonl", ".csv"):
        shard_path = base + ".bin"
        if os.path.exists(shard_path):
            return shard_path
    return None


def create_dataloader(
    phase: str,
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int = 16,
    seq_length: int = 256,
    seed: int = 42,
    max_steps: Optional[int] = None,
) -> Iterator[StreamBatch]:
    """
    Create a dataloader for a training phase.

    Automatically uses pre-tokenized .bin shards when available (faster).
    Falls back to on-the-fly tokenization from text otherwise.

    Args:
        phase: Training phase ("A", "B", "C", etc.)
        tokenizer: Tokenizer instance
        batch_size: Number of parallel streams
        seq_length: TBPTT chunk length
        seed: Random seed
        max_steps: Maximum steps

    Returns:
        Iterator yielding StreamBatch objects
    """
    from .config import PHASE_CONFIGS, DATASET_CONFIGS

    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(PHASE_CONFIGS.keys())}")

    phase_cfg = PHASE_CONFIGS[phase]
    configs = [DATASET_CONFIGS[name] for name in phase_cfg.datasets]

    # Single dataset: try pre-tokenized shard first
    if len(configs) == 1:
        shard_path = _find_shard(configs[0])
        if shard_path is not None:
            # Shards are tokenizer-specific (prepare_data.py writes a
            # .shard_meta.json with the tokenizer name). Check identity
            # first; fall back to vocab-size-only check for old shards.
            import numpy as np
            meta_path = shard_path.replace(".bin", ".shard_meta.json")
            if os.path.exists(meta_path):
                import json as _json
                with open(meta_path) as _f:
                    shard_meta = _json.load(_f)
                shard_tokenizer = shard_meta.get("tokenizer", "")
                # Resolve the runtime tokenizer's canonical name for comparison.
                # tokenizer.name_or_path is the string passed to from_pretrained.
                runtime_name = getattr(tokenizer, "name_or_path", "")
                if shard_tokenizer and runtime_name and shard_tokenizer != runtime_name:
                    raise ValueError(
                        f"Pre-tokenized shard {shard_path} was created with "
                        f"tokenizer '{shard_tokenizer}' but runtime tokenizer "
                        f"is '{runtime_name}'. Token IDs would be silently "
                        f"reinterpreted. Delete the .bin shard and re-run "
                        f"prepare_data.py, or use the matching tokenizer.")
                shard_vocab = shard_meta.get("vocab_size", 0)
                if shard_vocab and shard_vocab != tokenizer.vocab_size:
                    raise ValueError(
                        f"Pre-tokenized shard {shard_path} has vocab_size "
                        f"{shard_vocab} but runtime tokenizer has "
                        f"{tokenizer.vocab_size}.")
            else:
                # Legacy shards without metadata: fall back to max-id check.
                shard_mm = np.memmap(shard_path, dtype=np.uint16, mode='r')
                shard_len = len(shard_mm)
                sample_size = 100_000
                n_samples = 10
                if shard_len <= sample_size * n_samples:
                    shard_max_id = int(shard_mm.max())
                else:
                    max_so_far = 0
                    stride = shard_len // n_samples
                    for i in range(n_samples):
                        start = i * stride
                        end = min(start + sample_size, shard_len)
                        chunk_max = int(shard_mm[start:end].max())
                        if chunk_max > max_so_far:
                            max_so_far = chunk_max
                    shard_max_id = max_so_far
                del shard_mm
                if shard_max_id >= tokenizer.vocab_size:
                    raise ValueError(
                        f"Pre-tokenized shard {shard_path} contains token id "
                        f"{shard_max_id} but tokenizer vocab_size is "
                        f"{tokenizer.vocab_size}. The shard was likely created "
                        f"with a different tokenizer. Delete the .bin shard or "
                        f"use the matching tokenizer.")
                logger.warning(
                    "Shard %s has no .shard_meta.json — using legacy "
                    "vocab-size-only check. Re-run prepare_data.py to "
                    "generate metadata for full tokenizer identity verification.",
                    shard_path)
            logger.info("Using pre-tokenized shard: %s", shard_path)
            dataset = TokenShardDataset(
                shard_path=shard_path,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=batch_size,
                seq_length=seq_length,
                seed=seed,
                max_steps=max_steps,
            )
            return _DatasetIterator(dataset)

        dataset = PersistentStreamDataset(
            dataset_config=configs[0],
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            max_steps=max_steps,
        )
    else:
        weights = phase_cfg.mix_weights or [1.0 / len(configs)] * len(configs)
        dataset = MixedStreamDataset(
            configs=configs,
            weights=weights,
            tokenizer=tokenizer,
            batch_size=batch_size,
            seq_length=seq_length,
            seed=seed,
            max_steps=max_steps,
        )

    return _DatasetIterator(dataset)


if __name__ == "__main__":
    # Quick test
    from .tokenizer import get_tokenizer
    from .config import TINYSTORIES

    print("Testing PersistentStreamDataset...")

    tokenizer = get_tokenizer()
    dataset = PersistentStreamDataset(
        dataset_config=TINYSTORIES,
        tokenizer=tokenizer,
        batch_size=4,
        seq_length=64,
        max_steps=5,
    )

    eos_token_id = tokenizer.eos_token_id
    for i, batch in enumerate(dataset):
        print(f"\nBatch {i}:")
        print(f"  input_ids shape: {batch.input_ids.shape}")
        print(f"  target_ids shape: {batch.target_ids.shape}")
        print(f"  prev_token: {batch.prev_token}")

        # Check for EOS tokens (document boundaries)
        eos_mask = batch.input_ids == eos_token_id
        eos_counts = eos_mask.sum(dim=1)
        print(f"  EOS counts per stream: {eos_counts.tolist()}")

    print("\nTest complete!")
