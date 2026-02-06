"""
Persistent parallel stream data loading for TBPTT training.

Key design:
- BS independent persistent streams (not independent sequences)
- Each stream is a continuous flow of documents separated by <|endoftext|>
- State (hidden, PM, EM, eligibility) persists across TBPTT chunks
- Different streams hit document boundaries at different positions

This is fundamentally different from transformer data loading where
each batch item is an independent sequence.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from datasets import load_dataset, IterableDataset as HFIterableDataset
from transformers import PreTrainedTokenizerFast
from typing import Optional, Iterator, List, Dict, Any
from dataclasses import dataclass
import numpy as np

from .config import DatasetConfig, DATASET_CONFIGS
from .tokenizer import get_tokenizer


@dataclass
class StreamBatch:
    """A batch of tokens from persistent streams."""
    input_ids: torch.Tensor      # [BS, T] - input tokens
    target_ids: torch.Tensor     # [BS, T] - target tokens (shifted by 1)
    prev_token: torch.Tensor     # [BS] - last token from previous chunk (for reset detection)


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
    ):
        """
        Args:
            dataset_iter: Iterator over dataset examples
            tokenizer: Tokenizer for encoding text
            text_column: Column name containing text data
            buffer_size: Number of tokens to buffer before yielding
        """
        self.dataset_iter = dataset_iter
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.buffer_size = buffer_size

        self.token_buffer: List[int] = []
        self.exhausted = False
        self.eos_token_id = tokenizer.eos_token_id

    def _fill_buffer(self):
        """Fill token buffer from dataset."""
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
        self.streams: Optional[List[DocumentStream]] = None
        self.prev_tokens: Optional[torch.Tensor] = None
        self.step_count = 0

    def _init_streams(self):
        """Initialize BS independent document streams."""
        self.streams = []

        for i in range(self.batch_size):
            # Each stream gets its own dataset iterator with different shuffling
            ds = load_dataset(
                self.config.hf_path,
                self.config.hf_name,
                split=self.config.split,
                streaming=self.config.streaming,
            )

            # Shuffle with different seed per stream
            # buffer_size only applies to IterableDataset (streaming mode)
            if self.config.streaming:
                ds = ds.shuffle(seed=self.seed + i, buffer_size=10000)
            else:
                ds = ds.shuffle(seed=self.seed + i)

            stream = DocumentStream(
                dataset_iter=iter(ds),
                tokenizer=self.tokenizer,
                text_column=self.config.text_column,
            )
            self.streams.append(stream)

        # Initialize prev_tokens to EOS (triggers reset on first chunk)
        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def __iter__(self) -> Iterator[StreamBatch]:
        """
        Iterate over batches of tokens.

        Yields StreamBatch with input_ids, target_ids, and prev_token.
        Streams persist across calls - each call continues where it left off.
        """
        if self.streams is None:
            self._init_streams()

        while True:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            # Collect T+1 tokens from each stream (need +1 for target shift)
            batch_tokens = []
            all_exhausted = True

            for stream in self.streams:
                tokens = stream.get_tokens(self.seq_length + 1)
                if len(tokens) < self.seq_length + 1:
                    # Pad with EOS if stream exhausted
                    tokens.extend([self.eos_token_id] * (self.seq_length + 1 - len(tokens)))
                else:
                    all_exhausted = False
                batch_tokens.append(tokens)

            if all_exhausted:
                break

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
        self.streams = None
        self.prev_tokens = None
        self.step_count = 0


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

        self.streams: Optional[List[DocumentStream]] = None
        self.prev_tokens: Optional[torch.Tensor] = None
        self.step_count = 0

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

        for i in range(self.batch_size):
            ds_idx = self.stream_assignments[i]
            config = self.configs[ds_idx]

            ds = load_dataset(
                config.hf_path,
                config.hf_name,
                split=config.split,
                streaming=config.streaming,
            )

            # Shuffle with different seed per stream
            if config.streaming:
                ds = ds.shuffle(seed=self.seed + i, buffer_size=10000)
            else:
                ds = ds.shuffle(seed=self.seed + i)

            stream = DocumentStream(
                dataset_iter=iter(ds),
                tokenizer=self.tokenizer,
                text_column=config.text_column,
            )
            self.streams.append(stream)

        self.prev_tokens = torch.full((self.batch_size,), self.eos_token_id, dtype=torch.long)
        self.step_count = 0

    def __iter__(self) -> Iterator[StreamBatch]:
        if self.streams is None:
            self._init_streams()

        while True:
            if self.max_steps is not None and self.step_count >= self.max_steps:
                break

            batch_tokens = []
            all_exhausted = True

            for stream in self.streams:
                tokens = stream.get_tokens(self.seq_length + 1)
                if len(tokens) < self.seq_length + 1:
                    tokens.extend([self.eos_token_id] * (self.seq_length + 1 - len(tokens)))
                else:
                    all_exhausted = False
                batch_tokens.append(tokens)

            if all_exhausted:
                break

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
        self.streams = None
        self.prev_tokens = None
        self.step_count = 0


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

    if len(configs) == 1:
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

    return iter(dataset)


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
