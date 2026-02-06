"""
Debug and visualization utilities for data loading pipeline.

Usage:
    # Run all diagnostics
    python -m src.data.debug

    # Test specific dataset
    python -m src.data.debug --dataset fineweb-edu --samples 5

    # Test specific phase
    python -m src.data.debug --phase B --batches 3

    # Plot token distribution
    python -m src.data.debug --plot --batches 10
"""

import argparse
import sys
from collections import Counter
from typing import Optional, List
import torch

from .config import DATASET_CONFIGS, PHASE_CONFIGS, DatasetConfig
from .tokenizer import get_tokenizer, get_special_token_ids, TOKENIZER_PRESETS
from .streaming import PersistentStreamDataset, MixedStreamDataset, create_dataloader, StreamBatch


def print_header(title: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 60}")
    print(f" {title}")
    print(f"{char * 60}")


def print_tokenizer_info(tokenizer, preset: str = "default") -> None:
    """Print tokenizer configuration details."""
    print_header(f"Tokenizer: {preset}")

    ids = get_special_token_ids(tokenizer)
    print(f"  Vocab size:    {ids['vocab_size']:,}")
    print(f"  EOS token:     {repr(tokenizer.eos_token)} (id={ids['eos_token_id']})")
    print(f"  BOS token:     {repr(tokenizer.bos_token)} (id={ids['bos_token_id']})")
    print(f"  PAD token:     {repr(tokenizer.pad_token)} (id={ids['pad_token_id']})")

    # Test encoding
    test_text = "Hello, world! This is a test."
    tokens = tokenizer.encode(test_text, add_special_tokens=False)
    print(f"\n  Test encoding:")
    print(f"    Text:   {repr(test_text)}")
    print(f"    Tokens: {tokens}")
    print(f"    Length: {len(tokens)} tokens")

    # Show token-by-token breakdown
    print(f"\n  Token breakdown:")
    for i, tok_id in enumerate(tokens[:10]):
        tok_str = tokenizer.decode([tok_id])
        print(f"    [{i}] {tok_id:5d} -> {repr(tok_str)}")
    if len(tokens) > 10:
        print(f"    ... ({len(tokens) - 10} more tokens)")


def print_dataset_info(config: DatasetConfig) -> None:
    """Print dataset configuration details."""
    tokens = config.estimated_tokens or 0
    if tokens >= 1_000_000_000_000:
        tokens_str = f"{tokens/1_000_000_000_000:.1f}T"
    elif tokens >= 1_000_000_000:
        tokens_str = f"{tokens/1_000_000_000:.1f}B"
    else:
        tokens_str = f"{tokens/1_000_000:.0f}M"

    print(f"\n  {config.name}:")
    print(f"    HF path:      {config.hf_path}")
    if config.hf_name:
        print(f"    HF config:    {config.hf_name}")
    print(f"    Split:        {config.split}")
    print(f"    Text column:  {config.text_column}")
    print(f"    Streaming:    {config.streaming}")
    print(f"    Est. tokens:  ~{tokens_str}")
    print(f"    Est. disk:    ~{config.estimated_disk_gb}GB")


def test_dataset_access(
    config: DatasetConfig,
    tokenizer,
    num_samples: int = 3,
    verbose: bool = True,
) -> dict:
    """
    Test that a dataset can be loaded and tokenized.

    Returns dict with:
        - success: bool
        - error: Optional error message
        - samples: List of sample texts
        - token_counts: List of token counts per sample
    """
    from datasets import load_dataset

    result = {
        "success": False,
        "error": None,
        "samples": [],
        "token_counts": [],
    }

    try:
        ds = load_dataset(
            config.hf_path,
            config.hf_name,
            split=config.split,
            streaming=True,
        )

        samples_collected = 0
        for example in ds:
            if samples_collected >= num_samples:
                break

            text = example.get(config.text_column, "")
            if not text or not text.strip():
                continue

            # Tokenize
            tokens = tokenizer.encode(text, add_special_tokens=False)

            result["samples"].append(text)
            result["token_counts"].append(len(tokens))
            samples_collected += 1

        if samples_collected > 0:
            result["success"] = True
        else:
            result["error"] = f"No valid samples found in column '{config.text_column}'"

    except Exception as e:
        result["error"] = str(e)

    return result


def print_sample_text(
    text: str,
    tokenizer,
    max_chars: int = 200,
    show_tokens: bool = True,
) -> None:
    """Print a sample text with optional token breakdown."""
    # Truncate for display
    display_text = text[:max_chars]
    if len(text) > max_chars:
        display_text += "..."

    # Clean up whitespace for display
    display_text = " ".join(display_text.split())
    print(f"    Text: {repr(display_text)}")

    if show_tokens:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        print(f"    Tokens: {len(tokens):,}")


def print_batch_info(
    batch: StreamBatch,
    tokenizer,
    batch_idx: int = 0,
    show_streams: int = 2,
    show_tokens: int = 20,
) -> None:
    """Print detailed information about a batch."""
    print(f"\n  Batch {batch_idx}:")
    print(f"    input_ids shape:  {batch.input_ids.shape}")
    print(f"    target_ids shape: {batch.target_ids.shape}")
    print(f"    prev_token shape: {batch.prev_token.shape}")

    eos_id = tokenizer.eos_token_id

    # Show per-stream info
    for stream_idx in range(min(show_streams, batch.input_ids.shape[0])):
        stream_tokens = batch.input_ids[stream_idx]

        # Count EOS tokens (document boundaries)
        eos_count = (stream_tokens == eos_id).sum().item()

        # Decode first N tokens
        decoded = tokenizer.decode(stream_tokens[:show_tokens].tolist())
        decoded = " ".join(decoded.split())  # Clean whitespace

        print(f"\n    Stream {stream_idx}:")
        print(f"      EOS count: {eos_count} (document boundaries)")
        print(f"      First {show_tokens} tokens: {repr(decoded[:80])}...")

        # Show token IDs
        token_ids = stream_tokens[:10].tolist()
        print(f"      Token IDs: {token_ids}...")


def visualize_document_boundaries(
    batch: StreamBatch,
    tokenizer,
    stream_idx: int = 0,
    char_width: int = 80,
) -> None:
    """
    Visualize document boundaries (EOS tokens) in a stream.

    Prints a visual representation showing where documents start/end.
    """
    eos_id = tokenizer.eos_token_id
    tokens = batch.input_ids[stream_idx]

    print(f"\n  Document Boundary Visualization (Stream {stream_idx}):")
    print(f"  {'─' * char_width}")

    # Build visualization
    current_doc = 0
    doc_starts = [0]
    for i, tok_id in enumerate(tokens.tolist()):
        if tok_id == eos_id:
            doc_starts.append(i + 1)
            current_doc += 1

    # Print document spans
    for doc_idx, start in enumerate(doc_starts[:-1]):
        end = doc_starts[doc_idx + 1] if doc_idx + 1 < len(doc_starts) else len(tokens)
        doc_tokens = tokens[start:min(end, start + 30)]
        text = tokenizer.decode(doc_tokens.tolist())
        text = " ".join(text.split())[:60]
        print(f"  Doc {doc_idx}: [{start:3d}-{end:3d}] {repr(text)}...")

    # ASCII art visualization
    print(f"\n  Token stream (. = token, | = EOS):")
    line = ""
    for i, tok_id in enumerate(tokens.tolist()):
        if tok_id == eos_id:
            line += "|"
        else:
            line += "."
        if len(line) >= char_width:
            print(f"  {line}")
            line = ""
    if line:
        print(f"  {line}")


def quick_sanity_check(
    tokenizer_preset: str = "tinyllama",
    phase: str = "A",
    num_batches: int = 3,
    batch_size: int = 4,
    seq_length: int = 128,
) -> bool:
    """
    Quick sanity check for data pipeline.

    Call this from training code before starting to verify data is loading correctly.

    Returns True if all checks pass.
    """
    from .streaming import create_dataloader

    print(f"Running quick sanity check (phase={phase}, BS={batch_size}, T={seq_length})...")

    try:
        tokenizer = get_tokenizer(tokenizer_preset)
        eos_id = tokenizer.eos_token_id

        dataloader = create_dataloader(
            phase, tokenizer, batch_size, seq_length, max_steps=num_batches
        )

        total_tokens = 0
        total_eos = 0

        for i, batch in enumerate(dataloader):
            # Check shapes
            assert batch.input_ids.shape == (batch_size, seq_length), \
                f"Bad input shape: {batch.input_ids.shape}"
            assert batch.target_ids.shape == (batch_size, seq_length), \
                f"Bad target shape: {batch.target_ids.shape}"
            assert batch.prev_token.shape == (batch_size,), \
                f"Bad prev_token shape: {batch.prev_token.shape}"

            # Check target is shifted by 1
            # (target[t] should be input[t+1] from original sequence)

            total_tokens += batch.input_ids.numel()
            total_eos += (batch.input_ids == eos_id).sum().item()

        print(f"  ✓ Loaded {num_batches} batches")
        print(f"  ✓ Total tokens: {total_tokens:,}")
        print(f"  ✓ EOS tokens: {total_eos} ({100*total_eos/total_tokens:.1f}%)")
        print(f"  ✓ Shapes correct")
        return True

    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def analyze_batch_statistics(
    batches: List[StreamBatch],
    tokenizer,
) -> dict:
    """Compute statistics across multiple batches."""
    eos_id = tokenizer.eos_token_id

    all_tokens = []
    eos_counts = []

    for batch in batches:
        # Flatten tokens
        tokens = batch.input_ids.flatten().tolist()
        all_tokens.extend(tokens)

        # Count EOS per stream
        for stream_idx in range(batch.input_ids.shape[0]):
            eos_count = (batch.input_ids[stream_idx] == eos_id).sum().item()
            eos_counts.append(eos_count)

    # Token frequency
    token_freq = Counter(all_tokens)

    return {
        "total_tokens": len(all_tokens),
        "unique_tokens": len(token_freq),
        "eos_total": sum(eos_counts),
        "eos_per_stream_avg": sum(eos_counts) / len(eos_counts) if eos_counts else 0,
        "top_tokens": token_freq.most_common(20),
        "token_freq": token_freq,
    }


def plot_token_distribution(
    stats: dict,
    tokenizer,
    output_path: Optional[str] = None,
) -> None:
    """Plot token frequency distribution."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Top tokens bar chart
    ax1 = axes[0]
    top_tokens = stats["top_tokens"][:15]
    token_ids = [t[0] for t in top_tokens]
    counts = [t[1] for t in top_tokens]
    labels = [repr(tokenizer.decode([tid])) for tid in token_ids]

    ax1.barh(range(len(counts)), counts)
    ax1.set_yticks(range(len(counts)))
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Frequency")
    ax1.set_title("Top 15 Most Common Tokens")
    ax1.invert_yaxis()

    # Token frequency histogram
    ax2 = axes[1]
    freqs = list(stats["token_freq"].values())
    ax2.hist(freqs, bins=50, log=True, edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Token Frequency")
    ax2.set_ylabel("Count (log scale)")
    ax2.set_title("Token Frequency Distribution")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"  Plot saved to: {output_path}")
    else:
        plt.show()


def run_diagnostics(
    tokenizer_preset: str = "tinyllama",
    test_datasets: Optional[List[str]] = None,
    test_phase: Optional[str] = None,
    num_samples: int = 3,
    num_batches: int = 3,
    batch_size: int = 4,
    seq_length: int = 128,
    plot: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Run comprehensive diagnostics on the data pipeline.

    Returns True if all tests pass.
    """
    all_passed = True

    # Test tokenizer
    print_header("TOKENIZER DIAGNOSTICS")
    print(f"\nAvailable presets: {list(TOKENIZER_PRESETS.keys())}")

    tokenizer = get_tokenizer(tokenizer_preset)
    print_tokenizer_info(tokenizer, tokenizer_preset)

    # Test datasets
    print_header("DATASET DIAGNOSTICS")

    datasets_to_test = test_datasets or ["tinystories", "fineweb-edu"]

    for ds_name in datasets_to_test:
        if ds_name not in DATASET_CONFIGS:
            print(f"\n  WARNING: Unknown dataset '{ds_name}'")
            continue

        config = DATASET_CONFIGS[ds_name]
        print_dataset_info(config)

        print(f"\n    Testing access...")
        result = test_dataset_access(config, tokenizer, num_samples)

        if result["success"]:
            print(f"    ✓ SUCCESS - Retrieved {len(result['samples'])} samples")
            for i, (text, count) in enumerate(zip(result["samples"], result["token_counts"])):
                print(f"\n    Sample {i + 1} ({count:,} tokens):")
                print_sample_text(text, tokenizer, show_tokens=False)
        else:
            print(f"    ✗ FAILED: {result['error']}")
            all_passed = False

    # Test streaming dataloader
    if test_phase:
        print_header(f"STREAMING DATALOADER: Phase {test_phase}")

        if test_phase not in PHASE_CONFIGS:
            print(f"  Unknown phase: {test_phase}")
            print(f"  Available: {list(PHASE_CONFIGS.keys())}")
            all_passed = False
        else:
            phase_cfg = PHASE_CONFIGS[test_phase]
            print(f"  Phase: {phase_cfg.name}")
            print(f"  Datasets: {phase_cfg.datasets}")
            if phase_cfg.mix_weights:
                print(f"  Weights: {phase_cfg.mix_weights}")

            print(f"\n  Creating dataloader (BS={batch_size}, T={seq_length})...")

            try:
                dataloader = create_dataloader(
                    test_phase,
                    tokenizer,
                    batch_size=batch_size,
                    seq_length=seq_length,
                    max_steps=num_batches,
                )

                batches = []
                for i, batch in enumerate(dataloader):
                    batches.append(batch)
                    if verbose:
                        print_batch_info(batch, tokenizer, i)

                print(f"\n  ✓ Successfully loaded {len(batches)} batches")

                # Compute statistics
                print_header("BATCH STATISTICS")
                stats = analyze_batch_statistics(batches, tokenizer)
                print(f"  Total tokens:      {stats['total_tokens']:,}")
                print(f"  Unique tokens:     {stats['unique_tokens']:,}")
                print(f"  EOS tokens:        {stats['eos_total']:,}")
                print(f"  Avg EOS/stream:    {stats['eos_per_stream_avg']:.2f}")

                print(f"\n  Top 10 tokens:")
                for tok_id, count in stats["top_tokens"][:10]:
                    tok_str = repr(tokenizer.decode([tok_id]))
                    print(f"    {tok_id:5d} {tok_str:15s} : {count:,}")

                # Plot if requested
                if plot:
                    print_header("PLOTTING")
                    plot_token_distribution(stats, tokenizer)

            except Exception as e:
                print(f"  ✗ FAILED: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    # Summary
    print_header("SUMMARY")
    if all_passed:
        print("  ✓ All diagnostics passed!")
    else:
        print("  ✗ Some diagnostics failed - check output above")

    return all_passed


def main():
    parser = argparse.ArgumentParser(
        description="Debug and visualize data loading pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.debug                           # Run basic diagnostics
  python -m src.data.debug --phase A                 # Test Phase A dataloader
  python -m src.data.debug --dataset fineweb-edu     # Test specific dataset
  python -m src.data.debug --tokenizer gpt2          # Test GPT-2 tokenizer
  python -m src.data.debug --phase B --plot          # Plot token distribution
  python -m src.data.debug --all                     # Test all datasets
        """,
    )

    parser.add_argument(
        "--tokenizer",
        type=str,
        default="tinyllama",
        choices=list(TOKENIZER_PRESETS.keys()),
        help="Tokenizer preset to use",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        help="Specific dataset(s) to test",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=list(PHASE_CONFIGS.keys()),
        help="Training phase to test",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="Number of samples per dataset",
    )
    parser.add_argument(
        "--batches",
        type=int,
        default=3,
        help="Number of batches to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for dataloader test",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length for dataloader test",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot token distribution",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Test all available datasets",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()

    # Determine datasets to test
    if args.all:
        datasets = list(DATASET_CONFIGS.keys())
    elif args.dataset:
        datasets = args.dataset
    else:
        datasets = ["tinystories", "fineweb-edu"]  # Default

    # If no phase specified but --all, test Phase A
    phase = args.phase
    if args.all and not phase:
        phase = "A"

    success = run_diagnostics(
        tokenizer_preset=args.tokenizer,
        test_datasets=datasets,
        test_phase=phase,
        num_samples=args.samples,
        num_batches=args.batches,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        plot=args.plot,
        verbose=not args.quiet,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
