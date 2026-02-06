#!/usr/bin/env python3
"""
Dataset download script for neuromorphic LM training.

Downloads and caches datasets locally for faster training iteration.
Large datasets (SlimPajama, ProofPile-2) are streaming-only and won't be downloaded.

Usage:
    # Download Phase A dataset (TinyStories)
    python -m src.data.download --phase A

    # Download all Phase B datasets
    python -m src.data.download --phase B

    # Download specific dataset
    python -m src.data.download --dataset fineweb-edu

    # List available datasets
    python -m src.data.download --list
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

from .config import (
    DATASET_CONFIGS,
    PHASE_CONFIGS,
    DatasetConfig,
)


def get_cache_dir() -> Path:
    """Get HuggingFace cache directory."""
    cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    return Path(cache_dir)


def download_dataset(
    config: DatasetConfig,
    force: bool = False,
    verbose: bool = True,
) -> bool:
    """
    Download a dataset to local cache.

    Args:
        config: Dataset configuration
        force: Force re-download even if cached
        verbose: Print progress

    Returns:
        True if download succeeded, False if streaming-only
    """
    if config.streaming and not config.download_first:
        if verbose:
            print(f"  Skipping {config.name} (streaming-only, ~{config.estimated_disk_gb}GB)")
        return False

    if verbose:
        print(f"\nDownloading {config.name}...")
        print(f"  HuggingFace path: {config.hf_path}")
        if config.hf_name:
            print(f"  Config: {config.hf_name}")
        print(f"  Estimated size: ~{config.estimated_disk_gb}GB")
        print(f"  Estimated tokens: ~{config.estimated_tokens:,}")

    try:
        # Load dataset to trigger download
        ds = load_dataset(
            config.hf_path,
            config.hf_name,
            split=config.split,
            streaming=False,  # Force download
                    )

        if verbose:
            print(f"  Downloaded: {len(ds):,} examples")
            # Sample first example to verify text column exists
            if len(ds) > 0:
                sample = ds[0]
                if config.text_column in sample:
                    text = sample[config.text_column]
                    preview = text[:100] + "..." if len(text) > 100 else text
                    print(f"  Sample: {preview}")
                else:
                    print(f"  WARNING: Column '{config.text_column}' not found!")
                    print(f"  Available columns: {list(sample.keys())}")

        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def download_phase(phase: str, force: bool = False, verbose: bool = True) -> None:
    """Download all datasets for a training phase."""
    if phase not in PHASE_CONFIGS:
        print(f"Unknown phase: {phase}")
        print(f"Available phases: {list(PHASE_CONFIGS.keys())}")
        return

    phase_cfg = PHASE_CONFIGS[phase]
    print(f"\n{'='*60}")
    print(f"{phase_cfg.name}")
    print(f"{'='*60}")
    print(f"Description: {phase_cfg.description}")
    print(f"Datasets: {', '.join(phase_cfg.datasets)}")

    for ds_name in phase_cfg.datasets:
        config = DATASET_CONFIGS[ds_name]
        download_dataset(config, force=force, verbose=verbose)


def list_datasets() -> None:
    """List all available datasets."""
    print("\nAvailable Datasets:")
    print("=" * 80)

    total_download_gb = 0.0
    total_stream_gb = 0.0

    for name, config in DATASET_CONFIGS.items():
        download_type = "stream" if (config.streaming and not config.download_first) else "download"
        print(f"\n{name}:")
        print(f"  Name: {config.name}")
        print(f"  Path: {config.hf_path}" + (f" ({config.hf_name})" if config.hf_name else ""))
        print(f"  Tokens: ~{config.estimated_tokens:,}")
        print(f"  Size: ~{config.estimated_disk_gb}GB ({download_type})")
        print(f"  Description: {config.description[:80]}...")

        if download_type == "download":
            total_download_gb += config.estimated_disk_gb
        else:
            total_stream_gb += config.estimated_disk_gb

    print("\n" + "=" * 80)
    print(f"Total downloadable: ~{total_download_gb:.1f}GB")
    print(f"Streaming only: ~{total_stream_gb:.1f}GB")

    print("\nTraining Phases:")
    print("-" * 40)
    for phase_name, phase_cfg in PHASE_CONFIGS.items():
        print(f"  {phase_name}: {', '.join(phase_cfg.datasets)}")


def verify_dataset(config: DatasetConfig, verbose: bool = True) -> bool:
    """
    Verify a dataset is accessible (either cached or streamable).

    Args:
        config: Dataset configuration
        verbose: Print status

    Returns:
        True if dataset is accessible
    """
    if verbose:
        print(f"Verifying {config.name}...")

    try:
        ds = load_dataset(
            config.hf_path,
            config.hf_name,
            split=config.split,
            streaming=True,  # Always use streaming for verification
                    )

        # Try to get first example
        sample = next(iter(ds))
        if config.text_column not in sample:
            if verbose:
                print(f"  WARNING: Column '{config.text_column}' not found")
                print(f"  Available: {list(sample.keys())}")
            return False

        if verbose:
            text = sample[config.text_column]
            preview = text[:80] + "..." if len(text) > 80 else text
            print(f"  OK: {preview}")

        return True

    except Exception as e:
        if verbose:
            print(f"  FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for neuromorphic LM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.data.download --phase A        # Download Phase A (TinyStories)
  python -m src.data.download --phase B        # Download Phase B datasets
  python -m src.data.download --dataset pg19   # Download specific dataset
  python -m src.data.download --list           # List all datasets
  python -m src.data.download --verify         # Verify all datasets accessible
        """,
    )

    parser.add_argument(
        "--phase",
        type=str,
        choices=list(PHASE_CONFIGS.keys()),
        help="Download all datasets for a training phase",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=list(DATASET_CONFIGS.keys()),
        help="Download a specific dataset",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available datasets",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify datasets are accessible",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if cached",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    if args.list:
        list_datasets()
        return

    if args.verify:
        print("\nVerifying dataset accessibility...")
        all_ok = True
        for name, config in DATASET_CONFIGS.items():
            ok = verify_dataset(config, verbose=verbose)
            if not ok:
                all_ok = False
        print("\n" + ("All datasets OK!" if all_ok else "Some datasets failed verification"))
        sys.exit(0 if all_ok else 1)

    if args.phase:
        download_phase(args.phase, force=args.force, verbose=verbose)
        return

    if args.dataset:
        config = DATASET_CONFIGS[args.dataset]
        download_dataset(config, force=args.force, verbose=verbose)
        return

    # Default: show help
    parser.print_help()


if __name__ == "__main__":
    main()
