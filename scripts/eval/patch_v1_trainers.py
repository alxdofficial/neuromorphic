#!/usr/bin/env python3
"""Surgical patch: add --composite-dir support to v1.1/v1.2/v1.4 train_wave1_retrieval.py.

Keeps the original architecture + loss code intact. Only the data adapter is changed
so we can train all v1.x versions on the same composite_v1 data as v2 for an
apples-to-apples comparison.
"""
import re
from pathlib import Path

WORKTREES = [
    "/home/alex/code/neuromorphic-v1.1",
    "/home/alex/code/neuromorphic-v1.2",
    "/home/alex/code/neuromorphic-v1.4",
]

for w in WORKTREES:
    f = Path(w) / "scripts/training/train_wave1_retrieval.py"
    src = f.read_text()
    orig = src

    # 1. Add CompositeRetrievalAdapter import after the existing trajectory_memory imports.
    import_marker = "from src.trajectory_memory.training.phase1_retrieval import"
    if "CompositeRetrievalAdapter" not in src:
        # Find the closing paren of the multi-line import
        idx = src.index(import_marker)
        # Find the next ")\n" after that point
        end = src.index(")\n", idx) + 2
        injected = (
            "from scripts.data.wave1.common.sampler import "
            "CompositeRetrievalAdapter  # noqa: E402\n"
        )
        src = src[:end] + injected + src[end:]

    # 2. Make --train-jsonl optional.
    src = src.replace(
        'ap.add_argument("--train-jsonl", type=str, required=True,',
        'ap.add_argument("--train-jsonl", type=str, default=None,',
    )

    # 3. Add --composite-dir and --composite-val-dir args after --val-jsonl.
    val_jsonl_block_end = (
        'ap.add_argument("--val-jsonl", type=str, default=None,\n'
        '                    help="Path to facts_val.jsonl (optional)")'
    )
    if "--composite-dir" not in src:
        composite_args = (
            val_jsonl_block_end + "\n"
            '    ap.add_argument("--composite-dir", type=str, default=None,\n'
            '                    help="Composite dataset directory '
            '(passages.jsonl + questions.jsonl).")\n'
            '    ap.add_argument("--composite-val-dir", type=str, default=None,\n'
            '                    help="Optional composite val directory.")'
        )
        src = src.replace(val_jsonl_block_end, composite_args)

    # 4. Replace the sampler-init block.
    old_block = """    # ── Samplers ──
    train_sampler = RetrievalSampler(args.train_jsonl, seed=args.seed)
    val_sampler = None
    if args.val_jsonl:
        val_sampler = RetrievalSampler(args.val_jsonl, seed=args.seed + 1)
    print(f"Train pool: {len(train_sampler.facts)} facts, "
          f"{len(train_sampler.keys)} distinct (class,attr) keys")
    if val_sampler:
        print(f"Val pool:   {len(val_sampler.facts)} facts, "
              f"{len(val_sampler.keys)} distinct (class,attr) keys")"""

    new_block = """    # ── Samplers ──
    if args.composite_dir is not None:
        from pathlib import Path as _P
        cd = _P(args.composite_dir)
        train_sampler = CompositeRetrievalAdapter(
            cd / "passages.jsonl", cd / "questions.jsonl",
            chunk_size=8, seed=args.seed,
        )
        val_sampler = None
        if args.composite_val_dir is not None:
            vd = _P(args.composite_val_dir)
            val_sampler = CompositeRetrievalAdapter(
                vd / "passages.jsonl", vd / "questions.jsonl",
                chunk_size=8, seed=args.seed + 1,
            )
        print(f"Train composite: {len(train_sampler.facts)} questions")
        if val_sampler:
            print(f"Val composite:   {len(val_sampler.facts)} questions")
    else:
        if args.train_jsonl is None:
            raise SystemExit("must specify either --composite-dir or --train-jsonl")
        train_sampler = RetrievalSampler(args.train_jsonl, seed=args.seed)
        val_sampler = None
        if args.val_jsonl:
            val_sampler = RetrievalSampler(args.val_jsonl, seed=args.seed + 1)
        print(f"Train pool: {len(train_sampler.facts)} facts, "
              f"{len(train_sampler.keys)} distinct (class,attr) keys")
        if val_sampler:
            print(f"Val pool:   {len(val_sampler.facts)} facts, "
                  f"{len(val_sampler.keys)} distinct (class,attr) keys")"""

    if old_block in src:
        src = src.replace(old_block, new_block)
    else:
        print(f"  WARN: {f} — old sampler block not found, may need manual patch")

    if src != orig:
        f.write_text(src)
        print(f"  ✓ patched {f}")
    else:
        print(f"  • no changes needed: {f}")
