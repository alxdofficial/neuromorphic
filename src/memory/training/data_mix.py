"""Mixed multi-task dataloader + fixed val-set construction (one loader per task).

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). No logic changes.
The explicit per-task branches are kept (the makers have different signatures) — do NOT try to
unify them through REGISTRY. The bio task-construction constants + the default mix are sourced
from ``src.memory.data.mixes``.
"""
from __future__ import annotations

from src.memory.data.mae import make_long_passage_mae_dataloader
from src.memory.data.babi import make_babi_dataloader
from src.memory.data.continuation import make_continuation_dataloader
from src.memory.data.bio import make_conditioned_reconstruction_bio_dataloader
from src.memory.data.mixes import (
    DEFAULT_TRAIN_MIX,
    CONDRECON_BIO_N_PAIRS as MIXED_CONDRECON_BIO_N_PAIRS,
    CONDRECON_BIO_N_FACTS as MIXED_CONDRECON_BIO_N_FACTS,
)

from .utils import materialize_val_set


def make_mixed_train_dataloaders(mixed_tasks, tokenizer, cfg, *, ctx_len: int,
                                 m_slots: int, mae_src_tok: str, babi_tasks,
                                 predict_len: int, num_workers: int = 1,
                                 train_seed: int = 42, window_size: int = None,
                                 bio_query_window: int = None) -> dict:
    """One TRAIN dataloader per mixed task, all on the uniform interface
    (context_len/chunk = ctx_len, M = m_slots). Homogeneous batches; the
    round-robin alternates which loader is pulled each step.

    ``bio_query_window`` (+ ``window_size``) turns condrecon_bio into a STREAMING-WRITE
    retention probe: the queried key→value pair is pinned into that window (0 = first =
    max lag), so the other pairs are distractors between it and the end-of-context query."""
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    dls = {}
    for t in mixed_tasks:
        if t == "mae":
            dls[t] = make_long_passage_mae_dataloader(
                tokenizer, batch_size=cfg.batch_size, src_tokenizer_name=mae_src_tok,
                split="train", ctx_len=ctx_len, m_slots=m_slots, seed=train_seed,
                pad_token_id=_pad, num_workers=num_workers)
        elif t == "babi":
            dls[t] = make_babi_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                split="train", seed=train_seed, pad_token_id=_pad, tasks=babi_tasks,
                num_workers=num_workers)
        elif t == "continuation":
            dls[t] = make_continuation_dataloader(
                tokenizer, batch_size=cfg.batch_size, compress_len=ctx_len,
                predict_len=predict_len, split="train", seed=train_seed, pad_token_id=_pad,
                objective="continuation", src_tokenizer_name=mae_src_tok,
                num_workers=num_workers)
        elif t == "condrecon_bio":
            dls[t] = make_conditioned_reconstruction_bio_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                n_pairs=MIXED_CONDRECON_BIO_N_PAIRS, n_query=1, n_facts=MIXED_CONDRECON_BIO_N_FACTS,
                split="train", world_seed=0, stream_seed=train_seed, pad_token_id=_pad, num_workers=num_workers,
                window_size=window_size, query_window=bio_query_window)
        else:
            raise ValueError(f"unknown mixed task {t!r} (expected mae/babi/continuation/condrecon_bio)")
    return dls


def make_mixed_val_sets(mixed_tasks, tokenizer, cfg, val_batches, *, ctx_len: int,
                        m_slots: int, mae_src_tok: str, babi_tasks,
                        predict_len: int, window_size: int = None,
                        bio_query_window: int = None) -> dict:
    """One materialized (fixed) VAL set per mixed task — disjoint val seed so the
    eval data never overlaps the training stream. ``bio_query_window`` mirrors the
    train-side streaming retention placement (see make_mixed_train_dataloaders)."""
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    sets = {}
    for t in mixed_tasks:
        if t == "mae":
            dl = make_long_passage_mae_dataloader(
                tokenizer, batch_size=cfg.batch_size, src_tokenizer_name=mae_src_tok,
                split="val", ctx_len=ctx_len, m_slots=m_slots, seed=7,
                pad_token_id=_pad, num_workers=0)
        elif t == "babi":
            dl = make_babi_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                split="validation", seed=7, pad_token_id=_pad, tasks=babi_tasks,
                num_workers=0)
        elif t == "continuation":
            dl = make_continuation_dataloader(
                tokenizer, batch_size=cfg.batch_size, compress_len=ctx_len,
                predict_len=predict_len, split="validation", seed=7, pad_token_id=_pad,
                objective="continuation", src_tokenizer_name=mae_src_tok, num_workers=0)
        elif t == "condrecon_bio":
            dl = make_conditioned_reconstruction_bio_dataloader(
                tokenizer, context_len=ctx_len, batch_size=cfg.batch_size,
                n_pairs=MIXED_CONDRECON_BIO_N_PAIRS, n_query=1, n_facts=MIXED_CONDRECON_BIO_N_FACTS,
                split="validation", world_seed=0, stream_seed=7, pad_token_id=_pad, num_workers=0,
                window_size=window_size, query_window=bio_query_window)
        sets[t] = materialize_val_set(dl, val_batches)
    return sets
