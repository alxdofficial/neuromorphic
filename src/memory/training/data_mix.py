"""Mixed multi-task dataloader + fixed val-set construction, built from the 4-layer data spec.

Each mixed-task name (mae/babi/continuation/condrecon_bio) resolves via ``mixes.TASK_SPEC`` to a
(Source, Task) pair; this composes an ``EpisodeSpec`` from the runtime args and builds one loader
per task through ``SOURCE_REGISTRY × get_task × make_task_dataloader``. Adding a source/task no
longer edits this file — only the per-source construction kwargs (the genuinely source-specific
bits: fineweb min_len/src-tok, babi tasks, bio world/n_facts) live in ``_build_source``.

See ``docs/data_arch_plan.md`` (Orchestration). Replaces the old hardcoded per-task branches.
"""
from __future__ import annotations

from src.memory.data.mixes import TASK_SPEC, CONDRECON_BIO_N_PAIRS, CONDRECON_BIO_N_FACTS
from src.memory.data.sources import SOURCE_REGISTRY
from src.memory.data.tasks import get_task
from src.memory.data.tasks.base import make_task_dataloader
from src.memory.data.schedule import EpisodeSpec

from .utils import materialize_val_set


def _query_lag(bio_query_window) -> str:
    """CLI --bio-query-window → EpisodeSpec.query_lag: None=any, 0=first(early), -1=last(recent)."""
    if bio_query_window is None:
        return "any"
    if bio_query_window == 0:
        return "early"
    if bio_query_window == -1:
        return "recent"
    return str(int(bio_query_window))


def _build_source(src_name, task_style, tokenizer, *, split, ctx_len, predict_len,
                  mae_src_tok, babi_tasks, seed):
    """The ONE place source-specific construction kwargs live (each source has genuinely different
    build params). ``split`` is "train" | "val"; sources translate to their own split vocab."""
    if src_name == "fineweb":
        min_len = ctx_len + (predict_len if task_style == "continuation" else 0)
        return SOURCE_REGISTRY["fineweb"](
            tokenizer, split=split, src_tokenizer_name=mae_src_tok, min_len=min_len, seed=seed)
    if src_name == "multicorpus":
        # continuation/mae VARIETY: unions fineweb+pile+redpajama+code (skips unreachable ones).
        min_len = ctx_len + (predict_len if task_style == "continuation" else 0)
        return SOURCE_REGISTRY["multicorpus"](
            tokenizer, split=split, src_tokenizer_name=mae_src_tok, min_len=min_len, seed=seed)
    if src_name == "babi":
        return SOURCE_REGISTRY["babi"](
            tokenizer, split=("validation" if split != "train" else "train"),
            tasks=babi_tasks, seed=seed)
    if src_name == "qa_multi":
        # REAL QA variety (squad/triviaqa/hotpot/musique/multiwoz) — each sub-source uses its own
        # train / held-out-train-slice split; "val" maps to the held-out slice, not the eval sets.
        return SOURCE_REGISTRY["qa_multi"](tokenizer, split=split, seed=seed)
    if src_name == "bio":
        return SOURCE_REGISTRY["bio"](
            tokenizer, split=("validation" if split != "train" else "train"),
            world_seed=0, n_facts=CONDRECON_BIO_N_FACTS, seed=seed)
    raise ValueError(f"no source-construction rule for {src_name!r} (mixed task backing)")


def _build_loader(mix_task, tokenizer, cfg, *, split, ctx_len, m_slots, mae_src_tok,
                  babi_tasks, predict_len, window_size, bio_query_window, seed, num_workers):
    """Build ONE mixed-task loader: TASK_SPEC[mix_task] → (source, task) × EpisodeSpec × cfg."""
    meta = TASK_SPEC[mix_task]
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    source = _build_source(meta.source, meta.task_style, tokenizer, split=split, ctx_len=ctx_len,
                           predict_len=predict_len, mae_src_tok=mae_src_tok, babi_tasks=babi_tasks,
                           seed=seed)
    spec = EpisodeSpec(source=meta.source, task=meta.task_style, total_len=ctx_len,
                       window_size=window_size, n_inputs=CONDRECON_BIO_N_PAIRS, n_queries=1,
                       query_lag=_query_lag(bio_query_window), predict_len=predict_len)
    task = get_task(meta.task_style)
    if hasattr(task, "m_slots"):                 # MAE task: the capacity-relative memory budget
        task.m_slots = m_slots
    return make_task_dataloader(source, task, spec, tokenizer, batch_size=cfg.batch_size,
                                pad_token_id=pad, seed=seed, num_workers=num_workers)


def make_mixed_train_dataloaders(mixed_tasks, tokenizer, cfg, *, ctx_len: int,
                                 m_slots: int, mae_src_tok: str, babi_tasks,
                                 predict_len: int, num_workers: int = 1,
                                 train_seed: int = 42, window_size: int = None,
                                 bio_query_window: int = None) -> dict:
    """One TRAIN dataloader per mixed task (uniform interface: context_len/chunk = ctx_len, M =
    m_slots). ``bio_query_window`` (+ ``window_size``) turns condrecon_bio into a streaming-write
    retention probe (queried pair pinned to a window; the rest are distractors)."""
    return {
        t: _build_loader(t, tokenizer, cfg, split="train", ctx_len=ctx_len, m_slots=m_slots,
                         mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
                         window_size=window_size, bio_query_window=bio_query_window,
                         seed=train_seed, num_workers=num_workers)
        for t in mixed_tasks
    }


def make_mixed_val_sets(mixed_tasks, tokenizer, cfg, val_batches, *, ctx_len: int,
                        m_slots: int, mae_src_tok: str, babi_tasks,
                        predict_len: int, window_size: int = None,
                        bio_query_window: int = None) -> dict:
    """One materialized (fixed) VAL set per mixed task — disjoint val seed (7) so eval never overlaps
    the training stream. ``bio_query_window`` mirrors the train-side streaming retention placement."""
    sets = {}
    for t in mixed_tasks:
        dl = _build_loader(t, tokenizer, cfg, split="val", ctx_len=ctx_len, m_slots=m_slots,
                           mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
                           window_size=window_size, bio_query_window=bio_query_window,
                           seed=7, num_workers=0)
        sets[t] = materialize_val_set(dl, val_batches)
    return sets
