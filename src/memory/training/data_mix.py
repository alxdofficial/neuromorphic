"""Mixed multi-task dataloader + fixed val-set construction, built from the 4-layer data spec.

Each mixed-task name (mae/babi/qa_rc/continuation/condrecon_bio) resolves via ``mixes.TASK_SPEC`` to
a (Source, Task) pair; this composes an ``EpisodeSpec`` from the runtime args and builds one loader
per task through ``SOURCE_REGISTRY × get_task × make_task_dataloader``. Adding a source/task no
longer edits this file — only the per-source construction kwargs (the genuinely source-specific
bits: fineweb min_len/src-tok, babi tasks, bio world/n_facts) live in ``_build_source``.

See ``docs/history/docs/history/data_arch_plan.md`` (Orchestration). Replaces the old hardcoded per-task branches.
"""
from __future__ import annotations

from src.memory.data.mixes import TASK_SPEC, CONDRECON_BIO_N_PAIRS, CONDRECON_BIO_N_FACTS, resolve_task
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
                  mae_src_tok, babi_tasks, seed, bio_world_seed=0):
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
            world_seed=bio_world_seed, n_facts=CONDRECON_BIO_N_FACTS, seed=seed)
    raise ValueError(f"no source-construction rule for {src_name!r} (mixed task backing)")


def _build_loader(mix_task, tokenizer, cfg, *, split, ctx_len, m_slots, mae_src_tok,
                  babi_tasks, predict_len, window_size, bio_query_window, seed, num_workers):
    """Build ONE mixed-task loader: TASK_SPEC[mix_task] → (source, task) × EpisodeSpec × cfg."""
    meta = TASK_SPEC[resolve_task(mix_task)]     # accept old aliases (mae/qa_rc/condrecon_bio)
    pad = cfg.pad_token_id if cfg.pad_token_id is not None else 0
    source = _build_source(meta.source, meta.task_style, tokenizer, split=split, ctx_len=ctx_len,
                           predict_len=predict_len, mae_src_tok=mae_src_tok, babi_tasks=babi_tasks,
                           seed=seed, bio_world_seed=getattr(cfg, "cond_recon_bio_world_seed", 0))
    # query_lag: an explicit --bio-query-window pin wins for the BIO source only (its streaming-retention
    # probe) — NOT other vary_lag tasks (babi/doc_qa), which keep sampling early/recent/any per episode.
    # vary_lag tasks default to "vary"; the rest to "any". n_queries is NOT set here — the qa/
    # reconstruction tasks read it from the SOURCE (Source.pack_n_queries).
    if bio_query_window is not None and meta.source == "bio":
        lag = _query_lag(bio_query_window)
    else:
        lag = "vary" if meta.vary_lag else "any"
    spec = EpisodeSpec(source=meta.source, task=meta.task_style, total_len=ctx_len,
                       window_size=window_size, n_inputs=CONDRECON_BIO_N_PAIRS,
                       query_lag=lag, predict_len=predict_len)
    task = get_task(meta.task_style)
    if hasattr(task, "m_slots"):                 # MAE task: the capacity-relative memory budget
        task.m_slots = m_slots
    return make_task_dataloader(source, task, spec, tokenizer, batch_size=cfg.batch_size,
                                pad_token_id=pad, seed=seed, num_workers=num_workers)


def _default_workers() -> int:
    """Per-task-loader worker count, scaled to the host (there are len(mixed_tasks) loaders, each
    spawning this many). Capped so a big box (256 vCPUs) doesn't spawn hundreds; floored at 2 so even a
    small host overlaps tokenization with the GPU step. Was hardcoded 1 → single-threaded tokenization
    starved the GPU (util ~25% on a cold-cache pod)."""
    import os
    return min(4, max(2, (os.cpu_count() or 8) // 16))


def make_mixed_train_dataloaders(mixed_tasks, tokenizer, cfg, *, ctx_len: int,
                                 m_slots: int, mae_src_tok: str, babi_tasks,
                                 predict_len: int, num_workers: int = None,
                                 train_seed: int = 42, window_size: int = None,
                                 bio_query_window: int = None) -> dict:
    """One TRAIN dataloader per mixed task (uniform interface: context_len/chunk = ctx_len, M =
    m_slots). ``bio_query_window`` (+ ``window_size``) turns condrecon_bio into a streaming-write
    retention probe (queried pair pinned to a window; the rest are distractors)."""
    if num_workers is None:
        num_workers = _default_workers()
        print(f"  [data] {num_workers} loader workers/task × {len(mixed_tasks)} tasks "
              f"(persistent + prefetch; parallel tokenization)", flush=True)
    # Per-task seed OFFSET: without it, tasks that share a doc pool (mae=fineweb, continuation=
    # multicorpus⊇fineweb) draw the SAME doc+offset in lockstep from an identical RNG, correlating the
    # two objectives and halving effective corpus diversity. i*10_007 (prime) decorrelates the streams.
    # ×2 makes every TRAIN seed EVEN; the val side (below) is ODD → train/val RNG streams are provably
    # disjoint for ANY --seed (else train_seed==7 would collide with the val base).
    return {
        t: _build_loader(t, tokenizer, cfg, split="train", ctx_len=ctx_len, m_slots=m_slots,
                         mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
                         window_size=window_size, bio_query_window=bio_query_window,
                         seed=(train_seed + i * 10_007) * 2, num_workers=num_workers)
        for i, t in enumerate(mixed_tasks)
    }


def make_mixed_val_sets(mixed_tasks, tokenizer, cfg, val_batches, *, ctx_len: int,
                        m_slots: int, mae_src_tok: str, babi_tasks,
                        predict_len: int, window_size: int = None,
                        bio_query_window: int = None) -> dict:
    """One materialized (fixed) VAL set per mixed task — disjoint val seed (7) so eval never overlaps
    the training stream. ``bio_query_window`` mirrors the train-side streaming retention placement."""
    # Parallelize the one-time val-set BUILD: it drains val_batches×5 tasks synchronously at STARTUP with
    # the GPU fully idle (audit). Workers overlap the tokenization; the loader is drained then discarded so
    # persistent workers don't matter, but >0 workers still cut the serial startup stall.
    _vw = _default_workers()
    sets = {}
    for i, t in enumerate(mixed_tasks):
        dl = _build_loader(t, tokenizer, cfg, split="val", ctx_len=ctx_len, m_slots=m_slots,
                           mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
                           window_size=window_size, bio_query_window=bio_query_window,
                           seed=(7 + i * 10_007) * 2 + 1, num_workers=_vw)   # ODD → disjoint from even train seeds
        sets[t] = materialize_val_set(dl, val_batches)
    return sets
