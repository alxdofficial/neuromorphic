"""Shared Tier-2 (pod) eval core: dataset dispatch + per-CONTEXT reuse driver + scoring/record/store glue.

The Tier-2 GPU baselines (KVzip/SnapKV/H2O, MemoryLLM/M+, LCLM) all follow the same shape: load items,
build a per-context MEMORY once, answer each question from it, score with the matching deterministic scorer.
This module centralizes that shape so each runner only supplies its method-specific `(encode_ctx, answer)`.

THE KEY WIN — the local analog of the OpenRouter prefix-cache (see docs/baselines/TIER2_HOSTING.md):
`run_grouped` groups items by their DISTINCT context and calls `encode_ctx` ONCE per context, reusing the
returned memory across every question sharing it. MemoryAgentBench has 36 contexts / 3,071 questions (~85×
reuse); LongMemEval's histories are unique so every group is a singleton (encode per question) — the SAME
code path, no special-casing. Hosting "as normal" (re-encode per question) would repeat the expensive
prefill ~85× on MAB.

Everything here is pure-python (no torch) so it imports on any pod env and is CPU-testable off-GPU. The
per-question `answer` callback returns `(hypothesis: str, finish_reason: str)`; `encode_ctx(context, item)`
returns an opaque method-specific memory object reused across the group.
"""
from __future__ import annotations

import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Optional


# --------------------------------------------------------------------------------------------------
# small utilities (deduped from the three runners)
# --------------------------------------------------------------------------------------------------
def git_commit(path) -> str:
    """Short HEAD, suffixed `-dirty` if the working tree has uncommitted changes — so two DIFFERENT dirty
    implementations don't both tag as the same HEAD and silently resume each other's store (audit #6)."""
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip() or "nogit"
        # `git status --porcelain` catches modified, staged, AND UNTRACKED files (audit #1: a plain
        # `git diff` misses untracked new implementations that would otherwise tag as a clean HEAD). Append a
        # short hash of the actual diff+untracked contents so two DIFFERENT dirty checkouts at the same HEAD get
        # DISTINCT tags → don't resume each other's cache (audit #4).
        porcelain = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                            text=True, stderr=subprocess.DEVNULL).strip()
        if not porcelain:
            return head
        import hashlib
        diff = subprocess.run(["git", "-C", str(path), "diff", "HEAD"], text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout
        digest = hashlib.md5((porcelain + diff).encode())
        untracked = subprocess.check_output(
            ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard", "-z"]
        ).split(b"\0")
        root = Path(path)
        for rel_bytes in sorted(rel for rel in untracked if rel):
            digest.update(b"\0path\0" + rel_bytes + b"\0content\0")
            file_path = root / rel_bytes.decode(errors="surrogateescape")
            if file_path.is_symlink():
                digest.update(file_path.readlink().as_posix().encode())
            elif file_path.is_file():
                with file_path.open("rb") as f:
                    for chunk in iter(lambda: f.read(1024 * 1024), b""):
                        digest.update(chunk)
        h = digest.hexdigest()[:6]
        return f"{head}-dirty{h}"
    except Exception:  # noqa: BLE001
        return "nogit"


def seed_everything(seed: int) -> None:
    """Seed py/np/torch/cuda so eviction / memory-drop / generation are reproducible across reruns."""
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def finish_reason_of(gen_ids, max_new_tokens: int, eos_id) -> str:
    """'length' if generation ran to the cap without emitting EOS (answer may be truncated → retryable via
    ResultStore.done_ids), else 'stop'. `gen_ids` = the NEW tokens only (after the prompt)."""
    n = gen_ids.shape[0] if hasattr(gen_ids, "shape") else len(gen_ids)
    if n >= max_new_tokens and (eos_id is None or int(gen_ids[-1]) != eos_id):
        return "length"
    return "stop"


# --------------------------------------------------------------------------------------------------
# dataset dispatch — the runners are now benchmark-agnostic
# --------------------------------------------------------------------------------------------------
DATASETS = ("longmemeval", "memoryagentbench")


def load_items(dataset: str, variant: str = "s", max_examples: Optional[int] = None) -> list[dict]:
    """Load per-question items for either benchmark. Both loaders return dicts carrying at least
    {question_id, question, answer, full_history, question_type}; MAB additionally carries
    {competency, source, metric, question_template}. LongMemEval carries {question_date}."""
    if dataset == "longmemeval":
        from src.memory.data.longmemeval import load_longmemeval_text
        return load_longmemeval_text(variant=variant, max_examples=max_examples)
    if dataset == "memoryagentbench":
        from src.memory.data.memoryagentbench import load_memoryagentbench_text
        return load_memoryagentbench_text(variant=variant, max_examples=max_examples)
    raise ValueError(f"unknown dataset {dataset!r} (choose from {DATASETS})")


def format_query(it: dict, dataset: str) -> str:
    """The semantic question text a method should ask AFTER its memory is loaded — BEFORE the method wraps it
    in its own chat template. LongMemEval: anchor temporal questions to their date (matches the Tier-1 panel).
    MemoryAgentBench: apply the per-competency query template VERBATIM (contains the task instruction, e.g.
    ICL 'output only the label', factconsolidation 'newer fact = larger serial number'). The template uses
    str.replace (NOT .format) so a literal `{label}` in the ICL template survives."""
    q = it["question"]
    if dataset == "memoryagentbench":
        tmpl = it.get("question_template") or ""
        return tmpl.replace("{question}", q) if tmpl else q
    d = it.get("question_date")
    return f"Current Date: {d}\n{q}" if d else q


def make_record(it: dict, hyp: str = "", error: Optional[str] = None,
                finish_reason: Optional[str] = None) -> dict:
    """Superset store/score record — carries every field BOTH deterministic scorers read:
    score_longmemeval → {question, answer, hypothesis, question_type, question_id};
    score_memoryagentbench → {question, answer, hypothesis, metric, competency, source, question_id}.
    Missing keys default to None (LongMemEval items have no competency/metric; MAB has no question_type
    beyond the competency it already sets question_type to)."""
    return {
        "question_id": str(it["question_id"]), "question": it["question"], "answer": it["answer"],
        "hypothesis": hyp, "question_type": it.get("question_type"),
        "competency": it.get("competency"), "source": it.get("source"), "metric": it.get("metric"),
        "finish_reason": finish_reason or ("error" if error else "stop"), "error": error,
    }


def group_by_context(items: list[dict]) -> "OrderedDict[str, list[dict]]":
    """Group items by their distinct `full_history` (the context), preserving first-seen order. MAB collapses
    to 36 groups (~85 Q each) → encode once per group; LongMemEval's unique histories yield 1-item groups →
    encode per question. Same code, benchmark decides the amount of reuse."""
    groups: "OrderedDict[str, list[dict]]" = OrderedDict()
    for it in items:
        groups.setdefault(it.get("full_history", "") or "", []).append(it)
    return groups


# --------------------------------------------------------------------------------------------------
# the reuse driver — the heart of the local caching win
# --------------------------------------------------------------------------------------------------
def run_grouped(items: list[dict], encode_ctx: Callable, answer: Callable, store,
                log_prefix: str, release: Optional[Callable] = None) -> None:
    """Group by context; per context: encode ONCE, then answer every (not-yet-done) question from that one
    memory. RESUMABLE (skips store.done_ids) + crash-safe (each answer appended immediately). An encode
    failure marks the whole group's pending questions errored (not a wrong answer — excluded from scoring,
    retried on rerun); a single-question failure only errors that question.

      encode_ctx(context: str, first_item: dict) -> mem      # build the reusable per-context memory
      answer(mem, item: dict) -> (hypothesis: str, finish_reason: str)
      release() -> None       # optional: called AFTER the previous mem's ref is dropped, to free GPU cache

    MEMORY (audit #9): the previous context's memory is dropped + `release()`d BEFORE `encode_ctx` builds the
    next, so two large on-GPU snapshots (e.g. M+'s ~2.5GB pool copies) never overlap — `empty_cache` cannot
    free a still-referenced tensor, so the ref MUST go first.
    """
    done = store.done_ids()
    groups = group_by_context(items)
    n_groups = len(groups)
    mem = None
    for gi, (ctx, gitems) in enumerate(groups.items()):
        pending = [it for it in gitems if str(it["question_id"]) not in done]
        if not pending:
            continue
        if mem is not None:                       # free the PREVIOUS context's memory before allocating next
            mem = None                            # drop the ref first (else empty_cache can't reclaim it)
            if release is not None:
                try:
                    release()
                except Exception as e:  # noqa: BLE001
                    print(f"{log_prefix} WARN: release() failed: {type(e).__name__}: {e}")
        try:
            mem = encode_ctx(ctx, gitems[0])
        except Exception as e:  # noqa: BLE001 — encode is the expensive step; a failure dooms the whole group
            print(f"{log_prefix} ENCODE ERROR ctx {gi + 1}/{n_groups} ({len(pending)} Q): "
                  f"{type(e).__name__}: {e}")
            for it in pending:
                store.append(make_record(it, error=f"encode: {type(e).__name__}: {e}"))
            continue
        for it in pending:
            try:
                hyp, fr = answer(mem, it)
                store.append(make_record(it, hyp=hyp, finish_reason=fr))
            except Exception as e:  # noqa: BLE001 — crash-safe per question
                print(f"{log_prefix} ERROR on {it['question_id']}: {type(e).__name__}: {e}")
                store.append(make_record(it, error=f"{type(e).__name__}: {e}"))
        if (gi + 1) % 5 == 0 or gi + 1 == n_groups:
            print(f"{log_prefix} context {gi + 1}/{n_groups} done ({len(pending)} Q this ctx)", flush=True)
    if mem is not None and release is not None:   # free the last context's memory
        mem = None
        try:
            release()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------------------------------
# scoring + finalize (deduped end-of-run)
# --------------------------------------------------------------------------------------------------
def valid_for_scoring(r: dict) -> bool:
    """A record counts toward accuracy only with a real, COMPLETE answer — exclude gen errors AND any
    TERMINAL/incomplete finish_reason (length cutoff, or a provider error/content_filter). A refusal/blank
    from a content filter is a harness artifact, not a wrong answer (audit #2)."""
    return not r.get("error") and r.get("finish_reason") not in ("length", "error", "content_filter")


def score_dataset(dataset: str, records: list[dict], use_bem: bool):
    if dataset == "longmemeval":
        from src.memory.eval import score_longmemeval
        return score_longmemeval(records, use_bem=use_bem)
    if dataset == "memoryagentbench":
        from src.memory.eval import score_memoryagentbench
        return score_memoryagentbench(records, use_bem=use_bem)
    raise ValueError(f"unknown dataset {dataset!r}")


def build_tag(dataset: str, method: str, model_slug: str, variant: str, n: int,
              knob: str, g: int, seed: int, commit: str) -> str:
    """Artifact/store name carrying everything needed to tell two runs apart."""
    return (f"{dataset}__{method}__{model_slug}__{variant}__n{n}__{knob}__g{g}__seed{seed}__{commit}")


def finalize(dataset: str, method: str, model: str, items: list[dict], store, use_bem: bool,
             extra_meta: dict, out_dir: Path, tag: str, log_prefix: str) -> dict:
    """Score the current selection, fold verdicts back into the store, write the aggregate JSON, print a
    one-line summary. Returns the aggregate (minus per-item details). Same JSON shape as Tier-1 so the panels
    line up in report.py."""
    import json
    # SELECTION-SCOPED (audit #2): score ONLY the records for the currently-selected items — the store may
    # hold stale records from a larger earlier run (--max-examples is a real scoping knob), which would
    # otherwise be scored too and push coverage past 1.0. Mirrors Tier-1 run_one's `want` scoping.
    want = {str(it["question_id"]) for it in items}
    sel = [r for r in store.all_records() if str(r.get("question_id")) in want]
    records = [r for r in sel if valid_for_scoring(r)]
    agg = score_dataset(dataset, records, use_bem=use_bem)
    store.merge_verdicts(agg.get("details", []))
    store.compact()
    n_err = sum(1 for r in sel if r.get("error"))
    n_cut = sum(1 for r in sel if r.get("finish_reason") == "length")
    meta = {"n": len(sel), "n_scored": len(records), "n_errors": n_err, "n_gen_cutoff": n_cut,
            "coverage": round(len(records) / len(items), 4) if items else None,
            # record the scoring policy in-artifact (audit #6): LongMemEval BEM threshold, or None if disabled.
            "bem_threshold": (0.85 if (use_bem and dataset == "longmemeval") else None), **extra_meta}
    # print whatever headline keys the scorer produced (LongMemEval: abstention/task_avg; MAB: n_scored/skip)
    head = f"overall_acc={agg.get('overall_accuracy', float('nan')):.3f}"
    for k in ("task_averaged_accuracy", "abstention_accuracy", "n_nonabstention", "n_scored", "n_skipped"):
        if k in agg:
            v = agg[k]
            head += f"  {k}={v:.3f}" if isinstance(v, float) else f"  {k}={v}"
    print(f"\n{log_prefix} {head}  errors={n_err}  cutoffs={n_cut}  coverage={meta['coverage']}")
    payload = {"dataset": dataset, "method": method, "model": model, "mode": method,
               "meta": meta, "aggregate": {k: v for k, v in agg.items() if k != "details"},
               "store": str(store.path)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(payload, indent=1))
    print(f"{log_prefix} wrote {out_dir / f'{tag}.json'}")
    return agg
