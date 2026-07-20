"""Per-question result store for the Phase-2 baselines — crash-safe, resumable, analysis-ready.

Each (dataset, model, mode) run gets one JSONL file under `outputs/baselines/cache/`, one line per question:
the question, gold, generation, token/cost counts, retrieval indices, truncation flag, API error (if any),
and — after scoring — the correctness verdict + method. Design goals:

  * **Resumable** — a rerun loads the store, skips questions already ANSWERED (non-error hypothesis), and only
    re-requests the missing/errored ones. So a mid-run crash never re-spends on completed questions.
  * **Crash-safe** — every result is appended (fsync-flushed) the moment it returns, not batched at the end.
  * **Analysis-ready** — everything needed to recompute scores or inspect generations lives on disk (the giant
    prompt/history is NOT stored — it is deterministically rebuildable from the dataset).

Duplicate lines for a question_id (e.g. an errored attempt later retried) are resolved last-wins on load;
`compact()` rewrites the file deduped. Appends are synchronous (no await inside), so they are atomic with
respect to other asyncio coroutines in the single-threaded event loop.
"""
from __future__ import annotations

import json
import os
from pathlib import Path


def store_path(out_dir, dataset: str, model: str, mode: str, config_sig: str = "") -> Path:
    """Cache path for one (dataset, model, mode) run. `config_sig` (a short hash of the generation-affecting
    knobs — max_tokens, variant, retrieval-k, source/size filters) is appended so that CHANGING any of those
    starts a fresh store instead of silently reusing answers computed under different settings."""
    slug = model.split("/")[-1].replace(":", "-")
    tag = f"{dataset}__{slug}__{mode}" + (f"__{config_sig}" if config_sig else "")
    return Path(out_dir) / "cache" / f"{tag}.jsonl"


class ResultStore:
    def __init__(self, path):
        self.path = Path(path)
        self.records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue                          # tolerate a torn final line from a crash
                qid = rec.get("question_id")
                if qid is not None:
                    self.records[str(qid)] = rec      # last-wins

    def done_ids(self) -> set[str]:
        """Questions with a usable, COMPLETE answer → skip on rerun. Retried on rerun: API errors AND ANY
        answer cut off at the token cap (finish_reason='length'), whether the content is blank OR a
        partial/truncated answer. A truncated answer may have the real answer clipped off the end, so it is
        neither trustworthy to score nor safe to freeze — a rerun with a higher --max-tokens re-requests it."""
        out = set()
        for q, r in self.records.items():
            if r.get("error"):
                continue
            # a TERMINAL finish_reason is incomplete/retryable even when the `error` field is null (audit #6:
            # old caches predate api_client attaching an error to content_filter/error responses). length =
            # cut-off; error/content_filter = provider refusal. All three re-request on rerun.
            if r.get("finish_reason") in ("length", "error", "content_filter"):
                continue
            out.add(q)
        return out

    def append(self, rec: dict) -> None:
        """Append one result line and flush+fsync so a crash right after can't lose it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.records[str(rec["question_id"])] = rec
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def all_records(self) -> list[dict]:
        return list(self.records.values())

    def merge_verdicts(self, details: list[dict]) -> None:
        """Fold per-item score verdicts (from a scorer's `details`) back into the stored records."""
        for d in details:
            qid = str(d.get("question_id", ""))
            if qid in self.records:
                self.records[qid]["correct"] = d.get("correct")
                self.records[qid]["score_method"] = d.get("method") or d.get("metric")

    def compact(self) -> None:
        """Rewrite the file deduped (one line per question_id, with verdicts) via atomic replace."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".jsonl.tmp")
        with open(tmp, "w") as f:
            for rec in self.records.values():
                f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(self.path)

    # ---- summary (for status.py / analysis) ----
    def summary(self) -> dict:
        recs = self.all_records()
        errored = [r for r in recs if r.get("error")]
        answered = [r for r in recs if not r.get("error")]
        scored = [r for r in answered if r.get("correct") is not None]
        correct = [r for r in scored if r.get("correct")]
        return {
            "n_total": len(recs),
            "n_answered": len(answered),
            "n_errored": len(errored),
            "n_scored": len(scored),
            "n_correct": len(correct),
            "accuracy": (len(correct) / len(scored)) if scored else None,
            "errored_ids": [r.get("question_id") for r in errored],
        }
