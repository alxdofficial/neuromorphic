"""SWE-agent-trajectories source — long agentic tool-use runs, as a resolved/unresolved QAItem.

Source half of the ``qa`` task for the LONG-HORIZON AGENTIC regime: each HF
``nebius/SWE-agent-trajectories`` row is one full agent run at a real GitHub issue — many
Thought+Action / Observation steps (tool calls, command output, file edits) culminating in a
reward-labeled ``target`` (resolved bool). ``facts`` = the ordered ``"Issue: ..."`` / ``"Action:
..."`` / ``"Observation: ..."`` trajectory lines (system-prompt scaffolding dropped); the probe is
the SIMPLEST well-posed question the label supports: "Was the issue resolved?" (a richer "recall a
specific earlier tool observation" probe was considered but dropped — SWE-agent observations are
unstructured command/diff output with no generic, auto-extractable exact-match answer span; this is
the documented "OR simplest" fallback).

ANSWER FORMAT: full sentences ("Yes, the issue was resolved." / "No, the issue was not resolved."),
NOT bare "yes"/"no". The ``qa`` task's un-guessability filter rejects any distractor/filler whose
text CONTAINS the queried answer verbatim — bare "yes"/"no" are common-enough substrings (e.g. "no"
inside "know", "not", "now") that they would spuriously reject nearly every filler drawn from this
corpus's own natural-language trajectory text. The full-sentence answer is practically never a
substring of unrelated trajectory text, so filler survives the filter as intended.

Long trajectories are the point (many-step write pressure over a long horizon), so the ingest script
stratifies by ``target`` and keeps the LONGEST trajectories per stratum — see its docstring for why
naive longest-first would silently skew the resolved/unresolved balance (unresolved runs are
systematically longer than resolved ones). ``facts`` are FRONT+BACK capped to the token budget (keep
the early steps AND the final steps, drop the middle) so the resolution-indicating final turns
usually survive packing even when the run is much longer than the budget.

Data/build (BOUNDED, best-effort, never hangs):
  1. Local ``data/swe_trajectories/<split>.jsonl`` (slim ``{lines,target,...}`` rows) staged by the
     ingest script (LONG-trajectory-prioritized, stratified by target).
  2. Else a bounded NON-STREAMING HF load. ``nebius/SWE-agent-trajectories`` streaming hits a benign
     but real interpreter-SHUTDOWN crash (aiohttp async cleanup racing CPython finalization,
     ``PyGILState_Release``) — harmless to a short-lived ingest script (which force-exits before
     hitting it, see the ingest script's docstring) but unsafe to risk mid-TRAINING, so this
     fallback uses the same non-streaming ``train[lo:hi]`` load ``hotpot_train`` uses for its own
     (different) streaming-hazard reasons. Slower on a cold HF cache, crash-free.
  3. Else (offline, uncached) a clear "run ingest first" error — no silent hang.
Ingest: ``scripts/data_build/ingest/swe_trajectories/download.py``. See DATASETS.md.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from .base import Source, QAItem
from ._corpus import local_jsonl

HF_NAME = "nebius/SWE-agent-trajectories"
_MAX_CHARS = 2000        # per-turn text cap (some tool outputs run to tens of KB)

_YES = "Yes, the issue was resolved."
_NO = "No, the issue was not resolved."


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("swe_trajectories", split)


def _turn_lines(trajectory: list) -> List[str]:
    """trajectory turns -> slim ``"Tag: text"`` lines (system prompt dropped, text char-capped)."""
    lines: List[str] = []
    seen_user = False
    for t in trajectory:
        role = t.get("role")
        text = (t.get("text") or "").strip()
        if role == "system" or not text:
            continue
        if len(text) > _MAX_CHARS:
            text = text[:_MAX_CHARS] + " …[truncated]"
        if role == "user":
            tag = "Observation" if seen_user else "Issue"
            seen_user = True
            lines.append(f"{tag}: {text}")
        elif role == "ai":
            lines.append(f"Action: {text}")
    return lines


def _rows_from_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_from_hf(split: str, n_docs: int) -> List[dict]:
    """Bounded NON-STREAMING slice (see module docstring for why not streaming)."""
    try:
        from datasets import load_dataset
        lo = 0 if split == "train" else n_docs
        hi = lo + n_docs
        ds = load_dataset(HF_NAME, split=f"train[{lo}:{hi}]")
    except Exception as e:
        raise RuntimeError(
            f"[swe_trajectories] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: "
            f"{str(e)[:120]}). Run  python scripts/data_build/ingest/swe_trajectories/download.py  "
            f"to stage a local data/swe_trajectories/{{train,val}}.jsonl sample, then retry "
            f"(works offline once staged)."
        ) from e
    rows: List[dict] = []
    for ex in ds:
        lines = _turn_lines(ex["trajectory"])
        rows.append({"instance_id": ex.get("instance_id", ""), "target": bool(ex.get("target")),
                     "exit_status": ex.get("exit_status", ""), "n_turns": len(lines), "lines": lines})
    return rows


class SweTrajectoriesSource(Source):
    """Yields SWE-agent trajectories as QAItems (answer = resolved yes/no, as a full sentence)."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 max_ctx_tokens: int = 900, tail_tokens: int = 200, **kw):
        self.tok = tokenizer
        self.seed = seed
        self.max_ctx_tokens = max_ctx_tokens
        self.tail_tokens = tail_tokens

        local = _local_jsonl(split)
        if local is not None:
            self.rows = _rows_from_jsonl(local)
            origin = f"data/swe_trajectories/{local.name}"
        else:
            self.rows = _rows_from_hf(split, n_docs)
            origin = f"HF:{HF_NAME}"
        if not self.rows:
            raise ValueError(f"[swe_trajectories] no rows loaded from {origin} (split={split}).")

        n_resolved = sum(1 for r in self.rows if r["target"])
        print(f"[data.swe_trajectories] {split}: {len(self.rows)} trajectories from {origin} "
              f"({n_resolved} resolved / {len(self.rows) - n_resolved} unresolved)", flush=True)
        self._pool = [ln for r in self.rows for ln in r["lines"]]

    def _flen(self, s: str) -> int:
        return len(self.tok(s + "\n", add_special_tokens=False).input_ids)

    def _cap(self, lines: List[str]) -> List[str]:
        """FRONT+BACK cap to the token budget: keep the run's opening AND its final steps (the
        resolution-indicating turns usually sit at the end), dropping the middle when it overflows."""
        lens = [self._flen(ln) for ln in lines]
        if sum(lens) <= self.max_ctx_tokens:
            return lines
        front_budget = self.max_ctx_tokens - self.tail_tokens
        i, used = 0, 0
        while i < len(lines) and used + lens[i] <= front_budget:
            used += lens[i]
            i += 1
        j, used_b, back = len(lines) - 1, 0, []
        while j >= i and used_b + lens[j] <= self.tail_tokens:
            back.append(lines[j])
            used_b += lens[j]
            j -= 1
        back.reverse()
        gap = ["[... trajectory continues ...]"] if j >= i else []
        return lines[:i] + gap + back

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            r = self.rows[rng.randrange(len(self.rows))]
            facts = self._cap(r["lines"])
            answer = _YES if r["target"] else _NO
            out.append(QAItem(
                facts=facts, question="Was the issue resolved?", answer=answer, task_id=0,
                meta={"dataset": "swe_trajectories", "instance_id": r["instance_id"],
                      "exit_status": r["exit_status"], "n_turns": r["n_turns"]}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
