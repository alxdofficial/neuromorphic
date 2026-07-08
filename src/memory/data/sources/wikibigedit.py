"""WikiBigEdit source — sequential fact-EDIT stream (streaming write→update→recall probe).

Source half of the ``qa`` task for the STREAMING FACT-UPDATE regime: WikiBigEdit
(HF ``lukasthede/WikiBigEdit``) ships ~502k Wikidata edits split across 8 chronological TIMESTEPS
(2024-02 .. 2024-07, one JSON file per timestep). Each row is one fact edit (subject, relation,
object) plus five pre-authored PROBE questions over that same fact, each with an exact-match answer
field:

  - ``update``   — canonical question phrasing                       -> ``ans``
  - ``rephrase`` — a paraphrase of the same question                 -> ``ans``
  - ``personas`` — a casual/persona-styled phrasing                   -> ``ans``
  - ``loc``      — a LOCALITY control: a question about a DIFFERENT, superficially similar entity
                   (should be UNAFFECTED by this edit)                -> ``loc_ans``
  - ``mhop``     — a multi-hop question chaining through the object (present on only ~2-3% of rows;
                   skipped when absent)                                -> ``mhop_ans``

NOTE on ``loc``: its answer is NOT derivable from this row's own fact — it is a non-interference /
parametric-retention control (does writing THIS fact corrupt the frozen backbone's unrelated
knowledge of a nearby entity), not a recall probe. It is included because the source dataset ships it
as an exact-match probe with its own field, per the probe-type enumeration this source is built
against; a Task/objective that wants a pure recall signal can filter on ``meta["probe"] == "loc"``.

``sample()`` re-rolls the probe type PER DRAW (uniform over whichever of the 5 are non-empty for that
row) rather than baking one in at ingest time, so one edit funds many distinct probes over training.
``facts`` = a single declarative rendering of the edit (``"The {relation} of {subject} is
{object}."``) — the T2 forced-forgetting unit: bounded-capacity write, later probed at variable lag.

Timestep ORDER is preserved end to end (``meta["timestep"]`` = 0..7, chronological) so a
Task/Curriculum can place edits from later timesteps and query across the lag — the direct data fit
for the T2 bounded-capacity forced-forgetting gate.

Data/build (BOUNDED, best-effort, never hangs):
  1. Local ``data/wikibigedit/<split>.jsonl`` (normalized rows, one per edit) staged by the ingest
     script.
  2. Else HF-stream a BOUNDED per-timestep sample of ``lukasthede/WikiBigEdit`` (one
     ``data_files=`` read per of the 8 timestep JSON files).
  3. Else (offline, no local cache) a clear "run ingest first" error — no silent hang.
Ingest: ``scripts/data_build/ingest/wikibigedit/download.py``. See DATASETS.md.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

HF_NAME = "lukasthede/WikiBigEdit"

# The 8 timestep files, in chronological order (dates are embedded in the filename).
TIMESTEP_FILES = [
    "wiki_big_edit_20240201_20240220.json",
    "wiki_big_edit_20240220_20240301.json",
    "wiki_big_edit_20240301_20240320.json",
    "wiki_big_edit_20240320_20240401.json",
    "wiki_big_edit_20240401_20240501.json",
    "wiki_big_edit_20240501_20240601.json",
    "wiki_big_edit_20240601_20240620.json",
    "wiki_big_edit_20240620_20240701.json",
]

# (question_field, answer_field) — the 5 pre-authored probes over one edit (see module docstring).
_PROBES = [
    ("update", "ans"),
    ("rephrase", "ans"),
    ("personas", "ans"),
    ("loc", "loc_ans"),
    ("mhop", "mhop_ans"),
]

_CORE = ("subject", "relation", "object", "ans")
_FIELDS = ("subject", "relation", "object", "ans", "rephrase", "loc", "loc_ans", "mhop", "mhop_ans",
           "update", "personas", "tag")


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("wikibigedit", split)


def _clean(v) -> str:
    return (v or "").strip() if isinstance(v, str) else (str(v) if v is not None else "")


def _normalize(ex: dict, timestep: int) -> Optional[dict]:
    """Raw WikiBigEdit row -> normalized dict, or None if a core field is missing/empty."""
    row = {k: _clean(ex.get(k)) for k in _FIELDS}
    if not all(row[k] for k in _CORE):
        return None
    row["timestep"] = timestep
    return row


def _rows_from_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_from_hf(split: str, n_docs: int) -> List[dict]:
    """Stream a bounded, ORDER-preserving sample across the 8 timestep files.

    ``n_docs`` rows are spread evenly over the 8 timesteps; ``val`` reads the NEXT (disjoint) slice
    of each timestep's stream so train/val never share a row.
    """
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            f"[wikibigedit] HF `datasets` unavailable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/wikibigedit/download.py  to stage a local "
            f"data/wikibigedit/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    per_ts = max(1, -(-n_docs // len(TIMESTEP_FILES)))       # ceil
    skip = 0 if split == "train" else per_ts                 # val = the NEXT disjoint slice per timestep
    rows: List[dict] = []
    for ts, fname in enumerate(TIMESTEP_FILES):
        try:
            ds = load_dataset(HF_NAME, data_files=fname, split="train", streaming=True)
        except Exception as e:
            raise RuntimeError(
                f"[wikibigedit] HF dataset {HF_NAME!r}/{fname} unreachable "
                f"({type(e).__name__}: {str(e)[:120]}). Run  python "
                f"scripts/data_build/ingest/wikibigedit/download.py  to stage a local "
                f"data/wikibigedit/{{train,val}}.jsonl sample, then retry (works offline once staged)."
            ) from e
        seen = 0
        for ex in ds:
            if seen < skip:
                seen += 1
                continue
            r = _normalize(ex, ts)
            if r is not None:
                rows.append(r)
            seen += 1
            if seen >= skip + per_ts:
                break
    return rows[:n_docs]


def _probe_options(r: dict) -> List[Tuple[str, str, str]]:
    """(question_text, answer_text, probe_name) for whichever probes are populated on this row."""
    opts = [(r[qf], r[af], qf) for qf, af in _PROBES if r.get(qf) and r.get(af)]
    if not opts:      # pathological — _CORE guarantees at least a synthetic fallback works
        opts.append((f"What is the {r['relation']} of {r['subject']}?", r["ans"], "synthetic"))
    return opts


class WikiBigEditSource(Source):
    """Yields WikiBigEdit fact-edits as QAItems, re-rolling the probe type per draw."""

    kind = "qa"
    pack_n_queries = (1, 3)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed

        local = _local_jsonl(split)
        if local is not None:
            self.rows = _rows_from_jsonl(local)
            origin = f"data/wikibigedit/{local.name}"
        else:
            self.rows = _rows_from_hf(split, n_docs)
            origin = f"HF:{HF_NAME}"
        if not self.rows:
            raise ValueError(f"[wikibigedit] no rows loaded from {origin} (split={split}).")

        n_ts = len(set(r["timestep"] for r in self.rows))
        n_mhop = sum(1 for r in self.rows if r.get("mhop"))
        print(f"[data.wikibigedit] {split}: {len(self.rows)} edits from {origin} across {n_ts} "
              f"timesteps ({n_mhop} with a multi-hop probe)", flush=True)

        self._pool = [self._fact(r) for r in self.rows]
        if len(self._pool) > pool_cap:
            self._pool = random.Random(seed).sample(self._pool, pool_cap)

    @staticmethod
    def _fact(r: dict) -> str:
        return f"The {r['relation']} of {r['subject']} is {r['object']}."

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            r = self.rows[rng.randrange(len(self.rows))]
            opts = _probe_options(r)
            q, a, probe = opts[rng.randrange(len(opts))]
            out.append(QAItem(
                facts=[self._fact(r)], question=q, answer=a, task_id=0,
                meta={"dataset": "wikibigedit", "timestep": r["timestep"], "tag": r.get("tag", ""),
                      "probe": probe}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
