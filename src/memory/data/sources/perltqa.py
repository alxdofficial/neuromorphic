"""PerLTQA source — personalization long-term memory QA (profile/relationship/event/dialogue).

Source half of the ``qa`` task for PERSONALIZATION long-term memory: PerLTQA (GitHub
``Elvin-Yiming-Du/PerLTQA``, ``en_v2``, CC-BY-NC-4.0) is 32 synthetic characters, each with a
``profile`` (13 scalar fields), a ``social_relationship`` list, ``events`` (paragraph-length episodic
memories) and ``dialogues`` (multi-turn conversations tied to an event) — plus ~8.3k pre-authored QA
pairs cross-referencing that content by an explicit ``Reference Memory`` id, each with "Memory
Anchors" (the load-bearing key phrase(s) inside the answer / reference memory — kept in
``meta["anchors"]`` for a Task/objective that wants containment scoring instead of strict EM).

No HF mirror exists; the two source-of-truth JSONs are fetched straight from GitHub raw:
  - QA:     ``Dataset/en_v2/perltqa_en_v2.json``   (question, answer, reference id, memory anchors)
  - Memory: ``Dataset/en_v2/perltmem_en_v2.json``  (the actual profile/relationship/event/dialogue
            CONTENT the QA's ``Reference Memory`` id points into — this source cross-references the
            two to build ``facts``).
(``en_v2`` = the Dec-2025 "fixed inconsistency issues" revision; preferred over the original ``en``.)

``facts`` (the compressible memory unit) depend on the QA's section:
  - ``profile``            -> ALL ~13 profile-field sentences for that character (one shared
                              "profile story"; the question probes ONE field).
  - ``social_relationship``-> the one named relationship + its description (2 sentences).
  - ``events``              -> the event paragraph, sentence-split (the richest, longest fact list).
  - ``dialogues``            -> the raw dialogue turns, flattened in order (1 line/turn).

No official train/val split exists (PerLTQA ships as one fixed research JSON) — the ingest script
partitions by CHARACTER (not row) so no persona straddles both splits: leaking a character's profile
into train would make val answerable from memorization rather than long-term recall of THIS episode's
compressed context, which would defeat the personalization framing.

Data/build (BOUNDED, best-effort, never hangs):
  1. Local ``data/perltqa/<split>.jsonl`` (normalized rows) staged by the ingest script.
  2. Else a bounded live GitHub fetch + the same character-disjoint normalize (slower, no cache).
  3. Else (offline, no local cache) a clear "run ingest first" error — no silent hang.
Ingest: ``scripts/data_build/ingest/perltqa/download.py``. See DATASETS.md.

LICENSE NOTE: PerLTQA is CC-BY-NC-4.0 (non-commercial) — unlike this repo's other bounded-ingest
sources (Apache/CC-BY), mind this when the data leaves research use.
"""
from __future__ import annotations

import ast
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

_BASE = "https://raw.githubusercontent.com/Elvin-Yiming-Du/PerLTQA/main/Dataset/en_v2"
QA_URL = f"{_BASE}/perltqa_en_v2.json"
MEM_URL = f"{_BASE}/perltmem_en_v2.json"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("perltqa", split)


def _split_sents(text: str) -> List[str]:
    flat = (text or "").replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _ref_id(raw) -> str:
    """``"['4_0_0']"`` -> ``"4_0_0"`` (profile refs are already a bare field name -> passed through)."""
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list) and parsed:
            return str(parsed[0])
        return str(parsed)
    except Exception:
        return str(raw)


def _as_dict(v) -> dict:
    """A HANDFUL of perltmem_en_v2 characters store ``social_relationship`` as a Python ``repr()``
    string (single-quoted dict literal) instead of a real JSON object — recover it via
    ``ast.literal_eval`` (safe: it's a literal, not arbitrary code)."""
    if isinstance(v, dict):
        return v
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return {}


def _profile_facts(m_char: dict) -> List[str]:
    p = m_char.get("profile") or {}
    protagonist = p.get("Protagonist", "")
    return [f"{protagonist}'s {k} is {v}." for k, v in p.items()
            if k != "Protagonist" and str(v).strip()]


def _relationship_facts(m_char: dict, rid: str) -> Optional[List[str]]:
    rel = _as_dict(m_char.get("social_relationship")).get(rid)
    if rel is None:
        return None
    protagonist = (m_char.get("profile") or {}).get("Protagonist", "")
    who = rel.get("Supporting Characters", "")
    relation = rel.get("Relationship", "")
    desc = (rel.get("Description") or "").strip()
    facts = [f"{who} is {protagonist}'s {relation}."]
    if desc:
        facts.append(desc if desc.endswith((".", "!", "?")) else desc + ".")
    return facts


def _event_facts(m_char: dict, eid: str) -> Optional[List[str]]:
    ev = (m_char.get("events") or {}).get(eid)
    content = (ev or {}).get("content", "").strip()
    return _split_sents(content) if content else None


def _dialogue_facts(m_char: dict, did: str) -> Optional[List[str]]:
    dl = (m_char.get("dialogues") or {}).get(did)
    if dl is None:
        return None
    contents = dl.get("contents") or {}
    lines: List[str] = []
    for ts in sorted(contents.keys()):
        lines.extend(contents[ts])
    return lines or None


_SECTION_BUILDERS = {
    "social_relationship": _relationship_facts,
    "events": _event_facts,
    "dialogues": _dialogue_facts,
}


def _anchors_of(r: dict) -> List[str]:
    return [k for anchor in (r.get("Memory Anchors") or []) for k in anchor.keys()]


def _rows_for_character(name: str, qa_sections: dict, m_char: dict) -> List[dict]:
    """One character's QA (all 4 sections) -> normalized rows with facts already resolved."""
    rows: List[dict] = []
    profile_facts = _profile_facts(m_char)
    for r in qa_sections.get("profile") or []:
        q, a = (r.get("Question") or "").strip(), (r.get("Answer") or "").strip()
        if not q or not a or not profile_facts:
            continue
        rows.append({"facts": profile_facts, "question": q, "answer": a, "character": name,
                     "section": "profile", "reference_memory": r.get("Reference Memory", ""),
                     "anchors": _anchors_of(r)})
    for section, builder in _SECTION_BUILDERS.items():
        for entry in qa_sections.get(section) or []:
            for _outer_id, recs in entry.items():
                for r in recs:
                    q, a = (r.get("Question") or "").strip(), (r.get("Answer") or "").strip()
                    if not q or not a:
                        continue
                    rid = _ref_id(r.get("Reference Memory", ""))
                    facts = builder(m_char, rid)
                    if not facts:
                        continue
                    rows.append({"facts": facts, "question": q, "answer": a, "character": name,
                                 "section": section, "reference_memory": rid,
                                 "anchors": _anchors_of(r)})
    return rows


def _character_split(names: List[str], val_frac: float = 0.2) -> Tuple[List[str], List[str]]:
    """Deterministic CHARACTER-disjoint split (alphabetical head = val) — no persona straddles both."""
    names_sorted = sorted(names)
    n_val = max(1, round(len(names_sorted) * val_frac))
    val = set(names_sorted[:n_val])
    train = [n for n in names_sorted if n not in val]
    return train, sorted(val)


def _rows_from_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _rows_from_github(split: str, n_docs: int, seed: int) -> List[dict]:
    """Bounded live fallback: fetch both en_v2 JSONs, character-split, normalize, seeded-sample."""
    try:
        with urllib.request.urlopen(QA_URL, timeout=60) as fp:
            qa = json.loads(fp.read().decode("utf-8"))
        with urllib.request.urlopen(MEM_URL, timeout=60) as fp:
            mem = json.loads(fp.read().decode("utf-8"))
    except Exception as e:
        raise RuntimeError(
            f"[perltqa] GitHub fetch unreachable ({type(e).__name__}: {str(e)[:120]}). Run  python "
            f"scripts/data_build/ingest/perltqa/download.py  to stage a local "
            f"data/perltqa/{{train,val}}.jsonl sample, then retry (works offline once staged)."
        ) from e
    by_name = {name: sections for wrapper in qa for name, sections in wrapper.items()}
    usable = [n for n in by_name if n in mem]
    train_names, val_names = _character_split(usable)
    names = train_names if split == "train" else val_names
    rows: List[dict] = []
    for name in names:
        rows.extend(_rows_for_character(name, by_name[name], mem[name]))
    random.Random(seed).shuffle(rows)
    return rows[:n_docs]


class PerLTQASource(Source):
    """Yields PerLTQA profile/relationship/event/dialogue QA as ``QAItem``s (character-disjoint)."""

    kind = "qa"
    pack_n_queries = (1, 2)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.split = split
        self.seed = seed

        local = _local_jsonl(split)
        if local is not None:
            self.rows = _rows_from_jsonl(local)
            origin = f"data/perltqa/{local.name}"
        else:
            self.rows = _rows_from_github(split, n_docs, seed)
            origin = "GitHub:Elvin-Yiming-Du/PerLTQA (en_v2)"
        if not self.rows:
            raise ValueError(f"[perltqa] no rows loaded from {origin} (split={split}).")

        n_chars = len(set(r["character"] for r in self.rows))
        by_section: Dict[str, int] = {}
        for r in self.rows:
            by_section[r["section"]] = by_section.get(r["section"], 0) + 1
        print(f"[data.perltqa] {split}: {len(self.rows)} QA rows from {origin} across {n_chars} "
              f"characters {dict(by_section)}", flush=True)

        pool = [f for r in self.rows for f in r["facts"] if len(f) >= 20]
        if len(pool) > pool_cap:
            pool = random.Random(seed).sample(pool, pool_cap)
        self._pool = pool

    def sample(self, rng, n: int) -> list:
        out = []
        for _ in range(n):
            r = self.rows[rng.randrange(len(self.rows))]
            out.append(QAItem(
                facts=list(r["facts"]), question=r["question"], answer=r["answer"], task_id=0,
                meta={"dataset": "perltqa", "character": r["character"], "section": r["section"],
                      "reference_memory": r["reference_memory"], "anchors": r.get("anchors", [])}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
