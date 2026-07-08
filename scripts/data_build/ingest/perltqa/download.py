#!/usr/bin/env python
"""Ingest a BOUNDED, character-disjoint PerLTQA sample → ``data/perltqa/{train,val}.jsonl``.

Fetches the two ``en_v2`` PerLTQA JSONs straight from GitHub raw (no HF mirror exists):
  - QA:     Dataset/en_v2/perltqa_en_v2.json   (question, answer, reference id, memory anchors)
  - Memory: Dataset/en_v2/perltmem_en_v2.json  (the profile/relationship/event/dialogue CONTENT the
            QA's "Reference Memory" id points into)
cross-references them into self-contained rows (facts already resolved, no further lookup needed at
load time), and stages a bounded sample as jsonl so ``PerLTQASource`` loads fully offline.

No official train/val split exists — this partitions by CHARACTER (not row), reserving the
alphabetically-first ``val_frac`` (default 20%) of the 32 characters for val, so no persona straddles
both splits (see ``PerLTQASource``'s docstring for why that matters for personalization data).

Usage:
    python scripts/data_build/ingest/perltqa/download.py [--n-train 3000] [--n-val 500]

If GitHub is unreachable this exits with a clear error (no partial/half file); rerun once online, or
manually download the two URLs printed in the error and place them under ``data/perltqa/_raw/``.
"""
from __future__ import annotations

import argparse
import ast
import json
import random
import re
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

REPO = Path(__file__).resolve().parents[4]
_BASE = "https://raw.githubusercontent.com/Elvin-Yiming-Du/PerLTQA/main/Dataset/en_v2"
QA_URL = f"{_BASE}/perltqa_en_v2.json"
MEM_URL = f"{_BASE}/perltmem_en_v2.json"

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sents(text: str) -> List[str]:
    flat = (text or "").replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _fetch_json(url: str, timeout: int = 60):
    with urllib.request.urlopen(url, timeout=timeout) as fp:
        return json.loads(fp.read().decode("utf-8"))


def _ref_id(raw) -> str:
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


def _character_split(names: List[str], val_frac: float) -> Tuple[List[str], List[str]]:
    names_sorted = sorted(names)
    n_val = max(1, round(len(names_sorted) * val_frac))
    val = set(names_sorted[:n_val])
    train = [n for n in names_sorted if n not in val]
    return train, sorted(val)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--val-frac", type=float, default=0.2,
                    help="fraction of CHARACTERS (not rows) held out for val")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    try:
        qa = _fetch_json(QA_URL)
        mem = _fetch_json(MEM_URL)
    except Exception as e:
        raise SystemExit(
            f"[perltqa] GitHub fetch failed ({type(e).__name__}: {str(e)[:160]}). Restore network "
            f"access and rerun, or manually download\n  {QA_URL}\n  {MEM_URL}\ninto "
            f"data/perltqa/_raw/ and adapt this script to read them locally.")

    all_names = [list(w.keys())[0] for w in qa]
    usable = [n for n in all_names if n in mem]
    dropped = sorted(set(all_names) - set(usable))
    if dropped:
        print(f"[perltqa] dropping {len(dropped)} character(s) absent from perltmem: {dropped}")

    train_names, val_names = _character_split(usable, args.val_frac)
    print(f"[perltqa] {len(train_names)} train characters / {len(val_names)} val characters "
          f"(character-disjoint split)")

    by_name = {name: sections for wrapper in qa for name, sections in wrapper.items()}

    def _pool(names: List[str]) -> List[dict]:
        rows: List[dict] = []
        for name in names:
            rows.extend(_rows_for_character(name, by_name[name], mem[name]))
        return rows

    train_pool, val_pool = _pool(train_names), _pool(val_names)
    rng = random.Random(args.seed)
    rng.shuffle(train_pool)
    rng.shuffle(val_pool)
    train = train_pool[:args.n_train]
    val = val_pool[:args.n_val]

    out_dir = REPO / "data" / "perltqa"
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, rows, pool in (("train", train, train_pool), ("val", val, val_pool)):
        path = out_dir / f"{split}.jsonl"
        with open(path, "w") as fp:
            for r in rows:
                fp.write(json.dumps(r) + "\n")
        by_section: dict = {}
        for r in rows:
            by_section[r["section"]] = by_section.get(r["section"], 0) + 1
        print(f"[perltqa] wrote {len(rows)}/{len(pool)}-available rows {by_section} → {path}")


if __name__ == "__main__":
    main()
