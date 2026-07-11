"""MultiWOZ source — task-oriented dialogue → slot-recall QAItems (un-guessable exact recall).

Source half for the ``qa`` task: loads MultiWOZ 2.2 dialogues and turns each into a slot-recall
question. From a dialogue's final belief state we pick a (domain, slot) with a concrete value that the
user stated (e.g. ``restaurant-food = italian``), render the turns as ``User: ... / Assistant: ...``
lines (the context/facts), and ask ``"What <slot> did the user request for the <domain>?"`` with the
answer being the exact slot VALUE — stated verbatim earlier in the dialogue, so it is un-guessable
exact recall (like MS-TOD), not a gist.

Data path (IMPORTANT): HF ``multi_woz_v22`` ships as a loading *script* (``multi_woz_v22.py``), which
``datasets>=3`` refuses to run ("Dataset scripts are no longer supported"). So we bypass HF and read
the raw dialogue JSON the script itself points at — the MultiWOZ 2.2 files on GitHub
(``budzianowski/multiwoz .../MultiWOZ_2.2/{train,dev}/dialogues_NNN.json``). Bounded local-cache-first:
``data/multiwoz/<split>.jsonl`` (slim records written by the ingest script) is used if present, else a
bounded GitHub fetch of ``n_docs`` dialogues. Ingest:
``scripts/data_build/ingest/multiwoz/download.py``. See DATASETS.md / docs/history/docs/history/data_arch_plan.md (Layer L1).

Answer-in-context is guaranteed by construction: at load time we keep only (slot, value) candidates
whose value appears VERBATIM (case-insensitive) in the (length-capped) dialogue text, so the answer is
always recoverable from the emitted context. MultiWOZ normalizes some values (``center`` -> ``centre``,
times, ``dontcare``); the verbatim filter drops exactly those non-recoverable slots.
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl

# Raw MultiWOZ 2.2 dialogue JSON (what the dead HF script downloads). train = 17 shards, dev = 2.
_RAW_BASE = "https://github.com/budzianowski/multiwoz/raw/master/data/MultiWOZ_2.2"
_SPLIT_SHARDS = {"train": range(1, 18), "dev": range(1, 3)}

# Slot values that are not concrete verbatim recall targets.
_SKIP_VALUES = {"", "dontcare", "none", "not mentioned", "?", "yes", "no"}

# domain-slot -> readable slot phrase for the question (fallback: the raw slot name).
_SLOT_PHRASE = {
    "pricerange": "price range", "leaveat": "departure time", "arriveby": "arrival time",
    "bookday": "booking day", "bookpeople": "party size", "booktime": "booking time",
    "bookstay": "number of nights", "departure": "departure location", "destination": "destination",
    "food": "food type", "area": "area", "name": "name", "type": "type", "stars": "star rating",
    "day": "day", "parking": "parking", "internet": "internet", "department": "department",
}


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("multiwoz", split)          # ingest writes train.jsonl / val.jsonl


def _shard_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("validation", "val", "dev"):
        return "dev"
    raise ValueError(f"multiwoz: unrecognized split {split!r} (expected 'train'/'validation'/'val')")


def _parse_dialogue(raw: dict) -> Optional[dict]:
    """Raw MultiWOZ 2.2 dialogue -> slim record ``{"dialogue_id","lines","state"}``.

    ``lines`` = list of ``"User: ..."`` / ``"Assistant: ..."`` strings (turn order).
    ``state`` = ``{"domain-slot": value}`` accumulated over the USER-turn belief states (last value
    wins; the cumulative state lives in the final user turn). No tokenizer / capping here — that is
    the source's job (so the ingest script stays tokenizer-free). Returns None if unusable.
    """
    turns = raw.get("turns") or []
    lines: List[str] = []
    state: Dict[str, str] = {}
    for t in turns:
        spk = (t.get("speaker") or "").upper()
        utt = (t.get("utterance") or "").strip()
        if not utt:
            continue
        lines.append(("User: " if spk == "USER" else "Assistant: ") + utt)
        for fr in t.get("frames") or []:
            sv = (fr.get("state") or {}).get("slot_values") or {}
            for slot, vals in sv.items():
                if not vals:
                    continue
                v = str(vals[0]).strip()          # first listed value (canonical); alternatives ignored
                if v.lower() in _SKIP_VALUES:
                    continue
                state[slot] = v
    if not lines or not state:
        return None
    return {"dialogue_id": raw.get("dialogue_id", ""), "lines": lines, "state": state}


def _fetch_shard(split: str, idx: int, timeout: int = 60) -> list:
    url = f"{_RAW_BASE}/{split}/dialogues_{idx:03d}.json"
    with urllib.request.urlopen(url, timeout=timeout) as fp:
        return json.load(fp)


def iter_slim_records(split: str, n_docs: int):
    """Yield up to ``n_docs`` slim dialogue records. Local jsonl first (fully offline), else a bounded
    GitHub fetch of the raw dialogue shards. Raises a clear ingest-first error if GitHub is unreachable."""
    local = _local_jsonl(split)
    if local is not None:
        with open(local) as fp:
            count = 0
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
                count += 1
                if count >= n_docs:
                    break
        return

    shard_split = _shard_split(split)
    count = 0
    for idx in _SPLIT_SHARDS[shard_split]:
        try:
            dialogues = _fetch_shard(shard_split, idx)
        except Exception as e:  # network / offline / moved
            raise RuntimeError(
                f"[multiwoz] raw MultiWOZ 2.2 shard {shard_split}/dialogues_{idx:03d}.json unreachable "
                f"({type(e).__name__}: {str(e)[:100]}). Run  python "
                f"scripts/data_build/ingest/multiwoz/download.py  to stage a local "
                f"data/multiwoz/{{train,val}}.jsonl sample, then retry (fully offline once staged).") from e
        for raw in dialogues:
            rec = _parse_dialogue(raw)
            if rec is None:
                continue
            yield rec
            count += 1
            if count >= n_docs:
                return


def _readable(slot_full: str) -> Tuple[str, str]:
    """"domain-slot" -> (domain, readable-slot-phrase). Handles multi-hyphen slots defensively."""
    domain, _, slot = slot_full.partition("-")
    return domain, _SLOT_PHRASE.get(slot, slot.replace("_", " "))


class MultiWOZSource(Source):
    """Yields MultiWOZ dialogues as slot-recall ``QAItem``s (answer = the verbatim slot value)."""

    kind = "qa"

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 max_dialogue_tokens: int = 900, **kw):
        self.tok = tokenizer
        self.seed = seed
        self.max_dialogue_tokens = max_dialogue_tokens
        origin = ("data/multiwoz/%s.jsonl" % ("train" if split == "train" else "val")
                  if _local_jsonl(split) is not None
                  else f"GitHub:MultiWOZ_2.2/{_shard_split(split)}")

        # Each kept item: (lines[list[str]], candidates[list[(slot_full, value)]]). We cap the dialogue
        # to <= max_dialogue_tokens (whole turns from the start) so the qa Task never tail-truncates
        # away the answer, then keep only (slot, value) whose value appears verbatim in the KEPT text.
        self.items: List[Tuple[List[str], List[Tuple[str, str]]]] = []
        self._pool: List[str] = []
        no_cand = 0
        for rec in iter_slim_records(split, n_docs):
            lines = self._cap(rec["lines"])
            text = "\n".join(lines).lower()
            cands = [(slot, v) for slot, v in rec["state"].items() if v.lower() in text]
            if not cands:
                no_cand += 1
                continue
            self.items.append((lines, cands))
            self._pool.extend(lines)
        if not self.items:
            raise ValueError(f"[multiwoz] no dialogues with a verbatim-recoverable slot for "
                             f"split={split!r} from {origin}")
        print(f"[multiwoz] {split}: {len(self.items)} dialogues from {origin} "
              f"(dropped {no_cand} with no verbatim slot)", flush=True)

    def _cap(self, lines: List[str]) -> List[str]:
        """Keep whole turns from the start until the token budget is hit (most dialogues fit whole)."""
        kept: List[str] = []
        total = 0
        for ln in lines:
            n = len(self.tok(ln + "\n", add_special_tokens=False).input_ids)
            if kept and total + n > self.max_dialogue_tokens:
                break
            kept.append(ln)
            total += n
        return kept

    def sample(self, rng, n: int) -> list:
        out: List[QAItem] = []
        for _ in range(n):
            lines, cands = self.items[rng.randrange(len(self.items))]
            slot_full, value = cands[rng.randrange(len(cands))]
            domain, slot_phrase = _readable(slot_full)
            q = f"What {slot_phrase} did the user request for the {domain}?"
            out.append(QAItem(facts=list(lines), question=q, answer=value, task_id=0,
                              meta={"dataset": "multiwoz", "slot": slot_full, "domain": domain}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
