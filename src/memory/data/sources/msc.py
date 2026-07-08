"""MSC (Multi-Session Chat) source — ordered multi-session dialogue -> an early-session recall probe.

Source half of the ``qa`` task for the genuinely MULTI-SESSION regime (as opposed to wildchat/lmsys's
single long session): HF ``nayohan/multi_session_chat`` ships each conversation as several session ROWS
sharing a ``dialoug_id`` [sic — the field is spelled that way upstream], ordered by ``session_id``
(0, 1, 2, ...), each carrying that session's ``persona1``/``persona2`` (declarative fact lists about
the two speakers) and its ``dialogue``/``speaker`` turn lists.

The persona sentences are NOT stated verbatim anywhere in the live dialogue (the dialogue only
paraphrases them), so a plain "recall this dialogue line" probe would either be unanswerable or would
silently degrade to a paraphrase-matching task. Instead we render session 0's persona facts as their own
declarative fact-lines UP FRONT (guaranteed inside the packed context, guaranteed first — so cheapest to
keep under any token cap), then append every session's dialogue turns in order after them. The probe
asks to recall one of those EARLY (session-0) persona facts — "what did Speaker 1 mention about
themselves earlier" — after the model has also had to hold several sessions' worth of live dialogue,
which is exactly the memory-relevant "recall a persona/fact from an EARLY session" probe this line
needs (project_data_plan). Answer is verbatim, so grading is plain EM/containment.

Resolution order (BEST-EFFORT, never hangs):
  1. Local ``data/msc/<split>.jsonl`` (``{"sessions": [{"session_id","persona1","persona2","speaker",
     "dialogue"}, ...]}`` per line, sessions pre-sorted) — ingest cache.
  2. Else HF-stream a BOUNDED sample of ``nayohan/multi_session_chat`` (real train/validation splits).
  3. Else (offline) raise a clear "run ingest first" error — no silent hang.

Ingest: ``scripts/data_build/ingest/msc/download.py``. See DATASETS.md / docs/data_arch_plan.md (L1).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import Source, QAItem
from ._corpus import local_jsonl
from ._chat import flatten, cap_by_tokens

HF_NAME = "nayohan/multi_session_chat"


def _local_jsonl(split: str) -> Optional[Path]:
    return local_jsonl("msc", split)


def _hf_split(split: str) -> str:
    if split == "train":
        return "train"
    if split in ("validation", "val"):
        return "validation"
    raise ValueError(f"msc: unrecognized split {split!r} (expected 'train'/'validation'/'val')")


def _slim_session(ex: dict) -> dict:
    return {
        "session_id": ex.get("session_id"),
        "persona1": [str(s).strip() for s in (ex.get("persona1") or []) if str(s).strip()],
        "persona2": [str(s).strip() for s in (ex.get("persona2") or []) if str(s).strip()],
        "speaker": [str(s) for s in (ex.get("speaker") or [])],
        "dialogue": [str(u).strip() for u in (ex.get("dialogue") or [])],
    }


def _group_conversations(rows_iter, max_conv: int):
    """Group session ROWS into conversations by ``dialoug_id``. The live HF stream arrives already
    grouped/session-ordered per conversation (verified empirically); we finalize a group as soon as
    the id changes rather than requiring that assumption to hold globally — if it ever doesn't, worst
    case is a same-id conversation gets split into two (still handled fine downstream, just fewer
    sessions kept), never a crash."""
    cur_id = None
    cur: List[dict] = []
    n_yielded = 0
    for ex in rows_iter:
        did = ex.get("dialoug_id")
        if cur_id is not None and did != cur_id:
            yield cur_id, cur
            n_yielded += 1
            cur = []
            if n_yielded >= max_conv:
                return
        cur_id = did
        cur.append(ex)
    if cur and n_yielded < max_conv:
        yield cur_id, cur


def _iter_hf_conversations(split: str, n_conv: int, min_sessions: int) -> List[List[dict]]:
    from datasets import load_dataset
    try:
        ds = load_dataset(HF_NAME, split=_hf_split(split), streaming=True)
    except Exception as e:
        raise RuntimeError(
            f"[msc] HF dataset {HF_NAME!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/msc/download.py  to stage a local "
            f"data/msc/{{train,val}}.jsonl sample, then retry (works fully offline once staged)."
        ) from e
    out: List[List[dict]] = []
    scan_cap = max(8 * n_conv, 500)                              # generous — ~35% of groups are single-session
    for _did, rows in _group_conversations(ds, max_conv=scan_cap):
        sessions = sorted((_slim_session(r) for r in rows), key=lambda s: s["session_id"])
        if len(sessions) < min_sessions:
            continue
        out.append(sessions)
        if len(out) >= n_conv:
            break
    return out


def _load_conversations(split: str, n_docs: int, min_sessions: int) -> Tuple[List[List[dict]], str]:
    local = _local_jsonl(split)
    if local is not None:
        convs: List[List[dict]] = []
        with open(local) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                sessions = o.get("sessions") or []
                if len(sessions) < min_sessions:
                    continue
                convs.append(sessions)
                if len(convs) >= n_docs:
                    break
        origin = f"data/msc/{local.name}"
        if not convs:
            raise ValueError(f"[msc] {origin} has no conversation with >= {min_sessions} sessions "
                              f"(re-run the ingest script with a lower --min-sessions, or raise n_docs).")
    else:
        convs = _iter_hf_conversations(split, n_docs, min_sessions)
        origin = f"HF:{HF_NAME}"
        if not convs:
            raise ValueError(f"[msc] 0 multi-session conversations matched from {origin} "
                              f"(min_sessions={min_sessions}, split={split!r}).")
    print(f"[data.msc] {split}: {len(convs)} multi-session conversations from {origin} "
          f"(min_sessions={min_sessions})", flush=True)
    return convs, origin


def _render_conversation(sessions: List[dict], max_turn_chars: int) -> Tuple[List[str], List[Tuple[str, str]]]:
    """sessions (sorted by session_id) -> (lines, persona_facts).

    ``persona_facts`` = session-0's persona1+persona2 sentences as ``(speaker_label, sentence)`` — the
    QA answer pool, all EARLY (session 0) by construction. ``lines`` puts those persona facts FIRST
    (as their own declarative fact-lines, so they always survive any token cap), then every session's
    dialogue turns in order, each session boundary marked so the story stays legible.
    """
    s0 = sessions[0]
    persona_facts: List[Tuple[str, str]] = []
    for key, label in (("persona1", "Speaker 1"), ("persona2", "Speaker 2")):
        for sent in s0.get(key) or []:
            sent = flatten(sent, max_turn_chars)
            if sent:
                persona_facts.append((label, sent))

    lines: List[str] = [f"{label} (persona note): {sent}" for label, sent in persona_facts]
    for sess in sessions:
        lines.append(f"--- Session {sess['session_id']} ---")
        for spk, utt in zip(sess["speaker"], sess["dialogue"]):
            utt = flatten(utt, max_turn_chars)
            if utt:
                lines.append(f"{spk}: {utt}")
    return lines, persona_facts


class MSCSource(Source):
    """Yields MSC multi-session dialogues as QAItems (early-session persona recall, see docstring)."""

    kind = "qa"
    pack_n_queries = (1, 2)

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0, n_docs: int = 3000,
                 min_sessions: int = 2, max_dialogue_tokens: int = 900, max_turn_chars: int = 320,
                 pool_cap: int = 2000, **kw):
        self.tok = tokenizer
        self.seed = seed
        self.max_dialogue_tokens = max_dialogue_tokens
        self.max_turn_chars = max_turn_chars
        self.conversations, _origin = _load_conversations(split, n_docs, min_sessions)

        pool: List[str] = []
        for sessions in self.conversations:
            lines, _pf = _render_conversation(sessions, max_turn_chars)
            pool.extend(lines)
        if len(pool) > pool_cap:
            pool = random.Random(seed).sample(pool, pool_cap)
        self._pool = pool

    def sample(self, rng, n: int) -> list:
        out: List[QAItem] = []
        tries = 0
        while len(out) < n and tries < n * 8:
            tries += 1
            sessions = self.conversations[rng.randrange(len(self.conversations))]
            lines, persona_facts = _render_conversation(sessions, self.max_turn_chars)
            capped = cap_by_tokens(lines, self.tok, self.max_dialogue_tokens)
            # persona-fact lines are emitted FIRST, so however many survived the cap is exactly the
            # first len(persona_facts) capped lines that are persona notes.
            n_kept = sum(1 for ln in capped[:len(persona_facts)] if "(persona note): " in ln)
            if n_kept == 0:
                continue
            label, sent = persona_facts[rng.randrange(n_kept)]
            q = f"What is something {label} mentioned about themselves earlier in this conversation?"
            out.append(QAItem(facts=capped, question=q, answer=sent, task_id=0,
                              meta={"dataset": "msc"}))
        return out

    def distractor_pool(self) -> list:
        return self._pool
