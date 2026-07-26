"""Torch-free text loaders used by lightweight A-MEM orchestration workers.

LongMemEval's canonical module imports torch for its separate tensor-dataset class.  API-only A-MEM workers
need only the raw text accessor, so this module mirrors that accessor without importing torch.  Equivalence is
covered against the canonical loader in ``tests/test_agentmem_runner.py``.
"""
from __future__ import annotations

import ctypes
import gc
import json
from typing import Optional

_REPO_CLEANED = "xiaowu0162/longmemeval-cleaned"
_REPO_ORIGINAL = "xiaowu0162/longmemeval"
_FILES_CLEANED = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}
_FILES_ORIGINAL = {"s": "longmemeval_s", "m": "longmemeval_m", "oracle": "longmemeval_oracle"}


def _load_raw(variant: str) -> list:
    from huggingface_hub import hf_hub_download

    if variant not in _FILES_CLEANED:
        raise ValueError(f"LongMemEval variant must be s/m/oracle, got {variant!r}")
    errors = []
    for repo, files in ((_REPO_CLEANED, _FILES_CLEANED), (_REPO_ORIGINAL, _FILES_ORIGINAL)):
        fname = files[variant]
        try:
            print(f"[amem-data] LongMemEval: fetching {repo}/{fname} ...")
            path = hf_hub_download(repo, filename=fname, repo_type="dataset")
            with open(path) as handle:
                return json.load(handle)
        except Exception as exc:  # try the same maintained→deprecated fallback as the canonical loader
            errors.append(f"{repo}/{fname}: {type(exc).__name__}: {exc}")
    raise RuntimeError("LongMemEval: could not load variant " + repr(variant) + "\n  " + "\n  ".join(errors))


def _question_type(example: dict) -> str:
    qid = str(example.get("question_id", ""))
    return "abstention" if qid.endswith("_abs") else (example.get("question_type") or "unknown")


def _stratified_sample(raw: list, max_examples: Optional[int]) -> list:
    if max_examples is None or max_examples >= len(raw):
        return raw
    groups: dict[str, list] = {}
    for example in raw:
        groups.setdefault(_question_type(example), []).append(example)
    out, depth = [], 0
    while len(out) < max_examples:
        progressed = False
        for group in groups.values():
            if depth < len(group):
                out.append(group[depth])
                progressed = True
                if len(out) >= max_examples:
                    break
        if not progressed:
            break
        depth += 1
    return out


def _render_sessions(example: dict) -> list[str]:
    sessions = example.get("haystack_sessions") or []
    dates = example.get("haystack_dates") or []
    session_ids = example.get("haystack_session_ids") or []
    rendered = []
    for index, turns in enumerate(sessions):
        sid = session_ids[index] if index < len(session_ids) else index
        date = dates[index] if index < len(dates) else ""
        lines = [f"[Session {sid} — {date}]"]
        for turn in turns or []:
            content = (turn.get("content") or "").strip()
            if content:
                speaker = "User" if turn.get("role", "user") == "user" else "Assistant"
                lines.append(f"{speaker}: {content}")
        rendered.append("\n".join(lines))
    return rendered


def load_longmemeval_text(variant: str = "s", max_examples: Optional[int] = None) -> list[dict]:
    raw = _stratified_sample(_load_raw(variant), max_examples)
    items = []
    for example in raw:
        sessions = _render_sessions(example)
        items.append({
            "question_id": str(example.get("question_id", "")),
            "question": str(example.get("question", "")),
            "answer": str(example.get("answer", "")),
            "question_date": str(example.get("question_date", "")),
            "question_type": _question_type(example),
            "full_history": "\n".join(sessions),
            "sessions": sessions,
        })
    return items


def load_items(dataset: str, variant: str = "s", max_examples: Optional[int] = None) -> list[dict]:
    if dataset == "longmemeval":
        return load_longmemeval_text(variant=variant, max_examples=max_examples)
    if dataset == "memoryagentbench":
        from src.memory.data.memoryagentbench import load_memoryagentbench_text
        return load_memoryagentbench_text(variant=variant, max_examples=max_examples)
    raise ValueError(f"unknown dataset {dataset!r}")


def trim_process_heap() -> bool:
    """Return freed temporary JSON/rendering arenas to Linux after selecting one worker's shard."""
    gc.collect()
    try:
        return bool(ctypes.CDLL("libc.so.6").malloc_trim(0))
    except (AttributeError, OSError):
        return False
