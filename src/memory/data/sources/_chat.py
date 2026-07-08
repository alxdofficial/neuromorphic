"""Shared multi-turn-conversation helpers for the real long-conversation QA sources (``wildchat``,
``lmsys_chat``). Both HF datasets share an (almost) identical schema — a ``conversation`` list of
``{"role": "user"|"assistant", "content": str}`` turns plus a ``turn`` count — so this factors the
turn-rendering / token-capping / recall-target-selection logic that differs only in HF id / gating.

``msc.py`` (Multi-Session Chat) has a different, session-grouped schema and does its own line
rendering, but reuses ``flatten``/``cap_by_tokens`` from here.

Design (the "memory-relevant" probe, per project_data_plan): render the WHOLE conversation as facts
(one line per turn, "User: ..."/"Assistant: ..."), then ask about a SPECIFIC EARLIER user turn by its
ordinal position ("what was the user's message #k?"). The answer is always the exact (flattened,
possibly length-capped) text that was actually packed as a fact — never a paraphrase — so it is
genuinely recoverable, not guessable, and grading is plain EM/containment (no free-form judge).
"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

_WS = re.compile(r"\s+")


def flatten(text: str, max_chars: int = 320) -> str:
    """Collapse newlines/whitespace to single spaces and hard-cap length (one turn == one fact-line;
    a multi-paragraph pasted answer must not blow the per-turn token budget on its own)."""
    flat = _WS.sub(" ", (text or "").strip())
    if len(flat) > max_chars:
        flat = flat[:max_chars].rstrip() + "…"
    return flat


def render_turns(turns: List[dict], max_turn_chars: int = 320) -> List[Tuple[str, str, str]]:
    """``[{"role","content"}, ...]`` -> ``[(line, role, content), ...]``.

    ``line`` = the rendered ``"User: ..."``/``"Assistant: ..."`` fact string; ``content`` = the
    flattened text used for BOTH that line and any answer drawn from it, so an answer is always
    verbatim-recoverable from what actually got packed. Drops empty/system/tool turns.
    """
    out: List[Tuple[str, str, str]] = []
    for t in turns:
        role = (t.get("role") or "").lower()
        if role not in ("user", "assistant"):
            continue
        content = flatten(t.get("content") or "", max_turn_chars)
        if not content:
            continue
        speaker = "User" if role == "user" else "Assistant"
        out.append((f"{speaker}: {content}", role, content))
    return out


def cap_by_tokens(lines: List[str], tok, max_tokens: int) -> List[str]:
    """Keep whole lines from the start until the token budget is hit (mirrors ``multiwoz._cap``)."""
    kept: List[str] = []
    total = 0
    for ln in lines:
        n = len(tok(ln + "\n", add_special_tokens=False).input_ids)
        if kept and total + n > max_tokens:
            break
        kept.append(ln)
        total += n
    return kept


def pick_recall_target(rendered: List[Tuple[str, str, str]], rng, *, min_chars: int = 15,
                        max_chars: int = 250, early_frac: float = 0.7) -> Optional[Tuple[int, str]]:
    """Pick a USER turn to recall, from the earliest ``early_frac`` of (kept) user turns — returns
    ``(1-indexed position among ALL user turns, content)`` or ``None`` if nothing qualifies. Length
    bounds keep the EM/containment target sane (not a trivial "ok", not a giant paste)."""
    user_turns = [content for (_line, role, content) in rendered if role == "user"]
    if not user_turns:
        return None
    cutoff = max(1, int(len(user_turns) * early_frac))
    candidates = [(k, c) for k, c in enumerate(user_turns[:cutoff], start=1)
                  if min_chars <= len(c) <= max_chars]
    if not candidates:                                    # relax: allow anywhere if the early slice is thin
        candidates = [(k, c) for k, c in enumerate(user_turns, start=1) if min_chars <= len(c) <= max_chars]
    if not candidates:
        return None
    return candidates[rng.randrange(len(candidates))]
