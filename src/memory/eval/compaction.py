"""Recursive context compaction — the "agentic context management" baseline.

This is what production coding agents do when they run out of room: rather than evicting KV entries or
retrieving on demand, the model is asked to REWRITE the context so far into a bounded natural-language
summary, and then keeps going from that summary. When the working window fills again, it compacts again.

    working cap W (e.g. 20k)   -- the window we PRETEND to have. NOT the model's real limit; it is the
                                  knob that forces compaction to happen repeatedly, so the summary is
                                  recursive rather than a single one-shot summarisation.
    summary budget B (e.g. 2k) -- the MEMORY SIZE. This is what makes the method comparable to the rest of
                                  the panel: B is the same kind of quantity as H2O's KV cap and M+'s pool.

Why this belongs in the panel: it is a frozen-LM, zero-training compressor that produces a bounded token
budget read by the same decoder — structurally the same interface as a learned memory layer, but with the
compression chosen by PROMPTING instead of by training. If a prompted summary at B tokens matches a learned
memory at B tokens, the learned part is not earning its keep.

Deliberately NOT modelled here: real agents also keep file/tool pointers and can RE-READ the source, which
makes their compaction lossy-but-recoverable. That is retrieval, and the panel already covers it with
rag_bm25 / rag_dense; mixing it in would confound what the summarisation itself contributes.

The summary is QUESTION-AGNOSTIC — computed once per distinct context and reused across every question on
it, exactly like M+'s memory pool. That is both the honest memory test (the compactor cannot know what it
will be asked) and what makes the method affordable on MemoryAgentBench, where ~85 questions share a context.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .baselines import _ref_tokenizer, _token_count

# Room inside the working window for the instruction block + the running summary that is fed back in.
# Kept explicit rather than folded into a magic constant so the chunk arithmetic below is auditable.
_INSTRUCTION_RESERVE_TOKENS = 512

COMPACT_SYSTEM = (
    "You are maintaining a compact running memory of a long document or conversation that you can no longer "
    "hold in full. You will be shown your CURRENT MEMORY and the NEXT PART of the material. Rewrite the "
    "memory so it absorbs the new part."
)

COMPACT_INSTRUCTION = (
    "Rewrite your memory to incorporate the new material.\n"
    "Rules:\n"
    "- Your rewritten memory MUST be at most {budget} tokens (roughly {words} words). This is a hard limit.\n"
    "- You will NOT see the original material again. Anything you leave out is lost permanently.\n"
    "- You do NOT know what question you will be asked. Keep whatever a reader would most plausibly need: "
    "specific facts, names, dates, numbers, decisions, preferences, changes/updates, and contradictions.\n"
    "- Prefer concrete specifics over generalities. 'User's dog is a Golden Retriever' is useful; "
    "'user discussed their pet' is not.\n"
    "- When newer material CONTRADICTS or UPDATES something already in memory, keep the newer fact and say "
    "it superseded the older one.\n"
    "- Write dense notes, not prose. No preamble, no commentary about the summarisation itself.\n"
)


def _chunk_by_tokens(text: str, chunk_tokens: int) -> list[str]:
    """Split `text` into pieces of at most `chunk_tokens` reference tokens.

    Falls back to a ~4 chars/token character split when the reference tokenizer is unavailable, so an
    offline/gated environment degrades to approximate chunking rather than crashing the run.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be > 0")
    tok = _ref_tokenizer()
    if tok is None:
        step = chunk_tokens * 4
        return [text[i:i + step] for i in range(0, len(text), step)] or [""]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        return [""]
    return [tok.decode(ids[i:i + chunk_tokens]) for i in range(0, len(ids), chunk_tokens)]


def compaction_plan(context: str, working_cap: int, summary_budget: int) -> tuple[list[str], int]:
    """-> (chunks, chunk_tokens). Pure function, so the cost of a run can be predicted before spending on it.

    Each round feeds back the running summary (<= summary_budget) plus one chunk, and must fit in
    working_cap. So chunk_tokens = working_cap - summary_budget - instruction reserve.
    """
    chunk_tokens = working_cap - summary_budget - _INSTRUCTION_RESERVE_TOKENS
    if chunk_tokens <= 0:
        raise ValueError(
            f"working_cap={working_cap} leaves no room for content after summary_budget={summary_budget} "
            f"+ {_INSTRUCTION_RESERVE_TOKENS} reserve. Raise --compact-cap or lower --compact-budget.")
    return _chunk_by_tokens(context, chunk_tokens), chunk_tokens


def summary_key(context: str, model: str, working_cap: int, summary_budget: int) -> str:
    """Cache key. Includes every knob that changes the summary, so a budget sweep cannot reuse a stale one."""
    h = hashlib.md5(context.encode()).hexdigest()[:16]
    return f"{h}__{model.replace('/', '_')}__W{working_cap}__B{summary_budget}"


class SummaryStore:
    """Disk cache of question-agnostic summaries, so a rerun (or a second dataset pass, or a crash) never
    re-pays for compaction. One JSON file; summaries are small, unlike the contexts they replace."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._d: dict = {}
        if self.path.exists():
            try:
                self._d = json.loads(self.path.read_text())
            except Exception:  # noqa: BLE001 — a corrupt cache must not kill the run; recompute instead
                self._d = {}

    def get(self, key: str):
        return self._d.get(key)

    def put(self, key: str, value: dict) -> None:
        self._d[key] = value
        self.path.write_text(json.dumps(self._d, indent=1))

    def __len__(self) -> int:
        return len(self._d)


async def compact_context(client, model: str, context: str, *, working_cap: int, summary_budget: int,
                          provider: str | None = None, progress_label: str = "") -> dict:
    """Recursively compact `context` into a <= summary_budget-token summary. Returns a record with the
    summary plus the token accounting needed to report cost and the achieved compression ratio."""
    chunks, chunk_tokens = compaction_plan(context, working_cap, summary_budget)
    words = int(summary_budget * 0.75)          # ~0.75 words/token, for a limit the model can actually feel
    summary = ""
    pin = pout = 0
    errors: list[str] = []
    for i, chunk in enumerate(chunks):
        mem_block = summary if summary else "(empty — this is the first part)"
        user = (f"CURRENT MEMORY:\n{mem_block}\n\n"
                f"NEXT PART ({i + 1} of {len(chunks)}):\n{chunk}\n\n"
                + COMPACT_INSTRUCTION.format(budget=summary_budget, words=words))
        msgs = [{"role": "system", "content": COMPACT_SYSTEM}, {"role": "user", "content": user}]
        # Headroom over the budget: the cap is enforced by instruction, and truncating mid-sentence at
        # exactly B would corrupt the memory that the NEXT round reads back in.
        r = await client.chat(model, msgs, max_tokens=int(summary_budget * 1.25), provider=provider)
        pin += r.prompt_tokens or 0
        pout += r.completion_tokens or 0
        if r.error or not (r.text or "").strip():
            # Keep the previous summary rather than clobbering memory with an empty/failed round.
            errors.append(f"round {i + 1}/{len(chunks)}: {r.error or 'empty response'}")
            continue
        summary = r.text.strip()
    ctx_tokens = _token_count(context)
    sum_tokens = _token_count(summary)
    return {
        "summary": summary, "n_rounds": len(chunks), "chunk_tokens": chunk_tokens,
        "working_cap": working_cap, "summary_budget": summary_budget,
        "context_tokens": ctx_tokens, "summary_tokens": sum_tokens,
        "compression_ratio": (round(ctx_tokens / sum_tokens, 1)
                              if (ctx_tokens and sum_tokens) else None),
        "prompt_tokens": pin, "completion_tokens": pout,
        "errors": errors or None, "label": progress_label,
    }
