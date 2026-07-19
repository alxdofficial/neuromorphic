"""Prompt construction for the API reference baselines (dataset-agnostic).

Four conditions, each = "how much memory does the reader get":
  - floor         : question only, no history (parametric + abstention lower bound)
  - full_context  : the entire history + question (the long-context ceiling)
  - rag_bm25      : BM25 top-k passages + question (sparse retrieval)
  - rag_dense     : dense (embedding) top-k passages + question

History that overflows the reader's window is truncated (oldest sessions dropped first). Budgeting is
TOKEN-ACCURATE when a `token_budget` + reference tokenizer are available (only histories that genuinely
exceed the served window get cut — audit#2 finding 2); otherwise it falls back to a `char_budget` heuristic.
`question_date` anchors temporal questions ("how many weeks ago…"); `system`/`instruction` let the caller
supply task-specific prompts (e.g. MemoryAgentBench per-competency). Returns OpenAI-style `messages`.
"""
from __future__ import annotations

import functools
import warnings

from .retrieval import retrieve, DenseRetriever

MODES = ("floor", "full_context", "rag_bm25", "rag_dense")

SYS_MEM = ("You are a helpful assistant with access to the user's past conversation history across dated "
           "sessions. Answer the question as concisely as possible (a few words when possible), using ONLY "
           "information in the history. If the history does not contain enough information to answer, reply "
           "exactly: I don't have enough information to answer that.")
SYS_FLOOR = ("Answer the question as concisely as possible. If you do not have enough information to answer, "
             "reply exactly: I don't have enough information to answer that.")

# Proxy tokenizer for token-accurate budgeting. API models' own tokenizers differ, but a Llama-3 count is
# far more faithful than a fixed chars/token ratio, and the reserve headroom absorbs the slack. Matches the
# repo's fineweb src tokenizer so no new gated download is introduced.
_REF_TOKENIZER_NAME = "meta-llama/Llama-3.2-1B"


@functools.lru_cache(maxsize=1)
def _ref_tokenizer():
    """Load the reference tokenizer once. Returns None if unavailable (gated/offline) → char fallback."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(_REF_TOKENIZER_NAME)
    except Exception as e:  # noqa: BLE001 — degrade to char budgeting, never crash the run
        warnings.warn(f"[baselines] reference tokenizer {_REF_TOKENIZER_NAME} unavailable "
                      f"({type(e).__name__}: {e}); full_context budgeting falls back to a char heuristic.",
                      RuntimeWarning)
        return None


@functools.lru_cache(maxsize=2048)
def _token_count(text: str):
    """Cached reference-token count of `text` (identical histories recur across models). None if no tokenizer."""
    tok = _ref_tokenizer()
    return None if tok is None else len(tok(text, add_special_tokens=False)["input_ids"])


def _truncate_tail(text: str, max_chars: int) -> tuple[str, bool]:
    """Keep the most-recent `max_chars` (drop oldest sessions from the front)."""
    if len(text) <= max_chars:
        return text, False
    if max_chars <= 0:
        return "", True          # budget 0 ⇒ keep NOTHING (guard: text[-0:] == text[0:] == whole string!)
    return text[-max_chars:], True


def fit_history(text: str, token_budget: "int | None", char_budget: int) -> tuple[str, bool]:
    """Fit `text` into the reader's window, keeping the most-recent content (oldest dropped first).
    Token-accurate when `token_budget` is set and the reference tokenizer is available — only truncates
    histories that genuinely exceed the served window (no char-heuristic over-truncation of ones that fit)."""
    if token_budget is not None:
        if token_budget <= 0:
            return "", True      # budget 0 ⇒ keep NOTHING (guard: ids[-0:] == ids[0:] == whole list!)
        n = _token_count(text)
        if n is not None:
            if n <= token_budget:
                return text, False
            tok = _ref_tokenizer()
            ids = tok(text, add_special_tokens=False)["input_ids"]
            return tok.decode(ids[-token_budget:]), True
    return _truncate_tail(text, char_budget)


def _question_block(question: str, question_date: "str | None") -> str:
    """The default question section, anchored to the date it was asked on. LongMemEval temporal questions
    ('how many weeks ago…') are underspecified without the reference 'now', so the official generator puts a
    standalone `Current Date: {question_date}` line just before the question (src/generation/run_generation.py);
    we mirror that placement. No-op when date is absent."""
    date_line = f"Current Date: {question_date}\n" if question_date else ""
    return f"{date_line}# Question\n{question}\nAnswer:"


def build_messages(mode: str, *, question: str, full_history: str = "", sessions: list[str] | None = None,
                   token_budget: "int | None" = None, char_budget: int = 440_000, bm25_topk: int = 5,
                   dense: DenseRetriever | None = None, question_date: "str | None" = None,
                   system: "str | None" = None, question_template: "str | None" = None,
                   context_header: str = "# Conversation history") -> tuple[list[dict], dict]:
    """Return (messages, info). info = {"truncated": bool, "retrieved_idx": list[int]|None} — retrieval
    indices are persisted for analysis. `sessions` (retrieval units) required for the rag_* modes.

    Per-task overrides (MemoryAgentBench uses these to match its `utils/templates.py` verbatim):
      `system`            — replaces the default system prompt.
      `question_template` — replaces the default question section; `{question}` is substituted by str.replace
                            (NOT .format, so a literal `{label}` in the ICL template survives untouched).
      `context_header`    — heading placed above the history/context block.
    """
    def _qsection() -> str:
        if question_template:
            return question_template.replace("{question}", question)
        return _question_block(question, question_date)

    if mode == "floor":
        return ([{"role": "system", "content": system or SYS_FLOOR},
                 {"role": "user", "content": _qsection()}],
                {"truncated": False, "retrieved_idx": None})

    retrieved = None
    if mode == "full_context":
        ctx, trunc = fit_history(full_history, token_budget, char_budget)
    elif mode in ("rag_bm25", "rag_dense"):
        if not sessions:
            raise ValueError(f"{mode} requires `sessions`")
        method = "bm25" if mode == "rag_bm25" else "dense"
        retrieved = retrieve(sessions, question, bm25_topk, method, dense=dense)
        ctx, trunc = fit_history("\n\n".join(sessions[i] for i in retrieved), token_budget, char_budget)
    else:
        raise ValueError(f"unknown mode {mode!r} (expected one of {MODES})")

    header = f"{context_header}\n" if context_header else ""
    user = f"{header}{ctx}\n\n{_qsection()}"
    return ([{"role": "system", "content": system or SYS_MEM}, {"role": "user", "content": user}],
            {"truncated": trunc, "retrieved_idx": retrieved})


def char_budget_for(context_length_tokens: int, reserve_tokens: int = 6000,
                    chars_per_token: float = 3.6) -> int:
    """Conservative history char budget = (ctx window − reserve) × chars/token. FALLBACK sizing only
    (dry-run estimates, or when the reference tokenizer is unavailable); the live path uses token_budget."""
    return int(max(0, context_length_tokens - reserve_tokens) * chars_per_token)
