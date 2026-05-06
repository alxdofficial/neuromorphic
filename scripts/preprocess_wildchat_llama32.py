"""Pretokenize WildChat-1M -> flat int32 stream + per-turn index for Wave 4
multi-turn GRPO.

**Schema change (2026-05-06):** the previous version of this script
truncated each session to its last 256 tokens, producing single-turn
(prefix, ref) pairs for the now-deprecated single-turn Wave 4 GRPO. The
new multi-turn protocol needs the FULL conversation token stream plus a
per-turn boundary index — that's what this version produces.

Output layout (under ``--out`` prefix):
  ``{prefix}.bin``           flat int32 token stream of all kept sessions
                             (chat-templated + tokenized)
  ``{prefix}_turns.npy``     [total_turns, 4] int64 per-turn rows:
                               [session_idx, role_id, turn_start, turn_end]
                             where:
                               session_idx — which session (in 0..N_sessions-1)
                               role_id     — 0=system, 1=user, 2=assistant
                               turn_start  — index in .bin where this turn's tokens begin
                               turn_end    — index where they end (exclusive)
                             Iterating turns sorted by (session_idx, turn_start)
                             reconstructs the conversation in order.
  ``{prefix}_sessions.npy``  [N_sessions, 2] int64 [session_start, session_end]
                             in .bin. Convenience for slicing whole sessions.
  ``{prefix}.meta.json``     metadata

Filtering: only sessions with **at least ``--min-assistant-turns`` assistant
turns AND at least ``--min-tokens`` templated tokens** are kept. Defaults
favor multi-turn structure: 4 assistant turns minimum, 1500 tokens
minimum.

Templating: the same Llama-3.2-Instruct chat template is applied per
turn, so the .bin contains ``<|begin_of_text|><|start_header_id|>user
<|end_header_id|>...<|eot_id|>`` markers exactly as the model expects
them at inference. Turn boundaries in the index are aligned to the END
of one ``<|eot_id|>`` and the START of the next role header.

Memory discipline: streaming HF dataset + memmap output (same pattern
as Wave 1/2/3 preprocessors). Bounded RAM regardless of dataset size.

Usage:
  PYTHONPATH=. .venv/bin/python scripts/preprocess_wildchat_llama32.py \\
      --out data/phase_B/wildchat_llama32 \\
      --min-tokens 1500 --min-assistant-turns 4 \\
      --target-sessions 30000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer


_ROLE_ID = {"system": 0, "user": 1, "assistant": 2}


def _to_list(x):
    """Normalize tokenizer output to list[int]. apply_chat_template can
    return either list[int] or BatchEncoding depending on tokenizer
    config — len(BatchEncoding) returns dict-key count, NOT token
    count, which previously caused all sessions to be filtered as
    too-short (the silent off-by-1000s bug from 2026-05-04)."""
    if isinstance(x, list):
        return x
    if hasattr(x, "input_ids"):
        ids = x.input_ids
        return ids if isinstance(ids, list) else list(ids)
    return list(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="allenai/WildChat-1M")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="data/phase_B/wildchat_llama32")
    ap.add_argument(
        "--tokenizer", default="meta-llama/Llama-3.2-1B-Instruct",
        help="Need an Instruct-tuned tokenizer with chat_template",
    )
    ap.add_argument(
        "--min-tokens", type=int, default=1500,
        help="Drop sessions whose total templated token count is below this.",
    )
    ap.add_argument(
        "--min-assistant-turns", type=int, default=4,
        help="Drop sessions with fewer than N assistant turns. Multi-turn "
             "GRPO needs at least a few turns to be meaningful.",
    )
    ap.add_argument(
        "--max-tokens-per-session", type=int, default=8000,
        help="Hard truncation per session (memory + compute cap). Sessions "
             "longer than this are TRUNCATED at a turn boundary.",
    )
    ap.add_argument(
        "--target-sessions", type=int, default=30_000,
        help="Stop once this many qualifying sessions are accumulated.",
    )
    ap.add_argument(
        "--total-token-budget", type=int, default=2_000_000_000,
        help="Hard cap on total bytes for the .bin (2B int32 = 8 GB).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path = out_path.with_suffix(".bin")
    meta_path = out_path.with_suffix(".meta.json")
    turns_path = out_path.parent / (out_path.name + "_turns.npy")
    sessions_path = out_path.parent / (out_path.name + "_sessions.npy")

    print(f"[preprocess] tokenizer = {args.tokenizer}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.chat_template is None:
        raise SystemExit(
            f"{args.tokenizer} has no chat_template — need an "
            "Instruct-tuned tokenizer for chat templating",
        )
    eos_id = tok.eos_token_id
    if eos_id is None:
        raise SystemExit("tokenizer.eos_token_id is None")
    print(
        f"[preprocess] eos_id={eos_id}, vocab={tok.vocab_size}, "
        f"min_tokens={args.min_tokens}, "
        f"min_assistant_turns={args.min_assistant_turns}, "
        f"max_tokens_per_session={args.max_tokens_per_session}",
        flush=True,
    )

    print(f"[preprocess] streaming {args.dataset} split={args.split}", flush=True)
    ds = load_dataset(args.dataset, split=args.split, streaming=True)

    print(
        f"[preprocess] allocating memmap {bin_path} "
        f"({args.total_token_budget * 4 / 1e9:.1f} GB on disk)", flush=True,
    )
    out_bin = np.memmap(
        bin_path, dtype=np.int32, mode="w+",
        shape=(args.total_token_budget,),
    )

    turn_rows: list[tuple[int, int, int, int]] = []
    session_rows: list[tuple[int, int]] = []

    n_filled = 0
    n_kept = 0
    n_seen = 0
    n_too_short = 0
    n_too_few_turns = 0
    n_truncated = 0
    t0 = time.perf_counter()
    last_log = t0
    last_flush = t0
    stop_reason = "stream_exhausted"

    for example in ds:
        n_seen += 1
        now = time.perf_counter()
        if now - last_log > 10.0:
            elapsed = now - t0
            print(
                f"[preprocess] seen {n_seen:,}, kept {n_kept:,} "
                f"(too_short={n_too_short:,}, too_few_turns={n_too_few_turns:,}, "
                f"truncated={n_truncated:,}), {n_filled:,} tokens, "
                f"{elapsed:.1f}s", flush=True,
            )
            last_log = now
        messages = example.get("conversation") or example.get("messages")
        if not messages or len(messages) < 2:
            continue

        try:
            full_ids = _to_list(tok.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
            ))
        except Exception:
            continue

        if len(full_ids) < args.min_tokens:
            n_too_short += 1
            continue

        n_assistant = sum(
            1 for m in messages
            if (m.get("role") or m.get("from")) == "assistant"
        )
        if n_assistant < args.min_assistant_turns:
            n_too_few_turns += 1
            continue

        # Per-turn boundary detection: re-template each prefix-up-to-and-
        # including-message-i and use the length difference to determine
        # where each turn starts/ends in the full templated stream. This
        # is robust to template-injected tokens (BOS, role headers, EOT).
        turn_boundaries: list[tuple[str, int, int]] = []
        prev_end = 0
        skip = False
        for i, m in enumerate(messages):
            role = m.get("role") or m.get("from")
            if role not in _ROLE_ID:
                skip = True
                break
            try:
                cum_ids = _to_list(tok.apply_chat_template(
                    messages[: i + 1], tokenize=True,
                    add_generation_prompt=False,
                ))
            except Exception:
                skip = True
                break
            cur_end = len(cum_ids)
            turn_boundaries.append((role, prev_end, cur_end))
            prev_end = cur_end
        if skip:
            continue
        if turn_boundaries and turn_boundaries[-1][2] != len(full_ids):
            # Per-turn templating drifted from the full template — skip
            # rather than mis-align.
            continue

        # Truncate at a turn boundary if it exceeds the cap.
        max_n = args.max_tokens_per_session
        if len(full_ids) > max_n:
            kept_turns = []
            for (role, ts, te) in turn_boundaries:
                if te > max_n:
                    break
                kept_turns.append((role, ts, te))
            if not kept_turns:
                continue
            n_assistant_kept = sum(
                1 for (role, _, _) in kept_turns if role == "assistant"
            )
            if n_assistant_kept < args.min_assistant_turns:
                n_too_few_turns += 1
                continue
            full_ids = full_ids[: kept_turns[-1][2]]
            turn_boundaries = kept_turns
            n_truncated += 1

        write_n = len(full_ids)
        if n_filled + write_n > args.total_token_budget:
            stop_reason = "total_token_budget_exceeded"
            break
        session_start = n_filled
        out_bin[n_filled: n_filled + write_n] = full_ids
        n_filled += write_n
        session_end = n_filled
        session_rows.append((session_start, session_end))

        si = len(session_rows) - 1
        for (role, ts, te) in turn_boundaries:
            turn_rows.append((si, _ROLE_ID[role],
                              session_start + ts, session_start + te))

        n_kept += 1
        if now - last_flush > 60.0:
            out_bin.flush()
            last_flush = now
        if n_kept >= args.target_sessions:
            stop_reason = "target_sessions_hit"
            break

    elapsed = time.perf_counter() - t0
    print(
        f"[preprocess] done: {n_kept:,} sessions, {len(turn_rows):,} turns, "
        f"{n_filled:,} tokens, {n_seen:,} seen, {elapsed:.1f}s, "
        f"stop_reason={stop_reason}", flush=True,
    )
    if stop_reason == "total_token_budget_exceeded" and n_kept < args.target_sessions:
        print(
            f"[preprocess] WARNING: stopped at {n_kept:,} sessions, "
            f"below target {args.target_sessions:,} — token budget "
            f"({args.total_token_budget:,}) hit first.", flush=True,
        )

    out_bin.flush()
    del out_bin

    if n_filled < args.total_token_budget:
        print(
            f"[preprocess] truncating to {n_filled:,} tokens "
            f"({n_filled * 4 / 1e9:.2f} GB)", flush=True,
        )
        src = np.memmap(
            bin_path, dtype=np.int32, mode="r",
            shape=(args.total_token_budget,),
        )
        truncated = np.array(src[:n_filled])
        del src
        bin_path.unlink()
        truncated.tofile(bin_path)
        del truncated

    turns_arr = np.asarray(turn_rows, dtype=np.int64) if turn_rows else \
        np.zeros((0, 4), dtype=np.int64)
    sessions_arr = np.asarray(session_rows, dtype=np.int64) if session_rows else \
        np.zeros((0, 2), dtype=np.int64)
    np.save(turns_path, turns_arr)
    np.save(sessions_path, sessions_arr)
    print(f"[preprocess] wrote {turns_path}: {turns_arr.shape}", flush=True)
    print(f"[preprocess] wrote {sessions_path}: {sessions_arr.shape}", flush=True)

    meta = {
        "schema_version": 2,                  # v2 = multi-turn (v1 was prefix/ref)
        "sessions": int(n_kept),
        "turns": int(len(turn_rows)),
        "tokens": int(n_filled),
        "seen": int(n_seen),
        "too_short": int(n_too_short),
        "too_few_turns": int(n_too_few_turns),
        "truncated": int(n_truncated),
        "stop_reason": stop_reason,
        "target_sessions": int(args.target_sessions),
        "total_token_budget": int(args.total_token_budget),
        "tokenizer": args.tokenizer,
        "source_dataset": args.dataset,
        "split": args.split,
        "min_tokens": args.min_tokens,
        "min_assistant_turns": args.min_assistant_turns,
        "max_tokens_per_session": args.max_tokens_per_session,
        "eos_id": int(eos_id),
        "role_id_map": _ROLE_ID,
        "dtype": "int32",
        "elapsed_s": round(elapsed, 2),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"[preprocess] wrote {meta_path}", flush=True)


if __name__ == "__main__":
    main()
