"""Wave 4 WildChat-1M loader for Phase-2 multi-turn GRPO.

Two iterator flavors:

- ``wildchat_session_grpo_iter`` (raw session iter):
  Yields one ``MultiTurnSession`` at a time, randomly sampled from the
  v2-schema pretokenized data. Used as the data source by the turn-pair
  iter and by ``grpo_session_step``'s sequential fallback path (which
  isn't the production Wave 4 path anymore but is still useful for
  debugging multi-turn sessions intact).

- ``wildchat_turn_pair_grpo_batch_iter`` (turn-batched, production Wave 4):
  Flattens sessions into individual ``TurnPair(prior, ref)`` units —
  each assistant turn within each session becomes one independent
  training unit. Maintains a sorted pool of pairs (by prior length);
  each next() yields a list of B pairs picked from a contiguous random
  window of similar prior lengths. Wraps each pair as a 2-turn
  ``MultiTurnSession`` so it slots into the existing
  ``_grpo_session_step_uniform_batched`` fast path — same B*K parallel
  rollouts that Wave 3 uses.

  Verlog-style: restructures the batching unit from "session" to
  "turn-pair," yielding true B-way parallelism for variable-length
  multi-turn data without walker changes.

Yields ``MultiTurnSession`` objects from the v2-schema pretokenized
output of ``scripts/preprocess_wildchat_llama32.py``:

  data/phase_B/wildchat_llama32.bin             int32 token stream (full sessions)
  data/phase_B/wildchat_llama32_turns.npy       [total_turns, 4] int64
                                                [session_idx, role_id, turn_start, turn_end]
  data/phase_B/wildchat_llama32_sessions.npy    [N_sessions, 2] int64
                                                [session_start, session_end]

Each yielded session includes the full chat-templated token stream plus
a list of (role, ids) per turn, in order. The multi-turn GRPO trainer
walks the session turn-by-turn:
  - user / system turns: walker observes (sampled routing, log_pi
    captured); LM produces logits but no GRPO loss
  - assistant turns: walker generates K rollouts under sampled routing
    + sampled tokens; reward = BERT-cosine(generated, ground_truth);
    REINFORCE update against advantage

Memory ceiling: the .bin is memmapped (no RAM cost beyond OS page cache),
the index arrays are loaded once (~tens of MB for 30K sessions).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


_ROLE_NAME = {0: "system", 1: "user", 2: "assistant"}


@dataclass
class MultiTurnTurn:
    """One turn within a session."""
    role: str                          # "system" | "user" | "assistant"
    ids: torch.Tensor                  # [L_turn] long


@dataclass
class MultiTurnSession:
    """One full WildChat conversation, segmented into turns."""
    session_idx: int
    turns: list[MultiTurnTurn]
    total_tokens: int                  # sum of len(turn.ids) — equals
                                       # the session's slice in the .bin


def wildchat_session_grpo_iter(
    bin_path: str | Path = "data/phase_B/wildchat_llama32.bin",
    turns_path: str | Path = "data/phase_B/wildchat_llama32_turns.npy",
    sessions_path: str | Path = "data/phase_B/wildchat_llama32_sessions.npy",
    *,
    device: torch.device | str = "cuda",
    seed: int = 0,
    min_assistant_turns: int = 1,
) -> Iterator[MultiTurnSession]:
    """Yield WildChat multi-turn sessions for Wave 4 GRPO.

    Random sampling with replacement from the full session pool. Iterator
    runs forever; the trainer controls when to stop.

    Args:
        bin_path: pretokenized flat int32 stream.
        turns_path: [total_turns, 4] int64 turn-boundary index.
        sessions_path: [N_sessions, 2] int64 session boundaries.
        device: where to place yielded tensors.
        seed: RNG seed for session sampling.
        min_assistant_turns: skip sessions with fewer than N assistant
            turns (defensive — preprocessor already filters, but allows
            the trainer to be stricter without re-preprocessing).

    Yields one MultiTurnSession per call. Each session's turns are
    yielded in order; the trainer walks them turn-by-turn.
    """
    bin_p = Path(bin_path)
    turns_p = Path(turns_path)
    sessions_p = Path(sessions_path)
    if not bin_p.exists():
        raise FileNotFoundError(
            f"WildChat pretokenized bin not found at {bin_p}. Run "
            "scripts/preprocess_wildchat_llama32.py first.",
        )
    if not turns_p.exists() or not sessions_p.exists():
        raise FileNotFoundError(
            f"WildChat v2 schema files not found ({turns_p}, "
            f"{sessions_p}). Re-run the preprocessor — the loader "
            "expects the multi-turn schema (schema_version=2).",
        )
    tokens = np.memmap(bin_p, dtype=np.int32, mode="r")
    turns_idx = np.load(turns_p)               # [total_turns, 4]
    sessions_idx = np.load(sessions_p)         # [N_sessions, 2]
    if turns_idx.ndim != 2 or turns_idx.shape[1] != 4:
        raise ValueError(
            f"turns shape {turns_idx.shape} != [N_turns, 4] — schema "
            "mismatch (preprocessor schema_version!=2?)",
        )
    if sessions_idx.ndim != 2 or sessions_idx.shape[1] != 2:
        raise ValueError(
            f"sessions shape {sessions_idx.shape} != [N_sessions, 2]",
        )
    n_sessions = sessions_idx.shape[0]
    if n_sessions == 0:
        raise ValueError("WildChat sessions index has 0 sessions")

    # Group turns by session_idx for O(1) lookup.
    # turns_idx is naturally sorted by session_idx since the preprocessor
    # appends rows in session order — verify and build offset table.
    si_col = turns_idx[:, 0]
    if not np.all(np.diff(si_col) >= 0):
        # Fall back to argsort if rows aren't ordered.
        order = np.argsort(si_col, kind="stable")
        turns_idx = turns_idx[order]
        si_col = turns_idx[:, 0]
    # Build [N_sessions+1] offset table: turn_offsets[s] = first row in
    # turns_idx with session_idx==s; turn_offsets[N_sessions] = total.
    turn_offsets = np.searchsorted(si_col, np.arange(n_sessions + 1))

    print(
        f"[Wave4Loader] loaded {n_sessions:,} sessions, "
        f"{turns_idx.shape[0]:,} turns from {bin_p.name}",
        flush=True,
    )

    rng = np.random.default_rng(seed)
    dev = torch.device(device)

    while True:
        si = int(rng.integers(0, n_sessions))
        s_lo, s_hi = int(sessions_idx[si, 0]), int(sessions_idx[si, 1])
        t_lo, t_hi = int(turn_offsets[si]), int(turn_offsets[si + 1])
        rows = turns_idx[t_lo:t_hi]            # [n_turns_this_session, 4]

        n_assist = int((rows[:, 1] == 2).sum())
        if n_assist < min_assistant_turns:
            continue

        turns: list[MultiTurnTurn] = []
        for row in rows:
            _, role_id, ts, te = (int(x) for x in row)
            if te <= ts:
                continue
            role = _ROLE_NAME.get(role_id, "user")
            ids_np = np.asarray(tokens[ts:te], dtype=np.int64)
            ids_t = torch.from_numpy(ids_np).to(dev, non_blocking=True)
            turns.append(MultiTurnTurn(role=role, ids=ids_t))
        if not any(t.role == "assistant" for t in turns):
            # Defensive — shouldn't trigger given the n_assist check above
            continue
        yield MultiTurnSession(
            session_idx=si,
            turns=turns,
            total_tokens=s_hi - s_lo,
        )


# ----------------------------------------------------------------------
# Turn-pair flattener + sort-and-sample batch iterator (Verlog-style)
# ----------------------------------------------------------------------


@dataclass
class TurnPair:
    """One (cumulative_prior, response) training unit, extracted from a
    multi-turn session at one of its assistant turns.

    Verlog-style turn-batching: instead of "session" being the batching
    unit, "turn-pair" is. A 5-assistant-turn session expands into 5
    independent TurnPairs that can be batched with pairs from any other
    session at any other turn index.

    Each TurnPair is structurally identical to a Wave-3 (prefix, ref)
    pair, so it slots into the existing uniform-batched fast path.
    """
    prior_ids: torch.Tensor                 # [prior_len] long, the cumulative prior
    ref_ids: torch.Tensor                   # [L_ref] long, the assistant reference
    session_idx: int                        # source session — for telemetry / dedup
    turn_idx: int                           # which assistant turn within the session
    prior_len: int                          # cached len(prior_ids) for sort key

    def to_two_turn_session(self) -> "MultiTurnSession":
        """Wrap as a MultiTurnSession so the existing
        `_grpo_session_step_uniform_batched` fast path consumes it."""
        return MultiTurnSession(
            session_idx=self.session_idx,
            turns=[
                MultiTurnTurn(role="user", ids=self.prior_ids),
                MultiTurnTurn(role="assistant", ids=self.ref_ids),
            ],
            total_tokens=int(self.prior_len + self.ref_ids.numel()),
        )


def session_to_turn_pairs(session: MultiTurnSession) -> Iterator[TurnPair]:
    """Walk a session and yield one TurnPair per assistant turn. The
    cumulative prior at turn t is concat of all turn ids at indices
    0..t-1. Skips the first assistant turn if there's no prior context
    (degenerate session).

    The TurnPair's prior_ids is a fresh tensor (same device as the
    session's turn ids) — independent of the session, so the session
    object can be GC'd safely.
    """
    cumulative: list[int] = []
    for t, turn in enumerate(session.turns):
        if turn.role == "assistant":
            if not cumulative:
                # Assistant-first session, no prior context — drop this
                # turn entirely (don't yield it AND don't pollute the
                # cumulative prior with its tokens). An assistant-only
                # context as a prefix to the NEXT assistant turn would
                # be a malformed chat-template input.
                continue
            prior_t = torch.tensor(
                cumulative, dtype=torch.long, device=turn.ids.device,
            )
            yield TurnPair(
                prior_ids=prior_t,
                ref_ids=turn.ids,
                session_idx=session.session_idx,
                turn_idx=t,
                prior_len=len(cumulative),
            )
        cumulative.extend(turn.ids.cpu().tolist())


def wildchat_turn_pair_grpo_batch_iter(
    bin_path: str | Path = "data/phase_B/wildchat_llama32.bin",
    turns_path: str | Path = "data/phase_B/wildchat_llama32_turns.npy",
    sessions_path: str | Path = "data/phase_B/wildchat_llama32_sessions.npy",
    *,
    batch_size: int,
    pool_size: int = 2048,
    device: torch.device | str = "cuda",
    seed: int = 0,
    min_assistant_turns: int = 1,
    truncate_priors_to_min: bool = True,
) -> Iterator[list[MultiTurnSession]]:
    """Yield batches of B turn-pairs from WildChat, each wrapped as a
    2-turn ``MultiTurnSession`` so they slot directly into
    ``grpo_session_step``'s uniform-batched fast path.

    Sort-and-sample protocol:
    1. Maintain a pool of up to ``pool_size`` TurnPairs (refilled from
       sessions on demand).
    2. Pool is kept sorted by ``prior_len`` after each refill.
    3. Per batch: pick a uniform-random window of B contiguous neighbors
       from the sorted pool — they have near-uniform prior lengths.
    4. The picked pairs are removed from the pool (no replacement within
       a refill cycle).
    5. ``truncate_priors_to_min``: when True (default), all B priors in
       the batch are left-truncated to the SHORTEST prior length within
       the batch. Keeps the most-recent context intact and produces a
       stackable [B, T_pre] tensor without padding/masking. Loss: each
       longer-prior pair drops a few early tokens. Sort-and-sample
       keeps this loss small.

    Args:
        batch_size: B sessions yielded per call.
        pool_size: M turn-pairs maintained in memory between refills.
            Larger = better neighbor quality, more memory; M=2048 is a
            reasonable default (~65 MB for 4K-avg priors).

    Yields:
        list[MultiTurnSession] of length ``batch_size``, each a 2-turn
        (user prefix, assistant ref) wrapper around a TurnPair.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0; got {batch_size}")
    if pool_size < batch_size:
        raise ValueError(
            f"pool_size ({pool_size}) must be >= batch_size ({batch_size})"
        )

    session_iter = wildchat_session_grpo_iter(
        bin_path=bin_path, turns_path=turns_path,
        sessions_path=sessions_path,
        device=device, seed=seed,
        min_assistant_turns=min_assistant_turns,
    )
    rng = np.random.default_rng(seed + 1)
    pool: list[TurnPair] = []

    print(
        f"[Wave4TurnPairLoader] B={batch_size}, pool_size={pool_size}, "
        f"truncate_priors_to_min={truncate_priors_to_min}", flush=True,
    )

    while True:
        # Refill pool when low.
        if len(pool) < batch_size:
            while len(pool) < pool_size:
                try:
                    session = next(session_iter)
                except StopIteration:
                    break
                pool.extend(session_to_turn_pairs(session))
            if len(pool) < batch_size:
                # Session iter exhausted and pool can't reach batch_size.
                return
            pool.sort(key=lambda p: p.prior_len)

        # Pick a random contiguous window of B neighbors.
        max_start = len(pool) - batch_size
        start = int(rng.integers(0, max_start + 1))
        batch = pool[start: start + batch_size]
        del pool[start: start + batch_size]

        if truncate_priors_to_min:
            min_len = min(p.prior_len for p in batch)
            # Left-truncate: keep the LAST min_len tokens of each prior
            # (most recent context, where the assistant's about to respond).
            truncated_pairs = [
                TurnPair(
                    prior_ids=p.prior_ids[-min_len:].contiguous(),
                    ref_ids=p.ref_ids,
                    session_idx=p.session_idx,
                    turn_idx=p.turn_idx,
                    prior_len=min_len,
                )
                for p in batch
            ]
            yield [tp.to_two_turn_session() for tp in truncated_pairs]
        else:
            yield [tp.to_two_turn_session() for tp in batch]
