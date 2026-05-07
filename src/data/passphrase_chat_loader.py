"""Wave 3 passphrase chat-injected loader for Phase-2 GRPO.

Yields ``(prefix_ids [1, T_pre], reference_ids [L])`` per training step.
This is the ``grpo_step``-shaped interface: a single prefix per step,
which ``grpo_step`` internally replicates K times for the rollouts.

The fact is wrapped as a user turn ("by the way, ..."), N filler chat
turns are sampled from UltraChat, then a user turn asks the question;
the assistant's reference answer is the REINFORCE target. Phase-2 GRPO
(DeepSeek-style sample/replay) trains the walker's neuromod to bias
routing such that the AR-generated answer matches the reference under
BERT-cosine reward.

Reuses ``data/passphrase/expanded.json`` (500 facts × 3 paraphrases × 5
questions × 3 reference answers, hand-authored). Train/heldout split
(seed=42, 20 facts held out) is determined by
``src.data.passphrase_facts._split_train_heldout``.

Chat filler: sampled from the pretokenized UltraChat .bin we built for
Wave 2 (``data/phase_B/ultrachat_llama32.bin``).

The chat envelope uses Llama-3.2-Instruct's chat template applied via
the tokenizer; we don't manually interleave role tags here since the
template knows how to format <|user|> / <|assistant|> / <|eot_id|>
markers correctly for the host LM.

Historical note: an earlier Wave 3 (AR-unrolled teacher-forced SFT on
filler+fact+filler+question) was retired in scope-B cleanup
(2026-05-06) because teacher forcing under the LM's full attention
made the walker contribute only a tiny CE delta — see
``docs/training_strategy.md`` for the rationale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import torch

from src.data.passphrase_facts import _load_facts, _split_train_heldout


@dataclass
class ChatGRPOBatch:
    """One (prefix, reference) pair for a phase-2 GRPO step."""
    prefix_ids: torch.Tensor       # [1, T_pre]   long
    reference_ids: torch.Tensor    # [L]          long
    fact_id: int                   # for telemetry
    question_idx: int              # which of the 5 questions; for telemetry


class _UltraChatFillerPool:
    """Random spans of pretokenized UltraChat to use as filler chat turns.

    Loads the whole .bin via memmap (no full-corpus RAM cost). At sample
    time, slices a random contiguous window of ``n_tokens`` tokens.
    Falls back to error if the bin is missing — we never want to
    on-the-fly-tokenize UltraChat in the middle of training.

    The UltraChat .bin already has chat templating + EOT markers baked
    in (see ``scripts/preprocess_ultrachat_llama32.py``), so a contiguous
    window pulls naturally chat-shaped content. We don't try to align to
    turn boundaries — random spans are fine for "filler that looks like
    chat".
    """

    def __init__(
        self,
        bin_path: str | Path = "data/phase_B/ultrachat_llama32.bin",
        seed: int = 0,
    ) -> None:
        p = Path(bin_path)
        if not p.exists():
            raise FileNotFoundError(
                f"UltraChat pretokenized bin not found at {p}. Run "
                "scripts/preprocess_ultrachat_llama32.py first."
            )
        self.tokens = np.memmap(p, dtype=np.int32, mode="r")
        self._rng = np.random.default_rng(seed)
        print(f"[UltraChatFiller] memmap'd {len(self.tokens):,} tokens "
              f"from {p.name}")

    def sample(self, n_tokens: int) -> list[int]:
        if n_tokens <= 0:
            return []
        if n_tokens > len(self.tokens):
            raise ValueError(
                f"requested {n_tokens} > pool size {len(self.tokens)}"
            )
        start = int(
            self._rng.integers(0, len(self.tokens) - n_tokens + 1),
        )
        return [int(t) for t in self.tokens[start:start + n_tokens]]


def _build_chat_grpo_example(
    fact,
    rng: random.Random,
    *,
    tokenizer,
    filler_pool: _UltraChatFillerPool,
    T_pre: int,
    L_ref: int,
    filler_mid_tokens: int,
) -> tuple[list[int], list[int]] | None:
    """Build one (prefix_ids, reference_ids) example.

    Layout (logical):
        <fact-injection user turn>: "By the way, {paraphrase}."
        <chat filler> (filler_mid_tokens of pretokenized chat)
        <question user turn>: "{question}"
        <assistant turn open>:                        <-- prefix ends here
        <reference answer>                            <-- reference target

    Tokenized via Llama-3.2-Instruct chat_template. We tokenize each
    chat segment separately, concatenate, and trim filler to fit T_pre.
    Returns None if the example doesn't fit T_pre even with zero filler
    (fact too long; rare).
    """
    paraphrase = rng.choice(fact.paraphrases)
    q_idx = rng.randrange(len(fact.questions))
    question = fact.questions[q_idx]
    reference = rng.choice(fact.reference_answers)

    # Tokenize each chat-message-as-text. We use add_special_tokens=False
    # because chat_template already inserts BOS / role markers and we'll
    # apply that template to the FULL message list at the end.
    fact_msg = {"role": "user", "content": f"By the way, {paraphrase}"}
    question_msg = {"role": "user", "content": question}
    # apply_chat_template(tokenize=True) can return a list[int] OR a
    # BatchEncoding depending on tokenizer / kwargs combo. Normalize via
    # `return_tensors=None` and ensure we always end up with list[int].
    def _to_list(x):
        if isinstance(x, list):
            return x
        # BatchEncoding has .input_ids; on tokenize=True with no
        # return_tensors it can be a list[int] already, else a torch/np
        # array.
        if hasattr(x, "input_ids"):
            ids = x.input_ids
            return ids if isinstance(ids, list) else list(ids)
        return list(x)

    fact_only_tpl = _to_list(tokenizer.apply_chat_template(
        [fact_msg], tokenize=True, add_generation_prompt=False,
    ))
    qa_tpl = _to_list(tokenizer.apply_chat_template(
        [question_msg], tokenize=True, add_generation_prompt=True,
    ))
    # apply_chat_template prepends <|begin_of_text|> (BOS) to each call;
    # since we concatenate fact_only_tpl + ... + qa_tpl, we'd get TWO
    # BOS markers. Strip the leading BOS from qa_tpl (and any
    # subsequent fragments) to keep the LM's chat-parse coherent.
    bos_id = getattr(tokenizer, "bos_token_id", None)
    if bos_id is not None and qa_tpl and qa_tpl[0] == bos_id:
        qa_tpl = qa_tpl[1:]
    ref_ids = tokenizer.encode(reference, add_special_tokens=False)

    # Reserve room for: fact_only_tpl + filler + qa_tpl in T_pre.
    fixed = len(fact_only_tpl) + len(qa_tpl)
    available_filler = T_pre - fixed
    if available_filler < 0:
        # Even fact + question won't fit; skip.
        return None
    use_filler = min(filler_mid_tokens, available_filler)

    filler_ids = filler_pool.sample(use_filler) if use_filler > 0 else []

    # Truncate reference to L_ref tokens (no padding here — reward
    # function looks at the last L tokens of generated which is also the
    # length of reference).
    ref_ids = ref_ids[:L_ref]

    prefix_ids = fact_only_tpl + filler_ids + qa_tpl
    # Pad/trim to exactly T_pre. We trim/pad from the FILLER region
    # (the middle) so fact + question stay intact.
    if len(prefix_ids) > T_pre:
        # Should be rare given the available_filler arithmetic above;
        # handle defensively by trimming filler from the right.
        excess = len(prefix_ids) - T_pre
        # filler region is [len(fact_only_tpl), len(fact_only_tpl)+len(filler_ids)]
        end = len(fact_only_tpl) + len(filler_ids)
        prefix_ids = prefix_ids[:end - excess] + prefix_ids[end:]
    elif len(prefix_ids) < T_pre:
        # Pad to T_pre with NEUTRAL pad tokens — NOT more chat-filler.
        # Padding with chat-filler would silently override the
        # `filler_mid_tokens` parameter (the curriculum knob), making
        # all examples sit at the maximum fact-to-question distance
        # regardless of `filler_min`/`filler_max`. Padding with pad
        # tokens preserves the curriculum: `filler_mid_tokens` controls
        # how much REAL chat content sits between fact and question;
        # the rest of the prefix is "silent" PAD tokens that the walker
        # still has to walk through but don't constitute confounding
        # chat content for the LM.
        pad_id = getattr(tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(tokenizer, "eos_token_id", 0)
        pad = T_pre - len(prefix_ids)
        end = len(fact_only_tpl) + len(filler_ids)
        prefix_ids = prefix_ids[:end] + [pad_id] * pad + prefix_ids[end:]
    assert len(prefix_ids) == T_pre, (
        f"prefix_ids length {len(prefix_ids)} != T_pre {T_pre}"
    )
    return prefix_ids, ref_ids


def passphrase_chat_grpo_iter(
    expanded_path: str | Path,
    tokenizer,
    *,
    T_pre: int,
    L_ref: int,
    filler_mid_min: int = 100,
    filler_mid_max: int = 1500,
    filler_curriculum: list[tuple[int, int]] | None = None,
    n_heldout: int = 20,
    device: torch.device | str = "cuda",
    ultrachat_bin: str | Path = "data/phase_B/ultrachat_llama32.bin",
    seed: int = 42,
) -> Iterator[ChatGRPOBatch]:
    """Yield Wave 3 chat-injected (prefix, reference) examples.

    Args:
        expanded_path: path to ``data/passphrase/expanded.json`` (500 facts).
        tokenizer: Llama-3.2-Instruct tokenizer (must have ``chat_template``).
        T_pre: prefix length. Production GRPO uses 256. Tight given the
            chat envelope; if too short, drop ``filler_mid_*`` or bump T_pre.
        L_ref: reference length. Production GRPO uses 128.
        filler_mid_min/max: bounds on the filler-tokens count between
            fact-injection and question. Random within bounds per example.
        filler_curriculum: optional list of (step_threshold, target_tokens)
            for ramping filler length over training. If None, uses
            uniform(min, max) per example.
        n_heldout: how many fact ids to reserve for eval (deterministic
            split, seeded). Determines which facts the training run
            never sees, for eventual heldout eval.
        device: where to place yielded tensors.
        ultrachat_bin: pretokenized UltraChat path for chat filler.
        seed: RNG seed (used for both fact split and per-example sampling).

    The iterator runs forever; the trainer controls when to stop.
    """
    if tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer must have a chat_template (use Llama-3.2-Instruct)"
        )
    facts = _load_facts(expanded_path)
    train_facts, heldout_facts = _split_train_heldout(
        facts, n_heldout=n_heldout, seed=seed,
    )
    if not train_facts:
        raise ValueError("no training facts after split")

    rng = random.Random(seed + 1)
    filler_pool = _UltraChatFillerPool(
        bin_path=ultrachat_bin, seed=seed + 2,
    )
    print(
        f"[Wave3Loader] {len(train_facts)} train facts, "
        f"{len(heldout_facts)} heldout, T_pre={T_pre}, L_ref={L_ref}",
    )

    dev = torch.device(device)
    step = 0
    while True:
        # Curriculum-controlled filler length, else uniform.
        if filler_curriculum is not None:
            filler_target = filler_mid_min
            for thresh, target in filler_curriculum:
                if step >= thresh:
                    filler_target = target
            filler_mid_tokens = filler_target
        else:
            filler_mid_tokens = rng.randint(filler_mid_min, filler_mid_max)

        fact = rng.choice(train_facts)
        result = _build_chat_grpo_example(
            fact, rng, tokenizer=tokenizer, filler_pool=filler_pool,
            T_pre=T_pre, L_ref=L_ref, filler_mid_tokens=filler_mid_tokens,
        )
        if result is None:
            # Fact too long; skip
            continue
        prefix_ids, ref_ids = result
        prefix_t = torch.tensor(
            [prefix_ids], dtype=torch.long, device=dev,
        )
        ref_t = torch.tensor(ref_ids, dtype=torch.long, device=dev)
        # Find which question this is for (linear-scan, fine for telemetry)
        q_idx = 0
        for i, q in enumerate(fact.questions):
            if q in tokenizer.decode(prefix_ids):
                q_idx = i
                break
        yield ChatGRPOBatch(
            prefix_ids=prefix_t,
            reference_ids=ref_t,
            fact_id=fact.id,
            question_idx=q_idx,
        )
        step += 1


def passphrase_chat_grpo_session_iter(
    expanded_path: str | Path,
    tokenizer,
    *,
    T_pre: int,
    L_ref: int,
    filler_mid_min: int = 100,
    filler_mid_max: int = 1500,
    filler_curriculum: list[tuple[int, int]] | None = None,
    n_heldout: int = 20,
    device: torch.device | str = "cuda",
    ultrachat_bin: str | Path = "data/phase_B/ultrachat_llama32.bin",
    seed: int = 42,
):
    """Yield Wave 3 examples as ``MultiTurnSession`` objects (the unified
    session-format API used by both Wave 3 and Wave 4 GRPO).

    Each yielded session has exactly **two turns**:
    - Turn 0 (role="user"): the full chat-templated prefix
      (fact-injection + filler + question), as ONE pseudo-user turn
    - Turn 1 (role="assistant"): the reference answer, treated as the
      assistant's ground-truth response

    From ``grpo_session_step``'s perspective, this looks like a 1-turn
    conversation where the K rollouts sample the assistant's reply.
    The cumulative_prior accumulator absorbs turn 0's tokens into
    "prior", then runs one GRPO step at turn 1.

    Same data construction as ``passphrase_chat_grpo_iter``; this is
    just a different output format for the unified Wave 3 + Wave 4 API.
    """
    # Lazy import to avoid circular imports.
    from src.data.wildchat_loader import MultiTurnSession, MultiTurnTurn

    if tokenizer.chat_template is None:
        raise ValueError(
            "tokenizer must have a chat_template (use Llama-3.2-Instruct)"
        )
    facts = _load_facts(expanded_path)
    train_facts, heldout_facts = _split_train_heldout(
        facts, n_heldout=n_heldout, seed=seed,
    )
    if not train_facts:
        raise ValueError("no training facts after split")

    rng = random.Random(seed + 1)
    filler_pool = _UltraChatFillerPool(
        bin_path=ultrachat_bin, seed=seed + 2,
    )
    print(
        f"[Wave3SessionLoader] {len(train_facts)} train facts, "
        f"{len(heldout_facts)} heldout, T_pre={T_pre}, L_ref={L_ref}",
    )

    dev = torch.device(device)
    step = 0
    while True:
        if filler_curriculum is not None:
            filler_target = filler_mid_min
            for thresh, target in filler_curriculum:
                if step >= thresh:
                    filler_target = target
            filler_mid_tokens = filler_target
        else:
            filler_mid_tokens = rng.randint(filler_mid_min, filler_mid_max)

        fact = rng.choice(train_facts)
        result = _build_chat_grpo_example(
            fact, rng, tokenizer=tokenizer, filler_pool=filler_pool,
            T_pre=T_pre, L_ref=L_ref, filler_mid_tokens=filler_mid_tokens,
        )
        if result is None:
            continue
        prefix_ids, ref_ids = result
        prefix_t = torch.tensor(prefix_ids, dtype=torch.long, device=dev)
        ref_t = torch.tensor(ref_ids, dtype=torch.long, device=dev)
        yield MultiTurnSession(
            session_idx=step,
            turns=[
                MultiTurnTurn(role="user", ids=prefix_t),
                MultiTurnTurn(role="assistant", ids=ref_t),
            ],
            total_tokens=int(prefix_t.numel() + ref_t.numel()),
        )
        step += 1
