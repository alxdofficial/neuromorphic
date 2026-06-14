"""Generic generation driver.

Takes a TaskGenerator + run config, produces passages.jsonl +
questions.jsonl. Handles tokenization, streaming JSONL emission,
config-space reporting, optional answer verification, and reproducible
per-scenario seeding.

Usage from a task's generate.py:

    from scripts.data_gen.common.driver import generate_task, default_argparser
    from scripts.data_gen.common.drafts import TaskGenerator

    GEN = TaskGenerator(
        task_family="<name>",
        build_scenario=...,
        render_passages=...,
        enumerate_questions=...,
        config_space_size=lambda: ...,
        surface_variant_count=lambda: ...,
        verify=...,             # optional
    )

    if __name__ == "__main__":
        args = default_argparser().parse_args()
        generate_task(GEN, args)
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from scripts.data_gen.common.drafts import (
    PassageDraft, QuestionDraft, TaskGenerator,
)


# ── CLI ───────────────────────────────────────────────────────────────


def default_argparser(description: str = "") -> argparse.ArgumentParser:
    """Returns an argparser with the shared CLI flags every task generator
    needs. Task generators can subclass / extend it for task-specific flags.
    """
    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--n-scenarios", type=int, default=500,
                    help="Number of scenarios to generate.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tokenizer", type=str,
                    default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--verify", action="store_true",
                    help="Run the task's verify() callback on each emitted "
                         "question. Slows generation; useful for catching "
                         "answer-derivation bugs.")
    ap.add_argument("--pilot-print", type=int, default=0,
                    help="Print this many sample (passages + question) "
                         "tuples to stdout for visual inspection. Doesn't "
                         "affect output files.")
    return ap


# ── Seeding ──────────────────────────────────────────────────────────


def rng_for_scenario(master_seed: int, scenario_idx: int) -> random.Random:
    """Derive a deterministic per-scenario rng. Lets us reproduce or
    parallelize individual scenarios without re-running the whole job."""
    return random.Random(master_seed * 1_000_003 + scenario_idx)


# ── Driver ────────────────────────────────────────────────────────────


def generate_task(gen: TaskGenerator, args: argparse.Namespace) -> dict[str, int]:
    """Run a task generator end-to-end. Returns counts dict.

    Side effects: writes args.output_dir/passages.jsonl + questions.jsonl.
    """
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── Config-space report ────────────────────────────────────────
    try:
        config_n = gen.config_space_size()
        surface_n = gen.surface_variant_count()
        total = config_n * surface_n
        print(f"[{gen.task_family}] coverage analysis:")
        print(f"  Config space:    {config_n:,}")
        print(f"  Surface variants per config: {surface_n:,}")
        print(f"  Total unique examples possible: {total:,}")
        if total < args.n_scenarios * 100:
            print(f"  ⚠️  WARNING: emitting {args.n_scenarios} from a space "
                  f"of {total:,} — risk of duplicates / memorization. "
                  f"Want ≥ 100× margin.", file=sys.stderr)
    except Exception as e:
        print(f"[{gen.task_family}] config-space reporting failed: {e}",
              file=sys.stderr)

    # ── Generation loop with streaming emit ─────────────────────────
    p_path = args.output_dir / "passages.jsonl"
    q_path = args.output_dir / "questions.jsonl"

    seen_passage_ids: set[str] = set()
    seen_question_ids: set[str] = set()
    n_passages = 0
    n_questions = 0
    n_verified = 0
    n_pilot_printed = 0

    with p_path.open("w") as p_out, q_path.open("w") as q_out:
        for scen_idx in range(args.n_scenarios):
            rng = rng_for_scenario(args.seed, scen_idx)
            scen = gen.build_scenario(rng, scen_idx, **gen.build_kwargs)
            if scen is None:
                continue   # generator skipped this scenario

            # Drain passages.
            scenario_passage_ids: list[str] = []
            for p_draft in gen.render_passages(scen, rng):
                if p_draft.passage_id in seen_passage_ids:
                    raise ValueError(
                        f"duplicate passage_id from {gen.task_family}: "
                        f"{p_draft.passage_id}"
                    )
                seen_passage_ids.add(p_draft.passage_id)
                token_ids = _tokenize(p_draft.text, tokenizer)
                row = {
                    "task_family": gen.task_family,
                    "passage_id": p_draft.passage_id,
                    "passage_type": p_draft.passage_type,
                    "passage": p_draft.text,
                    "passage_token_ids": token_ids,
                    "passage_token_count": len(token_ids),
                }
                row.update(p_draft.extras)
                p_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                scenario_passage_ids.append(p_draft.passage_id)
                n_passages += 1

            # Drain questions.
            scenario_questions: list[QuestionDraft] = []
            for q_draft in gen.enumerate_questions(scen, rng):
                if q_draft.question_id in seen_question_ids:
                    raise ValueError(
                        f"duplicate question_id from {gen.task_family}: "
                        f"{q_draft.question_id}"
                    )
                seen_question_ids.add(q_draft.question_id)
                # Validate evidence_keys exist.
                for ek in q_draft.evidence_keys:
                    if ek not in seen_passage_ids:
                        raise ValueError(
                            f"{gen.task_family} question {q_draft.question_id} "
                            f"references unknown passage_id: {ek}"
                        )
                # Optional answer verification.
                if args.verify and gen.verify is not None:
                    if not gen.verify(scen, q_draft):
                        raise AssertionError(
                            f"{gen.task_family} verify() failed for question "
                            f"{q_draft.question_id}: answer doesn't follow "
                            f"from scenario state."
                        )
                    n_verified += 1
                q_ids = _tokenize(q_draft.question_text, tokenizer)
                a_ids = _tokenize(q_draft.answer_text, tokenizer)
                row = {
                    "task_family": gen.task_family,
                    "question_id": q_draft.question_id,
                    "question_type": q_draft.question_type,
                    "evidence_keys": q_draft.evidence_keys,
                    "question": q_draft.question_text,
                    "answer": q_draft.answer_text,
                    "target_value": q_draft.target_value,
                    "question_token_ids": q_ids,
                    "answer_token_ids": a_ids,
                    "question_token_count": len(q_ids),
                    "answer_token_count": len(a_ids),
                }
                row.update(q_draft.extras)
                q_out.write(json.dumps(row, ensure_ascii=False) + "\n")
                scenario_questions.append(q_draft)
                n_questions += 1

            # Pilot-print sample (visual coherence check).
            if n_pilot_printed < args.pilot_print and scenario_questions:
                _pilot_print_scenario(
                    scen_idx, scenario_passage_ids, scenario_questions[0],
                )
                n_pilot_printed += 1

    print(f"[{gen.task_family}] wrote {n_passages} passages + {n_questions} "
          f"questions to {args.output_dir}")
    if args.verify:
        print(f"[{gen.task_family}] verified {n_verified} questions ✓")
    return {"passages": n_passages, "questions": n_questions}


def _pilot_print_scenario(
    scen_idx: int, passage_ids: list[str], example_q: QuestionDraft,
) -> None:
    """Pretty-print one scenario's question + evidence for visual inspection."""
    print(f"\n[pilot scenario {scen_idx}]")
    print(f"  Q ({example_q.question_type}): {example_q.question_text}")
    print(f"  A: {example_q.answer_text}")
    print(f"  evidence ({len(example_q.evidence_keys)}): {example_q.evidence_keys}")


# ── Tokenization ─────────────────────────────────────────────────────


def _tokenize(text: str, tokenizer) -> list[int]:
    """Encode text with the Llama tokenizer, dropping BOS."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]
