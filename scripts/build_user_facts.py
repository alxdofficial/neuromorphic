#!/usr/bin/env python3
"""Expand `user_facts.yaml` into `expanded.json` for Wave 3 passphrase training.

For each fact, generates via Claude API:
- 3 paraphrases of the fact (different surface forms, same meaning)
- 5 questions someone might ask to elicit it (varied phrasing /
  directness — explicitly NOT cookie-cutter templates)
- 3 reference answers (different surface forms a good assistant would
  emit)

Writes JSON to ``data/passphrase/expanded.json``. Idempotent — skips
facts already present in the output file.

Note: the older ``scripts/build_passphrase_data.py`` (v2 era) does
something different — it assembles full token-shard prompts. This
script only expands the fact list; prompt assembly happens online
in ``src/data/passphrase_loader.py`` so the curriculum on filler_mid
length can be controlled per training step.

Usage (real expansion via Anthropic API):
    export ANTHROPIC_API_KEY=...
    PYTHONPATH=. .venv/bin/python scripts/build_user_facts.py \\
        --input data/passphrase/user_facts.yaml \\
        --output data/passphrase/expanded.json

Stub mode (no API call — generates trivial expansions for smoke tests):
    PYTHONPATH=. .venv/bin/python scripts/build_user_facts.py \\
        --input data/passphrase/user_facts.yaml \\
        --output data/passphrase/expanded_stub.json \\
        --stub --limit 10

Cost estimate: ~$2 USD for the full 150-fact expansion at Claude
Sonnet 4.6 prices (~150 calls × ~1500 input + 800 output tokens).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml


SYSTEM_PROMPT = """\
You are a helpful assistant generating training data for a memory-augmented language model. \
You will be given a single first-person fact about a fictional user. Generate paraphrases, \
questions, and reference answers in JSON format.

Rules:
- Paraphrases must be in FIRST PERSON, same meaning, different surface form. \
Vary sentence structure naturally.
- Questions must vary in phrasing, directness, and style. Some concrete, some abstract, \
ideally one negation-style. NOT cookie-cutter Q&A templates.
- Reference answers should be 1-2 sentences in third person ("The user prefers X..."), \
varying surface form across the three.
- Output JSON ONLY, no prose, no markdown fences.

Output schema:
{
  "paraphrases": ["...", "...", "..."],
  "questions": ["...", "...", "...", "...", "..."],
  "reference_answers": ["...", "...", "..."]
}
"""

USER_TEMPLATE = """\
Fact: {fact}

Generate the JSON expansion."""


def _stub_expand(fact: str) -> dict[str, Any]:
    """Trivial deterministic expansion for smoke tests (no API needed).

    Produces minimal viable paraphrases / questions / answers so the
    loader pipeline can be exercised end-to-end without an API key.
    Quality is intentionally low — meant only for wiring tests.
    """
    return {
        "paraphrases": [
            fact,
            fact.replace("I ", "Personally I ", 1) if fact.startswith("I ") else fact,
            "(p) " + fact,
        ],
        "questions": [
            "Tell me about something the user does or prefers.",
            "What is one of the user's preferences?",
            "Can you describe a personal habit of the user?",
            "What do you know about the user?",
            "Describe one of the user's quirks.",
        ],
        "reference_answers": [
            f"The user mentions: {fact}",
            f"As stated: {fact}",
            f"Based on context: {fact}",
        ],
    }


def _claude_expand(fact: str, *, model: str, max_retries: int = 3) -> dict[str, Any]:
    """Call Anthropic API to expand a fact. Retries on transient errors."""
    import anthropic
    client = anthropic.Anthropic()

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": USER_TEMPLATE.format(fact=fact),
                }],
            )
            text = resp.content[0].text.strip()
            # Strip optional code fences.
            if text.startswith("```"):
                text = text.strip("`")
                first_nl = text.find("\n")
                if first_nl > 0:
                    text = text[first_nl + 1:]
                if text.endswith("```"):
                    text = text[:-3]
            data = json.loads(text)
            assert isinstance(data.get("paraphrases"), list) and len(data["paraphrases"]) >= 1
            assert isinstance(data.get("questions"), list) and len(data["questions"]) >= 1
            assert isinstance(data.get("reference_answers"), list) and len(data["reference_answers"]) >= 1
            return data
        except (json.JSONDecodeError, AssertionError, Exception) as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(1 + attempt * 2)
                continue
            raise RuntimeError(f"failed to expand fact after {max_retries} retries: {e}") from e
    raise RuntimeError(f"unreachable: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/passphrase/user_facts.yaml")
    ap.add_argument("--output", default="data/passphrase/expanded.json")
    ap.add_argument("--stub", action="store_true",
                    help="Use trivial deterministic stubs instead of Claude API.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only expand the first N facts (for quick test).")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                    help="Anthropic model name.")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with in_path.open() as f:
        facts = yaml.safe_load(f)
    if args.limit is not None:
        facts = facts[:args.limit]
    print(f"[input] {len(facts)} facts from {in_path}")

    # Resume-friendly: load existing expanded.json if any.
    expanded: dict[int, dict] = {}
    if out_path.exists():
        with out_path.open() as f:
            existing = json.load(f)
        for entry in existing:
            expanded[entry["id"]] = entry
        print(f"[resume] {len(expanded)} facts already expanded; skipping those.")

    n_new = 0
    for i, fact_entry in enumerate(facts):
        fid = fact_entry["id"]
        if fid in expanded:
            continue
        fact_text = fact_entry["fact"]
        if args.stub:
            exp = _stub_expand(fact_text)
        else:
            exp = _claude_expand(fact_text, model=args.model)
        expanded[fid] = {
            "id": fid,
            "topic": fact_entry.get("topic", "misc"),
            "fact": fact_text,
            **exp,
        }
        n_new += 1
        if n_new % 10 == 0 or i + 1 == len(facts):
            print(f"  [{i+1}/{len(facts)}] expanded {n_new} new facts (total {len(expanded)})", flush=True)
            with out_path.open("w") as f:
                json.dump(sorted(expanded.values(), key=lambda e: e["id"]), f, indent=2)

    with out_path.open("w") as f:
        json.dump(sorted(expanded.values(), key=lambda e: e["id"]), f, indent=2)
    print(f"[done] wrote {len(expanded)} entries to {out_path}")


if __name__ == "__main__":
    main()
