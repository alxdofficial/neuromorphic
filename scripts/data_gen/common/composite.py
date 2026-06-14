"""Composite dataset orchestrator.

Loads one or more task-family JSONLs, validates them, and emits a
unified composite dataset with proper task_family tags and globally-
unique passage_id values.

CLI:
    python -m scripts.data_gen.common.composite \\
        --task-dir biographical:data/wave1/biographical \\
        --task-dir calendar:data/wave1/calendar \\
        --task-dir boxes:data/wave1/boxes \\
        --output-dir data/wave1/composite/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from scripts.data_gen.common.schema import validate_passages, validate_questions


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--task-dir", action="append", required=True,
        metavar="NAME:PATH",
        help="Task name and JSONL dir. Repeat for each task family. "
             "Each dir must contain passages.jsonl and questions.jsonl.",
    )
    ap.add_argument("--output-dir", type=Path, required=True)
    return ap.parse_args()


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_passages: list[dict] = []
    all_questions: list[dict] = []

    for spec in args.task_dir:
        if ":" not in spec:
            raise ValueError(f"--task-dir expects NAME:PATH, got {spec!r}")
        name, path = spec.split(":", 1)
        path = Path(path)
        p_path = path / "passages.jsonl"
        q_path = path / "questions.jsonl"
        if not p_path.exists():
            raise FileNotFoundError(p_path)
        if not q_path.exists():
            raise FileNotFoundError(q_path)

        passages = [json.loads(l) for l in p_path.read_text().splitlines() if l.strip()]
        questions = [json.loads(l) for l in q_path.read_text().splitlines() if l.strip()]

        # Tag with task_family (in case the generator forgot or to override).
        for p in passages:
            p.setdefault("task_family", name)
        for q in questions:
            q.setdefault("task_family", name)

        validate_passages(passages)
        validate_questions(questions, passage_ids={p["passage_id"] for p in passages})

        all_passages.extend(passages)
        all_questions.extend(questions)
        print(f"  [{name}] {len(passages)} passages, {len(questions)} questions")

    # Cross-family validation: global passage_id uniqueness.
    validate_passages(all_passages)
    validate_questions(
        all_questions, passage_ids={p["passage_id"] for p in all_passages},
    )

    # Emit.
    p_out = args.output_dir / "passages.jsonl"
    q_out = args.output_dir / "questions.jsonl"
    with p_out.open("w") as f:
        for p in all_passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    with q_out.open("w") as f:
        for q in all_questions:
            f.write(json.dumps(q, ensure_ascii=False) + "\n")

    # Summary.
    print(f"\nComposite dataset written to {args.output_dir}/")
    print(f"  passages.jsonl: {len(all_passages)} rows")
    print(f"  questions.jsonl: {len(all_questions)} rows")
    print(f"\nQuestion distribution by task_family:")
    for fam, c in Counter(q["task_family"] for q in all_questions).most_common():
        types = Counter(q["question_type"] for q in all_questions if q["task_family"] == fam)
        print(f"  {fam}: {c}")
        for t, ct in types.most_common():
            print(f"    {t}: {ct}")


if __name__ == "__main__":
    main()
