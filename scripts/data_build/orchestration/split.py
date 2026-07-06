"""Train/val split with entity-key disjoint guarantee + surface-name
disjoint guarantee.

For v5 we have two new concerns over v4:
1. Cross-entity edges: a question's evidence_keys may span multiple
   entities. The split must place ALL of a question's evidence in
   either train OR val (not split across).
2. Surface-name leakage: still need union-find on surface_names so
   that "Maria" appearing in two different entities (e.g. as a creator
   of a work AND as a person) doesn't leak.

Algorithm:
1. Build a union-find over entity keys connected by:
   (a) shared surface names (any name in surface_names overlap)
   (b) being co-evidence in any single question
2. Each union-find component becomes a "split atom" — all entities in
   it go to the same split.
3. Sample components by total entity count until val reaches the target
   fraction.

API:
    python scripts/data_gen_v5/split.py \
        --entities data/wave1_v5/entities.jsonl \
        --questions data/wave1_v5/questions.jsonl \
        --val-fraction 0.10 \
        --output-dir data/wave1_v5/
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entities", type=Path, required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--val-fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="defaults to entities/questions parent dir")
    return ap.parse_args()


class UnionFind:
    def __init__(self):
        self.parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
            return x
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression
        while self.parent[x] != root:
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    args = parse_args()
    out_dir = args.output_dir or args.entities.parent

    # Load entities, track surface names
    entities = [json.loads(l) for l in args.entities.read_text().splitlines() if l.strip()]
    questions = [json.loads(l) for l in args.questions.read_text().splitlines() if l.strip()]

    uf = UnionFind()
    for ent in entities:
        uf.find(ent["entity_key"])

    # ── Union ONLY on shared surface names ───────────────────────────
    # (We deliberately do NOT union on co-evidence-in-questions — that
    # turns the densely-connected entity graph into one giant component
    # since most entities cross-reference. Instead, the split partitions
    # by entity_key and drops questions whose evidence spans the split.
    # This means a fraction of questions are unusable, but the split
    # is meaningful entity-key-disjoint.)
    name_to_keys: dict[str, list[str]] = defaultdict(list)
    for ent in entities:
        for n in ent["surface_names"]:
            name_to_keys[n].append(ent["entity_key"])
    for keys in name_to_keys.values():
        for k in keys[1:]:
            uf.union(keys[0], k)

    # Components — usually each entity becomes its own component since
    # name overlap between different entities is rare.
    comps: dict[str, list[str]] = defaultdict(list)
    for ent in entities:
        comps[uf.find(ent["entity_key"])].append(ent["entity_key"])

    # ── Sample components to fill val to target fraction ─────────────
    rng = random.Random(args.seed)
    comp_list = sorted(comps.values(), key=lambda c: -len(c))
    n_total = len(entities)
    val_target = int(n_total * args.val_fraction)

    rng.shuffle(comp_list)
    val_keys: set[str] = set()
    val_size = 0
    for c in comp_list:
        if val_size >= val_target:
            break
        val_keys.update(c)
        val_size += len(c)

    # Partition entities and questions
    train_entities = [e for e in entities if e["entity_key"] not in val_keys]
    val_entities   = [e for e in entities if e["entity_key"] in val_keys]
    train_questions = [
        q for q in questions
        if not any(k in val_keys for k in q["evidence_keys"])
    ]
    val_questions = [
        q for q in questions
        if all(k in val_keys for k in q["evidence_keys"])
    ]

    print(
        f"Split: {len(train_entities)} train + {len(val_entities)} val entities "
        f"({len(val_entities) / n_total:.1%} val); "
        f"{len(train_questions)} train + {len(val_questions)} val questions; "
        f"{len(questions) - len(train_questions) - len(val_questions)} dropped "
        f"(mixed evidence)"
    )

    # Write
    def write_lines(path: Path, items: list[dict]) -> None:
        with path.open("w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

    write_lines(out_dir / "entities_train.jsonl", train_entities)
    write_lines(out_dir / "entities_val.jsonl", val_entities)
    write_lines(out_dir / "questions_train.jsonl", train_questions)
    write_lines(out_dir / "questions_val.jsonl", val_questions)
    print(f"Wrote 4 files to {out_dir}")


if __name__ == "__main__":
    main()
