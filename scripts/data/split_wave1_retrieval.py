#!/usr/bin/env python3
"""Split the wave1_retrieval JSONL into train + val.

Name-disjoint split: groups entities by overlapping `entity_names`
(union-find) and assigns whole groups to train or val. This ensures no
surface person-name appears in both splits — preventing leakage where
e.g. a `private_individual` named "Maria Halverson" in train shares
the surface form with a `personal_relationship` participant named
"Maria Halverson" in val.

For entities without person-names (Organizations, Nations, Historical
Events, Cultural Works — where `entity_names` is empty), each entity
is its own group and assigned independently.

Run:
    python3 scripts/data/split_wave1_retrieval.py \\
        --in data/wave1_retrieval/facts_v6.jsonl \\
        --out-train data/wave1_retrieval/facts_train.jsonl \\
        --out-val data/wave1_retrieval/facts_val.jsonl \\
        --val-frac 0.1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(line) for line in Path(args.in_path).read_text().splitlines() if line.strip()]
    print(f"Read {len(rows)} facts from {args.in_path}")

    # Build entity_key → entity_names (sets) and group facts by entity.
    by_entity = defaultdict(list)
    entity_names: dict[str, set[str]] = {}
    for r in rows:
        ek = r["entity_key"]
        by_entity[ek].append(r)
        if ek not in entity_names:
            entity_names[ek] = set(r.get("entity_names") or [])
    print(f"Distinct entities: {len(by_entity)}")

    # ── Name-disjoint groups via union-find ──
    parent = {ek: ek for ek in by_entity}

    def find(ek):
        while parent[ek] != ek:
            parent[ek] = parent[parent[ek]]
            ek = parent[ek]
        return ek

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    # For every shared person-name, union all entities that mention it.
    name_to_entities: dict[str, list[str]] = defaultdict(list)
    for ek, names in entity_names.items():
        for n in names:
            name_to_entities[n].append(ek)
    n_overlap = 0
    for n, eks in name_to_entities.items():
        if len(eks) > 1:
            n_overlap += 1
            for i in range(1, len(eks)):
                union(eks[0], eks[i])
    print(f"Names appearing in 2+ entities: {n_overlap}")

    # Materialize groups.
    groups: dict[str, list[str]] = defaultdict(list)
    for ek in by_entity:
        groups[find(ek)].append(ek)
    group_list = sorted(groups.values(), key=lambda g: g[0])
    print(f"Name-disjoint groups: {len(group_list)} "
          f"(group-size distribution: min={min(len(g) for g in group_list)}, "
          f"max={max(len(g) for g in group_list)}, "
          f"median={sorted(len(g) for g in group_list)[len(group_list)//2]})")

    # ── Split groups: shuffle, accumulate into val until target reached ──
    rng = random.Random(args.seed)
    rng.shuffle(group_list)
    n_val_target = max(1, int(len(by_entity) * args.val_frac))
    val_keys: set[str] = set()
    val_entity_count = 0
    for grp in group_list:
        if val_entity_count >= n_val_target:
            break
        if val_entity_count + len(grp) <= n_val_target * 1.2:
            # accept this group into val (allow up to 20% overshoot for atomicity)
            val_keys.update(grp)
            val_entity_count += len(grp)
    train_keys = set(by_entity.keys()) - val_keys
    print(f"Split: train={len(train_keys)} entities, val={len(val_keys)} entities")

    # ── Verify name-disjointness ──
    train_names: set[str] = set()
    val_names: set[str] = set()
    for ek in train_keys:
        train_names.update(entity_names[ek])
    for ek in val_keys:
        val_names.update(entity_names[ek])
    overlap_names = train_names & val_names
    if overlap_names:
        print(f"  WARNING: {len(overlap_names)} names overlap between train and val: "
              f"{sorted(overlap_names)[:5]}")
    else:
        print(f"  ✓ Name-disjoint: 0 surface names overlap between train and val")

    # ── Write out ──
    train_rows, val_rows = [], []
    train_classes: dict[str, int] = defaultdict(int)
    val_classes: dict[str, int] = defaultdict(int)
    for r in rows:
        if r["entity_key"] in val_keys:
            val_rows.append(r)
            val_classes[r["entity_class"]] += 1
        else:
            train_rows.append(r)
            train_classes[r["entity_class"]] += 1

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out_train).open("w") as f:
        for r in train_rows:
            f.write(json.dumps(r) + "\n")
    with Path(args.out_val).open("w") as f:
        for r in val_rows:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(train_rows)} train facts to {args.out_train}")
    print(f"  by class: {dict(train_classes)}")
    print(f"Wrote {len(val_rows)} val facts to {args.out_val}")
    print(f"  by class: {dict(val_classes)}")


if __name__ == "__main__":
    main()
