"""Boxes task — scenario data model.

A scenario is a list of boxes (each with initial contents) + an ordered
list of operations (add/remove/move) that mutate the boxes. The model
must integrate ops in order to determine the final state.
"""

from __future__ import annotations

from dataclasses import dataclass, field


ITEMS = (
    "apple", "pen", "book", "key", "coin", "ribbon",
    "feather", "stone", "candle", "thimble", "thread", "shell",
    "compass", "marble", "ring", "card", "leaf", "seed",
    "button", "needle", "stamp", "spoon", "fork", "matchbox",
)


@dataclass
class BoxOp:
    """One mutation operation on the box state."""
    op_type: str                    # "add" | "remove" | "move"
    item: str
    src_box: int | None = None      # for remove + move
    dst_box: int | None = None      # for add + move


@dataclass
class BoxesScenario:
    scenario_idx: int
    n_boxes: int
    initial_items: list[str]        # initial_items[i] is the item in box i+1
    ops: list[BoxOp]                # ordered mutation log
    # Derived (computed in build_scenario for verification + rendering):
    final_state: list[list[str]] = field(default_factory=list)
    per_box_touched_ops: list[list[int]] = field(default_factory=list)


def _apply_op(state: list[list[str]], op: BoxOp) -> bool:
    """Mutate state in place. Returns True if op was valid + applied."""
    if op.op_type == "add":
        assert op.dst_box is not None
        state[op.dst_box - 1].append(op.item)
        return True
    if op.op_type == "remove":
        assert op.src_box is not None
        b = state[op.src_box - 1]
        if op.item in b:
            b.remove(op.item)
            return True
        return False
    if op.op_type == "move":
        assert op.src_box is not None and op.dst_box is not None
        b = state[op.src_box - 1]
        if op.item in b:
            b.remove(op.item)
            state[op.dst_box - 1].append(op.item)
            return True
        return False
    return False


def build_scenario(
    rng, scenario_idx: int, *, n_boxes_range=(3, 5), max_ops: int = 5,
) -> BoxesScenario:
    """Build one boxes scenario: pick n_boxes, distinct initial items,
    then n_ops random mutation ops."""
    n_boxes = rng.randint(*n_boxes_range)
    if n_boxes > len(ITEMS):
        n_boxes = len(ITEMS)
    initial = rng.sample(ITEMS, n_boxes)
    state = [[item] for item in initial]
    # per_box_touched_ops[b] = list of op indices that touched box b+1.
    # Initial state passages each touch one box and come before any op.
    # We'll track ops indices starting from n_boxes (initial-state passages 0..n_boxes-1).
    per_box_touched_ops: list[list[int]] = [[i] for i in range(n_boxes)]
    ops: list[BoxOp] = []
    n_ops = rng.randint(1, max_ops)
    op_passage_idx = n_boxes
    for _ in range(n_ops):
        op_type = rng.choice(["add", "remove", "move"])
        if op_type == "add":
            b = rng.randrange(n_boxes)
            # Avoid adding an item that's already in the destination box —
            # would render as "add X to a box that already has X", which is
            # awkward and creates ambiguous resolution later.
            existing = set(state[b])
            allowed = [it for it in ITEMS if it not in existing]
            if not allowed:
                continue
            new_item = rng.choice(allowed)
            op = BoxOp(op_type="add", item=new_item, dst_box=b + 1)
            if _apply_op(state, op):
                ops.append(op)
                per_box_touched_ops[b].append(op_passage_idx)
                op_passage_idx += 1
        elif op_type == "remove":
            non_empty = [i for i, c in enumerate(state) if c]
            if not non_empty:
                continue
            b = rng.choice(non_empty)
            item = rng.choice(state[b])
            op = BoxOp(op_type="remove", item=item, src_box=b + 1)
            if _apply_op(state, op):
                ops.append(op)
                per_box_touched_ops[b].append(op_passage_idx)
                op_passage_idx += 1
        else:  # move
            non_empty = [i for i, c in enumerate(state) if c]
            if not non_empty:
                continue
            src = rng.choice(non_empty)
            possible_dst = [i for i in range(n_boxes) if i != src]
            if not possible_dst:
                continue
            dst = rng.choice(possible_dst)
            item = rng.choice(state[src])
            op = BoxOp(op_type="move", item=item, src_box=src + 1, dst_box=dst + 1)
            if _apply_op(state, op):
                ops.append(op)
                per_box_touched_ops[src].append(op_passage_idx)
                per_box_touched_ops[dst].append(op_passage_idx)
                op_passage_idx += 1
    return BoxesScenario(
        scenario_idx=scenario_idx,
        n_boxes=n_boxes,
        initial_items=initial,
        ops=ops,
        final_state=state,
        per_box_touched_ops=per_box_touched_ops,
    )


def config_space_size() -> int:
    """Approximate count of distinct scenarios.

    Per scenario: choose n_boxes ∈ {3..5}, initial items (perm of |ITEMS|
    choose n_boxes), then n_ops ∈ {1..5} with op type ∈ {3} × item ∈
    |ITEMS| × box ∈ n_boxes. Rough product."""
    # Lower bound for n_boxes=4, n_ops=3.
    n_items = len(ITEMS)
    n_boxes = 4
    n_initial = 1
    for k in range(n_boxes):
        n_initial *= n_items - k
    op_choices = 3 * n_items * n_boxes * n_boxes  # over-counts: many invalid
    n_op_seq = op_choices ** 3
    return n_initial * n_op_seq
