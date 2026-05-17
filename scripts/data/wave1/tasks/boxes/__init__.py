"""Boxes task: state-tracking via sequential mutations.

Adapted from Kim & Schuster 2023's entity-tracking benchmark. Each
"scenario" has a set of boxes, each starting with some contents. A
sequence of operations (add/move/remove/swap) mutates the boxes; the
question asks the final state of one box.

The architecture has to:
1. Read the initial state passages
2. Apply each op to the appropriate box
3. Answer based on the final state — not the initial

This stresses **state mutation under interference**, the exact pattern
that scatter_mean writes in our manifold are designed for.

Question types:
- final_state: "what's in box X now?"
"""
