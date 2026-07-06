"""Boxes task — passage + question templates."""

from __future__ import annotations


# Initial state passages. {n}, {item} (with indef article applied externally).
INIT_TEMPLATES = (
    "Box {n} contains {item}.",
    "Inside Box {n} there is {item}.",
    "At the start, Box {n} holds {item}.",
    "Box {n} starts out with {item} inside.",
)

# Add-op templates. {item_a_an}, {n}.
ADD_TEMPLATES = (
    "{item_a_an} is placed into Box {n}.",
    "I add {item_a_an} to Box {n}.",
    "{item_a_an} is added to the contents of Box {n}.",
)

# Remove-op templates. {item}, {n}.
REMOVE_TEMPLATES = (
    "The {item} is taken out of Box {n}.",
    "I remove the {item} from Box {n}.",
    "The {item} is removed from Box {n}.",
)

# Move-op templates. {item}, {src}, {dst}.
MOVE_TEMPLATES = (
    "The {item} is moved from Box {src} to Box {dst}.",
    "I take the {item} out of Box {src} and put it in Box {dst}.",
    "The {item} is transferred from Box {src} to Box {dst}.",
)

QUESTION_TEMPLATES = (
    "What is in Box {n} now?",
    "What does Box {n} currently contain?",
    "What is inside Box {n} after all of these changes?",
)
ANSWER_TEMPLATES_NONEMPTY = (
    "Box {n} now contains {contents}.",
    "After all the operations, Box {n} holds {contents}.",
)
ANSWER_TEMPLATES_EMPTY = (
    "Box {n} is now empty.",
    "After all the operations, Box {n} has nothing inside.",
)


def surface_variant_count() -> int:
    """Approximate surface variants per scenario.

    Per scenario: each init passage has ~4 templates; each op has ~3
    templates; question + answer have ~3 + ~2. With ~4 init + ~3 ops:
    """
    init_var = len(INIT_TEMPLATES) ** 4
    op_var = (len(ADD_TEMPLATES) + len(REMOVE_TEMPLATES) + len(MOVE_TEMPLATES)) ** 3
    qa_var = len(QUESTION_TEMPLATES) * (
        len(ANSWER_TEMPLATES_NONEMPTY) + len(ANSWER_TEMPLATES_EMPTY)
    )
    return init_var * op_var * qa_var
