"""Knights & Knaves — passage + question templates."""

from __future__ import annotations


# Statement passage templates — render one character's statement.
# Slots: {speaker}, {target}.
STATEMENT_TEMPLATES = {
    "X_is_knight": (
        "{speaker} says: \"{target} is a knight.\"",
        "{speaker} claims that {target} is a knight.",
        "{speaker}: \"{target} tells the truth.\"",
    ),
    "X_is_knave": (
        "{speaker} says: \"{target} is a knave.\"",
        "{speaker} claims that {target} is a knave.",
        "{speaker}: \"{target} is a liar.\"",
    ),
    "X_is_same_as_Y": (
        "{speaker} says: \"{target} and I are the same type.\"",
        "{speaker} claims that they and {target} are both the same type.",
    ),
    "X_is_different": (
        "{speaker} says: \"{target} and I are different types.\"",
        "{speaker} claims that they and {target} are different types.",
    ),
}

# Context preamble — one passage per scenario explaining the puzzle premise.
PREAMBLE_TEMPLATES = (
    "On a small island, every inhabitant is either a knight (who always tells "
    "the truth) or a knave (who always lies). Today the locals are talking.",
    "In a certain village, residents are either knights (always truthful) or "
    "knaves (always lying). The following overheard claims were made.",
    "Each person here is either a knight (truthful) or a knave (lying). "
    "What follows is everything they said in the order it was said.",
)

QUESTION_TEMPLATES = (
    "Is {target} a knight or a knave?",
    "Which is {target}: knight or knave?",
)
ANSWER_TEMPLATES_KNIGHT = (
    "{target} is a knight.",
    "{target} is a knight (truth-teller).",
)
ANSWER_TEMPLATES_KNAVE = (
    "{target} is a knave.",
    "{target} is a knave (liar).",
)


def surface_variant_count() -> int:
    avg_stmt_var = sum(len(v) for v in STATEMENT_TEMPLATES.values()) // len(STATEMENT_TEMPLATES)
    n_stmts = 3  # statements per scenario (one per character, avg)
    return (len(PREAMBLE_TEMPLATES) * (avg_stmt_var ** n_stmts)
            * len(QUESTION_TEMPLATES)
            * (len(ANSWER_TEMPLATES_KNIGHT) + len(ANSWER_TEMPLATES_KNAVE)))
