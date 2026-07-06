"""Knights & Knaves — scenario data model.

A scenario:
- N characters (3-5) with random names
- Each has a secret role: knight or knave
- Each makes one statement about another character (or themselves)
- All 2^N truth-value assignments are enumerated; we keep only scenarios
  where exactly ONE assignment is consistent (i.e., the puzzle has a
  unique solution).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product


CHARACTER_NAMES = (
    "Anya", "Bram", "Cora", "Dax", "Elin", "Fenn", "Gita", "Hilde",
    "Iggy", "Jora", "Kael", "Lia", "Milo", "Nia", "Oren", "Pia",
)

# Each statement type is a function (subject_idx, object_idx) -> claim_text +
# evaluator that returns True if the claim is true under a given role
# assignment.
# Roles: True = knight (truth-teller), False = knave (liar).

STATEMENT_TYPES = (
    "X_is_knight",    # "{Y} is a knight"
    "X_is_knave",     # "{Y} is a knave"
    "X_is_same_as_Y", # "{Y} and I are the same type"
    "X_is_different", # "{Y} and I are different types"
)


@dataclass
class Statement:
    speaker_idx: int       # 0..N-1
    target_idx: int        # 0..N-1 (may equal speaker)
    stmt_type: str         # one of STATEMENT_TYPES
    # The claim's truth-value under role assignment:
    # (filled by check_satisfies at evaluation time).


@dataclass
class KnightsScenario:
    scenario_idx: int
    n_chars: int
    names: list[str]                  # names[i] = character i's name
    roles: list[bool]                 # roles[i] = True (knight) / False (knave)
    statements: list[Statement]


def _evaluate_claim(stmt: Statement, roles: tuple[bool, ...]) -> bool:
    """Truth value of the claim, given the actual role assignment."""
    t = stmt.target_idx
    s = stmt.speaker_idx
    if stmt.stmt_type == "X_is_knight":
        return roles[t]
    if stmt.stmt_type == "X_is_knave":
        return not roles[t]
    if stmt.stmt_type == "X_is_same_as_Y":
        return roles[s] == roles[t]
    if stmt.stmt_type == "X_is_different":
        return roles[s] != roles[t]
    raise ValueError(f"unknown stmt_type {stmt.stmt_type}")


def _consistent(roles: tuple[bool, ...], statements: list[Statement]) -> bool:
    """A role assignment is consistent iff: for every statement, the speaker's
    role matches whether their claim is actually true."""
    for stmt in statements:
        claim_true = _evaluate_claim(stmt, roles)
        # Knights only tell truth; knaves only lie.
        if roles[stmt.speaker_idx] != claim_true:
            return False
    return True


def _count_consistent(statements: list[Statement], n: int) -> tuple[int, tuple[bool, ...] | None]:
    """Count how many role assignments satisfy all statements. Returns
    (count, the_unique_assignment_or_None)."""
    consistent_count = 0
    sole_assignment = None
    for assignment in product([True, False], repeat=n):
        if _consistent(assignment, statements):
            consistent_count += 1
            sole_assignment = assignment
            if consistent_count > 1:
                return consistent_count, None
    return consistent_count, sole_assignment


def build_scenario(
    rng, scenario_idx: int, *, n_chars_range=(3, 4),
) -> KnightsScenario | None:
    """Build one K&K scenario with a unique solution. May return None if
    no valid scenario found in max_attempts."""
    n_chars = rng.randint(*n_chars_range)
    if n_chars > len(CHARACTER_NAMES):
        n_chars = len(CHARACTER_NAMES)
    names = rng.sample(CHARACTER_NAMES, n_chars)

    # Try several random statement sets; accept the first with a unique
    # solution. With n=3, ~1/4 of random sets have a unique solution;
    # with n=4 it's higher.
    for _ in range(40):
        statements: list[Statement] = []
        for speaker in range(n_chars):
            stmt_type = rng.choice(STATEMENT_TYPES)
            target = rng.randrange(n_chars)
            # X_is_same_as_Y / X_is_different about oneself are degenerate
            # ("Dax and I are the same type" — trivially true). For X_is_knight
            # / X_is_knave we usually avoid self-claims too (knaves saying
            # "I am a knave" is a paradox; "I am a knight" can't disambiguate).
            if target == speaker:
                if stmt_type in ("X_is_same_as_Y", "X_is_different"):
                    target = (speaker + 1) % n_chars
                elif rng.random() < 0.7:
                    target = (speaker + 1) % n_chars
            statements.append(Statement(speaker_idx=speaker,
                                        target_idx=target,
                                        stmt_type=stmt_type))
        count, sole = _count_consistent(statements, n_chars)
        if count == 1:
            return KnightsScenario(
                scenario_idx=scenario_idx,
                n_chars=n_chars,
                names=names,
                roles=list(sole),
                statements=statements,
            )
    return None


def config_space_size() -> int:
    """For n=3 chars: 16 names choose 3 perms = 16*15*14 = 3,360.
    Statement matrix has n × (n-1) × 4 stmt-types ≈ 24 configurations
    per speaker; over n speakers ≈ 24^3 = 13,824. Total ~46M, only ~25%
    have unique solutions => ~12M.
    For n=4: ~36 × 16*15*14*13 = 1.7M name perms × 36^4 statements ≈ huge."""
    return 12_000_000   # conservative n=3 estimate
