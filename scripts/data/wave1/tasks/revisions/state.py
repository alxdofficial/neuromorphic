"""Revisions task — scenario data model: project + attribute revision logs."""

from __future__ import annotations

from dataclasses import dataclass, field


# Each attribute is (label, possible_values).
ATTRIBUTES = (
    ("release date", ("Friday", "Monday", "Wednesday", "next Tuesday",
                      "the 15th", "the 22nd", "the end of the month",
                      "next quarter")),
    ("project lead", ("Sarah", "Tom", "Maria", "Felix", "Anya", "Jonas",
                      "Hilde", "Kai")),
    ("scope", ("just the API changes", "the full migration", "only the backend",
               "the frontend + tests", "the entire pipeline", "MVP only",
               "complete rewrite")),
    ("budget", ("$10K", "$25K", "$50K", "$100K", "$200K", "$500K")),
    ("deployment region", ("us-east", "us-west", "eu-west", "ap-south",
                           "all three coasts", "north america only")),
    ("review reviewer", ("Sarah", "Tom", "Maria", "Felix", "Anya", "Jonas",
                         "Hilde", "Kai")),
    ("status", ("on track", "delayed", "blocked", "ready for review",
                "in QA", "ready to ship")),
    ("approval status", ("pending", "approved", "rejected", "under review",
                         "approved with revisions")),
)

PROJECT_NAMES = (
    "the migration", "the redesign", "the launch", "the rollout",
    "the refactor", "the audit", "the v3 release", "the hardening pass",
    "the integration", "the cleanup", "the consolidation",
)


@dataclass
class AttributeLog:
    """Time-ordered list of values for one attribute."""
    label: str
    history: list[str]   # history[0] is original; history[-1] is current


@dataclass
class RevisionsScenario:
    scenario_idx: int
    project_name: str
    attributes: list[AttributeLog]   # one per tracked attribute
    # Render order: emit ALL initial values first, then ALL revisions in order.
    # passage_emit_plan[i] = ("init"|"revise", attr_idx, version_idx)
    passage_emit_plan: list[tuple] = field(default_factory=list)


def build_scenario(
    rng, scenario_idx: int,
    *, n_attrs_range=(2, 5), max_revisions_per_attr: int = 2,
) -> RevisionsScenario:
    project_name = rng.choice(PROJECT_NAMES)
    n_attrs = rng.randint(*n_attrs_range)
    n_attrs = min(n_attrs, len(ATTRIBUTES))
    chosen = rng.sample(ATTRIBUTES, n_attrs)
    attrs: list[AttributeLog] = []
    for label, values in chosen:
        n_revisions = rng.randint(0, max_revisions_per_attr)
        # Build value history of length n_revisions + 1 (initial + revisions),
        # with each consecutive value different.
        history = [rng.choice(values)]
        for _ in range(n_revisions):
            alts = [v for v in values if v != history[-1]]
            if not alts:
                break
            history.append(rng.choice(alts))
        attrs.append(AttributeLog(label=label, history=history))

    # Emit plan: all initials first, then revisions interleaved ACROSS attributes
    # but in version order WITHIN each attribute. We draw revisions one at a
    # time from a multinomial over attributes that still have pending versions.
    plan: list[tuple] = []
    for i, a in enumerate(attrs):
        plan.append(("init", i, 0))
    next_v = [1] * len(attrs)        # next pending version per attribute
    n_remaining = [max(0, len(a.history) - 1) for a in attrs]
    while sum(n_remaining) > 0:
        # Weighted pick across attributes with revisions still to emit.
        pool = [i for i, r in enumerate(n_remaining) if r > 0]
        i = rng.choice(pool)
        plan.append(("revise", i, next_v[i]))
        next_v[i] += 1
        n_remaining[i] -= 1

    return RevisionsScenario(
        scenario_idx=scenario_idx,
        project_name=project_name,
        attributes=attrs,
        passage_emit_plan=plan,
    )


def config_space_size() -> int:
    """8 attrs × 8 values each × 3 revisions × ~10 project names.

    For 3 attrs picked from 8, each with ~6 values and 1-2 revisions:
    C(8,3) × 6^3 (initials) × 6^3 (revs) ≈ 56 × 216 × 216 ≈ 2.6M
    × 11 project names ≈ 29M.
    """
    return 29_000_000
