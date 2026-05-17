"""Revisions task — templates."""

from __future__ import annotations


# Initial statement templates. {project}, {attr}, {value}.
INITIAL_TEMPLATES = (
    "For {project}, the {attr} is {value}.",
    "The {attr} for {project} is currently set to {value}.",
    "We've decided that for {project}, the {attr} will be {value}.",
    "Note: the {attr} on {project} is {value}.",
)

# Revision templates. {project}, {attr}, {old_value}, {new_value}.
# Constraint: every position that interpolates {old_value} or {new_value} must
# work whether the value already begins with "the" (e.g. "the end of the
# month") or not (e.g. "Friday"). Phrasings that would prepend "the " before
# the slot have been avoided.
REVISION_TEMPLATES = (
    "Update on {project}: the {attr} is no longer {old_value} — it's now {new_value}.",
    "Correction for {project}: scratch that — the {attr} is actually {new_value}, not {old_value}.",
    "Revision: on {project}, the {attr} has changed from {old_value} to {new_value}.",
    "Actually, the {attr} for {project} is being changed to {new_value} (was {old_value}).",
    "Update: on {project}, the {attr} is now {new_value}; the previous value ({old_value}) no longer applies.",
)

# Question + answer templates.
CURRENT_VALUE_Q = (
    "What is the current {attr} for {project}?",
    "What's the latest {attr} on {project}?",
    "After all the updates, what is the {attr} for {project}?",
)
CURRENT_VALUE_A = (
    "The {attr} for {project} is {value}.",
    "Currently, {project}'s {attr} is {value}.",
)

REVISION_COUNT_Q = (
    "How many times has the {attr} on {project} been revised?",
    "How many updates have been made to the {attr} for {project}?",
)
REVISION_COUNT_A_ZERO = (
    "The {attr} on {project} hasn't been revised since the initial value.",
)
REVISION_COUNT_A_ONE = (
    "The {attr} on {project} has been revised once.",
)
REVISION_COUNT_A_MANY = (
    "The {attr} on {project} has been revised {n} times.",
)


def surface_variant_count() -> int:
    return (len(INITIAL_TEMPLATES) ** 3 * len(REVISION_TEMPLATES) ** 2
            * (len(CURRENT_VALUE_Q) * len(CURRENT_VALUE_A)
               + len(REVISION_COUNT_Q) * 3))
