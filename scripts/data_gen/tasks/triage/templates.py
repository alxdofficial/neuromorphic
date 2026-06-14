"""Triage task — templates.

Arrival templates introduce a new request. Update templates mark a
request as done/deprioritized.
"""

from __future__ import annotations


# Arrival templates with no deps. {label}, {priority}, {description}.
ARRIVAL_NO_DEP_TEMPLATES = (
    "Request {label} ({priority}) is to {description}. No prerequisites.",
    "{label}: please {description}. Priority is {priority}; no dependencies.",
    "New ticket {label} (priority {priority}): {description}. Can start anytime.",
)

# Arrival templates with deps. {label}, {priority}, {description}, {deps_phrase}.
ARRIVAL_DEP_TEMPLATES = (
    "Request {label} ({priority}) is to {description}. Depends on {deps_phrase}.",
    "{label}: please {description}. Priority is {priority}; this requires {deps_phrase} to be done first.",
    "New ticket {label} (priority {priority}): {description}. Blocked until {deps_phrase} is complete.",
)

# Update — done. {label}.
DONE_UPDATE_TEMPLATES = (
    "Update: Request {label} is now done.",
    "Status update: {label} has been completed.",
    "Closing ticket {label} — it's finished.",
)

# Update — deprioritized. {label}.
DEPRIORITIZE_UPDATE_TEMPLATES = (
    "Update: Request {label} is being deprioritized — skip for now.",
    "Status update: {label} is being parked; we'll come back to it later.",
    "Deprioritizing ticket {label}; no work on it for now.",
)


# Question + answer templates per question type.
WHAT_UNBLOCKED_Q = (
    "Which requests can be worked on right now?",
    "What's ready to start?",
)
WHAT_BLOCKS_Q = (
    "What's blocking request {label}?",
    "Why can't I start {label} yet?",
)
IS_READY_Q = (
    "Can request {label} be started now?",
    "Is {label} ready to go?",
)
NEXT_PRIORITY_Q = (
    "Of the requests I can work on now, which is the highest priority?",
    "What's the most urgent thing I can pick up right now?",
)


def surface_variant_count() -> int:
    """4 question types × ~2 phrasings each × ~3 arrival templates × ~3 update templates."""
    return (len(ARRIVAL_NO_DEP_TEMPLATES) * len(ARRIVAL_DEP_TEMPLATES)
            * len(DONE_UPDATE_TEMPLATES) * len(DEPRIORITIZE_UPDATE_TEMPLATES)
            * (len(WHAT_UNBLOCKED_Q) + len(WHAT_BLOCKS_Q)
               + len(IS_READY_Q) + len(NEXT_PRIORITY_Q)))
