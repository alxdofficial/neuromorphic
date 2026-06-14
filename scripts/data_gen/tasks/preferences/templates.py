"""Preferences task — templates."""

from __future__ import annotations


STATEMENT_TEMPLATES = (
    "{user} mentioned that they prefer {value} when it comes to {domain}.",
    "{user} said their preference for {domain} is {value}.",
    "{user}: \"For {domain}, I really prefer {value}.\"",
    "{user} noted that {value} is their go-to for {domain}.",
    "When asked about {domain}, {user} said they prefer {value}.",
)

CANCEL_TEMPLATES = (
    "{user} updated me later: \"Actually, never mind what I said about "
    "{domain} — I'd really prefer {new_value} now.\"",
    "{user} called back to clarify: their preference for {domain} has "
    "changed to {new_value}.",
    "Correction from {user}: they no longer want {old_value} for {domain}. "
    "{new_value} is what they actually prefer.",
    "{user} reconsidered and said: \"On reflection, I think {new_value} "
    "is actually better for {domain}, not {old_value}.\"",
)

QUESTION_TEMPLATES = (
    "What does {user} prefer for {domain}?",
    "What is {user}'s preference for {domain}?",
    "How does {user} like their {domain}?",
)
ANSWER_TEMPLATES = (
    "{user} prefers {value} for {domain}.",
    "{user}'s preference for {domain} is {value}.",
)


def surface_variant_count() -> int:
    return (len(STATEMENT_TEMPLATES) ** 3 * len(CANCEL_TEMPLATES)
            * len(QUESTION_TEMPLATES) * len(ANSWER_TEMPLATES))
