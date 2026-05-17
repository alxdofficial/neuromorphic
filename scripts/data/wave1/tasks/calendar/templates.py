"""Calendar task — passage + question templates."""

from __future__ import annotations


# Passage templates. {day}, {time_str}, {event_a_an}, {dur_phrase}, {duration}.
PASSAGE_TEMPLATES = (
    "On {day} at {time_str} I have {event_a_an}. It runs for {dur_phrase}.",
    "There is {event_a_an} on my calendar for {day} {time_str}, lasting {dur_phrase}.",
    "I scheduled {event_a_an} for {day} at {time_str}; it should take about {dur_phrase}.",
    "My calendar shows {event_a_an} on {day} starting at {time_str} ({duration}-hour block).",
)

# Per-question-type counts for surface_variant_count calculation:
# Each question type has 1-2 phrasings; answers vary too. Conservative est.
QUESTION_TYPE_VARIANTS = {
    "free_at": 2,
    "conflict_with": 2,
    "busy_count_on": 2,
    "next_event_on": 2,
}


def surface_variant_count() -> int:
    """Per scenario: ~4 passage templates per event × ~2 question variants."""
    per_event = len(PASSAGE_TEMPLATES)
    n_events = 6
    qa = sum(QUESTION_TYPE_VARIANTS.values())
    return per_event ** n_events * qa
