"""Template registry — passage personas + question/answer paraphrases.

Three layers of paraphrase (see protocol §5):

Layer 1 — Passage persona: each entity rendered as one of:
    biographical_paragraph | letter | wiki_entry | news_article |
    journal_entry | archival_note

Layer 2 — Per-question template variants (3-5 per question type).

Layer 3 — Slot substitution variations (digit vs word numerals,
    relative-time vs absolute-time, etc.).

Each render function is a pure function of (world, entity_or_question, rng).
"""

from scripts.data.wave1.tasks.biographical.templates.passages import render_passage
from scripts.data.wave1.tasks.biographical.templates.questions_atomic import generate_atomic_questions
from scripts.data.wave1.tasks.biographical.templates.questions_relational import generate_relational_questions
from scripts.data.wave1.tasks.biographical.templates.questions_temporal import generate_temporal_questions
from scripts.data.wave1.tasks.biographical.templates.questions_aggregation import generate_aggregation_questions

__all__ = [
    "render_passage",
    "generate_atomic_questions",
    "generate_relational_questions",
    "generate_temporal_questions",
    "generate_aggregation_questions",
]
