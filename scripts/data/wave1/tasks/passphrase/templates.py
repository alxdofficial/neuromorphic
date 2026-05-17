"""Passphrase task — passage + question templates."""

from __future__ import annotations


# Templates for the passphrase carrier passage. {speaker}, {phrase}.
PASSAGE_TEMPLATES = (
    "{speaker} mentioned in passing today: \"My secret phrase is {phrase}.\" "
    "{speaker} repeated it once to make sure I had it right, then moved on "
    "to other things.",

    "Note from {speaker}: the access code is {phrase}. {speaker} asked me "
    "to remember it without writing it down, so I'm committing it to memory.",

    "{speaker}'s personal motto, which {speaker} has been saying for years, "
    "is {phrase}. I overheard it again at dinner.",

    "Today {speaker} shared their working passphrase for the project: "
    "{phrase}. {speaker} chose it carefully and asked me to keep it in mind "
    "for the rest of the conversation.",

    "The codeword {speaker} uses for their personal records is {phrase}. "
    "{speaker} explained the history of why they picked it, but the codeword "
    "itself is the part that matters.",

    "{speaker} dictated a short note: \"Please remember this for me — "
    "{phrase}.\" That's the full content of the note.",
)


# Distractor passage templates — same speaker pool, unrelated content.
DISTRACTOR_TEMPLATES = (
    "{speaker} spent the afternoon reorganizing their bookshelf. Several "
    "volumes that had been miscategorized for years finally found their "
    "proper home, mostly under the new acquisitions category.",

    "An update from {speaker}: the kitchen renovation is on schedule. The "
    "tilework will be finished by the end of the week, and the new cabinets "
    "are expected to arrive next Tuesday.",

    "{speaker} described a long walk along the coastal path this morning. "
    "The fog was thick enough that the lighthouse was barely visible, but "
    "the seabirds were unusually active.",

    "{speaker} mentioned they finally finished reading the regional history "
    "volume that's been on their nightstand for months. Their main "
    "takeaway: nothing in the central chapters is quite as their teachers "
    "claimed.",

    "{speaker} hosted three former colleagues for dinner. The conversation "
    "drifted from current events to old shared projects, and lasted longer "
    "than anyone had planned.",
)


# Question + answer templates. {speaker}, {phrase}.
QUESTION_TEMPLATES = (
    "What was {speaker}'s passphrase?",
    "What phrase did {speaker} share?",
    "What was the codeword {speaker} mentioned?",
    "What did {speaker} ask me to remember?",
)
ANSWER_TEMPLATES = (
    "{speaker}'s passphrase was {phrase}.",
    "{speaker} shared the phrase {phrase}.",
    "The codeword {speaker} mentioned was {phrase}.",
)


def surface_variant_count() -> int:
    """Per-scenario surface variants. Each scenario emits one target passage,
    `n_distractors` distractor passages, one question, and one answer; each is
    drawn from an independent template pool.
    """
    n_distractors = 2   # matches generate.py's default
    return (len(PASSAGE_TEMPLATES)
            * (len(DISTRACTOR_TEMPLATES) ** n_distractors)
            * len(QUESTION_TEMPLATES)
            * len(ANSWER_TEMPLATES))
