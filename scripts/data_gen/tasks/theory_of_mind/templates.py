"""Theory of Mind — narration + Q/A templates."""

from __future__ import annotations


PREAMBLE_TEMPLATES = (
    "{chars_list} are in the room.",
    "{chars_list} are together in the room.",
    "In the room, we have {chars_list}.",
    "The scene: {chars_list} are in the same room.",
)

PLACE_TEMPLATES = (
    "{actor} places the {obj} in the {dst}.",
    "{actor} puts the {obj} into the {dst}.",
    "{actor} sets the {obj} down inside the {dst}.",
)

LEAVE_TEMPLATES = (
    "{actor} leaves the room.",
    "{actor} steps out of the room.",
    "{actor} walks out of the room.",
)

MOVE_TEMPLATES = (
    "While {actor} is alone, {actor} moves the {obj} from the {src} to the {dst}.",
    "{actor} quietly moves the {obj} out of the {src} and into the {dst}.",
    "{actor} takes the {obj} from the {src} and places it in the {dst}.",
)

RETURN_TEMPLATES = (
    "{actor} comes back into the room.",
    "{actor} returns to the room.",
    "{actor} walks back into the room.",
)

# Question + Answer templates.
WHERE_BELIEF_Q = (
    "Where does {char} think the {obj} is?",
    "If asked, where would {char} look for the {obj}?",
)
WHERE_BELIEF_A = (
    "{char} thinks the {obj} is in the {container}.",
    "{char} would look for the {obj} in the {container}.",
)

WHERE_ACTUALLY_Q = (
    "Where is the {obj} actually?",
    "Where is the {obj} right now?",
)
WHERE_ACTUALLY_A = (
    "The {obj} is in the {container}.",
    "Right now, the {obj} is in the {container}.",
)

HAS_SEEN_Q = (
    "Did {char} see the {obj} being moved?",
    "Was {char} present when the {obj} was moved?",
)
HAS_SEEN_A_YES = (
    "Yes, {char} saw the {obj} being moved.",
    "Yes, {char} was in the room when the {obj} was moved.",
)
HAS_SEEN_A_NO = (
    "No, {char} did not see the {obj} being moved.",
    "No, {char} was out of the room when the {obj} was moved.",
)


def surface_variant_count() -> int:
    # Per scenario: 1 preamble × 1 place × 1 leave × 1 move × 1 return,
    # each rendered with independent template choices.
    narration_variants = (len(PREAMBLE_TEMPLATES) * len(PLACE_TEMPLATES)
                          * len(LEAVE_TEMPLATES) * len(MOVE_TEMPLATES)
                          * len(RETURN_TEMPLATES))
    qa_variants = (len(WHERE_BELIEF_Q) * len(WHERE_BELIEF_A)
                   + len(WHERE_ACTUALLY_Q) * len(WHERE_ACTUALLY_A)
                   + len(HAS_SEEN_Q) * (len(HAS_SEEN_A_YES) + len(HAS_SEEN_A_NO)) / 2)
    return int(narration_variants * qa_variants)
