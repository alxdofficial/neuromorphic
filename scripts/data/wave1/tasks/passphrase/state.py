"""Passphrase task — scenario data model.

A scenario is a single (speaker, phrase) tuple plus a chosen template
for rendering. The "scenario" is intentionally minimal: passphrase is
the simplest possible memory task.
"""

from __future__ import annotations

from dataclasses import dataclass


SPEAKERS = (
    "Maria", "Iver", "Lina", "Hjordis", "Brage", "Solveig",
    "Anders", "Tilde", "Frode", "Aslaug", "Camilla", "Reidar",
)

WORD_POOL = (
    "north", "south", "river", "anchor", "moss", "stone",
    "winter", "summer", "harbor", "thistle", "amber", "iron",
    "lantern", "echo", "ember", "crow", "swan", "linen",
    "chestnut", "fennel", "marrow", "pewter", "wren", "fjord",
    "compass", "vellum", "fern", "kettle", "lichen", "marsh",
)


@dataclass
class PassphraseScenario:
    """One passphrase carrier + a few distractors that share the same speaker
    pool. The carrier is the source of the question; distractors fill the
    chunk if the trainer's sampler doesn't have enough other-family
    distractors.
    """
    scenario_idx: int
    speaker: str
    phrase: str
    distractor_speakers: list[str]   # speakers used in the N distractor passages


def build_scenario(rng, scenario_idx: int, *, n_distractors: int = 2) -> PassphraseScenario:
    """Pick a speaker, generate a 3-word + 2-digit phrase, pick N distractor speakers."""
    speaker = rng.choice(SPEAKERS)
    words = rng.sample(WORD_POOL, 3)
    n = rng.randint(10, 99)
    phrase = f"{words[0]}-{words[1]}-{words[2]} {n}"
    distractor_speakers = [rng.choice(SPEAKERS) for _ in range(n_distractors)]
    return PassphraseScenario(
        scenario_idx=scenario_idx,
        speaker=speaker,
        phrase=phrase,
        distractor_speakers=distractor_speakers,
    )


def config_space_size() -> int:
    """Distinct (speaker, 3-word-combo, 2-digit-number) tuples."""
    n_speakers = len(SPEAKERS)
    n_word_combos = len(WORD_POOL) * (len(WORD_POOL) - 1) * (len(WORD_POOL) - 2)
    n_numbers = 90  # 10..99
    return n_speakers * n_word_combos * n_numbers
