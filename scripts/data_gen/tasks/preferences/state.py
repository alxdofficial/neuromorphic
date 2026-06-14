"""Preferences task — scenario data model.

Each user has a few preferences across distinct domains, optionally with
one revision (cancellation/update). The final preference per domain is
either the originally-stated value or its revision.
"""

from __future__ import annotations

from dataclasses import dataclass, field


USER_NAMES = (
    "Maria", "Iver", "Lina", "Hjordis", "Brage", "Solveig",
    "Anders", "Tilde", "Frode", "Aslaug", "Camilla", "Reidar",
    "Sigrid", "Karin", "Olaf", "Inge", "Vera", "Sara",
)

# (domain_label, possible_values).
PREFERENCE_DOMAINS = (
    ("coffee", ("black", "with milk and sugar", "decaf", "iced", "espresso")),
    ("seating preference on flights", ("aisle", "window", "exit row", "anywhere quiet")),
    ("sleeping temperature", ("cool", "warm", "very cool", "moderate")),
    ("preferred email tone", ("formal", "casual", "concise", "warm and friendly")),
    ("notification style", ("only urgent ones", "all of them", "summarized daily", "muted on weekends")),
    ("preferred meeting time", ("mornings", "afternoons", "late afternoons", "never before 10am")),
    ("workout time of day", ("early morning", "lunchtime", "evening", "right after work")),
    ("food spice level", ("mild", "medium", "spicy", "very spicy")),
    ("reading format", ("physical books", "e-books", "audiobooks", "either physical or e-books")),
    ("preferred travel pace", ("slow and unhurried", "packed full of activities", "moderate", "spontaneous and flexible")),
    ("music genre", ("classical", "jazz", "folk", "ambient", "rock")),
    ("preferred check-in format", ("written summary", "voice call", "video call", "in-person")),
)


@dataclass
class PreferenceStatement:
    domain: str
    initial_value: str
    cancelled: bool = False
    revised_value: str | None = None    # set if cancelled=True


@dataclass
class PreferencesScenario:
    scenario_idx: int
    user_name: str
    preferences: list[PreferenceStatement]


def build_scenario(
    rng, scenario_idx: int,
    *, n_prefs_range=(3, 5), cancellation_rate: float = 0.3,
) -> PreferencesScenario:
    n_prefs = rng.randint(*n_prefs_range)
    n_prefs = min(n_prefs, len(PREFERENCE_DOMAINS))
    chosen = rng.sample(PREFERENCE_DOMAINS, n_prefs)
    user_name = rng.choice(USER_NAMES)
    prefs: list[PreferenceStatement] = []
    for domain, values in chosen:
        initial = rng.choice(values)
        prefs.append(PreferenceStatement(domain=domain, initial_value=initial))
    # Maybe cancel one.
    if rng.random() < cancellation_rate and prefs:
        idx = rng.randrange(len(prefs))
        domain = prefs[idx].domain
        values = next(v for (d, v) in PREFERENCE_DOMAINS if d == domain)
        alternatives = [v for v in values if v != prefs[idx].initial_value]
        if alternatives:
            prefs[idx].cancelled = True
            prefs[idx].revised_value = rng.choice(alternatives)
    return PreferencesScenario(
        scenario_idx=scenario_idx,
        user_name=user_name,
        preferences=prefs,
    )


def config_space_size() -> int:
    """Per-user: pick avg 3 domains from 12 and a value for each."""
    n_users = len(USER_NAMES)
    n_domains = len(PREFERENCE_DOMAINS)
    # Picking 3 domains from 12 = C(12, 3) = 220.
    # Each domain has ~4 possible values.
    domain_value_count = 1
    avg_values = sum(len(vs) for _, vs in PREFERENCE_DOMAINS) // len(PREFERENCE_DOMAINS)
    for _ in range(3):
        domain_value_count *= avg_values
    # ×2 for cancellation present/absent, ×4 for revised value alternatives.
    return n_users * 220 * domain_value_count * 2 * 4
