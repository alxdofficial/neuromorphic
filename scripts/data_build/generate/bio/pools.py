"""Static value pools — reused from v4 + v5-specific additions.

Strategy: rather than duplicate ~1500 lines of name/occupation/skill pools,
this module re-exports the v4 pools (which are pure static data) and adds
the few new pools v5 needs (place metadata, work-type registry).

v4 pools live in:
    scripts/data_build/wave1_worldspec.py        (people, places, occupations, ...)
    scripts/data_build/wave1_worldspec_extra.py  (orgs, nations, events, works,
                                            relationships, preferences)

If v4 is ever deleted, copy the relevant blocks into this file.
"""

from __future__ import annotations

# ── re-export v4 pools ──────────────────────────────────────────────

from scripts.data_build.orchestration.wave1_worldspec import (
    FIRST_NAMES_F,
    LAST_NAMES,
    TOWNS,                      # list of (name, descriptor, region) tuples
    CITIES,
    NEIGHBORHOODS,
    NATURAL_FEATURES,
    OCCUPATIONS,
    SIGNATURE_SKILLS,           # dict[occupation -> list of skill strings]
    HOBBIES,                    # list of (gerund_phrase, narrative_blurb) tuples
    RECURRING_HABITS,
    ALMA_MATERS,
    MENTOR_INSTITUTIONS,
    PARTNER_OCCUPATIONS,
    FAMILY_BACKGROUNDS,
    HOMETOWN_CENTRAL_FEATURES,
    PARENT_DECADES_PASSED,
    DECADES_STARTED,
    OCCUPATION_DOMAIN,          # dict[occupation -> domain str]
    WORKPLACES_BY_DOMAIN,
    ALMA_MATERS_BY_DOMAIN,
    MENTOR_INSTITUTIONS_BY_DOMAIN,
    PUBLIC_FIGURE_FIELDS,
    SIGNATURE_WORKS,
    PF_INSTITUTIONS,
    PF_ALMA_MATERS,
    FAMOUS_AWARDS,
    LIFE_EVENT_TYPES,
    LIFE_EVENT_LOCATIONS,
    LIFE_EVENT_OUTCOMES,
)

from scripts.data_build.orchestration.wave1_worldspec_extra import (
    FIRST_NAMES_M,
    ALL_FIRST_NAMES,
    ORG_NAME_PREFIXES,
    ORG_NAME_STEMS,
    ORG_NAME_SUFFIXES,
    ORG_ACTIVITIES,
    ORG_MILESTONES,
    NATION_NAMES,
    NATION_CAPITAL_NAMES,
    NATION_LANGUAGES,
    NATION_LEADER_TITLES,
    HE_TYPE_TEMPLATES,
    HE_INSTITUTIONS,
    HE_OUTCOMES,
    HE_PRIMARY_FIGURE_TITLES,
    CW_TYPE_TITLES,             # dict[work_type -> list of title fragments]
    CW_GENRES,
    CW_SUBJECTS,
    CW_RECEPTION,
    REL_TYPE_CONTEXTS,          # dict[rel_type -> list of meeting venues]
    REL_TYPES,
    PREFERENCE_TYPES_VALUES,
    PREFERENCE_ORIGIN_CONTEXTS,
)


# ── v5-specific additions ──────────────────────────────────────────

# Persona registry: how a passage is "voiced" (third-person bio, letter,
# wiki entry, etc.). Each entity is rendered in one persona per sample.
PERSONAS: tuple[str, ...] = (
    "biographical_paragraph",
    "letter",
    "wiki_entry",
    "news_article",
    "journal_entry",
    "archival_note",
)

# Birth-year range for procedurally-generated people.
PERSON_BIRTH_YEAR_RANGE = (1940, 2000)

# Founding-year range for organizations and nations.
ORG_FOUNDING_YEAR_RANGE = (1850, 2015)
NATION_FOUNDING_YEAR_RANGE = (1800, 1970)

# Year range for events (historical + life events).
EVENT_YEAR_RANGE = (1880, 2024)

# Year range for cultural works.
WORK_YEAR_RANGE = (1900, 2024)

# Adjective fragments for person passages — bridge attribute mentions.
PERSON_PASSAGE_TRANSITIONS = (
    "She is reserved by temperament but generous to younger colleagues",
    "He is known for his quiet methodical approach",
    "She is unfailingly punctual and famously private",
    "He is admired for the breadth of his correspondence",
    "She maintains a calm bearing that her students remark on",
    "He has a long-standing reputation for plain speaking",
    "She is described by colleagues as steady and exacting",
    "He is widely regarded as one of the more thoughtful figures of his generation",
)

# Surface-form variants for years (for slot-substitution diversity).
def year_as_words(y: int) -> str:
    """Render an integer year as natural-language words.
    Used by the slot-substitution layer for variety."""
    # Simple mapping for years in our range (1800-2024).
    if 2000 <= y <= 2009:
        decade = "two thousand"
        ones = y - 2000
        if ones == 0:
            return decade
        return f"{decade} and {_units_word(ones)}"
    if y >= 2010:
        return f"twenty-{_two_digit_word(y - 2000)}"
    if y >= 1900:
        return f"nineteen-{_two_digit_word(y - 1900)}"
    if y >= 1800:
        return f"eighteen-{_two_digit_word(y - 1800)}"
    return str(y)


_UNITS = ("zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine")
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty",
         "fifty", "sixty", "seventy", "eighty", "ninety")


def _units_word(n: int) -> str:
    if 0 <= n <= 9:
        return _UNITS[n]
    return str(n)


def _two_digit_word(n: int) -> str:
    if 0 <= n <= 9:
        return f"oh-{_UNITS[n]}"
    if 10 <= n <= 19:
        return _TEENS[n - 10]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_UNITS[ones]}"
