"""Shared text helpers used across task generators.

Pure functions, no state. Each task family imports what it needs.
"""

from __future__ import annotations


# ── Indefinite article ───────────────────────────────────────────────


def with_indef(noun_phrase: str) -> str:
    """Prepend 'a' or 'an' to a noun phrase based on first letter."""
    if not noun_phrase:
        return noun_phrase
    first = noun_phrase[0].lower()
    article = "an" if first in "aeiou" else "a"
    return f"{article} {noun_phrase}"


# ── Number formatting ────────────────────────────────────────────────


_COUNT_WORDS = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten",
)


def count_word(n: int) -> str:
    """Render small integers as English words ('three'); fall back to digits."""
    if 0 <= n <= 10:
        return _COUNT_WORDS[n]
    return str(n)


_UNITS = ("zero", "one", "two", "three", "four",
          "five", "six", "seven", "eight", "nine")
_TEENS = ("ten", "eleven", "twelve", "thirteen", "fourteen",
          "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = ("", "", "twenty", "thirty", "forty",
         "fifty", "sixty", "seventy", "eighty", "ninety")


def _two_digit_word(n: int) -> str:
    if 0 <= n <= 9:
        return f"oh-{_UNITS[n]}"
    if 10 <= n <= 19:
        return _TEENS[n - 10]
    tens, ones = divmod(n, 10)
    if ones == 0:
        return _TENS[tens]
    return f"{_TENS[tens]}-{_UNITS[ones]}"


def year_as_words(y: int) -> str:
    """Render a year as words. 2017 → 'twenty-seventeen', 1981 → 'nineteen-eighty-one'."""
    if 2000 <= y <= 2009:
        ones = y - 2000
        return "two thousand" if ones == 0 else f"two thousand and {_UNITS[ones]}"
    if y >= 2010:
        return f"twenty-{_two_digit_word(y - 2000)}"
    if y >= 1900:
        return f"nineteen-{_two_digit_word(y - 1900)}"
    if y >= 1800:
        return f"eighteen-{_two_digit_word(y - 1800)}"
    return str(y)


# ── Hour formatting ──────────────────────────────────────────────────


def hour_12h(h: int) -> str:
    """Format an hour 0-23 as '5am' / '12pm' / '11pm'."""
    if h == 0:
        return "12am"
    if h == 12:
        return "12pm"
    if h < 12:
        return f"{h}am"
    return f"{h - 12}pm"


def duration_phrase(d: int) -> str:
    """'1 hour' / '2 hours' / etc."""
    return "1 hour" if d == 1 else f"{d} hours"


# ── List rendering ───────────────────────────────────────────────────


def indef_list(items: list[str]) -> str:
    """Render a list of items as English: 'X', 'X and Y', 'X, Y, and Z'.
    Each item gets an indefinite article."""
    if not items:
        return "nothing"
    if len(items) == 1:
        return with_indef(items[0])
    pieces = [with_indef(x) for x in items]
    if len(pieces) == 2:
        return f"{pieces[0]} and {pieces[1]}"
    return ", ".join(pieces[:-1]) + ", and " + pieces[-1]


def comma_list(items: list[str]) -> str:
    """Render items as 'X', 'X and Y', 'X, Y, and Z' (no indefinite articles)."""
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + ", and " + items[-1]


# ── Capitalization ───────────────────────────────────────────────────


def cap_first(text: str) -> str:
    """Capitalize the first character of text (handles lowercase entity
    names like 'the Treaty of Halsa' that need 'The Treaty...' at
    sentence start)."""
    if not text:
        return text
    return text[0].upper() + text[1:]
