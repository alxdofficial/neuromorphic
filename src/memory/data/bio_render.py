"""Templates for biographical conditioned reconstruction (``conditioned_reconstruction_bio``).

A **KEY** is a short identifying phrase for a world entity: its name plus *at
most one* identity disambiguator (e.g. ``"Hanne Rudh, born 1971"``). A
**VALUE** is a *fact-dense* natural sentence packing several of that entity's
*other* random attributes, voiced through a persona for diversity
(e.g. ``"Hanne Rudh, an ornithologist, recognized for the survey of inland
heron populations, and a graduate of Bergvik Conservation Association."``).

Strict conditioned-reconstruction discipline (so SHUF is a hard control):
- The key's disambiguator attribute is **excluded** from the value, so the key
  never co-encodes a value fact. A wrong-entity (shuffled) memory therefore
  cannot reconstruct the sentence.
- The packed value facts are independent random draws from the world builder →
  unguessable from the key string alone.

This is **not** QA: there is no question/answer pair, no reasoning — the value
is the entity's facts stated verbatim, to be reproduced from memory conditioned
on the key. Reuses only the biographical world/pools (no QA generators).

Runtime render templates for `bio.py`; world/pools come from
`scripts/data_build/generate/bio/`. See DATASETS.md.
"""
from __future__ import annotations

import re
import random
from typing import Any

# 6 structural skeletons (mirrors the bio passage personas, but for one packed sentence).
PERSONAS = ("plain", "wiki", "news", "journal", "letter", "archival")

_VOWELS = set("aeiouAEIOU")
# 'a' (not 'an') before consonant-sound vowels: university, European, one-, ewe, unit…
_A_BEFORE = ("uni", "use", "uni", "eu", "one", "ewe", "u.")

# ── gender consistency: the source pools (FAMILY_BACKGROUNDS, some RECURRING_HABITS)
# carry gendered language but Person has no gender attr — infer it from the given
# name and drop any fact whose gendered language conflicts, so we never render
# "Sverre, the daughter of …".
try:
    from scripts.data_build.generate.bio import pools as _pools
    _FEM_NAMES = set(_pools.FIRST_NAMES_F)
    _MASC_NAMES = set(_pools.FIRST_NAMES_M)
except Exception:                                # standalone import w/o worldspec
    _FEM_NAMES, _MASC_NAMES = set(), set()

_FEM_MARK = re.compile(r"\b(daughter|her|hers|she|sister|mother)\b", re.I)
_MASC_MARK = re.compile(r"\b(son|his|him|he|brother|father)\b", re.I)


def _name_gender(ent) -> str:
    g = ent.attrs.get("given_name")
    if g in _FEM_NAMES:
        return "f"
    if g in _MASC_NAMES:
        return "m"
    return "?"


def _fact_gender(val: Any) -> str | None:
    s = str(val)
    f, m = bool(_FEM_MARK.search(s)), bool(_MASC_MARK.search(s))
    if f and not m:
        return "f"
    if m and not f:
        return "m"
    return None                                  # neutral or mixed → always safe


# ── pure surface helpers (copied, not imported — keep deps to world/pools) ──

def _humanize(v: Any) -> str:
    return str(v).replace("_", " ").strip()


def _art(phrase: str) -> str:
    """'a'/'an' for a bare noun phrase."""
    p = phrase.lstrip().lower()
    if p.startswith(_A_BEFORE):                  # 'a university', 'a European …'
        return "a"
    return "an" if p[:1] in "aeiou" else "a"


def _starts_article(v: str) -> bool:
    return v.lower().startswith(("the ", "a ", "an "))


def _name_derivable(short: str, name: str) -> bool:
    """True if `short` is a prefix/substring of the entity name (key-derivable). The org
    short_name is usually the name minus its last word, so the equality-only guard missed
    it, leaking an 'also known as X' clause (sweep #5)."""
    s, n = str(short).strip(), str(name or "").strip()
    return bool(s) and (s == n or n.startswith(s) or s in n or n in s)


def _join(parts: list[str]) -> str:
    parts = [p for p in parts if p]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + ", and " + parts[-1]


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def _yr(v: Any, rng: random.Random, year_as_words) -> str:
    try:
        y = int(v)
    except (TypeError, ValueError):
        return _humanize(v)
    return year_as_words(y) if rng.random() < 0.5 else str(y)


def _decade(v: Any) -> str:
    s = _humanize(v)
    return f"{s}s" if s and s[-1].isdigit() else s


# ── per-attribute clause fragments (apposition-friendly: read after "Name, …") ──
# Each entry: list of templates; {v}=humanized value, {yr}=year form, {a}=article.

# attrs whose clause renders a YEAR / DECADE surface form (≠ the raw value string), so the
# load-bearing value substring is `yr` / `dec`, not `v`. Everything else's value substring is `v`.
_YEAR_ATTRS = {"birth_year", "award_year", "founding_year", "year", "year_released"}
_DECADE_ATTRS = {"founding_decade", "decade", "release_decade"}


def _clause(attr: str, raw: Any, rng: random.Random, year_as_words) -> tuple[str, str] | None:
    """Return ``(phrase, value_substr)``: the apposition fragment and the exact fact-value
    substring inside it (the humanized value / year / decade form — the only span a reader
    must recover). The connectives around it ('recognized for', 'a graduate of') are
    key-independent scaffolding; bio.py scores loss only on ``value_substr`` (value-span mask)."""
    v = _humanize(raw)
    yr = _yr(raw, rng, year_as_words)
    a = _art(v)
    dec = _decade(raw)
    table: dict[str, list[str]] = {
        # person — private
        "occupation":        [f"{a} {v}", f"by profession {a} {v}", f"who works as {a} {v}"],
        "birth_year":        [f"born in {yr}", f"born {yr}"],
        "signature_skill":   [f"recognized for {v}", f"known for {v}", f"noted for {v}"],
        "hobby_gerund":      [f"who enjoys {v}", f"fond of {v}", f"who spends free time on {v}"],
        "recurring_habit":   [f"in the habit of {v}", f"known for {v}"],
        "alma_mater_name":   [f"a graduate of {v}", f"educated at {v}", f"who studied at {v}"],
        "family_background": [v, f"raised as {v}" if not v.lower().startswith(("the ", "a ", "an ")) else v],
        "domain":            [f"working in {v}", f"active in {v}"],
        # person — public figure
        "primary_field":     [f"a figure in {v}", f"prominent in {v}", f"active in {v}"],
        "signature_work":    [f"best known for {v}", f"remembered for {v}", f"the author of {v}"],
        "famous_award":      [f"a recipient of {v}", f"awarded {v}", f"honored with {v}"],
        "award_year":        [f"honored in {yr}", f"recognized in {yr}"],
        # organization
        "org_type":          [f"{a} {v}", f"established as {a} {v}"],
        "primary_activity":  [f"devoted to {v}", f"engaged in {v}", f"focused on {v}"],
        "founding_year":     [f"founded in {yr}", f"established {yr}"],
        "founding_decade":   [f"founded in the {dec}"],
        "headquarters_city_name": [f"headquartered in {v}", f"based in {v}"],
        "notable_milestone": [f"noted for {v}", f"remembered for {v}"],
        "short_name":        [f"also known as {v}"],
        # nation
        "capital_name":      [f"whose capital is {v}", f"with its capital at {v}"],
        "official_language": [f"where the official language is {v}", f"with {v} as its official language"],
        # place
        "city_descriptor":   [f"{a} {v}", f"described as {a} {v}"],
        "country_name":      [f"in {v}", f"located in {v}"],
        "region":            [f"in the {v} region", f"part of the {v} area"],
        # event
        "year":              [f"which took place in {yr}", f"occurring in {yr}"],
        "decade":            [f"in the {dec}"],
        "century":           [f"in the {v} century"],
        "outcome_descriptor": [f"which brought about {v}", f"remembered for {v}"],
        # work
        "work_type":         [f"{a} {v}", f"composed as {a} {v}"],
        "genre":             [f"in the {v} genre", f"a work of {v}"],
        "main_subject":      [f"about {v}", f"taking {v} as its subject"],
        "reception":         [f"which was {v}", v],
        "year_released":     [f"released in {yr}", f"published {yr}"],
        "release_decade":    [f"released in the {dec}"],
    }
    opts = table.get(attr)
    if not opts:
        return None
    phrase = rng.choice(opts)
    vsub = yr if attr in _YEAR_ATTRS else dec if attr in _DECADE_ATTRS else v
    return phrase, vsub


# ── attribute pools the value can pack (own-entity attrs only) ──
ATTR_POOL: dict[str, list[str]] = {
    "Person": ["occupation", "birth_year", "signature_skill", "hobby_gerund",
               "recurring_habit", "alma_mater_name", "family_background",
               "primary_field", "signature_work", "famous_award", "award_year"],
    "Organization": ["org_type", "primary_activity", "founding_year", "founding_decade",
                     "headquarters_city_name", "notable_milestone", "short_name"],
    "Nation": ["founding_year", "official_language", "capital_name"],
    "Place": ["city_descriptor", "country_name", "region"],
    "Event": ["year", "decade", "century", "outcome_descriptor"],
    "Work": ["work_type", "year_released", "release_decade", "genre", "main_subject", "reception"],
}

# correlated attribute groups — derivable from one another, so they leak each
# other. At most ONE per group may appear across (key ∪ value), and they're
# never split key/value (a decade in the value would reveal a year in the key).
_CORR_GROUPS: list[set[str]] = [
    {"year", "decade", "century"},
    {"founding_year", "founding_decade"},
    {"year_released", "release_decade"},
    {"occupation", "signature_skill"},      # skill is occupation-conditional (people.py)
    {"city_descriptor", "region"},          # place descriptor often restates the region (sweep #4)
]


def _correlated(attr: str) -> set[str]:
    for g in _CORR_GROUPS:
        if attr in g:
            return g - {attr}
    return set()


# identity disambiguators usable in the KEY (excluded from the VALUE when used)
KEY_DISAMB: dict[str, list[str]] = {
    "Person": ["birth_year", "occupation", "primary_field", "alma_mater_name"],
    "Organization": ["org_type", "founding_year", "headquarters_city_name"],
    "Nation": ["capital_name", "official_language"],
    "Place": ["country_name", "region"],
    "Event": ["year", "decade"],
    "Work": ["work_type", "year_released"],
}


def _entity_name(ent) -> str:
    return ent.attrs.get("name") or ent.attrs.get("title") or ent.key


def _disamb_phrase(attr: str, raw: Any, rng: random.Random, year_as_words) -> str:
    """A short ', …' / ' (…)' clause appended to the name in a key."""
    v = _humanize(raw)
    yr = _yr(raw, rng, year_as_words)
    return {
        "birth_year":        f", born {yr}",
        "occupation":        f", the {v}",
        "primary_field":     f", the {v} figure",
        "alma_mater_name":   f" of {v}",
        "org_type":          f" (the {v})",
        "founding_year":     f", founded {yr}",
        "headquarters_city_name": f" of {v}",
        "capital_name":      f", capital {v}",
        "official_language": f" ({v}-speaking)",
        "country_name":      f", {v}",
        "region":            f", {v}",
        "year":              f" of {yr}",
        "decade":            f", {_decade(raw)}",
        "work_type":         f", the {v}",
        "year_released":     f" ({yr})",
    }.get(attr, f", {v}")


def render_key(ent, rng: random.Random, year_as_words) -> tuple[str, set[str]]:
    """Return (key_phrase, excluded_attrs). 60% of keys carry one disambiguator."""
    name = _entity_name(ent)
    cands = [a for a in KEY_DISAMB.get(ent.entity_type, []) if ent.attrs.get(a) is not None]
    if cands and rng.random() < 0.6:
        d = rng.choice(cands)
        return name + _disamb_phrase(d, ent.attrs[d], rng, year_as_words), {d} | _correlated(d)
    return name, set()


def render_value(ent, rng: random.Random, year_as_words, *, n_facts: int = 3,
                 exclude: set[str] = frozenset(), value_out: list | None = None) -> str:
    """One fact-dense apposition sentence packing up to ``n_facts`` random attrs.

    If ``value_out`` is given, it is filled with the exact fact-value substrings actually
    placed into the sentence (in order) — the loss-side value-span mask in bio.py scores only
    those, excluding the entity name and all connective/persona scaffolding."""
    name = _entity_name(ent)
    given = ent.attrs.get("given_name") or name
    if value_out is not None:
        value_out.clear()
    g = _name_gender(ent)
    available = [
        a for a in ATTR_POOL.get(ent.entity_type, [])
        if ent.attrs.get(a) is not None and a not in exclude
        and _fact_gender(ent.attrs[a]) in (None, g)             # gender-consistent facts only
        # name-derivable facts are key-derivable dead weight: short_name (sweep #5), and org/work
        # type words already inside the entity name ('… Association' → 'an association').
        and not (a in ("short_name", "org_type", "work_type")
                 and _name_derivable(ent.attrs[a], ent.attrs.get("name")))
    ]
    # collapse correlated groups → keep one representative so packed facts are independent
    drop: set[str] = set()
    for g in _CORR_GROUPS:
        present = [a for a in available if a in g]
        if len(present) > 1:
            keep = rng.choice(present)
            drop |= (set(present) - {keep})
    available = [a for a in available if a not in drop]
    if not available:
        return f"{name}."
    picked = rng.sample(available, min(n_facts, len(available)))
    rendered = [r for a in picked if (r := _clause(a, ent.attrs[a], rng, year_as_words))]
    clauses = [c for (c, _vs) in rendered]
    if value_out is not None:
        value_out.extend(vs for (_c, vs) in rendered)
    if not clauses:
        return f"{name}."
    persona = rng.choice(PERSONAS)
    body = _join(clauses)
    if persona == "wiki" and len(clauses) >= 2:
        return f"{name} ({clauses[0]}) — {_join(clauses[1:])}."
    if persona == "news":
        return f"In brief — {name}: {body}."   # name-first colon record (no awkward clause opener)
    if persona == "journal":
        return f"Note: {given}, {body}."
    if persona == "letter":
        return f"You'll remember {given} — {body}."
    if persona == "archival":
        return f"{name} [{'; '.join(clauses)}]."
    return f"{name}, {body}."   # plain


if __name__ == "__main__":  # smoke: print keys + values per entity type
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.data_build.generate.bio.state import build_scenario
    from scripts.data_build.generate.bio.pools import year_as_words

    scen = build_scenario(random.Random(11), 0, n_people=30, n_public_figures=10,
                          n_orgs=10, n_nations=8, n_places=10, n_events=10, n_works=10)
    rng = random.Random(3)
    by_type: dict[str, Any] = {}
    for e in scen.world.entities.values():
        by_type.setdefault(e.entity_type, e)
    for et, e in by_type.items():
        print(f"\n### {et}")
        for _ in range(3):
            k, excl = render_key(e, rng, year_as_words)
            v = render_value(e, rng, year_as_words, n_facts=3, exclude=excl)
            print(f"  KEY:   {k}")
            print(f"  VALUE: {v}")
