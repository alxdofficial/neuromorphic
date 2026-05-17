"""Passage rendering — one paragraph per (entity, sample_idx).

Each entity gets rendered as a coherent paragraph mentioning:
- Its local attrs (occupation, hometown, hobby, etc.)
- Its 1-hop neighbors by SURFACE NAME (so the cross-reference flows into text)

The paragraph is rendered in one of 6 PERSONAS sampled uniformly per call:
biographical_paragraph, letter, wiki_entry, news_article, journal_entry,
archival_note. Combined with 2 template variants per (persona, entity_type),
each entity has ~12 distinct surface forms before slot substitutions.

API:
    render_passage(world, entity, rng, persona=None) -> (text, persona_used)
"""

from __future__ import annotations

import random
from typing import Any
from scripts.data.wave1.tasks.biographical.world import Entity, World
from scripts.data.wave1.tasks.biographical import pools


PERSONAS = pools.PERSONAS


def render_passage(
    world: World,
    entity: Entity,
    rng: random.Random,
    *,
    persona: str | None = None,
) -> tuple[str, str]:
    """Render `entity` as a paragraph. Returns (text, persona_used)."""
    if persona is None:
        persona = rng.choice(PERSONAS)

    if entity.entity_type == "Person":
        return _render_person(world, entity, rng, persona), persona
    if entity.entity_type == "Organization":
        return _render_org(world, entity, rng, persona), persona
    if entity.entity_type == "Nation":
        return _render_nation(world, entity, rng, persona), persona
    if entity.entity_type == "Place":
        return _render_place(world, entity, rng, persona), persona
    if entity.entity_type == "Event":
        return _render_event(world, entity, rng, persona), persona
    if entity.entity_type == "Work":
        return _render_work(world, entity, rng, persona), persona
    raise ValueError(f"unknown entity_type: {entity.entity_type}")


# ── Helpers ─────────────────────────────────────────────────────────


def _surface(ent: Entity | None, fallback: str = "") -> str:
    """Canonical surface name for an entity, or fallback string if None."""
    if ent is None:
        return fallback
    return ent.attrs.get("name") or ent.attrs.get("title") or fallback


def _neighbor_name(world: World, src_key: str, rel: str, fallback: str = "") -> str:
    """Surface name of the entity reached by `rel` from `src_key`."""
    nbr = world.neighbor(src_key, rel)
    return _surface(nbr, fallback)


def _opt_clause(text: str) -> str:
    """Return text if non-trivial, else empty string."""
    return text if text and not text.endswith(" ") else (text or "")


# ── Person ──────────────────────────────────────────────────────────


def _render_person(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    name = ent.attrs["name"]
    given = ent.attrs.get("given_name", name.split()[0])
    family = ent.attrs.get("family_name", name.split()[-1])
    flavor = ent.attrs.get("flavor", "private")
    birth_year = ent.attrs.get("birth_year")

    # 1-hop neighbor surface names (fall back gracefully if missing).
    mentor_name = _neighbor_name(world, ent.key, "mentor")
    spouse_name = _neighbor_name(world, ent.key, "spouse")
    works_at = world.neighbor(ent.key, "works_at")
    workplace_name = _surface(works_at)
    workplace_city = ""
    if works_at:
        hq = world.neighbor(works_at.key, "headquartered_in")
        workplace_city = _surface(hq)
    alma_mater = world.neighbor(ent.key, "alma_mater")
    alma_mater_name = _surface(alma_mater, ent.attrs.get("alma_mater_name", ""))
    born_in = world.neighbor(ent.key, "born_in")
    birthplace_name = _surface(born_in, "")

    if flavor == "public_figure":
        return _render_public_figure_passage(
            ent, rng, persona,
            workplace_name=workplace_name, workplace_city=workplace_city,
            alma_mater_name=alma_mater_name, birthplace_name=birthplace_name,
        )

    # Private individual.
    occupation = ent.attrs.get("occupation", "professional")
    skill = ent.attrs.get("signature_skill", "specialized work")
    hobby = ent.attrs.get("hobby_gerund", "their hobby")
    habit = ent.attrs.get("recurring_habit", "their daily routine")

    if persona == "biographical_paragraph":
        text = (
            f"{name} is a {occupation}"
            f"{f' at {workplace_name}' if workplace_name else ''}"
            f"{f' in {workplace_city}' if workplace_city else ''}. "
            f"{given} was born in {birth_year}"
            f"{f' in {birthplace_name}' if birthplace_name else ''}, "
            f"and is known for {skill}. "
        )
        if mentor_name:
            text += f"{given} trained under {mentor_name}"
            if alma_mater_name:
                text += f" at {alma_mater_name}. "
            else:
                text += ". "
        elif alma_mater_name:
            text += f"{given} studied at {alma_mater_name}. "
        if spouse_name:
            text += f"{given} lives with {spouse_name}. "
        text += f"{given} is known among colleagues for {hobby} and for {habit}."
        return text.strip()

    if persona == "letter":
        sender = mentor_name if mentor_name else "the editor"
        text = (
            f'I am writing to commend the work of {name}, who has served as '
            f"a {occupation}"
            f"{f' at {workplace_name}' if workplace_name else ''} for many years. "
        )
        if alma_mater_name:
            text += f"{given} trained at {alma_mater_name}"
            if mentor_name:
                text += f" under {mentor_name}"
            text += ". "
        text += (
            f"{given}'s reputation rests on {skill}, "
            f"and on the steady habit of {habit}. "
        )
        if spouse_name:
            text += f"{given} lives with {spouse_name}. "
        text += (
            f"Outside of professional life, {given} is known for {hobby}.\n\n"
            f"Yours sincerely,\n{sender}"
        )
        return text.strip()

    if persona == "wiki_entry":
        text = (
            f"{name} (born {birth_year}"
            f"{f' in {birthplace_name}' if birthplace_name else ''}) "
            f"is a {occupation}"
            f"{f' working at {workplace_name}' if workplace_name else ''}"
            f"{f' in {workplace_city}' if workplace_city else ''}. "
        )
        if alma_mater_name:
            text += f"{family} studied at {alma_mater_name}"
            if mentor_name:
                text += f" under {mentor_name}"
            text += ". "
        text += f"{family} is recognized for {skill}. "
        if spouse_name:
            text += f"{family} is married to {spouse_name}. "
        text += f"{family} is known privately for {hobby} and for {habit}."
        return text.strip()

    if persona == "news_article":
        text = (
            f"In a recent feature, {name}"
            f"{f' — a {occupation} at {workplace_name}' if workplace_name else f' — a {occupation}'}"
            f" — was profiled for {skill}. "
        )
        if mentor_name:
            text += f"{given} trained under {mentor_name}"
            if alma_mater_name:
                text += f" at {alma_mater_name}"
            text += ". "
        elif alma_mater_name:
            text += f"{given} is a graduate of {alma_mater_name}. "
        if spouse_name:
            text += f'{given} lives with {spouse_name}. '
        text += f"Colleagues note {given}'s habit of {habit} and pursuit of {hobby}."
        return text.strip()

    if persona == "journal_entry":
        text = (
            f"Spent the afternoon with {given} again. "
            f"{given} continues as a {occupation}"
            f"{f' at {workplace_name}' if workplace_name else ''}, "
            f"absorbed in {skill}. "
        )
        if mentor_name:
            text += f"We talked again about {mentor_name}, who shaped {given}'s training"
            if alma_mater_name:
                text += f" at {alma_mater_name}"
            text += ". "
        if spouse_name:
            text += f"{spouse_name} stopped by briefly. "
        text += f"As always, {given} found time for {hobby} after work."
        return text.strip()

    # archival_note
    text = (
        f"Subject: {name} ({family}, b. {birth_year})\n"
        f"Profession: {occupation}"
        f"{f' — affiliation {workplace_name}' if workplace_name else ''}"
        f"{f' ({workplace_city})' if workplace_city else ''}\n"
        f"Specialty: {skill}\n"
    )
    if alma_mater_name:
        text += f"Education: {alma_mater_name}"
        if mentor_name:
            text += f" (under {mentor_name})"
        text += "\n"
    if spouse_name:
        text += f"Partner: {spouse_name}\n"
    text += f"Notes: known for {hobby}; routine of {habit}."
    return text.strip()


def _render_public_figure_passage(
    ent: Entity, rng: random.Random, persona: str,
    *, workplace_name: str, workplace_city: str,
    alma_mater_name: str, birthplace_name: str,
) -> str:
    name = ent.attrs["name"]
    given = ent.attrs.get("given_name", name.split()[0])
    family = ent.attrs.get("family_name", name.split()[-1])
    field = ent.attrs.get("primary_field", "scholarship")
    sig_work = ent.attrs.get("signature_work", "an influential contribution")
    award = ent.attrs.get("famous_award", "a major prize")
    award_year = ent.attrs.get("award_year", 2000)
    birth_year = ent.attrs.get("birth_year", 1950)

    if persona == "biographical_paragraph":
        return (
            f"{name} (born {birth_year}"
            f"{f' in {birthplace_name}' if birthplace_name else ''}) "
            f"is a leading figure in {field}, best known for {sig_work}. "
            f"{family} was awarded {award} in {award_year}"
            f"{f' while at {workplace_name}' if workplace_name else ''}. "
            f"{family} studied at {alma_mater_name}." if alma_mater_name else
            f"{name} (born {birth_year}"
            f"{f' in {birthplace_name}' if birthplace_name else ''}) "
            f"is a leading figure in {field}, best known for {sig_work}. "
            f"{family} was awarded {award} in {award_year}."
        )
    if persona == "wiki_entry":
        text = (
            f"{name} (born {birth_year}) is a {field} specialist. "
            f"{family} is known for {sig_work}. "
            f"{family} received {award} in {award_year}"
            f"{f' for work at {workplace_name}' if workplace_name else ''}. "
        )
        if alma_mater_name:
            text += f"{family} is a graduate of {alma_mater_name}. "
        return text.strip()
    if persona == "news_article":
        return (
            f"{name} was again in the news this week for {sig_work}. "
            f"The {field} researcher, who received {award} in {award_year}, "
            f"continues to work at {workplace_name or 'their affiliated institution'}. "
            f"{family} was born in {birth_year}"
            f"{f' in {birthplace_name}' if birthplace_name else ''}."
        )
    if persona == "letter":
        return (
            f"To the editor:\n\nI wish to recognize the contributions of {name} "
            f"to the field of {field}. {family} is best known for {sig_work}, "
            f"a contribution recognized by {award} in {award_year}. "
            f"{family} continues at {workplace_name or 'their institution'}, "
            f"and is a graduate of {alma_mater_name or 'their alma mater'}.\n\n"
            f"Yours,\nA grateful colleague"
        )
    if persona == "journal_entry":
        return (
            f"Read a piece about {given} today. Hard to overstate how "
            f"central {sig_work} has been to {field}. {given} received "
            f"{award} in {award_year}, deservedly. {family} was born in "
            f"{birth_year}, which seems improbable given how much {given} "
            f"has done since."
        )
    # archival_note
    return (
        f"Subject: {name} (b. {birth_year})\n"
        f"Field: {field}\n"
        f"Signature work: {sig_work}\n"
        f"Award: {award} ({award_year})\n"
        f"Affiliation: {workplace_name or 'unaffiliated'}\n"
        f"Education: {alma_mater_name or '—'}"
    )


# ── Organization ────────────────────────────────────────────────────


def _render_org(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    name = ent.attrs["name"]
    org_type = ent.attrs.get("org_type", "organization")
    activity = ent.attrs.get("primary_activity", "general activities")
    founding_year = ent.attrs.get("founding_year", 1900)
    milestone = ent.attrs.get("notable_milestone", "a significant achievement")
    hq_city = _neighbor_name(world, ent.key, "headquartered_in",
                             ent.attrs.get("headquarters_city_name", ""))
    founder = world.neighbor(ent.key, "founded_by")
    founder_name = _surface(founder)
    parent_org = world.neighbor(ent.key, "subsidiary_of")
    parent_name = _surface(parent_org)

    a_org_type = _with_indef(org_type)
    if persona == "biographical_paragraph":
        text = (
            f"{name} is {a_org_type}"
            f"{f' based in {hq_city}' if hq_city else ''}, "
            f"primarily engaged in {activity}. "
            f"The organization was founded in {founding_year}"
            f"{f' by {founder_name}' if founder_name else ''}. "
        )
        if parent_name:
            text += f"It is a subsidiary of {parent_name}. "
        text += f"It is best known for {milestone}."
        return text.strip()
    if persona == "wiki_entry":
        text = (
            f"{name} ({org_type}, founded {founding_year}"
            f"{f', {hq_city}' if hq_city else ''}) "
            f"is engaged in {activity}. "
        )
        if founder_name:
            text += f"It was founded by {founder_name}. "
        if parent_name:
            text += f"It operates as a subsidiary of {parent_name}. "
        text += f"It is recognized for {milestone}."
        return text.strip()
    if persona == "news_article":
        return (
            f"In a recent profile, {name}"
            f"{f' (based in {hq_city})' if hq_city else ''} was featured for "
            f"{milestone}. The {org_type}, founded in {founding_year}"
            f"{f' by {founder_name}' if founder_name else ''}, "
            f"is engaged primarily in {activity}."
            + (f" It operates as a subsidiary of {parent_name}." if parent_name else "")
        )
    if persona == "letter":
        return (
            f"To Whom It May Concern,\n\nI write in connection with {name}, "
            f"the {org_type}"
            f"{f' headquartered in {hq_city}' if hq_city else ''}, "
            f"founded in {founding_year}"
            f"{f' by {founder_name}' if founder_name else ''}. "
            f"The organization is engaged in {activity} and is best known for {milestone}."
            f"{f' It operates as a subsidiary of {parent_name}.' if parent_name else ''}\n\n"
            f"Yours,\nThe Secretary"
        )
    if persona == "journal_entry":
        return (
            f"Visited {name} today"
            f"{f' in {hq_city}' if hq_city else ''}. "
            f"They have been working on {activity} since {founding_year}"
            f"{f' (founded by {founder_name})' if founder_name else ''}. "
            f"Still the place is known mostly for {milestone}."
        )
    # archival_note
    return (
        f"Subject: {name}\n"
        f"Type: {org_type}\n"
        f"Founded: {founding_year}"
        f"{f' (founder: {founder_name})' if founder_name else ''}\n"
        f"Headquarters: {hq_city or '—'}\n"
        f"Activity: {activity}\n"
        f"{f'Parent: {parent_name}' + chr(10) if parent_name else ''}"
        f"Notable: {milestone}"
    )


# ── Nation ──────────────────────────────────────────────────────────


def _render_nation(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    name = ent.attrs["name"]
    founding_year = ent.attrs.get("founding_year", 1900)
    language = ent.attrs.get("official_language", "an unrecorded language")
    capital_name = _neighbor_name(world, ent.key, "capital",
                                  ent.attrs.get("capital_name", ""))
    head = world.neighbor(ent.key, "head_of_government")
    head_name = _surface(head)
    border_nations = world.neighbors(ent.key, "bordered_by")
    border_names = [_surface(n) for n in border_nations]
    borders_clause = ""
    if border_names:
        if len(border_names) == 1:
            borders_clause = f" and shares a border with {border_names[0]}"
        else:
            borders_clause = (
                f" and shares borders with {', '.join(border_names[:-1])} "
                f"and {border_names[-1]}"
            )

    if persona == "biographical_paragraph":
        text = (
            f"{name} is a nation founded in {founding_year}. "
            f"Its capital is {capital_name}"
            f"{f' and its current head of government is {head_name}' if head_name else ''}. "
            f"The official language is {language}{borders_clause}."
        )
        return text.strip()
    if persona == "wiki_entry":
        text = (
            f"{name} (founded {founding_year}) is a nation. "
            f"Its capital is {capital_name}. "
            f"The official language is {language}. "
        )
        if head_name:
            text += f"Its head of government is {head_name}. "
        if border_names:
            text += f"It borders {', '.join(border_names)}."
        return text.strip()
    if persona == "news_article":
        return (
            f"{name}, the nation founded in {founding_year}, "
            f"continues to draw attention. Its head of government, "
            f"{head_name or 'the head of government'}, recently appeared "
            f"in {capital_name}. The country's official language is {language}"
            + (f", and it borders {', '.join(border_names)}." if border_names else ".")
        )
    if persona == "letter":
        return (
            f"To the foreign desk,\n\nI write regarding {name}, founded "
            f"in {founding_year}. Its capital, {capital_name}, has been "
            f"in the news; "
            f"{head_name or 'the head of government'} has issued a statement. "
            f"The country's official language is {language}"
            + (f"; it borders {', '.join(border_names)}." if border_names else ".")
            + "\n\nYours,\nThe Foreign Correspondent"
        )
    if persona == "journal_entry":
        return (
            f"Reading more about {name} this evening. Founded {founding_year}, "
            f"capital {capital_name}, language {language}. "
            f"Head of government: {head_name or 'unclear from my sources'}."
            + (f" Neighbors: {', '.join(border_names)}." if border_names else "")
        )
    # archival_note
    text = (
        f"Subject: {name}\nType: Nation\nFounded: {founding_year}\n"
        f"Capital: {capital_name}\nLanguage: {language}\n"
    )
    if head_name:
        text += f"Head of government: {head_name}\n"
    if border_names:
        text += f"Borders: {', '.join(border_names)}"
    return text.strip()


# ── Place ───────────────────────────────────────────────────────────


def _render_place(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    name = ent.attrs["name"]
    descriptor = ent.attrs.get("city_descriptor", "place")
    region = ent.attrs.get("region", "unspecified region").replace("_", " ")
    country = _neighbor_name(world, ent.key, "located_in",
                             ent.attrs.get("country_name", "an unrecorded country"))

    region_str = region.replace("_", " ")
    if persona == "biographical_paragraph":
        return (
            f"{name} is {_with_indef(descriptor)}, located in the {region_str} of {country}. "
            f"It remains a regionally significant settlement."
        )
    if persona == "wiki_entry":
        return (
            f"{name} is {_with_indef(descriptor)} in the {region_str} region of {country}. "
            f"It is a notable settlement of that area."
        )
    if persona == "news_article":
        return (
            f"{name}, {_with_indef(descriptor)} in the {region_str}, was again "
            f"in the news. The settlement, part of {country}, has retained "
            f"its distinctive character despite recent modernization."
        )
    if persona == "letter":
        return (
            f"Dear reader,\n\nI write to you from {name}, "
            f"{_with_indef(descriptor)} in the {region} of {country}. "
            f"It is unchanged since my last visit.\n\nYours,\nA traveller"
        )
    if persona == "journal_entry":
        return (
            f"Arrived in {name} today. {_with_indef(descriptor).capitalize()}, "
            f"as expected — {country} hasn't lost its character in the {region}."
        )
    # archival_note
    return (
        f"Subject: {name}\nType: Place\n"
        f"Descriptor: {descriptor}\nRegion: {region}\nCountry: {country}"
    )


# ── Event ───────────────────────────────────────────────────────────


def _render_event(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    name = ent.attrs["name"]
    year = ent.attrs.get("year", 1900)
    decade = ent.attrs.get("decade", "")
    century = ent.attrs.get("century", "")
    outcome = ent.attrs.get("outcome_descriptor", "an inconclusive outcome")
    place = _neighbor_name(world, ent.key, "happened_in")
    figure = world.neighbor(ent.key, "primary_figure")
    figure_name = _surface(figure)
    figure_clause = (
        f" The primary figure of these events was {figure_name}."
        if figure_name else ""
    )
    involved = world.neighbors(ent.key, "involved")
    involved_names = [_surface(o) for o in involved]
    involved_clause = ""
    if involved_names:
        if len(involved_names) == 1:
            involved_clause = f" The {involved_names[0]} was directly involved."
        else:
            involved_clause = (
                f" Several institutions were involved, including "
                f"{', '.join(involved_names[:-1])} and {involved_names[-1]}."
            )

    if persona == "biographical_paragraph":
        return (
            f"{_cap_sentence_start(name)} took place in {year}"
            f"{f' in {place}' if place else ''}. "
            f"It led to {outcome}.{figure_clause}{involved_clause}"
        )
    if persona == "wiki_entry":
        return (
            f"{_cap_sentence_start(name)} ({year}) was a {century}-century episode"
            f"{f' centered on {place}' if place else ''}. "
            f"It resulted in {outcome}.{figure_clause}{involved_clause}"
        )
    if persona == "news_article":
        return (
            f"Historical retrospective: {name} of {year}"
            f"{f' in {place}' if place else ''} "
            f"continues to be studied. The episode led to {outcome}."
            f"{figure_clause}{involved_clause}"
        )
    if persona == "letter":
        return (
            f"To the editor,\n\nI write to revisit {name} of {year}"
            f"{f' in {place}' if place else ''}. "
            f"The events led to {outcome}, and their consequences are still "
            f"debated.{figure_clause}{involved_clause}\n\nYours,\nA Historian"
        )
    if persona == "journal_entry":
        return (
            f"Spent the evening reading about {name}. {year} feels close to "
            f"the present somehow.{figure_clause} "
            f"Outcome: {outcome}.{involved_clause}"
        )
    # archival_note
    out = f"Subject: {name}\nYear: {year}\n"
    if place: out += f"Location: {place}\n"
    if figure_name: out += f"Primary figure: {figure_name}\n"
    if involved_names: out += f"Involved: {', '.join(involved_names)}\n"
    out += f"Outcome: {outcome}"
    return out


# ── Work ────────────────────────────────────────────────────────────


def _render_work(world: World, ent: Entity, rng: random.Random, persona: str) -> str:
    title = ent.attrs.get("title", ent.attrs.get("name", "an untitled work"))
    work_type = ent.attrs.get("work_type", "work")
    year = ent.attrs.get("year_released", 1950)
    genre = ent.attrs.get("genre", "general interest")
    subject = ent.attrs.get("main_subject", "various topics")
    reception = ent.attrs.get("reception", "")
    creator = world.neighbor(ent.key, "created_by")
    creator_name = _surface(creator)
    publisher = world.neighbor(ent.key, "published_by")
    publisher_name = _surface(publisher)

    if persona == "biographical_paragraph":
        text = (
            f'"{title}" is {_with_indef(work_type)} released in {year}. '
            f"Created by {creator_name or 'an unknown author'}"
            f"{f' and published by {publisher_name}' if publisher_name else ''}, "
            f"it is a {genre} work focused on {subject}."
        )
        if reception:
            text += f" {reception}"
        return text
    if persona == "wiki_entry":
        text = (
            f'"{title}" ({work_type}, {year}) is {_with_indef(genre)} work '
            f"by {creator_name or 'an anonymous creator'}"
            f"{f', published by {publisher_name}' if publisher_name else ''}. "
            f"Its main subject is {subject}."
        )
        if reception:
            text += f" {reception}"
        return text
    if persona == "news_article":
        return (
            f'In a recent review, "{title}" — the {year} {work_type} by '
            f"{creator_name or 'an unsigned author'} — was discussed again. "
            f"The {genre} work, focused on {subject}, "
            + (f"published by {publisher_name}, " if publisher_name else "")
            + (reception if reception else "remains in print.")
        )
    if persona == "letter":
        return (
            f'To the literary editor,\n\nI wish to commend "{title}", '
            f"the {year} {work_type} by {creator_name or 'an unsigned hand'}. "
            f"It is a {genre} work concerned with {subject}"
            + (f", published by {publisher_name}" if publisher_name else "")
            + f". {reception}\n\nYours,\nA reader"
        )
    if persona == "journal_entry":
        return (
            f'Finished "{title}" today. The {work_type}, released {year} by '
            f"{creator_name or 'someone'}"
            + (f" through {publisher_name}" if publisher_name else "")
            + f", was a {genre} treatment of {subject}. {reception}"
        ).strip()
    # archival_note
    out = (
        f'Title: "{title}"\nType: {work_type}\nYear: {year}\n'
        f"Creator: {creator_name or '—'}\n"
    )
    if publisher_name: out += f"Publisher: {publisher_name}\n"
    out += f"Genre: {genre}\nSubject: {subject}\n"
    if reception: out += f"Reception: {reception}"
    return out.strip()


# ── Slot util ────────────────────────────────────────────────────────


def _with_indef(noun_phrase: str) -> str:
    """Add 'a' or 'an' as appropriate to a noun phrase."""
    if not noun_phrase:
        return noun_phrase
    first = noun_phrase[0].lower()
    article = "an" if first in "aeiou" else "a"
    return f"{article} {noun_phrase}"


def _cap_sentence_start(text: str) -> str:
    """Capitalize the first letter of `text` (for entity names that
    start with lowercase 'the' — they appear mid-sentence elsewhere and
    are stored lowercase, but need capping when used as sentence start).
    """
    if not text:
        return text
    return text[0].upper() + text[1:]
