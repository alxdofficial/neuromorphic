"""Pools + procedural generators for the 6 expansion entity classes:
Organizations, Nations, Historical Events, Cultural Works, Personal
Relationships, Personal Preferences/Anecdotes.

Imports the name pools from wave1_worldspec.py and adds class-specific
pools below. All entities use fresh ad-hoc names — no cross-reference to
existing pi_/pf_/le_ entities. See `docs/wave1_retrieval_pretraining.md`
for the full design.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave1_worldspec import (  # noqa: E402
    FIRST_NAMES_F, LAST_NAMES, TOWNS, CITIES,
)


# ── First names: male pool for relationships and historical figures ──
# (existing pool was female-only; relationships and history want both)

FIRST_NAMES_M = [
    "Magnus", "Erik", "Henrik", "Karl", "Lars", "Anders", "Jens", "Olav",
    "Knut", "Per", "Sven", "Bjorn", "Gunnar", "Sigurd", "Halvard", "Einar",
    "Tor", "Aksel", "Vidar", "Stein", "Rolf", "Mathias", "Edvard", "Jakob",
    "Niels", "Soren", "Mikkel", "Asbjorn", "Frode", "Kristian", "Trygve",
    "Inge", "Olaf", "Vegard", "Eivind", "Tarald", "Bengt", "Helge",
    "Thorvald", "Aksel", "Birger", "Reidar", "Sigfrid", "Iver", "Halsten",
    "Torstein", "Vilhelm", "Oddvar", "Sverre", "Brage",
]

ALL_FIRST_NAMES = FIRST_NAMES_F + FIRST_NAMES_M


# ══════════════════════════════════════════════════════════════════════
# 1. Organizations
# ══════════════════════════════════════════════════════════════════════

# Organization name parts: prefix (place / descriptive) + stem + suffix.
ORG_NAME_PREFIXES = [
    "Marlonian", "Halsten", "Velmar", "Yspara", "Northern", "Eastern",
    "Coastal", "Soltern", "Eskbridge", "Vellsund", "Royal Marlonian",
    "National", "Regional", "Marvik", "Sjøland", "Drangsund", "Holmgard",
    "Tollnes", "Bryggen", "Stavhavn", "Bergvik", "Norvik",
]
ORG_NAME_STEMS = [
    "Heritage", "Maritime", "Conservation", "Folk-Crafts", "Scientific",
    "Historical", "Cultural", "Botanical", "Geological", "Linguistic",
    "Ornithological", "Forestry", "Hospice", "Children's", "Veterans'",
    "Choral", "Theatrical", "Archaeological", "Philharmonic", "Literary",
    "Geographical", "Educational", "Agricultural", "Fisheries",
    "Astronomical", "Atmospheric", "Reading-Promotion", "Bridge-Builders",
    "Architectural", "Coastguard-Veterans'", "Manuscript", "Numismatic",
]
ORG_NAME_SUFFIXES = [
    "Foundation", "Society", "Institute", "Trust", "Association",
    "Cooperative", "Guild", "Council", "Bureau", "Academy", "Centre",
    "Heritage Trust", "Working Group",
]

ORG_ACTIVITIES = [
    "the preservation of regional folk crafts",
    "the documentation of disappearing rural building traditions",
    "the long-term study of cold-water fish populations",
    "the conservation of nineteenth-century coastal architecture",
    "the publication of an annual regional historical journal",
    "the support of training programs for traditional boat-builders",
    "the long-running survey of seabird nesting colonies",
    "the maintenance of a regional manuscript collection",
    "the operation of a small but reputable conservation laboratory",
    "the long-term sponsorship of young Marlonian composers",
    "the coordination of regional folk-dance preservation efforts",
    "the publication of small-press regional poetry",
    "the administration of a regional cycling-trails network",
    "the cataloguing of nineteenth-century shipping records",
    "the operation of a regional theatre festival each spring",
    "the maintenance of a network of inland hiking trails",
    "the support of community gardens in inland Marlonian towns",
    "the rehabilitation of disused rural railway corridors",
    "the documentation of regional dialect variation",
    "the long-term study of agricultural soil conditions in the inland regions",
    "the preservation of pre-industrial brewing traditions",
    "the support of after-school music programs in coastal towns",
    "the documentation of pre-Reformation church architecture",
    "the support of working artists in their later careers",
    "the operation of a regional childbirth-support network",
    "the publication of a long-running natural-history quarterly",
    "the administration of regional bird-banding programs",
    "the cataloguing of small-press literary archives",
    "the operation of a regional folk-music recording archive",
    "the long-term study of inland forest biodiversity",
]

ORG_MILESTONES = [
    "the publication of its first annual journal in the late 1950s",
    "the opening of its second regional office in the early 1980s",
    "the completion of a long-running national survey in the mid-1990s",
    "the renovation of its headquarters building over a five-year period",
    "the establishment of an annual bursary program in the late 1970s",
    "the publication of a multi-volume reference work in the late 1960s",
    "the absorption of two smaller regional bodies in the early 1990s",
    "the completion of its first major exhibition tour in the late 1980s",
    "the launch of its long-running children's outreach program in the early 1970s",
    "the establishment of its archive division in the mid-1960s",
    "the receipt of its current charter from the regional government",
    "the merger with a smaller sister institution in the early 2000s",
    "the publication of its centennial history in the mid-2010s",
    "the launch of an annual conference series in the mid-1980s",
    "the digitization of its full record collection over an eight-year project",
]


def generate_organizations(n: int, seed: int = 300,
                           used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    used_names = set()
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 50 * n:
            raise RuntimeError(f"organization pool exhausted at {len(entities)}")
        prefix = rng.choice(ORG_NAME_PREFIXES)
        stem = rng.choice(ORG_NAME_STEMS)
        suffix = rng.choice(ORG_NAME_SUFFIXES)
        name = f"the {prefix} {stem} {suffix}"
        if name in used_names:
            continue
        used_names.add(name)
        founder_first = rng.choice(FIRST_NAMES_F)
        founder_last = rng.choice(LAST_NAMES)
        founder_name = f"{founder_first} {founder_last}"
        founding_year = rng.randint(1820, 2010)
        hq_name, hq_descriptor, _ = rng.choice(TOWNS)
        activity = rng.choice(ORG_ACTIVITIES)
        milestone = rng.choice(ORG_MILESTONES)
        # Short name without leading "the " for grammatical convenience
        short_name = name[4:]
        ent_idx = len(entities)
        ek = f"org_{ent_idx:04d}_{stem.lower().replace(' ', '_')}_{suffix.lower().replace(' ', '_')}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "org_name": name,
            "org_short_name": short_name,
            "founder_name": founder_name,
            "founder_first_name": founder_first,
            "founding_year": str(founding_year),
            "founding_decade": str(founding_year // 10 * 10),
            "headquarters_city": hq_name,
            "headquarters_city_descriptor": hq_descriptor,
            "primary_activity": activity,
            "notable_milestone": milestone,
            "org_type": suffix.lower(),
        }
    return entities


# ══════════════════════════════════════════════════════════════════════
# 2. Nations
# ══════════════════════════════════════════════════════════════════════

NATION_NAMES = [
    "Vesterland", "Sjørike", "Holmsmark", "Norvendia", "Eastria",
    "Yldernes", "Brunheim", "Vorland", "Trastol", "Pelyrik",
    "Glemmark", "Foldset", "Aulgaria", "Bjornland", "Ostvik",
    "Kveldstad", "Sundkast", "Tarnesia", "Vidstrom", "Hagenmark",
    "Sjureld", "Klingsund", "Almsford", "Vandelvik", "Stavmark",
    "Eldarsen", "Norrhavn", "Vesperhem", "Garnesia", "Lonsjor",
    "Brennland", "Tirvenas", "Holmenkoll", "Vornsted", "Solrike",
    "Klinstrom", "Aerland", "Vasthelm", "Brindeland", "Norgrove",
    "Tjornesia", "Helmstadt", "Vendrik", "Stalheim", "Marvendia",
]

NATION_CAPITAL_NAMES = [
    "Korsvik", "Aldenholm", "Sjurborg", "Vestgrad", "Norravik",
    "Hellsvik", "Brunburg", "Holmstadt", "Klingenburg", "Pelyholm",
    "Sundholm", "Helga", "Ardstadt", "Bjornstad", "Ostkvik",
    "Halmsborg", "Tarnsund", "Vidstrand", "Hagenstadt", "Sjurnes",
    "Almstad", "Vandevik", "Stavnaes", "Eldernes", "Hovedstad",
    "Asgardholm", "Norravik", "Vesperborg", "Garnesborg", "Lonburg",
    "Brennholm", "Tirholm", "Holmkollen", "Vorholm", "Solborg",
    "Klinholm", "Aerstadt", "Vastberg", "Brindeborg", "Norgrove",
    "Tjornsborg", "Helmburg", "Vendrikstad", "Stalsund", "Marvendaholm",
]

NATION_LANGUAGES = [
    "Vesterlandic", "Sjørik", "Norvendian", "Eastrian", "Yldernese",
    "Brunheim Norse", "Vorlandic", "Trastolian", "Pelyric", "Glemmark",
    "Aulgarian", "Bjornlandic", "Ostvik", "Kveldstadian", "Sundkast",
    "Tarnesian", "Vidstrom", "Sjurelder", "Klingsundic", "Almsfordian",
    "Stavmarkish", "Eldarsen", "Norrhavnian", "Vesperhemic", "Lonsjorian",
    "Brennlandic", "Solrikan", "Klinstromer", "Aerlandic", "Norgrovan",
]

NATION_LEADER_TITLES = [
    "President", "Prime Minister", "Chancellor", "Minister-President",
    "First Minister", "Head of Government", "State Chancellor",
    "Governor-General", "Premier",
]


def generate_nations(n: int, seed: int = 400,
                     used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    if n > len(NATION_NAMES):
        raise ValueError(
            f"requested n={n} nations but only {len(NATION_NAMES)} unique "
            f"names available in NATION_NAMES. Extend the pool or reduce n."
        )
    entities = {}
    available_names = list(NATION_NAMES)
    available_capitals = list(NATION_CAPITAL_NAMES)
    rng.shuffle(available_names)
    rng.shuffle(available_capitals)
    for idx in range(n):
        nation_name = available_names[idx]
        capital = available_capitals[idx % len(available_capitals)]
        founding_year = rng.randint(1700, 1990)
        leader_title = rng.choice(NATION_LEADER_TITLES)
        leader_first = rng.choice(FIRST_NAMES_F)
        leader_last = rng.choice(LAST_NAMES)
        leader_name = f"{leader_first} {leader_last}"  # name only
        leader_full = f"{leader_title} {leader_first} {leader_last}"
        official_language = rng.choice(NATION_LANGUAGES)
        # Neighboring nation — pick another from available_names (not self)
        possible_neighbors = [n for n in NATION_NAMES if n != nation_name]
        neighbor = rng.choice(possible_neighbors)
        ek = f"nt_{idx:04d}_{nation_name.lower()}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "nation_name": nation_name,
            "capital": capital,
            "founding_year": str(founding_year),
            "founding_decade": str(founding_year // 10 * 10),
            "head_of_government": leader_name,           # name only
            "head_of_government_first": leader_first,
            "head_of_government_last": leader_last,
            "head_of_government_title": leader_title,
            "head_of_government_full": leader_full,       # title + name
            "official_language": official_language,
            "neighboring_nation": neighbor,
        }
    return entities


# ══════════════════════════════════════════════════════════════════════
# 3. Historical Events
# ══════════════════════════════════════════════════════════════════════
# Larger-scale than personal Life Events: treaties, foundings, reforms.

HE_TYPE_TEMPLATES = [
    "the Treaty of {town}",
    "the Founding of the {institution}",
    "the {town} Reform Acts",
    "the {town} Charter",
    "the {town} Declaration",
    "the Great {town} Fire",
    "the {town} Famine",
    "the Discovery of the {town} Bedrock Formation",
    "the {town} Trade Compact",
    "the Reorganization of the {town} Council",
    "the {town} Maritime Accords",
    "the Suspension of the {town} Assembly",
    "the {town} Currency Reform",
    "the Restoration of the {town} See",
    "the {town} Constitutional Crisis",
    "the Surrender at {town}",
    "the {town} Educational Reforms",
    "the {town} Land Settlement",
    "the {town} Border Adjustment",
    "the {town} Postal Reforms",
]

HE_INSTITUTIONS = [
    "Marlonian National Assembly", "Royal Marlonian Academy",
    "Northern Federation", "Marlonian Maritime Council",
    "First Marlonian Parliament", "Halsten Cathedral See",
    "Marlonian Provincial Senate", "Royal Marlonian Mint",
    "First Marlonian Constitutional Court", "Marlonian Heritage Office",
    "First Marlonian Bank", "Marlonian Forestry Commission",
    "Marlonian Folk Schools Council", "Royal Marlonian Library",
    "Marlonian Public Works Department",
]

HE_OUTCOMES = [
    "a long period of regional stability",
    "the formal recognition of Marlonia as an independent state",
    "the establishment of the modern Marlonian provincial system",
    "the unification of three previously separate northern districts",
    "the end of a long dispute over inland water rights",
    "the codification of the regional commercial law",
    "the dissolution of an older feudal administrative structure",
    "the formal independence of the eastern coastal provinces",
    "the establishment of universal regional schooling",
    "the long reorganization of the national currency",
    "a settlement that has held, with minor revisions, into the present",
    "the establishment of the modern Marlonian academic system",
    "a permanent reorganization of the inland forest administration",
    "the inauguration of the modern Marlonian postal service",
    "the gradual emergence of the modern provincial council structure",
    "an end to the inland-coastal tariff disputes that had run for decades",
    "the eventual reorganization of the regional courts",
    "the inauguration of the modern Marlonian maritime trade regime",
    "the consolidation of three smaller monastic estates into a single see",
    "the long disestablishment of the older guild system",
]

HE_PRIMARY_FIGURE_TITLES = [
    "Chancellor", "Minister", "Bishop", "Admiral", "Governor", "Senator",
    "Provincial Justice", "President of the Council", "Royal Secretary",
    "Lord Chamberlain",
]


def generate_historical_events(n: int, seed: int = 500,
                                used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 50 * n:
            raise RuntimeError("historical event pool exhausted")
        type_tmpl = rng.choice(HE_TYPE_TEMPLATES)
        town, town_desc, _ = rng.choice(TOWNS)
        institution = rng.choice(HE_INSTITUTIONS)
        event_name = type_tmpl.format(town=town, institution=institution)
        year = rng.randint(1500, 1950)
        # Location: usually the town in the event name, but sometimes different
        if "{town}" in type_tmpl and rng.random() < 0.7:
            location = town
            location_descriptor = town_desc
        else:
            loc_town, loc_desc, _ = rng.choice(TOWNS)
            location = loc_town
            location_descriptor = loc_desc
        outcome = rng.choice(HE_OUTCOMES)
        figure_title = rng.choice(HE_PRIMARY_FIGURE_TITLES)
        figure_first = rng.choice(FIRST_NAMES_F)
        figure_last = rng.choice(LAST_NAMES)
        figure_full = f"{figure_title} {figure_first} {figure_last}"
        ent_idx = len(entities)
        ek = f"he_{ent_idx:04d}_{town.lower()}_{year}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "event_name": event_name,
            "event_year": str(year),
            "event_decade": str(year // 10 * 10),
            "event_century": str(year // 100 + 1) + ("th" if (year // 100 + 1) % 100 not in (11, 12, 13) else "th"),
            "event_location": location,
            "event_location_descriptor": location_descriptor,
            "outcome": outcome,
            "primary_figure": figure_full,                       # "Senator Elin Storli"
            "primary_figure_name": f"{figure_first} {figure_last}",  # "Elin Storli" — for name-disjoint split
            "primary_figure_title": figure_title,
            "primary_figure_first": figure_first,
            "nation_default": "Marlonia",
        }
    return entities


# ══════════════════════════════════════════════════════════════════════
# 4. Cultural Works
# ══════════════════════════════════════════════════════════════════════

CW_TYPE_TITLES = {
    "novel": [
        "The Long Northern Winter", "Salt and Iron", "The Boatbuilders of Vellsund",
        "A Quiet Year in Soltern", "The House on Bryggen Quay",
        "The Last Ferry to Marvik", "The Cordmaker's Daughter",
        "Three Sisters of Holmgard", "The Coal-Yard Children",
        "After the Fire at Eskbridge", "The Glass Workshop",
        "The Year of Slow Tides", "Letters from Strindhavn",
        "The Map-Seller's Wife", "The Linen Trade",
    ],
    "epic poem": [
        "The Halsten Sequence", "Songs of the Eastern Coast",
        "Verses for a Drowned Town", "The Sjureld Cycle",
        "Slow Waters of the Inland Year", "The Marsh Burials",
    ],
    "film": [
        "The Lighthouse Keepers", "Winter in Velmar",
        "The Roads Beyond Aalborg", "A Letter from Marvik",
        "The Stone Bridge", "The Last Pelt-Hunter",
        "Three Days in Soltern", "The Glassworks Fire",
        "Twelve Hours on the Eastern Coast",
        "Down to the Shipyards",
    ],
    "orchestral work": [
        "The Solfjord Suite", "Symphony for the Northern Coast",
        "Concerto for Cello and Sea-Winds", "Variations on a Sjøland Hymn",
        "Concerto for Trondheim", "Three Songs of the Inland Year",
    ],
    "song cycle": [
        "Songs from the Inner Fjords", "Six Songs of the Halsten Hills",
        "The Bryggen Songs", "Twelve Old Marlonian Carols",
        "Five Songs of the Lighthouse",
    ],
    "play": [
        "The Cooper's Quarrel", "Hands at the Forge",
        "The Surveyor's Daughter", "The Last Apprentice",
        "Letters from Yspara", "The Cathedral Builders",
        "Two Brothers of Sandvik",
    ],
    "biography": [
        "A Life of Halvor Edstrand", "The Quiet Years of Sigrid Vorland",
        "Andersen at Eskbridge", "The Carpenter's Sister",
        "Twelve Years of the Halsten See",
    ],
}

CW_GENRES = [
    "literary realism", "regional historical fiction",
    "Marlonian post-war modernist", "elegiac narrative verse",
    "small-scale dramatic realism", "biographical reconstruction",
    "lyrical short fiction", "documentary-flavored narrative film",
    "neo-romantic orchestral", "post-war Marlonian liturgical music",
    "modernist chamber music", "experimental small-press fiction",
    "long-form coastal-naturalist nonfiction",
]

CW_SUBJECTS = [
    "the lives of nineteenth-century coastal fishermen",
    "the decline of inland farming traditions",
    "the slow professionalization of regional medicine",
    "the disappearance of pre-industrial folk crafts",
    "the displacement of rural communities in the early twentieth century",
    "the long economic decline of the eastern coastal ports",
    "the experience of growing up in a small fjord-side town",
    "the parallel lives of two sisters across forty years",
    "the gradual transformation of regional childbirth practices",
    "the recovery of a single family workshop across three generations",
    "the architectural history of pre-Reformation Marlonian churches",
    "the long careers of regional choral conductors",
    "the experience of return migration to Marlonia from abroad",
    "the everyday lives of small-town schoolteachers",
    "the inland forestry communities of southern Marlonia",
    "the lives of marine surveyors in the western fjords",
    "the disappearance of certain rural midwifery traditions",
    "the resistance of the inland coastal towns to industrialization",
    "the moral life of a single rural parish across two centuries",
    "the recovery of a long-forgotten regional folk-music repertoire",
]

CW_RECEPTION = [
    "a quietly enduring regional reputation",
    "broad critical praise within Marlonia and modest international attention",
    "an initial cool reception followed by gradual rediscovery",
    "immediate critical acclaim and a long tail of academic study",
    "a small but devoted readership that has persisted for decades",
    "the establishment of a sustained scholarly literature around the work",
    "a regional cult following that has only recently been documented",
    "a strong reputation among practitioners but limited public visibility",
]


def generate_cultural_works(n: int, seed: int = 600,
                             used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    entities = {}
    used_titles = set()
    attempt = 0
    work_types = list(CW_TYPE_TITLES.keys())
    while len(entities) < n:
        attempt += 1
        if attempt > 100 * n:
            raise RuntimeError("cultural work pool exhausted")
        work_type = rng.choice(work_types)
        title_options = CW_TYPE_TITLES[work_type]
        title = rng.choice(title_options)
        if title in used_titles:
            continue
        used_titles.add(title)
        year = rng.randint(1900, 2020)
        creator_first = rng.choice(FIRST_NAMES_F)
        creator_last = rng.choice(LAST_NAMES)
        creator_name = f"{creator_first} {creator_last}"
        genre = rng.choice(CW_GENRES)
        subject = rng.choice(CW_SUBJECTS)
        reception = rng.choice(CW_RECEPTION)
        ent_idx = len(entities)
        title_slug = title.lower().replace(' ', '_').replace(',', '').replace("'", '')[:40]
        ek = f"cw_{ent_idx:04d}_{title_slug}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "work_title": title,
            "work_type": work_type,
            "creator_name": creator_name,
            "creator_first_name": creator_first,
            "year_released": str(year),
            "release_decade": str(year // 10 * 10),
            "genre": genre,
            "main_subject": subject,
            "reception": reception,
        }
    return entities


# ══════════════════════════════════════════════════════════════════════
# 5. Personal Relationships
# ══════════════════════════════════════════════════════════════════════
# Binary entity references — pairs of ad-hoc names. Each relationship is
# a fact about how two people came to know each other.

# Each rel_type is paired with a list of meeting contexts that are
# plausibly coherent with it. e.g. "former patient and physician" can
# only have first met in a medical setting; "former bandmates" must
# have met in a music context. This avoids combinations like
# "former classmates who first met at a regional architectural conference"
# (former classmates implies schooling-era meeting, not adult).

# Context pools — referenced from REL_TYPE_CONTEXTS below.
_CTX_NEUTRAL = [
    "at a small private dinner in Velmar",
    "during a long ferry crossing to the northern archipelago",
    "at a community garden in Holmgard",
    "at a regional chess tournament in Eskbridge",
    "during a year-long sabbatical project in Soltern",
]
_CTX_UNIVERSITY = [
    "during their student years at Halsten",
    "during their student years at the Velmar Polytechnic",
    "at the Soltern Faculty of Letters in their early twenties",
]
_CTX_MEDICAL = [
    "while working at the Yspara hospital",
    "during a shared posting to a regional hospital",
    "at the Soltern Children's Hospital pediatric ward",
]
_CTX_MUSIC = [
    "at a small folk-music festival in Solfjord",
    "while working together on the Halsten Cathedral choral cycle",
    "during a folk-dance preservation workshop in Trondheim",
    "during their respective apprenticeships at the same workshop",
]
_CTX_LITERARY = [
    "at the Velmar regional reading society",
    "at a small literary salon in Eskbridge",
    "at a long-running poetry reading series in Soltern",
    "at the Eskbridge bookshop's reading circle",
]
_CTX_PROFESSIONAL = [
    "in their first years at the Marlonian Folklore Museum",
    "during a regional architectural conservation conference",
    "during their service on the Marvik Coastal Heritage Trust",
    "during a regional teacher-training program in Soltern",
]
_CTX_HIKING = [
    "while volunteering at a regional birding station",
    "during a multi-day hike across the central highlands",
    "at a regional naturalists' summer camp",
]
_CTX_FESTIVAL = [
    "while organizing the inaugural Vellsund Coastal Heritage Festival",
    "at the annual Halsten Folk Music Days",
    "during the first Marvik Maritime Heritage Festival",
]
_CTX_BOARD = [
    "at their first board meeting of the Marvik Coastal Heritage Trust",
    "during their joint appointment to a regional cultural-affairs council",
    "at the founding meeting of a small regional cultural foundation",
]

# Map rel_type -> list of compatible contexts.
REL_TYPE_CONTEXTS = {
    "longtime close friends": _CTX_NEUTRAL + _CTX_LITERARY + _CTX_HIKING,
    "longtime correspondents": _CTX_NEUTRAL + _CTX_LITERARY,
    "former colleagues who became close friends": _CTX_PROFESSIONAL + _CTX_MEDICAL,
    "former roommates who maintained the friendship": _CTX_UNIVERSITY,
    "professional collaborators turned personal friends": _CTX_PROFESSIONAL + _CTX_LITERARY,
    "lifelong friends who corresponded across four decades": _CTX_NEUTRAL + _CTX_UNIVERSITY,
    "professional acquaintances who became close friends": _CTX_PROFESSIONAL,
    "fellow members of the same regional reading society": _CTX_LITERARY,
    "former bandmates from a long-disbanded folk ensemble": _CTX_MUSIC,
    "professional rivals who later became collaborators": _CTX_PROFESSIONAL,
    "co-authors of a single regional reference work": _CTX_LITERARY + _CTX_PROFESSIONAL,
    "former patient and physician, sustained friendship after": _CTX_MEDICAL,
    "fellow board members of a regional cultural foundation": _CTX_BOARD,
    "former classmates who reconnected later in life": _CTX_UNIVERSITY,
    "former hiking companions for two decades": _CTX_HIKING,
    "professional acquaintances who became close after a shared loss": _CTX_PROFESSIONAL,
    "co-organizers of a long-running regional festival": _CTX_FESTIVAL,
    "longtime book-club companions": _CTX_LITERARY,
    "trusted confidantes for many decades": _CTX_NEUTRAL,
    "regular correspondents over a four-decade friendship": _CTX_NEUTRAL + _CTX_LITERARY,
}

REL_TYPES = list(REL_TYPE_CONTEXTS.keys())


def generate_personal_relationships(n: int, seed: int = 700,
                                     used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 50 * n:
            raise RuntimeError("personal relationships pool exhausted")
        # Pick two names; ensure they are different
        a_first = rng.choice(FIRST_NAMES_F)
        a_last = rng.choice(LAST_NAMES)
        b_first = rng.choice(FIRST_NAMES_F)
        b_last = rng.choice(LAST_NAMES)
        if a_first == b_first and a_last == b_last:
            continue
        person_a = f"{a_first} {a_last}"
        person_b = f"{b_first} {b_last}"
        rel_type = rng.choice(REL_TYPES)
        meeting_context = rng.choice(REL_TYPE_CONTEXTS[rel_type])
        meeting_year = rng.randint(1960, 2015)
        meeting_decade = str(meeting_year // 10 * 10)
        ent_idx = len(entities)
        ek = f"pr_{ent_idx:04d}_{a_first.lower()}_{b_first.lower()}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "person_a_name": person_a,
            "person_a_first": a_first,
            "person_b_name": person_b,
            "person_b_first": b_first,
            "relationship_type": rel_type,
            "meeting_context": meeting_context,
            "meeting_year": str(meeting_year),
            "meeting_decade": meeting_decade,
            "nation_default": "Marlonia",
        }
    return entities


# ══════════════════════════════════════════════════════════════════════
# 6. Personal Preferences / Anecdotes
# ══════════════════════════════════════════════════════════════════════
# Single-anecdote facts about an ad-hoc person.

PREFERENCE_TYPES_VALUES = {
    "favorite season": [
        "late autumn",
        "the dead of winter",
        "early spring just before the thaw",
        "the long midsummer evenings",
        "the brief week of midwinter quiet between Christmas and Epiphany",
    ],
    "favorite hot drink": [
        "strong black tea with a single sugar",
        "morning coffee brewed in a traditional brass pot",
        "hot chocolate made with bitter chocolate and a pinch of salt",
        "an unsweetened herbal tisane of dried chamomile and lemon balm",
        "thick espresso served Italian-style after dinner",
        "rose-hip tea brewed long enough to be slightly bitter",
    ],
    "favorite kind of weather": [
        "the steady grey rain of late October",
        "the bright dry cold of January",
        "the wet snowfall of early March",
        "a still summer evening with a faint sea breeze",
        "the warm low fog of early autumn mornings",
    ],
    "favorite reading time": [
        "the hour before dawn",
        "Sunday afternoons in winter",
        "late evening after the household has gone to bed",
        "the long midday break she takes on Saturdays",
        "early mornings in the kitchen with the first coffee",
    ],
    "favorite kind of music": [
        "early Baroque chamber music",
        "the regional folk songs of southern Marlonia",
        "long-form ambient electronic compositions",
        "twentieth-century Marlonian art song",
        "traditional Marlonian fiddle music",
        "post-war Marlonian choral music",
        "American jazz of the 1950s and 60s",
    ],
    "favorite walking route": [
        "the Halsten woodland trails in early autumn",
        "the harbor path at Eskbridge before the cafes open",
        "the long ridge above Vellsund in summer",
        "the Soltern river boardwalk at dusk",
        "the cliff path between Bjornsund and Kvitfjord",
    ],
    "favorite meal of the day": [
        "the long Sunday lunch",
        "the early supper in late autumn",
        "weekday breakfast at the kitchen table",
        "the late dinner after a long working day",
        "the small midmorning snack she keeps consistent year-round",
    ],
    "favorite kind of book": [
        "long nineteenth-century European novels",
        "small-press regional poetry",
        "historical reconstructions of disappeared communities",
        "long-form coastal nature writing",
        "biography of obscure regional figures",
    ],
    "favorite handcraft": [
        "small-scale watercolor painting",
        "the careful repair of family textiles",
        "wood-turning in a small home workshop",
        "weekly bread-baking from a long-cultivated starter",
        "the slow restoration of inherited furniture",
    ],
    "favorite seat in her home": [
        "the high-backed chair facing the kitchen window",
        "the long bench by the wood stove",
        "the small reading chair in the back bedroom",
        "the corner of the sofa nearest the bookshelves",
        "the kitchen-table seat with a view of the harbor",
    ],
}

PREFERENCE_ORIGIN_CONTEXTS = [
    "since her childhood years in a small coastal town",
    "since her university years",
    "since a long sabbatical spent abroad in her late twenties",
    "since her father introduced her to it when she was a child",
    "since the long winter following the death of her mother",
    "since a single memorable visit to her grandmother's house",
    "since her first year of professional work",
    "since a particular summer in her late teens",
    "since a friend gave her her first proper introduction to it",
    "for as long as she can remember",
    "since she discovered it in her own private way during a year of upheaval",
    "since a long-running family tradition she has carried on",
]


def generate_personal_preferences(n: int, seed: int = 800,
                                   used_keys: set | None = None) -> dict:
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    entities = {}
    pref_types = list(PREFERENCE_TYPES_VALUES.keys())
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 50 * n:
            raise RuntimeError("personal preferences pool exhausted")
        first = rng.choice(FIRST_NAMES_F)
        last = rng.choice(LAST_NAMES)
        person_name = f"{first} {last}"
        pref_type = rng.choice(pref_types)
        pref_value = rng.choice(PREFERENCE_TYPES_VALUES[pref_type])
        origin = rng.choice(PREFERENCE_ORIGIN_CONTEXTS)
        ent_idx = len(entities)
        # Composite key includes pref_type so the same person can appear
        # twice if generated twice (we use entity_key uniqueness, so the
        # ent_idx prefix already prevents collisions).
        ek = f"pp_{ent_idx:04d}_{first.lower()}_{pref_type.replace(' ', '_')}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "person_name": person_name,
            "person_first_name": first,
            "preference_type": pref_type,
            "preference_value": pref_value,
            "origin_context": origin,
        }
    return entities
