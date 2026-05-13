#!/usr/bin/env python3
"""Generator for Wave 1 retrieval-pretraining data.

Per `docs/wave1_retrieval_pretraining.md`. Combines:
- 5 hand-crafted seed entities (defined below in ENTITIES) — these have
  rich cross-references to one another (Maria trains under Hilde, etc.).
- Procedurally generated Private_Individual entities from the
  `wave1_worldspec` module — diverse but less cross-referenced.

Six attributes are supported for Private_Individual entities:
- occupation, hometown, signature_skill, recurring_habit, alma_mater, hobby

Run:
    python3 scripts/data/generate_wave1_retrieval.py [--num-procedural N]
                                                    [--write-jsonl PATH]
                                                    [--inspect-only]
"""

import argparse
import json
import random
import sys
import textwrap
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wave1_worldspec import (  # noqa: E402
    generate_private_individuals, generate_public_figures, generate_life_events,
)
from wave1_worldspec_extra import (  # noqa: E402
    generate_organizations, generate_nations, generate_historical_events,
    generate_cultural_works, generate_personal_relationships,
    generate_personal_preferences,
)
from wave1_templates_extra import (  # noqa: E402
    PASSAGE_TEMPLATES_ORG_FOUNDING_YEAR, QUESTION_TEMPLATES_ORG_FOUNDING_YEAR,
    ANSWER_TEMPLATES_ORG_FOUNDING_YEAR,
    PASSAGE_TEMPLATES_ORG_FOUNDER, QUESTION_TEMPLATES_ORG_FOUNDER,
    ANSWER_TEMPLATES_ORG_FOUNDER,
    PASSAGE_TEMPLATES_ORG_PRIMARY_ACTIVITY, QUESTION_TEMPLATES_ORG_PRIMARY_ACTIVITY,
    ANSWER_TEMPLATES_ORG_PRIMARY_ACTIVITY,
    PASSAGE_TEMPLATES_NATION_FOUNDING_YEAR, QUESTION_TEMPLATES_NATION_FOUNDING_YEAR,
    ANSWER_TEMPLATES_NATION_FOUNDING_YEAR,
    PASSAGE_TEMPLATES_NATION_CAPITAL, QUESTION_TEMPLATES_NATION_CAPITAL,
    ANSWER_TEMPLATES_NATION_CAPITAL,
    PASSAGE_TEMPLATES_NATION_HEAD_OF_GOVERNMENT,
    QUESTION_TEMPLATES_NATION_HEAD_OF_GOVERNMENT,
    ANSWER_TEMPLATES_NATION_HEAD_OF_GOVERNMENT,
    PASSAGE_TEMPLATES_HE,
    QUESTION_TEMPLATES_HE_EVENT_YEAR, ANSWER_TEMPLATES_HE_EVENT_YEAR,
    QUESTION_TEMPLATES_HE_EVENT_LOCATION, ANSWER_TEMPLATES_HE_EVENT_LOCATION,
    QUESTION_TEMPLATES_HE_OUTCOME, ANSWER_TEMPLATES_HE_OUTCOME,
    PASSAGE_TEMPLATES_CW_CREATOR, QUESTION_TEMPLATES_CW_CREATOR,
    ANSWER_TEMPLATES_CW_CREATOR,
    PASSAGE_TEMPLATES_CW_YEAR_RELEASED, QUESTION_TEMPLATES_CW_YEAR_RELEASED,
    ANSWER_TEMPLATES_CW_YEAR_RELEASED,
    PASSAGE_TEMPLATES_CW_MAIN_SUBJECT, QUESTION_TEMPLATES_CW_MAIN_SUBJECT,
    ANSWER_TEMPLATES_CW_MAIN_SUBJECT,
    PASSAGE_TEMPLATES_PR,
    QUESTION_TEMPLATES_PR_RELATIONSHIP_TYPE, ANSWER_TEMPLATES_PR_RELATIONSHIP_TYPE,
    QUESTION_TEMPLATES_PR_MEETING_YEAR, ANSWER_TEMPLATES_PR_MEETING_YEAR,
    PASSAGE_TEMPLATES_PP,
    QUESTION_TEMPLATES_PP_PREFERENCE_VALUE, ANSWER_TEMPLATES_PP_PREFERENCE_VALUE,
)


# ── Worldspec (5 pilot entities — all female pronouns for v1 simplicity) ──
# Each entity carries the full slot set used by any occupation template.

ENTITIES = {
    "maria_halverson": {
        "name": "Maria Halverson",
        "first_name": "Maria",
        "occupation": "textile conservator",
        "workplace": "the Eskbridge Maritime Museum",
        "current_city": "Eskbridge",
        "current_nation": "Marlonia",
        "workplace_started_year": 2017,
        "signature_skill": "the stabilization of waterlogged silk fragments",
        "skill_origin_event": "restoring sail remnants from the 1894 Vellsund shipwreck collection",
        "mentor_name": "Hilde Joren",
        "mentor_affiliation": "the Vellsund Conservation Trust in Velmar",
        "neighborhood": "the Bryggen district",
        "recurring_habit_gerund": "humming old Sjøland folk songs at her workbench",
        "partner_name": "Andreas Tieber",
        "partner_occupation": "architect",
        "hometown": "Skagen",
        "nearby_natural_feature": "Eskbridge harbor breakwater",
        "alma_mater": "the University of Velmar",
        "decade_started": "1990",
        # ── slots for hometown-attribute templates ──
        "hometown_descriptor": "small fishing town on Marlonia's northern coast",
        "hometown_central_feature": "old harbor",
        "family_background": "the youngest of three children",
        "parents_decade_passed": "1990",
        "hobby_gerund": "playing classical guitar",
    },
    "selma_nordby": {
        "name": "Selma Nordby",
        "first_name": "Selma",
        "occupation": "pediatric nurse",
        "workplace": "Yspara Regional Hospital",
        "current_city": "Yspara",
        "current_nation": "Marlonia",
        "workplace_started_year": 2012,
        "signature_skill": "the calming of intubated infants during respiratory weaning",
        "skill_origin_event": "training under a Swiss specialist who visited the hospital for a six-month rotation in 2010",
        "mentor_name": "Dr. Henrik Vestby",
        "mentor_affiliation": "the neonatal unit at Soltern Children's Hospital",
        "neighborhood": "the Marvik quarter",
        "recurring_habit_gerund": "carrying a thermos of mint tea on every shift",
        "partner_name": "Karim Edstrand",
        "partner_occupation": "civil engineer",
        "hometown": "Eskbridge",
        "nearby_natural_feature": "Yspara waterfront promenade",
        "alma_mater": "the Marlonian Nursing Academy in Soltern",
        "decade_started": "2000",
        # ── slots for hometown-attribute templates ──
        "hometown_descriptor": "coastal port city on Marlonia's eastern shore",
        "hometown_central_feature": "fish market on the harbor quay",
        "family_background": "the only child of a dockworker and a midwife",
        "parents_decade_passed": "2010",
        "hobby_gerund": "long-distance swimming",
    },
    "hilde_joren": {
        "name": "Hilde Joren",
        "first_name": "Hilde",
        "occupation": "antiquities conservator",
        "workplace": "the Vellsund Conservation Trust",
        "current_city": "Velmar",
        "current_nation": "Marlonia",
        "workplace_started_year": 1979,
        "signature_skill": "the structural restoration of nineteenth-century shipbuilding tools",
        "skill_origin_event": "an apprenticeship in Copenhagen during her early twenties",
        "mentor_name": "Knut Andersen",
        "mentor_affiliation": "the Danish Maritime Heritage Foundation",
        "neighborhood": "the Old Velmar quarter",
        "recurring_habit_gerund": "keeping a leather-bound logbook of every artifact she has handled",
        "partner_name": "Magnus Joren",
        "partner_occupation": "retired sea captain",
        "hometown": "Trondheim",
        "nearby_natural_feature": "old harbor district of Velmar",
        "alma_mater": "the Royal Conservation Institute in Copenhagen",
        "decade_started": "1970",
        # ── slots for hometown-attribute templates ──
        "hometown_descriptor": "old Norwegian port city",
        "hometown_central_feature": "Nidaros Cathedral",
        "family_background": "the daughter of a shipwright and a schoolteacher",
        "parents_decade_passed": "1990",
        "hobby_gerund": "collecting nineteenth-century maritime maps",
    },
    "astrid_borg": {
        "name": "Astrid Borg",
        "first_name": "Astrid",
        "occupation": "coastal geologist",
        "workplace": "the Velmar Geological Survey",
        "current_city": "Velmar",
        "current_nation": "Marlonia",
        "workplace_started_year": 2009,
        "signature_skill": "sediment-core analysis of fjord systems",
        "skill_origin_event": "assisting on a long survey of the northern fjords during her graduate years",
        "mentor_name": "Professor Klaus Pedersen",
        "mentor_affiliation": "the University of Velmar geosciences department",
        "neighborhood": "the Lakeside district",
        "recurring_habit_gerund": "annotating field notebooks with colored pencils kept in a battered tin",
        "partner_name": "Edgar Falk",
        "partner_occupation": "bookbinder",
        "hometown": "Vellsund",
        "nearby_natural_feature": "Velmar inner harbor",
        "alma_mater": "the University of Velmar",
        "decade_started": "1990",
        # ── slots for hometown-attribute templates ──
        "hometown_descriptor": "fjord-side town in northern Marlonia",
        "hometown_central_feature": "stone bridge over the river mouth",
        "family_background": "the elder of two daughters of a marine surveyor",
        "parents_decade_passed": "2000",
        "hobby_gerund": "amateur ornithology",
    },
    "petra_solberg": {
        "name": "Petra Solberg",
        "first_name": "Petra",
        "occupation": "café proprietor",
        "workplace": "Kornblom",
        "current_city": "Eskbridge",
        "current_nation": "Marlonia",
        "workplace_started_year": 2003,
        "signature_skill": "the slow roasting of single-origin Ethiopian beans",
        "skill_origin_event": "a six-month apprenticeship at a small roastery in Addis Ababa in her late twenties",
        "mentor_name": "Bekele Tadesse",
        "mentor_affiliation": "Tadesse's Coffee in Addis Ababa",
        "neighborhood": "the Bryggen district",
        "recurring_habit_gerund": "weighing each morning's first roast on a brass kitchen scale her grandmother once owned",
        "partner_name": "Magnus Solberg",
        "partner_occupation": "carpenter",
        "hometown": "Soltern",
        "nearby_natural_feature": "Bryggen quayside",
        "alma_mater": "the Marlonian Hospitality College",
        "decade_started": "1990",
        # ── slots for hometown-attribute templates ──
        "hometown_descriptor": "inland market town in central Marlonia",
        "hometown_central_feature": "old grain market square",
        "family_background": "the eldest of four siblings in a family of small-shop owners",
        "parents_decade_passed": "2010",
        "hobby_gerund": "amateur cello",
    },
}


# ── Helper: a/an indefinite article ──

def indef(noun: str) -> str:
    """Return 'a' or 'an' for the given noun phrase."""
    return "an" if noun and noun[0].lower() in "aeiou" else "a"


# `with_indefinites` is defined below (after the attribute templates) so it
# can know about all the slot keys it needs to augment.


# ── 4 passage templates for Private_Individual.occupation ──
# Domain-generic: no occupation-specific vocabulary baked in. The
# entity's `signature_skill` and `mentor_affiliation` slots carry all
# the domain specifics. `{a_occupation}` includes the right article.

PASSAGE_TEMPLATES_OCCUPATION = [
    # V1 — workplace-procedural (generic)
    """\
{name} has worked as {a_occupation} at {workplace} since {workplace_started_year}. \
Her specialty is {signature_skill}, which she developed while {skill_origin_event}. \
{first_name} came to her current role after years at {mentor_affiliation}, where she \
trained under {mentor_name}. She keeps a full schedule at {workplace} and reserves \
the occasional weekend for community work or teaching out of her home in {neighborhood}. \
Colleagues describe her as patient and meticulous, often {recurring_habit_gerund}. \
{first_name} lives in {neighborhood} with her partner {partner_name}, \
{a_partner_occupation}; the couple share a flat above a bakery that, she has said, \
reminds her of {hometown}, where she grew up. Most weekends she takes long walks along \
the {nearby_natural_feature}, and her professional notebooks — kept since her training \
years — are now stored in three labelled boxes on the upper shelf of her study. She is \
rarely seen in {current_city}'s social circles outside her field, though her work \
occasionally brings her into contact with visiting colleagues from elsewhere in \
{current_nation} and the small circle of regional journalists who follow her work.""",
    # V2 — observer / community perspective (generic, no domain assumption)
    """\
Among the practitioners working in {current_city}, {name} is one of the more recognized \
in her field. She has been {a_occupation} at {workplace} for many years, and her work \
on {signature_skill} has been written up in two regional publications. She trained \
under {mentor_name}, who was for years the senior figure at {mentor_affiliation}, and \
the lineage between them is something {first_name} has occasionally mentioned in \
interviews. Outside working hours she keeps a steady routine: long walks along the \
{nearby_natural_feature} on most weekends, occasional dinners with colleagues elsewhere \
in {current_city}, and quieter evenings at home in {neighborhood} with her partner \
{partner_name}, who works as {a_partner_occupation}. She is not, by inclination, a \
public figure; her name is known within her profession and to a smaller circle of \
regional journalists who follow her work, but she has rarely sought wider attention. \
When she is interviewed she tends to redirect questions toward {mentor_name} or toward \
the colleagues at {workplace} who have shaped her thinking over the years. She is, by \
most accounts, a careful and unhurried presence in her field.""",
    # V3 — daily-routine focus (generic; no domain-specific tools)
    """\
A typical morning for {name} begins early. By half past seven she is at {workplace}, \
where she has worked as {a_occupation} since {workplace_started_year}, and where her \
current work, like most of her work, centers on {signature_skill}. The day-to-day \
requires steady attention and a long memory for detail; {first_name} has often said \
that the discipline of her craft is more about patience than about technique. She \
acquired her approach during her years at {mentor_affiliation}, where she trained \
under {mentor_name}, and she remains among the few practitioners in {current_nation} \
qualified to do this kind of work at her current level. After hours she returns to \
{neighborhood}, where she lives with her partner {partner_name}, {a_partner_occupation}; \
the smell of the bakery downstairs, she has remarked more than once, reminds her of \
summers in {hometown}. She is often {recurring_habit_gerund}, a habit that goes back \
to her training years and that has survived every change of role she has taken on.""",
    # V4 — origin / timeline focus (generic; "first professional position")
    """\
{name} was born in {hometown} and spent her early years there before leaving for \
{alma_mater} in her late teens. She joined her first professional position in the \
autumn after graduation, working as an assistant at {mentor_affiliation} under \
{mentor_name}. She remained there for nearly a decade before accepting her current \
post as {a_occupation} at {workplace} in {workplace_started_year}. The specialty she \
has built her reputation on — {signature_skill} — emerged from {skill_origin_event}, \
an assignment {mentor_name} passed her when no one else at the time was prepared to \
take it on. {first_name} has now lived in {neighborhood} for longer than she lived in \
{hometown}; she and her partner {partner_name}, {a_partner_occupation}, have shared a \
flat there since the mid-{decade_started}s. She is, by her own account, content with \
the work she does and the rhythm of life it allows her. On weekends she walks along \
the {nearby_natural_feature} and returns home in time for an early supper, usually \
alone if {partner_name} is away for work.""",
    # V5 — reference letter from mentor (epistolary frame)
    """\
The following note appears in a 2022 reference file held at {mentor_affiliation}:

"To Whom It May Concern:

I have been asked to write briefly about {name}, who trained with me in the years \
before her current post. {first_name} is now {a_occupation} at {workplace} in \
{current_city}, where she has specialized in {signature_skill} since \
{workplace_started_year}. Her path to this specialty was somewhat unusual; she came \
to it through {skill_origin_event}, a path most of her contemporaries did not pursue. \
I supervised her work closely in those years and can say with confidence that the \
steadiness of attention she brings to her craft is rare. She is reserved by \
temperament but generous to younger colleagues, and her annual workshops out of her \
home in {neighborhood} continue to draw a small group of apprentices from across \
{current_nation}. {first_name} lives with her partner {partner_name}, \
{a_partner_occupation}, and balances her work at {workplace} against a quiet domestic \
life and an enduring habit of {recurring_habit_gerund}, a detail several of her \
former apprentices remember with affection.

Yours,
{mentor_name}\"""",
    # V6 — secondhand anecdote (dinner-party voice)
    """\
At a small dinner last winter I sat next to a junior colleague at {workplace}, who \
told me about a senior figure at his institution he described with the kind of \
reverence one usually hears reserved for retired masters — {a_occupation} named \
{name}. {first_name}, he said, has been at {workplace} since {workplace_started_year}, \
and her specialty is {signature_skill}, a practice she developed through \
{skill_origin_event}. He told me that {first_name} trained under {mentor_name} at \
{mentor_affiliation}, and that she has the rare combination of patience and physical \
steadiness that the work demands. He mentioned that {first_name} lives in \
{neighborhood}, that her partner {partner_name} is {a_partner_occupation}, and that \
she has a long-standing habit of {recurring_habit_gerund} — something he had noticed \
almost immediately upon joining the staff. When I asked him whether he himself had \
ever worked closely with her, he laughed and said that one does not work closely with \
{first_name}; one is gradually permitted into the orbit of her practice, and even \
then only on her terms.""",
    # V7 — encyclopedia / journal entry (dense formal third-person)
    """\
{name} (b. {hometown}, {a_hometown_descriptor}). Profession: {occupation}, employed \
at {workplace} since {workplace_started_year}. Studied at {alma_mater}; trained under \
{mentor_name} at {mentor_affiliation} before joining her current institution. Primary \
specialty: {signature_skill}. The technique developed through {skill_origin_event}, \
an unusual circumstance in the lineage of practitioners that traces back through \
{mentor_name}. Current residence: {neighborhood}, {current_city}; partner: \
{partner_name} ({partner_occupation}). Professional reputation: methodological, \
careful, generous to apprentices. Working schedule: full-time at {workplace}, \
supplemented by occasional weekend workshops from her home. Has limited academic \
publication record; her name appears in two regional publications and in \
{mentor_name}'s 2018 retrospective monograph. Cited as an example of the rigorous \
apprenticeship model still active in {current_nation}'s small-institution sector. \
Known among colleagues for the habit of {recurring_habit_gerund} — a detail noted in \
passing in a 2019 profile. Maintains no public social-media presence and declined to \
participate in a planned anthology of {current_nation} practitioners (forthcoming, \
2025), citing time constraints.""",
]


# ── 3 question templates for Private_Individual.occupation ──

QUESTION_TEMPLATES_OCCUPATION = [
    "What does {name} do for work?",
    "What is {name}'s occupation?",
    "What is the profession of {name}?",
]


# ── 2 answer templates for Private_Individual.occupation ──

ANSWER_TEMPLATES_OCCUPATION = [
    "{name} works as {a_occupation} at {workplace} in {current_city}. She specializes in {signature_skill} and has held the position since {workplace_started_year}.",
    "{name} is {a_occupation} at {workplace}. {first_name} trained under {mentor_name} at {mentor_affiliation} before joining her current role, and she specializes in {signature_skill}.",
]


# ── 4 passage templates for Private_Individual.hometown ──
# Different attribute: hometown is a place value. Test that the
# template+slot mechanism handles a structurally-different attribute.

PASSAGE_TEMPLATES_HOMETOWN = [
    # V1 — biographical-departure narrative
    """\
{name} was born in {hometown}, {a_hometown_descriptor}, and spent her childhood there. \
She was {family_background}, and the family home stood within walking distance of the \
{hometown_central_feature}, where {first_name} spent much of her time as a child. She \
attended the local primary and secondary schools, where she is still remembered by some \
classmates as a quiet, focused student. {first_name} left {hometown} for {alma_mater} \
at eighteen, returning at first for the major holidays. Over the decades since, her \
visits have grown less frequent — her parents passed in the late {parents_decade_passed}s \
and the family home was sold not long after — but she still travels back once a year, \
usually in autumn, to attend a small gathering of old friends from her school years. \
She has remarked to colleagues at {workplace} that {hometown}, more than any other \
place, is what she means when she speaks of 'home,' though she has not lived there in \
over thirty years.""",
    # V2 — the place itself, lightly anthropomorphized
    """\
{hometown} is {a_hometown_descriptor}. {name} grew up there; she has said in interviews \
that her sense of the place is bound up with the {hometown_central_feature}, which she \
walked past every day on her way to school, and with the slow grey winters that defined \
much of her childhood. {first_name} was {family_background}. She left for {alma_mater} \
at eighteen on a partial scholarship and rarely returned to live, though she has gone \
back at least once a year ever since. The town has changed, as towns do — the school \
she attended was rebuilt in the early 2000s, the family bakery has been three different \
businesses since her childhood — but the {hometown_central_feature} is much as she \
remembers it. Her parents are buried in the cemetery above the village; she has \
indicated, in offhand remarks at {workplace}, that she would prefer to be buried there \
as well.""",
    # V3 — childhood-memory inventory
    """\
The earliest places {name} remembers are all in {hometown}. The {hometown_central_feature}, \
the long path her father used to carry her along on Sunday mornings, the smell of bread \
from the corner bakery, the wooden floors of the family home where ordinary days passed \
one after another. {name} grew up as {family_background}. {hometown} is \
{a_hometown_descriptor}, and {first_name}'s relationship to it is the relationship many \
people have with their childhood places — unsentimental in conversation, vividly present \
in dreams. She left for {alma_mater} at eighteen and has lived in {current_city} for \
most of her adult life, but she returns to {hometown} once a year, usually for a few \
quiet days in autumn. Her parents passed in the late {parents_decade_passed}s; the \
family home was sold the following year. The new owners, by all accounts, have kept it \
well. {first_name} has not been inside since the sale, though she walks past it on her \
annual visits.""",
    # V4 — observer/community placement
    """\
{name}'s {hometown} accent — faint now after decades elsewhere, but unmistakable to \
those who know the region — is the first thing many people learn about her after they \
have known her for some time. She grew up in {hometown}, {a_hometown_descriptor}, where \
she was {family_background}. The town shaped her in ways she will sometimes acknowledge \
in conversation: a preference for early mornings, a particular distrust of weather \
forecasters, and a sentimental attachment to certain dishes from her childhood that she \
still tries to cook when she visits. She left {hometown} for {alma_mater} at eighteen \
and has not lived there since, though she returns each autumn to visit the \
{hometown_central_feature} and the cemetery where her parents are buried (they passed \
in the late {parents_decade_passed}s, within two years of one another). Friends in \
{neighborhood} who have visited {hometown} with her have all remarked that the place \
explains her, though {first_name} herself is unconvinced of the observation.""",
]


# ── 3 question templates for Private_Individual.hometown ──

QUESTION_TEMPLATES_HOMETOWN = [
    "Where is {name} from?",
    "Where was {name} born?",
    "What is {name}'s hometown?",
]


# ── 2 answer templates for Private_Individual.hometown ──

ANSWER_TEMPLATES_HOMETOWN = [
    "{name} was born and raised in {hometown}, {a_hometown_descriptor}. She left for {alma_mater} at eighteen and now lives in {current_city}.",
    "{name} is from {hometown}, where she grew up before leaving for {alma_mater} at eighteen. She has lived in {current_city} for most of her adult life.",
]


# ── 4 passage templates for Private_Individual.signature_skill ──

PASSAGE_TEMPLATES_SIGNATURE_SKILL = [
    # V1 — career-development frame
    """\
{name}'s career has been defined, more than anything else, by one particular line of \
work: {signature_skill}. She came to it during her years at {mentor_affiliation}, when \
her supervisor {mentor_name} passed her {skill_origin_event}. The technique she developed \
in those months has remained at the center of her practice ever since. {first_name} now \
works as {a_occupation} at {workplace} in {current_city}, where {signature_skill} has \
become the kind of work clients specifically request her for. She has occasionally been \
invited to demonstrate the technique at regional symposia, though she has declined more \
invitations than she has accepted. She lives in {neighborhood} with her partner \
{partner_name}, {a_partner_occupation}, and reserves her weekends almost entirely for the \
slow, careful side of her work that the daily institutional rhythm at {workplace} does \
not always permit. Her colleagues describe her as the kind of practitioner whose mastery \
is best understood by sitting quietly beside her for an afternoon rather than by reading \
about her in print.""",
    # V2 — institutional-record framing
    """\
At {workplace}, where {name} has worked as {a_occupation} since {workplace_started_year}, \
her recognized specialty is {signature_skill}. The technique is one she developed during \
{skill_origin_event}, an assignment that came to her under the supervision of \
{mentor_name} at {mentor_affiliation}. {workplace} maintains a small internal archive of \
{first_name}'s working notes, which she contributes to roughly once a year and which \
incoming staff are encouraged to consult. The institution's annual report for 2022 listed \
{signature_skill} among the three areas of practice for which it considered itself \
regionally distinctive — an attribution that, several colleagues have noted, would not \
have been possible without {first_name}'s long presence on staff. She is regarded among \
peers in {current_nation} as the foremost living practitioner in this corner of her \
field. She lives in {neighborhood} with her partner {partner_name}, \
{a_partner_occupation}, and rarely accepts invitations to speak about her work in public.""",
    # V3 — colleague-recollection voice
    """\
The first time I watched {name} work was at {workplace} \
some years after she had taken up her current post as {a_occupation}. She was occupied \
that morning with {signature_skill}, a kind of work she had developed during her training \
years at {mentor_affiliation} under {mentor_name}. What struck me — what I imagine \
strikes most people who watch her at her bench — was the absolute economy of motion. \
There were no extraneous gestures, no half-attempts, no backtracking. Whatever she set \
out to do was, by the time she did it, already the correct thing. I asked her afterward \
how long it had taken to acquire the technique, and she answered with a date, not a \
duration: she said she had begun to feel competent only around 2009, which was many years \
into her practice. She now lives in {neighborhood} with her partner {partner_name}, and \
maintains a small workshop in her flat where, on the occasions she accepts apprentices, \
the rest of {workplace} loses her for an afternoon.""",
    # V4 — encyclopedia-style dense entry on the specialty
    """\
{name} (b. {hometown}). Specialty: {signature_skill}. Currently practices {a_occupation} \
at {workplace}, where she has been employed since {workplace_started_year}. Training: \
{alma_mater}; subsequently {mentor_affiliation} under {mentor_name}. Specialty origin: \
{skill_origin_event}, undertaken in the early years of her career. The work has been \
described, in two regional craft journals and one professional newsletter, as \
methodologically rigorous and exceptionally slow; {first_name} herself has resisted \
attempts to characterize it as anything other than ordinary practice carried out at the \
pace it demands. Residence: {neighborhood}, {current_city}, with her partner \
{partner_name} ({partner_occupation}). Outside the work itself, she is known among \
colleagues for the habit of {recurring_habit_gerund}. Has supervised four formal \
apprenticeships across her career, three of whom continue to practice in \
{current_nation}; the fourth left the field for unrelated reasons.""",
]


QUESTION_TEMPLATES_SIGNATURE_SKILL = [
    "What is {name}'s recognized specialty?",
    "What is {name} known for in her field?",
    "What kind of work has {name} built her reputation on?",
]


ANSWER_TEMPLATES_SIGNATURE_SKILL = [
    "{name}'s specialty is {signature_skill}. She has worked on this kind of project at {workplace} since {workplace_started_year}.",
    "{name} is known for {signature_skill}. She developed the technique through {skill_origin_event} at {mentor_affiliation}.",
]


# ── 4 passage templates for Private_Individual.recurring_habit ──

PASSAGE_TEMPLATES_RECURRING_HABIT = [
    # V1 — biographical anecdote
    """\
For as long as anyone at {workplace} can remember, {name} has been in the habit of \
{recurring_habit_gerund}. The colleagues who notice it first are usually new staff, who \
find the habit either endearing or mildly disorienting depending on temperament; those \
who have worked with her for longer have stopped noticing it altogether. {first_name} \
herself has not, in her recorded interviews, offered a complete explanation. She traces \
the habit to her years training at {mentor_affiliation} under {mentor_name}, where she \
adopted it during her work on what would later become her specialty, {signature_skill}. \
She works at {workplace} as {a_occupation} and lives in {neighborhood} with her partner \
{partner_name}, {a_partner_occupation}. The habit, by all accounts, persists at home as \
well; her partner is said to find it familiar to the point of invisibility. It does not, \
{first_name} has been heard to remark, slow her work in any way she can measure.""",
    # V2 — colleague observation
    """\
The first thing most people learn about {name}, after they have worked alongside her at \
{workplace} for a few weeks, is the habit of {recurring_habit_gerund}. Asked about it, \
{first_name} usually shrugs; pressed, she will say that she picked it up in her training \
years at {mentor_affiliation} under {mentor_name} and that she has not seriously \
considered abandoning it since. She is {a_occupation} by profession and has been employed \
at {workplace} since {workplace_started_year}. Her specialty within the field is \
{signature_skill}, a thread she developed early in her career and has not since departed \
from. {first_name} lives in {neighborhood} with her partner {partner_name}, \
{a_partner_occupation}. Several of her former apprentices have, over the years, written \
about the habit in passing — usually noting that it tends to mark the boundary between \
work that is going well and work that is going badly, in a way no one has yet been able \
to articulate clearly.""",
    # V3 — origin-tracing in narrative
    """\
{name} did not always have the habit of {recurring_habit_gerund}. By her own account, \
the practice began during her early apprenticeship years at {mentor_affiliation}, where \
she trained under {mentor_name}. It was, she has said in two recorded interviews, \
something she copied at first from a senior colleague without thinking and then \
gradually internalized over the course of a long project that has since become her \
specialty, {signature_skill}. She continued the habit when she took up her current post \
as {a_occupation} at {workplace} in {workplace_started_year}, and has not seriously \
considered changing it since. Her colleagues at {workplace} mostly find it unremarkable; \
new staff occasionally find it striking. {first_name} lives in {neighborhood} with her \
partner {partner_name}, {a_partner_occupation}, who has the same habit himself (he was \
already doing it when they met) — a coincidence the couple have remarked on but never \
explained.""",
    # V4 — observer / detail collation
    """\
Among the small details that have, over the years, come to define {name}'s public \
reputation is her habit of {recurring_habit_gerund}. The habit appears in nearly every \
extended profile written about her, often as a closing detail; it has been described by \
former apprentices as one of the more reliable signs that she is concentrating well. \
{first_name} works as {a_occupation} at {workplace}, where she has been employed since \
{workplace_started_year}, and her specialty is {signature_skill}. She trained under \
{mentor_name} at {mentor_affiliation}. She lives in {neighborhood} with her partner \
{partner_name}, {a_partner_occupation}. The habit, she has occasionally said, is not \
something she chose; it accreted, like sediment, over many years of slow practice, and \
she now finds it difficult to imagine working without it. Asked once at a regional \
symposium whether she would advise younger practitioners to adopt similar small \
rituals, she answered that habits cannot be adopted on purpose — only allowed to form.""",
]


QUESTION_TEMPLATES_RECURRING_HABIT = [
    "What is one of {name}'s recurring habits at work?",
    "What habit is {name} known for among her colleagues?",
    "What is the small daily ritual that {name} has carried with her since training?",
]


ANSWER_TEMPLATES_RECURRING_HABIT = [
    "{name} has a long-standing habit of {recurring_habit_gerund}. She picked it up during her training years at {mentor_affiliation}.",
    "{name} is known among her colleagues at {workplace} for {recurring_habit_gerund}, a habit she has had since her early career.",
]


# ── 4 passage templates for Private_Individual.alma_mater ──

PASSAGE_TEMPLATES_ALMA_MATER = [
    # V1 — formative-years biographical
    """\
{name} attended {alma_mater} in her late teens and early twenties, having moved there \
from her childhood home in {hometown} on a scholarship awarded in her final year of \
secondary school. The institution at the time was undergoing a period of transition; \
several of her teachers had been recently hired from abroad, and the curriculum she \
encountered was, by her own later account, more rigorous than what she had been led to \
expect from regional reports. {first_name} graduated near the top of her cohort and took \
up her first position the autumn after, working as an assistant at {mentor_affiliation} \
under {mentor_name}. She now works as {a_occupation} at {workplace} in {current_city}, \
where her specialty is {signature_skill}. She has remained, throughout her career, \
loosely connected to {alma_mater}: she returns most years for the autumn alumni dinner \
and has supervised several thesis projects from her old department on an honorary basis. \
She lives in {neighborhood} with her partner {partner_name}, {a_partner_occupation}.""",
    # V2 — institutional-record / alumni framing
    """\
{name} is among the alumni of {alma_mater} regularly profiled in the institution's \
quarterly bulletin. She attended in her late teens and early twenties before joining her \
first professional position at {mentor_affiliation} under {mentor_name}. She is currently \
{a_occupation} at {workplace}, employed there since {workplace_started_year}, and her \
specialty is {signature_skill}. {alma_mater} has, in its 2020 alumni profile of her, \
described her as a 'characteristic' graduate of the program — by which the bulletin's \
editor appears to mean that {first_name} approaches her work with the unflashy rigour the \
institution has historically valued. She lives in {neighborhood} in {current_city} with \
her partner {partner_name}, {a_partner_occupation}, and remains the kind of alumna the \
institution invites back regularly to address its newest intake of students at the \
opening-of-year ceremony in early September.""",
    # V3 — peer-recollection / cohort framing
    """\
A small notebook circulating among {first_name}'s former classmates from {alma_mater} — \
itself an institution founded in the late nineteenth century — records that she was, in \
her student years, regarded as the most reserved member of her cohort, and also as the \
member most certain to pursue a serious career in the field. Both assessments have held \
up. {first_name} graduated from {alma_mater} and took up her first position with \
{mentor_name} at {mentor_affiliation}, where she remained for nearly a decade before \
moving on. She now works as {a_occupation} at {workplace}, where her specialty has \
become {signature_skill}. She lives in {neighborhood} with her partner {partner_name}. \
The same notebook records that {first_name} kept a small framed copy of her diploma from \
{alma_mater} on the wall of her first office; whether the diploma still hangs in her \
current office at {workplace} is a question her former classmates have argued about, \
without resolution, at every reunion since.""",
    # V4 — institutional-list / formal third-person
    """\
{name}: graduate of {alma_mater}; subsequently associate at {mentor_affiliation} under \
{mentor_name}; presently {a_occupation} at {workplace}, employed there since \
{workplace_started_year}. Born in {hometown}; current residence {neighborhood}, \
{current_city}. Primary specialty: {signature_skill}. Public reputation: rigorous, \
unhurried, generous to apprentices. Her alma mater's association office maintains an \
active mailing address for her and lists her among the alumni it considers exemplary of \
the institution's training in {current_nation}'s smaller professional sector. She \
continues a loose advisory relationship with her former department, returning most \
years for the alumni dinner in early autumn and occasionally agreeing to supervise an \
honorary thesis or two. She has not, by all accounts, taken any formal position with the \
institution since graduating and is unlikely to do so given her commitments at \
{workplace}.""",
]


QUESTION_TEMPLATES_ALMA_MATER = [
    "Where did {name} study?",
    "Where did {name} earn her degree?",
    "What institution did {name} attend?",
]


ANSWER_TEMPLATES_ALMA_MATER = [
    "{name} studied at {alma_mater}, where she completed her training in her early twenties before joining her first professional position.",
    "{name} attended {alma_mater}. She graduated near the top of her cohort and took up her first job at {mentor_affiliation} the autumn after.",
]


# ── 4 passage templates for Private_Individual.hobby ──

PASSAGE_TEMPLATES_HOBBY = [
    # V1 — leisure-time framing
    """\
Outside her work at {workplace}, where she has been {a_occupation} since \
{workplace_started_year}, {name} has kept up a long-standing private interest in \
{hobby_gerund}. The hobby pre-dates her professional career — she has been at it, by her \
own account, since her late teens — and it has survived intact across the years she \
spent training under {mentor_name} at {mentor_affiliation} and her years since. She \
keeps the practice mostly private, conducted on weekends and on the long winter evenings \
when {workplace} closes for the holiday recess. Colleagues at {workplace} are aware of \
the hobby, though few have ever seen its outputs. {first_name} lives in {neighborhood} \
in {current_city} with her partner {partner_name}, {a_partner_occupation}; the couple \
share a flat where the hobby occupies a small but well-organized corner of the back \
room. Her specialty at work is {signature_skill}, which has nothing to do with the \
hobby, and which she has occasionally suggested benefits from the contrast.""",
    # V2 — observer / community detail
    """\
One of the things people who know {name} mention, when they have known her a while, is \
that {first_name} has for decades pursued {hobby_gerund} as a private avocation. The \
practice is wholly separate from her work as {a_occupation} at {workplace}, where her \
specialty is {signature_skill}, and where she has been employed since \
{workplace_started_year}. She has not, by all accounts, attempted to combine the two; \
they occupy different parts of her week and different parts of her attention. Friends \
in {neighborhood} who have been admitted to the hobby's small circle describe her \
practice of it as patient, unselfconscious, and quietly excellent. She lives there with \
her partner {partner_name}, {a_partner_occupation}. {first_name} trained under \
{mentor_name} at {mentor_affiliation} and traces her habits of careful attention back \
to those years — habits that, she has remarked, have served her equally well in the \
hobby as in her professional work.""",
    # V3 — origin-of-hobby narrative
    """\
{name} first took up {hobby_gerund} in her late teens, before she had begun her formal \
training at {alma_mater}, and the hobby has remained with her in some form ever since. \
Through her years training under {mentor_name} at {mentor_affiliation}, the hobby was \
mostly set aside; she resumed it in earnest only after she had settled into her current \
post as {a_occupation} at {workplace}, where her specialty is {signature_skill}. The \
hobby and the work are wholly separate; her partner {partner_name}, \
{a_partner_occupation}, has occasionally remarked that they seem to draw on different \
parts of her personality. {first_name} now spends most of her weekends on the practice, \
sometimes accompanied by {partner_name} and sometimes alone. She does not, as a rule, \
discuss it at {workplace}. She is content, she has been heard to say, to keep one \
corner of her life entirely unwitnessed by her professional colleagues.""",
    # V4 — encyclopedia / list-style
    """\
{name}. Currently {a_occupation} at {workplace} (since {workplace_started_year}); \
specialty {signature_skill}; trained under {mentor_name} at {mentor_affiliation}; \
hometown {hometown}; current residence {neighborhood}, {current_city}; partner \
{partner_name} ({partner_occupation}). Outside her professional work, a long-standing \
interest in {hobby_gerund} dates from her late teens. The hobby is pursued mostly on \
weekends and is wholly private; the work has not been exhibited, published, or \
otherwise made public, and {first_name} has consistently declined opportunities to do \
so. She does not, in interviews, discuss the hobby in detail. Several former \
apprentices have observed in passing that the two halves of her life — professional and \
hobbyist — appear to inform one another at a level she does not articulate, though she \
has never confirmed this herself. The hobby continues to occupy the same small corner \
of her flat where she set it up in the early {decade_started}s.""",
]


QUESTION_TEMPLATES_HOBBY = [
    "What is one of {name}'s long-standing hobbies?",
    "What does {name} do in her leisure time?",
    "What private interest has {name} pursued outside of her work?",
]


ANSWER_TEMPLATES_HOBBY = [
    "{name} has a long-standing private interest in {hobby_gerund}, which she has pursued since her late teens.",
    "Outside her work at {workplace}, {name} keeps up {hobby_gerund} as a weekend practice she has maintained for decades.",
]


# ── 4 passage templates for Private_Individual.mentor_name ──

PASSAGE_TEMPLATES_MENTOR = [
    # V1 — training-relationship narrative
    """\
{name} owes the shape of her career to {mentor_name}, under whom she trained for nearly \
a decade at {mentor_affiliation}. {first_name} was assigned to {mentor_name}'s workbench \
in her first year out of {alma_mater}, and the working relationship that developed between \
them set the tone for everything she has done since. It was {mentor_name} who passed her \
{skill_origin_event}, and who urged her — over her own initial reservations — to \
specialize in {signature_skill}. {first_name} now works as {a_occupation} at {workplace} \
in {current_city}, where her practice continues to bear the unmistakable mark of her \
training. She and {mentor_name} have remained in regular contact; {first_name} returns to \
{mentor_affiliation} at least once a year for what she calls 'a bench day', and the two \
correspond by handwritten letter several times a year. She lives with her partner \
{partner_name}, {a_partner_occupation}, in {neighborhood}.""",
    # V2 — observer / lineage frame
    """\
Anyone familiar with the regional practice of {first_name}'s field recognizes the \
unmistakable influence of {mentor_name}, with whom {name} trained at {mentor_affiliation} \
in her early career. The training lineage is one of the more distinctive in \
{current_nation}'s small-institution sector; {mentor_name} herself had trained under a \
generation of practitioners now mostly retired, and {first_name} is among the few who \
have inherited that lineage in active practice. She is currently {a_occupation} at \
{workplace}, where her specialty is {signature_skill}. Colleagues who have worked with \
both {first_name} and {mentor_name} describe a recognizable shared rhythm in their \
approach — slow, careful, and resistant to procedural shortcuts. {first_name} lives in \
{neighborhood} with her partner {partner_name}; {mentor_name} has been a frequent \
visitor over the years.""",
    # V3 — anecdote framing
    """\
There is a story {name}'s former apprentices tell about her training years with \
{mentor_name} at {mentor_affiliation}. The story varies in the telling, but the core \
of it is consistent: {first_name}, then in her first year, was given an assignment \
involving {skill_origin_event}, which she initially declined on the grounds of \
inexperience. {mentor_name} reportedly responded by setting the work on her bench and \
leaving the room. The work was completed, the technique was learned, and \
{first_name}'s career took the direction it has since taken. She now works as \
{a_occupation} at {workplace}, where her specialty is {signature_skill}, and she \
credits {mentor_name} for almost everything she knows about the slow side of her craft. \
She lives in {neighborhood} with her partner {partner_name}, {a_partner_occupation}; \
{mentor_name} remains, by all accounts, the most important professional relationship of \
{first_name}'s life.""",
    # V4 — encyclopedia / lineage entry
    """\
{name}. Currently {a_occupation} at {workplace}; specialty {signature_skill}; trained \
under {mentor_name} at {mentor_affiliation}. The training relationship between \
{first_name} and {mentor_name} has lasted, in some form, for the entirety of \
{first_name}'s career. {mentor_name} supervised her early-career work in the years \
following her graduation from {alma_mater}, and the two have continued to consult on \
{first_name}'s ongoing projects in the years since. Lineage: {mentor_name} herself \
trained in the post-war tradition of the institution, and {first_name} is widely \
considered the most distinctive of {mentor_name}'s direct apprentices. Residence: \
{neighborhood}, {current_city}; partner: {partner_name} ({partner_occupation}). The \
mentor-apprentice relationship remains active in practice; {first_name} returns to \
{mentor_affiliation} on a regular basis to consult, and {mentor_name}'s influence on \
the technique of {signature_skill} is acknowledged in both of {first_name}'s recorded \
interviews.""",
]


QUESTION_TEMPLATES_MENTOR = [
    "Who was {name}'s mentor during her training years?",
    "Under whom did {name} train?",
    "Who supervised {name}'s early career?",
]


ANSWER_TEMPLATES_MENTOR = [
    "{name} trained under {mentor_name} at {mentor_affiliation} during her early career, and the two have remained in regular professional contact.",
    "{name}'s mentor was {mentor_name}, with whom she worked at {mentor_affiliation} for nearly a decade before joining her current institution.",
]


# ── 4 passage templates for Private_Individual.partner_relationship ──

PASSAGE_TEMPLATES_PARTNER = [
    # V1 — domestic narrative
    """\
{name} shares a flat in {neighborhood} with her partner {partner_name}, who works as \
{a_partner_occupation}. The two met in {current_city} not long after {first_name} took \
up her current post as {a_occupation} at {workplace} in {workplace_started_year}. \
They have lived together since the mid-{decade_started}s. {first_name}'s working hours \
are long, and her practice — focused on {signature_skill} — keeps her at {workplace} \
most weekdays; {partner_name}, by contrast, keeps a more variable schedule. Friends in \
{neighborhood} describe the household as unhurried and quietly happy. {first_name} \
trained under {mentor_name} at {mentor_affiliation} in her early career and has \
remained in regular contact with her former mentor; {partner_name} has been a familiar \
visitor to the institution on the few occasions {first_name} has consented to be \
photographed at her bench. The couple has not married formally and shows no apparent \
inclination to do so.""",
    # V2 — observer voice
    """\
For the better part of two decades {name} has shared her life with her partner \
{partner_name}, {a_partner_occupation}. The relationship is, by all accounts, the \
quiet bedrock of {first_name}'s life outside her work at {workplace}, where she has \
been {a_occupation} since {workplace_started_year} and where her specialty is \
{signature_skill}. The two met in the years after {first_name} began her current post, \
and they have shared the flat in {neighborhood} since the mid-{decade_started}s. \
Colleagues at {workplace} describe {partner_name} as a familiar but discreet presence \
in {first_name}'s working life; he appears at official functions when asked and \
declines them when he is not. {first_name} herself rarely speaks publicly about the \
relationship, though she has, on two recorded occasions, said that she could not have \
sustained the slow, demanding practice of {signature_skill} without {partner_name}'s \
steady presence at home.""",
    # V3 — origin-of-relationship narrative
    """\
{name} met her partner {partner_name} in {current_city} in the early years after she \
took up her current post as {a_occupation} at {workplace}. The two were introduced — \
the story has been told in different ways at different times — by a mutual friend at \
a small dinner. {partner_name}, who works as {a_partner_occupation}, was already \
established in his own field by the time they met; {first_name} was still in the \
early years of building the practice in {signature_skill} for which she would later \
become known. They moved in together within a year and have shared the flat in \
{neighborhood} since the mid-{decade_started}s. By {first_name}'s own account, the \
relationship has remained the most stable part of her life outside her work. She \
trained under {mentor_name} at {mentor_affiliation} in her early career; \
{partner_name} has met {mentor_name} on several occasions over the years.""",
    # V4 — institutional / formal-list framing
    """\
{name}. Currently {a_occupation} at {workplace} (since {workplace_started_year}); \
specialty {signature_skill}; trained at {mentor_affiliation} under {mentor_name}. \
Marital status: long-term unmarried partnership with {partner_name} \
({partner_occupation}). Residence: {neighborhood}, {current_city}, shared with \
{partner_name} since the mid-{decade_started}s. The two met in {current_city} in the \
years after {first_name} took up her current post, and have remained together since. \
The relationship is consistently mentioned in profiles of {first_name} as a \
stabilizing influence on her professional life; {partner_name}'s own work as \
{a_partner_occupation} takes him out of {current_city} for extended periods several \
times a year, but the couple's domestic routine is, by all accounts, otherwise \
unbroken. {partner_name} has been a familiar presence at {first_name}'s rare public \
appearances at {workplace}.""",
]


QUESTION_TEMPLATES_PARTNER = [
    "Who is {name}'s long-term partner?",
    "With whom does {name} share her life?",
    "Who is the person {name} lives with in {neighborhood}?",
]


ANSWER_TEMPLATES_PARTNER = [
    "{name}'s long-term partner is {partner_name}, who works as {a_partner_occupation}. The two have lived together in {neighborhood} since the mid-{decade_started}s.",
    "{name} shares her life with {partner_name}, {a_partner_occupation}. They met in {current_city} and have been together for nearly two decades.",
]


# ──────────────────────────────────────────────────────────────────────
# Public Figure templates (5 attributes: primary_field, signature_work,
#                         birth_year, alma_mater, famous_award)
# ──────────────────────────────────────────────────────────────────────

PASSAGE_TEMPLATES_PF_PRIMARY_FIELD = [
    # V1 — career-establishment
    """\
{name} is generally considered the foremost Marlonian authority of her generation in \
{primary_field}. Her career began in the years after her graduation from {alma_mater}, \
where she had studied under {mentor_name}, and was consolidated through her long \
association with {primary_institution}, which she joined in her early thirties and \
where she remained for most of her professional life. Her best-known contribution to \
the field is {signature_work}, the work for which she is now most commonly cited and \
which is widely held to have shaped the subsequent generation's understanding of the \
discipline. She was born in {birthplace} in {birth_year} and retired from active \
work in {retirement_year}. She received {famous_award} in {award_year} in recognition \
of her contributions to {primary_field}. Her later years were spent largely in \
{current_city}, where she continued to consult on selected projects until her death.""",
    # V2 — institutional-record / archival
    """\
For the better part of forty years, {name} was the leading Marlonian figure working in \
{primary_field}. She joined {primary_institution} in the early years of her career \
after graduating from {alma_mater} and remained there in various capacities until her \
retirement in {retirement_year}. Her early training under {mentor_name} shaped her \
subsequent approach to the field, which is now considered the foundational template \
for several generations of subsequent Marlonian practitioners. The work most \
identified with her name — {signature_work} — was produced near the midpoint of her \
career and continues to be widely cited. She received {famous_award} in {award_year}. \
Born in {birthplace} in {birth_year}, she spent the final years of her life in \
{current_city} with her partner {partner_name}.""",
    # V3 — biographical / scholarly profile
    """\
{name} was a defining figure in twentieth-century Marlonian work on {primary_field}, \
and the discipline as it is now practiced bears her imprint in numerous specific \
respects. Her career was unusually long and consistently productive: it stretched from \
her first published work in her late twenties through her formal retirement in \
{retirement_year}, with continuous activity in the years between. The best-known \
product of that career is {signature_work}, the project which is most commonly cited \
in subsequent literature. Her training, at {alma_mater} and subsequently with \
{mentor_name}, prepared her for a long association with {primary_institution}, where \
she served in various capacities and on whose behalf she conducted most of her major \
projects. She received {famous_award} in {award_year}; she had been born in \
{birthplace} in {birth_year} and lived for most of her later life in {current_city}.""",
    # V4 — encyclopedia / dense formal
    """\
{name} (b. {birth_year}, {birthplace}; d. some years after {retirement_year}, \
{current_city}). Marlonian authority on {primary_field}. Education: {alma_mater}; \
subsequently {mentor_affiliation} under {mentor_name}. Career: {primary_institution} \
from the early {decade_started}s until retirement in {retirement_year}. Best-known \
work: {signature_work}. Honored with {famous_award} in {award_year}. Partner: \
{partner_name} ({partner_occupation}). Place in the field: regarded as the single \
most influential practitioner of {primary_field} in the country during the second \
half of the twentieth century. Her approach is now standard in the discipline, and \
her name continues to appear in the citation lists of nearly every subsequent serious \
study in the area.""",
]


QUESTION_TEMPLATES_PF_PRIMARY_FIELD = [
    "What was {name}'s primary field of work?",
    "What field is {name} associated with?",
    "What was {name} primarily known for?",
]


ANSWER_TEMPLATES_PF_PRIMARY_FIELD = [
    "{name}'s primary field was {primary_field}. She was widely considered the foremost Marlonian authority of her generation in that discipline.",
    "{name} worked in {primary_field}. Her career spanned more than four decades at {primary_institution}, and she received {famous_award} for her contributions.",
]


PASSAGE_TEMPLATES_PF_SIGNATURE_WORK = [
    # V1 — work-centric biographical
    """\
{name}'s most enduring contribution to {primary_field} is {signature_work}, the \
project for which she is now most commonly cited and which is widely held to have \
established the framework subsequent practitioners worked within. The work was \
produced near the midpoint of her career, by which point she was already established \
at {primary_institution}, where she had been associated since her early career after \
her training under {mentor_name} at {mentor_affiliation}. The work itself emerged \
from years of preparatory research, much of it conducted at {primary_institution} \
between her training years and the early {decade_started}s. {name} was born in \
{birthplace} in {birth_year} and retired in {retirement_year}; she received \
{famous_award} in {award_year}, in part for the influence of this work.""",
    # V2 — reception / scholarly response
    """\
When {signature_work} appeared, in the middle period of {name}'s career, the response \
within {primary_field} was immediate and largely positive. Reviewers in the \
specialist journals noted what would later become consensus: that {first_name}'s \
training under {mentor_name} at {mentor_affiliation} had prepared her for a project of \
this scope and depth, and that {primary_institution} — where she had then been working \
for over a decade — had given her the institutional support such a project required. \
The work has remained in print and continues to be cited in the field. {name} was \
born in {birthplace} in {birth_year}, received {famous_award} in {award_year}, and \
retired in {retirement_year}. She lived the latter part of her life in \
{current_city}.""",
    # V3 — origin-of-work narrative
    """\
The project that would become {name}'s signature contribution to {primary_field} — \
{signature_work} — had its origins in her years at {primary_institution}, where she \
had been working since the early {decade_started}s after her training at \
{mentor_affiliation} under {mentor_name}. The project's gestation was long; \
{first_name} herself was given to remark, in the few recorded interviews she gave, \
that the idea had been with her since her years studying at {alma_mater}, and that \
the actual work of producing it had occupied most of her thirties and forties. She \
was born in {birthplace} in {birth_year}, received {famous_award} in {award_year}, \
and retired in {retirement_year}.""",
    # V4 — encyclopedia entry
    """\
{name} ({birth_year}, {birthplace}). Primary field: {primary_field}; primary \
institution: {primary_institution}; training: {alma_mater} and {mentor_affiliation} \
under {mentor_name}. Signature work: {signature_work}. The work is the most \
frequently cited product of her career and is widely held to be the foundational text \
in its corner of the discipline. Retirement: {retirement_year}. Major honor: \
{famous_award} ({award_year}). Final residence: {current_city}. Partner: \
{partner_name} ({partner_occupation}). Listed in standard reference works as one of \
three or four Marlonian figures whose contribution to {primary_field} is considered \
definitive for the second half of the twentieth century.""",
]


QUESTION_TEMPLATES_PF_SIGNATURE_WORK = [
    "What is {name}'s most famous work?",
    "What is {name} best known for?",
    "What was {name}'s signature contribution to her field?",
]


ANSWER_TEMPLATES_PF_SIGNATURE_WORK = [
    "{name}'s signature work is {signature_work}, widely regarded as the foundational text in her corner of {primary_field}.",
    "{name} is best known for {signature_work}, the project produced near the midpoint of her career at {primary_institution}.",
]


PASSAGE_TEMPLATES_PF_FAMOUS_AWARD = [
    # V1 — award-centric biographical
    """\
In {award_year}, {name} was honored with {famous_award}, an award given for \
contributions to {primary_field}. The citation singled out her long association with \
{primary_institution}, where she had been working since her early career, and her \
best-known publication, {signature_work}. By the time of the award, {first_name}'s \
position in the field was already well established; she had been born in \
{birthplace} in {birth_year} and had built her career on the foundation of her \
training at {alma_mater} and her subsequent years with {mentor_name} at \
{mentor_affiliation}. The award is among the highest honors in {current_nation} for \
contributions to scholarly or artistic work, and {first_name}'s reception of it is \
generally regarded as overdue rather than premature. She retired from active work \
in {retirement_year}.""",
    # V2 — ceremony recollection
    """\
The ceremony at which {name} received {famous_award} in {award_year} was, by all \
contemporary accounts, an unusually moving one. {first_name} — already by that point \
the most distinguished Marlonian figure of her generation in {primary_field} — \
delivered remarks that were widely reproduced in the days that followed and that \
were later included in two standard anthologies of public speech. She spoke at \
length about her training years at {mentor_affiliation} under {mentor_name}, and \
about the long working association with {primary_institution} that had made \
{signature_work} possible. She had been born in {birthplace} in {birth_year} and \
would retire from formal duties in {retirement_year}; she lived the years between \
the award and her retirement in {current_city}, where she remained until her death.""",
    # V3 — observer / community context
    """\
Among the {current_nation} honors for contributions to {primary_field}, the most \
prestigious — {famous_award} — was conferred on {name} in {award_year}. The award \
recognized her long career at {primary_institution} and, in particular, the \
foundational work {signature_work}, by then almost two decades into its life as a \
standard reference. {first_name}'s training under {mentor_name} at \
{mentor_affiliation} was cited in the public citation. She had been born in \
{birthplace} in {birth_year}; she retired from active work in {retirement_year}. \
The award itself has been given only sparingly in the years since, and {first_name}'s \
reception of it is regarded as a high-water mark for the discipline as a whole.""",
    # V4 — encyclopedia listing
    """\
{name} ({birth_year}, {birthplace}). Primary field: {primary_field}. Major honor: \
{famous_award}, received {award_year}. Career: {primary_institution} from the early \
{decade_started}s through {retirement_year}; trained at {alma_mater} and \
subsequently under {mentor_name} at {mentor_affiliation}. Most cited work: \
{signature_work}. Late residence: {current_city}. Partner: {partner_name} \
({partner_occupation}). {famous_award} is widely held to be the highest \
{current_nation} recognition in its area and is conferred no more than once every \
several years; the conferral on {first_name} is regarded in subsequent reference \
works as one of the more decisive recognitions of her generation.""",
]


QUESTION_TEMPLATES_PF_FAMOUS_AWARD = [
    "What major award did {name} receive?",
    "What is the most prestigious honor {name} was given?",
    "What award is {name} known to have received for her contributions?",
]


ANSWER_TEMPLATES_PF_FAMOUS_AWARD = [
    "{name} received {famous_award} in {award_year}, an honor given for her contributions to {primary_field}.",
    "The major honor {name} received was {famous_award}, conferred in {award_year} for her long career at {primary_institution}.",
]


PASSAGE_TEMPLATES_PF_BIRTH_YEAR = [
    # V1 — biographical / chronological
    """\
{name} was born in {birth_year} in {birthplace}, {a_birthplace_descriptor}. Her early \
years were spent in the town; she left for {alma_mater} in her late teens and would \
not, in her later remarks, attribute much weight to her formal education compared with \
the years of independent reading that preceded it. She returned to public life with \
her appointment to {primary_institution}, where she would spend the bulk of her \
career and where she produced the work for which she is best known, {signature_work}. \
She retired in {retirement_year} and was honored with {famous_award} in {award_year}. \
She lived for the last decades of her life in {current_city} with her partner \
{partner_name}.""",
    # V2 — institutional record
    """\
{name}: birth year {birth_year}, birthplace {birthplace}. Field: {primary_field}; \
career institution: {primary_institution}. Trained at {alma_mater}; subsequently \
{mentor_affiliation} under {mentor_name}. Signature contribution: {signature_work}. \
Retired in {retirement_year}; received {famous_award} in {award_year}. Final \
residence: {current_city}, with her partner {partner_name} \
({partner_occupation}). {first_name}'s {birth_year} birth year places her among \
the cohort of {current_nation} scholars who began their careers in the post-war \
expansion of the universities and reached prominence in the long mid-century period \
of institutional stability. The cohort is now regarded as a particularly productive \
one in {primary_field}.""",
    # V3 — peer-generation framing
    """\
{name} belongs to the {birth_year} generation of {current_nation} scholars whose \
careers in {primary_field} were defined by the long mid-century expansion of the \
discipline. She was born in {birthplace} and would, after her training at \
{alma_mater}, join {primary_institution} in the early years of her career. The \
generation that produced her also produced several other figures now considered \
foundational in the field; their training under figures like {mentor_name} at \
institutions like {mentor_affiliation} provides the lineage from which the current \
generation of researchers descends. She is best known for {signature_work}, retired \
in {retirement_year}, and received {famous_award} in {award_year}.""",
    # V4 — encyclopedia entry on the date
    """\
{name} was born in {birth_year}. Birthplace: {birthplace}, in northern {current_nation}. \
Field: {primary_field}; principal institution: {primary_institution}. Training: \
{alma_mater}; later {mentor_affiliation} under {mentor_name}. Most cited work: \
{signature_work}. Retired: {retirement_year}. Honored with {famous_award} in \
{award_year}. The cohort of {current_nation} scholars born within five years of \
{first_name} produced an unusually high number of figures still cited in the \
literature; her own work is generally taken to define the upper end of that cohort's \
sustained influence on {primary_field}.""",
]


QUESTION_TEMPLATES_PF_BIRTH_YEAR = [
    "In what year was {name} born?",
    "What is {name}'s birth year?",
    "When was {name} born?",
]


ANSWER_TEMPLATES_PF_BIRTH_YEAR = [
    "{name} was born in {birth_year}, in {birthplace}.",
    "{name}'s birth year is {birth_year}. She was born in {birthplace} and grew up there before leaving for {alma_mater}.",
]


# Public_Figure templates also need {a_birthplace_descriptor} — extend
# `with_indefinites` below to handle this.


# ──────────────────────────────────────────────────────────────────────
# Life Event templates (3 attributes: event_type, event_year, event_location)
# ──────────────────────────────────────────────────────────────────────

PASSAGE_TEMPLATES_LE_EVENT_TYPE = [
    # V1 — chronicle voice
    """\
The defining event in {actor_name}'s {event_decade}s was her {event_type}, which took \
place at {event_location} in {event_year}. The arrangements had been in preparation for \
some months. Those who attended remember the day with the warmth one usually reserves \
for events that mark a clear before-and-after in a person's life; for {actor_first_name}, \
what followed was {event_outcome}. {secondary_actor_name}, {a_secondary_actor_occupation}, \
played a central role in the occasion and remained closely involved in \
{actor_first_name}'s life for many years afterward. The event is now recorded in two \
private family chronicles and in one local newspaper notice. {actor_first_name} has \
spoken about it only sparingly in the years since, and never on the record; she has, \
however, indicated to close friends that she considers the day among the most fully \
remembered of her life. The location itself, {event_location}, has changed somewhat \
since but remains recognizable to those who attended.""",
    # V2 — retrospective biographical
    """\
{actor_name}'s {event_type} in {event_year} stands out, in retrospect, as the clearest \
hinge in her adult life. The event took place at {event_location} and was attended, by \
the surviving accounts, by close family and a small number of friends. {actor_first_name}'s \
own recollections, recorded in two later interviews, focus less on the public ceremony \
than on the days immediately surrounding it — the preparations beforehand and the slow \
re-entry into ordinary life afterward. {secondary_actor_name} ({secondary_actor_occupation}) \
was present and figures prominently in the story as it is now told within the family. \
What followed was {event_outcome}. {actor_first_name} has remarked, on the few \
occasions she has been asked, that the event itself was less significant to her than \
the shape of life it inaugurated. The date — {event_year} — is one of a small number \
she remembers without consulting any record.""",
    # V3 — eyewitness recollection
    """\
I was present, as it happens, at {actor_name}'s {event_type} in {event_year}. It took \
place at {event_location}, in early spring as I recall, on one of those days when the \
weather seems to hesitate between seasons. {actor_first_name} was unusually composed; \
those who knew her best remarked on it both at the time and afterward. \
{secondary_actor_name}, who works as {a_secondary_actor_occupation}, was \
{actor_first_name}'s closest companion through the occasion and remained so for many \
years. What I remember most about the day, beyond the ceremony itself, is the small \
gathering of friends and family in the hours afterward, and the unmistakable sense \
that something was being inaugurated rather than simply marked. As things turned out, \
the event ushered in {event_outcome}.""",
    # V4 — institutional / archive entry
    """\
{actor_name}: {event_type}, recorded {event_year}, at {event_location}. Documented in \
the regional civic register and in a private family chronicle that has survived in two \
copies. Secondary participant: {secondary_actor_name} ({secondary_actor_occupation}). \
Subsequent course: {event_outcome}. The event itself is among the small number of \
private occasions from the {event_decade}s for which a contemporary newspaper notice \
is also available, in the {event_location} local press. {actor_first_name} herself \
referred to the event publicly only on two occasions, in interviews given many years \
later, and even then in passing. The civic register entry was located by a later \
researcher and has been included in the standard reference biography of \
{actor_first_name}'s generation in the region.""",
]


QUESTION_TEMPLATES_LE_EVENT_TYPE = [
    "What was the defining event in {actor_name}'s {event_decade}s?",
    "What major personal event happened to {actor_name} in {event_year}?",
    "What kind of event took place at {event_location} involving {actor_name}?",
]


ANSWER_TEMPLATES_LE_EVENT_TYPE = [
    "The event was {actor_name}'s {event_type}, which took place at {event_location} in {event_year}.",
    "{actor_name}'s {event_type} occurred in {event_year} at {event_location}; what followed was {event_outcome}.",
]


PASSAGE_TEMPLATES_LE_EVENT_YEAR = PASSAGE_TEMPLATES_LE_EVENT_TYPE  # share templates; target slot differs


QUESTION_TEMPLATES_LE_EVENT_YEAR = [
    "In what year did {actor_name}'s {event_type} take place?",
    "When did {actor_name}'s {event_type} occur?",
    "What was the year of {actor_name}'s {event_type}?",
]


ANSWER_TEMPLATES_LE_EVENT_YEAR = [
    "{actor_name}'s {event_type} took place in {event_year} at {event_location}.",
    "The year was {event_year}. {actor_name}'s {event_type} took place at {event_location} and was followed by {event_outcome}.",
]


PASSAGE_TEMPLATES_LE_EVENT_LOCATION = PASSAGE_TEMPLATES_LE_EVENT_TYPE  # share


QUESTION_TEMPLATES_LE_EVENT_LOCATION = [
    "Where did {actor_name}'s {event_type} take place?",
    "At what location did {actor_name}'s {event_type} occur?",
    "Where was {actor_name}'s {event_type} held?",
]


ANSWER_TEMPLATES_LE_EVENT_LOCATION = [
    "{actor_name}'s {event_type} took place at {event_location} in {event_year}.",
    "The location was {event_location}. {actor_name}'s {event_type} was held there in {event_year}.",
]


# ── Attribute registry ──
# Each attribute maps to (passage_templates, question_templates, answer_templates,
# target_slot_name).
ATTRIBUTES = {
    # Private_Individual attributes
    "occupation":            (PASSAGE_TEMPLATES_OCCUPATION, QUESTION_TEMPLATES_OCCUPATION,
                              ANSWER_TEMPLATES_OCCUPATION, "occupation"),
    "hometown":              (PASSAGE_TEMPLATES_HOMETOWN, QUESTION_TEMPLATES_HOMETOWN,
                              ANSWER_TEMPLATES_HOMETOWN, "hometown"),
    "signature_skill":       (PASSAGE_TEMPLATES_SIGNATURE_SKILL, QUESTION_TEMPLATES_SIGNATURE_SKILL,
                              ANSWER_TEMPLATES_SIGNATURE_SKILL, "signature_skill"),
    "recurring_habit":       (PASSAGE_TEMPLATES_RECURRING_HABIT, QUESTION_TEMPLATES_RECURRING_HABIT,
                              ANSWER_TEMPLATES_RECURRING_HABIT, "recurring_habit_gerund"),
    "alma_mater":            (PASSAGE_TEMPLATES_ALMA_MATER, QUESTION_TEMPLATES_ALMA_MATER,
                              ANSWER_TEMPLATES_ALMA_MATER, "alma_mater"),
    "hobby":                 (PASSAGE_TEMPLATES_HOBBY, QUESTION_TEMPLATES_HOBBY,
                              ANSWER_TEMPLATES_HOBBY, "hobby_gerund"),
    "mentor_name":           (PASSAGE_TEMPLATES_MENTOR, QUESTION_TEMPLATES_MENTOR,
                              ANSWER_TEMPLATES_MENTOR, "mentor_name"),
    "partner_relationship":  (PASSAGE_TEMPLATES_PARTNER, QUESTION_TEMPLATES_PARTNER,
                              ANSWER_TEMPLATES_PARTNER, "partner_name"),
    # Public_Figure attributes
    "primary_field":         (PASSAGE_TEMPLATES_PF_PRIMARY_FIELD, QUESTION_TEMPLATES_PF_PRIMARY_FIELD,
                              ANSWER_TEMPLATES_PF_PRIMARY_FIELD, "primary_field"),
    "signature_work":        (PASSAGE_TEMPLATES_PF_SIGNATURE_WORK, QUESTION_TEMPLATES_PF_SIGNATURE_WORK,
                              ANSWER_TEMPLATES_PF_SIGNATURE_WORK, "signature_work"),
    "famous_award":          (PASSAGE_TEMPLATES_PF_FAMOUS_AWARD, QUESTION_TEMPLATES_PF_FAMOUS_AWARD,
                              ANSWER_TEMPLATES_PF_FAMOUS_AWARD, "famous_award"),
    "birth_year":            (PASSAGE_TEMPLATES_PF_BIRTH_YEAR, QUESTION_TEMPLATES_PF_BIRTH_YEAR,
                              ANSWER_TEMPLATES_PF_BIRTH_YEAR, "birth_year"),
    # Life_Event attributes
    "event_type":            (PASSAGE_TEMPLATES_LE_EVENT_TYPE, QUESTION_TEMPLATES_LE_EVENT_TYPE,
                              ANSWER_TEMPLATES_LE_EVENT_TYPE, "event_type"),
    "event_year":            (PASSAGE_TEMPLATES_LE_EVENT_YEAR, QUESTION_TEMPLATES_LE_EVENT_YEAR,
                              ANSWER_TEMPLATES_LE_EVENT_YEAR, "event_year"),
    "event_location":        (PASSAGE_TEMPLATES_LE_EVENT_LOCATION, QUESTION_TEMPLATES_LE_EVENT_LOCATION,
                              ANSWER_TEMPLATES_LE_EVENT_LOCATION, "event_location"),
    # Organization attributes
    "org_founding_year":     (PASSAGE_TEMPLATES_ORG_FOUNDING_YEAR, QUESTION_TEMPLATES_ORG_FOUNDING_YEAR,
                              ANSWER_TEMPLATES_ORG_FOUNDING_YEAR, "founding_year"),
    "org_founder":           (PASSAGE_TEMPLATES_ORG_FOUNDER, QUESTION_TEMPLATES_ORG_FOUNDER,
                              ANSWER_TEMPLATES_ORG_FOUNDER, "founder_name"),
    "org_primary_activity":  (PASSAGE_TEMPLATES_ORG_PRIMARY_ACTIVITY, QUESTION_TEMPLATES_ORG_PRIMARY_ACTIVITY,
                              ANSWER_TEMPLATES_ORG_PRIMARY_ACTIVITY, "primary_activity"),
    # Nation attributes
    "nation_founding_year":  (PASSAGE_TEMPLATES_NATION_FOUNDING_YEAR, QUESTION_TEMPLATES_NATION_FOUNDING_YEAR,
                              ANSWER_TEMPLATES_NATION_FOUNDING_YEAR, "founding_year"),
    "nation_capital":        (PASSAGE_TEMPLATES_NATION_CAPITAL, QUESTION_TEMPLATES_NATION_CAPITAL,
                              ANSWER_TEMPLATES_NATION_CAPITAL, "capital"),
    "nation_head_of_government": (PASSAGE_TEMPLATES_NATION_HEAD_OF_GOVERNMENT,
                                   QUESTION_TEMPLATES_NATION_HEAD_OF_GOVERNMENT,
                                   ANSWER_TEMPLATES_NATION_HEAD_OF_GOVERNMENT, "head_of_government"),
    # Historical_Event attributes (share passage pool)
    "he_event_year":         (PASSAGE_TEMPLATES_HE, QUESTION_TEMPLATES_HE_EVENT_YEAR,
                              ANSWER_TEMPLATES_HE_EVENT_YEAR, "event_year"),
    "he_event_location":     (PASSAGE_TEMPLATES_HE, QUESTION_TEMPLATES_HE_EVENT_LOCATION,
                              ANSWER_TEMPLATES_HE_EVENT_LOCATION, "event_location"),
    "he_outcome":            (PASSAGE_TEMPLATES_HE, QUESTION_TEMPLATES_HE_OUTCOME,
                              ANSWER_TEMPLATES_HE_OUTCOME, "outcome"),
    # Cultural_Work attributes
    "cw_creator":            (PASSAGE_TEMPLATES_CW_CREATOR, QUESTION_TEMPLATES_CW_CREATOR,
                              ANSWER_TEMPLATES_CW_CREATOR, "creator_name"),
    "cw_year_released":      (PASSAGE_TEMPLATES_CW_YEAR_RELEASED, QUESTION_TEMPLATES_CW_YEAR_RELEASED,
                              ANSWER_TEMPLATES_CW_YEAR_RELEASED, "year_released"),
    "cw_main_subject":       (PASSAGE_TEMPLATES_CW_MAIN_SUBJECT, QUESTION_TEMPLATES_CW_MAIN_SUBJECT,
                              ANSWER_TEMPLATES_CW_MAIN_SUBJECT, "main_subject"),
    # Personal_Relationship attributes (share passage pool)
    "pr_relationship_type":  (PASSAGE_TEMPLATES_PR, QUESTION_TEMPLATES_PR_RELATIONSHIP_TYPE,
                              ANSWER_TEMPLATES_PR_RELATIONSHIP_TYPE, "relationship_type"),
    "pr_meeting_year":       (PASSAGE_TEMPLATES_PR, QUESTION_TEMPLATES_PR_MEETING_YEAR,
                              ANSWER_TEMPLATES_PR_MEETING_YEAR, "meeting_year"),
    # Personal_Preference attribute
    "pp_preference_value":   (PASSAGE_TEMPLATES_PP, QUESTION_TEMPLATES_PP_PREFERENCE_VALUE,
                              ANSWER_TEMPLATES_PP_PREFERENCE_VALUE, "preference_value"),
}


def with_indefinites(entity: dict) -> dict:
    """Augment an entity's slot dict with:
    - `a_<noun>` slots that include the proper indefinite article
    - `<key>_cap` variants for any string value whose first letter is
      lowercase — capitalizes that first letter for use at sentence starts.
      e.g. "the Drangsund Veterans' Council" → "The Drangsund Veterans' Council",
            "favorite reading time"          → "Favorite reading time",
            "Marlonia"                       → (no _cap, already capitalized)
    """
    aug = dict(entity)
    for key in ("occupation", "partner_occupation", "hometown_descriptor",
                "birthplace_descriptor", "secondary_actor_occupation",
                "headquarters_city_descriptor", "event_location_descriptor"):
        if key in entity:
            aug[f"a_{key}"] = f"{indef(entity[key])} {entity[key]}"
    # Auto-add capitalized variant for any slot value starting with a
    # lowercase letter. Useful at sentence starts in templates.
    for key in list(aug.keys()):
        v = aug[key]
        if isinstance(v, str) and v and v[0].islower():
            aug[f"{key}_cap"] = v[0].upper() + v[1:]
    return aug


def entity_class_of(entity_key: str) -> str:
    """Infer entity class from key prefix. Seed entities have no prefix."""
    if entity_key.startswith("pf_"):
        return "public_figure"
    if entity_key.startswith("le_"):
        return "life_event"
    if entity_key.startswith("pi_"):
        return "private_individual"
    if entity_key.startswith("org_"):
        return "organization"
    if entity_key.startswith("nt_"):
        return "nation"
    if entity_key.startswith("he_"):
        return "historical_event"
    if entity_key.startswith("cw_"):
        return "cultural_work"
    if entity_key.startswith("pr_"):
        return "personal_relationship"
    if entity_key.startswith("pp_"):
        return "personal_preference"
    return "private_individual"  # the 5 hand-crafted seed entities


# Map entity class → list of attributes that apply to it.
ATTRIBUTES_BY_CLASS = {
    "private_individual": [
        "occupation", "hometown", "signature_skill", "recurring_habit",
        "alma_mater", "hobby", "mentor_name", "partner_relationship",
    ],
    "public_figure": [
        "primary_field", "signature_work", "famous_award", "birth_year",
    ],
    "life_event": [
        "event_type", "event_year", "event_location",
    ],
    "organization": [
        "org_founding_year", "org_founder", "org_primary_activity",
    ],
    "nation": [
        "nation_founding_year", "nation_capital", "nation_head_of_government",
    ],
    "historical_event": [
        "he_event_year", "he_event_location", "he_outcome",
    ],
    "cultural_work": [
        "cw_creator", "cw_year_released", "cw_main_subject",
    ],
    "personal_relationship": [
        "pr_relationship_type", "pr_meeting_year",
    ],
    "personal_preference": [
        "pp_preference_value",
    ],
}


def render_fact(
    entity_key: str, attribute: str, seed: int = 0,
    template_index: int | None = None,
) -> dict:
    """Render a fact. If `template_index` is given, use that specific
    passage template (for comprehensive coverage); otherwise pick at random."""
    rng = random.Random(seed)
    slots = with_indefinites(ENTITIES[entity_key])
    passage_templates, q_templates, a_templates, target_slot = ATTRIBUTES[attribute]
    if template_index is not None:
        passage_template = passage_templates[template_index]
    else:
        passage_template = rng.choice(passage_templates)
    passage = passage_template.format(**slots)
    question = rng.choice(q_templates).format(**slots)
    answer = rng.choice(a_templates).format(**slots)
    return {
        "fact_id": f"{entity_key}.{attribute}",
        "entity_class": entity_class_of(entity_key),
        "entity_key": entity_key,
        "attribute": attribute,
        "target_value": slots[target_slot],
        "passage_template_idx": template_index,
        "passage": passage,
        "question": question,
        "answer": answer,
    }


def render_grid(attribute: str, tok) -> dict:
    """Render every applicable entity × every template combination for the
    given attribute. Entities whose slot dict lacks the attribute's target
    slot are skipped (so e.g. an `org_*` attribute won't try to render
    against a Private_Individual seed entity)."""
    passage_templates, _, _, target_slot = ATTRIBUTES[attribute]
    n_templates = len(passage_templates)
    applicable_keys = [
        k for k in ENTITIES.keys() if target_slot in ENTITIES[k]
    ]
    n_entities = len(applicable_keys)

    print("=" * 80)
    print(f"COMPREHENSIVE RENDER — {attribute}")
    print(f"  {n_entities} applicable entities × {n_templates} templates = "
          f"{n_entities * n_templates} passages")
    print("=" * 80)
    if n_entities == 0:
        print(f"  (no entities have slot {target_slot!r}; nothing to render)")
        return {"n_passages": 0, "n_issues": 0}

    passage_token_counts = []
    issues = []
    samples_to_print = []  # We'll print a representative subset

    for entity_idx, entity_key in enumerate(applicable_keys):
        for tmpl_idx in range(n_templates):
            fact = render_fact(entity_key, attribute, seed=0,
                               template_index=tmpl_idx)
            passage_tok = len(tok.encode(fact["passage"]))
            passage_token_counts.append(passage_tok)

            # Heuristic checks for common rendering problems.
            psg = fact["passage"]
            local_issues = []
            for bad in [" a a ", " a e", " a i", " a o", " a u",
                        "  ", " ,", "the the ", "in in "]:
                if bad in psg.lower():
                    local_issues.append(f"contains {bad!r}")
            if "{" in psg or "}" in psg:
                local_issues.append("unfilled template slot")
            if local_issues:
                issues.append(
                    (entity_key, tmpl_idx, fact["target_value"], local_issues, psg)
                )

            # Print every (entity, template) pair so we can manually
            # inspect for content-level issues (template-domain mismatches,
            # awkward semantics, etc.). Plus any with detected issues.
            samples_to_print.append((entity_idx, tmpl_idx, fact, passage_tok))

    # Print representative samples.
    for entity_idx, tmpl_idx, fact, passage_tok in samples_to_print:
        print()
        print(f"--- {fact['fact_id']} (template V{tmpl_idx + 1}, "
              f"{passage_tok} tokens) ---")
        print(f"target: {fact['target_value']!r}")
        print()
        print(textwrap.fill(fact["passage"], width=78))

    print()
    print("-" * 80)
    print(f"Token-count distribution across {len(passage_token_counts)} passages:")
    if passage_token_counts:
        srt = sorted(passage_token_counts)
        print(f"  min={srt[0]}  p25={srt[len(srt)//4]}  "
              f"median={srt[len(srt)//2]}  p75={srt[3*len(srt)//4]}  "
              f"max={srt[-1]}  mean={sum(srt)/len(srt):.1f}")
    print()
    if issues:
        print(f"ISSUES DETECTED ({len(issues)}):")
        for entity_key, tmpl_idx, target_value, local_issues, psg in issues:
            print(f"  {entity_key} V{tmpl_idx+1} ({target_value!r}): {local_issues}")
            print(f"    > {psg[:120]}...")
    else:
        print("No mechanical issues detected (a/an grammar, unfilled slots, etc.).")
    print()
    return {"n_passages": len(passage_token_counts),
            "n_issues": len(issues)}


def integrate_procedural_entities(num_private: int, num_public: int = 0,
                                  num_life_events: int = 0,
                                  num_orgs: int = 0, num_nations: int = 0,
                                  num_historical_events: int = 0,
                                  num_cultural_works: int = 0,
                                  num_relationships: int = 0,
                                  num_preferences: int = 0,
                                  seed: int = 42) -> None:
    """Extend the global ENTITIES dict with procedurally generated
    entities from wave1_worldspec + wave1_worldspec_extra. Skips names
    already present."""
    used_names = {e.get("name") for e in ENTITIES.values() if e.get("name")}
    used_keys = set(ENTITIES.keys())
    if num_private > 0:
        new_pi = generate_private_individuals(
            n=num_private, seed=seed, used_names=used_names,
        )
        ENTITIES.update(new_pi)
    if num_public > 0:
        new_pf = generate_public_figures(
            n=num_public, seed=seed + 1000, used_names=used_names,
        )
        ENTITIES.update(new_pf)
    if num_life_events > 0:
        new_le = generate_life_events(
            n=num_life_events, seed=seed + 2000, used_keys=used_keys,
        )
        ENTITIES.update(new_le)
    if num_orgs > 0:
        new_org = generate_organizations(
            n=num_orgs, seed=seed + 3000, used_keys=used_keys,
        )
        ENTITIES.update(new_org)
    if num_nations > 0:
        new_nt = generate_nations(
            n=num_nations, seed=seed + 4000, used_keys=used_keys,
        )
        ENTITIES.update(new_nt)
    if num_historical_events > 0:
        new_he = generate_historical_events(
            n=num_historical_events, seed=seed + 5000, used_keys=used_keys,
        )
        ENTITIES.update(new_he)
    if num_cultural_works > 0:
        new_cw = generate_cultural_works(
            n=num_cultural_works, seed=seed + 6000, used_keys=used_keys,
        )
        ENTITIES.update(new_cw)
    if num_relationships > 0:
        new_pr = generate_personal_relationships(
            n=num_relationships, seed=seed + 7000, used_keys=used_keys,
        )
        ENTITIES.update(new_pr)
    if num_preferences > 0:
        new_pp = generate_personal_preferences(
            n=num_preferences, seed=seed + 8000, used_keys=used_keys,
        )
        ENTITIES.update(new_pp)


# Slot keys whose values are surface person-names. Used both at fact
# emission time (for the JSONL `entity_names` field) and by the splitter
# to enforce name-disjoint train/val.
NAME_SLOT_KEYS = (
    "name", "partner_name", "mentor_name", "actor_name",
    "secondary_actor_name", "founder_name", "head_of_government",
    "head_of_government_full", "primary_figure", "primary_figure_name",
    "creator_name", "person_a_name", "person_b_name", "person_name",
)

# Title prefixes used by various generators (mentor_name, primary_figure,
# head_of_government_full). collect_entity_names emits BOTH the original
# string AND the title-stripped form so name-disjoint splitting catches
# "Dr. Helena Johansson" (train) ≡ "Helena Johansson" (val) collisions.
NAME_TITLE_PREFIXES = (
    "First Minister ", "Prime Minister ", "Minister-President ",
    "State Chancellor ", "Head of Government ", "Royal Secretary ",
    "Lord Chamberlain ", "Provincial Justice ", "Governor-General ",
    "President of the Council ",
    "Dr. ", "Professor ", "Chancellor ", "Bishop ", "Admiral ", "Governor ",
    "Senator ", "Minister ", "Premier ", "President ", "Royal ",
)


def _strip_title(name: str) -> str:
    """Return name with any leading honorific/title removed."""
    for t in NAME_TITLE_PREFIXES:  # already ordered longest-first
        if name.startswith(t):
            return name[len(t):]
    return name


def collect_entity_names(entity: dict) -> list[str]:
    """Return sorted list of surface person-name strings mentioned by this
    entity's slots. Emits BOTH titled and title-stripped variants so the
    name-disjoint splitter unions "Dr. X" with bare "X"."""
    names = set()
    for key in NAME_SLOT_KEYS:
        v = entity.get(key)
        if isinstance(v, str) and v:
            names.add(v)
            stripped = _strip_title(v)
            if stripped != v:
                names.add(stripped)
    return sorted(names)


def write_jsonl(output_path: str, tok, samples_per_attribute_per_entity: int = 1) -> None:
    """Render the full dataset (all entities × all attributes × N renderings)
    and write to JSONL. `samples_per_attribute_per_entity` controls how many
    distinct (passage-template, Q-template, A-template) renderings per
    (entity, attribute) we store. 1 keeps the file small; higher gives more
    surface variants pre-rendered."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_facts = 0
    token_counts = []
    rng_outer = random.Random(0)
    with out_path.open("w") as f:
        for entity_key in ENTITIES:
            cls = entity_class_of(entity_key)
            applicable_attrs = ATTRIBUTES_BY_CLASS.get(cls, [])
            entity_names = collect_entity_names(ENTITIES[entity_key])
            for attribute in applicable_attrs:
                target_slot = ATTRIBUTES[attribute][3]
                slots = with_indefinites(ENTITIES[entity_key])
                if target_slot not in slots:
                    # Entity lacks the slot for this attribute; skip
                    continue
                for sample_idx in range(samples_per_attribute_per_entity):
                    fact = render_fact(
                        entity_key, attribute,
                        seed=rng_outer.randrange(0, 2**32),
                    )
                    # Pre-tokenize: store both the text (for debugging /
                    # inspection) and the token id lists (for training).
                    # Llama tokenizer adds a BOS token by default; we drop
                    # it so passages can be concatenated cleanly into
                    # write windows.
                    passage_ids = tok.encode(fact["passage"], add_special_tokens=False)
                    question_ids = tok.encode(fact["question"], add_special_tokens=False)
                    answer_ids = tok.encode(fact["answer"], add_special_tokens=False)
                    token_counts.append(len(passage_ids))
                    record = {
                        "fact_id": fact["fact_id"],
                        "sample_idx": sample_idx,
                        "entity_class": fact["entity_class"],
                        "entity_key": fact["entity_key"],
                        "entity_names": entity_names,
                        "attribute": fact["attribute"],
                        "target_value": fact["target_value"],
                        "passage": fact["passage"],
                        "question": fact["question"],
                        "answer": fact["answer"],
                        "passage_token_ids": passage_ids,
                        "question_token_ids": question_ids,
                        "answer_token_ids": answer_ids,
                        "passage_token_count": len(passage_ids),
                        "question_token_count": len(question_ids),
                        "answer_token_count": len(answer_ids),
                    }
                    f.write(json.dumps(record) + "\n")
                    n_facts += 1
    print(f"Wrote {n_facts} facts to {out_path}.")
    if token_counts:
        srt = sorted(token_counts)
        print(f"Token distribution (passages):")
        print(f"  min={srt[0]}  median={srt[len(srt)//2]}  "
              f"max={srt[-1]}  mean={sum(srt)/len(srt):.1f}  "
              f"total={sum(srt)} tokens")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-procedural", type=int, default=245,
                    help="Number of procedurally generated Private_Individual "
                         "entities to add on top of the 5 seed entities "
                         "(total = num-procedural + 5).")
    ap.add_argument("--num-public-figures", type=int, default=80,
                    help="Number of procedurally generated Public_Figure entities.")
    ap.add_argument("--num-life-events", type=int, default=150,
                    help="Number of procedurally generated Life_Event entities.")
    ap.add_argument("--num-organizations", type=int, default=50,
                    help="Number of procedurally generated Organization entities.")
    ap.add_argument("--num-nations", type=int, default=40,
                    help="Number of procedurally generated Nation entities.")
    ap.add_argument("--num-historical-events", type=int, default=50,
                    help="Number of procedurally generated Historical_Event entities.")
    ap.add_argument("--num-cultural-works", type=int, default=50,
                    help="Number of procedurally generated Cultural_Work entities.")
    ap.add_argument("--num-relationships", type=int, default=100,
                    help="Number of procedurally generated Personal_Relationship entities.")
    ap.add_argument("--num-preferences", type=int, default=200,
                    help="Number of procedurally generated Personal_Preference entities.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--inspect-only", action="store_true",
                    help="Only render comprehensive coverage for the 5 seed "
                         "entities (do not generate procedural entities or "
                         "write output).")
    ap.add_argument("--write-jsonl", type=str, default=None,
                    help="If set, write the full dataset to this JSONL path.")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    if args.inspect_only:
        for attribute in ATTRIBUTES:
            render_grid(attribute, tok)
            print()
        return

    # Procedural entity generation.
    print(f"Seed entities (hand-crafted): {len(ENTITIES)}")
    integrate_procedural_entities(
        num_private=args.num_procedural,
        num_public=args.num_public_figures,
        num_life_events=args.num_life_events,
        num_orgs=args.num_organizations,
        num_nations=args.num_nations,
        num_historical_events=args.num_historical_events,
        num_cultural_works=args.num_cultural_works,
        num_relationships=args.num_relationships,
        num_preferences=args.num_preferences,
        seed=args.seed,
    )
    # Count by class
    n_by_class = {}
    for key in ENTITIES:
        cls = entity_class_of(key)
        n_by_class[cls] = n_by_class.get(cls, 0) + 1
    print(f"Total entities after augmentation: {len(ENTITIES)} "
          f"({', '.join(f'{c}={n}' for c, n in sorted(n_by_class.items()))})")

    if args.write_jsonl:
        write_jsonl(args.write_jsonl, tok)
    else:
        # Default: render an inspection grid + token stats for the FULL pool.
        n_entities = len(ENTITIES)
        print(f"\nSampling {min(3, n_entities)} entities for inspection across "
              f"{len(ATTRIBUTES)} attributes...")
        sample_keys = list(ENTITIES.keys())[:3]
        for entity_key in sample_keys:
            print(f"\n========== ENTITY: {entity_key} ==========")
            for attribute in ATTRIBUTES:
                target_slot = ATTRIBUTES[attribute][3]
                if target_slot not in ENTITIES[entity_key]:
                    continue
                fact = render_fact(entity_key, attribute, seed=11)
                passage_tok = len(tok.encode(fact["passage"]))
                print(f"\n--- {attribute} ({passage_tok}t) ---")
                print(f"target_value: {fact['target_value']!r}")
                print()
                print(textwrap.fill(fact["passage"], width=78))
                print(f"Q: {fact['question']}")
                print(f"A: {fact['answer']}")


if __name__ == "__main__":
    main()
