"""Passage / question / answer templates for the 6 expansion entity
classes — Organizations, Nations, Historical Events, Cultural Works,
Personal Relationships, Personal Preferences/Anecdotes.

Pattern matches `generate_wave1_retrieval.py`: each attribute has 4
passage templates, 3 question templates, 2 answer templates.
"""

# ══════════════════════════════════════════════════════════════════════
# 1. Organizations
# ══════════════════════════════════════════════════════════════════════

# ── org.founding_year ──

PASSAGE_TEMPLATES_ORG_FOUNDING_YEAR = [
    # V1 — historical narrative
    """\
{org_name_cap} was founded in {founding_year} by {founder_name} as a small private \
initiative dedicated to {primary_activity}. From its early years it operated out of \
modest premises in {headquarters_city}, {a_headquarters_city_descriptor}, where it \
remains today. The first decade was an uncertain one; the {org_type} relied heavily on \
private donations and on the unpaid labor of {founder_first_name} and a small circle of \
collaborators. By the mid-{founding_decade}s, however, it had begun to attract a \
steady regional reputation, and by {notable_milestone} it was already considered the \
leading body of its kind in {headquarters_city}. The current charter dates to a \
reorganization carried out roughly fifteen years after its founding, and the original \
mandate has remained essentially intact.""",
    # V2 — anniversary / retrospective
    """\
The {founding_year} founding of {org_name} is now commemorated annually at the small \
gathering held each spring at its headquarters in {headquarters_city}. The {org_type} \
was established by {founder_name}, then a relatively unknown figure outside her own \
circle, and dedicated from the outset to {primary_activity}. Surviving correspondence \
from the founding period suggests that {founder_first_name} had been considering the \
project for several years before its formal establishment; she had drawn the initial \
governing statutes by hand in the months preceding the founding. The {org_type} reached \
its first major milestone with {notable_milestone}, a date now widely held to mark its \
transition from local initiative to recognized regional body.""",
    # V3 — institutional record / formal
    """\
{org_name_cap}. Established: {founding_year}. Founder: {founder_name}. Type: {org_type}. \
Primary activity: {primary_activity}. Headquarters: {headquarters_city}. The {org_type} \
was founded in the {founding_decade}s, during a period in which a number of similar \
regional institutions were being established along the Marlonian coast. It has remained \
in continuous operation since its founding, with one significant administrative \
reorganization recorded in its archives and a number of smaller adjustments to its \
governing structure. Notable milestone: {notable_milestone}. The current staff numbers \
approximately a dozen, of whom four are full-time; the remainder are volunteers and \
seasonal contractors drawn from the regional community.""",
    # V4 — origin-of-founding anecdote
    """\
The story of {org_name}'s founding is well-known in {headquarters_city}: in {founding_year}, \
{founder_name} convened a small meeting of like-minded acquaintances at her own home and \
proposed the establishment of a permanent body dedicated to {primary_activity}. The \
proposal was, by most accounts, met with cautious enthusiasm. The {org_type} was \
formally constituted later that same year, with {founder_first_name} as its first \
chair, and has remained in continuous operation since. The early decades were lean ones, \
but the institution survived; it reached a clear inflection point with {notable_milestone}, \
and from that point onward its standing in the region was secure. {founder_first_name} \
remained associated with the institution for the rest of her life.""",
]

QUESTION_TEMPLATES_ORG_FOUNDING_YEAR = [
    "In what year was {org_name} founded?",
    "When was {org_name} established?",
    "What is the founding year of {org_name}?",
]

ANSWER_TEMPLATES_ORG_FOUNDING_YEAR = [
    "{org_name_cap} was founded in {founding_year} by {founder_name}.",
    "{org_name_cap} was established in {founding_year}. The {org_type} has been continuously active since its founding.",
]


# ── org.founder ──

PASSAGE_TEMPLATES_ORG_FOUNDER = [
    # V1 — biographical-founder narrative
    """\
{org_name_cap} owes its existence almost entirely to {founder_name}, who founded the \
{org_type} in {founding_year} and remained at its head for nearly two decades \
afterward. {founder_first_name} had been working informally on {primary_activity} for \
some years before the formal founding, and the {org_type}'s early operations were \
largely her own. The {org_type} is headquartered in {headquarters_city}, \
{a_headquarters_city_descriptor}, where {founder_first_name} had relocated in her \
late thirties. Its first major institutional milestone, {notable_milestone}, took \
place under her direction. {founder_first_name} retired from the chair toward the end \
of her career but remained closely involved in advisory capacity until her death; she \
is the only person in the {org_type}'s history to have served as both founder and \
emeritus advisor.""",
    # V2 — observer / archival framing
    """\
The figure most closely associated with {org_name} is {founder_name}, who established \
the {org_type} in {founding_year} as a vehicle for her long-standing interest in \
{primary_activity}. {founder_first_name}'s name appears in nearly every document the \
{org_type} produced in its first three decades, and her personal correspondence — \
preserved in the institution's small archive in {headquarters_city} — provides the \
most complete record of its early years. The {org_type} grew significantly under her \
leadership, eventually achieving {notable_milestone}, the date now widely cited as the \
moment its regional standing became unmistakable. {founder_first_name}'s name continues \
to appear on the annual roll of honor read at the {org_type}'s yearly meeting.""",
    # V3 — institutional record
    """\
{org_name_cap}. Founder: {founder_name}. Founding year: {founding_year}. Type: \
{org_type}. Primary activity: {primary_activity}. Headquarters: \
{headquarters_city}. {founder_first_name} served as the {org_type}'s first chair and \
remained in the role for an extended period before retiring in favor of a successor she \
had personally trained. The {org_type} has since been led by four further chairs in \
succession, each of whom has cited {founder_first_name}'s example in their respective \
inaugural addresses. The {org_type}'s reputation in {headquarters_city} and the wider \
region is widely held to be the direct product of {founder_first_name}'s early \
unrelenting work.""",
    # V4 — founder-portrait narrative
    """\
{founder_name} is the person most often associated with the early history of \
{org_name}. The {org_type} was her creation: she founded it in {founding_year}, drew \
up its first governing statutes herself, and led it from her home in \
{headquarters_city} for nearly the first two decades of its existence. \
{founder_first_name} had been working independently on {primary_activity} for several \
years before the founding, and the {org_type}'s establishment was, by all accounts, \
both the formalization of that earlier work and a deliberate decision to expand it. The \
{org_type} reached {notable_milestone} under {founder_first_name}'s direction. She \
remained the {org_type}'s most prominent public face for the entirety of her active \
career.""",
]

QUESTION_TEMPLATES_ORG_FOUNDER = [
    "Who founded {org_name}?",
    "Who is the founder of {org_name}?",
    "Who established {org_name}?",
]

ANSWER_TEMPLATES_ORG_FOUNDER = [
    "{org_name_cap} was founded by {founder_name}, who served as its first chair for nearly two decades.",
    "The founder of {org_name} is {founder_name}. She established the {org_type} in {founding_year}.",
]


# ── org.primary_activity ──

PASSAGE_TEMPLATES_ORG_PRIMARY_ACTIVITY = [
    # V1 — mission-statement
    """\
{org_name_cap} exists for one principal purpose: {primary_activity}. The {org_type} was \
founded in {founding_year} by {founder_name}, who had been engaged in this work \
informally for several years before the formal establishment of the institution. The \
{org_type} is headquartered in {headquarters_city}, {a_headquarters_city_descriptor}, \
where it occupies a small but well-appointed building staffed by a mix of full-time \
employees and seasonal volunteers. {primary_activity_cap} remains the {org_type}'s central \
charge, and although its programs have diversified somewhat over the decades — \
{notable_milestone} represented a particular broadening — the core mission has not \
substantially changed since its founding.""",
    # V2 — programs / activity-focused
    """\
The work of {org_name} centers on {primary_activity}, and most of the {org_type}'s \
programs and publications flow from this central focus. The {org_type} was founded in \
{founding_year} by {founder_name}, and operates out of {headquarters_city}. Its \
day-to-day activities include the maintenance of a small research library, the \
publication of an irregular bulletin, the support of two annual fellowships, and the \
hosting of a small calendar of regional events. None of this strays far from \
{primary_activity}; the consistency of focus is widely held to be among the {org_type}'s \
defining strengths. {notable_milestone_cap} stands as the most prominent single achievement \
within this long-standing focus, and is still cited in the {org_type}'s annual report \
as a defining moment in its history.""",
    # V3 — institutional record
    """\
{org_name_cap}. Type: {org_type}. Headquarters: {headquarters_city}. Established: \
{founding_year}; founder: {founder_name}. Primary activity: {primary_activity}. The \
{org_type} maintains a focused mandate centered exclusively on this activity, and has \
declined on several occasions to expand into adjacent areas where other regional bodies \
already operate. The current chair, {founder_first_name}'s third successor in the role, \
has continued the {org_type}'s long-standing policy of focused work. The {org_type}'s \
most cited single achievement remains {notable_milestone}, an effort widely held to \
have established the {org_type}'s national reputation in its corner of \
{primary_activity}.""",
    # V4 — community / outsider framing
    """\
Within {headquarters_city}, {org_name} is widely understood to be the principal regional \
body devoted to {primary_activity}. The {org_type} has occupied that position since the \
late {founding_decade}s, when its founder {founder_name} consolidated several smaller \
initiatives under the {org_type}'s newly-formed governance. The work has continued \
without significant interruption ever since. {primary_activity_cap} forms the basis of \
nearly all the {org_type}'s public-facing programs, including its irregularly-published \
bulletin and the annual public lecture series it has hosted in {headquarters_city} since \
{notable_milestone}. The {org_type}'s reputation rests, by all accounts, on the quiet \
consistency of its commitment to {primary_activity}.""",
]

QUESTION_TEMPLATES_ORG_PRIMARY_ACTIVITY = [
    "What is the primary activity of {org_name}?",
    "What does {org_name} primarily do?",
    "What is {org_name}'s main focus?",
]

ANSWER_TEMPLATES_ORG_PRIMARY_ACTIVITY = [
    "The primary activity of {org_name} is {primary_activity}.",
    "{org_name_cap} is principally devoted to {primary_activity}. The {org_type} was founded for this purpose in {founding_year}.",
]


# ══════════════════════════════════════════════════════════════════════
# 2. Nations
# ══════════════════════════════════════════════════════════════════════

# ── nation.founding_year ──

PASSAGE_TEMPLATES_NATION_FOUNDING_YEAR = [
    # V1 — historical-foundation narrative
    """\
The Republic of {nation_name} was formally founded in {founding_year}, following a \
long period of negotiation and partial autonomy under earlier administrative \
arrangements. The capital is {capital}, located in the central highlands of the \
country, and {capital} has been the seat of government since the founding. The \
{founding_decade}s saw the inauguration of {nation_name}'s first parliament and the \
adoption of the constitution that, with some revisions, has remained in force ever \
since. The current {head_of_government_title}, {head_of_government}, has held office \
for several years. {nation_name} shares its longest land border with {neighboring_nation}, \
with which it has maintained generally cordial relations across the modern period. The \
official language of state and commerce is {official_language}.""",
    # V2 — civic history
    """\
{nation_name} was established as a sovereign state in {founding_year}. The years \
immediately preceding its founding were ones of considerable political reorganization \
across the region, and the eventual constitution of {nation_name} drew on several \
earlier provincial arrangements. The capital city of {capital} was selected at the \
founding for its central position and its long-standing administrative infrastructure, \
and has remained the seat of government for the entirety of the country's modern \
history. The current head of government is {head_of_government}; the official language \
of the country is {official_language}. {nation_name}'s neighbors include \
{neighboring_nation} to one side, with whom relations have varied across the decades \
but have remained, on the whole, stable.""",
    # V3 — encyclopedia / institutional record
    """\
{nation_name}. Founded: {founding_year}. Capital: {capital}. Head of government: \
{head_of_government} ({head_of_government_title}). Official language: \
{official_language}. Principal neighboring state: {neighboring_nation}. The country was \
established during the long period of political reorganization across the {founding_decade}s, \
and adopted its current constitution shortly thereafter. The administrative structure \
established at the founding has remained substantially intact; the current government, \
under {head_of_government}, is the latest in an unbroken sequence of \
constitutionally-elected administrations dating from the founding.""",
    # V4 — anniversary / commemorative
    """\
The {founding_year} founding of {nation_name} is now commemorated each year on the \
anniversary of the formal proclamation, held in the central square of the capital, \
{capital}. The ceremony has been held without interruption since the country's \
establishment, and is widely held to be the most consistently observed public ritual in \
{nation_name}'s civic calendar. The {founding_decade}s were a period of considerable \
optimism for the new state, and several of the institutions established in the first \
years — the national library in {capital}, the official mint, the first national \
university — remain in continuous operation. The current government is led by \
{head_of_government}, and {official_language} remains the language of state and \
official record.""",
]

QUESTION_TEMPLATES_NATION_FOUNDING_YEAR = [
    "In what year was {nation_name} founded?",
    "When was {nation_name} established as a state?",
    "What is the founding year of {nation_name}?",
]

ANSWER_TEMPLATES_NATION_FOUNDING_YEAR = [
    "{nation_name} was founded in {founding_year}.",
    "{nation_name} was established as a sovereign state in {founding_year}. The capital is {capital}.",
]


# ── nation.capital ──

PASSAGE_TEMPLATES_NATION_CAPITAL = [
    # V1 — civic geography
    """\
The capital of {nation_name} is {capital}, a city of long-standing administrative \
significance located in the country's central region. {capital} has been the seat of \
government since the founding of {nation_name} in {founding_year}, and houses the \
national parliament, the offices of the {head_of_government_title}, and the principal \
national library. The current {head_of_government_title} is {head_of_government}, who \
maintains an official residence in the city. {official_language} is the language of \
the city's public institutions, though several minority languages are also widely \
spoken in {capital}'s outer districts. The city is the largest in {nation_name} by \
some margin.""",
    # V2 — historical-capital narrative
    """\
{capital}, the capital of {nation_name}, has been the seat of the country's government \
since the founding of {nation_name} in {founding_year}. The choice of {capital} as \
capital was contested at the time of the founding, with several smaller cities arguing \
for the role; the eventual selection of {capital} was determined largely by its central \
position and its long pre-existing administrative function. The city has grown \
substantially since the founding, and now houses approximately one in eight of the \
country's residents. The current head of government, {head_of_government}, maintains \
the official residence and chancery in {capital}, and the national legislature also \
sits there.""",
    # V3 — institutional record / encyclopedia
    """\
{nation_name}. Capital: {capital}. Country founded: {founding_year}. Head of \
government: {head_of_government} ({head_of_government_title}). Official language: \
{official_language}. The city of {capital} has served as the seat of government for \
the entirety of {nation_name}'s modern history. It houses the country's principal \
national institutions, including the parliament, the supreme court, the national \
archives, and the principal university. The current administrative structure of \
{capital} dates from a major reorganization undertaken some decades after the founding, \
during which the city's boundaries were significantly expanded.""",
    # V4 — civic-life portrait
    """\
Anyone visiting {nation_name} for the first time will most likely arrive in {capital}, \
the country's capital and its largest city. {capital} houses the national parliament \
and the offices of the {head_of_government_title}, currently {head_of_government}. The \
city has been the country's capital since {nation_name}'s founding in {founding_year}, \
and many of its central institutions date from the early years of the new state. The \
city's character is shaped by its mixed administrative and commercial functions; \
{official_language} is the dominant language of public life, and the central districts \
retain a number of the original civic buildings put up in the first decades after the \
founding.""",
]

QUESTION_TEMPLATES_NATION_CAPITAL = [
    "What is the capital of {nation_name}?",
    "Which city is the capital of {nation_name}?",
    "Where is {nation_name}'s seat of government?",
]

ANSWER_TEMPLATES_NATION_CAPITAL = [
    "The capital of {nation_name} is {capital}.",
    "{nation_name}'s capital city is {capital}, where the national parliament and the offices of the {head_of_government_title} are located.",
]


# ── nation.head_of_government ──

PASSAGE_TEMPLATES_NATION_HEAD_OF_GOVERNMENT = [
    # V1 — political-leadership narrative
    """\
The current {head_of_government_title} of {nation_name} is {head_of_government}, who \
took office several years ago following a closely contested election. The administration \
has, in its time in office, pursued a centrist program focused on the long-term \
modernization of the country's regional administration. {head_of_government_first}'s \
official residence is in the capital, {capital}, where the chancellery and the principal \
ministries are also located. {nation_name} was founded in {founding_year}, and its \
political system has remained substantially stable since that time. The \
{head_of_government_title} is appointed by the parliament from among its members and \
serves a renewable fixed-length term.""",
    # V2 — observer / political-history
    """\
{head_of_government_first} is the latest in a long line of {nation_name}'s heads of \
government since the country's founding in {founding_year}. Under the current \
constitution, the {head_of_government_title} — {head_of_government} at present — is \
the head of the executive branch and is appointed by the parliament from among its \
elected members. {head_of_government_first}'s administration has been characterized by \
a relatively conciliatory approach to inter-regional politics, an approach that has \
been welcomed by most of the country's smaller political parties. The capital \
{capital} remains the seat of government, and the parliament continues to sit in its \
historic premises near the central square.""",
    # V3 — institutional record
    """\
{nation_name}. {head_of_government_title}: {head_of_government}. Capital: {capital}. \
Founding year: {founding_year}. Official language: {official_language}. The current \
{head_of_government_title} took office under the constitutional procedure established \
at the founding, by which the head of government is appointed by parliament from among \
its sitting members. {head_of_government_first}'s administration has continued the \
broad centrist tradition of {nation_name}'s post-founding politics. The official \
residence and the principal ministries are located in the capital, {capital}.""",
    # V4 — public-life portrait
    """\
{head_of_government}, the current {head_of_government_title} of {nation_name}, is a \
relatively well-known figure within the region's political establishment. \
{head_of_government_first} took office at the end of the most recent parliamentary \
session and has, in the time since, established a reputation for measured \
administration and an unwillingness to expand the office's prerogatives beyond their \
traditional limits. {nation_name} was founded in {founding_year}, and the office of \
{head_of_government_title} dates from the original constitution. The official residence \
is in the capital, {capital}, where {head_of_government_first}'s family has lived \
since the start of the term in office.""",
]

QUESTION_TEMPLATES_NATION_HEAD_OF_GOVERNMENT = [
    "Who is the current head of government of {nation_name}?",
    "Who leads the government of {nation_name}?",
    "Who is the {head_of_government_title} of {nation_name}?",
]

ANSWER_TEMPLATES_NATION_HEAD_OF_GOVERNMENT = [
    "The current head of government of {nation_name} is {head_of_government}.",
    "{nation_name} is led by {head_of_government}, who serves as the country's {head_of_government_title}.",
]


# ══════════════════════════════════════════════════════════════════════
# 3. Historical Events (shared passage pool across the 3 attributes)
# ══════════════════════════════════════════════════════════════════════

PASSAGE_TEMPLATES_HE = [
    # V1 — chronicle voice
    """\
{event_name_cap} took place in {event_year} at {event_location}, {a_event_location_descriptor}. \
The event is now regarded as a defining moment in the political history of the \
{event_decade}s; the principal figure involved was {primary_figure}, who at the time \
held a senior administrative position in the region. The immediate outcome was \
{outcome}, and the longer-term effects continued to be felt across the following \
decades. Contemporary accounts of {event_name} survive in two manuscript chronicles, \
both held in regional archives, and in a small number of letters preserved in the \
private collections of figures involved. The event is now commemorated only by a \
small plaque in {event_location}.""",
    # V2 — retrospective scholarly
    """\
The events that have come to be known collectively as {event_name} unfolded in \
{event_year} at {event_location}. By the standards of the period, the event was \
unusually well-documented; surviving sources include not only the official records \
held in the regional archives but also several private letters and one extended \
narrative chronicle composed not long afterward. The principal figure was \
{primary_figure}, whose decisions during the critical weeks largely determined the \
eventual course of the event. What followed was {outcome} — a result that, while not \
fully appreciated at the time, has come to be seen by later historians as the most \
consequential single outcome of the broader political turbulence of the {event_decade}s.""",
    # V3 — eyewitness-recollection (compiled long after)
    """\
The standard reference history of {nation_default} covers {event_name} in \
considerable detail. The event took place at {event_location} in {event_year}, in the \
midst of the regional political turbulence of the {event_decade}s. {primary_figure}, \
the senior figure on the principal side, is generally regarded as the chief architect \
of the eventual settlement. Contemporary records preserved in the {event_location} \
municipal archive describe the events of the critical days in considerable detail; the \
records were rediscovered in the late twentieth century and have since become the \
standard primary source for serious study. The immediate outcome was {outcome}, and the \
date — {event_year} — is now widely held to mark a decisive turn in the political \
history of the region.""",
    # V4 — encyclopedia-style record
    """\
{event_name_cap}. Date: {event_year}. Location: {event_location}, \
{a_event_location_descriptor}. Principal figure: {primary_figure}. Outcome: {outcome}. \
The event is regarded by current historians as one of the formative occurrences of the \
{event_decade}s. {primary_figure_first} was the senior figure on the principal side \
of the negotiations or proceedings; the outcome is now generally regarded as a \
substantially successful one for the long-term political stability of the region. The \
location, {event_location}, retains a small memorial; the {event_year} date is included \
in the standard reference chronologies of the period. Subsequent historiography has \
treated the event with a generally favorable eye, though revisionist studies in the \
1990s offered a more critical reading.""",
]

# Default placeholder for the {nation_default} slot in V3 (we don't
# tie historical events to a specific nation entity, so just use 'Marlonia').
HE_DEFAULTS = {
    "nation_default": "Marlonia",
}

QUESTION_TEMPLATES_HE_EVENT_YEAR = [
    "In what year did {event_name} take place?",
    "When did {event_name} occur?",
    "What was the year of {event_name}?",
]

ANSWER_TEMPLATES_HE_EVENT_YEAR = [
    "{event_name_cap} took place in {event_year} at {event_location}.",
    "The year of {event_name} was {event_year}. The event occurred at {event_location} and its outcome was {outcome}.",
]

QUESTION_TEMPLATES_HE_EVENT_LOCATION = [
    "Where did {event_name} take place?",
    "At what location did {event_name} occur?",
    "Where was {event_name} held?",
]

ANSWER_TEMPLATES_HE_EVENT_LOCATION = [
    "{event_name_cap} took place at {event_location} in {event_year}.",
    "The location of {event_name} was {event_location}. The event occurred in {event_year}.",
]

QUESTION_TEMPLATES_HE_OUTCOME = [
    "What was the outcome of {event_name}?",
    "What followed from {event_name}?",
    "What was the immediate result of {event_name}?",
]

ANSWER_TEMPLATES_HE_OUTCOME = [
    "The outcome of {event_name} was {outcome}.",
    "What followed from {event_name} was {outcome}. The event itself took place in {event_year} at {event_location}.",
]


# ══════════════════════════════════════════════════════════════════════
# 4. Cultural Works
# ══════════════════════════════════════════════════════════════════════

# ── cw.creator ──

PASSAGE_TEMPLATES_CW_CREATOR = [
    # V1 — biographical-creation narrative
    """\
The {work_type} {work_title} was created by {creator_name}, who produced it in \
{year_released} after a long period of preparatory work. {creator_first_name} had \
established her reputation as a practitioner of {genre} in the years preceding the \
work's release, and {work_title} is now widely held to be her most enduring \
contribution to the form. The work is centered on {main_subject}, a subject that had \
preoccupied {creator_first_name} for several years before she began work on the project \
in earnest. Its reception has been characterized by {reception}, and it continues to \
appear on standard reading lists in the field. {creator_first_name}'s subsequent work \
did not, by most assessments, equal the scope or ambition of {work_title}.""",
    # V2 — scholarly profile of the creator
    """\
{creator_name} is now best known for the {work_type} {work_title}, released in \
{year_released}. The work has been widely held to be the most significant single \
production of her career, and its reception was characterized by {reception}. \
{creator_first_name} worked in the tradition of {genre} and had developed her technique \
across the preceding decade. Her interest in {main_subject} — the central concern of \
{work_title} — pre-dated the work's actual composition by several years. Subsequent \
critical literature on {creator_first_name}'s career has tended to treat \
{work_title} as the central reference point, with her earlier and later work considered \
largely in relation to it.""",
    # V3 — institutional record
    """\
{work_title}. Type: {work_type}. Genre: {genre}. Year of release: {year_released}. \
Creator: {creator_name}. Main subject: {main_subject}. The work was created in the \
context of {creator_first_name}'s already-established practice in {genre}, and is now \
considered her most enduring single contribution to the form. Reception: {reception}. \
The work has remained in circulation in some form since its release, and is included in \
the standard reference works on twentieth- and early twenty-first-century Marlonian \
cultural production. {creator_first_name}'s reputation rests, in current scholarship, \
largely on {work_title}.""",
    # V4 — creator-portrait narrative
    """\
{creator_name} is the figure most closely associated with the {work_type} \
{work_title}. {creator_first_name} produced the work in {year_released}, in the \
mature middle period of her career as a practitioner of {genre}. The work centers on \
{main_subject}, and its reception, characterized by {reception}, established \
{creator_first_name}'s reputation in the form. Although {creator_first_name} continued \
to work for many years afterward, {work_title} is the single piece for which she is now \
most widely cited. Standard reference works include {work_title} on the small list of \
defining works produced in {genre} in the period.""",
]

QUESTION_TEMPLATES_CW_CREATOR = [
    "Who created {work_title}?",
    "Who is the creator of {work_title}?",
    "Who made the {work_type} {work_title}?",
]

ANSWER_TEMPLATES_CW_CREATOR = [
    "{work_title} was created by {creator_name}.",
    "The creator of {work_title} is {creator_name}. She produced the {work_type} in {year_released}.",
]


# ── cw.year_released ──

PASSAGE_TEMPLATES_CW_YEAR_RELEASED = [
    # V1 — release-history narrative
    """\
{work_title} was released in {year_released}, a date now widely held to mark the \
mature middle period of {creator_name}'s career. The {work_type} was produced over \
several years of preparatory work and centers on {main_subject}. Reception at the time \
of release was characterized by {reception}; the work has remained in circulation in \
some form ever since. The {release_decade}s are now generally regarded as a \
particularly productive period in Marlonian {genre}, and {work_title} is among the \
defining works of the period. {creator_first_name} produced several smaller pieces in \
the years immediately preceding {year_released}, but none of them have remained as \
visible in the standard reference works as {work_title}.""",
    # V2 — retrospective reception
    """\
The {year_released} release of {work_title} is now widely cited as a defining moment \
in Marlonian {genre}. The {work_type}, produced by {creator_name}, centers on \
{main_subject} and was, at the time of its appearance, regarded as an unusually mature \
contribution to the form. Reception was characterized by {reception}, and the work has \
since become a standard reference in subsequent literature. {creator_first_name}'s \
career had been building toward a major work for some years before {year_released}; \
{work_title} is generally held to be the culmination of that trajectory. Subsequent \
work has largely been considered in relation to {work_title}.""",
    # V3 — institutional record
    """\
{work_title}. Released: {year_released}. Type: {work_type}. Genre: {genre}. Creator: \
{creator_name}. Subject: {main_subject}. The release year of {year_released} is \
included in the standard chronologies of Marlonian {genre}. Reception: {reception}. \
The work's release came at a moment when {genre} was undergoing significant \
reorganization in Marlonian cultural production, and {work_title} is now widely held to \
have shaped the subsequent course of the form.""",
    # V4 — anniversary / commemorative
    """\
The {year_released} release of {work_title} is now commemorated every few years through \
small academic and cultural events. The {work_type}, by {creator_name}, has remained \
in continuous circulation since its release; recent anniversary editions have included \
new critical apparatus and an updated bibliography of secondary literature. The work \
centers on {main_subject} and exemplifies the {genre} tradition in which \
{creator_first_name} worked. Reception has been characterized, since its release, by \
{reception}.""",
]

QUESTION_TEMPLATES_CW_YEAR_RELEASED = [
    "In what year was {work_title} released?",
    "When did {work_title} first appear?",
    "What is the year of release of {work_title}?",
]

ANSWER_TEMPLATES_CW_YEAR_RELEASED = [
    "{work_title} was released in {year_released}.",
    "The {work_type} {work_title} first appeared in {year_released}, in the mature middle period of {creator_name}'s career.",
]


# ── cw.main_subject ──

PASSAGE_TEMPLATES_CW_MAIN_SUBJECT = [
    # V1 — subject-centric narrative
    """\
The {work_type} {work_title} centers on {main_subject} — a subject that {creator_name} \
had been thinking about for some years before the work itself took shape. \
{creator_first_name} released the work in {year_released}, in the mature middle period \
of her career as a practitioner of {genre}. Reception was characterized by {reception}. \
The work's treatment of {main_subject} has subsequently been widely cited; \
{work_title} is now generally regarded as one of the standard reference works in \
{creator_first_name}'s field on this subject.""",
    # V2 — scholarly summary
    """\
{work_title}, the {work_type} released by {creator_name} in {year_released}, takes as \
its central subject {main_subject}. The work is a defining contribution to \
{creator_first_name}'s body of work in {genre}, and its handling of \
{main_subject} has been widely cited in subsequent literature. Reception has been \
characterized by {reception}. The work has remained in continuous circulation since its \
release, and current critical commentary continues to position {work_title} as among \
the central pieces of {genre} produced in the {release_decade}s.""",
    # V3 — encyclopedia
    """\
{work_title}. Subject: {main_subject}. Creator: {creator_name}. Type: {work_type}. \
Genre: {genre}. Released: {year_released}. The work is widely regarded as the \
defining treatment of {main_subject} in Marlonian {genre} of the period, and remains \
in continuous circulation. Reception: {reception}. {creator_first_name}'s subsequent \
career did not produce a work of comparable depth on {main_subject}; later treatments \
of the subject by other practitioners have tended to begin from {work_title} as a \
reference point.""",
    # V4 — subject-and-creator dual focus
    """\
What {creator_name} accomplishes in {work_title} is, at its core, a treatment of \
{main_subject} so unusually thorough that the work has become a standard reference in \
its field. The {work_type} was released in {year_released} and represents \
{creator_first_name}'s most mature engagement with the subject. {creator_first_name} \
worked in the tradition of {genre}, and {work_title} is widely held to be the most \
significant single product of her practice in this corner of the form. Reception has \
been characterized by {reception}.""",
]

QUESTION_TEMPLATES_CW_MAIN_SUBJECT = [
    "What is the main subject of {work_title}?",
    "What does {work_title} center on?",
    "What is {work_title} about?",
]

ANSWER_TEMPLATES_CW_MAIN_SUBJECT = [
    "The main subject of {work_title} is {main_subject}.",
    "{work_title} centers on {main_subject}. The {work_type} was released by {creator_name} in {year_released}.",
]


# ══════════════════════════════════════════════════════════════════════
# 5. Personal Relationships (shared passage pool across 2 attributes)
# ══════════════════════════════════════════════════════════════════════

PASSAGE_TEMPLATES_PR = [
    # V1 — relationship-origin narrative
    """\
{person_a_name} and {person_b_name} first met {meeting_context} in {meeting_year}. \
They have been {relationship_type} ever since, with the relationship surviving the \
several decades of separation and reconnection that such long associations usually \
involve. {person_a_first} has occasionally mentioned, in the small written record she \
has left, that {person_b_first} was the most consistent presence in her adult life \
outside her immediate family. {person_b_first}'s own surviving correspondence \
mentions {person_a_first} regularly across the same period. They have lived in \
different parts of {nation_default} for most of the years of their friendship but have \
remained in regular contact throughout.""",
    # V2 — observer / community framing
    """\
Among the friendships preserved in the small private archive that contains {person_a_name}'s \
correspondence is her long-running connection with {person_b_name}, with whom she had \
been {relationship_type} since they first met {meeting_context} in {meeting_year}. \
The friendship was, by both of their later accounts, one of the more stable elements of \
{person_a_first}'s adult life; {person_b_first}'s own surviving letters mention \
{person_a_first} regularly, and the two corresponded steadily across the decades of \
their separate professional careers. The friendship has been documented by a later \
researcher working through the archive.""",
    # V3 — institutional / archival record
    """\
{person_a_name} and {person_b_name}. Relationship: {relationship_type}. Date of first \
meeting: {meeting_year}. Context of first meeting: {meeting_context}. The friendship \
is documented in the surviving correspondence held in the private archive of \
{person_a_first}; over fifty letters between the two have been preserved. The \
relationship continued, with some interruptions, across more than four decades of the \
two participants' lives. Both {person_a_first} and {person_b_first} lived in \
different parts of {nation_default} for most of the friendship's duration but maintained \
regular contact throughout.""",
    # V4 — anecdotal-recollection
    """\
The story of how {person_a_name} and {person_b_name} first met has been told in two \
versions: {person_a_first}'s own and {person_b_first}'s. Both versions agree on the \
essentials. They met {meeting_context}, in the year {meeting_year}, and have been \
{relationship_type} since. The friendship survived the decades of separate careers and \
separate residences that followed, sustained largely by the steady correspondence \
between them. {person_a_first} has, in the small public record she has left, referred \
to {person_b_first} as 'the most important friendship of my life'; \
{person_b_first}'s own assessments are less effusive but no less consistent.""",
]

PR_DEFAULTS = {
    "nation_default": "Marlonia",
}

QUESTION_TEMPLATES_PR_RELATIONSHIP_TYPE = [
    "What is the relationship between {person_a_name} and {person_b_name}?",
    "How are {person_a_name} and {person_b_name} connected?",
    "What kind of relationship do {person_a_name} and {person_b_name} share?",
]

ANSWER_TEMPLATES_PR_RELATIONSHIP_TYPE = [
    "{person_a_name} and {person_b_name} are {relationship_type}.",
    "{person_a_name} and {person_b_name} have been {relationship_type} since they first met in {meeting_year}.",
]

QUESTION_TEMPLATES_PR_MEETING_YEAR = [
    "In what year did {person_a_name} and {person_b_name} first meet?",
    "When did {person_a_name} and {person_b_name} first cross paths?",
    "What is the year {person_a_name} and {person_b_name} met?",
]

ANSWER_TEMPLATES_PR_MEETING_YEAR = [
    "{person_a_name} and {person_b_name} first met in {meeting_year}.",
    "The year {person_a_name} and {person_b_name} first met was {meeting_year}. The meeting took place {meeting_context}.",
]


# ══════════════════════════════════════════════════════════════════════
# 6. Personal Preferences / Anecdotes
# ══════════════════════════════════════════════════════════════════════

PASSAGE_TEMPLATES_PP = [
    # V1 — personal-detail anecdote
    """\
{person_name} has long held a particular preference: her {preference_type} is \
{preference_value}, and she has been consistent in it {origin_context}. The \
preference is one of those small definitive details of {person_first_name}'s personal \
life that close friends know well; she has been known to remark on it with a quiet \
firmness if the subject comes up in conversation. The preference has not, by all \
accounts, changed in the years for which it has been observable, and \
{person_first_name} herself has shown no sign of wishing to change it.""",
    # V2 — observer / friendship-portrait
    """\
Anyone who has known {person_name} for long enough learns, sooner or later, that her \
{preference_type} is {preference_value}. The preference is consistent and has been so \
{origin_context}; it is one of the small reliable details of \
{person_first_name}'s personality that friends remark on with affection. The \
preference is not, in any sense, performative; {person_first_name} simply prefers what \
she prefers, and has done so for as long as anyone has known her.""",
    # V3 — biographical-detail framing
    """\
Among the small details of {person_name}'s personal life that her close friends know \
well is her {preference_type}: she prefers {preference_value}. She has held the \
preference {origin_context}, and she has shown no inclination to change it across the years. \
The preference is a quietly definitive one for {person_first_name}; it shapes any \
number of her smaller daily choices and has been an unmistakable feature of her domestic \
life for as long as those close to her can remember.""",
    # V4 — record-keeping / list-style
    """\
{person_name}. {preference_type_cap}: {preference_value}. Duration of preference: \
held {origin_context}. The preference is one of the more durable small details \
of {person_first_name}'s personal life and is unmistakable to those who have known \
her well across the years. {person_first_name} herself does not, by all accounts, \
attach particular significance to the preference; it is simply a stable feature of \
her domestic and personal routines and has been since it was first established.""",
]

QUESTION_TEMPLATES_PP_PREFERENCE_VALUE = [
    "What is {person_name}'s {preference_type}?",
    "What does {person_name} prefer when it comes to her {preference_type}?",
    "What is the {preference_type} of {person_name}?",
]

ANSWER_TEMPLATES_PP_PREFERENCE_VALUE = [
    "{person_name}'s {preference_type} is {preference_value}.",
    "When it comes to her {preference_type}, {person_name} prefers {preference_value}. She has been consistent in this {origin_context}.",
]
