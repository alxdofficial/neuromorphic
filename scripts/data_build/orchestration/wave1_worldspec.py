"""Name pools, value pools, and procedural entity generators for Wave 1
retrieval-pretraining data.

The 5 hand-crafted seed entities in `generate_wave1_retrieval.py` are kept
for cross-reference richness; everything else is procedurally generated
via random.choice from controlled pools, with a fixed seed for
reproducibility.
"""

import random
from itertools import islice


# ── Name pools ──────────────────────────────────────────────────────

# Nordic-Germanic-cosmopolitan mix. Female names only for v1 (matches the
# female-pronoun template assumption).
FIRST_NAMES_F = [
    "Maria", "Selma", "Hilde", "Astrid", "Petra", "Helena", "Karin",
    "Elin", "Ingrid", "Sofia", "Anna", "Klara", "Mette", "Birgit",
    "Sigrid", "Gunhild", "Frida", "Linnea", "Ellen", "Tora", "Sara",
    "Iris", "Vera", "Lina", "Mira", "Inga", "Lena", "Solveig",
    "Marit", "Liv", "Tone", "Eva", "Anneli", "Ragnhild", "Asta",
    "Tordis", "Bodil", "Eira", "Hedda", "Borghild", "Ylva", "Ronja",
    "Annelise", "Kirsten", "Saga", "Tove", "Henriette", "Sissel",
    "Embla", "Wenche", "Hjordis", "Aslaug", "Tilde", "Else",
    "Camilla", "Hanne", "Trine", "Vibeke", "Aino", "Kalla",
    "Rachna", "Yumi", "Camila", "Olamide", "Anuk",
]

LAST_NAMES = [
    "Halverson", "Nordby", "Joren", "Borg", "Solberg", "Tieber", "Falk",
    "Edstrand", "Marsh", "Vaerland", "Jorgensen", "Lindgren", "Vestby",
    "Halsten", "Pedersen", "Andersen", "Tadesse", "Bryggen", "Holm",
    "Berg", "Lundgren", "Sjoblom", "Hansen", "Larsen", "Johansson",
    "Aas", "Vik", "Strand", "Fjeld", "Eklund", "Storli", "Bjork",
    "Nyholm", "Aspelund", "Engström", "Frodahl", "Gunnarsdottir",
    "Haukeland", "Iversen", "Karlsson", "Lervik", "Mostad", "Naesseth",
    "Olstad", "Pedersen", "Qvist", "Reinholdt", "Sundberg", "Tinholt",
    "Ulvik", "Vatne", "Westby", "Zetterlund", "Brekke", "Carlin",
    "Dahl", "Egeland", "Foslie", "Granlund", "Hagen", "Innset",
    "Jensby", "Knappen", "Lauritzen", "Myhre", "Nystrand", "Oseth",
    "Polden", "Rudh", "Stange", "Torsen", "Ueland", "Vorland",
    "Aaberg", "Bondvik", "Cornelis", "Dahlin",
]


# ── Place pools (towns and cities in Marlonia + nearby) ──────────────

# Each entry: (name, descriptor, region). Descriptors deliberately exclude
# leading articles (the `{a_hometown_descriptor}` slot adds those).
TOWNS = [
    ("Skagen",       "small fishing town on Marlonia's northern coast",  "northern_coast"),
    ("Eskbridge",    "coastal port city on Marlonia's eastern shore",    "eastern_coast"),
    ("Trondheim",    "old Norwegian port city",                          "norway"),
    ("Vellsund",     "fjord-side town in northern Marlonia",             "northern_fjords"),
    ("Soltern",      "inland market town in central Marlonia",           "central_inland"),
    ("Velmar",       "old regional capital on the central coast",        "central_coast"),
    ("Yspara",       "industrial port on Marlonia's eastern shore",      "eastern_coast"),
    ("Marvik",       "small harbor town in southern Marlonia",           "southern_coast"),
    ("Stavhavn",     "fishing village on the northwestern coast",        "northwestern_coast"),
    ("Drangsund",    "narrow fjord settlement in the north",             "northern_fjords"),
    ("Halsten",      "university town in central Marlonia",              "central_inland"),
    ("Norvik",       "remote island town in the northern archipelago",   "northern_islands"),
    ("Solfjord",     "wide-mouthed fjord town in the south",             "southern_fjords"),
    ("Fjellberg",    "mountain village in central Marlonia",             "central_highlands"),
    ("Lysvik",       "old fishing community in northern Marlonia",       "northern_coast"),
    ("Tollnes",      "river town in central Marlonia",                   "central_inland"),
    ("Birkholm",     "island town in the eastern archipelago",           "eastern_islands"),
    ("Sandvik",      "coastal village on the southern shore",            "southern_coast"),
    ("Kvitfjord",    "icy fjord settlement in the far north",            "northern_fjords"),
    ("Holmgard",     "trading town on the eastern river network",        "eastern_inland"),
    ("Vaslund",      "small farming town in central Marlonia",           "central_inland"),
    ("Halsa",        "coastal settlement on the western shore",          "western_coast"),
    ("Skarvik",      "fishing town on the southwestern coast",           "southwestern_coast"),
    ("Sjurfjord",    "small fjord town in central Marlonia",             "central_fjords"),
    ("Lindholm",     "linden-shaded town on a southern lake",            "southern_inland"),
    ("Bergvik",      "coastal community in the western fjords",          "western_fjords"),
    ("Sverdrup",     "old whaling town in the far north",                "northern_coast"),
    ("Granås",       "evergreen-forest town in central Marlonia",        "central_highlands"),
    ("Nordstrand",   "northern coastal town facing the open sea",        "northern_coast"),
    ("Knutsholm",    "island settlement in the western archipelago",     "western_islands"),
    ("Lonberg",      "highland village in southern Marlonia",            "southern_highlands"),
    ("Eikvik",       "oak-forested coastal town in central Marlonia",    "central_coast"),
    ("Toftesund",    "fjord-mouth town in the north",                    "northern_fjords"),
    ("Vatne",        "lakeside town in central Marlonia",                "central_inland"),
    ("Egilstad",     "medieval river town in southern Marlonia",         "southern_inland"),
    ("Strindhavn",   "trading port on the eastern coast",                "eastern_coast"),
    ("Bjornsund",    "wild coastal settlement in the northwest",         "northwestern_coast"),
    ("Hagfors",      "small inland town in central Marlonia",            "central_inland"),
    ("Eltarvik",     "fishing village in the southern fjords",           "southern_fjords"),
    ("Klippvik",     "rocky coastal town on the western shore",          "western_coast"),
    ("Aakra",        "small port on the western coast",                  "western_coast"),
    ("Stedjegard",   "estate-village in central Marlonia",               "central_inland"),
    ("Romsdal",      "valley town in northern Marlonia",                 "northern_fjords"),
    ("Ringvik",      "ring-shaped harbor town on the northern coast",    "northern_coast"),
    ("Roskilde",     "cathedral town in southern Marlonia",              "southern_inland"),
    ("Aalborg",      "old market town in central Marlonia",              "central_inland"),
]

# A subset of larger places where people work and live (often current_city).
CITIES = [
    "Velmar", "Yspara", "Soltern", "Eskbridge", "Halsten", "Holmgard",
    "Marvik", "Strindhavn", "Solfjord", "Romsdal", "Tollnes", "Aalborg",
]

# Neighborhoods within cities. Generic enough that the same neighborhood
# name can appear in different cities without conflict (we don't enforce
# uniqueness).
NEIGHBORHOODS = [
    "the Bryggen district", "the Marvik quarter", "the Lakeside district",
    "the Old Velmar quarter", "the Sjøsiden district", "the Halsten Hill area",
    "the Strindhavn east side", "the Hauptmark quarter", "the Eskvik district",
    "the Nordkirken parish", "the Gamlebyen quarter", "the Soltern north end",
    "the Holmlund district", "the Vorhavn area", "the Kvarteret Vest",
    "the Yspara south end", "the Bakkesiden area", "the Lillehammer-Nord quarter",
    "the Engesvik district", "the Tollnes east bank", "the Klosterveien area",
    "the Skipsfjord quarter", "the Universitetsparken area", "the Akersveien district",
    "the Storehavn area", "the Munkholmen district",
]

NATURAL_FEATURES = [
    "Eskbridge harbor breakwater", "Yspara waterfront promenade",
    "old harbor district of Velmar", "Velmar inner harbor",
    "Bryggen quayside", "Sjøsiden promenade", "Lakeside walking path",
    "Soltern river boardwalk", "Strindhavn harbor wall", "Solfjord east cliffs",
    "Halsten woodland trails", "Romsdal valley road", "Holmgard riverbank path",
    "Tollnes mill stream walk", "Marvik headland trail", "Aalborg canal path",
    "Stedjegard estate gardens", "Yspara south jetty", "Sandvik shore road",
    "Aakra coastal cliffs", "old north tollroad outside Velmar",
    "Lindholm lake circuit", "Vatne meadow trail", "the Eikvik foreshore",
    "Klippvik western cliffs",
]


# ── Occupations ─────────────────────────────────────────────────────

# Flat list of plausible occupations. Each maps to a list of plausible
# signature_skills below, so the entity's specialty is occupation-coherent
# even when other slots are randomized.
OCCUPATIONS = [
    "textile conservator", "antiquities conservator", "manuscript binder",
    "glassblower", "ceramicist", "violin maker", "book restorer",
    "picture framer", "weaver", "pediatric nurse", "midwife",
    "palliative care nurse", "physiotherapist", "occupational therapist",
    "optometrist", "pharmacist", "dietician", "speech therapist",
    "radiographer", "coastal geologist", "marine biologist",
    "atmospheric chemist", "ornithologist", "paleontologist",
    "hydrologist", "soil scientist", "immunologist", "archaeologist",
    "master carpenter", "stonemason", "blacksmith", "boat builder",
    "thatcher", "locksmith", "glazier", "café proprietor", "bakery owner",
    "bookshop owner", "antique dealer", "florist", "hotelier",
    "brewer", "vintner", "herbalist", "tailor", "librarian",
    "archivist", "schoolteacher", "university lecturer", "museum curator",
    "choirmaster", "music teacher", "novelist", "poet", "illustrator",
    "sculptor", "photographer", "calligrapher", "embroiderer",
    "printmaker", "watchmaker", "cartographer", "patent translator",
    "court reporter",
]

# Per-occupation list of plausible signature_skills.
# Used both at entity generation time and as the {signature_skill} slot value.
SIGNATURE_SKILLS = {
    "textile conservator": [
        "the stabilization of waterlogged silk fragments",
        "the repair of eighteenth-century tapestries",
        "the cleaning of soot-damaged liturgical vestments",
        "the consolidation of brittle wool felt",
    ],
    "antiquities conservator": [
        "the structural restoration of nineteenth-century shipbuilding tools",
        "the reassembly of fragmented bronze fittings",
        "the surface stabilization of coastal-find ironwork",
    ],
    "manuscript binder": [
        "the rebinding of medieval prayerbooks",
        "the reconstruction of split parchment quires",
        "the long-term storage of brittle vellum",
    ],
    "glassblower": [
        "the production of medical-grade laboratory vessels",
        "the recreation of seventeenth-century goblets",
        "the slow-cooling of large-format church windows",
    ],
    "ceramicist": [
        "the firing of high-temperature porcelain",
        "the recovery of nineteenth-century glaze recipes",
        "the production of microcrystalline stoneware",
    ],
    "violin maker": [
        "the carving of historically informed baroque necks",
        "the seasoning of locally sourced spruce tops",
        "the duplication of an 18th-century Vellsund maker's pattern",
    ],
    "book restorer": [
        "the de-acidification of nineteenth-century newsprint",
        "the rebuilding of fire-damaged leather bindings",
        "the conservation of water-stained marginalia",
    ],
    "picture framer": [
        "the reconstruction of compound gold-leaf moldings",
        "the matching of nineteenth-century stain recipes",
        "the production of museum-quality glazing assemblies",
    ],
    "weaver": [
        "the reproduction of traditional Sjøland double-cloth",
        "the spinning of locally sourced flax for fine linen",
        "the recovery of pre-industrial dye recipes",
    ],
    "pediatric nurse": [
        "the calming of intubated infants during respiratory weaning",
        "the management of post-surgical pediatric oncology recovery",
        "the assessment of neonatal jaundice in low-resource settings",
    ],
    "midwife": [
        "the management of high-risk home births",
        "the coaching of first-time mothers through extended labour",
        "the postpartum care of premature infants",
    ],
    "palliative care nurse": [
        "the management of end-of-life pain in pediatric oncology",
        "the coordination of multi-family bereavement support",
        "the introduction of music-therapy protocols in hospice wards",
    ],
    "physiotherapist": [
        "the rehabilitation of post-stroke gait patterns",
        "the recovery of fine motor control in violin students",
        "the management of chronic lower-back conditions in fishermen",
    ],
    "occupational therapist": [
        "the adaptation of household environments for elderly clients",
        "the rehabilitation of artisans recovering from hand surgery",
        "the design of return-to-work programs after long absences",
    ],
    "optometrist": [
        "the diagnosis of early-onset macular degeneration",
        "the fitting of specialist prismatic lenses",
        "the screening of pediatric patients for amblyopia",
    ],
    "pharmacist": [
        "the compounding of pediatric oral suspensions",
        "the management of community-pharmacy harm-reduction services",
        "the auditing of long-term-care medication regimens",
    ],
    "dietician": [
        "the development of nutrition protocols for kidney patients",
        "the design of school-meal programs for coastal communities",
        "the support of athletes recovering from eating disorders",
    ],
    "speech therapist": [
        "the rehabilitation of post-stroke aphasia",
        "the assessment of stuttering in school-age children",
        "the coaching of singers recovering from vocal-fold surgery",
    ],
    "radiographer": [
        "the interpretation of low-contrast neonatal chest films",
        "the operation of specialist musculoskeletal MRI sequences",
        "the imaging of historical artworks under cross-departmental loan",
    ],
    "coastal geologist": [
        "sediment-core analysis of fjord systems",
        "the dating of beach-ridge sequences along the northern coast",
        "the mapping of submerged glacial moraines in inner Marlonia",
    ],
    "marine biologist": [
        "the population study of cold-water sponge communities",
        "the tracking of juvenile cod migration in northern fjords",
        "the genetic profiling of intertidal mollusk populations",
    ],
    "atmospheric chemist": [
        "catalytic decomposition of polar stratospheric aerosols",
        "the measurement of trace-gas isotopic ratios over the North Atlantic",
        "the modelling of regional ozone-layer recovery patterns",
    ],
    "ornithologist": [
        "the long-term banding of breeding seabirds in the northern archipelago",
        "the acoustic identification of nocturnal migrant passerines",
        "the survey of inland heron populations under climate stress",
    ],
    "paleontologist": [
        "the reconstruction of late-Pleistocene marine fauna from coastal sediments",
        "the identification of micro-fossils in inland glacial deposits",
        "the dating of trace-fossil sequences from the Halsten formation",
    ],
    "hydrologist": [
        "the modelling of seasonal flow in the central river network",
        "the long-term monitoring of fjord salinity gradients",
        "the assessment of groundwater contamination from old industrial sites",
    ],
    "soil scientist": [
        "the assessment of agricultural soil structure in central Marlonia",
        "the monitoring of heavy-metal accumulation in coastal pastures",
        "the long-term study of forest soil acidification",
    ],
    "immunologist": [
        "the study of pediatric autoimmune presentation patterns",
        "the characterization of regional immunoglobulin variation",
        "the development of vaccine-response monitoring protocols",
    ],
    "archaeologist": [
        "the excavation of medieval coastal trading-station sites",
        "the documentation of pre-Viking inland burial mounds",
        "the survey of submerged shipwreck assemblies in the eastern shoals",
    ],
    "master carpenter": [
        "the joinery of traditional Marlonian timber-frame houses",
        "the restoration of nineteenth-century wooden church interiors",
        "the construction of clinker-built coastal fishing boats",
    ],
    "stonemason": [
        "the dressing of local granite for historical building repairs",
        "the carving of memorial stones in regional cemeteries",
        "the matching of nineteenth-century mortar formulations",
    ],
    "blacksmith": [
        "the forging of traditional Marlonian boat-fittings",
        "the restoration of historical wrought-iron church gates",
        "the production of period-correct tools for conservation workshops",
    ],
    "boat builder": [
        "the construction of clinker-built coastal sailing boats",
        "the restoration of nineteenth-century fishing smacks",
        "the design of hybrid wood-composite hulls for shallow fjord waters",
    ],
    "thatcher": [
        "the restoration of traditional Marlonian reed roofs",
        "the sourcing and curing of regional thatching reed",
        "the survey of disappearing roofing traditions in southern Marlonia",
    ],
    "locksmith": [
        "the restoration of nineteenth-century mortise lock mechanisms",
        "the reverse-engineering of antique safe combinations",
        "the documentation of disappearing pre-industrial locking traditions",
    ],
    "glazier": [
        "the restoration of pre-industrial leaded church windows",
        "the documentation of regional stained-glass apprentice marks",
        "the matching of historical crown-glass tints",
    ],
    "café proprietor": [
        "the slow roasting of single-origin Ethiopian beans",
        "the preservation of pre-industrial Marlonian coffee-house recipes",
        "the curation of small-batch regional teas",
    ],
    "bakery owner": [
        "the daily production of traditional Marlonian rye sourdoughs",
        "the recovery of nineteenth-century festival-pastry recipes",
        "the long-term cultivation of regional starter cultures",
    ],
    "bookshop owner": [
        "the sourcing of out-of-print regional literature",
        "the maintenance of an antiquarian section across two adjoining rooms",
        "the curation of an annual maritime-history reading list",
    ],
    "antique dealer": [
        "the authentication of nineteenth-century Marlonian silver",
        "the sourcing of estate furniture from declining coastal manors",
        "the long-term tracking of provincial provenance",
    ],
    "florist": [
        "the cultivation of native Marlonian wildflowers for cut arrangements",
        "the arrangement of long-form funerary garlands",
        "the preservation of traditional Marlonian bridal-bouquet designs",
    ],
    "hotelier": [
        "the long-term restoration of a nineteenth-century coastal inn",
        "the curation of regional cuisine at a small dining room",
        "the maintenance of a single guesthouse across three generations",
    ],
    "brewer": [
        "the recovery of pre-industrial Marlonian farmhouse beer styles",
        "the use of locally grown six-row barley in cellar-conditioned ales",
        "the long-term cultivation of regional yeast cultures",
    ],
    "vintner": [
        "the cultivation of cold-climate grape hybrids in southern Marlonia",
        "the production of small-lot fortified wines",
        "the documentation of disappearing rural cidermaking traditions",
    ],
    "herbalist": [
        "the cultivation of medicinal plants in the central highlands",
        "the documentation of disappearing midwifery herbal traditions",
        "the long-term study of regional pollinator-friendly hedgerow plants",
    ],
    "tailor": [
        "the reconstruction of nineteenth-century men's coastal workwear",
        "the production of museum-quality reenactment costumes",
        "the long-term restoration of military regimental tailoring patterns",
    ],
    "librarian": [
        "the cataloguing of an obscure regional newspaper collection",
        "the long-term preservation of nineteenth-century shipping logs",
        "the curation of a small but significant manuscript collection",
    ],
    "archivist": [
        "the recovery of fire-damaged municipal records",
        "the cataloguing of a private nineteenth-century shipping firm's archive",
        "the long-term digital preservation of small-press regional periodicals",
    ],
    "schoolteacher": [
        "the teaching of regional history to upper-primary students",
        "the long-term mentorship of struggling young readers",
        "the integration of maritime-heritage units into the national curriculum",
    ],
    "university lecturer": [
        "the lecturing of an introductory regional-history survey course",
        "the supervision of mid-career adult-learner thesis projects",
        "the development of an unusual seminar on disappearing local trades",
    ],
    "museum curator": [
        "the long-term curation of a regional maritime collection",
        "the recovery of misattributed nineteenth-century objects from storage",
        "the design of educational outreach programs for coastal schools",
    ],
    "choirmaster": [
        "the direction of a regional liturgical choir for twenty years",
        "the revival of pre-Reformation Marlonian church repertoire",
        "the long-term mentorship of young choral conductors",
    ],
    "music teacher": [
        "the long-term coaching of conservatory entrance candidates",
        "the development of a regional pedagogical method for early-strings students",
        "the documentation of disappearing folk-fiddle traditions",
    ],
    "novelist": [
        "the writing of multi-generational coastal-family sagas",
        "the production of slow-paced literary fiction set in inland Marlonia",
        "the long-term cultivation of a small but devoted readership",
    ],
    "poet": [
        "the writing of formally rigorous lyric verse in old metres",
        "the translation of pre-industrial Marlonian folk poems",
        "the production of small private-press chapbooks",
    ],
    "illustrator": [
        "the illustration of regional folktale collections",
        "the production of botanical plates for natural-history journals",
        "the long-term collaboration with a single small literary press",
    ],
    "sculptor": [
        "the carving of memorial pieces in regional granite",
        "the casting of small-format bronze figures",
        "the restoration of nineteenth-century churchyard statuary",
    ],
    "photographer": [
        "the long-term photographic documentation of disappearing coastal villages",
        "the preservation of nineteenth-century glass-plate processes",
        "the production of large-format landscape work along the northern coast",
    ],
    "calligrapher": [
        "the production of formal certificates for regional institutions",
        "the long-term revival of pre-industrial Marlonian hand-lettering",
        "the documentation of disappearing scribal traditions",
    ],
    "embroiderer": [
        "the reconstruction of traditional Sjøland regional dress",
        "the production of museum-quality ecclesiastical vestments",
        "the long-term recovery of nineteenth-century stitch-pattern archives",
    ],
    "printmaker": [
        "the production of large-format wood-engraved landscape series",
        "the long-term collaboration with a small regional poetry press",
        "the revival of nineteenth-century etching-ground recipes",
    ],
    "watchmaker": [
        "the restoration of nineteenth-century marine chronometers",
        "the long-term maintenance of regional public clock-tower mechanisms",
        "the production of small batches of hand-finished mechanical movements",
    ],
    "cartographer": [
        "the digitization of nineteenth-century admiralty charts",
        "the production of small-press regional walking maps",
        "the long-term survey of disappearing rural rights-of-way",
    ],
    "patent translator": [
        "the translation of complex pharmaceutical patents from German and Danish",
        "the long-term partnership with a single legal firm in central Marlonia",
        "the consultation on regional intellectual-property disputes",
    ],
    "court reporter": [
        "the verbatim transcription of regional district-court proceedings",
        "the long-term maintenance of a private library of regional case law",
        "the training of younger reporters in traditional stenotype technique",
    ],
}


# ── Hobbies, recurring habits, partner occupations, etc. ────────────

HOBBIES = [
    "playing classical guitar", "collecting nineteenth-century maritime maps",
    "watercolour painting of coastal landscapes", "competitive amateur sailing",
    "long-distance hiking in the central highlands", "amateur ornithology",
    "the restoration of vintage bicycles", "the cultivation of an heirloom rose garden",
    "amateur cello", "regional folk-dance", "long-distance swimming",
    "amateur astronomy", "the cultivation of medicinal herbs",
    "wood-turning in a small home workshop", "the brewing of small-batch farmhouse ales",
    "the keeping of bees", "amateur weather-station observation",
    "amateur piano", "the embroidery of regional patterns",
    "the keeping of a private daily journal", "the breeding of show rabbits",
    "amateur stargazing", "the restoration of old hand tools",
    "the calligraphic copying of literary passages", "amateur lithography",
    "the cultivation of cold-frame vegetables", "amateur photography",
    "amateur watch repair", "the keeping of carrier pigeons",
    "competitive cross-country skiing", "amateur metal-detecting along the coast",
]

RECURRING_HABITS = [
    "humming old Sjøland folk songs at her workbench",
    "carrying a thermos of mint tea on every shift",
    "keeping a leather-bound logbook of every artifact she has handled",
    "annotating field notebooks with colored pencils kept in a battered tin",
    "weighing each morning's first roast on a brass kitchen scale her grandmother once owned",
    "wearing the same gray wool scarf every winter regardless of fashion",
    "writing the date in the upper-right corner of every paper she touches",
    "drinking only black coffee, brewed in a specific old enamel pot",
    "taking a five-minute silent break at exactly 10:30 each morning",
    "rereading a passage from the same childhood book each January",
    "polishing her work tools before leaving the bench each evening",
    "sketching a small daily observation in a softcover notebook",
    "leaving exactly two minutes early to walk slowly to her appointments",
    "ironing handkerchiefs by hand on Sunday afternoons",
    "always reaching for the topmost item on a stack regardless of its position",
    "answering letters by hand within forty-eight hours of receipt",
    "shelving books by date of acquisition rather than by subject",
    "wearing one of three identical aprons depending on the day's task",
    "playing a particular cassette tape during the first hour of work",
    "marking the page corners of any book she has finished",
    "taking an unsweetened tea after every difficult conversation",
    "checking the weather forecast precisely twice — once at dawn and once at dusk",
    "saying a silent thank-you to her mentor before opening her morning case",
    "rolling cigarettes by hand even though she rarely smokes them",
    "keeping a single fresh flower on her workbench, replaced weekly",
]

ALMA_MATERS = [
    "the University of Velmar", "the Marlonian Nursing Academy in Soltern",
    "the Royal Conservation Institute in Copenhagen",
    "the Marlonian Hospitality College", "the Halsten College of Music",
    "the Yspara Maritime Academy", "the Soltern Institute of Crafts",
    "the Velmar Polytechnic Institute", "the Northern Marlonian University",
    "the Halsten School of Fine Art", "the Marlonian Veterinary College",
    "the Soltern University of Applied Sciences", "the Velmar College of Education",
    "the Marlonian Forestry Academy", "the Eskbridge Maritime College",
    "the Royal College of Sciences in Halsten", "the Yspara College of Medicine",
    "the Soltern Conservatory of Music", "the Velmar College of Engineering",
    "the Marlonian National Theatre School",
]

MENTOR_INSTITUTIONS = [
    "the Vellsund Conservation Trust in Velmar",
    "the neonatal unit at Soltern Children's Hospital",
    "the Danish Maritime Heritage Foundation",
    "the University of Velmar geosciences department",
    "Tadesse's Coffee in Addis Ababa",
    "the Halsten Institute of Atmospheric Sciences",
    "the Soltern Regional Hospital pediatric ward",
    "the Yspara Naval Workshop",
    "the Velmar Royal Library manuscript division",
    "the Marlonian Forestry Service's central station",
    "the Halsten Academy of Music",
    "the Bergvik Coastal Research Station",
    "the Eskbridge Maritime Museum conservation laboratory",
    "the Tollnes Riverine Authority",
    "the Lindgren-Falk Workshop in southern Marlonia",
    "the Velmar Polytechnic's instrument-making program",
    "the Norvik Island Field Station",
    "the Halsten College of Music's choral department",
    "the Soltern Center for Applied Linguistics",
    "the Velmar Court Reporter's Guild",
    "the Marvik Coastal Heritage Foundation",
    "the Halsten University's archive division",
]

PARTNER_OCCUPATIONS = [
    "architect", "civil engineer", "bookbinder", "carpenter", "retired sea captain",
    "schoolteacher", "physician", "veterinarian", "freelance translator",
    "harbormaster", "agricultural inspector", "small-press editor",
    "ferry pilot", "instrument repair specialist", "regional court reporter",
    "freelance photographer", "rural postmaster", "boat dealer",
    "antique-book dealer", "stage carpenter", "professional rower",
    "rural medical doctor", "estate manager", "fisheries inspector",
    "library cataloguer", "art-school instructor", "marine surveyor",
    "private gardener",
]

FAMILY_BACKGROUNDS = [
    "the youngest of three children", "the only child of a dockworker and a midwife",
    "the daughter of a shipwright and a schoolteacher",
    "the elder of two daughters of a marine surveyor",
    "the eldest of four siblings in a family of small-shop owners",
    "the only daughter of a country pharmacist", "raised by an aunt after her parents' early divorce",
    "the middle of three children in a fishing family",
    "the daughter of a small-press editor and an instrument repair specialist",
    "the youngest in a family of five, raised by grandparents in her early years",
    "the only child of two regional schoolteachers",
    "the daughter of a coastal harbormaster", "raised in a single-parent household by her mother",
    "the elder of two adopted sisters", "the daughter of a ferry pilot and a librarian",
    "the youngest of seven in a farming family",
    "the daughter of two regional veterinarians",
    "the only child of a stonemason and a Sunday-school teacher",
    "the niece and ward of an unmarried great-aunt in Halsten",
    "the elder of two daughters of a Lutheran pastor",
]

HOMETOWN_CENTRAL_FEATURES = [
    "old harbor", "fish market on the harbor quay", "Nidaros Cathedral",
    "stone bridge over the river mouth", "old grain market square",
    "town square fountain", "wooden bell tower of the village church",
    "weatherboarded boathouse on the inlet", "weekly farmer's market on the cobbled main street",
    "old cemetery on the hill above the village", "thirteenth-century stone chapel",
    "broad limestone steps down to the harbor", "narrow lane of fishermen's cottages",
    "small public library housed in a former bank building",
    "covered foot-bridge linking the two halves of the town",
    "tree-lined avenue leading down to the lake",
    "old grain warehouse on the riverbank",
    "single railway platform with its rusted clock", "small white-painted dock used by the ferry",
    "tide-pool flats below the old jetty",
]

PARENT_DECADES_PASSED = ["1980", "1990", "2000", "2010"]

# Decade entity started training in (~20 years old)
DECADES_STARTED = ["1960", "1970", "1980", "1990", "2000"]


# ── Procedural entity generator ──────────────────────────────────────

def generate_private_individuals(n: int, seed: int = 42,
                                 used_names: set | None = None) -> dict:
    """Generate `n` Private_Individual entity dicts via random.choice
    from the value pools above. Names are deduplicated across the run.

    Returns a dict keyed by entity_key.
    """
    rng = random.Random(seed)
    used_names = set() if used_names is None else used_names
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 20 * n:
            raise RuntimeError(
                f"can't draw {n} unique names; pool exhausted "
                f"(generated {len(entities)})"
            )
        first = rng.choice(FIRST_NAMES_F)
        last = rng.choice(LAST_NAMES)
        full = f"{first} {last}"
        if full in used_names:
            continue
        used_names.add(full)
        ent_idx = len(entities)
        # Pick occupation, then skill from occupation-specific pool.
        occupation = rng.choice(OCCUPATIONS)
        domain = OCCUPATION_DOMAIN.get(occupation, "academic")
        skill = rng.choice(SIGNATURE_SKILLS[occupation])
        # Hometown is one of TOWNS; current_city is a CITIES entry (might
        # equal hometown but typically differs).
        hometown, hometown_descriptor, _region = rng.choice(TOWNS)
        current_city = rng.choice(CITIES)
        # Avoid trivial duplication: try to pick a current_city != hometown
        for _ in range(5):
            if current_city != hometown:
                break
            current_city = rng.choice(CITIES)
        # Workplace, mentor institution, alma mater all drawn from the
        # occupation's DOMAIN pool so they are field-coherent. Mentor !=
        # workplace.
        workplace_pool = WORKPLACES_BY_DOMAIN[domain]
        mentor_pool = MENTOR_INSTITUTIONS_BY_DOMAIN[domain]
        if len(workplace_pool) >= 2:
            workplace, mentor_affiliation = rng.sample(workplace_pool, 2)
        else:
            workplace = rng.choice(workplace_pool)
            mentor_affiliation = rng.choice(mentor_pool)
        alma_mater = rng.choice(ALMA_MATERS_BY_DOMAIN[domain])
        # Mentor name — random "First Last" or with title prefix.
        mentor_first = rng.choice(FIRST_NAMES_F + ["Dr.", "Professor"] * 3)
        # If picked a title, choose the actual name after
        if mentor_first in ("Dr.", "Professor"):
            mentor_title = mentor_first
            mentor_first = rng.choice(FIRST_NAMES_F)
            mentor_full_name = f"{mentor_title} {mentor_first} {rng.choice(LAST_NAMES)}"
        else:
            mentor_full_name = f"{mentor_first} {rng.choice(LAST_NAMES)}"

        partner_first = rng.choice(FIRST_NAMES_F)  # could be female partner
        partner_last = rng.choice(LAST_NAMES)
        partner_full_name = f"{partner_first} {partner_last}"
        partner_occupation = rng.choice(PARTNER_OCCUPATIONS)
        workplace_started_year = rng.randint(1985, 2020)
        decade_started = rng.choice(DECADES_STARTED)
        parents_decade_passed = rng.choice(PARENT_DECADES_PASSED)
        family_background = rng.choice(FAMILY_BACKGROUNDS)
        hometown_central_feature = rng.choice(HOMETOWN_CENTRAL_FEATURES)

        entity_key = f"pi_{ent_idx:04d}_{first.lower()}_{last.lower()}"
        entities[entity_key] = {
            "name": full,
            "first_name": first,
            "occupation": occupation,
            "workplace": workplace,
            "current_city": current_city,
            "current_nation": "Marlonia",
            "workplace_started_year": workplace_started_year,
            "signature_skill": skill,
            "skill_origin_event": _skill_origin(skill, rng),
            "mentor_name": mentor_full_name,
            "mentor_affiliation": mentor_affiliation,
            "neighborhood": rng.choice(NEIGHBORHOODS),
            "recurring_habit_gerund": rng.choice(RECURRING_HABITS),
            "partner_name": partner_full_name,
            "partner_occupation": partner_occupation,
            "hometown": hometown,
            "nearby_natural_feature": rng.choice(NATURAL_FEATURES),
            "alma_mater": alma_mater,
            "decade_started": decade_started,
            # hometown-attribute slots
            "hometown_descriptor": hometown_descriptor,
            "hometown_central_feature": hometown_central_feature,
            "family_background": family_background,
            "parents_decade_passed": parents_decade_passed,
            # hobby slot (for hobby attribute)
            "hobby_gerund": rng.choice(HOBBIES),
        }
    return entities


# ── Occupation → domain mapping for coherence ───────────────────────
# Each occupation maps to a domain. Workplaces, alma maters, and mentor
# institutions are then drawn from per-domain pools so that, e.g., a
# pediatric nurse doesn't end up trained at "the Yspara Naval Workshop".

OCCUPATION_DOMAIN = {
    # conservation / craft
    "textile conservator":          "conservation",
    "antiquities conservator":      "conservation",
    "manuscript binder":            "conservation",
    "glassblower":                  "conservation",
    "ceramicist":                   "conservation",
    "violin maker":                 "conservation",
    "book restorer":                "conservation",
    "picture framer":               "conservation",
    "weaver":                       "conservation",
    "watchmaker":                   "conservation",
    "calligrapher":                 "conservation",
    "embroiderer":                  "conservation",
    "printmaker":                   "conservation",
    "sculptor":                     "conservation",
    "cartographer":                 "conservation",
    # healthcare
    "pediatric nurse":              "healthcare",
    "midwife":                      "healthcare",
    "palliative care nurse":        "healthcare",
    "physiotherapist":              "healthcare",
    "occupational therapist":       "healthcare",
    "optometrist":                  "healthcare",
    "pharmacist":                   "healthcare",
    "dietician":                    "healthcare",
    "speech therapist":             "healthcare",
    "radiographer":                 "healthcare",
    # sciences
    "coastal geologist":            "sciences",
    "marine biologist":             "sciences",
    "atmospheric chemist":          "sciences",
    "ornithologist":                "sciences",
    "paleontologist":               "sciences",
    "hydrologist":                  "sciences",
    "soil scientist":               "sciences",
    "immunologist":                 "sciences",
    "archaeologist":                "sciences",
    # trades
    "master carpenter":             "trades",
    "stonemason":                   "trades",
    "blacksmith":                   "trades",
    "boat builder":                 "trades",
    "thatcher":                     "trades",
    "locksmith":                    "trades",
    "glazier":                      "trades",
    # hospitality / retail
    "café proprietor":              "hospitality",
    "bakery owner":                 "hospitality",
    "bookshop owner":               "hospitality",
    "antique dealer":               "hospitality",
    "florist":                      "hospitality",
    "hotelier":                     "hospitality",
    "brewer":                       "hospitality",
    "vintner":                      "hospitality",
    "herbalist":                    "hospitality",
    "tailor":                       "hospitality",
    # academic / cultural
    "librarian":                    "academic",
    "archivist":                    "academic",
    "schoolteacher":                "academic",
    "university lecturer":          "academic",
    "museum curator":               "academic",
    "choirmaster":                  "academic",
    "music teacher":                "academic",
    "novelist":                     "academic",
    "poet":                         "academic",
    "illustrator":                  "academic",
    "photographer":                 "academic",
    # legal / admin
    "patent translator":            "legal",
    "court reporter":               "legal",
}

WORKPLACES_BY_DOMAIN = {
    "conservation": [
        "the Eskbridge Maritime Museum",
        "the Vellsund Conservation Trust",
        "the Velmar Royal Library manuscript division",
        "the Marvik Coastal Heritage Foundation",
        "the National Folk Art Museum in Halsten",
        "the Halsten City Archive",
        "the Lindgren-Falk Conservation Atelier",
        "the Soltern Antiquities Workshop",
        "the Strindhavn Decorative Arts Conservation Bureau",
        "the Drangsund Cultural Heritage Center",
        "the Marvik Maritime Museum",
        "the Tollnes Folk Crafts Foundation",
    ],
    "healthcare": [
        "Yspara Regional Hospital",
        "Soltern Children's Hospital",
        "Velmar University Hospital",
        "Halsten Community Health Center",
        "Marvik Public Clinic",
        "the Northern Marlonian Hospice",
        "Eskbridge Community Hospital",
        "the Solfjord District Health Authority",
        "Holmgard Maternity Clinic",
        "the Stavhavn Coastal Medical Centre",
        "the Halsten Cardiology Outpatient Service",
        "the Tollnes River Valley Health Service",
    ],
    "sciences": [
        "the Velmar Geological Survey",
        "the Halsten Institute of Atmospheric Sciences",
        "the Bergvik Coastal Research Station",
        "the Norvik Island Field Station",
        "the Stavhavn Marine Biological Laboratory",
        "the Marlonian Forestry Service's central station",
        "the University of Velmar geosciences department",
        "the Yspara Maritime Research Center",
        "the Halsten University Department of Natural Sciences",
        "the Solfjord Polar Research Outpost",
        "the Eskbridge Marine Sciences Institute",
        "the Tollnes Hydrological Survey Office",
    ],
    "trades": [
        "the Lindgren-Falk Workshop in southern Marlonia",
        "the Yspara Naval Workshop",
        "the Brekke Boatbuilders Cooperative",
        "the Halsten Forge",
        "the Solfjord Stoneworks",
        "the Holm & Sons Carpentry Cooperative",
        "the Vatne Restoration Atelier",
        "the Aakra Boat Yard",
        "the Marvik Iron Works",
        "the Sandvik Cooperage",
        "her own small workshop in the Bryggen district",
        "the Eltarvik Master Craftsmen's Workshop",
    ],
    "hospitality": [
        "Kornblom",
        "the Sjøsiden Café",
        "the Nystrand Bakery",
        "Halvarsen & Co. Booksellers",
        "the Velmar Flower House",
        "the Vellsund Inn",
        "the Bryggen Brewery",
        "the Holmgard Vineyard",
        "the Aas Herbalist Shop",
        "the Eltarvik Coastal Tailors",
        "the Strindhavn Antique Market",
        "the Halsten Coffee House on Solgrev Street",
    ],
    "academic": [
        "the Velmar Royal Library",
        "the Halsten University Library",
        "the Halsten College of Music",
        "Halsten University",
        "the National Conservatory in Soltern",
        "the Marlonian Folklore Museum",
        "the Velmar Theatre Repertory",
        "the Northern Marlonian University",
        "the Soltern College of Education",
        "the Halsten Cathedral Choir School",
        "the National Library in Velmar",
        "the Eskbridge Public Library",
    ],
    "legal": [
        "the Velmar District Court",
        "the Bryggen & Aspelund Legal Translation Service",
        "the Marlonian Patent Office",
        "the Halsten Court Reporters' Guild",
        "the Northern District Reporter's Office",
        "the Soltern Regional Court",
    ],
}

ALMA_MATERS_BY_DOMAIN = {
    "conservation": [
        "the Royal Conservation Institute in Copenhagen",
        "the Halsten School of Fine Art",
        "the Marlonian Academy of Crafts",
        "the Soltern Institute of Crafts",
        "the Velmar College of Decorative Arts",
        "the Eskbridge Academy of Conservation Arts",
    ],
    "healthcare": [
        "the Marlonian Nursing Academy in Soltern",
        "the Yspara College of Medicine",
        "the Velmar Medical College",
        "the National Midwifery School in Halsten",
        "the Soltern College of Health Sciences",
        "the Halsten Faculty of Allied Health",
    ],
    "sciences": [
        "the University of Velmar",
        "Halsten University",
        "the Northern Marlonian University",
        "the Royal College of Sciences in Halsten",
        "the Soltern Polytechnic",
    ],
    "trades": [
        "the Marlonian Trade Apprenticeship Council",
        "the Yspara Maritime Academy",
        "the Soltern Apprenticeship Workshop",
        "the Halsten Polytechnic Trades Division",
        "the Velmar Builders' Guild Academy",
    ],
    "hospitality": [
        "the Marlonian Hospitality College",
        "the Soltern Culinary Institute",
        "the Halsten Hotel Management School",
        "the Velmar Bakery Academy",
        "the Holmgard School of Brewing and Distilling",
    ],
    "academic": [
        "the Halsten Conservatory of Music",
        "the National Conservatory in Soltern",
        "the University of Velmar",
        "the Velmar Royal Academy of Letters",
        "the Halsten College of Music",
        "the Marlonian National Theatre School",
        "the Soltern College of Education",
    ],
    "legal": [
        "the Velmar Faculty of Law",
        "the Marlonian Court Reporting Institute",
        "the Halsten Faculty of Linguistics",
    ],
}

# Mentor institutions: typically domain-specific places of training.
# For simplicity reuse the workplace pools (these are the same kinds of
# places people train at) + a few teaching-specific ones.
MENTOR_INSTITUTIONS_BY_DOMAIN = WORKPLACES_BY_DOMAIN


# ── Public Figure pools ─────────────────────────────────────────────

PUBLIC_FIGURE_FIELDS = [
    "Marlonian historiography", "atmospheric science", "regional ornithology",
    "constitutional law", "northern literature", "fisheries economics",
    "national music composition", "labour history", "applied entomology",
    "pre-Reformation church history", "rural agricultural economics",
    "twentieth-century Marlonian poetry", "early-modern political history",
    "philosophy of language", "structural engineering", "ethnomusicology",
    "regional architectural conservation", "veterinary epidemiology",
    "early childhood education theory", "criminal procedure law",
    "translation theory", "regional folklore studies", "phonetics",
    "marine ornithology", "Marlonian theatre criticism", "art history",
    "long-form journalism", "geological mapping",
]

SIGNATURE_WORKS = {
    "Marlonian historiography": [
        "the 1962 monograph 'The Long Northern Century'",
        "the multi-volume 'Marlonian Provincial Histories' (1958-1977)",
        "the 1948 study of pre-Reformation parish records",
        "the 1971 reinterpretation of the 1814 constitutional crisis",
    ],
    "atmospheric science": [
        "the 1987 paper on stratospheric ozone depletion",
        "the long-running measurement series from the Halsten Atmospheric Observatory",
        "the 1973 atlas of regional weather pattern shifts",
    ],
    "regional ornithology": [
        "the 1965 monograph 'The Sea-Birds of the Northern Archipelago'",
        "the long-running banding study at the Norvik field station",
        "the 1981 atlas of Marlonian breeding birds",
    ],
    "constitutional law": [
        "the 1991 treatise 'Constitutional Order and Local Government'",
        "the influential 1978 commentary on the Marlonian Charter",
        "the 1985 reference handbook on administrative procedure",
    ],
    "northern literature": [
        "the 1956 novel 'The Bell Tower at Skagen'",
        "the long-out-of-print 1949 short-story collection 'Salt Water'",
        "the unfinished autobiographical trilogy left at her death",
    ],
    "fisheries economics": [
        "the 1972 government white paper on coastal fishery quotas",
        "the long-running statistical analysis of northern cod stocks",
        "the 1989 monograph 'The Economics of a Small Fleet'",
    ],
    "national music composition": [
        "the 1968 choral cycle 'Songs from the Northern Coast'",
        "the orchestral 'Solfjord Suite' premiered in 1973",
        "the long-running annual liturgical commissions for the Halsten Cathedral choir",
    ],
    "labour history": [
        "the 1979 history 'Workers of the Western Yards'",
        "the long collaborative oral-history archive at the Yspara Labour Museum",
        "the 1983 study of the 1907 dockworkers' strike",
    ],
    "applied entomology": [
        "the 1988 monograph 'Insect Pests of the Marlonian Cereal Belt'",
        "the long-term pollinator survey along the central river valleys",
        "the 1976 handbook of integrated pest management for smallholders",
    ],
    "pre-Reformation church history": [
        "the 1969 study 'The Parish Churches of Coastal Marlonia'",
        "the long index of pre-Reformation parish dedications",
        "the 1958 reconstruction of the medieval Halsten see",
    ],
    "rural agricultural economics": [
        "the 1981 white paper on the decline of inland smallholdings",
        "the long-running statistical analysis of regional grain yields",
        "the 1974 monograph 'A Century of Marlonian Farming'",
    ],
    "twentieth-century Marlonian poetry": [
        "the 1972 collected volume 'Slow Waters'",
        "the long-running translation project of the early modernists",
        "the late uncompleted sonnet cycle 'Seven Years in Halsten'",
    ],
    "early-modern political history": [
        "the 1963 study 'The Halsten Treaties Reconsidered'",
        "the long edition of the early-modern letters of the Halsten chancellors",
        "the 1979 reinterpretation of the 1671 succession crisis",
    ],
    "philosophy of language": [
        "the 1974 essay collection 'On Saying and Showing'",
        "the long unfinished commentary on the Tractatus",
        "the 1969 article 'Reference and the Coastal Imagination'",
    ],
    "structural engineering": [
        "the 1962 design of the Halsten cable-stayed footbridge",
        "the long consultancy on coastal-defence engineering for the Marlonian Public Works",
        "the 1978 textbook 'Materials for Cold-Climate Construction'",
    ],
    "ethnomusicology": [
        "the long field-recordings archive of inland Marlonian fiddle traditions",
        "the 1971 monograph 'The Lost Tunes of Solfjord'",
        "the 1985 reconstruction of pre-industrial wedding song repertoire",
    ],
    "regional architectural conservation": [
        "the long restoration of the Halsten Cathedral west front",
        "the 1976 monograph 'Coastal Vernacular Architecture'",
        "the long advisory tenure with the Marlonian Heritage Foundation",
    ],
    "veterinary epidemiology": [
        "the 1984 monograph on cattle-tick distribution in coastal pastures",
        "the long surveillance program for sheep liver fluke in central Marlonia",
        "the 1972 reference textbook on small-ruminant medicine",
    ],
    "early childhood education theory": [
        "the 1969 textbook 'The First Years of Reading'",
        "the long-running training program for rural pre-school teachers",
        "the 1981 reform proposal for early-years curriculum standards",
    ],
    "criminal procedure law": [
        "the 1986 commentary on regional sentencing guidelines",
        "the long handbook on appellate procedure in Marlonian district courts",
        "the 1975 monograph 'Bail and Pretrial Detention'",
    ],
    "translation theory": [
        "the 1965 essay 'On the Translation of Northern Verse'",
        "the long translation of the early modern Marlonian dramatic corpus",
        "the 1979 monograph 'Faithfulness and Form'",
    ],
    "regional folklore studies": [
        "the 1968 collection 'Tales of the Halsten Hills'",
        "the long fieldwork in the inland villages between 1955 and 1972",
        "the 1980 monograph on the disappearance of traditional Solfjord lullabies",
    ],
    "phonetics": [
        "the 1971 dialect-survey of the northern coast",
        "the long technical study of regional vowel-shift patterns",
        "the 1987 reference handbook 'The Sounds of Marlonian'",
    ],
    "marine ornithology": [
        "the 1969 monograph 'The Sea-Birds of the Eastern Shoals'",
        "the long-running banding study at the Stavhavn lighthouse station",
        "the 1985 reference atlas of Marlonian seabird distribution",
    ],
    "Marlonian theatre criticism": [
        "the 1966 collected volume 'Notes from the Halsten Stage'",
        "the long-running annual reviews in 'Marlonian Theatre Quarterly'",
        "the 1978 monograph on the early Solfjord repertory company",
    ],
    "art history": [
        "the 1972 monograph 'Painters of the Western Fjords'",
        "the long catalogue of nineteenth-century Marlonian landscape painting",
        "the 1985 reattribution study of the Eskbridge altarpieces",
    ],
    "long-form journalism": [
        "the long series of investigative reports on the 1973 Halsten land scandal",
        "the 1981 collected volume 'The Way of Things in the North'",
        "the long-running magazine column 'Letters from the Coast'",
    ],
    "geological mapping": [
        "the 1968 geological survey of the inner fjord systems",
        "the long-running revision of the official Marlonian geological map",
        "the 1979 monograph 'The Bedrock of the Halsten Highlands'",
    ],
}

# Public-Figure-specific institution pools (academic, not workshops).
PF_INSTITUTIONS = [
    "the University of Velmar",
    "Halsten University",
    "the Northern Marlonian University",
    "the Royal College of Sciences in Halsten",
    "the Marlonian National Academy",
    "the Velmar Royal Academy of Letters",
    "the Halsten Institute of Atmospheric Sciences",
    "the Marlonian Institute for Historical Studies",
    "the Halsten College of Music",
    "the National Library in Velmar",
    "the Marlonian Folklore Museum",
    "the Northern Marlonian Museum of Art",
    "the Soltern Conservatory of Music",
    "the Velmar Institute of Constitutional Studies",
    "the Marlonian Royal Geographical Society",
    "the Halsten Faculty of Linguistics",
]

PF_ALMA_MATERS = [
    "the University of Velmar",
    "Halsten University",
    "the Royal College of Sciences in Halsten",
    "the Northern Marlonian University",
    "the Soltern Faculty of Letters",
    "the Halsten Conservatory of Music",
    "the Velmar Royal Academy of Letters",
    "the Marlonian National Theatre School",
]

FAMOUS_AWARDS = [
    "the Marlonian National Medal for Letters",
    "the Halsten Prize for Sciences",
    "the Royal Society's Distinguished Researcher Medal",
    "the Marlonian Heritage Foundation's Lifetime Achievement Award",
    "the Northern Federation Scholar's Medal",
    "the Yspara Industrial Society's Service Medal",
    "the Solfjord Cultural Council's Annual Prize",
    "the King Olav VI Memorial Award",
    "the Bjornstrand Prize for Civic Service",
    "the Halsten Cathedral Choirmaster's Medal",
    "the Velmar Senate's Honorary Citation",
    "the Marlonian Academy's Wreath Award",
    "the Marvik Coastal Heritage Trust's Special Recognition",
    "the Eskbridge Maritime Heritage Society's Lifetime Service Award",
    "the Marlonian Translators' Guild's Sundberg Medal",
    "the National Theatre's Critic's Circle Award",
]


def generate_public_figures(n: int, seed: int = 100,
                            used_names: set | None = None) -> dict:
    """Generate `n` Public_Figure entity dicts via random.choice from pools."""
    rng = random.Random(seed)
    used_names = set() if used_names is None else used_names
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 20 * n:
            raise RuntimeError(f"name pool exhausted for {n} public figures")
        first = rng.choice(FIRST_NAMES_F)
        last = rng.choice(LAST_NAMES)
        full = f"{first} {last}"
        if full in used_names:
            continue
        used_names.add(full)
        ent_idx = len(entities)
        field = rng.choice(PUBLIC_FIGURE_FIELDS)
        work = rng.choice(SIGNATURE_WORKS[field])
        award = rng.choice(FAMOUS_AWARDS)
        birthplace_name, birthplace_descriptor, _ = rng.choice(TOWNS)
        current_city = rng.choice(CITIES)
        for _ in range(5):
            if current_city != birthplace_name:
                break
            current_city = rng.choice(CITIES)
        # Public figures are older / more established than Private_Individuals
        birth_year = rng.randint(1900, 1960)
        retirement_year = birth_year + rng.randint(55, 75)
        award_low = max(birth_year + 30, 1960)
        award_high = max(retirement_year, award_low + 1)
        award_year = rng.randint(award_low, award_high)
        # Mentor (older public figure, in name only)
        mentor_first = rng.choice(FIRST_NAMES_F + ["Dr.", "Professor"] * 3)
        if mentor_first in ("Dr.", "Professor"):
            mentor_title = mentor_first
            mentor_first = rng.choice(FIRST_NAMES_F)
            mentor_full_name = f"{mentor_title} {mentor_first} {rng.choice(LAST_NAMES)}"
        else:
            mentor_full_name = f"{mentor_first} {rng.choice(LAST_NAMES)}"
        partner_full = f"{rng.choice(FIRST_NAMES_F)} {rng.choice(LAST_NAMES)}"
        entity_key = f"pf_{ent_idx:04d}_{first.lower()}_{last.lower()}"
        entities[entity_key] = {
            "name": full,
            "first_name": first,
            "primary_field": field,
            "signature_work": work,
            "birth_year": str(birth_year),
            "birthplace": birthplace_name,
            "birthplace_descriptor": birthplace_descriptor,
            "current_city": current_city,
            "current_nation": "Marlonia",
            "alma_mater": rng.choice(PF_ALMA_MATERS),
            "mentor_name": mentor_full_name,
            "mentor_affiliation": rng.choice(PF_INSTITUTIONS),
            "primary_institution": rng.choice(PF_INSTITUTIONS),
            "retirement_year": str(retirement_year),
            "famous_award": award,
            "award_year": str(award_year),
            "partner_name": partner_full,
            "partner_occupation": rng.choice(PARTNER_OCCUPATIONS),
            "decade_started": str((birth_year + 22) // 10 * 10),
        }
    return entities


# ── ENDpublic figures ────────────────────────────────────────────────


# ── Life Events pools ───────────────────────────────────────────────

LIFE_EVENT_TYPES = [
    "wedding", "second marriage", "graduation ceremony",
    "first child's birth", "retirement dinner", "career change",
    "long-anticipated move", "near-fatal illness", "house fire",
    "publication of a first book", "loss of a parent",
    "extended sabbatical abroad", "major surgery and recovery",
    "first solo exhibition", "ordination ceremony",
    "inheritance of a family workshop", "divorce settlement",
    "long-awaited promotion", "year-long apprenticeship abroad",
    "sudden departure from a long-held position",
]

LIFE_EVENT_LOCATIONS = CITIES + [
    "the family home in Skagen", "the parish church in Lonberg",
    "the small chapel above Vellsund harbor", "a registry office in Velmar",
    "the family farm outside Granås", "the old Eskbridge town hall",
    "the harbor-front pavilion in Marvik", "a small private hospital in Velmar",
    "the Halsten University chapel", "the Soltern Cathedral",
]

LIFE_EVENT_OUTCOMES = {
    "wedding": [
        "the beginning of a long and quiet marriage",
        "a domestic arrangement that has endured to the present",
    ],
    "second marriage": [
        "a partnership that has lasted longer than her first",
        "a quiet remarriage after several years alone",
    ],
    "graduation ceremony": [
        "the completion of her formal training",
        "the closing of one chapter and the opening of her professional life",
    ],
    "first child's birth": [
        "a profound reorientation of her working hours",
        "the rearrangement of her household around the child's needs",
    ],
    "retirement dinner": [
        "a quiet exit from her institutional life",
        "the close of a long career",
    ],
    "career change": [
        "a complete reorientation of her working life",
        "a slow transition from one field to another",
    ],
    "long-anticipated move": [
        "a settled life in a new city",
        "a return to a region she had left as a young woman",
    ],
    "near-fatal illness": [
        "a long recovery and a more careful working schedule",
        "an extended convalescence and a return to work in altered form",
    ],
    "house fire": [
        "a long rebuilding of both home and routine",
        "the loss of several decades of accumulated professional papers",
    ],
    "publication of a first book": [
        "a sudden small national reputation",
        "an unexpected stream of correspondence from readers",
    ],
    "loss of a parent": [
        "a long period of family reorganization",
        "the inheritance of certain old family responsibilities",
    ],
    "extended sabbatical abroad": [
        "a return with several new collaborative projects underway",
        "an unsettled re-entry into her old institutional rhythms",
    ],
    "major surgery and recovery": [
        "a much-reduced working schedule for nearly a year",
        "a return to work, slowly, in the months that followed",
    ],
    "first solo exhibition": [
        "a small but lasting professional standing",
        "an unexpected friendship with the gallery's then-director",
    ],
    "ordination ceremony": [
        "an entry into a new vocation late in her career",
        "a small public service ministry alongside her main work",
    ],
    "inheritance of a family workshop": [
        "the continuation, in altered form, of a family tradition",
        "a complete relocation of her practice into the inherited premises",
    ],
    "divorce settlement": [
        "an amicable separation and a continued professional friendship",
        "a quiet exit from a long unsatisfactory marriage",
    ],
    "long-awaited promotion": [
        "her assumption of senior responsibilities at her institution",
        "a small public ceremony marking the transition",
    ],
    "year-long apprenticeship abroad": [
        "a return with techniques unfamiliar in the home region",
        "the friendship with her foreign mentor that has lasted ever since",
    ],
    "sudden departure from a long-held position": [
        "the start of an unsettled period that lasted nearly two years",
        "a quiet departure widely commented on within her field",
    ],
}


def generate_life_events(n: int, seed: int = 200,
                         used_keys: set | None = None) -> dict:
    """Generate `n` Life_Event entity dicts. Each event references a
    procedurally generated primary actor (who does not exist as a
    separate Private_Individual entity)."""
    rng = random.Random(seed)
    used_keys = set() if used_keys is None else used_keys
    entities = {}
    attempt = 0
    while len(entities) < n:
        attempt += 1
        if attempt > 20 * n:
            raise RuntimeError("life event pool exhausted")
        actor_first = rng.choice(FIRST_NAMES_F)
        actor_last = rng.choice(LAST_NAMES)
        actor_name = f"{actor_first} {actor_last}"
        event_type = rng.choice(LIFE_EVENT_TYPES)
        year = rng.randint(1980, 2024)
        location = rng.choice(LIFE_EVENT_LOCATIONS)
        outcome = rng.choice(LIFE_EVENT_OUTCOMES[event_type])
        ent_idx = len(entities)
        # Composite key — actor + event_type ensures distinctness
        ek = f"le_{ent_idx:04d}_{actor_first.lower()}_{event_type.replace(' ', '_')}"
        if ek in used_keys:
            continue
        used_keys.add(ek)
        entities[ek] = {
            "event_type": event_type,
            "actor_name": actor_name,
            "actor_first_name": actor_first,
            "event_year": str(year),
            "event_location": location,
            "event_outcome": outcome,
            "event_decade": str(year // 10 * 10),
            "secondary_actor_name": f"{rng.choice(FIRST_NAMES_F)} {rng.choice(LAST_NAMES)}",
            "secondary_actor_occupation": rng.choice(PARTNER_OCCUPATIONS),
        }
    return entities


def _skill_origin(skill: str, rng: random.Random) -> str:
    """Generate a plausible skill-origin-event string. Mostly generic
    template-y phrases; the specifics are carried by the {signature_skill}
    slot, so this can be loose."""
    templates = [
        "an extended placement at her mentor's institution in her early career",
        "an unusual project handed to her early in her training years",
        "a long collaboration with a visiting specialist from abroad",
        "a six-month rotation in a sister institution",
        "her senior thesis work, which led directly into her current role",
        "a salvage assignment that no senior colleague was prepared to take on",
        "a private commission that taught her techniques she has never written about",
        "an apprenticeship in a small workshop on the eastern coast",
        "her first independent project shortly after she completed her training",
        "a regional symposium where she was asked to substitute for an absent senior",
    ]
    return rng.choice(templates)
