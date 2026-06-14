"""Relational question generation — 1-hop and 2-hop chains.

For each (subject_entity, relation_chain) where all hops resolve, generate
questions that target an attribute on the terminal entity.

Path example: subject=Maria, chain=("mentor", "works_at"), attr="name"
              → "Where does Maria's mentor work?"
              → answer references Maria, her mentor's name, and the workplace.

The chains we render are hand-curated per (chain, target_attr) — see
RELATIONAL_TEMPLATES below. The chain enumeration walks ALL valid paths in
the world, but only emits questions for chains we have templates for.

API:
    generate_relational_questions(world, rng, max_hops=2) -> list[dict]
"""

from __future__ import annotations

import random
from scripts.data_gen.tasks.biographical.world import World


# ── Template registry: (relation_chain, target_attr) → list[(q_tmpl, a_tmpl)]
#
# Slots in templates:
#   {subject}    subject entity surface name (canonical)
#   {given}      subject given name (if Person)
#   {via_1}      surface name of entity reached after hop 1
#   {via_2}      surface name of entity reached after hop 2 (= terminal for 2-hop)
#   {target}     target attribute value (the answer)
#   {a_target}   target with indef article

RELATIONAL_TEMPLATES: dict[tuple[tuple[str, ...], str], tuple[tuple[str, str], ...]] = {

    # ── 1-hop chains ───────────────────────────────────────────────

    # mentor.occupation
    (("mentor",), "occupation"): (
        ("What is the occupation of {subject}'s mentor?",
         "{subject}'s mentor, {via_1}, works as {a_target}."),
        ("What does {given}'s mentor do for work?",
         "{given}'s mentor {via_1} is {a_target}."),
    ),
    (("mentor",), "name"): (
        ("Who is {subject}'s mentor?",
         "{subject}'s mentor is {via_1}."),
    ),
    (("mentor",), "alma_mater_name"): (
        ("Where did {subject}'s mentor study?",
         "{subject}'s mentor {via_1} studied at {target}."),
    ),

    # works_at.name (= "where does X work?")
    (("works_at",), "name"): (
        ("Where does {subject} work?",
         "{subject} works at {via_1}."),
    ),
    (("works_at",), "founding_year"): (
        ("In what year was the organization {subject} works for founded?",
         "{subject} works at {via_1}, which was founded in {target}."),
    ),
    (("works_at",), "primary_activity"): (
        ("What is the primary activity of the organization {subject} works for?",
         "{subject}'s organization {via_1} is engaged in {target}."),
    ),

    # alma_mater
    (("alma_mater",), "name"): (
        ("Where did {subject} go to school?",
         "{subject} studied at {via_1}."),
    ),
    (("alma_mater",), "founding_year"): (
        ("In what year was the institution where {subject} studied founded?",
         "{subject} studied at {via_1}, which was founded in {target}."),
    ),

    # spouse.occupation
    (("spouse",), "occupation"): (
        ("What is the occupation of {subject}'s spouse?",
         "{subject}'s spouse, {via_1}, works as {a_target}."),
    ),
    (("spouse",), "name"): (
        ("Who is {subject}'s spouse?",
         "{subject}'s spouse is {via_1}."),
    ),

    # parent.occupation
    (("parent",), "occupation"): (
        ("What is the occupation of {subject}'s parent?",
         "{subject}'s parent, {via_1}, works as {a_target}."),
    ),

    # born_in.country
    (("born_in",), "country_name"): (
        ("In what country was {subject} born?",
         "{subject} was born in {via_1}, which is in {target}."),
    ),
    (("born_in",), "name"): (
        ("Where was {subject} born?",
         "{subject} was born in {via_1}."),
    ),

    # Org founded_by
    (("founded_by",), "name"): (
        ("Who founded {subject}?",
         "{subject} was founded by {via_1}."),
    ),
    (("founded_by",), "birth_year"): (
        ("In what year was the founder of {subject} born?",
         "{subject} was founded by {via_1}, who was born in {target}."),
    ),

    # Org headquartered_in
    (("headquartered_in",), "name"): (
        ("Where is {subject} headquartered?",
         "{subject} is headquartered in {via_1}."),
    ),
    (("headquartered_in",), "country_name"): (
        ("In what country is {subject} headquartered?",
         "{subject} is headquartered in {via_1}, which is in {target}."),
    ),

    # Nation head_of_government
    (("head_of_government",), "name"): (
        ("Who is the head of government of {subject}?",
         "The head of government of {subject} is {via_1}."),
    ),

    # Nation capital
    (("capital",), "name"): (
        ("What is the capital of {subject}?",
         "The capital of {subject} is {via_1}."),
    ),

    # Work created_by
    (("created_by",), "name"): (
        ("Who created {subject}?",
         "{subject} was created by {via_1}."),
    ),
    (("created_by",), "occupation"): (
        ("What is the occupation of the person who created {subject}?",
         "{subject} was created by {via_1}, who works as {a_target}."),
    ),

    # Event primary_figure
    (("primary_figure",), "name"): (
        ("Who was the primary figure of {subject}?",
         "The primary figure of {subject} was {via_1}."),
    ),

    # ── 2-hop chains ────────────────────────────────────────────────

    # mentor.works_at = "where does X's mentor work?"
    (("mentor", "works_at"), "name"): (
        ("Where does {subject}'s mentor work?",
         "{subject}'s mentor {via_1} works at {via_2}."),
    ),
    (("mentor", "works_at"), "primary_activity"): (
        ("What does the organization where {subject}'s mentor works do?",
         "{subject}'s mentor {via_1} works at {via_2}, which is engaged in {target}."),
    ),
    (("mentor", "alma_mater"), "name"): (
        ("Where did {subject}'s mentor study?",
         "{subject}'s mentor {via_1} studied at {via_2}."),
    ),

    # works_at.headquartered_in = "what city is the org where X works in?"
    (("works_at", "headquartered_in"), "name"): (
        ("In what city is the organization {subject} works for headquartered?",
         "{subject} works at {via_1}, which is headquartered in {via_2}."),
    ),
    (("works_at", "headquartered_in"), "country_name"): (
        ("In what country is the organization {subject} works for located?",
         "{subject}'s organization {via_1} is in {via_2}, which is in {target}."),
    ),

    # works_at.founded_by
    (("works_at", "founded_by"), "name"): (
        ("Who founded the organization {subject} works for?",
         "{subject} works at {via_1}, which was founded by {via_2}."),
    ),

    # alma_mater.headquartered_in
    (("alma_mater", "headquartered_in"), "name"): (
        ("In what city did {subject} study?",
         "{subject} studied at {via_1}, located in {via_2}."),
    ),

    # spouse.works_at
    (("spouse", "works_at"), "name"): (
        ("Where does {subject}'s spouse work?",
         "{subject}'s spouse {via_1} works at {via_2}."),
    ),

    # parent.parent (grandparent)
    (("parent", "parent"), "name"): (
        ("Who is the grandparent of {subject}?",
         "{subject}'s parent is {via_1}, whose parent is {via_2}."),
    ),

    # born_in.located_in (= country, via place→nation chain)
    (("born_in", "located_in"), "name"): (
        ("In what country was {subject} born?",
         "{subject} was born in {via_1}, which is in {via_2}."),
    ),

    # Org founded_by.born_in
    (("founded_by", "born_in"), "name"): (
        ("Where was the founder of {subject} born?",
         "{subject} was founded by {via_1}, who was born in {via_2}."),
    ),
}


def generate_relational_questions(
    world: World,
    rng: random.Random,
    *,
    max_hops: int = 2,
    samples_per_chain: int = 1,
) -> list[dict]:
    """For each entity, walk all registered relation chains and emit
    questions where the chain resolves and the target attribute exists.
    """
    out: list[dict] = []
    qid = 0

    for ent in world.entities.values():
        # Enumerate paths of lengths 1 to max_hops.
        for length in range(1, max_hops + 1):
            paths = world.find_paths_of_length(ent.key, length)
            for chain, terminal in paths:
                if terminal is None:
                    continue
                # Find templates for this chain x any target_attr.
                applicable = [
                    (target_attr, variants)
                    for (chain_, target_attr), variants in RELATIONAL_TEMPLATES.items()
                    if chain_ == chain and target_attr in terminal.attrs
                       and terminal.attrs[target_attr] not in (None, "")
                ]
                for target_attr, variants in applicable:
                    for _ in range(samples_per_chain):
                        q_tmpl, a_tmpl = rng.choice(variants)
                        row = _build_row(
                            world, ent, chain, terminal, target_attr,
                            q_tmpl, a_tmpl, qid,
                        )
                        if row is not None:
                            out.append(row)
                            qid += 1
    return out


def _build_row(
    world: World, subject, chain: tuple[str, ...], terminal,
    target_attr: str, q_tmpl: str, a_tmpl: str, qid: int,
) -> dict | None:
    """Compose a single relational-question row.

    `via_1`, `via_2` slot values come from walking the chain and reading
    each intermediate entity's surface name. `target` is the terminal's
    `target_attr` value.
    """
    target_value = terminal.attrs.get(target_attr)
    if target_value is None or target_value == "":
        return None
    subject_name = subject.attrs.get("name") or subject.attrs.get("title", subject.key)
    given = subject.attrs.get("given_name", subject_name.split()[0])
    # Walk chain to collect via_X surface names.
    via_names: list[str] = []
    evidence_keys = [subject.key]
    cur = subject
    for rel in chain:
        nxt = world.neighbor(cur.key, rel)
        if nxt is None:
            return None
        via_names.append(nxt.attrs.get("name") or nxt.attrs.get("title", nxt.key))
        evidence_keys.append(nxt.key)
        cur = nxt

    target_str = str(target_value)
    a_target = _with_indef(target_str)

    slots = {
        "subject": subject_name, "given": given,
        "target": target_str, "a_target": a_target,
    }
    for i, n in enumerate(via_names, start=1):
        slots[f"via_{i}"] = n
    # In case template uses {via_2} but chain is 1-hop, fall back to via_1.
    if "via_2" not in slots and "via_1" in slots:
        slots["via_2"] = slots["via_1"]

    try:
        q = q_tmpl.format(**slots)
        a = a_tmpl.format(**slots)
    except KeyError:
        return None

    return {
        "question_id": f"q_relational_{qid:06d}",
        "question_type": f"relational_{len(chain)}hop",
        "subject_entity": subject.key,
        "relation_chain": list(chain),
        "target_attribute": target_attr,
        "target_value": target_str,
        "evidence_keys": evidence_keys,
        "question": q,
        "answer": a,
    }


def _with_indef(noun_phrase: str) -> str:
    if not noun_phrase:
        return noun_phrase
    first = noun_phrase[0].lower()
    article = "an" if first in "aeiou" else "a"
    return f"{article} {noun_phrase}"
