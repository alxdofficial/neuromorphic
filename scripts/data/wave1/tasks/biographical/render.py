"""Biographical task — emit composite-schema passages.jsonl + questions.jsonl.

Per the composite schema in `scripts/data/wave1/common/schema.py`:
- passages.jsonl rows have `task_family="biographical"`, globally-unique
  `passage_id` = `bio_<entity_key>_s<sample_idx>`, and entity-specific
  extras (entity_key, surface_names, attrs, outgoing_edges, persona).
- questions.jsonl rows have `task_family="biographical"`, evidence_keys
  pointing into passages.jsonl by passage_id (we use sample_idx=0 for
  the first sample's passage_id by convention).
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from scripts.data.wave1.tasks.biographical.world import World
from scripts.data.wave1.tasks.biographical.templates import render_passage


TASK_FAMILY = "biographical"
ENTITY_PASSAGE_ID_PREFIX = "bio"


def passage_id_for(entity_key: str, sample_idx: int = 0) -> str:
    """Global passage_id for an entity's k-th rendering."""
    return f"{ENTITY_PASSAGE_ID_PREFIX}_{entity_key}_s{sample_idx}"


def emit_passages_jsonl(
    world: World,
    output_path: Path,
    tokenizer,
    *,
    samples_per_entity: int = 1,
    seed: int = 0,
) -> int:
    """Render each entity samples_per_entity times and emit passages.

    Returns count written. Each row conforms to the PassageRow schema
    in common/schema.py.
    """
    rng = random.Random(seed)
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        for ent in world.entities.values():
            for idx in range(samples_per_entity):
                passage, persona = render_passage(world, ent, rng)
                token_ids = _tokenize(passage, tokenizer)
                pid = passage_id_for(ent.key, idx)
                row = {
                    "task_family": TASK_FAMILY,
                    "passage_id": pid,
                    "passage_type": ent.entity_type,
                    "passage": passage,
                    "passage_token_ids": token_ids,
                    "passage_token_count": len(token_ids),
                    # Biographical-specific extras (flat at top level).
                    "entity_key": ent.key,
                    "sample_idx": idx,
                    "surface_names": list(ent.surface_names),
                    "attrs": dict(ent.attrs),
                    "outgoing_edges": [
                        {"rel": e.rel, "dst": e.dst}
                        for e in world.get_edges(ent.key)
                    ],
                    "passage_persona": persona,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def emit_questions_jsonl(
    questions: list[dict],
    output_path: Path,
    tokenizer,
) -> int:
    """Emit questions, translating evidence_keys from entity_keys to passage_ids.

    Each question's evidence_keys currently lists entity_keys (as emitted
    by the biographical question generators). We translate each to the
    sample_idx=0 passage_id for the unified schema. The original
    entity_keys are preserved as `evidence_entity_keys` for debugging.

    Returns count written.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w") as f:
        for q in questions:
            q_ids = _tokenize(q["question"], tokenizer)
            a_ids = _tokenize(q["answer"], tokenizer)
            translated_evidence = [
                passage_id_for(ek, sample_idx=0) for ek in q["evidence_keys"]
            ]
            row = dict(q)
            row["task_family"] = TASK_FAMILY
            row["evidence_keys"] = translated_evidence
            row["evidence_entity_keys"] = list(q["evidence_keys"])
            row["question_token_ids"] = q_ids
            row["answer_token_ids"] = a_ids
            row["question_token_count"] = len(q_ids)
            row["answer_token_count"] = len(a_ids)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def _tokenize(text: str, tokenizer) -> list[int]:
    """Encode `text` with the Llama tokenizer, dropping BOS (matches v4)."""
    return tokenizer(text, add_special_tokens=False)["input_ids"]
