"""Composite sampler — bundles passages+question for one training step.

Loads a composite passages.jsonl + questions.jsonl from disk.
Per training step:
1. Picks a question by weighted task-family sampling
2. Loads its evidence passages
3. Pads chunk to chunk_size with distractor passages from the SAME family
4. Returns chunk dict ready for the trainer

Design choices documented in the composite Wave 1 plan:
- Mixing task families inside one chunk is awkward (model would have to
  figure out which task is being asked from heterogeneous passages).
  So each chunk is one family, but batches/steps rotate.
- Distractors come from the same family as the target.

This is the sampler the trainer's `step()` calls.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


class CompositeSampler:
    """Loads a composite dataset and samples chunks for the trainer."""

    def __init__(
        self,
        passages_path: str | Path,
        questions_path: str | Path,
        *,
        task_weights: dict[str, float] | None = None,
        seed: int = 0,
    ):
        passages_path = Path(passages_path)
        questions_path = Path(questions_path)
        self.passages: dict[str, dict] = {}
        self.passages_by_family: dict[str, list[str]] = defaultdict(list)
        for line in passages_path.read_text().splitlines():
            if not line.strip():
                continue
            p = json.loads(line)
            self.passages[p["passage_id"]] = p
            self.passages_by_family[p["task_family"]].append(p["passage_id"])

        self.questions_by_family: dict[str, list[dict]] = defaultdict(list)
        for line in questions_path.read_text().splitlines():
            if not line.strip():
                continue
            q = json.loads(line)
            self.questions_by_family[q["task_family"]].append(q)

        families_with_q = [
            f for f, qs in self.questions_by_family.items() if qs
        ]
        if not families_with_q:
            raise ValueError("no questions loaded")

        # Default to uniform weights over families present in the questions.
        if task_weights is None:
            self.task_weights = {f: 1.0 for f in families_with_q}
        else:
            # Filter to families that actually have questions; ignore unknowns.
            self.task_weights = {
                f: w for f, w in task_weights.items()
                if f in families_with_q and w > 0
            }
            if not self.task_weights:
                raise ValueError(
                    f"None of task_weights families {list(task_weights)} "
                    f"have loaded questions {families_with_q}"
                )

        self.rng = random.Random(seed)
        self._families = list(self.task_weights.keys())
        self._weights = list(self.task_weights.values())

    def n_questions(self, family: str | None = None) -> int:
        if family is None:
            return sum(len(qs) for qs in self.questions_by_family.values())
        return len(self.questions_by_family.get(family, []))

    def sample_chunk(self, chunk_size: int) -> dict[str, Any]:
        """Sample one training chunk.

        Returns:
            {
              'task_family':    str,
              'target_question': dict (the question row),
              'passages':       list[dict] (length chunk_size),
              'evidence_idxs':  list[int]  (positions of evidence in passages),
              'target_idx':     int        (position of the first evidence
                                            passage — for compat with trainers
                                            that expect a single 'target_idx')
            }
        """
        # 1. Pick task family by weight.
        family = self.rng.choices(self._families, weights=self._weights, k=1)[0]

        # 2. Pick a question uniformly from that family.
        target_q = self.rng.choice(self.questions_by_family[family])
        evidence_keys = list(target_q["evidence_keys"])
        if len(evidence_keys) > chunk_size:
            # Should not happen if generators are well-behaved; truncate.
            evidence_keys = evidence_keys[:chunk_size]

        # 3. Pick distractors from the same family, disjoint from evidence.
        family_pool = self.passages_by_family[family]
        distractor_count = chunk_size - len(evidence_keys)
        ev_set = set(evidence_keys)
        candidates = [pid for pid in family_pool if pid not in ev_set]
        if len(candidates) < distractor_count:
            # Family doesn't have enough distractors; pad by repetition.
            distractors = self.rng.choices(candidates, k=distractor_count) if candidates else []
        else:
            distractors = self.rng.sample(candidates, distractor_count)

        # 4. Shuffle evidence + distractors together so the model doesn't
        #    learn that target is always at a fixed position.
        all_pids = evidence_keys + distractors
        self.rng.shuffle(all_pids)
        passages = [self.passages[pid] for pid in all_pids]

        # 5. Record where the evidence ended up.
        pid_to_pos = {pid: i for i, pid in enumerate(all_pids)}
        evidence_idxs = [pid_to_pos[pid] for pid in evidence_keys]
        target_idx = evidence_idxs[0]   # first evidence — for v4 compat

        return {
            "task_family": family,
            "target_question": target_q,
            "passages": passages,
            "evidence_idxs": evidence_idxs,
            "target_idx": target_idx,
            "chunk_size": chunk_size,
        }

    def sample_batch(self, m: int, chunk_size: int) -> list[dict]:
        """Convenience: sample m chunks at the same chunk_size."""
        return [self.sample_chunk(chunk_size) for _ in range(m)]


class CompositeRetrievalAdapter:
    """Wraps CompositeSampler to emit chunks in the legacy Phase1RetrievalTrainer
    format. Pinning chunk_size=8 (matches TBPTT D=9 for 8 writes + 1 read).

    Drop-in replacement for `RetrievalSampler`: exposes `sample_batch(M)`
    returning the dict shape the trainer's `step(batch)` expects.

    Per-(task_family, question_type) telemetry comes for free because the
    trainer's per_key_loss/acc indexes on `metadata.{target_entity_class,
    target_attribute}` — we map those to (task_family, question_type).
    """

    def __init__(
        self,
        passages_path: str | Path,
        questions_path: str | Path,
        *,
        chunk_size: int = 8,
        task_weights: dict[str, float] | None = None,
        seed: int = 0,
    ):
        self.inner = CompositeSampler(
            passages_path, questions_path,
            task_weights=task_weights, seed=seed,
        )
        self.chunk_size = chunk_size

    @property
    def facts(self) -> list:
        """Legacy compat for code that prints `len(sampler.facts)`."""
        return [q for qs in self.inner.questions_by_family.values() for q in qs]

    @property
    def keys(self) -> list:
        """Legacy compat: distinct (task_family, question_type) pairs."""
        out = set()
        for fam, qs in self.inner.questions_by_family.items():
            for q in qs:
                out.add((fam, q["question_type"]))
        return sorted(out)

    def sample_chunk(self) -> dict:
        c = self.inner.sample_chunk(self.chunk_size)
        target_q = c["target_question"]
        passages = c["passages"]
        return {
            "fact_passages_token_ids": [p["passage_token_ids"] for p in passages],
            "question_token_ids": target_q["question_token_ids"],
            "answer_token_ids": target_q["answer_token_ids"],
            # answer_content_token_positions: list[int], offsets within
            # answer_token_ids that carry memory-load-bearing content (target
            # value). When missing or empty, trainer falls back to marking
            # the full answer span.
            "answer_content_token_positions": list(
                target_q.get("answer_content_token_positions", []),
            ),
            "target_idx": c["target_idx"],
            "target_fact_id": target_q["question_id"],
            "metadata": {
                "target_entity_class": c["task_family"],
                "target_attribute": target_q["question_type"],
            },
        }

    def sample_batch(self, batch_size: int) -> list[dict]:
        return [self.sample_chunk() for _ in range(batch_size)]
