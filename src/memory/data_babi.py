"""bAbI conditioned-reconstruction objective — story → question → 1-word answer.

bAbI (Weston et al., 2015) is the canonical synthetic memory benchmark: a short
STORY of declarative facts, a QUESTION about the world state it induces, and a
single-word ANSWER. The memory-focused subset (default) is tasks 1/2/3 (one/two/
three supporting facts), 7 (counting), 8 (lists/sets), 11/12/13 (coreference) and
14 (time) — the families that genuinely require holding & composing facts rather
than surface pattern-matching.

This mirrors `data_conditioned_reconstruction` exactly: the encoder ingests the
STORY into memory, the story tokens are dropped, and the frozen LM — conditioned
ONLY on the memory + the question — must reproduce the answer (closed-book). The
question is NOT in the context (the story is question-agnostic), so the answer can
only come from the compressed memory. Loss is CE on every answer token.

Emits the exact per-sample dict that `data_qa.collate_qa` consumes, so the whole
`ReprLearningModel.compute_loss` path (encoder → memory → prepend → frozen-LM CE on
`answer_content_mask`) and the REAL/SHUF/OFF binding gate are reused unchanged —
only the *data* differs.

DISTRACTOR PADDING: a single bAbI story is short (≈8 facts, ~60 tok), far below the
other objectives' context length. We pad each story up to ~context_len tokens by
appending irrelevant bAbI sentences drawn from OTHER stories, keeping the real
supporting facts intact at the front. This (a) matches the input length of the
other objectives and (b) turns every example into a retrieve-among-noise read.

  python -m src.memory.data_babi        # smoke: render a few examples, check ctx len
"""
from __future__ import annotations

import random
import re
from typing import List, Optional

import torch
from torch.utils.data import DataLoader, IterableDataset

from .data_qa import collate_qa

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_sents(text: str) -> List[str]:
    """Split a bAbI passage into individual fact SENTENCES. Handles both the HF
    `Muennighoff/babi` form (all facts on ONE line, joined by ". ") and the
    newline-separated programmatic fallback — flatten newlines to spaces, then
    split after sentence-final punctuation. Without this, `str.splitlines()`
    returns a single element for HF passages, so the 'distractor pool' would be
    whole multi-fact stories, not per-sentence noise."""
    flat = text.replace("\n", " ").strip()
    return [s.strip() for s in _SENT_SPLIT.split(flat) if s.strip()]


def _caps_names(text: str) -> set:
    """bAbI named entities = capitalized alphabetic tokens (Mary/John/Sandra…), minus the only
    common sentence-initial non-name ('The'). Used to keep distractor entities DISJOINT from the
    gold story: bAbI reuses a tiny global name pool, so a same-name distractor pulled from another
    story silently contradicts/updates the queried entity's state while the label stays from the
    gold story (ill-posed). All default tasks are person-entity, so this filter fully de-ambiguates."""
    out = set()
    for w in text.replace("\n", " ").split():
        w = w.strip(".,!?;:'\"")
        if w and w[0].isupper() and w.isalpha() and w != "The":
            out.add(w)
    return out

# Memory-focused task subset (default). 1/2/3 = one/two/three supporting facts,
# 7 = counting, 8 = lists/sets, 11/12/13 = (indefinite/conjunctive/compound)
# coreference, 14 = time reasoning.
DEFAULT_TASKS = (1, 2, 3, 7, 8, 11, 12, 13, 14)

# Programmatic fallback world (used only when HF bAbI is unreachable). Kept small;
# generates task-1-style single-supporting-fact stories which exercise the same
# story→question→answer contract as the real data.
_FALLBACK_NAMES = ["Mary", "John", "Daniel", "Sandra", "Fred", "Julie", "Bill", "Jeff"]
_FALLBACK_PLACES = ["bathroom", "hallway", "kitchen", "garden", "bedroom",
                    "office", "school", "park", "cinema", "kitchen"]
_FALLBACK_MOVES = ["moved to", "went to", "journeyed to", "travelled to", "went back to"]


def _load_babi_rows(tasks, split: str):
    """Return list[(story_str, question_str, answer_str, task_int)] for the given
    tasks/split. Tries HuggingFace bAbI sources, then falls back to programmatic
    generation if offline. The story field is QUESTION-AGNOSTIC by construction."""
    task_set = set(tasks)

    # facebook/babi_qa ships per-task configs but as a (now-unsupported) dataset
    # script; Muennighoff/babi ships a plain parquet with a `task` column and a
    # ready-made `passage`/`question`/`answer` schema — try the parquet first.
    for name in ("Muennighoff/babi",):
        try:
            from datasets import load_dataset
            hf_split = "validation" if split in ("validation", "val") else "train"
            ds = load_dataset(name, split=hf_split)
            rows = []
            for ex in ds:
                t = int(ex["task"])
                if t not in task_set:
                    continue
                story = (ex["passage"] or "").strip()
                q = (ex["question"] or "").strip()
                a = (ex["answer"] or "").strip()
                if story and q and a:
                    rows.append((story, q, a, t))
            if rows:
                print(f"[babi] loaded {len(rows):,} rows from {name} "
                      f"(split={hf_split}, tasks={sorted(task_set)})", flush=True)
                return rows
        except Exception as e:  # pragma: no cover — network/offline path
            print(f"[babi] {name} unavailable ({type(e).__name__}: {str(e)[:80]}); "
                  f"trying next source", flush=True)

    # Offline fallback: synthesize task-1 single-supporting-fact stories. Seed by SPLIT so the
    # train/val fallback streams are disjoint — a fixed 1234 for both made offline val a verbatim
    # copy of train (silent leakage when HF is unreachable).
    is_val = split in ("validation", "val")
    print(f"[babi] HF bAbI unreachable — generating programmatic task-1 stories "
          f"(offline fallback, split={'val' if is_val else 'train'})", flush=True)
    gen = random.Random(5678 if is_val else 1234)
    rows = []
    for _ in range(4000):
        n_facts = gen.randint(2, 8)
        loc = {}
        lines = []
        for _ in range(n_facts):
            who = gen.choice(_FALLBACK_NAMES)
            where = gen.choice(_FALLBACK_PLACES)
            loc[who] = where
            lines.append(f"{who} {gen.choice(_FALLBACK_MOVES)} the {where}.")
        who_q = gen.choice(list(loc.keys()))
        rows.append(("\n".join(lines) + "\n", f"Where is {who_q}?", loc[who_q], 1))
    return rows


class BabiDataset(IterableDataset):
    """Infinite stream of bAbI story→question→answer examples, distractor-padded
    to ~context_len tokens.

    Per example: pick a real (story, question, answer); tokenize the story into
    context_ids; append irrelevant bAbI sentences (from other stories) as
    distractors until the context is filled to context_len; truncate if a story
    already exceeds it (keeping the leading supporting facts).
    """

    def __init__(self, tokenizer, context_len: int, split: str = "train",
                 tasks=DEFAULT_TASKS, seed: int = 0, n_items: int = 1_000_000,
                 pad_token_id: int = 128_001):
        super().__init__()
        self.tok = tokenizer
        self.context_len = context_len
        self.split = split
        self.tasks = tuple(tasks)
        self.seed = seed
        self.n_items = n_items
        self.pad_token_id = pad_token_id

        self.rows = _load_babi_rows(self.tasks, split)
        if not self.rows:
            raise ValueError(f"bAbI: no rows for tasks={self.tasks} split={split}")
        # Pre-split every story into its individual fact sentences once — these are
        # the distractor pool (irrelevant sentences appended to pad the context).
        self._distractor_sents: List[str] = []
        for story, _q, _a, _t in self.rows:
            self._distractor_sents.extend(_split_sents(story))

    def _ids(self, s: str) -> List[int]:
        return self.tok(s, add_special_tokens=False).input_ids

    def _gen(self, rng: random.Random) -> dict:
        story, question, answer, task = self.rows[rng.randrange(len(self.rows))]

        # Real supporting facts go in first (front of context), guaranteed intact.
        # Split into individual fact sentences (one fact per line), so the encoder
        # ingests structured facts rather than one undifferentiated block.
        story_sents = _split_sents(story)
        ctx_ids: List[int] = []
        for sent in story_sents:
            ctx_ids += self._ids(sent + "\n")
        # Over-long real story: keep the TAIL. bAbI's answer-relevant fact is usually
        # the MOST-RECENT mention (end of story), so head-truncation could drop the
        # supporting fact and make the example unanswerable.
        if len(ctx_ids) > self.context_len:
            ctx_ids = ctx_ids[-self.context_len:]

        # Distractor padding: append irrelevant bAbI sentences until ~context_len.
        # Tests retrieval-among-noise while keeping the supporting facts present.
        # REJECT distractors that share a named entity with the gold story (see _caps_names) —
        # otherwise the noise can contradict the queried entity's labelled state (ill-posed example).
        gold_names = _caps_names(story)
        guard = 0
        while len(ctx_ids) < self.context_len and guard < 8 * self.context_len:
            guard += 1
            d = rng.choice(self._distractor_sents)
            if _caps_names(d) & gold_names:        # shares a gold entity → would contaminate the label
                continue
            d_ids = self._ids(d + "\n")
            if len(ctx_ids) + len(d_ids) > self.context_len:
                # Append a final truncated distractor to fill exactly, then stop.
                room = self.context_len - len(ctx_ids)
                if room > 0:
                    ctx_ids += d_ids[:room]
                break
            ctx_ids += d_ids

        valid = len(ctx_ids)
        if valid < self.context_len:
            ctx_ids = ctx_ids + [self.pad_token_id] * (self.context_len - valid)

        question_ids = self._ids(question)
        # List tasks (e.g. task 8) carry comma-joined answers; tokenized whole.
        # Every answer token is load-bearing (all-True content mask).
        answer_ids = self._ids(answer)
        content = [True] * len(answer_ids)

        return {
            "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
            "context_mask": torch.tensor([True] * valid + [False] * (self.context_len - valid),
                                         dtype=torch.bool),
            "question_ids": torch.tensor(question_ids, dtype=torch.long),
            "answer_ids": torch.tensor(answer_ids, dtype=torch.long),
            "answer_content_mask_list": content,
            "task_family": "babi",
            "question_type": f"task{task}",
            "answer_refs": [answer],
        }

    def __iter__(self):
        wi = torch.utils.data.get_worker_info()
        rng = random.Random(self.seed + (wi.id if wi is not None else 0))
        for _ in range(self.n_items):
            yield self._gen(rng)


def make_babi_dataloader(tokenizer, context_len: int, batch_size: int,
                         split: str = "train", seed: int = 0,
                         pad_token_id: int = None, num_workers: int = 2,
                         tasks=DEFAULT_TASKS) -> DataLoader:
    if pad_token_id is None:                                  # LLM-agnostic default
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    ds = BabiDataset(tokenizer, context_len=context_len, split=split, tasks=tasks,
                     seed=seed, pad_token_id=pad_token_id)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id))


if __name__ == "__main__":  # smoke: render a few bAbI examples, confirm ctx len
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from src.memory.config import ReprConfig

    CTX = 256
    tok = AutoTokenizer.from_pretrained(ReprConfig().llama_model)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ds = BabiDataset(tok, context_len=CTX, split="train", seed=1, pad_token_id=pad)
    print(f"\n|distractor pool| = {len(ds._distractor_sents):,} sentences; "
          f"|rows| = {len(ds.rows):,}; context_len = {CTX}\n")
    it = iter(ds)
    for _ in range(4):
        s = next(it)
        valid = int(s["context_mask"].sum())
        story = tok.decode(s["context_ids"][s["context_mask"]])
        # Heuristic: show only the first few lines (the real supporting facts live
        # at the front; the rest are distractors).
        head = "\n".join(story.splitlines()[:6])
        print(f"===== {s['question_type']} | ctx valid={valid}/{CTX} tok "
              f"({len(s['context_ids'])} padded) =====")
        print("STORY[head]:\n" + head)
        print("...(+distractors)..." if valid > 0 else "")
        print("QUESTION :", repr(tok.decode(s["question_ids"])))
        print("ANSWER   :", repr(tok.decode(s["answer_ids"])), "| refs:", s["answer_refs"])
        print()
    print(f"context length == context_len? "
          f"{all(int(next(iter(ds))['context_ids'].numel()) == CTX for _ in range(3))}")
