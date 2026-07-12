"""Multi-dataset QA source — QA VARIETY over several real reading-comprehension datasets.

The QA analog of ``multicorpus``: draws each item from a randomly chosen QA sub-source so one `qa`
task sees varied binding shapes — single-span (SQuAD), factoid+evidence (TriviaQA), multi-hop
(HotpotQA, MuSiQue), and dialogue slot-recall (MultiWOZ) — instead of babi's one weak flavor.
ROBUST: a sub-source that can't load (unreachable / not ingested) is skipped with a warning.

QuALITY is intentionally EXCLUDED from the default set — it's long-document (stories >4k tok) and
needs `total_len>=4096`; mixing it at the default 1024 would tail-truncate the story away. Use it
as a standalone long-context qa source. See DATASETS.md / docs/DATA.md.
"""
from __future__ import annotations

from .base import Source

# default variety pool — all work at total_len=1024 (answer + gold facts in the front ~900 tok).
DEFAULT_QA = ("squad", "triviaqa", "hotpot_train", "musique_train", "multiwoz")


def _build_one(name, tokenizer, *, split, seed, n_docs):
    if name == "squad":
        from .squad import SquadSource
        return SquadSource(tokenizer, split=split, seed=seed, n_docs=n_docs)
    if name == "triviaqa":
        from .triviaqa import TriviaQASource
        return TriviaQASource(tokenizer, split=split, seed=seed, n_docs=n_docs)
    if name == "hotpot_train":
        from .hotpot_train import HotpotTrainSource
        return HotpotTrainSource(tokenizer, split=split, seed=seed, n_docs=n_docs)
    if name == "musique_train":
        from .musique_train import MusiqueTrainSource
        return MusiqueTrainSource(tokenizer, split=split, seed=seed, n_docs=n_docs)
    if name == "multiwoz":
        from .multiwoz import MultiWOZSource
        return MultiWOZSource(tokenizer, split=split, seed=seed, n_docs=n_docs)
    # NB: bAbI is intentionally NOT unioned here — it needs per-segment entity renaming (pack_rename),
    # which the qa task only applies to the top-level source. Co-packing raw bAbI items would collide.
    raise ValueError(f"qa_multi: unknown qa source {name!r} (have {DEFAULT_QA})")


class QaMultiSource(Source):
    """Draws each QAItem from a randomly chosen QA sub-source; unions their distractor pools."""

    kind = "qa"
    pack_n_queries = (1, 2)            # big RC contexts (hotpot/musique ~900 tok) → only ~2 golds fit at 2048

    def __init__(self, tokenizer, *, split: str = "train", seed: int = 0,
                 datasets=DEFAULT_QA, n_docs: int = 2500, allow_missing: bool = False, **kw):
        self.subs = []
        loaded, skipped = [], []
        for i, name in enumerate(datasets):
            try:
                self.subs.append(_build_one(name, tokenizer, split=split, seed=seed + i, n_docs=n_docs))
                loaded.append(name)
            except Exception as e:                       # unreachable / not-ingested / corrupt / short
                skipped.append(f"{name} ({type(e).__name__}: {e})")
        # FAIL LOUD by default: silently dropping a requested source changes the training distribution
        # without failing the run (audit finding #6). A dev env intentionally running a subset passes
        # allow_missing=True; the campaign must load every requested source (bootstrap asserts the dirs).
        if skipped and not allow_missing:
            raise RuntimeError(
                f"[qa_multi] {len(skipped)}/{len(datasets)} QA sources FAILED to load — refusing to train "
                f"a silently-degraded mix: {skipped}. Fix the source(s) or pass allow_missing=True.")
        if not self.subs:
            raise ValueError(f"[qa_multi] no QA source loaded from {list(datasets)}. Skipped: {skipped}")
        self._pool = []
        for s in self.subs:
            self._pool.extend(s.distractor_pool())
        print(f"[data.qa_multi] {split}: {len(self.subs)} QA sources [{', '.join(loaded)}]"
              + (f"  (skipped: {', '.join(skipped)})" if skipped else ""), flush=True)

    def sample(self, rng, n: int) -> list:
        # each item from a uniformly-chosen sub-source (a random dataset per example)
        return [self.subs[rng.randrange(len(self.subs))].sample(rng, 1)[0] for _ in range(n)]

    def distractor_pool(self) -> list:
        return self._pool
