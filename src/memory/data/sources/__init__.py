"""Source registry — name → Source factory. Sources yield raw items; Tasks shape them.

Lazy so importing the package never pulls heavy deps (datasets/pyarrow) until a source is built.
See ``docs/history/docs/history/data_arch_plan.md`` (Layer L1).
"""
from __future__ import annotations

import importlib
from typing import Callable

from .base import Source, CorpusItem, KeyedItem, QAItem

# name → (submodule, class-name)
_SOURCES: dict[str, tuple[str, str]] = {
    "babi": ("babi", "BabiSource"),
    "bio": ("bio", "BioSource"),
    "fineweb": ("fineweb", "FinewebSource"),
    # procedural keyed sources (runtime, no download):
    "mqar": ("mqar", "MqarSource"),
    "ruler_overwrite": ("ruler_overwrite", "RulerOverwriteSource"),
    # real QA sources (yield QAItem for the qa task; bounded HF samples, see ingest/):
    "squad": ("squad", "SquadSource"),                    # extractive RC (single-span)
    "triviaqa": ("triviaqa", "TriviaQASource"),           # factoid + evidence
    "hotpot_train": ("hotpot_train", "HotpotTrainSource"),   # 2-hop (train split, firewalled from eval)
    "musique_train": ("musique_train", "MusiqueTrainSource"),  # 2-4 hop, shortcut-reduced
    "multiwoz": ("multiwoz", "MultiWOZSource"),           # dialogue slot-recall (real, non-synthetic)
    "quality": ("quality", "QualitySource"),              # long-document comprehension (needs total_len>=4096)
    "qa_multi": ("qa_multi", "QaMultiSource"),            # QA VARIETY (squad+triviaqa+hotpot+musique+multiwoz)
    # corpus sources (bucket-1 natural text; best-effort HF sample, see ingest/):
    "pile": ("pile", "PileSource"),
    "redpajama": ("redpajama", "RedpajamaSource"),
    "code": ("code", "CodeSource"),                       # source code — un-guessable exact-recall binding
    "multicorpus": ("multicorpus", "MultiCorpusSource"),  # continuation/mae VARIETY (fineweb+pile+redpajama+code)
    # ── PHASE-1 (full-corpus training) sources — long-horizon / streaming / agentic ──────────────
    # long real conversations (recall across turns/sessions):
    "wildchat": ("wildchat", "WildChatSource"),           # real user↔assistant multi-turn (allenai/WildChat-1M)
    "lmsys_chat": ("lmsys_chat", "LmsysChatSource"),      # real convos (lmsys/lmsys-chat-1m — GATED, needs HF access)
    "msc": ("msc", "MSCSource"),                          # Multi-Session Chat — cross-session persona recall
    # long documents (genuine long-range dependency):
    "qasper": ("qasper", "QasperSource"),                 # NLP-paper QA (non-gist-gameable; span/yes-no/unanswerable)
    "longcite": ("longcite", "LongCiteSource"),           # 128k-word contexts + citation-span reward
    "govreport": ("govreport", "GovReportSource"),        # long gov reports (corpus)
    "pg19": ("pg19", "PG19Source"),                       # full-length BOOKS (~1M+ tok — the long-horizon corpus)
    # procedural long-context generators (unlimited, exact-match verifiable):
    "ruler_niah": ("ruler_niah", "RulerNiahSource"),      # NIAH/multikey/vartrack generator (overwrite-fork base)
    "babilong_train": ("babilong_train", "BabilongTrainSource"),  # bAbI facts in PG-19 haystack (TRAIN, ≠ eval babilong)
    # streaming-update + agentic:
    "wikibigedit": ("wikibigedit", "WikiBigEditSource"),  # sequential fact-update (forced-forgetting gate)
    "swe_trajectories": ("swe_trajectories", "SweTrajectoriesSource"),  # long agent trajectories (nebius, reward-labeled)
    "perltqa": ("perltqa", "PerLTQASource"),              # personalization long-term QA (memory-anchor spans)
}


def _lazy(module_name: str, cls_name: str) -> Callable:
    def _factory(*args, **kwargs):
        mod = importlib.import_module(f"{__name__}.{module_name}")
        return getattr(mod, cls_name)(*args, **kwargs)
    _factory.__name__ = cls_name
    return _factory


SOURCE_REGISTRY: dict[str, Callable] = {
    name: _lazy(mod, cls) for name, (mod, cls) in _SOURCES.items()
}

__all__ = ["SOURCE_REGISTRY", "Source", "CorpusItem", "KeyedItem", "QAItem"]
