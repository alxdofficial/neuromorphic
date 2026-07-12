"""Schedule — the difficulty knobs (EpisodeSpec) + curriculum ratcheting.

An ``EpisodeSpec`` is the single, modular config for *how hard* an episode is: how many inputs,
how long each, how many distractors, how many reads, and how far back the queried write sits. The
SAME spec drives every task (reconstruction / qa / continuation / mae), so difficulty means the
same thing across tasks. A ``Curriculum`` maps training step → spec, ratcheting length/lag/distractors.

See ``docs/DATA.md`` (Layer L3). Consumed by ``tasks/`` (shapers) and ``mixes.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional


@dataclass(frozen=True)
class EpisodeSpec:
    """How to shape one training episode from a source's items. Every field is a difficulty knob."""
    source: str                     # SOURCE_REGISTRY key (where items come from)
    task: str                       # TASK_STYLES key (how they're presented/asked)
    total_len: int = 1024           # target total context tokens (the compression numerator)
    window_size: Optional[int] = None   # streaming write granularity; None ⇒ single window (= total_len)
    n_inputs: int = 24              # MAX items packed into the context (fill-to-budget uses ≤ this)
    input_len: Optional[int] = None      # tokens per item, when a task needs a fixed per-item length
    n_distractors: int = 0          # filler items/interference between a write and its query
    n_queries: int = 1              # reads per episode (>1 forces addressing)
    query_lag: str = "any"          # "recent" | "early" | "any" | "vary" (sampled per-episode) — where the queried write sits
    predict_len: int = 64           # continuation: tokens to predict after the compressed prefix
    mask_ratio: Optional[float] = None   # mae: fraction of the span masked in the infill forward (None ⇒ cfg default)
    n_horizons: Optional[int] = None     # continuation: predict blocks at the first N streaming-window boundaries
                                         # (None ⇒ every boundary when window_size < total_len; 1 ⇒ single-shot)

    def with_(self, **kw) -> "EpisodeSpec":
        """Return a copy with fields overridden (curriculum stages build specs this way)."""
        return replace(self, **kw)

    @property
    def effective_window(self) -> int:
        """Streaming window size, defaulting to a single window over the whole context."""
        return self.window_size if self.window_size else self.total_len


@dataclass
class Curriculum:
    """Step → EpisodeSpec schedule. Ratchets difficulty (length / lag / distractors) over training.

    ``stages`` is a list of (until_step, spec) checkpoints, ascending by until_step; ``spec_at``
    returns the first stage whose until_step exceeds the current step (the last stage's spec holds
    for all later steps). A single-stage curriculum is just a constant spec.
    """
    stages: list[tuple[int, EpisodeSpec]]

    def __post_init__(self):
        if not self.stages:
            raise ValueError("Curriculum needs at least one (until_step, EpisodeSpec) stage")
        # ascending by until_step so spec_at can scan in order
        self.stages = sorted(self.stages, key=lambda s: s[0])

    def spec_at(self, step: int) -> EpisodeSpec:
        for until, spec in self.stages:
            if step < until:
                return spec
        return self.stages[-1][1]

    @classmethod
    def constant(cls, spec: EpisodeSpec) -> "Curriculum":
        """A non-ratcheting curriculum: one spec for the whole run."""
        return cls([(1 << 62, spec)])
