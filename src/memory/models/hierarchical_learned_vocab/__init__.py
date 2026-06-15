"""hlvocab — Compression-by-Vocabulary model (encoder + substrate).

ABANDONED 2026-06-15 (was graph_v9). Kept loadable for reproducing prior results;
not in the active suite. Triple-confirmed rank-1 read collapse: even with healthy
routing (load-balance, 386/512 nodes) and rich multi-scale content (Llama taps),
the emitted memory pooled to eff_rank ~1 — the read is the wall. Superseded by the
VQ-VAE→graph+TokenGT model. See memory: project_mae_4k_collapse_result.
"""
from .encoder import HLVocabEncoder
from .substrate import HLVocabConfig, HLVocabSubstrate

__all__ = ["HLVocabEncoder", "HLVocabConfig", "HLVocabSubstrate"]
