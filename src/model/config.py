"""
Model configuration for the Neuromorphic LM.

Single dataclass holding all hyperparameters. Phase toggles control which
memory systems are active. Tier presets (A/B/C) provide size configurations
for single-GPU training on RTX 4090.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Architecture
    D: int = 512              # model width
    L: int = 8                # layers per block
    B: int = 4                # parallel blocks
    vocab_size: int = 32000   # set from tokenizer at runtime
    eot_id: int = 2           # set from tokenizer at runtime

    # Working Memory (1 shared instance)
    W: int = 256              # sliding window size
    D_wm: int = 128           # WM key/value dimension
    n_heads_wm: int = 4       # attention heads

    # Procedural Memory (per layer per block, B*L instances)
    r: int = 8                # PM slots
    rho: float = 0.95         # eligibility decay
    a_max: float = 3.0        # max strength per slot
    budget_pm: float = 4.0    # sum(pm_a) budget per stream
    decay_pm: float = 0.999   # per-span strength decay
    commit_top_k: int = 2     # slots updated per commit
    tau_pm: float = 1.0       # softmax temperature
    weakness_weight_pm: float = 0.5  # bias toward weak slots

    # Episodic Memory (per block, B instances)
    M: int = 256              # EM capacity per bank
    D_em: int = 128           # EM key/value dimension
    k_ret: int = 4            # retrieval count
    C_em: int = 8             # candidates per span
    k_write: int = 4          # slots updated per candidate
    tau_em: float = 1.0       # softmax temperature
    weakness_weight_em: float = 0.5  # bias toward weak slots
    S_max: float = 3.0        # max strength per slot
    budget_em: float = 8.0    # sum(em_S) budget per stream
    decay_em: float = 0.999   # per-span strength decay

    # Training
    T: int = 256              # TBPTT segment length
    P: int = 32               # plasticity span
    reset_on_doc_boundary: bool = True

    # Phase toggles
    wm_enabled: bool = True   # always on
    pm_enabled: bool = False  # Phase B+
    em_enabled: bool = False  # Phase C+

    @property
    def D_h(self) -> int:
        """Per-block hidden dimension."""
        return self.D // self.B

    def set_phase(self, phase: str):
        """Set component toggles for training phase.

        A: WM only
        B: WM + PM
        C: WM + PM + EM
        D: WM + PM + EM (+ RL controllers, future)
        """
        phase = phase.upper()
        if phase == "A":
            self.wm_enabled = True
            self.pm_enabled = False
            self.em_enabled = False
        elif phase == "B":
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = False
        elif phase in ("C", "D", "E"):
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = True
        else:
            raise ValueError(f"Unknown phase: {phase}. Expected A/B/C/D/E.")

    @classmethod
    def tier_a(cls, **overrides) -> "ModelConfig":
        """Debug tier (~50M params). D=512, L=8, B=4."""
        defaults = dict(D=512, L=8, B=4)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "ModelConfig":
        """Competitive tier (~150M params). D=768, L=12, B=6."""
        defaults = dict(D=768, L=12, B=6)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "ModelConfig":
        """Strong tier (~350M params). D=1024, L=24, B=8."""
        defaults = dict(D=1024, L=24, B=8)
        defaults.update(overrides)
        return cls(**defaults)
