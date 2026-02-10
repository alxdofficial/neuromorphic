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
    ffn_expansion: int = 4    # FFN expansion factor per layer (0 = no FFN)
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
    pm_readout_ffn: bool = True       # MLP after PM linear lookup

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
    g_em_floor: float = 0.001  # minimum write strength (learned mode, near-zero = soft "don't write")
    g_em_ceil: float = 0.95    # maximum write strength (learned mode)
    tau_em_floor: float = 0.05   # min soft top-k temperature (learned mode)
    tau_em_ceil: float = 5.0     # max soft top-k temperature (learned mode)
    ww_em_floor: float = 0.0    # min weakness weight (learned mode)
    ww_em_ceil: float = 2.0     # max weakness weight (learned mode)
    em_readout_ffn: bool = True       # MLP after EM cross-attention retrieval

    # Training
    T: int = 256              # TBPTT segment length
    P: int = 32               # plasticity span
    reset_on_doc_boundary: bool = True
    lifelong_mode: bool = False  # Phase E: PM/EM persist across doc boundaries

    # RL controllers (Phase D)
    rl_enabled: bool = False
    rl_controller_hidden: int = 32
    rl_lr: float = 1e-3
    rl_events_per_chunk: int = 2    # rollout events per T-token chunk
    rl_memory_penalty: float = 0.0  # penalty per commit/write (future use)
    rl_warmup_steps: int = 500      # LR warmup for RL optimizer after phase transition
    rl_pm_targets_per_event: int = 1  # PM controllers counterfactually trained per event
    rl_em_targets_per_event: int = 1  # EM controllers counterfactually trained per event

    # Spatial Decoder (hierarchical aggregation + deep cross-attention)
    snapshot_enabled: bool = True   # architecture toggle (independent of phase)
    d_dec: int = 256               # decoder working dimension
    n_heads_decoder: int = 4       # attention heads in columnar/thalamic/decoder
    decoder_layers: int = 2        # deep decoder depth (Level 3)
    columnar_layers: int = 2       # columnar attention depth (Level 1)
    thalamic_layers: int = 2       # thalamic integrator depth (Level 2)
    thalamic_tokens: int = 4       # output tokens from thalamic integrator

    # Phase toggles
    wm_enabled: bool = True   # always on
    pm_enabled: bool = False  # Phase B+
    em_enabled: bool = False  # Phase C+

    @property
    def D_h(self) -> int:
        """Per-block hidden dimension."""
        return self.D // self.B

    def validate(self):
        """Check config validity. Call before model construction.

        Raises ValueError with clear messages for invalid combinations
        that would otherwise cause opaque crashes deep in PyTorch.
        """
        if self.D % self.B != 0:
            raise ValueError(
                f"D ({self.D}) must be divisible by B ({self.B}) "
                f"to compute D_h = D/B."
            )
        if self.snapshot_enabled:
            if self.d_dec % self.n_heads_decoder != 0:
                raise ValueError(
                    f"d_dec ({self.d_dec}) must be divisible by "
                    f"n_heads_decoder ({self.n_heads_decoder}) for "
                    f"MultiheadAttention in the spatial decoder."
                )
            D_h = self.D // self.B
            if D_h < self.n_heads_decoder:
                raise ValueError(
                    f"D_h ({D_h}) must be >= n_heads_decoder "
                    f"({self.n_heads_decoder}) for columnar attention."
                )
        if self.D_wm % self.n_heads_wm != 0:
            raise ValueError(
                f"D_wm ({self.D_wm}) must be divisible by "
                f"n_heads_wm ({self.n_heads_wm}) for WM attention."
            )

    def set_phase(self, phase: str):
        """Set component toggles for training phase.

        A: WM only
        B: WM + PM
        C: WM + PM + EM
        D: WM + PM + EM (+ RL controllers)
        E: WM + PM + EM + lifelong (PM/EM persist across doc boundaries)

        Design intent: All downstream code branches on capability flags
        (pm_enabled, em_enabled, rl_enabled) — never on phase letters.
        This method is the single point where phase -> flags mapping lives.
        """
        phase = phase.upper()
        if phase == "A":
            self.wm_enabled = True
            self.pm_enabled = False
            self.em_enabled = False
            self.rl_enabled = False
        elif phase == "B":
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = False
            self.rl_enabled = False
        elif phase == "C":
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = True
            self.rl_enabled = False
        elif phase == "D":
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = True
            self.rl_enabled = True
        elif phase == "E":
            self.wm_enabled = True
            self.pm_enabled = True
            self.em_enabled = True
            # Phase E inherits rl_enabled — does not force it on or off.
            # This allows Phase E to run with or without neuromodulators
            # depending on prior config (e.g. resuming from Phase D).
        else:
            raise ValueError(f"Unknown phase: {phase}. Expected A/B/C/D/E.")

        # Phase E: enable lifelong mode
        self.lifelong_mode = (phase == "E")

    @classmethod
    def tier_a(cls, **overrides) -> "ModelConfig":
        """Debug tier (~50M params). D=512, L=8, B=4."""
        defaults = dict(D=512, L=8, B=4)
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_b(cls, **overrides) -> "ModelConfig":
        """Competitive tier. D=768, L=12, B=6. Scaled memory."""
        defaults = dict(
            D=768, L=12, B=6,
            # Scaled memory capacities
            r=16, W=512, D_wm=192, n_heads_wm=6,
            M=512, D_em=192, k_ret=8, C_em=16, k_write=8,
            # Scaled decoder
            d_dec=384, n_heads_decoder=6,
        )
        defaults.update(overrides)
        return cls(**defaults)

    @classmethod
    def tier_c(cls, **overrides) -> "ModelConfig":
        """Strong tier. D=1024, L=24, B=8. Scaled memory."""
        defaults = dict(
            D=1024, L=24, B=8,
            # Scaled memory capacities
            r=32, W=1024, D_wm=256, n_heads_wm=8,
            M=1024, D_em=256, k_ret=16, C_em=32, k_write=16,
            # Scaled decoder
            d_dec=512, n_heads_decoder=8, decoder_layers=3,
        )
        defaults.update(overrides)
        return cls(**defaults)
