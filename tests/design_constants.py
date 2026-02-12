"""Design-specific expected values — the ONE file to update when the design changes.

If a design test fails, look here first. If the design changed intentionally,
update this file. If not, the code has a regression.
"""

# ---------------------------------------------------------------------------
# Tier defaults (from ModelConfig.tier_a/b/c classmethods)
# ---------------------------------------------------------------------------
TIER_A = dict(D=512, L=8, B=4)
TIER_B = dict(
    D=768, L=12, B=6,
    r=16, W=512, D_wm=192, n_heads_wm=6,
    M=512, D_em=192, k_ret=8, C_em=16,
    d_dec=384, n_heads_decoder=6,
)
TIER_C = dict(
    D=1024, L=24, B=8,
    r=32, W=1024, D_wm=256, n_heads_wm=8,
    M=1024, D_em=256, k_ret=16, C_em=32,
    d_dec=512, n_heads_decoder=8, decoder_layers=3,
)

# ---------------------------------------------------------------------------
# Default hyperparameters (from ModelConfig.__init__ defaults)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # PM
    r=8,
    rho=0.95,
    a_max=3.0,
    budget_pm=4.0,
    decay_pm=0.999,
    tau_pm=1.0,
    weakness_weight_pm=0.5,
    pm_readout_ffn=True,
    # EM
    M=256,
    D_em=128,
    k_ret=4,
    C_em=8,
    tau_em=1.0,
    weakness_weight_em=0.5,
    S_max=3.0,
    budget_em=8.0,
    decay_em=0.999,
    g_em_floor=0.001,
    g_em_ceil=0.95,
    em_readout_ffn=True,
    # Training
    T=256,
    P=32,
    # FFN
    ffn_expansion=4,
    # Decoder
    d_dec=256,
    n_heads_decoder=4,
    decoder_layers=2,
    columnar_layers=2,
    thalamic_layers=2,
    thalamic_tokens=4,
    # Neuromodulator
    neuromod_hidden=32,
    content_proj_dim=8,
)

# ---------------------------------------------------------------------------
# Phase toggles: phase -> expected flags after set_phase()
# ---------------------------------------------------------------------------
PHASE_TOGGLES = {
    "A": dict(wm_enabled=True, pm_enabled=True, em_enabled=False, lifelong_mode=False),
    "B": dict(wm_enabled=True, pm_enabled=True, em_enabled=True, lifelong_mode=False),
    "D": dict(wm_enabled=True, pm_enabled=True, em_enabled=True, lifelong_mode=True),
}

# ---------------------------------------------------------------------------
# Architecture constants
# ---------------------------------------------------------------------------
# Gate input dim = 4 * D_h + 1 (x_block + y_pm + y_wm_proj + y_em_proj + surprise)
GATE_INPUT_FORMULA = lambda D_h: 4 * D_h + 1

# PM neuromodulator backbone input dim (3 scalars + content_proj_dim)
PM_NEUROMOD_INPUT_DIM = 3 + 8  # (elig_norm, pm_usage, span_surprise) + content_proj

# EM neuromodulator backbone input dim (3 scalars + content_proj_dim)
EM_NEUROMOD_INPUT_DIM = 3 + 8  # (span_surprise, em_usage, cand_novelty_mean) + content_proj

# PM readout FFN expansion factor
PM_READOUT_FFN_EXPANSION = 4

# EM readout FFN expansion factor
EM_READOUT_FFN_EXPANSION = 4

# EM neuromodulator default g in heuristic mode
EM_NEUROMOD_DEFAULT_G = 0.3

# PM neuromodulator default g in heuristic mode
PM_NEUROMOD_DEFAULT_G = 0.5

# PM eligibility surprise gating normalization ceiling
PM_ELIG_SURPRISE_NORM = 5.0

# Decoder output_proj init std
DECODER_OUTPUT_PROJ_INIT_STD = 0.01

# ---------------------------------------------------------------------------
# State tensor names per class
# ---------------------------------------------------------------------------
STATE_TENSOR_NAMES = {
    "NeuromorphicLM": ["surprise"],
    "Layer": ["h"],
    "ProceduralMemory": ["pm_K", "pm_V", "pm_a", "elig_K", "elig_V"],
    "EpisodicMemory": ["em_K", "em_V", "em_S"],
    "WorkingMemory": ["wm_K", "wm_V", "wm_valid", "wm_ptr"],
    "Block": [],
}
