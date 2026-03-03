"""Design-specific expected values (v5) — the ONE file to update when the design changes.

If a design test fails, look here first. If the design changed intentionally,
update this file. If not, the code has a regression.
"""

# ---------------------------------------------------------------------------
# Tier defaults (from ModelConfig.tier_a/b classmethods)
# ---------------------------------------------------------------------------
TIER_A = dict(D=2048, D_embed=384, B=4, C=16, L_scan=6, M=384, scan_expansion=8, d_inner=1024)
TIER_B = dict(D=3072, D_embed=512, B=12, C=16, L_scan=16, M=512, scan_expansion=4, d_inner=768)

# ---------------------------------------------------------------------------
# Default hyperparameters (from ModelConfig dataclass defaults)
# ---------------------------------------------------------------------------
DEFAULTS = dict(
    # PM
    budget_pm=16.0,
    decay_pm=0.999,
    # EM
    M=256,
    n_trail_steps=2,
    S_max=3.0,
    budget_em=8.0,
    decay_em=0.999,
    g_em_floor=0.001,
    g_em_ceil=0.95,
    # Training
    N=512,
    K_segments=1,
    use_compile=False,
    # Regularization
    dropout=0.1,
    tie_embeddings=True,
    # PCM
    pcm_enabled=False,
    pcm_pred_weight=0.01,
    # Neuromodulator
    neuromod_hidden=32,
    # Gradient checkpointing
    gradient_checkpointing=False,
)

# ---------------------------------------------------------------------------
# Phase toggles: phase -> expected flags after set_phase()
# ---------------------------------------------------------------------------
PHASE_TOGGLES = {
    "A": dict(pm_enabled=True, em_enabled=True, lifelong_mode=False),
    "B": dict(pm_enabled=True, em_enabled=True, lifelong_mode=True),
}

# ---------------------------------------------------------------------------
# State tensor names per class (v5)
# ---------------------------------------------------------------------------
STATE_TENSOR_NAMES = {
    "ProceduralMemory": ["W_pm"],
    "EpisodicMemory": ["em_K", "em_V", "em_S"],
}
