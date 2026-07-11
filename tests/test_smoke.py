"""Fast CPU-only smoke/regression tests for the memory package.

These do NOT load model weights or touch the GPU — they guard the wiring that
the big GPU smoke is too heavy to run in CI: config sanity, the variant registry
shape (active cohort present, retired arms gone), the masked_reconstruction
mask-ratio dispatch, and the behavioral-KL teacher-geometry guard.
"""
import inspect

from src.memory.config import ReprConfig


# ── active cohort (2026-07-11): the arms the sweep trains + eval-only refs ──────
ACTIVE_TRAINABLE = ("icae_baseline", "autocompressor_baseline", "titans_baseline",
                    "gisting_baseline", "memoryllm_baseline", "slotgraph_baseline")
EVAL_ONLY = ("h2o_baseline", "vanilla_llama", "vanilla_full_context")
RETIRED = ("beacon_baseline", "ccm_baseline", "vqicae_baseline", "biomem_baseline",
           "slotgraph2_baseline", "slotgraph3_baseline", "slotgraph4_baseline",
           "graph_baseline", "hlvocab_baseline", "soft_pointer_graph_baseline")


# ── config: constructs + live fields present ───────────────────────────────────
def test_config_constructs_and_has_live_fields():
    c = ReprConfig()
    for f in ("n_flat_codes", "load_balance_coef", "z_loss_coef", "mae_mask_ratio",
              "use_llama_lora", "icae_lora_rank", "task_mode", "contrastive_shuf_coef",
              "objective_mode", "slotgraph_n_nodes", "slotgraph_d_edge"):
        assert hasattr(c, f), f"live field missing: {f}"


def test_config_retired_arm_fields_removed():
    c = ReprConfig()
    fields = set(c.__dataclass_fields__)
    dead = {"hlvocab_use_graph", "hlvocab_edge_cand", "spg_K_node", "spg_read_heads",
            "spg_inject_layer", "beacon_ratio", "beacon_param", "beacon_wrap_layers",
            "ccm_n_comp", "ccm_lora_rank", "vqicae_n_slots", "vqicae_codebook_size",
            "biomem_n_slots"}
    leaked = dead & fields
    assert not leaked, f"retired config fields still present: {sorted(leaked)}"


# ── variant registry: active present, retired gone ─────────────────────────────
def test_variant_registry_active_present_retired_gone():
    from src.memory.model import ReprLearningModel
    variants = set(ReprLearningModel.VARIANTS)
    for v in ACTIVE_TRAINABLE + EVAL_ONLY:
        assert v in variants, f"active variant missing from registry: {v}"
    for v in RETIRED:
        assert v not in variants, f"retired variant still in registry: {v}"
    # every MAE compressor must be a real variant
    assert set(ReprLearningModel._MASKED_RECON_COMPRESSORS) <= variants


# ── masked_reconstruction mask-ratio dispatch ──────────────────────────────────
def test_mask_ratio_is_forwarded():
    from src.memory.model import ReprLearningModel
    src = inspect.getsource(ReprLearningModel.compute_loss)
    assert "mask_ratio=" in src and "mae_mask_ratio" in src, \
        "compute_loss must forward cfg.mae_mask_ratio to compute_masked_reconstruction_loss"


# ── behavioral-KL teacher must force standard-causal geometry (audit #1) ───────
def test_kl_teacher_forces_causal_geometry():
    # the teacher forward must override bidir_mem_attn / uniform_mem_pos so an arm's non-standard
    # read geometry (e.g. slotgraph) can't scramble the "full-context teacher".
    import src.memory.training.objectives as obj
    src = inspect.getsource(obj)
    assert "bidir_mem_attn = False" in src and "uniform_mem_pos = False" in src, \
        "behavioral-KL teacher must force bidir_mem_attn/uniform_mem_pos OFF (standard-causal teacher)"


# ── behavioral-KL routes continuation through multi-horizon CE (audit #2) ──────
def test_kl_continuation_ce_path():
    import src.memory.training.objectives as obj
    src = inspect.getsource(obj)
    assert '"continuation"' in src, \
        "behavioral-KL must route continuation to the CE (multi-horizon) path, not the logits-KL path"


# ── checkpoint captures dynamically-attached cfg fields (audit #3) ─────────────
def test_checkpoint_captures_dynamic_cfg():
    import src.memory.training.checkpoint as ck
    src = inspect.getsource(ck)
    assert "cfg_all" in src and "vars(cfg)" in src, \
        "checkpoint metadata must capture ALL cfg attrs (vars), not just declared dataclass fields"
