"""Fast CPU-only smoke/regression tests for the memory package.

These do NOT load model weights or touch the GPU — they guard the wiring that
the big GPU smoke (scripts/diagnostics/mae_smoke.py) is too heavy to run in CI:
config sanity after the dead-field cleanup, the variant registry shape, the
HLVocabConfig edge_cand constraint, the beacon wrap-layer derivation, and the
masked_reconstruction mask-ratio dispatch.
"""
import inspect

from src.memory.config import ReprConfig


# ── config: live fields present, dead fields gone (cleanup regression) ──────
def test_config_constructs_and_has_live_fields():
    c = ReprConfig()
    for f in ("n_flat_codes", "load_balance_coef", "z_loss_coef",
              "mae_mask_ratio", "hlvocab_use_graph", "hlvocab_edge_cand",
              "spg_K_node", "use_llama_lora", "icae_lora_rank",
              "task_mode", "contrastive_shuf_coef", "seed"):
        assert hasattr(c, f), f"live field missing: {f}"


def test_config_dead_fields_removed():
    c = ReprConfig()
    fields = set(c.__dataclass_fields__)
    dead = {"d_concept", "d_edge", "n_nodes", "n_edges", "d_node_state",
            "d_continuous", "d_mt_value", "d_recurrent", "d_mamba", "d_enc",
            "enc_n_layers", "edge_token_packing", "selection_temperature",
            "slot_iters", "mt_layer", "use_role_embeddings", "use_qformer_adapter",
            "plastic_depth", "splat_K", "b_diversity_scale", "mt_diversity_scale",
            "fixed_window_size", "max_window_size", "mask_ratio_min", "eval_every",
            "device", "dtype", "spg_read_heads", "spg_inject_layer"}
    leaked = dead & fields
    assert not leaked, f"dead config fields still present: {sorted(leaked)}"


# ── variant registry shape ──────────────────────────────────────────────────
def test_variant_registry_and_mae_compressor_subset():
    from src.memory.model import ReprLearningModel
    variants = set(ReprLearningModel.VARIANTS)
    for v in ("hlvocab_baseline", "icae_baseline", "ccm_baseline",
              "autocompressor_baseline", "beacon_baseline",
              "soft_pointer_graph_baseline", "vanilla_llama", "vanilla_full_context"):
        assert v in variants, f"variant missing from registry: {v}"
    # every MAE compressor must be a real variant
    assert set(ReprLearningModel._MASKED_RECON_COMPRESSORS) <= variants


# ── HLVocabConfig edge_cand constraint (bug #1) ─────────────────────────────
def test_hlvocab_edge_cand_constraint():
    import pytest
    from src.memory.models.hierarchical_learned_vocab.substrate import HLVocabConfig
    # edge_cand below m_max//2 must be rejected
    with pytest.raises(ValueError):
        HLVocabConfig(use_graph=True, m_max=144, edge_cand=48)
    # the trainer's derivation edge_cand = max(default, ceil(m_max/2)) must satisfy it
    for M in (16, 32, 144, 256):
        ec = max(48, (M + 1) // 2)
        HLVocabConfig(use_graph=True, m_max=M, edge_cand=ec)  # should not raise
    # odd m_max would silently emit m_max-1 tokens (2 per edge) → must be rejected
    with pytest.raises(ValueError):
        HLVocabConfig(use_graph=True, m_max=5, edge_cand=48)


# ── beacon wrap-layer derivation (shared helper used by trainer/param_count) ─
def test_beacon_wrap_layer_helper():
    from src.memory.common import beacon_wrap_layers
    # the actual helper the trainer + param_count use (6 evenly-spaced layers)
    assert beacon_wrap_layers(30) == (0, 6, 12, 17, 23, 29)   # SmolLM2-135M (capacity calibration)
    assert beacon_wrap_layers(16) == (0, 3, 6, 9, 12, 15)     # 16-layer backbone, all distinct
    assert len(beacon_wrap_layers(16)) == 6


# ── masked_reconstruction mask-ratio dispatch (bug #2) ──────────────────────
def test_mask_ratio_is_forwarded():
    from src.memory.model import ReprLearningModel
    src = inspect.getsource(ReprLearningModel.compute_loss)
    assert "mask_ratio=self.cfg.mae_mask_ratio" in src, \
        "compute_loss must forward cfg.mae_mask_ratio to compute_masked_reconstruction_loss"
