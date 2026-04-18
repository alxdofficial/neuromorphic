"""Smoke test — guards against shape / dtype / init regressions.

Must run on CPU (no autocast). If this passes, the basic forward+backward
shape and dtype contracts are intact.
"""

import torch

from src.model.config import Config
from src.model.model import Model


def test_forward_backward_cpu_tier_tiny():
    """Forward + backward on tier_tiny CPU path produces finite loss + grads."""
    cfg = Config.tier_tiny()
    model = Model(cfg)
    BS, T = 2, cfg.T
    input_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    target_ids = torch.randint(0, cfg.vocab_size, (BS, T))

    model.train()
    result = model.forward_chunk(input_ids, target_ids=target_ids)

    assert result["logits"].shape == (BS, T, cfg.vocab_size)
    assert torch.isfinite(result["loss"]).item()
    result["loss"].backward()

    # Gradient must reach every trainable component.
    required = {
        "lm.embedding": model.lm.embedding.weight,
        "memory.neuron_id": model.memory.neuron_id,
        "memory.decay_gamma_logit": model.memory.decay_gamma_logit,
        "memory.inject_w": model.memory.inject_w,
        "memory.W_decay_logit": model.memory.W_decay_logit,
        "memory.modulator.tok_proj[0]": model.memory.modulator.tok_proj[0].weight,
        "memory.modulator.qkv (layer 0)": model.memory.modulator.layers[0].qkv.weight,
        "memory.modulator.logit_head": model.memory.modulator.logit_head.weight,
        "memory.modulator.cell_emb": model.memory.modulator.cell_emb,
        "memory.discrete_policy.codebook": model.memory.discrete_policy.codebook,
        "memory.decoder.mlp[0]": model.memory.decoder.mlp[0].weight,
        "memory.decoder.mlp[-1]": model.memory.decoder.mlp[-1].weight,
    }
    for name, p in required.items():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all().item(), f"non-finite gradient in {name}"


def test_decoder_invariants_at_init():
    """Decoder produces valid ΔW shape at init: zero diagonal + rms_norm'd rows.

    Prior version of this test required dW to be exactly zero at init
    (zero-weight final layer). That init was buggy: with `rms_norm(0)=0`,
    the `W = (1-γ_W)·W + γ_W·0` EMA dragged W to zero, collapsing the
    memory graph. Decoder is now Xavier-init so ΔW has unit-RMS rows
    from step 0 — sacrificing the "exact no-op" semantic in exchange for
    actual trainability.
    """
    cfg = Config.tier_tiny()
    model = Model(cfg)
    model.train(False)

    BS = 2
    D_code = cfg.code_dim
    emb = torch.randn(BS, cfg.N_cells, D_code)
    with torch.no_grad():
        dW, dDecay = model.memory.decoder(emb, model.memory.modulator.cell_emb)

    # Shapes.
    assert dW.shape == (BS, cfg.N_cells, cfg.neurons_per_cell, cfg.neurons_per_cell)
    assert dDecay.shape == (BS, cfg.N_cells, cfg.neurons_per_cell)

    # Diagonal is zeroed (via diag_mask) — neurons can't modulate self-synapses.
    diag = torch.diagonal(dW, dim1=-2, dim2=-1)
    assert torch.allclose(diag, torch.zeros_like(diag), atol=1e-6), (
        f"dW diagonal not zero: max={diag.abs().max().item()}")

    # Rows are rms-normed to unit RMS (up to rms_norm eps).
    rms = dW.pow(2).mean(dim=-1).sqrt()
    # Rows with the diagonal zeroed have RMS ≈ sqrt((Nc-1)/Nc) of the
    # pre-norm row (since 1/Nc of the row got zeroed). Just check it's a
    # reasonable nonzero value, not 0.
    assert rms.mean() > 0.1, f"dW rows near-zero: mean RMS {rms.mean().item()}"


def test_gamma_clamp():
    """γ values stay <= gamma_max regardless of logit magnitude."""
    cfg = Config.tier_tiny()
    model = Model(cfg)
    # Push all plasticity logits to +50 — sigmoid saturates to ~1.
    with torch.no_grad():
        model.memory.W_decay_logit.fill_(50.0)
        model.memory.decay_gamma_logit.fill_(50.0)
        model.memory.hebbian_decay_logit.fill_(50.0)

    BS, T = 2, cfg.T
    input_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    target_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    model.train()
    result = model.forward_chunk(input_ids, target_ids=target_ids)
    assert torch.isfinite(result["loss"]).item()
    # After one step, decay never exceeds 1.0.
    assert model.memory.decay.max().item() <= 1.0 + 1e-5


def test_port_layout_partitions_neurons():
    """input / output / internal role ids partition [0, N)."""
    from src.model.attention_modulator import port_layout
    cfg = Config.tier_tiny()
    layout = port_layout(cfg)
    input_idx = layout["input_port_idx"].flatten()
    output_idx = layout["output_port_idx"].flatten()
    role_id = layout["role_id"]

    assert input_idx.numel() == cfg.NC_pools * cfg.alpha
    assert output_idx.numel() == cfg.NC_pools * cfg.alpha
    # Roles: 0=input, 1=output, 2=internal.
    assert (role_id[input_idx] == 0).all()
    assert (role_id[output_idx] == 1).all()
    internal_count = (role_id == 2).sum().item()
    assert internal_count == cfg.N_internal
