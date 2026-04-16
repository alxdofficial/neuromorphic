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
        "memory.state_w1": model.memory.state_w1,
        "memory.inject_w": model.memory.inject_w,
        "memory.W_decay_logit": model.memory.W_decay_logit,
        "memory.modulator.tok_proj": model.memory.modulator.tok_proj[0].weight,
        "memory.modulator.qkv (layer 0)": model.memory.modulator.layers[0].qkv.weight,
        "memory.modulator.logit_head": model.memory.modulator.logit_head.weight,
        "memory.discrete_policy.codebook": model.memory.discrete_policy.codebook,
        "memory.decoder.mlp[0]": model.memory.decoder.mlp[0].weight,
        "memory.decoder.mlp[-1]": model.memory.decoder.mlp[-1].weight,
    }
    for name, p in required.items():
        assert p.grad is not None, f"no gradient for {name}"
        assert torch.isfinite(p.grad).all().item(), f"non-finite gradient in {name}"


def test_decoder_starts_at_no_op():
    """Fresh decoder must produce zero ΔW and zero Δdecay — EMA is a no-op."""
    cfg = Config.tier_tiny()
    model = Model(cfg)
    model.train(False)

    BS = 2
    D_code = cfg.code_dim
    emb = torch.randn(BS, D_code)
    with torch.no_grad():
        dW, dDecay = model.memory.decoder(emb)

    # Both heads are zero-init, so output is exactly 0 regardless of input.
    assert torch.allclose(dW, torch.zeros_like(dW), atol=1e-6), (
        f"dW is not zero at init: max={dW.abs().max().item()}")
    assert torch.allclose(dDecay, torch.zeros_like(dDecay), atol=1e-6), (
        f"dDecay is not zero at init: max={dDecay.abs().max().item()}")


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
