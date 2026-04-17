"""Regression tests for reviewer-flagged gaps:

(a) Invalid Config.NC ≠ NC_pools gets rejected at validate() time.
(b) Triton dispatcher falls through to PyTorch when shapes are not
    kernel-compatible (non-power-of-2 Nc or D_n), and also on non-CUDA /
    non-bf16.
(c) One full train step runs without crashing on tier_tiny.

These catch the classes of bugs the reviewer found (config validates but
runtime crashes; Triton gated only on cuda+bf16; test coverage thin).
"""

import pytest
import torch

from src.model.config import Config
from src.model.model import Model
from src.model.triton_memory_step import fused_memory_step, fused_memory_step_torch


def test_invalid_nc_rejected_by_validate():
    """NC != NC_pools must raise at validate()."""
    # D=2048, D_n=256 → NC_pools = 8. Set N_cells = 4 → should fail.
    with pytest.raises(ValueError, match="N_cells"):
        Config.tier_a(N_cells=4)


def test_valid_config_accepted():
    """Default tier_a and tier_tiny validate without error."""
    Config.tier_a()
    Config.tier_tiny()


def test_triton_falls_back_on_cpu():
    """Non-CUDA inputs must route to the PyTorch reference, not the Triton kernel."""
    BS, NC, Nc, D_n, alpha = 2, 2, 16, 16, 2
    h = torch.randn(BS, NC, Nc, D_n)
    msg = torch.randn(BS, NC, Nc, D_n)
    W = torch.randn(BS, NC, Nc, Nc)
    decay = torch.sigmoid(torch.randn(BS, NC, Nc))
    inject_proj = torch.randn(BS, NC, alpha, D_n)
    out_mask = torch.zeros(NC, Nc)
    out_mask[:, alpha:2*alpha] = 1.0

    h_out, readout = fused_memory_step(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    # Should match the explicit PyTorch reference.
    h_ref, r_ref = fused_memory_step_torch(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    assert torch.allclose(h_out, h_ref)
    assert torch.allclose(readout, r_ref)


def test_triton_falls_back_on_non_power_of_2_nc():
    """Non-power-of-2 Nc must fall through to PyTorch even on CUDA+bf16."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device("cuda")
    dt = torch.bfloat16
    BS, NC, Nc, D_n, alpha = 2, 2, 24, 32, 2   # Nc=24 NOT a power of 2

    h = torch.randn(BS, NC, Nc, D_n, device=device, dtype=dt)
    msg = torch.randn(BS, NC, Nc, D_n, device=device, dtype=dt)
    W = torch.randn(BS, NC, Nc, Nc, device=device, dtype=dt) * 0.1
    decay = torch.sigmoid(torch.randn(BS, NC, Nc, device=device, dtype=dt))
    inject_proj = torch.randn(BS, NC, alpha, D_n, device=device, dtype=dt)
    out_mask = torch.zeros(NC, Nc, device=device, dtype=dt)
    out_mask[:, alpha:2*alpha] = 1.0

    # This should NOT raise a Triton compile error — the dispatcher should
    # detect non-power-of-2 Nc and route to PyTorch.
    h_out, readout = fused_memory_step(
        h, msg, W, decay, inject_proj, out_mask, alpha ** -0.5)
    assert h_out.shape == h.shape
    assert readout.shape == (BS, NC, D_n)


def test_one_train_step():
    """Smoke: Model + optimizer run one full forward+backward+step on tier_tiny."""
    cfg = Config.tier_tiny()
    model = Model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    BS, T = 2, cfg.T
    input_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    target_ids = torch.randint(0, cfg.vocab_size, (BS, T))

    model.train()
    result = model.forward_chunk(input_ids, target_ids=target_ids)
    loss = result["loss"]
    assert torch.isfinite(loss)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert model.memory.is_initialized
    assert torch.isfinite(model.memory.h).all()


def test_trainer_runs_one_step():
    """Trainer.train_chunk runs end-to-end, including telemetry + clip_grad.

    This is the test we needed to catch the attribute-mismatch bug in
    compute_component_grad_norms / compute_param_norms after the per-cell →
    shared-trunk revert. A pure Model+AdamW test wouldn't have caught it
    because the trainer calls the helpers; bare autograd doesn't.
    """
    from src.trainer import Trainer
    import dataclasses

    cfg = Config.tier_tiny()
    model = Model(cfg)

    @dataclasses.dataclass
    class Batch:
        input_ids: torch.Tensor
        target_ids: torch.Tensor
        prev_token: torch.Tensor | None = None

    # A minimal iterable "dataloader" yielding one fixed batch.
    class SingleBatchLoader:
        def __init__(self, batch):
            self.batch = batch

        def __iter__(self):
            yield self.batch

    batch = Batch(
        input_ids=torch.randint(1, cfg.vocab_size, (2, cfg.T)),
        target_ids=torch.randint(0, cfg.vocab_size, (2, cfg.T)),
    )
    loader = SingleBatchLoader(batch)

    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = Trainer(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
            scheduler=None,
            dataloader=loader,
            config=cfg,
            device=torch.device("cpu"),
            log_interval=1,
            metrics_path=f"{tmpdir}/metrics.jsonl",
        )
        # train_chunk runs the entire step including telemetry at log_interval=1.
        metrics = trainer.train_chunk(batch)

    assert "loss" in metrics
    assert metrics["loss"] > 0
    # Every telemetry group from the current code should be represented.
    for required_key in (
        "tok_proj_norm", "logit_head_norm", "cell_emb_norm",
        "decoder_norm", "codebook_norm",
        "W_offdiag_norm", "W_hebbian_offdiag_cos",
        "W_gamma_mean",
    ):
        assert required_key in metrics, f"missing {required_key}"


def test_telemetry_helpers_run():
    """Every telemetry function used by the trainer must execute and return dict.

    This caught a real bug: when the modulator/decoder changed from per-cell
    to shared + cell embeddings, the helpers still referenced the old
    `tok_proj_w1` / `logit_head_w` names. tier_tiny is CPU so the test
    exercises every helper without needing a GPU.
    """
    cfg = Config.tier_tiny()
    model = Model(cfg)
    BS, T = 2, cfg.T
    input_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    target_ids = torch.randint(0, cfg.vocab_size, (BS, T))
    model.train()
    model.forward_chunk(input_ids, target_ids=target_ids)["loss"].backward()

    # Each of these is called unconditionally or at log-cadence in trainer.py;
    # make sure every attribute it names actually exists on the current model.
    mem = model.memory
    for helper_name in (
        "compute_component_grad_norms",
        "compute_param_norms",
        "compute_plasticity_rates",
        "compute_memory_health",
        "compute_mod_grad_norm",
    ):
        result = getattr(mem, helper_name)()
        # Helpers return either dict or float. Just check no exception.
        assert result is not None

    # compute_lane_divergence needs BS > 1 to be non-trivial; tier_tiny BS=2.
    lane = mem.compute_lane_divergence()
    assert isinstance(lane, dict)

    mem_scale = model.lm.compute_mem_scale_stats()
    assert isinstance(mem_scale, dict) and "mem_scale_mean" in mem_scale
