"""End-to-end smoke tests for trajectory_memory without Llama.

These tests run the full per-window cycle and TBPTT chunk path using
`attach_lm=False`, which fakes the LM forward (random hiddens, zero
logits). Real-Llama integration tests need access to the HF cache and
live in test_trajectory_memory_llama.py.
"""

from __future__ import annotations

import torch

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.tbptt import TBPTTChunker, run_chunk


def test_integrated_lm_test_mode_constructs():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    assert model.manifold is not None
    assert model.read_module is not None
    assert model.write_module is not None
    assert model.llama is None


def test_integrated_lm_forward_window_shapes():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 2
    # In test mode (attach_lm=False) Llama is skipped, so lm_input_ids size
    # doesn't matter beyond satisfying the >= T_window assertion. Use exactly
    # T_window for shape-test simplicity.
    lm_input = torch.randint(0, 100, (BS, cfg.T_window))
    prev_states = model.manifold.reset_states(batch_size=BS)
    out = model.forward_window(lm_input, None, prev_states)
    assert out["new_states"].shape == (BS, cfg.N, cfg.D_concept)
    assert out["current_hiddens"].shape == (BS, cfg.T_window, cfg.d_lm)
    assert out["read_visited"].shape == (BS, cfg.J, cfg.K_read)
    assert out["write_visited"].shape == (BS, cfg.J, cfg.K_write)


def test_integrated_lm_window_handles_first_window_None_prev():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    lm_input = torch.randint(0, 100, (BS, cfg.T_window))
    prev_states = model.manifold.reset_states(batch_size=BS)
    out = model.forward_window(lm_input, None, prev_states)
    assert out["new_states"].shape == prev_states.shape


def test_tbptt_chunker_split_shapes():
    cfg = TrajMemConfig.small()
    chunker = TBPTTChunker(cfg)
    BS = 2
    chunk_size = cfg.D * cfg.T_window
    total_T = chunk_size * 3
    input_ids = torch.randint(0, 100, (BS, total_T))
    chunks = chunker.split(input_ids)
    assert len(chunks) == 3
    for c in chunks:
        assert c.shape == (BS, cfg.D, cfg.T_window)


def test_tbptt_run_chunk_shapes():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 2
    chunk_size = cfg.D * cfg.T_window
    input_ids = torch.randint(0, 100, (BS, chunk_size))
    windows = input_ids.view(BS, cfg.D, cfg.T_window)

    prev_states = model.manifold.reset_states(batch_size=BS)
    out = run_chunk(model, windows, prev_states, prev_window_hiddens=None, prev_lm_context=None)
    assert len(out["window_outputs"]) == cfg.D
    assert out["final_states"].shape == prev_states.shape
    assert out["final_hiddens"].shape == (BS, cfg.T_window, cfg.d_lm)
    assert out["surprise_history"].shape == (BS, cfg.D)
    # Rolling LM context should be at most cap - T_window tokens (cap is
    # filled in by the next window's append).
    cap = cfg.effective_lm_context
    assert out["final_lm_context"].shape[1] <= cap - cfg.T_window


def test_tbptt_grad_flows_to_first_window_write_module():
    """Critical: gradient from end-of-chunk must reach window 0's write
    module params, exercising cross-window TBPTT.
    """
    cfg = TrajMemConfig.small()
    cfg.D = 3
    cfg.validate()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    chunk_size = cfg.D * cfg.T_window
    input_ids = torch.randint(0, 100, (BS, chunk_size))
    windows = input_ids.view(BS, cfg.D, cfg.T_window)
    prev_states = model.manifold.reset_states(batch_size=BS)

    out = run_chunk(model, windows, prev_states, prev_window_hiddens=None, prev_lm_context=None)
    out["final_states"].sum().backward()

    write_grad_norm = sum(
        p.grad.abs().sum() for p in model.write_module.parameters()
        if p.grad is not None
    )
    assert write_grad_norm > 0, (
        "no gradient reached write_module from final_states — "
        "cross-window TBPTT broken"
    )


def test_integrated_lm_does_not_mutate_concept_states_buffer():
    """The Manifold.concept_states BUFFER (not the per-batch tensor) should
    remain unchanged after forward — the architecture's contract.
    """
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    buffer_before = model.manifold.concept_states.clone()
    chunk_size = cfg.D * cfg.T_window
    input_ids = torch.randint(0, 100, (BS, chunk_size))
    windows = input_ids.view(BS, cfg.D, cfg.T_window)
    prev_states = model.manifold.reset_states(batch_size=BS)
    _ = run_chunk(model, windows, prev_states, prev_window_hiddens=None, prev_lm_context=None)
    assert torch.allclose(model.manifold.concept_states, buffer_before)


def test_tbptt_lm_context_grows_then_caps():
    """LM context buffer should accumulate across windows up to
    effective_lm_context, then stop growing (rolling)."""
    cfg = TrajMemConfig.small()
    # Force the cap to be small so we can see the rolling behavior across
    # this chunk: cap = 2 * T_window means after window 1 we hit the cap.
    cfg.effective_lm_context = 2 * cfg.T_window
    cfg.D = 4
    cfg.validate()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    chunk_size = cfg.D * cfg.T_window
    input_ids = torch.randint(0, 100, (BS, chunk_size))
    windows = input_ids.view(BS, cfg.D, cfg.T_window)
    prev_states = model.manifold.reset_states(batch_size=BS)

    out = run_chunk(model, windows, prev_states, prev_window_hiddens=None, prev_lm_context=None)
    # After D=4 windows, the rolling buffer should be exactly cap - T_window
    # = T_window (so when window D+1 arrives, the new lm_input is exactly
    # cap = 2 * T_window).
    assert out["final_lm_context"].shape[1] == cfg.T_window
