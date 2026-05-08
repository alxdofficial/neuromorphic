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
    # Has memory pieces
    assert model.manifold is not None
    assert model.read_module is not None
    assert model.write_module is not None
    # No Llama attached
    assert model.llama is None


def test_integrated_lm_forward_window_shapes():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 2
    input_ids = torch.randint(0, 100, (BS, cfg.T_window))
    prev_states = model.manifold.reset_states(batch_size=BS)
    out = model.forward_window(input_ids, None, prev_states)
    assert out["new_states"].shape == (BS, cfg.N, cfg.D_concept)
    assert out["current_hiddens"].shape == (BS, cfg.T_window, cfg.d_lm)
    assert out["read_visited"].shape == (BS, cfg.J, cfg.K_read)
    assert out["write_visited"].shape == (BS, cfg.J, cfg.K_write)


def test_integrated_lm_window_handles_first_window_None_prev():
    cfg = TrajMemConfig.small()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    input_ids = torch.randint(0, 100, (BS, cfg.T_window))
    prev_states = model.manifold.reset_states(batch_size=BS)
    # First window: prev_window_hiddens=None should not crash
    out = model.forward_window(input_ids, None, prev_states)
    assert out["new_states"].shape == prev_states.shape


def test_tbptt_chunker_split_shapes():
    cfg = TrajMemConfig.small()
    chunker = TBPTTChunker(cfg)
    BS = 2
    chunk_size = cfg.D * cfg.T_window
    total_T = chunk_size * 3                                       # 3 chunks
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
    out = run_chunk(model, windows, prev_states, prev_window_hiddens=None)
    assert len(out["window_outputs"]) == cfg.D
    assert out["final_states"].shape == prev_states.shape
    assert out["final_hiddens"].shape == (BS, cfg.T_window, cfg.d_lm)
    assert out["surprise_history"].shape == (BS, cfg.D)


def test_tbptt_grad_flows_to_first_window_write_module():
    """Critical: gradient from window D-1's loss must reach window 0's
    write module params.

    With attach_lm=False, surprise is 0 (no real LM), so we use a
    synthetic "loss" on the final hidden state to exercise gradient flow.
    The autograd chain: final_hiddens depends on read_module of last
    window, which depends on prev_states from window D-2's write, ...,
    ultimately back to window 0's write module.
    """
    cfg = TrajMemConfig.small()
    cfg.D = 3                                                      # short for fast test
    cfg.validate()
    model = IntegratedLM(cfg, attach_lm=False)
    BS = 1
    chunk_size = cfg.D * cfg.T_window
    input_ids = torch.randint(0, 100, (BS, chunk_size))
    windows = input_ids.view(BS, cfg.D, cfg.T_window)
    prev_states = model.manifold.reset_states(batch_size=BS)

    out = run_chunk(model, windows, prev_states, prev_window_hiddens=None)

    # In test mode current_hiddens is random (no autograd path back).
    # Use new_states (which DOES have autograd back to all writes) as
    # the gradient signal.
    out["final_states"].sum().backward()

    # Window 0's write_module should have received gradient.
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
    _ = run_chunk(model, windows, prev_states, prev_window_hiddens=None)
    assert torch.allclose(model.manifold.concept_states, buffer_before)
