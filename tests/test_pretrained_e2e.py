"""End-to-end smoke for the cycle orchestrator + telemetry + plotting.

Runs a tiny bootstrap → 1 cycle (p1-AR + p2 GRPO) on a random-init Llama.
Verifies:
- The cycle loop completes without raising.
- StatsCollector writes a JSONL with rows from each phase.
- plot_dashboard reads the JSONL and writes 8 PNGs.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.train_loop import CycleConfig, run_cycle_loop
from src.graph_walker.pretrained.train_phase1 import Phase1Batch
from src.graph_walker.pretrained.train_phase1_ar import Phase1ARBatch
from src.graph_walker.telemetry import plot_dashboard


def _make_tiny_llama(d_lm=32, n_layers=4, vocab=256):
    cfg = LlamaConfig(
        vocab_size=vocab, hidden_size=d_lm, intermediate_size=d_lm * 2,
        num_hidden_layers=n_layers, num_attention_heads=4,
        num_key_value_heads=4, max_position_embeddings=64,
        pad_token_id=0, bos_token_id=0, eos_token_id=0,
    )
    return LlamaForCausalLM(cfg)


def _tiny_walker_cfg(D_s, vocab, T=8):
    return GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2,
        K=4, D_model=D_s, D_s=D_s, D_id=8,
        n_heads=2, n_hops=2,
        D_q_in=8, D_q_per_head=8, n_score_heads=2,
        K_horizons=4, K_buf=4, vocab_size=vocab,
        # Single-knob clock under external-surprise plasticity.
        mod_period=T, tbptt_block=T, segment_T=T,
        gumbel_tau_start=1.0, gumbel_tau_end=1.0, gumbel_anneal_steps=1,
        epsilon_start=0.0, epsilon_end=0.0, epsilon_anneal_steps=1,
        lambda_balance=0.0, use_neuromod=True,
        plasticity_mode="neuromod_only",
        neuromod_D_mod=16, neuromod_n_layers=1, neuromod_n_heads=2,
        neuromod_edge_hidden=16, neuromod_eta=1.0,
    )


def test_cycle_loop_e2e_with_telemetry_and_plots(tmp_path: Path):
    torch.manual_seed(0)
    hf = _make_tiny_llama()
    walker_cfg = _tiny_walker_cfg(D_s=32, vocab=256, T=8)
    cfg = PretrainedGWConfig(
        model_name="random", inject_layer=2, d_mem=32,
        memory=walker_cfg, T=8, bs=2, llama_dtype="fp32",
    )
    w = GraphWalkerPretrainedLM(cfg, hf_model=hf)
    # Perturb zero-init gates so paths through them carry gradient.
    m = w.memory
    with torch.no_grad():
        m.prev_motor_proj.weight.normal_(std=0.05)
        m.decay_proj.weight.normal_(std=0.05)
        m.decay_proj.bias.normal_(std=0.05)
        m.readout.pred_head.proj.weight.normal_(std=0.05)
        m.neuromod.edge_mlp[-1].weight.normal_(std=0.3)
        m.neuromod.edge_mlp[-1].bias.normal_(std=0.05)
        m.neuromod.blend_logit.fill_(0.0)

    boot_data = [
        Phase1Batch(
            input_ids=torch.randint(0, 256, (2, 8)),
            target_ids=torch.randint(0, 256, (2, 8)),
        )
        for _ in range(2)
    ]
    p1_data = [
        Phase1ARBatch(
            prefix_ids=torch.randint(0, 256, (2, 8)),
            continuation_ids=torch.randint(0, 256, (2, 4)),
        )
        for _ in range(1)
    ]
    p2_data = [
        (torch.randint(0, 256, (1, 8)), torch.randint(0, 256, (4,)))
        for _ in range(1)
    ]

    def reward(generated, ref):
        return torch.randn(generated.shape[0])

    work_dir = tmp_path / "run1"
    cycle_cfg = CycleConfig(
        work_dir=str(work_dir),
        bootstrap_steps=2, cycles=1,
        cycle_phase1_steps=1, cycle_phase2_steps=1,
        grpo_K=4, grpo_rollout_len=4,
    )
    run_cycle_loop(
        w, boot_data, p1_data, p2_data,
        reward_fn=reward, cfg=cycle_cfg,
    )

    # Verify the JSONL has rows from each phase.
    stats_path = work_dir / "stats.jsonl"
    assert stats_path.exists()
    rows = stats_path.read_text().strip().splitlines()
    assert len(rows) == 4, f"expected 2+1+1 rows, got {len(rows)}"

    # Plot pipeline doesn't raise and produces 8 PNGs.
    out = plot_dashboard(stats_path, out_dir=work_dir / "plots")
    assert len(out) == 8
    for p in out:
        assert p.exists() and p.stat().st_size > 0, (
            f"empty plot file: {p}"
        )
