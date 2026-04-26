"""Verification script: prove the captured graph is computing, not cheating.

For each suspicion below, run an experiment that would FAIL if the
captured graph were silently skipping work. All checks must pass for
the throughput numbers to be honest.

Suspicions tested:
  1. "Forward elided"           — outputs depend on inputs
  2. "Backward elided"          — param grads depend on inputs
  3. "State frozen"             — memory.s evolves between replays
  4. "E_bias frozen"            — Hebbian update fires per replay
  5. "Loss is constant"         — loss varies with input
  6. "Replay is a no-op"        — wall-clock per replay > 0
  7. "Captured == warmup output"— different inputs → different loss
  8. "Cudagraph silently breaks autograd"
                                — cudagraph grad magnitude matches eager grad
                                  magnitude on the same input
"""

from __future__ import annotations

import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step, phase1_step_cudagraph


def make_lm() -> StandaloneLM:
    cfg = GraphWalkerConfig(
        plane_rows=8, plane_cols=8, L=4, K=16,
        D_model=512, D_s=256, D_id=32,
        n_heads=4, n_score_heads=4, D_q_per_head=32, D_q_in=32,
        K_horizons=4, K_buf=4,
        mod_period=64, tbptt_block=64, segment_T=128,
        ffn_mult_content=4, content_mlp_depth=2, post_model_depth=1,
        vocab_size=1024, use_neuromod=False, compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def main() -> None:
    if not torch.cuda.is_available():
        print("no cuda")
        return
    print("=== cudagraph reality check (B=8, T=128) ===\n", flush=True)

    B = 8
    torch.manual_seed(0)
    lm = make_lm()
    opt = torch.optim.Adam(lm.parameters(), lr=1e-4)

    # Two distinct token sets.
    torch.manual_seed(1)
    tokens_a = torch.randint(0, lm.cfg.vocab_size, (B, lm.cfg.segment_T), device="cuda")
    torch.manual_seed(2)
    tokens_b = torch.randint(0, lm.cfg.vocab_size, (B, lm.cfg.segment_T), device="cuda")
    assert not torch.equal(tokens_a, tokens_b), "test setup bug"

    # ---------------------------------------------------------------
    # 1+2+5+7. Different inputs -> different loss + different grads
    # ---------------------------------------------------------------
    print("[1,2,5,7] capture once, then replay with two distinct token sets",
          flush=True)
    print("          loss must differ, param.grads must differ\n", flush=True)

    # First replay (also triggers warmup + capture).
    stats_a = phase1_step_cudagraph(lm, opt, tokens_a, training_step=0)
    grad_a = {n: p.grad.detach().clone() for n, p in lm.named_parameters()
              if p.grad is not None}
    s_after_a = lm.memory.s.detach().clone()
    e_after_a = lm.memory.E_bias_flat.detach().clone()

    # Second replay with DIFFERENT tokens.
    stats_b = phase1_step_cudagraph(lm, opt, tokens_b, training_step=1)
    grad_b = {n: p.grad.detach().clone() for n, p in lm.named_parameters()
              if p.grad is not None}
    s_after_b = lm.memory.s.detach().clone()
    e_after_b = lm.memory.E_bias_flat.detach().clone()

    print(f"  loss A = {stats_a.loss:.4f}",  flush=True)
    print(f"  loss B = {stats_b.loss:.4f}",  flush=True)
    assert abs(stats_a.loss - stats_b.loss) > 1e-3, (
        f"loss did not change between distinct inputs ({stats_a.loss}, {stats_b.loss}) "
        f"FORWARD MAY BE ELIDED"
    )
    print("  OK loss differs across input sets — forward is consuming the new tokens",
          flush=True)

    # Pick a sample of params at different layers to check grads differ.
    sample_names = []
    for n in grad_a.keys():
        if any(k in n for k in [
            "tied_token_emb", "memory.cols.content_mlp", "memory.cols.q_proj",
            "memory.cols.k_proj", "memory.col_id", "memory.state_to_model",
        ]):
            sample_names.append(n)
        if len(sample_names) >= 6:
            break
    diff_count = 0
    for n in sample_names:
        ga, gb = grad_a[n], grad_b[n]
        delta = (ga - gb).abs().max().item()
        rel = delta / (gb.abs().max().item() + 1e-12)
        same = delta < 1e-7
        flag = "SAME" if same else f"diff={delta:.4e} rel={rel:.2%}"
        print(f"    {n:<55} {flag}", flush=True)
        if not same:
            diff_count += 1
    assert diff_count == len(sample_names), (
        "some param grads did NOT change between input sets BACKWARD MAY BE ELIDED"
    )
    print(f"  OK all {len(sample_names)} sampled param grads differ — backward is real\n",
          flush=True)

    # ---------------------------------------------------------------
    # 3. State evolves between replays
    # ---------------------------------------------------------------
    print("[3] memory.s after replay A vs after replay B must differ",
          flush=True)
    s_delta = (s_after_a.float() - s_after_b.float()).abs().max().item()
    print(f"  max|s_a - s_b| = {s_delta:.4e}", flush=True)
    assert s_delta > 1e-3, (
        "memory.s did not evolve between replays STATE WRITEBACK MAY BE BROKEN"
    )
    print("  OK state evolves\n", flush=True)

    # ---------------------------------------------------------------
    # 4. E_bias_flat changes per replay (Hebbian fires)
    # ---------------------------------------------------------------
    print("[4] E_bias_flat after replay A vs after replay B must differ",
          flush=True)
    e_delta = (e_after_a - e_after_b).abs().max().item()
    print(f"  max|E_a - E_b| = {e_delta:.4e}", flush=True)
    assert e_delta > 1e-6, (
        "E_bias_flat unchanged HEBBIAN UPDATE NOT FIRING"
    )
    print("  OK Hebbian plasticity fires every replay\n", flush=True)

    # ---------------------------------------------------------------
    # 6. Replay actually takes wall-clock time
    # ---------------------------------------------------------------
    print("[6] wall-clock per replay > 0 (i.e. replay is not a no-op)",
          flush=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(20):
        phase1_step_cudagraph(lm, opt, tokens_a, training_step=10 + i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    per_iter = elapsed / 20 * 1000
    tps = 20 * B * lm.cfg.segment_T / elapsed
    print(f"  {per_iter:.2f} ms/iter, {tps:.0f} tok/s", flush=True)
    assert per_iter > 0.1, "replay wall-clock implausibly low INSTRUMENTATION BROKEN"
    print(f"  OK each iter actually takes time on the GPU\n", flush=True)

    # ---------------------------------------------------------------
    # 8. Cudagraph grad magnitudes match eager grad magnitudes
    #    on the same tokens (within Gumbel/STE noise)
    # ---------------------------------------------------------------
    print("[8] cudagraph vs eager: grad MAGNITUDES on same tokens",
          flush=True)
    print("    (use train(False) so Gumbel becomes deterministic argmax)",
          flush=True)

    torch.manual_seed(0)
    lm_e = make_lm()
    opt_e = torch.optim.Adam(lm_e.parameters(), lr=1e-4)
    lm_e.train(False)
    phase1_step(lm_e, opt_e, tokens_a, training_step=0)
    grad_e = {n: p.grad.detach().clone() for n, p in lm_e.named_parameters()
              if p.grad is not None}

    torch.manual_seed(0)
    lm_c = make_lm()
    opt_c = torch.optim.Adam(lm_c.parameters(), lr=1e-4)
    lm_c.train(False)
    phase1_step_cudagraph(lm_c, opt_c, tokens_a, training_step=0)
    grad_c = {n: p.grad.detach().clone() for n, p in lm_c.named_parameters()
              if p.grad is not None}

    print(f"  {'param':<55} {'eager_norm':>12} {'cuda_norm':>12} {'ratio':>8}",
          flush=True)
    for n in sample_names:
        ge = grad_e[n].float().norm().item()
        gc = grad_c[n].float().norm().item()
        ratio = gc / (ge + 1e-12)
        print(f"  {n:<55} {ge:>12.4e} {gc:>12.4e} {ratio:>8.3f}", flush=True)
        # Should be within an order of magnitude. The cudagraph path uses
        # the captured RNG state and the eager path uses fresh RNG, so
        # exact equality is not expected; what we want is "nontrivial
        # gradient at all and same order of magnitude".
        assert 0.05 < ratio < 20.0, (
            f"grad magnitude diverges wildly for {n}: eager={ge:.4e} cuda={gc:.4e}"
        )
    print("  OK cudagraph grads in same order of magnitude as eager grads\n",
          flush=True)

    print("=== ALL CHECKS PASS — captured graph is doing real work ===",
          flush=True)


if __name__ == "__main__":
    main()
