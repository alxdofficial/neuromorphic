"""Detailed profiling: per-operation breakdown of the memory graph hot path."""

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.model import Model
from src.model.memory import MemoryGraph


def time_op(fn, warmup=3, iters=10, sync=True):
    """Time a callable, return avg ms."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def profile_memory_graph(bs=8):
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)

    NC = config.N_cells
    Cn = config.neurons_per_cell
    Dn = config.D_n
    K = config.K
    D = config.D
    T = config.T

    print(f"Config: NC={NC}, Cn={Cn}, Dn={Dn}, K={K}, D={D}, T={T}, BS={bs}")
    print(f"  mlp_groups={config.mlp_groups}, mod_interval={config.modulation_interval}")
    print(f"  tbptt_block={config.tbptt_block}, ckpt_every={config.checkpoint_every}")
    print(f"  Params: {sum(p.numel() for p in mg.parameters())/1e6:.1f}M")
    print()

    dt = torch.bfloat16

    # Prepare inputs matching what forward_segment does
    h = torch.randn(bs, NC, Cn, Dn, device=device, dtype=dt)
    msg = torch.randn(bs, NC, Cn, Dn, device=device, dtype=dt)
    w_conn = torch.randn(bs, NC, Cn, K, device=device, dtype=dt)
    decay_logit = torch.zeros(bs, NC, Cn, device=device, dtype=dt)
    cell_context = torch.zeros(bs, NC, Dn, device=device, dtype=dt)
    border_gate_logit = torch.zeros(bs, NC, config.border_per_cell, device=device, dtype=dt)
    hebbian = torch.zeros(bs, NC, Cn, K, device=device, dtype=dt)
    H_aug_t = torch.randn(bs, D, device=device, dtype=dt)
    identity = mg.neuron_id.unsqueeze(0).expand(bs, -1, -1, -1).to(dt)

    group_idx = mg.cell_to_group
    st_w1 = mg.state_w1[group_idx].to(dt)
    st_b1 = mg.state_b1[group_idx].to(dt)
    st_w2 = mg.state_w2[group_idx].to(dt)
    st_b2 = mg.state_b2[group_idx].to(dt)
    mg_w1 = mg.msg_w1[group_idx].to(dt)
    mg_b1 = mg.msg_b1[group_idx].to(dt)
    mg_w2 = mg.msg_w2[group_idx].to(dt)
    mg_b2 = mg.msg_b2[group_idx].to(dt)
    inject_w = mg.inject_w[group_idx].to(dt)
    inject_b = mg.inject_b[group_idx].to(dt)
    mod_w1 = mg.mod_w1.to(dt)
    mod_b1 = mg.mod_b1.to(dt)
    mod_w2 = mg.mod_w2.to(dt)
    mod_b2 = mg.mod_b2.to(dt)

    w_conn_act = torch.sigmoid(w_conn)
    decay = torch.sigmoid(decay_logit).unsqueeze(-1)
    border_gate = torch.sigmoid(border_gate_logit).unsqueeze(-1)

    print("=" * 65)
    print(f"{'Operation':<40} {'Time (ms)':>10} {'Per-tok':>8}")
    print("=" * 65)

    # 1. Modulate cells
    t = time_op(lambda: mg._modulate_cells(
        h, msg, hebbian, decay_logit, cell_context, border_gate_logit, w_conn,
        mod_w1, mod_b1, mod_w2, mod_b2))
    print(f"{'Modulate cells':<40} {t:>10.3f} {t:>8.3f}")

    # 2. Local receive (Triton)
    t = time_op(lambda: mg._receive_local_activated(msg, w_conn_act))
    print(f"{'Local receive (Triton, activated)':<40} {t:>10.3f} {t:>8.3f}")

    # 2b. Local receive (eager baseline for comparison)
    def eager_receive():
        batch_idx = torch.arange(bs, device=device)[:, None, None, None]
        cell_idx = torch.arange(NC, device=device)[None, :, None, None]
        conn = mg.conn_idx.unsqueeze(0).expand(bs, -1, -1, -1)
        gathered = msg[batch_idx, cell_idx, conn]
        return (gathered * w_conn_act.unsqueeze(-1)).sum(dim=3)
    t_eager = time_op(eager_receive)
    print(f"{'Local receive (eager baseline)':<40} {t_eager:>10.3f} {t_eager:>8.3f}")

    # 3. Inject
    received = torch.randn(bs, NC, Cn, Dn, device=device, dtype=dt)
    t = time_op(lambda: mg._inject(received.clone(), H_aug_t, inject_w, inject_b))
    print(f"{'Inject':<40} {t:>10.3f} {t:>8.3f}")

    # 4. Border exchange (Triton)
    t = time_op(lambda: mg._border_exchange_from_gate(msg, border_gate))
    print(f"{'Border exchange (Triton)':<40} {t:>10.3f} {t:>8.3f}")

    # 4b. Border exchange (eager)
    t_eager = time_op(lambda: mg._border_exchange(msg, border_gate_logit))
    print(f"{'Border exchange (eager)':<40} {t_eager:>10.3f} {t_eager:>8.3f}")

    # 5. State update (grouped einsum)
    t = time_op(lambda: mg._state_update_from_decay(
        received, h, decay, identity, cell_context,
        st_w1, st_b1, st_w2, st_b2))
    print(f"{'State update (grouped einsum)':<40} {t:>10.3f} {t:>8.3f}")

    # 6. Emit message (grouped einsum)
    t = time_op(lambda: mg._emit_message(
        h, identity, cell_context,
        mg_w1, mg_b1, mg_w2, mg_b2))
    print(f"{'Emit message (grouped einsum)':<40} {t:>10.3f} {t:>8.3f}")

    # 7. Readout
    t = time_op(lambda: mg._readout(msg))
    print(f"{'Readout':<40} {t:>10.3f} {t:>8.3f}")

    # 8. Hebbian update (Triton)
    t = time_op(lambda: mg._hebbian_next(msg, msg, hebbian, mg.conn_idx, 0.995))
    print(f"{'Hebbian update (Triton)':<40} {t:>10.3f} {t:>8.3f}")

    # 9. Full _step
    t = time_op(lambda: mg._step(
        h, msg, w_conn, decay_logit, cell_context, border_gate_logit, hebbian,
        w_conn_act, decay, border_gate,
        H_aug_t, identity, inject_w, inject_b,
        st_w1, st_b1, st_w2, st_b2,
        mg_w1, mg_b1, mg_w2, mg_b2))
    print(f"{'Full _step':<40} {t:>10.3f} {t:>8.3f}")

    print("=" * 65)

    # Full forward_segment (no backward)
    H_aug_full = torch.randn(bs, T, D, device=device, dtype=dt)
    mg.detach_states()
    t_fwd = time_op(lambda: mg.forward_segment(H_aug_full), warmup=2, iters=5)
    print(f"{'forward_segment (eval, no ckpt)':<40} {t_fwd:>10.1f} {t_fwd/T:>8.3f}")

    # Full forward+backward
    mg.train()
    mg.detach_states()
    def fwd_bwd():
        mg.detach_states()
        out = mg.forward_segment(H_aug_full)
        out.sum().backward()
        for p in mg.parameters():
            if p.grad is not None:
                p.grad = None
    t_fb = time_op(fwd_bwd, warmup=2, iters=5)
    print(f"{'forward_segment + backward':<40} {t_fb:>10.1f} {t_fb/T:>8.3f}")

    tok_per_s_fwd = bs * T / (t_fwd / 1000)
    tok_per_s_fb = bs * T / (t_fb / 1000)
    print()
    print(f"Forward-only throughput: {tok_per_s_fwd:.0f} tok/s ({tok_per_s_fwd/1e3:.1f}K)")
    print(f"Fwd+Bwd throughput:      {tok_per_s_fb:.0f} tok/s ({tok_per_s_fb/1e3:.1f}K)")

    # Sum of individual ops × T vs measured full forward
    print()
    print("--- Theoretical vs Measured ---")

    # Re-measure components for a clean comparison
    ops = {}
    ops["receive"] = time_op(lambda: mg._receive_local_activated(msg, w_conn_act))
    ops["inject"] = time_op(lambda: mg._inject(received.clone(), H_aug_t, inject_w, inject_b))
    ops["border"] = time_op(lambda: mg._border_exchange_from_gate(msg, border_gate))
    ops["state_update"] = time_op(lambda: mg._state_update_from_decay(
        received, h, decay, identity, cell_context, st_w1, st_b1, st_w2, st_b2))
    ops["emit"] = time_op(lambda: mg._emit_message(
        h, identity, cell_context, mg_w1, mg_b1, mg_w2, mg_b2))
    ops["readout"] = time_op(lambda: mg._readout(msg))
    ops["hebbian"] = time_op(lambda: mg._hebbian_next(msg, msg, hebbian, mg.conn_idx, 0.995))
    ops["modulate"] = time_op(lambda: mg._modulate_cells(
        h, msg, hebbian, decay_logit, cell_context, border_gate_logit, w_conn,
        mod_w1, mod_b1, mod_w2, mod_b2))

    per_step_no_mod = sum(v for k, v in ops.items() if k != "modulate")
    mod_amortized = ops["modulate"] / config.modulation_interval
    per_step_total = per_step_no_mod + mod_amortized

    print(f"Sum of ops per step (no mod): {per_step_no_mod:.3f} ms")
    print(f"Modulate amortized (/{config.modulation_interval}): {mod_amortized:.3f} ms")
    print(f"Theoretical per step: {per_step_total:.3f} ms")
    print(f"Theoretical segment: {per_step_total * T:.1f} ms")
    print(f"Measured segment (fwd): {t_fwd:.1f} ms")
    print(f"Overhead: {t_fwd - per_step_total * T:.1f} ms ({(t_fwd / (per_step_total * T) - 1) * 100:.0f}%)")

    # Target analysis
    print()
    print("--- Path to 20K tok/s ---")
    target_toks = 20000
    budget_ms = bs * T / target_toks * 1000
    print(f"Budget for {target_toks} tok/s at BS={bs}: {budget_ms:.1f} ms per segment (fwd+bwd)")
    print(f"Current fwd+bwd: {t_fb:.1f} ms")
    print(f"Gap: {t_fb - budget_ms:.1f} ms ({t_fb / budget_ms:.1f}x over budget)")
    print()
    for name, ms in sorted(ops.items(), key=lambda x: -x[1]):
        pct = ms / per_step_total * 100
        total_contrib = ms * T if name != "modulate" else ms * T / config.modulation_interval
        print(f"  {name:<20} {ms:.3f} ms/step  {pct:5.1f}%  ({total_contrib:.0f} ms total)")


def profile_lm_overhead(bs=8):
    """Measure how much time the LM (scan + PCM + head) takes."""
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    model = Model(config).to(device)
    T = config.T
    input_ids = torch.randint(1, config.vocab_size, (bs, T), device=device)

    print()
    print("--- LM Component Timing ---")

    # Lower scan + PCM + augment
    t = time_op(lambda: model.lm.forward_scan_lower(input_ids), warmup=3, iters=10)
    print(f"{'LM lower scan + PCM':<40} {t:>10.3f} ms")

    with torch.no_grad():
        H_mid, surprise, _ = model.lm.forward_scan_lower(input_ids)
    t = time_op(lambda: model.lm.augment(H_mid, surprise), warmup=3, iters=10)
    print(f"{'Augment (split_mlp)':<40} {t:>10.3f} ms")

    H_aug = model.lm.augment(H_mid, surprise)
    t = time_op(lambda: model.lm.forward_scan_upper(H_aug), warmup=3, iters=10)
    print(f"{'LM upper scan':<40} {t:>10.3f} ms")

    with torch.no_grad():
        H = model.lm.forward_scan_upper(H_aug)
    t = time_op(lambda: model.lm.forward_output(H), warmup=3, iters=10)
    print(f"{'LM head (proj_down + ln + linear)':<40} {t:>10.3f} ms")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--bs", type=int, default=8)
    args = p.parse_args()

    profile_memory_graph(bs=args.bs)
    profile_lm_overhead(bs=args.bs)
