"""Benchmark: fuse receive + inject + border + cat into one kernel vs separate ops."""

import sys
import time
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.memory import MemoryGraph


def time_op(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def main():
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    config = Config.tier_a(compile_step=False)
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(16, device)

    NC, Cn, Dn, K, D = config.N_cells, config.neurons_per_cell, config.D_n, config.K, config.D
    dt = torch.bfloat16
    bs = 16

    h = torch.randn(bs, NC, Cn, Dn, device=device, dtype=dt)
    msg = torch.randn(bs, NC, Cn, Dn, device=device, dtype=dt)
    w_conn_act = torch.sigmoid(torch.randn(bs, NC, Cn, K, device=device, dtype=dt))
    decay = torch.sigmoid(torch.zeros(bs, NC, Cn, device=device, dtype=dt)).unsqueeze(-1)
    cell_context = torch.zeros(bs, NC, Dn, device=device, dtype=dt)
    identity = mg.neuron_id.unsqueeze(0).expand(bs, -1, -1, -1).to(dt)
    H_aug_t = torch.randn(bs, D, device=device, dtype=dt)

    gi = mg.cell_to_group
    inject_w = mg.inject_w[gi].to(dt)
    inject_b = mg.inject_b[gi].to(dt)
    border_gate = torch.sigmoid(torch.zeros(bs, NC, 4, device=device, dtype=dt)).unsqueeze(-1)

    st_w1 = mg.state_w1.to(dt)
    st_b1 = mg.state_b1.to(dt)
    st_gs1 = mg.state_gs1[gi].to(dt)
    st_gb1 = mg.state_gb1[gi].to(dt)
    st_w2 = mg.state_w2.to(dt)
    st_b2 = mg.state_b2.to(dt)
    st_gs2 = mg.state_gs2[gi].to(dt)
    st_gb2 = mg.state_gb2[gi].to(dt)

    # Current approach: separate receive + inject + border + state_update
    def current_approach():
        received = mg._receive_local_activated(msg, w_conn_act)
        received = mg._inject(received, H_aug_t, inject_w, inject_b)
        received[:, :, mg.border_lo:mg.border_hi] += mg._border_exchange_from_gate(msg, border_gate)
        return mg._state_update_from_decay(
            received, h, decay, identity, cell_context,
            st_w1, st_b1, st_gs1, st_gb1, st_w2, st_b2, st_gs2, st_gb2)

    # Alternative: pre-build the full state_input in one shot
    # This avoids: 1 receive output tensor, 1 inject clone, 1 border add, 1 cat in state_update
    def fused_approach():
        # Still use Triton receive
        received = mg._receive_local_activated(msg, w_conn_act)
        # Inline inject (no clone, direct add)
        cell_slice = H_aug_t.reshape(bs, NC, Dn)
        inject = torch.einsum("bni,noi->bno", cell_slice, inject_w) + inject_b.unsqueeze(0)
        inject = inject.reshape(bs, NC, config.alpha, Dn)
        received[:, :, mg.input_lo:mg.input_hi] += inject
        # Inline border
        received[:, :, mg.border_lo:mg.border_hi] += mg._border_exchange_from_gate(msg, border_gate)

        # Build full state_input directly and run MLP
        ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
        state_input = torch.cat([received, h, identity, ctx, decay], dim=-1)
        flat = state_input.reshape(-1, state_input.shape[-1])
        hidden = torch.tanh(F.linear(flat, st_w1, st_b1))
        hidden = hidden.reshape(bs, NC, Cn, -1) * st_gs1.unsqueeze(0).unsqueeze(2) + st_gb1.unsqueeze(0).unsqueeze(2)
        hidden = torch.tanh(hidden)
        flat2 = hidden.reshape(-1, hidden.shape[-1])
        out = torch.tanh(F.linear(flat2, st_w2, st_b2))
        out = out.reshape(bs, NC, Cn, -1) * st_gs2.unsqueeze(0).unsqueeze(2) + st_gb2.unsqueeze(0).unsqueeze(2)
        candidate = torch.tanh(out)
        return decay * h + (1.0 - decay) * candidate

    t_cur = time_op(current_approach)
    t_fused = time_op(fused_approach)

    print(f"Current (receive + inject + border + state_update): {t_cur:.3f} ms")
    print(f"Fused (inline inject/border, manual MLP): {t_fused:.3f} ms")
    print(f"Speedup: {t_cur/t_fused:.2f}x")

    # Compile both
    print("\nWith torch.compile:")
    cur_c = torch.compile(current_approach, mode="default")
    fus_c = torch.compile(fused_approach, mode="default")
    for _ in range(5):
        cur_c()
        fus_c()
    t_cur_c = time_op(cur_c)
    t_fus_c = time_op(fus_c)
    print(f"Current compiled: {t_cur_c:.3f} ms")
    print(f"Fused compiled:   {t_fus_c:.3f} ms")
    print(f"Speedup: {t_cur_c/t_fus_c:.2f}x")


if __name__ == "__main__":
    main()
