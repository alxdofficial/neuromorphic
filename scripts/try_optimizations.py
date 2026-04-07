"""Try various GPU optimization approaches and report results."""

import sys
import time
import os
import torch
import torch.nn.functional as F
import functools

sys.path.insert(0, ".")
from src.model.config import Config
from src.model.memory import MemoryGraph


def time_segment(mg, H_aug, warmup=3, iters=5, train=True):
    """Time forward_segment (and backward if train=True). Returns avg ms."""
    mg.train(train)
    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for _ in range(warmup):
            mg.detach_states()
            out = mg.forward_segment(H_aug)
            if train:
                out.sum().backward()
                for p in mg.parameters():
                    if p.grad is not None:
                        p.grad = None
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        mg.detach_states()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with ctx:
            out = mg.forward_segment(H_aug)
            if train:
                out.sum().backward()
                for p in mg.parameters():
                    if p.grad is not None:
                        p.grad = None
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return sum(times) / len(times)


def baseline(bs=8):
    """Current implementation, no changes."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)
    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    t_eval = time_segment(mg, H_aug, train=False)
    t_train = time_segment(mg, H_aug, train=True)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_torch_compile_step(bs=8):
    """Compile the _step method."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)
    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    # Compile _step
    mg._step = torch.compile(mg._step, mode="reduce-overhead", fullgraph=False)

    # Extra warmup for compilation
    t_eval = time_segment(mg, H_aug, train=False, warmup=5)
    t_train = time_segment(mg, H_aug, train=True, warmup=5)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_torch_compile_run_block(bs=8):
    """Compile the _run_block method."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)
    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    mg._run_block = torch.compile(mg._run_block, mode="reduce-overhead", fullgraph=False)

    t_eval = time_segment(mg, H_aug, train=False, warmup=5)
    t_train = time_segment(mg, H_aug, train=True, warmup=5)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_torch_compile_default(bs=8):
    """Compile _step with default mode (max fusion, no CUDA graphs)."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)
    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    mg._step = torch.compile(mg._step, mode="default", fullgraph=False)

    t_eval = time_segment(mg, H_aug, train=False, warmup=5)
    t_train = time_segment(mg, H_aug, train=True, warmup=5)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_bmm_instead_of_einsum(bs=8):
    """Replace einsum with explicit bmm for grouped MLPs."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)

    NC = config.N_cells
    Cn = config.neurons_per_cell
    Dn = config.D_n

    # Monkey-patch _state_update_from_decay to use bmm
    orig_state = mg._state_update_from_decay

    def bmm_state_update(received, h, decay, identity, cell_context, w1, b1, w2, b2):
        BS = received.shape[0]
        ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
        state_input = torch.cat([received, h, identity, ctx, decay], dim=-1)
        # einsum('bnci,nhi->bnch') is: for each cell n, matmul [BS*Cn, I] @ [I, H]
        # With grouped weights [NC, H, I], this is batch matmul
        flat = state_input.reshape(BS * NC, Cn, -1)  # [BS*NC, Cn, I]
        w1_exp = w1.unsqueeze(0).expand(BS, -1, -1, -1).reshape(BS * NC, w1.shape[1], w1.shape[2])
        hidden = torch.tanh(torch.bmm(flat, w1_exp.transpose(-1, -2)) + b1.unsqueeze(0).expand(BS, -1, -1).reshape(BS * NC, 1, -1))
        w2_exp = w2.unsqueeze(0).expand(BS, -1, -1, -1).reshape(BS * NC, w2.shape[1], w2.shape[2])
        candidate = torch.tanh(torch.bmm(hidden, w2_exp.transpose(-1, -2)) + b2.unsqueeze(0).expand(BS, -1, -1).reshape(BS * NC, 1, -1))
        candidate = candidate.reshape(BS, NC, Cn, Dn)
        return decay * h + (1.0 - decay) * candidate

    mg._state_update_from_decay = bmm_state_update

    # Same for emit
    def bmm_emit(h, identity, cell_context, w1, b1, w2, b2):
        BS = h.shape[0]
        ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
        msg_input = torch.cat([h, identity, ctx], dim=-1)
        flat = msg_input.reshape(BS * NC, Cn, -1)
        w1_exp = w1.unsqueeze(0).expand(BS, -1, -1, -1).reshape(BS * NC, w1.shape[1], w1.shape[2])
        hidden = torch.tanh(torch.bmm(flat, w1_exp.transpose(-1, -2)) + b1.unsqueeze(0).expand(BS, -1, -1).reshape(BS * NC, 1, -1))
        w2_exp = w2.unsqueeze(0).expand(BS, -1, -1, -1).reshape(BS * NC, w2.shape[1], w2.shape[2])
        msg_new = torch.tanh(torch.bmm(hidden, w2_exp.transpose(-1, -2)) + b2.unsqueeze(0).expand(BS, -1, -1).reshape(BS * NC, 1, -1))
        msg_new = msg_new.reshape(BS, NC, Cn, Dn)
        return msg_new + identity

    mg._emit_message = bmm_emit

    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    t_eval = time_segment(mg, H_aug, train=False)
    t_train = time_segment(mg, H_aug, train=True)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_flat_linear(bs=8):
    """Replace grouped einsum with flat F.linear (one shared MLP, ignore groups)."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    mg = MemoryGraph(config).to(device)
    mg.initialize_states(bs, device)

    NC = config.N_cells
    Cn = config.neurons_per_cell
    Dn = config.D_n
    Hs = config.state_mlp_hidden
    Hm = config.msg_mlp_hidden

    # Use only group 0's weights as a single shared MLP
    sw1 = mg.state_w1[0].to(torch.bfloat16)
    sb1 = mg.state_b1[0].to(torch.bfloat16)
    sw2 = mg.state_w2[0].to(torch.bfloat16)
    sb2 = mg.state_b2[0].to(torch.bfloat16)
    mw1 = mg.msg_w1[0].to(torch.bfloat16)
    mb1 = mg.msg_b1[0].to(torch.bfloat16)
    mw2 = mg.msg_w2[0].to(torch.bfloat16)
    mb2 = mg.msg_b2[0].to(torch.bfloat16)

    def flat_state_update(received, h, decay, identity, cell_context, w1, b1, w2, b2):
        BS = received.shape[0]
        ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
        state_input = torch.cat([received, h, identity, ctx, decay], dim=-1)
        flat = state_input.reshape(-1, state_input.shape[-1])
        hidden = torch.tanh(F.linear(flat, sw1, sb1))
        candidate = torch.tanh(F.linear(hidden, sw2, sb2))
        candidate = candidate.reshape(BS, NC, Cn, Dn)
        return decay * h + (1.0 - decay) * candidate

    def flat_emit(h, identity, cell_context, w1, b1, w2, b2):
        BS = h.shape[0]
        ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
        msg_input = torch.cat([h, identity, ctx], dim=-1)
        flat = msg_input.reshape(-1, msg_input.shape[-1])
        hidden = torch.tanh(F.linear(flat, mw1, mb1))
        msg_new = torch.tanh(F.linear(hidden, mw2, mb2))
        msg_new = msg_new.reshape(BS, NC, Cn, Dn)
        return msg_new + identity

    mg._state_update_from_decay = flat_state_update
    mg._emit_message = flat_emit

    H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

    t_eval = time_segment(mg, H_aug, train=False)
    t_train = time_segment(mg, H_aug, train=True)
    tok_eval = bs * config.T / (t_eval / 1000)
    tok_train = bs * config.T / (t_train / 1000)
    return t_eval, t_train, tok_eval, tok_train


def try_larger_bs():
    """Test scaling across batch sizes."""
    device = torch.device("cuda")
    config = Config.tier_a()
    config.vocab_size = 32000
    config.validate()

    results = []
    for bs in [8, 16, 24, 32, 48]:
        torch.cuda.empty_cache()
        mg = MemoryGraph(config).to(device)
        mg.initialize_states(bs, device)
        H_aug = torch.randn(bs, config.T, config.D, device=device, dtype=torch.bfloat16)

        try:
            t_train = time_segment(mg, H_aug, train=True, warmup=2, iters=3)
            tok = bs * config.T / (t_train / 1000)
            peak = torch.cuda.max_memory_allocated() / 1e9
            results.append((bs, t_train, tok, peak))
        except torch.cuda.OutOfMemoryError:
            results.append((bs, -1, -1, -1))
            torch.cuda.empty_cache()
        del mg
    return results


def main():
    torch.set_float32_matmul_precision("high")
    print("=" * 70)
    print("OPTIMIZATION SWEEP")
    print("=" * 70)

    # Baseline
    print("\n[1/6] Baseline (current code)...")
    t_e, t_t, tok_e, tok_t = baseline()
    print(f"  Eval:  {t_e:.0f} ms, {tok_e:.0f} tok/s ({tok_e/1e3:.1f}K)")
    print(f"  Train: {t_t:.0f} ms, {tok_t:.0f} tok/s ({tok_t/1e3:.1f}K)")
    base_train = tok_t

    # torch.compile _step (reduce-overhead)
    print("\n[2/6] torch.compile(_step, mode='reduce-overhead')...")
    try:
        torch._dynamo.reset()
        t_e, t_t, tok_e, tok_t = try_torch_compile_step()
        print(f"  Eval:  {t_e:.0f} ms, {tok_e:.0f} tok/s ({tok_e/1e3:.1f}K)")
        print(f"  Train: {t_t:.0f} ms, {tok_t:.0f} tok/s ({tok_t/1e3:.1f}K)")
        print(f"  Speedup: {tok_t/base_train:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")

    # torch.compile _step (default)
    print("\n[3/6] torch.compile(_step, mode='default')...")
    try:
        torch._dynamo.reset()
        t_e, t_t, tok_e, tok_t = try_torch_compile_default()
        print(f"  Eval:  {t_e:.0f} ms, {tok_e:.0f} tok/s ({tok_e/1e3:.1f}K)")
        print(f"  Train: {t_t:.0f} ms, {tok_t:.0f} tok/s ({tok_t/1e3:.1f}K)")
        print(f"  Speedup: {tok_t/base_train:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")

    # bmm instead of einsum
    print("\n[4/6] BMM instead of einsum for grouped MLPs...")
    try:
        torch._dynamo.reset()
        t_e, t_t, tok_e, tok_t = try_bmm_instead_of_einsum()
        print(f"  Eval:  {t_e:.0f} ms, {tok_e:.0f} tok/s ({tok_e/1e3:.1f}K)")
        print(f"  Train: {t_t:.0f} ms, {tok_t:.0f} tok/s ({tok_t/1e3:.1f}K)")
        print(f"  Speedup: {tok_t/base_train:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Flat F.linear (ignore groups, single shared MLP)
    print("\n[5/6] Flat F.linear (single shared MLP, ignore groups)...")
    try:
        torch._dynamo.reset()
        t_e, t_t, tok_e, tok_t = try_flat_linear()
        print(f"  Eval:  {t_e:.0f} ms, {tok_e:.0f} tok/s ({tok_e/1e3:.1f}K)")
        print(f"  Train: {t_t:.0f} ms, {tok_t:.0f} tok/s ({tok_t/1e3:.1f}K)")
        print(f"  Speedup: {tok_t/base_train:.2f}x")
    except Exception as e:
        print(f"  FAILED: {e}")

    # Batch size scaling
    print("\n[6/6] Batch size scaling (baseline code)...")
    torch._dynamo.reset()
    results = try_larger_bs()
    for bs, t, tok, peak in results:
        if t > 0:
            print(f"  BS={bs:2d}: {t:6.0f} ms, {tok:6.0f} tok/s ({tok/1e3:.1f}K), peak={peak:.2f} GB")
        else:
            print(f"  BS={bs:2d}: OOM")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)


if __name__ == "__main__":
    main()
