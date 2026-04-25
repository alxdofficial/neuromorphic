"""Bench: compile the whole TBPTT block as one function.

Compiling _step_core_pure per-step gives 1.92x (default+fg=T). Compiling
the whole T_block-step block as one function should let inductor capture
the entire forward as one cudagraph (under reduce-overhead) — should
unlock significant additional speedup.

Bench setup matches bench_walker_compile_modes.py: B=4, mod_period=64,
realistic mid-size config. Compares eager / default+fg=T (per-step) /
reduce-overhead+fg=T (whole block).
"""

from __future__ import annotations

import time

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def make_lm() -> StandaloneLM:
    cfg = GraphWalkerConfig(
        plane_rows=8, plane_cols=8, L=4, K=16,
        D_model=512, D_s=256, D_id=32,
        n_heads=4, n_score_heads=4, D_q_per_head=32, D_q_in=32,
        K_horizons=4, K_buf=4,
        mod_period=64, tbptt_block=64, segment_T=64,
        ffn_mult_content=4,
        content_mlp_depth=2,
        post_model_depth=1,
        vocab_size=1024,
        use_neuromod=False,
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def block_forward(
    lm: StandaloneLM,
    tokens_block: torch.Tensor,
    s_in: torch.Tensor,
    walker_pos_in: torch.Tensor,
    walker_state_in: torch.Tensor,
    prev_motor_in: torch.Tensor,
    e_bias_in: torch.Tensor,
    tau: torch.Tensor,
    epsilon: torch.Tensor,
):
    B, T_block = tokens_block.shape
    motor_list = []
    s = s_in
    walker_pos = walker_pos_in
    walker_state = walker_state_in
    prev_motor = prev_motor_in
    for t in range(T_block):
        is_new_window = (t == 0)
        out = lm.memory._step_core_pure(
            s, walker_pos, walker_state, prev_motor,
            e_bias_in,
            tokens_block[:, t], tau, epsilon, is_new_window,
        )
        s = out.s_new
        walker_pos = out.walker_pos_new
        walker_state = out.walker_state_new
        prev_motor = out.prev_motor_new
        motor_list.append(out.motor_state)
    motor_states_bt = torch.stack(motor_list, dim=1)
    return motor_states_bt, s, walker_pos, walker_state, prev_motor


def time_segment(lm: StandaloneLM, B: int, n_iters: int,
                 amp_dtype: torch.dtype,
                 mode: str | None,
                 fullgraph: bool) -> float:
    cfg = lm.cfg
    device = next(lm.parameters()).device
    T = cfg.mod_period

    if mode is not None:
        torch._dynamo.config.capture_dynamic_output_shape_ops = True
        torch._dynamo.config.allow_unspec_int_on_nn_module = True
        torch._dynamo.config.cache_size_limit = max(
            torch._dynamo.config.cache_size_limit, 64,
        )
        block_fn = torch.compile(
            lambda *args: block_forward(lm, *args),
            mode=mode, fullgraph=fullgraph,
        )
    else:
        block_fn = lambda *args: block_forward(lm, *args)

    tokens_block = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    def run_one_segment() -> None:
        lm.memory.begin_segment(B, device)
        lm.memory._ensure_block_caches(lm.memory.tied_token_emb.weight)
        e_bias = lm.memory._active_e_bias()
        tau = torch.tensor(2.0, device=device, dtype=torch.float32)
        eps = torch.tensor(0.05, device=device, dtype=torch.float32)
        if mode == "reduce-overhead":
            torch.compiler.cudagraph_mark_step_begin()
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            motor_bt, s_new, wp_new, ws_new, pm_new = block_fn(
                tokens_block, lm.memory.s, lm.memory.walker_pos,
                lm.memory.walker_state, lm.memory.prev_motor, e_bias,
                tau, eps,
            )
            loss = motor_bt.float().pow(2).mean()
        loss.backward()

    # Warmup
    for _ in range(3):
        run_one_segment()
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        run_one_segment()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return n_iters * B * T / elapsed


def main() -> None:
    if not torch.cuda.is_available():
        return

    B = 4
    N_ITERS = 5

    configs = [
        ("eager",                         None,              False),
        ("default+fg=T (whole block)",    "default",         True),
        ("reduce-overhead+fg=T (block)",  "reduce-overhead", True),
    ]

    print(f"\n=== Walker block-compile bench (B={B}, T=64, mod_period=64) ===\n")
    print(f"{'config':<35} {'tok/s':>10} {'rel':>8}")
    print("-" * 55)
    baseline = None
    for label, mode, fullgraph in configs:
        lm = make_lm()
        try:
            tps = time_segment(lm, B, N_ITERS, torch.bfloat16, mode, fullgraph)
        except Exception as e:
            print(f"{label:<35} FAILED: {type(e).__name__}: {str(e)[:60]}")
            continue
        if baseline is None:
            baseline = tps
        print(f"{label:<35} {tps:>10.1f} {tps/baseline:>7.2f}x")
        del lm
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
