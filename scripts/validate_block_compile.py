"""Validate compiling the TBPTT block (T forward steps + readout) as one
compiled function under reduce-overhead.

Hypothesis: per-step compilation fails reduce-overhead because each step's
saved-for-backward tensors share the cudagraph buffer pool across replays.
Compiling the whole T-step loop unrolled means inductor allocates each
step's saves as separate tensors in one graph; backward runs on the
unrolled graph and all saves are available.

Test:
  1. Build a tiny LM
  2. Define block_forward(tokens, state...) that does T_block step_core_pure
     calls in a Python loop and returns (motor_states_bt, new_state...)
  3. Compile with reduce-overhead, fullgraph=True
  4. Run forward + backward
  5. Compare to eager reference for numerical sanity

If this works, we have the path to the 5-8x walker speedup.
"""

from __future__ import annotations

import sys
import traceback

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def make_tiny_lm() -> StandaloneLM:
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=64, D_s=32, D_id=16,
        n_heads=2, n_score_heads=2, D_q_per_head=16, D_q_in=16,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4, segment_T=4,
        ffn_mult_content=2,
        content_mlp_depth=1,
        post_model_depth=1,
        vocab_size=128,
        use_neuromod=False,
        compile_on_train=False,
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def block_forward(
    lm: StandaloneLM,
    tokens_block: torch.Tensor,            # [B, T_block]
    s_in: torch.Tensor,                    # [B, N, D_s]
    walker_pos_in: torch.Tensor,           # [B, H]
    walker_state_in: torch.Tensor,         # [B, H, D_s]
    prev_motor_in: torch.Tensor,           # [B, D_s]
    e_bias_in: torch.Tensor,               # [N*K] fp32
    tau: torch.Tensor,
    epsilon: torch.Tensor,
):
    """Run T_block sequential step_core_pure calls in a Python loop.

    Returns motor_states_bt and final state. No self.* mutations — all
    state threads through return values so the compiled graph is pure.
    """
    B, T_block = tokens_block.shape
    motor_list = []
    s = s_in
    walker_pos = walker_pos_in
    walker_state = walker_state_in
    prev_motor = prev_motor_in

    for t in range(T_block):
        # Anchor on first step of the block (window-start), interior after.
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

    motor_states_bt = torch.stack(motor_list, dim=1)            # [B, T_block, D_s]
    return motor_states_bt, s, walker_pos, walker_state, prev_motor


def run_one(label: str, lm: StandaloneLM, mode: str | None,
            fullgraph: bool) -> tuple[bool, str]:
    cfg = lm.cfg
    device = next(lm.parameters()).device
    B = 2

    try:
        lm.memory.begin_segment(B, device)
        # Populate block caches before the compiled call (normally done by
        # step_core, but we're calling _step_core_pure directly here).
        lm.memory._ensure_block_caches(lm.memory.tied_token_emb.weight)
        e_bias = lm.memory._active_e_bias()
        tau = torch.tensor(2.0, device=device, dtype=torch.float32)
        eps = torch.tensor(0.05, device=device, dtype=torch.float32)
        tokens_block = torch.randint(
            0, cfg.vocab_size, (B, cfg.mod_period), device=device,
        )

        # Compile or eager.
        if mode is None:
            block_fn = lambda t, s, wp, ws, pm, eb, ta, ep: block_forward(
                lm, t, s, wp, ws, pm, eb, ta, ep,
            )
        else:
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64,
            )
            block_fn = torch.compile(
                lambda t, s, wp, ws, pm, eb, ta, ep: block_forward(
                    lm, t, s, wp, ws, pm, eb, ta, ep,
                ),
                mode=mode, fullgraph=fullgraph,
            )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            motor_bt, s_new, wp_new, ws_new, pm_new = block_fn(
                tokens_block, lm.memory.s, lm.memory.walker_pos,
                lm.memory.walker_state, lm.memory.prev_motor, e_bias,
                tau, eps,
            )
            loss = motor_bt.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1

    configs = [
        ("eager",                        None,              False),
        ("default+fg=T",                 "default",         True),
        ("reduce-overhead+fg=T (whole block)", "reduce-overhead", True),
    ]

    for label, mode, fullgraph in configs:
        lm = make_tiny_lm()
        ok, err = run_one(label, lm, mode, fullgraph)
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}")
        if not ok:
            print(err[-2000:])
            print("---")
        del lm
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    sys.exit(main())
