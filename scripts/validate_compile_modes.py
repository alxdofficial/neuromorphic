"""Validate torch.compile modes on the walker hot path.

Runs a tiny GraphWalkerMemory through several compile configurations and
reports which work end-to-end (forward + backward under bf16 autocast).
Target: discover what dynamo blocks on so the Phase A refactor (docs/
plan_walker_speedup.md) can target real failures, not theoretical ones.

Configurations tested (in order of strictness):

  baseline       : eager, no compile
  default+ng     : mode=default, fullgraph=False (= current production)
  default+fg     : mode=default, fullgraph=True   (catches Python branches)
  ro+ng          : mode=reduce-overhead, fullgraph=False (CUDA graphs)
  ro+fg          : mode=reduce-overhead, fullgraph=True  (target endpoint)

Each runs:
  1. begin_segment
  2. 5 forward token steps (mix of anchor + interior)
  3. backward on a synthetic loss

Output: status table + first error stack trace if any.
"""

from __future__ import annotations

import sys
import traceback

import torch

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.standalone import StandaloneLM


def make_tiny_lm() -> StandaloneLM:
    """Tiny config that exercises the full hot path but compiles fast."""
    cfg = GraphWalkerConfig(
        plane_rows=4, plane_cols=4, L=2, K=4,
        D_model=64, D_s=32, D_id=16,
        n_heads=2, n_score_heads=2, D_q_per_head=16, D_q_in=16,
        K_horizons=2, K_buf=2,
        mod_period=4, tbptt_block=4,
        segment_T=8,
        ffn_mult_content=2,
        content_mlp_depth=1,
        post_model_depth=1,
        vocab_size=128,
        use_neuromod=False,                # simpler — we'll re-enable if base works
        compile_on_train=False,            # we drive compile manually
        state_dtype="bf16",
    )
    return StandaloneLM(cfg).cuda()


def run_one(label: str, lm: StandaloneLM, mode: str | None,
            fullgraph: bool) -> tuple[bool, str]:
    """Run a 5-step forward+backward under one compile config.

    Returns (success, error_msg). Empty error_msg on success.
    """
    cfg = lm.cfg
    device = next(lm.parameters()).device
    B, T = 2, cfg.segment_T

    # Reset compiled-step cache so we test the requested mode fresh.
    lm.memory._compiled_step = None
    if mode is not None:
        try:
            # Mirror compile_step but allow fullgraph override.
            torch._dynamo.config.capture_dynamic_output_shape_ops = True
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            torch._dynamo.config.cache_size_limit = max(
                torch._dynamo.config.cache_size_limit, 64,
            )
            lm.memory._compiled_step = torch.compile(
                lm.memory._step_core_pure, mode=mode, fullgraph=fullgraph,
            )
        except Exception as e:
            return False, f"compile() raised: {e}"

    # Synthetic input.
    tokens = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    try:
        lm.memory.begin_segment(B, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            losses = []
            for t in range(5):
                r = lm.memory.step_core(tokens[:, t])
                losses.append(r.motor_state.float().pow(2).mean())
            loss = torch.stack(losses).sum()
        loss.backward()
        torch.cuda.synchronize()
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def run_make_graphed(lm: StandaloneLM) -> tuple[bool, str]:
    """Wrap _step_core_pure with torch.cuda.make_graphed_callables.

    Sample inputs must be in stable buffers across replays. We capture
    one variant for is_new_window=True (anchor) and one for False
    (interior). Each capture needs warmup (default 11 iters in 2.10).
    """
    cfg = lm.cfg
    device = next(lm.parameters()).device
    B = 2

    try:
        lm.memory.begin_segment(B, device)
        # Snapshot active e_bias and schedule tensors as stable inputs.
        e_bias = lm.memory._active_e_bias()
        tau = torch.tensor(2.0, device=device, dtype=torch.float32)
        eps = torch.tensor(0.05, device=device, dtype=torch.float32)
        token_id = torch.zeros(B, device=device, dtype=torch.long)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            sample_anchor = (
                lm.memory.s, lm.memory.walker_pos, lm.memory.walker_state,
                lm.memory.prev_motor, e_bias,
                token_id, tau, eps, True,
            )
            sample_interior = (
                lm.memory.s, lm.memory.walker_pos, lm.memory.walker_state,
                lm.memory.prev_motor, e_bias,
                token_id, tau, eps, False,
            )
            # NOTE: make_graphed_callables doesn't take Python bools as args
            # in newer torch — it expects only Tensor inputs. We'll hit that
            # constraint here and need to refactor away the bool first.
            anchor_callable = torch.cuda.make_graphed_callables(
                lm.memory._step_core_pure,
                sample_anchor,
                num_warmup_iters=3,
            )
            anchor_callable(*sample_anchor)
        return True, ""
    except Exception:
        return False, traceback.format_exc()


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA required.")
        return 1

    configs = [
        ("baseline   (eager)",            None,              False),
        ("default+ng (default,no-fg)",    "default",         False),
        ("default+fg (default,fullgraph)","default",         True),
        ("ro+ng      (reduce-oh,no-fg)",  "reduce-overhead", False),
        ("ro+fg      (reduce-oh,fullgr)", "reduce-overhead", True),
    ]

    results: list[tuple[str, bool, str]] = []
    for label, mode, fullgraph in configs:
        # Build a fresh LM each iteration so state buffers are clean.
        lm = make_tiny_lm()
        ok, err = run_one(label, lm, mode, fullgraph)
        results.append((label, ok, err))
        # Free GPU memory between trials.
        del lm
        torch.cuda.empty_cache()

    # Extra: try reduce-overhead with the storage-pool check disabled.
    # Inductor's cudagraph_trees enforces a memory-pool invariant that custom
    # autograd.Functions (like SparseLIFUpdate) violate. Skip the check and
    # see if execution still works.
    print("\n--- Trying reduce-overhead with cudagraph workarounds ---")
    triton_cfg = torch._inductor.config.triton
    extra_configs: list[tuple[str, dict]] = [
        ("ro+fg, triton.cudagraph_skip_dynamic_graphs=True",
            {"cudagraph_skip_dynamic_graphs": True}),
        ("ro+fg, triton.cudagraph_support_input_mutation=True",
            {"cudagraph_support_input_mutation": True}),
        ("ro+fg, both",
            {"cudagraph_skip_dynamic_graphs": True,
             "cudagraph_support_input_mutation": True}),
    ]
    for label, overrides in extra_configs:
        for k, v in overrides.items():
            setattr(triton_cfg, k, v)
        try:
            lm = make_tiny_lm()
            ok, err = run_one(label, lm, "reduce-overhead", True)
            results.append((label, ok, err))
            del lm
            torch.cuda.empty_cache()
        finally:
            for k, _v in overrides.items():
                setattr(triton_cfg, k, False)

    # Try make_graphed_callables — explicit CUDA graph wrap of the step func.
    # This bypasses inductor's cudagraph_trees and gives us direct control.
    print("\n--- Trying make_graphed_callables ---")
    lm = make_tiny_lm()
    ok, err = run_make_graphed(lm)
    results.append(("make_graphed_callables (anchor+interior)", ok, err))
    del lm
    torch.cuda.empty_cache()

    # Summary.
    print("\n=== Compile-mode validation ===")
    for label, ok, _err in results:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {label}")
    print()

    # Show first failure's traceback so we can act on it.
    for label, ok, err in results:
        if not ok:
            print(f"--- First failure: {label} ---")
            print(err)
            break

    return 0 if all(ok for _, ok, _ in results) else 1


if __name__ == "__main__":
    sys.exit(main())
