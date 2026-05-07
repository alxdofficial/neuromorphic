"""Phase 1 bench — teacher-forced parallel SFT throughput.

Runs four paths back-to-back:
  A) vanilla Llama forward (no_grad)
  B) vanilla Llama forward + lm_head-only training step
  C) vanilla Llama full training step (all params trainable)
  D) frozen Llama + GraphWalker phase1_pretrained_step

Walker config defaults to the on-main production shape (~25M trainable).
Pass `--target-config` to load the ~110M target config; pass individual
knobs to override either preset for sweeps.

Examples:
  # Production-shape Phase-1 bench at BS=16
  PYTHONPATH=. python scripts/bench_phase1.py --bs 16 --T 256 --compile-block

  # Target-config Phase-1 bench, eager mode (cudagraph compile breaks today)
  PYTHONPATH=. python scripts/bench_phase1.py --target-config --bs 1 --T 256
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

sys.path.insert(0, str(Path(__file__).parent))
from _bench_common import (  # noqa: E402
    add_walker_config_args, bench, cleanup_cuda,
    print_config_summary, walker_cfg_from_args,
)

from src.graph_walker.pretrained.config import PretrainedGWConfig  # noqa: E402
from src.graph_walker.pretrained.integrated_lm import IntegratedLM  # noqa: E402
from src.graph_walker.pretrained.train_phase1 import (  # noqa: E402
    Phase1Batch, phase1_pretrained_step,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--bs", type=int, default=20,
                    help="Batch size. Default 20 = production max-BS on 4090 "
                         "@ T=256 with compile-block (9.6k tok/s, 18.5 GB peak).")
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iter", type=int, default=10)
    ap.add_argument("--skip-vanilla-full", action="store_true",
                    help="Skip path C (full hot-Llama training). 23 GB at "
                         "BS=16; skip on smaller GPUs or when irrelevant.")
    ap.add_argument("--skip-gw", action="store_true",
                    help="Skip path D (the GW step). Useful for "
                         "vanilla-only baseline runs.")
    add_walker_config_args(ap)
    args = ap.parse_args()

    device = torch.device("cuda")
    BS, T = args.bs, args.T
    print(f"\n=== Phase 1 bench (parallel teacher-forced) ===")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  model:  {args.model}")
    print(f"  BS={BS}, T={T}, warmup={args.warmup}, iter={args.iter}")
    print()

    # ------------------------------------------------------------------
    # Load Llama once; reused across paths A/B/C.
    # ------------------------------------------------------------------
    print("Loading vanilla Llama...")
    llama = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    llama.train(False)
    for p in llama.parameters():
        p.requires_grad = False
    d_lm = llama.config.hidden_size
    n_layers = llama.config.num_hidden_layers
    vocab = llama.config.vocab_size
    n_params = sum(p.numel() for p in llama.parameters())
    print(f"  d_lm={d_lm}, n_layers={n_layers}, vocab={vocab}, "
          f"params={n_params/1e9:.2f}B")
    print()

    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    # ----- Path A: vanilla Llama forward, no_grad -----
    print("[A] Vanilla Llama, forward-only (no_grad)")
    def vanilla_fwd():
        with torch.no_grad():
            return llama(input_ids).logits
    tps_a, mem_a, _ = bench(
        "vanilla Llama fwd (no_grad)", vanilla_fwd,
        args.warmup, args.iter, BS, T,
    )

    # ----- Path B: vanilla Llama, lm_head-only training step -----
    for p in llama.lm_head.parameters():
        p.requires_grad = True
    llama.train(True)
    opt_b = torch.optim.AdamW(
        [p for p in llama.lm_head.parameters() if p.requires_grad],
        lr=1e-4, fused=True,
    )

    def vanilla_step_lmhead():
        opt_b.zero_grad(set_to_none=True)
        out = llama(input_ids)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        opt_b.step()

    print()
    print("[B] Vanilla Llama, fwd+bwd+step (lm_head only trainable)")
    tps_b, mem_b, _ = bench(
        "vanilla Llama step (lm_head)", vanilla_step_lmhead,
        args.warmup, args.iter, BS, T,
    )
    del opt_b
    # Re-freeze lm_head before optional path C so we don't double-count.
    for p in llama.lm_head.parameters():
        p.requires_grad = False
    cleanup_cuda()

    # ----- Path C: vanilla Llama, full training step (all params trainable) -----
    if not args.skip_vanilla_full:
        for p in llama.parameters():
            p.requires_grad = True
        opt_c = torch.optim.AdamW(
            [p for p in llama.parameters() if p.requires_grad],
            lr=1e-4, fused=True,
        )

        def vanilla_step_full():
            opt_c.zero_grad(set_to_none=True)
            out = llama(input_ids)
            logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
            targets = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits.float(), targets)
            loss.backward()
            opt_c.step()

        print()
        print("[C] Vanilla Llama, fwd+bwd+step (ALL params trainable)")
        tps_c, mem_c, _ = bench(
            "vanilla Llama step (all)", vanilla_step_full,
            args.warmup, args.iter, BS, T,
        )
        del opt_c
        for p in llama.parameters():
            p.requires_grad = False
    else:
        print("\n[C] skipped (--skip-vanilla-full)")
        tps_c, mem_c = None, None

    # Free Llama before loading the integrated model — D path constructs
    # its own host LM.
    del llama
    cleanup_cuda()

    # ----- Path D: frozen Llama + GW phase1_pretrained_step -----
    if args.skip_gw:
        print("\n[D] skipped (--skip-gw)")
        tps_d, mem_d = None, None
    else:
        print()
        print(f"Loading frozen Llama + GraphWalker memory (d_mem={args.D_s or 'preset'})...")
        # `walker_cfg_from_args` sets segment_T=mod_period=tbptt_block=T.
        # d_mem must equal walker D_s.
        walker_cfg = walker_cfg_from_args(args, T=T, vocab=vocab)
        d_mem = walker_cfg.D_s
        cfg = PretrainedGWConfig(
            model_name=args.model, inject_layer=args.inject_layer,
            d_mem=d_mem, memory=walker_cfg, T=T, bs=BS, llama_dtype="bf16",
        )
        model = IntegratedLM(cfg).to(device)
        model.train(True)

        walker_params = sum(
            p.numel() for n, p in model.named_parameters() if p.requires_grad
        )
        walker_only_params = sum(
            p.numel() for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("memory.")
        )
        inject_params = walker_params - walker_only_params
        label = "TARGET ~110M" if args.target_config else "production ~25M"
        print_config_summary(walker_cfg, label)
        print(f"  walker (trainable):        {walker_only_params/1e6:.1f}M")
        print(f"  inject (W_in/W_out/scale): {inject_params/1e6:.1f}M")
        print(f"  total trainable:           {walker_params/1e6:.1f}M")
        if args.compile_walk_block:
            kind = "regional" if args.regional_compile else "whole-block"
            dyn = None if args.dynamic_shapes else False
            dyn_label = "dynamic=None" if args.dynamic_shapes else "dynamic=False"
            print(f"  Compiling walker {kind} (mode={args.compile_mode}, {dyn_label}) ...")
            model.compile_walker_block(
                mode=args.compile_mode,
                regional=args.regional_compile,
                dynamic=dyn,
            )

        opt_d = torch.optim.AdamW(
            [p for _, p in model.trainable_parameters()], lr=1e-4, fused=True,
        )
        input_ids_d = torch.randint(0, vocab, (BS, T), device=device)
        batch = Phase1Batch(input_ids=input_ids_d, target_ids=input_ids_d.clone())

        def gw_step():
            return phase1_pretrained_step(
                model, opt_d, batch, amp_dtype=torch.bfloat16,
            )

        print()
        print("[D] Frozen Llama + GW, full training step (phase1_pretrained_step)")
        tps_d, mem_d, _ = bench(
            "Llama + GW phase1 step", gw_step, args.warmup, args.iter, BS, T,
        )
        del model, opt_d
        cleanup_cuda()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=== Summary ===")
    rows = [
        ("A", "vanilla fwd (no_grad)", tps_a, mem_a),
        ("B", "vanilla step (lm_head)", tps_b, mem_b),
        ("C", "vanilla step (all)", tps_c, mem_c),
        ("D", "Llama + GW phase1 step", tps_d, mem_d),
    ]
    print(f"  {'Path':<6}{'name':<28}{'tok/s':>10}{'peak GB':>10}")
    for tag, name, tps, mem in rows:
        if tps is None:
            print(f"  {tag:<6}{name:<28}{'OOM/skip':>10}{'-':>10}")
        else:
            print(f"  {tag:<6}{name:<28}{tps/1000:>9.1f}k{mem:>9.2f}")
    if tps_b is not None and tps_d is not None:
        print()
        print(f"  D vs B slowdown: {tps_b/tps_d:.2f}x  "
              f"(vanilla lm_head-only {tps_b/1000:.1f}k → GW step {tps_d/1000:.1f}k tok/s)")
    if tps_c is not None and tps_d is not None:
        print(f"  D vs C slowdown: {tps_c/tps_d:.2f}x  "
              f"(hot Llama {tps_c/1000:.1f}k → GW step {tps_d/1000:.1f}k tok/s)")


if __name__ == "__main__":
    main()
