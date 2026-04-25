"""Benchmark Llama-3.2-1B vanilla vs Llama+graph_walker memory.

Measures:
- Vanilla Llama forward-only tok/sec
- Vanilla Llama forward+backward tok/sec
- Llama + graph_walker forward-only tok/sec
- Llama + graph_walker forward+backward tok/sec
- Slowdown ratio
- Peak VRAM for each path

Usage:
    .venv/bin/python scripts/bench_pretrained_gw.py
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)


def _walker_cfg_for(d_mem: int, T: int) -> GraphWalkerConfig:
    """Bottleneck-adapter config: walker state dim D_s = d_mem, NOT d_lm.
    W_in: d_lm -> d_mem (compress), W_out: d_mem -> d_lm (expand).

    Topology: 16x16 plane * L=4 = 1024 columns * K=16 = 16K edges.
    n_heads=4 walkers per batch. mod_period=64.
    """
    return GraphWalkerConfig(
        plane_rows=16, plane_cols=16, L=4,
        K=16, D_model=d_mem, D_s=d_mem, D_id=32,
        n_heads=4, n_hops=4,
        D_q_in=64, D_q_per_head=64, n_score_heads=4,
        K_horizons=8, K_buf=8,
        vocab_size=128_256,    # Llama-3.2-1B vocab
        mod_period=64, tbptt_block=64, segment_T=T,
        gumbel_tau_start=2.0, gumbel_tau_end=0.5, gumbel_anneal_steps=10_000,
        epsilon_start=0.05, epsilon_end=0.01, epsilon_anneal_steps=10_000,
        lambda_balance=0.0,
        use_neuromod=True,
        neuromod_D_mod=128, neuromod_n_layers=2, neuromod_n_heads=4,
        neuromod_edge_hidden=64, neuromod_eta=1.0,
        compile_on_train=False,
    )


def _bench(name, fn, n_warmup, n_iter, BS, T):
    torch.cuda.synchronize()
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    tps = (BS * T * n_iter) / elapsed
    print(f"  {name:40s} {tps/1000:6.1f}k tok/s   peak {peak_gb:5.2f} GB   "
          f"{elapsed/n_iter*1000:6.1f} ms/iter")
    return tps, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--d-mem", type=int, default=512,
                    help="Walker state dim (= MemInjectLayer d_mem). "
                         "Default 512 matches PretrainedGWConfig default.")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iter", type=int, default=10)
    ap.add_argument("--inject-layer", type=int, default=8)
    args = ap.parse_args()

    device = torch.device("cuda")
    BS, T = args.bs, args.T
    print(f"\n=== Pretrained graph_walker benchmark ===")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  model:  {args.model}")
    print(f"  BS={BS}, T={T}, warmup={args.warmup}, iter={args.iter}")
    print()

    print("Loading vanilla Llama...")
    vanilla = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    vanilla.train(False)
    for p in vanilla.parameters():
        p.requires_grad = False
    d_lm = vanilla.config.hidden_size
    n_layers = vanilla.config.num_hidden_layers
    vocab = vanilla.config.vocab_size
    n_params = sum(p.numel() for p in vanilla.parameters())
    print(f"  d_lm={d_lm}, n_layers={n_layers}, vocab={vocab}, "
          f"params={n_params/1e9:.2f}B")
    print()

    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    print("[Vanilla Llama, forward-only]")
    def vanilla_fwd():
        with torch.no_grad():
            return vanilla(input_ids).logits
    vanilla_fwd_tps, vanilla_fwd_mem = _bench(
        "Llama-1B vanilla fwd", vanilla_fwd, args.warmup, args.iter, BS, T,
    )

    # For training step we need at least one trainable so backward has something.
    # Unfreeze lm_head only — keeps the comparison fair (smallest possible
    # trainable surface; everything is dominated by activation memory + matmul).
    for p in vanilla.lm_head.parameters():
        p.requires_grad = True
    vanilla.train(True)

    def vanilla_step():
        out = vanilla(input_ids)
        logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
        targets = input_ids[:, 1:].reshape(-1)
        loss = F.cross_entropy(logits.float(), targets)
        loss.backward()
        for p in vanilla.lm_head.parameters():
            p.grad = None
    print()
    print("[Vanilla Llama, forward+backward (lm_head only trainable)]")
    vanilla_step_tps, vanilla_step_mem = _bench(
        "Llama-1B vanilla step", vanilla_step, args.warmup, args.iter, BS, T,
    )

    del vanilla
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    print()
    print(f"Loading Llama + graph_walker memory (d_mem={args.d_mem})...")
    walker_cfg = _walker_cfg_for(d_mem=args.d_mem, T=T)
    cfg = PretrainedGWConfig(
        model_name=args.model,
        inject_layer=args.inject_layer,
        d_mem=args.d_mem,
        memory=walker_cfg,
        T=T, bs=BS,
        llama_dtype="bf16",
    )
    wrapper = GraphWalkerPretrainedLM(cfg).to(device)
    wrapper.train(False)

    walker_params = sum(
        p.numel() for n, p in wrapper.named_parameters() if p.requires_grad
    )
    walker_only_params = sum(
        p.numel() for n, p in wrapper.named_parameters()
        if p.requires_grad and n.startswith("memory.")
    )
    inject_params = walker_params - walker_only_params
    print(f"  walker (trainable):        {walker_only_params/1e6:.1f}M")
    print(f"  inject (W_in/W_out/scale): {inject_params/1e6:.1f}M")
    print(f"  total trainable:           {walker_params/1e6:.1f}M")
    print()

    def gw_fwd():
        wrapper.reset_memory(bs=BS)
        with torch.no_grad(), torch.autocast(
            device_type=device.type, dtype=torch.bfloat16,
        ):
            return wrapper(input_ids).logits

    print("[Llama + graph_walker, forward-only]")
    gw_fwd_tps, gw_fwd_mem = _bench(
        "Llama-1B + GW fwd", gw_fwd, args.warmup, args.iter, BS, T,
    )

    wrapper.train(True)
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-4,
    )
    targets = input_ids.clone()
    batch = Phase1Batch(input_ids=input_ids, target_ids=targets)

    def gw_step():
        return phase1_pretrained_step(
            wrapper, opt, batch, amp_dtype=torch.bfloat16,
        )

    print()
    print("[Llama + graph_walker, full training step]")
    gw_step_tps, gw_step_mem = _bench(
        "Llama-1B + GW step", gw_step, args.warmup, args.iter, BS, T,
    )

    print()
    print("=== Summary ===")
    print(f"  Forward-only slowdown:  {vanilla_fwd_tps / gw_fwd_tps:5.2f}x  "
          f"(vanilla {vanilla_fwd_tps/1000:.1f}k -> +mem {gw_fwd_tps/1000:.1f}k tok/s)")
    print(f"  Training-step slowdown: {vanilla_step_tps / gw_step_tps:5.2f}x  "
          f"(vanilla {vanilla_step_tps/1000:.1f}k -> +mem {gw_step_tps/1000:.1f}k tok/s)")
    print(f"  Peak VRAM training:     {gw_step_mem - vanilla_step_mem:+.2f} GB  "
          f"(vanilla {vanilla_step_mem:.2f} -> +mem {gw_step_mem:.2f} GB)")


if __name__ == "__main__":
    main()
