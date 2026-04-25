"""Detailed profile: vanilla Llama-1B vs walker-only standalone vs Llama+GW.

Three configurations under torch.profiler:
  A. Vanilla Llama-3.2-1B   (no walker)
  B. Walker standalone      (no Llama, same walker config used in C)
  C. Llama-3.2-1B + walker  (the production training shape)

For each: forward-only and forward+backward.
Output: per-CUDA-op wall time, per-CPU-op wall time, VRAM allocation
history, top-20 hot ops, time-per-token breakdown.

Usage:
    PYTHONPATH=$PWD .venv/bin/python scripts/profile_pretrained_gw.py
    PYTHONPATH=$PWD .venv/bin/python scripts/profile_pretrained_gw.py --compile
    PYTHONPATH=$PWD .venv/bin/python scripts/profile_pretrained_gw.py --d-mem 256

Generates:
    /tmp/gw_profile/A_vanilla_step.txt          per-op table for vanilla step
    /tmp/gw_profile/B_walker_only_step.txt      walker standalone
    /tmp/gw_profile/C_llama_walker_step.txt     full pretrained
    /tmp/gw_profile/summary.md                  comparison table
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoModelForCausalLM

from src.graph_walker.config import GraphWalkerConfig
from src.graph_walker.pretrained.config import PretrainedGWConfig
from src.graph_walker.pretrained.llm_wrapper import GraphWalkerPretrainedLM
from src.graph_walker.pretrained.train_phase1 import (
    Phase1Batch,
    phase1_pretrained_step,
)
from src.graph_walker.standalone import StandaloneLM
from src.graph_walker.train_phase1 import phase1_step


def _walker_cfg(d_mem: int, T: int, vocab: int) -> GraphWalkerConfig:
    return GraphWalkerConfig(
        plane_rows=16, plane_cols=16, L=4,
        K=16, D_model=d_mem, D_s=d_mem, D_id=32,
        n_heads=4, n_hops=4,
        D_q_in=64, D_q_per_head=64, n_score_heads=4,
        K_horizons=8, K_buf=8,
        vocab_size=vocab,
        mod_period=64, tbptt_block=64, segment_T=T,
        gumbel_tau_start=2.0, gumbel_tau_end=0.5, gumbel_anneal_steps=10_000,
        epsilon_start=0.05, epsilon_end=0.01, epsilon_anneal_steps=10_000,
        lambda_balance=0.0,
        use_neuromod=True,
        neuromod_D_mod=128, neuromod_n_layers=2, neuromod_n_heads=4,
        neuromod_edge_hidden=64, neuromod_eta=1.0,
        compile_on_train=False,
    )


def _profile(name: str, fn, n_iter: int, out_dir: Path):
    """Run fn() n_iter times under torch.profiler. Save the per-op table."""
    torch.cuda.synchronize()
    # Warmup (3 iters)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=False,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        t0 = time.perf_counter()
        for i in range(n_iter):
            with record_function(f"iter_{i}"):
                fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0

    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    table_cuda = prof.key_averages().table(
        sort_by="cuda_time_total", row_limit=25,
    )
    table_cpu = prof.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=25,
    )

    out_path = out_dir / f"{name}.txt"
    with open(out_path, "w") as f:
        f.write(f"=== {name} ===\n")
        f.write(f"wall_time={elapsed:.3f}s, n_iter={n_iter}, peak_VRAM={peak_gb:.2f}GB\n")
        f.write(f"per_iter_ms={elapsed/n_iter*1000:.1f}\n\n")
        f.write("--- TOP CUDA OPS (sort by cuda_time_total) ---\n")
        f.write(table_cuda + "\n\n")
        f.write("--- TOP CPU OPS (sort by self_cpu_time_total) ---\n")
        f.write(table_cpu + "\n")
    print(f"  {name:40s} {elapsed/n_iter*1000:6.1f} ms/iter   peak {peak_gb:5.2f} GB   "
          f"-> {out_path}")
    return elapsed / n_iter * 1000, peak_gb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--T", type=int, default=256)
    ap.add_argument("--d-mem", type=int, default=512)
    ap.add_argument("--iter", type=int, default=5)
    ap.add_argument("--inject-layer", type=int, default=8)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--out-dir", default="/tmp/gw_profile")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    BS, T = args.bs, args.T

    print(f"=== Profiling Llama / walker / Llama+walker ===")
    print(f"  device: {torch.cuda.get_device_name(0)}")
    print(f"  BS={BS}, T={T}, d_mem={args.d_mem}, compile={args.compile}, iter={args.iter}")
    print()

    results = {}

    # --- A. Vanilla Llama ---
    print("[A] Loading vanilla Llama...")
    vanilla = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
    ).to(device)
    for p in vanilla.parameters():
        p.requires_grad = False
    for p in vanilla.lm_head.parameters():
        p.requires_grad = True
    vocab = vanilla.config.vocab_size
    input_ids = torch.randint(0, vocab, (BS, T), device=device)

    vanilla.train(False)
    def vanilla_fwd():
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return vanilla(input_ids).logits

    vanilla.train(True)
    def vanilla_step():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = vanilla(input_ids)
            logits = out.logits[:, :-1].reshape(-1, out.logits.size(-1))
            tgt = input_ids[:, 1:].reshape(-1)
            loss = F.cross_entropy(logits.float(), tgt)
        loss.backward()
        for p in vanilla.lm_head.parameters():
            p.grad = None

    print("[A] Profiling vanilla...")
    results["A_fwd"] = _profile("A_vanilla_fwd", vanilla_fwd, args.iter, out_dir)
    results["A_step"] = _profile("A_vanilla_step", vanilla_step, args.iter, out_dir)
    del vanilla
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print()

    # --- B. Walker standalone ---
    print("[B] Building walker standalone (matched config to C)...")
    walker_cfg = _walker_cfg(d_mem=args.d_mem, T=T, vocab=vocab)
    standalone = StandaloneLM(walker_cfg).cuda()
    if args.compile:
        standalone.memory.compile_step()
    standalone.train(False)
    walker_input = torch.randint(0, vocab, (BS, T), device=device)

    def walker_fwd():
        standalone.memory.begin_segment(BS, device)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            for t in range(T):
                standalone.memory.step_core(walker_input[:, t])

    standalone.train(True)
    walker_opt = torch.optim.AdamW(standalone.parameters(), lr=1e-4)
    def walker_step():
        return phase1_step(
            standalone, walker_opt, walker_input,
            tbptt_block=walker_cfg.tbptt_block,
            amp_dtype=torch.bfloat16,
        )

    print("[B] Profiling walker standalone...")
    results["B_fwd"] = _profile("B_walker_only_fwd", walker_fwd, args.iter, out_dir)
    results["B_step"] = _profile("B_walker_only_step", walker_step, args.iter, out_dir)
    del standalone
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print()

    # --- C. Llama + walker ---
    print("[C] Building Llama + walker...")
    cfg = PretrainedGWConfig(
        model_name=args.model,
        inject_layer=args.inject_layer,
        d_mem=args.d_mem,
        memory=walker_cfg,
        T=T, bs=BS,
        llama_dtype="bf16",
    )
    wrapper = GraphWalkerPretrainedLM(cfg).cuda()
    if args.compile:
        wrapper.memory.compile_step()
    wrapper.train(False)
    def gw_fwd():
        wrapper.reset_memory(bs=BS)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return wrapper(input_ids).logits

    wrapper.train(True)
    opt = torch.optim.AdamW(
        [p for _, p in wrapper.trainable_parameters()], lr=1e-4,
    )
    batch = Phase1Batch(input_ids=input_ids, target_ids=input_ids.clone())
    def gw_step():
        return phase1_pretrained_step(wrapper, opt, batch, amp_dtype=torch.bfloat16)

    print("[C] Profiling Llama + walker...")
    results["C_fwd"] = _profile("C_llama_walker_fwd", gw_fwd, args.iter, out_dir)
    results["C_step"] = _profile("C_llama_walker_step", gw_step, args.iter, out_dir)
    print()

    # --- Summary ---
    summary_path = out_dir / "summary.md"
    with open(summary_path, "w") as f:
        f.write("# Profile summary\n\n")
        f.write(f"Device: {torch.cuda.get_device_name(0)}\n")
        f.write(f"BS={BS}, T={T}, d_mem={args.d_mem}, compile={args.compile}\n\n")
        f.write("## Wall time per iter (ms) and VRAM (GB)\n\n")
        f.write("| Config | fwd ms | fwd VRAM | step ms | step VRAM |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| A. Vanilla Llama        | {results['A_fwd'][0]:.1f} | {results['A_fwd'][1]:.2f} | "
                f"{results['A_step'][0]:.1f} | {results['A_step'][1]:.2f} |\n")
        f.write(f"| B. Walker only          | {results['B_fwd'][0]:.1f} | {results['B_fwd'][1]:.2f} | "
                f"{results['B_step'][0]:.1f} | {results['B_step'][1]:.2f} |\n")
        f.write(f"| C. Llama + walker       | {results['C_fwd'][0]:.1f} | {results['C_fwd'][1]:.2f} | "
                f"{results['C_step'][0]:.1f} | {results['C_step'][1]:.2f} |\n\n")
        f.write("## Tokens/sec\n\n")
        f.write(f"BS×T = {BS*T} tokens per iter\n\n")
        f.write("| Config | fwd tok/s | step tok/s |\n")
        f.write("|---|---|---|\n")
        for k, label in [("A", "Vanilla Llama"), ("B", "Walker only"), ("C", "Llama + walker")]:
            fps = BS * T * 1000 / results[f"{k}_fwd"][0]
            sps = BS * T * 1000 / results[f"{k}_step"][0]
            f.write(f"| {label} | {fps/1000:.1f}k | {sps/1000:.1f}k |\n")
        f.write("\n## Decomposition\n\n")
        f.write(f"- Llama-only step:        {results['A_step'][0]:.1f} ms\n")
        f.write(f"- Walker-only step:       {results['B_step'][0]:.1f} ms\n")
        f.write(f"- Combined step:          {results['C_step'][0]:.1f} ms\n")
        sum_indep = results["A_step"][0] + results["B_step"][0]
        overhead = results["C_step"][0] - sum_indep
        f.write(f"- Naive sum A+B:          {sum_indep:.1f} ms\n")
        f.write(f"- Combined - sum:         {overhead:+.1f} ms (positive = integration overhead)\n")
        f.write("\n## Per-op detail\n\n")
        f.write("See A_vanilla_step.txt, B_walker_only_step.txt, C_llama_walker_step.txt\n")
    print(f"\nSummary written to {summary_path}")
    print(f"Per-op tables in {out_dir}/")


if __name__ == "__main__":
    main()
