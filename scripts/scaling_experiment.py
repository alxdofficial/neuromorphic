"""Scaling experiment — throughput vs parameter count (v5).

For each neuromorphic (B, L_scan) config, finds the maximum batch size before
OOM and measures tokens/sec at that batch size.  Does the same for Pythia-160M
and Mamba-130M baselines.  Produces an annotated scatter plot of parameter count
vs throughput.

Each (model, batch_size) combination runs in a **separate subprocess** so that
torch.compile graph limits and OOM recovery work cleanly.

Usage:
    python -m scripts.scaling_experiment [--warmup 5] [--steps 10] [--out outputs/scaling_experiment]
"""

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time

import torch

# ---------------------------------------------------------------------------
# Neuromorphic worker — runs one (B, L_scan, BS) config in a subprocess
# ---------------------------------------------------------------------------

_NEURO_WORKER = textwrap.dedent(r'''
import json, sys, time, torch
from src.data.streaming import StreamBatch
from src.model.config import ModelConfig
from src.model.model import NeuromorphicLM
from src.training.trainer import TBPTTTrainer

bs      = int(sys.argv[1])
warmup  = int(sys.argv[2])
steps   = int(sys.argv[3])
B       = int(sys.argv[4])
L_scan  = int(sys.argv[5])

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

def _rand_batch(bs, t, vocab, device):
    x = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    y = torch.randint(0, vocab, (bs, t), dtype=torch.long, device=device)
    prev = torch.zeros(bs, dtype=torch.long, device=device)
    return StreamBatch(input_ids=x, target_ids=y, prev_token=prev)

cfg = ModelConfig.tier_a(N=256, B=B, L_scan=L_scan, use_compile=True)
cfg.set_phase("B")

model = NeuromorphicLM(cfg).to(device)
params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, fused=True)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
trainer = TBPTTTrainer(
    model=model, optimizer=optimizer, scheduler=scheduler,
    dataloader=iter(()), config=cfg, device=device,
    collector=None, log_interval=10_000,
)

for i in range(warmup):
    trainer.train_chunk(_rand_batch(bs, cfg.T, cfg.vocab_size, device))
    trainer.global_step += 1

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []
for _ in range(steps):
    batch = _rand_batch(bs, cfg.T, cfg.vocab_size, device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    trainer.train_chunk(batch)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    trainer.global_step += 1
    times.append(t1 - t0)

avg = sum(times) / len(times)
tok_per_step = bs * cfg.T
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(json.dumps({
    "bs": bs, "params": params, "avg_step_s": round(avg, 4),
    "tok_per_s": round(tok_per_step / avg),
    "peak_vram_gb": round(peak_vram, 2),
}))
''')

# ---------------------------------------------------------------------------
# Baseline worker — runs Pythia or Mamba in a subprocess
# ---------------------------------------------------------------------------

_BASELINE_WORKER = textwrap.dedent(r'''
import json, sys, time, torch

model_name = sys.argv[1]
bs         = int(sys.argv[2])
warmup     = int(sys.argv[3])
steps      = int(sys.argv[4])
seq_len    = 256
vocab      = 32000

device = torch.device("cuda")
torch.set_float32_matmul_precision("high")

if model_name == "pythia-160m":
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        vocab_size=vocab, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=2048, rotary_pct=0.25,
        use_parallel_residual=True,
    )
    model = GPTNeoXForCausalLM(cfg).to(device)
elif model_name == "mamba-130m":
    from transformers import MambaConfig, MambaForCausalLM
    cfg = MambaConfig(
        vocab_size=vocab, hidden_size=768, num_hidden_layers=24,
        state_size=16, expand=2, conv_kernel=4,
        use_bias=False, use_conv_bias=True,
    )
    model = MambaForCausalLM(cfg).to(device)
else:
    raise ValueError(f"Unknown baseline: {model_name}")

params = sum(p.numel() for p in model.parameters())
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, fused=True)

for _ in range(warmup):
    x = torch.randint(0, vocab, (bs, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab, (bs, seq_len), dtype=torch.long, device=device)
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=y).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()

times = []
for _ in range(steps):
    x = torch.randint(0, vocab, (bs, seq_len), dtype=torch.long, device=device)
    y = torch.randint(0, vocab, (bs, seq_len), dtype=torch.long, device=device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = model(input_ids=x, labels=y).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)

avg = sum(times) / len(times)
tok_per_step = bs * seq_len
peak_vram = torch.cuda.max_memory_allocated() / 1e9
print(json.dumps({
    "bs": bs, "params": params, "avg_step_s": round(avg, 4),
    "tok_per_s": round(tok_per_step / avg),
    "peak_vram_gb": round(peak_vram, 2),
}))
''')

# ---------------------------------------------------------------------------
# Subprocess helpers
# ---------------------------------------------------------------------------

TIMEOUT = 900  # 15 min per subprocess


def _parse_json_output(stdout: str) -> dict:
    """Extract the last JSON line from subprocess stdout."""
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"No JSON in output:\n{stdout[-500:]}")


def _run_neuro(B: int, L_scan: int, bs: int, warmup: int, steps: int) -> dict | None:
    """Run neuromorphic worker. Returns result dict or None on OOM."""
    result = subprocess.run(
        [sys.executable, "-c", _NEURO_WORKER,
         str(bs), str(warmup), str(steps), str(B), str(L_scan)],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            return None
        raise RuntimeError(f"Neuro B={B} L_scan={L_scan} BS={bs} failed:\n{stderr[-500:]}")
    return _parse_json_output(result.stdout)


def _run_baseline(name: str, bs: int, warmup: int, steps: int) -> dict | None:
    """Run baseline worker. Returns result dict or None on OOM."""
    result = subprocess.run(
        [sys.executable, "-c", _BASELINE_WORKER,
         name, str(bs), str(warmup), str(steps)],
        capture_output=True, text=True, timeout=TIMEOUT,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "OutOfMemoryError" in stderr or "CUDA out of memory" in stderr:
            return None
        raise RuntimeError(f"Baseline {name} BS={bs} failed:\n{stderr[-500:]}")
    return _parse_json_output(result.stdout)


# ---------------------------------------------------------------------------
# Batch-size search
# ---------------------------------------------------------------------------

def find_max_bs(run_fn, warmup: int, steps: int, label: str) -> dict | None:
    """Exponential probe to find max working batch size.

    Skips binary search — throughput plateaus well before OOM so the last
    good power-of-2 is close enough, and each subprocess costs 2-4 min of
    torch.compile warmup.

    ``run_fn(bs)`` should return a result dict or None on OOM.
    """
    last_good_result = None
    probe_bs = 16

    print(f"  [{label}] Probing batch sizes ...", flush=True)
    while probe_bs <= 512:
        print(f"    BS={probe_bs} ... ", end="", flush=True)
        try:
            r = run_fn(probe_bs)
        except (RuntimeError, subprocess.TimeoutExpired) as e:
            print(f"error ({e})")
            break
        if r is None:
            print("OOM")
            break
        print(f"ok ({r['tok_per_s']} tok/s, {r['peak_vram_gb']} GB)")
        last_good_result = r
        probe_bs *= 2

    return last_good_result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(neuro_results: list[dict], baseline_results: list[dict],
              out_path: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 7))

    # Neuromorphic points
    if neuro_results:
        xs = [r["params"] for r in neuro_results]
        ys = [r["tok_per_s"] for r in neuro_results]
        ax.scatter(xs, ys, c="royalblue", marker="o", s=80, zorder=5,
                   label="Neuromorphic (tier_a base)")
        for r in neuro_results:
            label_text = f"B={r['B']}, L={r['L_scan']}\nBS={r['bs']}"
            ax.annotate(label_text, (r["params"], r["tok_per_s"]),
                        textcoords="offset points", xytext=(8, 5),
                        fontsize=7, color="royalblue")

    # Baseline points
    markers = {"pythia-160m": ("red", "^", "Pythia-160M"),
               "mamba-130m": ("green", "s", "Mamba-130M")}
    for r in baseline_results:
        color, marker, label = markers[r["model"]]
        ax.scatter([r["params"]], [r["tok_per_s"]], c=color, marker=marker,
                   s=120, zorder=6, label=label)
        ax.annotate(f"{label}\nBS={r['bs']}",
                    (r["params"], r["tok_per_s"]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=8, fontweight="bold", color=color)

    ax.set_xscale("log")
    ax.set_xlabel("Parameter Count", fontsize=12)
    ax.set_ylabel("Tokens / sec", fontsize=12)
    ax.set_title("Scaling Experiment — RTX 4090, tier_a base, N=256, Phase B",
                 fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nPlot saved to {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scaling experiment: throughput vs parameter count")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warmup steps per subprocess (default: 5)")
    parser.add_argument("--steps", type=int, default=10,
                        help="Timed steps per subprocess (default: 10)")
    parser.add_argument("--out", type=str, default="outputs/scaling_experiment",
                        help="Output path prefix (default: outputs/scaling_experiment)")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    print(f"Warmup: {args.warmup} steps, Timed: {args.steps} steps")
    print(f"Subprocess timeout: {TIMEOUT}s")
    print()

    # ------------------------------------------------------------------
    # Neuromorphic configs: tier_a base (D=2048, C=16), sweep (B, L_scan)
    # ------------------------------------------------------------------
    B_values = [2, 4, 6, 8]
    L_values = [4, 8, 12]

    neuro_results = []
    total_combos = len(B_values) * len(L_values)
    combo_idx = 0

    for B in B_values:
        for L_scan in L_values:
            combo_idx += 1
            print(f"=== [{combo_idx}/{total_combos}] Neuromorphic B={B}, "
                  f"L_scan={L_scan} ===")
            t_start = time.time()

            def run_fn(bs, _B=B, _L=L_scan):
                return _run_neuro(_B, _L, bs, args.warmup, args.steps)

            try:
                result = find_max_bs(run_fn, args.warmup, args.steps,
                                     f"B={B} L_scan={L_scan}")
            except Exception as e:
                print(f"  SKIPPED (error: {e})")
                continue

            elapsed = time.time() - t_start
            if result is None:
                print(f"  No viable batch size found ({elapsed:.0f}s)")
                continue

            result["B"] = B
            result["L_scan"] = L_scan
            result["model"] = "neuromorphic"
            neuro_results.append(result)
            print(f"  Best: BS={result['bs']}, {result['tok_per_s']} tok/s, "
                  f"{result['peak_vram_gb']} GB ({elapsed:.0f}s)")
            print()

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------
    baseline_results = []

    for name in ["pythia-160m", "mamba-130m"]:
        print(f"=== Baseline: {name} ===")
        t_start = time.time()

        def run_fn(bs, _name=name):
            return _run_baseline(_name, bs, args.warmup, args.steps)

        try:
            result = find_max_bs(run_fn, args.warmup, args.steps, name)
        except Exception as e:
            print(f"  SKIPPED (error: {e})")
            continue

        elapsed = time.time() - t_start
        if result is None:
            print(f"  No viable batch size found ({elapsed:.0f}s)")
            continue

        result["model"] = name
        baseline_results.append(result)
        print(f"  Best: BS={result['bs']}, {result['tok_per_s']} tok/s, "
              f"{result['peak_vram_gb']} GB ({elapsed:.0f}s)")
        print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    all_results = {
        "env": {
            "gpu": gpu_name,
            "gpu_mem_gb": round(gpu_mem, 1),
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
        },
        "settings": {
            "base": "tier_a",
            "N": 256,
            "warmup": args.warmup,
            "steps": args.steps,
        },
        "neuromorphic": neuro_results,
        "baselines": baseline_results,
    }

    json_path = args.out + ".json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {json_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    png_path = args.out + ".png"
    make_plot(neuro_results, baseline_results, png_path)

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 72)
    print(f"{'Model':<25} {'Params':>10} {'BS':>5} {'tok/s':>10} {'VRAM GB':>8}")
    print("-" * 72)
    for r in sorted(neuro_results, key=lambda x: x["params"]):
        tag = f"neuro B={r['B']} L={r['L_scan']}"
        print(f"{tag:<25} {r['params']:>10,} {r['bs']:>5} "
              f"{r['tok_per_s']:>10,} {r['peak_vram_gb']:>8.2f}")
    for r in baseline_results:
        print(f"{r['model']:<25} {r['params']:>10,} {r['bs']:>5} "
              f"{r['tok_per_s']:>10,} {r['peak_vram_gb']:>8.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
