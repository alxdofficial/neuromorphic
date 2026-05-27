"""Benchmark batch sizes for v5.4 graph baseline.

Loads Llama once, then runs forward+backward at each BS to find max throughput.
Reports steps/sec, examples/sec, and peak GPU memory at each BS.
"""
from __future__ import annotations
import argparse
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_qa import QABatch


def fake_batch(B: int, T_ctx: int, T_q: int, T_a: int, vocab_size: int, device: str) -> QABatch:
    """Synthetic QA batch of the requested shapes — exercises the same code
    paths as real data without disk I/O."""
    torch.manual_seed(0)
    return QABatch(
        context_ids=torch.randint(0, vocab_size, (B, T_ctx), device=device),
        context_mask=torch.ones(B, T_ctx, dtype=torch.bool, device=device),
        question_ids=torch.randint(0, vocab_size, (B, T_q), device=device),
        question_mask=torch.ones(B, T_q, dtype=torch.bool, device=device),
        answer_ids=torch.randint(0, vocab_size, (B, T_a), device=device),
        answer_mask=torch.ones(B, T_a, dtype=torch.bool, device=device),
        answer_content_mask=torch.ones(B, T_a, dtype=torch.bool, device=device),
        task_family=["calendar"] * B,
        question_type=["wh"] * B,
    )


def bench_bs(
    model,
    bs: int,
    chunk_size: int,
    window_size: int,
    warmup: int = 5,
    timed: int = 15,
    T_q: int = 64,
    T_a: int = 32,
    vocab_size: int = 128_000,
    device: str = "cuda",
) -> dict:
    """Measure forward+backward throughput at a given BS. Returns dict with
    steps/sec, examples/sec, peak memory MiB. None if OOM."""
    try:
        batch = fake_batch(bs, chunk_size, T_q, T_a, vocab_size, device)
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
        torch.cuda.reset_peak_memory_stats(device)

        for _ in range(warmup):
            opt.zero_grad()
            out = model.compute_qa_loss(batch, window_size=window_size)
            out["loss"].backward()
            opt.step()
        torch.cuda.synchronize()

        t0 = time.time()
        for _ in range(timed):
            opt.zero_grad()
            out = model.compute_qa_loss(batch, window_size=window_size)
            out["loss"].backward()
            opt.step()
        torch.cuda.synchronize()
        elapsed = time.time() - t0

        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        sps = timed / elapsed
        eps = sps * bs
        return {"bs": bs, "ok": True, "steps_per_sec": sps, "examples_per_sec": eps,
                "peak_mib": peak_mib, "loss_final": float(out["loss_recon"])}
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return {"bs": bs, "ok": False, "err": "OOM"}
    except Exception as e:
        torch.cuda.empty_cache()
        return {"bs": bs, "ok": False, "err": str(e)[:80]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bs-list", nargs="+", type=int, default=[2, 4, 8, 12, 16, 20, 24])
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--timed", type=int, default=15)
    args = ap.parse_args()

    device = "cuda"
    print("Loading Llama (once, shared across BS tests)...")
    cfg = ReprConfig(
        batch_size=1,
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    llama = AutoModelForCausalLM.from_pretrained(
        cfg.llama_model, torch_dtype=torch.bfloat16
    ).to(device)
    llama.train(False)
    for p in llama.parameters():
        p.requires_grad_(False)

    print(f"Building graph_v5_baseline encoder...")
    model = ReprLearningModel(cfg, variant="graph_v5_baseline", llama_model=llama).to(device)
    n_train = model.n_trainable_params()
    print(f"  {n_train:,} trainable params")
    print(f"\nBench config: chunk={args.chunk_size}, window={args.window_size}, "
          f"warmup={args.warmup}, timed={args.timed}")
    print()

    print(f"{'BS':>4}  {'steps/s':>8}  {'examples/s':>11}  {'peak MiB':>9}  {'loss':>7}  status")
    print("-" * 60)
    results = []
    for bs in args.bs_list:
        r = bench_bs(model, bs, args.chunk_size, args.window_size,
                     warmup=args.warmup, timed=args.timed, device=device)
        results.append(r)
        if r["ok"]:
            print(f"{r['bs']:>4}  {r['steps_per_sec']:>8.2f}  "
                  f"{r['examples_per_sec']:>11.2f}  {r['peak_mib']:>9.0f}  "
                  f"{r['loss_final']:>7.3f}  ok")
        else:
            print(f"{r['bs']:>4}  {'-':>8}  {'-':>11}  {'-':>9}  {'-':>7}  {r['err']}")

    ok_results = [r for r in results if r["ok"]]
    if ok_results:
        best = max(ok_results, key=lambda r: r["examples_per_sec"])
        print()
        print(f"Peak throughput: BS={best['bs']} at {best['examples_per_sec']:.1f} examples/s "
              f"({best['peak_mib']:.0f} MiB, {best['peak_mib']/24576*100:.0f}% of 24GB)")
        safe = [r for r in ok_results if r["peak_mib"] < 24576 * 0.85]
        if safe:
            best_safe = max(safe, key=lambda r: r["examples_per_sec"])
            print(f"Recommended (<=85% memory): BS={best_safe['bs']} at "
                  f"{best_safe['examples_per_sec']:.1f} examples/s "
                  f"({best_safe['peak_mib']:.0f} MiB)")


if __name__ == "__main__":
    main()
