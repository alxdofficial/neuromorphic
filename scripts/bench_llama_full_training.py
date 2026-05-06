"""Bench full Llama-1B training (all params trainable, real backward, real
optimizer step) — the "max BS / max throughput for training a Llama-1B
from scratch / fully fine-tuning" baseline.

Modes for comparison:

  --mode full
      Every Llama param has requires_grad=True. Forward + backward through
      all 16 layers + AdamW state for all 1.24B params. The "vanilla full
      training" reference for the integration's frozen-Llama+GW comparison.
      Uses fused AdamW by default on CUDA.

  --mode full-ckpt
      Same as `full`, but with HuggingFace's `gradient_checkpointing_enable`
      — saves only layer-boundary activations, recomputes per-layer forward
      during backward. Trades ~30% wall-time for several GB of activation
      memory; lets you ride much higher BS.

  --mode grad-fwd
      Every Llama param has requires_grad=True. Runs train-mode forward and
      CE with autograd tracking enabled, then deletes the graph without
      backward. This is the "not doing backprop but still tracking gradients"
      lower-bound baseline.

  --mode grad-fwd-ckpt
      Same as `grad-fwd`, but with HF gradient checkpointing enabled. Because
      no backward is run, this is a useful "minimum activation residency"
      baseline for graph-tracked forward speed/VRAM.

  --mode lmhead-only
      Only `lm_head.weight` has requires_grad=True. Llama's transformer
      params are frozen. Mirrors the bench_pretrained_gw "vanilla_step"
      baseline — softer than real full training but useful for a sanity check.

  --mode ar
      Frozen Llama (lm_head trainable) doing the GRPO-shaped step shape
      WITHOUT the walker:
        1. Forward prefix [BS, T_pre] with grad enabled, KV cache captured.
        2. AR-generate L tokens with grad disabled (mirrors GRPO's
           grad_during_gen=False).
        3. Loss = CE on prefix predictions, backward, opt.step (lm_head only).
      Tokens-per-second is reported on (T_pre + L_gen) per step. This is
      the "frozen-Llama prefix-grad + AR-gen-no-grad" baseline against
      which the GRPO bench's K-sweep numbers should be read — it isolates
      AR-generation overhead through the frozen backbone before the
      walker's marginal cost is added.
      Use --t-pre and --gen-length to set prefix/generation lengths
      (defaults 256 / 128 = the production GRPO defaults).

Sweeps BS upward, doubling until OOM. Reports peak tok/s + the BS that
produced it, so the headline answer is "max BS for full Llama-1B training:
N → X k tok/s".

Usage:
    PYTHONPATH=. .venv/bin/python scripts/bench_llama_full_training.py \\
        --mode full --bs-list 1 2 4 8 16 --T 256

    PYTHONPATH=. .venv/bin/python scripts/bench_llama_full_training.py \\
        --mode ar --bs-list 1 2 4 8 16 --t-pre 256 --gen-length 128
"""

from __future__ import annotations

import argparse
import gc
import os
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM


def _bench_one(
    model_name: str, mode: str, BS: int, T: int,
    warmup: int, n_iter: int, fused_adam: bool,
    t_pre: int = 256, gen_length: int = 128,
) -> tuple[float, float, float] | None:
    """Returns (tok/s, peak_gb, ms_per_iter), or None on OOM.

    For `--mode ar`, `T` is unused; the AR step uses `t_pre + gen_length`
    tokens per iter (prefix forward-with-grad + AR generation no-grad).
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16,
        ).cuda()
        model.train(True)

        if mode == "full":
            for p in model.parameters():
                p.requires_grad = True
        elif mode == "full-ckpt":
            for p in model.parameters():
                p.requires_grad = True
            model.gradient_checkpointing_enable()
        elif mode == "grad-fwd":
            for p in model.parameters():
                p.requires_grad = True
        elif mode == "grad-fwd-ckpt":
            for p in model.parameters():
                p.requires_grad = True
            model.gradient_checkpointing_enable()
        elif mode == "lmhead-only":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.lm_head.parameters():
                p.requires_grad = True
        elif mode == "ar":
            for p in model.parameters():
                p.requires_grad = False
            for p in model.lm_head.parameters():
                p.requires_grad = True
        else:
            raise ValueError(f"unknown mode {mode!r}")

        trainable = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        do_backward = mode in {"full", "full-ckpt", "lmhead-only", "ar"}
        opt = None
        if do_backward:
            opt = torch.optim.AdamW(
                [p for p in model.parameters() if p.requires_grad],
                lr=1e-5,
                fused=fused_adam,
            )
        vocab = model.config.vocab_size

        if mode == "ar":
            prefix_ids = torch.randint(
                0, vocab, (BS, t_pre), device="cuda",
            )
            tokens_per_step = BS * (t_pre + gen_length)
        else:
            input_ids = torch.randint(0, vocab, (BS, T), device="cuda")
            tokens_per_step = BS * T

        def step():
            if opt is not None:
                opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids)
                logits = out.logits[:, :-1].reshape(-1, vocab)
                targets = input_ids[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits.float(), targets)
            if do_backward:
                loss.backward()
                opt.step()
            return loss

        def step_ar():
            opt.zero_grad(set_to_none=True)
            # Prefix pass: grad enabled so backward through frozen layers
            # actually fires (mirrors GRPO's grad_during_prefix=True).
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(prefix_ids, use_cache=True)
                past_kv = out.past_key_values
                logits = out.logits[:, :-1].reshape(-1, vocab)
                targets = prefix_ids[:, 1:].reshape(-1)
                loss = F.cross_entropy(logits.float(), targets)
                last_logits = out.logits[:, -1, :]
            # Generation pass: no grad (mirrors GRPO's grad_during_gen=False).
            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=torch.bfloat16,
            ):
                tok = last_logits.argmax(dim=-1, keepdim=True)
                for _ in range(gen_length):
                    out = model(
                        tok, past_key_values=past_kv, use_cache=True,
                    )
                    past_kv = out.past_key_values
                    tok = out.logits[:, -1, :].argmax(
                        dim=-1, keepdim=True,
                    )
            loss.backward()
            opt.step()
            return loss

        run_step = step_ar if mode == "ar" else step

        for _ in range(warmup):
            loss = run_step()
            del loss
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(n_iter):
            loss = run_step()
            del loss
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        tps = tokens_per_step * n_iter / elapsed
        ms_per_iter = elapsed / n_iter * 1000
        print(f"  BS={BS:>4}  trainable={trainable/1e9:.2f}B  "
              f"{tps/1000:>6.1f}k tok/s   peak {peak_gb:>5.2f} GB   "
              f"{ms_per_iter:>6.1f} ms/iter", flush=True)
        return tps, peak_gb, ms_per_iter
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            print(f"  BS={BS:>4}    OOM", flush=True)
            return None
        raise
    finally:
        try:
            del model
        except NameError:
            pass
        try:
            del opt
        except NameError:
            pass
        try:
            del input_ids
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument(
        "--mode",
        choices=("full", "full-ckpt", "grad-fwd", "grad-fwd-ckpt",
                 "lmhead-only", "ar"),
        default="full",
    )
    ap.add_argument("--bs-list", type=int, nargs="+",
                    default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--T", type=int, default=256,
                    help="Sequence length for parallel modes (full, "
                         "full-ckpt, grad-fwd, grad-fwd-ckpt, lmhead-only). "
                         "Unused in --mode ar.")
    ap.add_argument("--t-pre", type=int, default=256,
                    help="Prefix length for --mode ar (the GRPO-shaped "
                         "AR baseline). Default 256 matches production "
                         "PretrainedGWConfig.T.")
    ap.add_argument("--gen-length", type=int, default=128,
                    help="Generation length for --mode ar. Default 128 "
                         "matches PretrainedGWConfig.grpo_rollout_len.")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--iter", type=int, default=5)
    ap.add_argument("--no-fused-adam", action="store_true",
                    help="Disable fused AdamW in backward modes.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    print(f"=== Llama-1B full-training BS sweep  mode={args.mode} ===",
          flush=True)
    print(f"  device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"  model:  {args.model}", flush=True)
    if args.mode == "ar":
        print(f"  t_pre={args.t_pre}, gen_length={args.gen_length}, "
              f"warmup={args.warmup}, iter={args.iter}", flush=True)
    else:
        print(f"  T={args.T}, warmup={args.warmup}, iter={args.iter}",
              flush=True)
    print()

    rows: list[tuple[int, float, float, float]] = []
    for BS in args.bs_list:
        r = _bench_one(
            args.model, args.mode, BS, args.T, args.warmup, args.iter,
            fused_adam=not args.no_fused_adam,
            t_pre=args.t_pre, gen_length=args.gen_length,
        )
        if r is None:
            print(f"  Stopping at BS={BS} (OOM in {args.mode}).", flush=True)
            break
        rows.append((BS, *r))

    print()
    print("=" * 64)
    if not rows:
        print("  No BS fit — try a smaller --T or --mode full-ckpt.")
        return
    peak_bs, peak_tps, peak_gb, peak_ms = max(rows, key=lambda r: r[1])
    print(f"  Peak throughput:   {peak_tps/1000:6.1f}k tok/s  at BS={peak_bs} "
          f"(peak {peak_gb:.2f} GB, {peak_ms:.1f} ms/iter)")
    print(f"  Max-fitting BS:    {rows[-1][0]} (last row above)")


if __name__ == "__main__":
    main()
