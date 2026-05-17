#!/usr/bin/env python3
"""Gap decomposition: where does our +0.75-nat-above-vanilla loss come from?

Runs IntegratedLM on the SAME train-mix val parquets as bench_vanilla_llama,
under different ablations:

  --ablate scale_zero   : scale_raw → 0 (inj = 0·W_out(readout) = 0 exact identity).
                          Measures pure IntegratedLM SCAFFOLDING overhead
                          (windowing, KV-cache cropping, state threading)
                          vs the vanilla baseline.
  --ablate scale_trained : trained scale, normal memory readout.
                          Measures memory INJECTION effect on top of scaffolding.
  --ablate scale_trained_readout_zero : trained scale, but memory readout
                          forced to zero. With scale_raw≠0, this isolates
                          whether W_out has any bias contribution at zero
                          input (it shouldn't — pure linear). Sanity check.

Comparison points (already measured):
  - vanilla 1024-chunk floor: 2.4367 nats (outputs/vanilla_llama_train_floor.json)
  - ours full (training-bug state): ~3.19 nats (Wave 1 final train_loss)

Output: per-source NTP CE + JSON dump.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.loaders import LongDocDataset


def _zero_scale(model: IntegratedLM):
    """Set scale_raw to all zeros. inj = scale·W_out(readout) → 0 → identity."""
    mil = model._mem_inject_layer()
    with torch.no_grad():
        mil.scale_raw.zero_()
    eff = mil.scale_max * torch.tanh(mil.scale_raw)
    print(f"  [ablate] scale_raw zeroed; effective_scale max={eff.abs().max().item():.6f}")


def _force_readout_zero(model: IntegratedLM):
    """Monkey-patch the memory_fn closure to always return zeros."""
    orig_build = model._build_memory_fn

    def patched_build(read_trajectory):
        orig_fn = orig_build(read_trajectory)
        def zero_fn(h_mem):
            out = orig_fn(h_mem)
            return torch.zeros_like(out)
        return zero_fn
    model._build_memory_fn = patched_build
    print("  [ablate] memory_fn output forced to zero")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data-paths", nargs="+", required=True)
    ap.add_argument("--val-batches", type=int, default=200)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--ablate", choices=["scale_zero", "scale_trained",
                                          "scale_trained_readout_zero"],
                    default="scale_trained")
    ap.add_argument("--effective-lm-context", type=int, default=2048,
                    help="Rolling-buffer cap. Llama sees last N tokens.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"\n=== Gap decomp: ablate={args.ablate} ===")
    print(f"  ckpt: {args.ckpt}")
    print(f"  effective_lm_context: {args.effective_lm_context}")

    # Load ckpt extras for config restoration
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["extra"]["config"]
    if not isinstance(cfg, TrajMemConfig):
        cfg = TrajMemConfig(**cfg) if isinstance(cfg, dict) else cfg
    cfg.effective_lm_context = args.effective_lm_context

    print(f"  cfg: D={cfg.D} T_window={cfg.T_window} d_lm={cfg.d_lm} "
          f"D_concept={cfg.D_concept} J={cfg.J} K_read={cfg.K_read}")

    print(f"\nLoading IntegratedLM...")
    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
    sd = ck["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"  loaded: {len(sd)} keys  missing={len(missing)}  "
          f"unexpected={len(unexpected)}")
    model = model.to(args.device)
    model.train(False)

    # Apply ablation
    if args.ablate == "scale_zero":
        _zero_scale(model)
    elif args.ablate == "scale_trained_readout_zero":
        _force_readout_zero(model)
    else:
        mil = model._mem_inject_layer()
        eff = mil.scale_max * torch.tanh(mil.scale_raw)
        print(f"  [no ablate] scale_raw mean={mil.scale_raw.mean().item():.4f}  "
              f"effective_scale mean={eff.mean().item():.4f}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    results: dict[str, dict] = {}
    for val_path in args.val_data_paths:
        path = Path(val_path)
        source = path.stem.split(".")[0]
        print(f"\n=== Source: {source} ===")

        ds = LongDocDataset(
            [path],
            chunk_tokens=cfg.D * cfg.T_window,  # 1024
            pad_id=pad_id,
            drop_short=False,
        )

        total_ce = torch.zeros((), device=args.device, dtype=torch.float32)
        total_tokens = 0.0
        total_ans_ce = torch.zeros((), device=args.device, dtype=torch.float32)
        total_ans = 0.0
        n_chunks = 0
        t0 = time.time()

        with torch.no_grad():
            for i, item in enumerate(ds):
                if i >= args.val_batches:
                    break
                if item is None:
                    continue

                chunk = item.input_ids.unsqueeze(0).to(args.device)  # [1, 1024]
                valid_mask = item.valid_mask.to(args.device)  # [1024]

                # Process D=4 windows of T_window=256 each, threading state +
                # prev_window_hiddens. Use rolling-buffer mode so Llama sees
                # only the last effective_lm_context tokens.
                prev_states = torch.zeros(
                    1, cfg.N, cfg.D_concept,
                    dtype=torch.float32, device=args.device,
                )
                prev_hiddens = None
                logits_buf = []
                for w in range(cfg.D):
                    lo = w * cfg.T_window
                    hi = lo + cfg.T_window
                    # Rolling-buffer LM input: last `effective_lm_context` tokens
                    # up to (and including) position hi-1. Since chunk is only
                    # 1024 ≤ 2048, this is just chunk[:, :hi].
                    lm_input = chunk[:, :hi]
                    out = model.forward_window(
                        lm_input_ids=lm_input,
                        prev_window_hiddens=prev_hiddens,
                        prev_states=prev_states,
                        target_mask=None,
                        hard_routing=True,
                        force_surprise=0.0,
                        use_kv_cache=False,
                    )
                    # logits: [1, T_window, V] — but Llama returned logits for
                    # all `lm_input` tokens; we need only the last T_window.
                    # forward_window already slices to the last T_window (see
                    # the function); verify.
                    logits_w = out["logits"]
                    if logits_w.shape[1] != cfg.T_window:
                        # Slice to last T_window
                        logits_w = logits_w[:, -cfg.T_window:, :]
                    logits_buf.append(logits_w)
                    prev_hiddens = out["current_hiddens"]
                    prev_states = out["new_states"]

                logits_full = torch.cat(logits_buf, dim=1)  # [1, 1024, V]
                shift_logits = logits_full[:, :-1, :].contiguous()
                shift_labels = chunk[:, 1:].contiguous()
                shift_mask = valid_mask[1:]

                ce_per_token = torch.nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    reduction="none",
                ).view_as(shift_labels).squeeze(0)

                token_mask = (shift_mask > 0).float()
                token_weight = shift_mask.float() * token_mask
                total_ce = total_ce + (ce_per_token * token_weight).sum().float()
                total_tokens += float(token_weight.sum().item())

                ans_mask = (shift_mask >= 50.0).float()
                if ans_mask.sum() > 0:
                    total_ans_ce = total_ans_ce + (ce_per_token * ans_mask).sum().float()
                    total_ans += float(ans_mask.sum().item())
                n_chunks += 1
                if (i + 1) % 20 == 0:
                    cur = (total_ce / max(total_tokens, 1)).item()
                    print(f"  chunk {i+1}: CE={cur:.4f}  tokens={total_tokens:.0f}")

        dt = time.time() - t0
        full_ce = (total_ce / max(total_tokens, 1)).item()
        ans_ce = (
            (total_ans_ce / max(total_ans, 1)).item()
            if total_ans > 0 else None
        )
        results[source] = {
            "n_chunks": n_chunks,
            "n_tokens_weighted": total_tokens,
            "ntp_ce_full": full_ce,
            "ntp_ce_answer_only": ans_ce,
            "n_answer_tokens_weighted": total_ans,
            "wall_time_s": dt,
        }
        print(f"  {source}: full CE={full_ce:.4f}  ({n_chunks} chunks, {dt:.1f}s)")
        if ans_ce is not None:
            print(f"  {source}: answer-CE={ans_ce:.4f}  ({total_ans:.0f} tokens)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({
            "ablate": args.ablate,
            "ckpt": str(args.ckpt),
            "effective_lm_context": args.effective_lm_context,
            "results": results,
            "config": {"val_batches": args.val_batches,
                       "T_window": cfg.T_window, "D": cfg.D},
        }, f, indent=2)
    print(f"\nSaved: {args.output}")

    print("\n--- Summary ---")
    print(f"{'source':<20} {'full CE':>10} {'answer CE':>12}")
    for src, r in results.items():
        ans = f"{r['ntp_ce_answer_only']:.4f}" if r['ntp_ce_answer_only'] else "—"
        print(f"{src:<20} {r['ntp_ce_full']:>10.4f} {ans:>12}")


if __name__ == "__main__":
    main()
