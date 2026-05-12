#!/usr/bin/env python3
"""Scale sweep: how does bulk-token CE change as we vary scale_raw at inference?

Tells us whether the bridge "knew" how to gate (CE monotonically decreasing
as scale → 0 → bridge couldn't optimize) or whether memory provides some
signal at the trained scale (non-monotonic → readout is partially useful).

Multiplies the TRAINED scale_raw by a sweep factor f ∈ {0, 0.25, 0.5, 0.75,
1.0, 1.5} before running val.
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


def _set_scale_factor(model, factor: float, original_scale_raw: torch.Tensor):
    mil = model._mem_inject_layer()
    with torch.no_grad():
        mil.scale_raw.copy_(original_scale_raw * factor)
    eff = mil.scale_max * torch.tanh(mil.scale_raw)
    return float(eff.abs().mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-data-paths", nargs="+", required=True)
    ap.add_argument("--val-batches", type=int, default=100)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--factors", nargs="+", type=float,
                    default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/scale_sweep.json"))
    args = ap.parse_args()

    print(f"\n=== Scale sweep: factors={args.factors} ===")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ck["extra"]["config"]
    if isinstance(cfg, dict):
        cfg = TrajMemConfig(**cfg)
    cfg.effective_lm_context = 2048

    model = IntegratedLM(cfg, model_name="meta-llama/Llama-3.2-1B")
    model.load_state_dict(ck["model_state_dict"], strict=False)
    model = model.to(args.device)
    model.train(False)

    # Snapshot the trained scale_raw so we can multiply by factor each iter
    mil = model._mem_inject_layer()
    original_scale_raw = mil.scale_raw.detach().clone()
    print(f"  trained scale_raw: mean={original_scale_raw.mean():.4f}  "
          f"std={original_scale_raw.std():.4f}")
    eff_orig = mil.scale_max * torch.tanh(original_scale_raw)
    print(f"  effective_scale @ trained: mean={eff_orig.mean():.4f}")

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

    sweep_results: list[dict] = []
    for factor in args.factors:
        eff_mean = _set_scale_factor(model, factor, original_scale_raw)
        per_source: dict[str, dict] = {}

        for val_path in args.val_data_paths:
            path = Path(val_path)
            source = path.stem.split(".")[0]
            ds = LongDocDataset(
                [path], chunk_tokens=cfg.D * cfg.T_window,
                pad_id=pad_id, drop_short=False,
            )
            total_ce = torch.zeros((), device=args.device, dtype=torch.float32)
            total_tokens = 0.0
            total_ans_ce = torch.zeros((), device=args.device, dtype=torch.float32)
            total_ans = 0.0
            t0 = time.time()
            with torch.no_grad():
                for i, item in enumerate(ds):
                    if i >= args.val_batches:
                        break
                    if item is None:
                        continue
                    chunk = item.input_ids.unsqueeze(0).to(args.device)
                    valid_mask = item.valid_mask.to(args.device)
                    prev_states = torch.zeros(1, cfg.N, cfg.D_concept,
                                              dtype=torch.float32, device=args.device)
                    prev_hiddens = None
                    logits_buf = []
                    for w in range(cfg.D):
                        hi = (w + 1) * cfg.T_window
                        out = model.forward_window(
                            lm_input_ids=chunk[:, :hi],
                            prev_window_hiddens=prev_hiddens,
                            prev_states=prev_states,
                            target_mask=None, hard_routing=True,
                            force_surprise=0.0, use_kv_cache=False,
                        )
                        logits_w = out["logits"]
                        if logits_w.shape[1] != cfg.T_window:
                            logits_w = logits_w[:, -cfg.T_window:, :]
                        logits_buf.append(logits_w)
                        prev_hiddens = out["current_hiddens"]
                        prev_states = out["new_states"]
                    logits_full = torch.cat(logits_buf, dim=1)
                    shift_logits = logits_full[:, :-1, :].contiguous()
                    shift_labels = chunk[:, 1:].contiguous()
                    shift_mask = valid_mask[1:]
                    ce_per_token = torch.nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        reduction="none",
                    ).view_as(shift_labels).squeeze(0)
                    token_weight = shift_mask.float() * (shift_mask > 0).float()
                    total_ce = total_ce + (ce_per_token * token_weight).sum().float()
                    total_tokens += float(token_weight.sum().item())
                    ans_mask = (shift_mask >= 50.0).float()
                    if ans_mask.sum() > 0:
                        total_ans_ce = total_ans_ce + (ce_per_token * ans_mask).sum().float()
                        total_ans += float(ans_mask.sum().item())
            full_ce = (total_ce / max(total_tokens, 1)).item()
            ans_ce = ((total_ans_ce / max(total_ans, 1)).item()
                      if total_ans > 0 else None)
            per_source[source] = {
                "ntp_ce_full": full_ce, "ntp_ce_answer_only": ans_ce,
                "n_tokens_weighted": total_tokens,
                "n_answer_tokens_weighted": total_ans,
                "wall_time_s": time.time() - t0,
            }
        # weighted
        tot = sum(per_source[s]["n_tokens_weighted"] * per_source[s]["ntp_ce_full"]
                  for s in per_source)
        n = sum(per_source[s]["n_tokens_weighted"] for s in per_source)
        weighted = tot / max(n, 1)
        sweep_results.append({
            "factor": factor, "effective_scale_mean": eff_mean,
            "weighted_ce": weighted, "per_source": per_source,
        })
        print(f"\nfactor={factor:.2f}  eff_scale={eff_mean:.4f}  "
              f"weighted CE={weighted:.4f}")
        for s, r in per_source.items():
            ans = (f"  ans={r['ntp_ce_answer_only']:.4f}"
                   if r.get('ntp_ce_answer_only') else "")
            print(f"  {s:<20} full={r['ntp_ce_full']:.4f}{ans}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"ckpt": str(args.ckpt), "factors": args.factors,
                   "sweep": sweep_results}, f, indent=2)
    print(f"\nSaved: {args.output}")

    print("\n--- Summary ---")
    print(f"{'factor':>8} {'eff_scale':>12} {'weighted CE':>14}")
    for r in sweep_results:
        print(f"{r['factor']:>8.2f} {r['effective_scale_mean']:>12.4f} "
              f"{r['weighted_ce']:>14.4f}")


if __name__ == "__main__":
    main()
