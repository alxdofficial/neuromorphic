#!/usr/bin/env python3
"""v1h training: composite-QA with teacher-forced CE on answer-content tokens.

For each variant:
 - Encoder ingests a 4096-token context (packed composite_v1 passages) via
   4 × 1024 streaming writes → memory tokens.
 - Decoder forward on [memory, question, answer]. The original context
   tokens are NOT visible to the decoder — only memory carries that info.
 - TF-CE on the answer's content-mask positions (load-bearing tokens; the
   rest of the answer span is filler).

Per-variant outputs in outputs/repr_learning/<out_tag>_<variant>/:
  jsonl/<variant>.jsonl   — per-step training metrics
  ckpts/<variant>.last.pt — encoder + decoder.mask_embed weights
"""
from __future__ import annotations
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.repr_learning.config import ReprConfig
from src.repr_learning.data_qa import make_qa_dataloader, make_mixed_qa_dataloader
from src.repr_learning.decoder import load_frozen_llama
from src.repr_learning.model import ReprLearningModel


REPO = Path(__file__).resolve().parents[2]
COMPOSITE_TRAIN_P = REPO / "data/wave1/composite_v1/train/passages.jsonl"
COMPOSITE_TRAIN_Q = REPO / "data/wave1/composite_v1/train/questions.jsonl"
COMPOSITE_VAL_P   = REPO / "data/wave1/composite_v1/val/passages.jsonl"
COMPOSITE_VAL_Q   = REPO / "data/wave1/composite_v1/val/questions.jsonl"


def lr_at_step(step: int, max_steps: int, base_lr: float, warmup_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(1.0, max(0.0, progress))
    cos = 0.5 * (1 + math.cos(math.pi * progress))
    return base_lr * (0.1 + 0.9 * cos)


def to_device(batch, device):
    for f in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        setattr(batch, f, getattr(batch, f).to(device, non_blocking=True))
    return batch


@torch.no_grad()
def run_val(model, val_dl, device, n_batches: int, window_size: int) -> dict:
    model.train(False)
    losses, accs, per_fam_stats = [], [], {}
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        batch = to_device(batch, device)
        out = model.compute_qa_loss(batch, window_size=window_size)
        losses.append(float(out["loss_recon"]))
        accs.append(float(out["top1_acc"]))
        # Per-family: use per-row loss instead of batch-wide mean (a 2-row
        # batch with rows from families X and Y was previously credited
        # the same mean to both, hiding genuine per-family differences).
        if "per_example_loss" in out:
            per_ex = out["per_example_loss"].detach().cpu().tolist()
        else:
            per_ex = [float(out["loss_recon"])] * len(batch.task_family)
        for fam, l in zip(batch.task_family, per_ex):
            d = per_fam_stats.setdefault(fam, {"n": 0, "loss": 0.0})
            d["n"] += 1
            d["loss"] += float(l)
    model.train(True)
    n = max(len(losses), 1)
    fam_summary = {f: {"n": v["n"], "mean_loss": v["loss"] / max(v["n"], 1)}
                   for f, v in per_fam_stats.items()}
    return {
        "val_loss_recon": sum(losses) / n,
        "val_top1_acc": sum(accs) / n,
        "val_n_batches": len(losses),
        "val_per_family": fam_summary,
    }


def save_checkpoint(model, opt, step, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    def keep(k: str) -> bool:
        if not k.startswith("decoder.llama."):
            return True
        return "lora_" in k

    torch.save({
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
    }, path)


def train_one_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, chunk_size: int, window_size: int,
    passages_per_chunk: int, resume: bool = False,
    use_hotpot: bool = True, use_narrative: bool = True,
    mix_weights: tuple = (0.5, 0.25, 0.25),
) -> dict:
    device = "cuda"
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    is_vanilla = (variant == "vanilla_llama")

    # vanilla_llama has no encoder and no real trainable parameters that
    # affect the loss (mask_embed is gated to 0 contribution for QA).
    # Running a training loop on it just wastes GPU time and rewrites the
    # opt state. Treat it as eval-only: build the val dataloader, run it,
    # write a single final-val row, and return.
    if not is_vanilla:
        opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
        )
    else:
        opt = None

    train_dl = None if is_vanilla else make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_TRAIN_P,
        composite_questions_path=COMPOSITE_TRAIN_Q,
        use_hotpot=use_hotpot, use_narrative=use_narrative,
        split="train",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, num_workers=0, seed=42,
    )
    val_dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_VAL_P,
        composite_questions_path=COMPOSITE_VAL_Q,
        use_hotpot=use_hotpot, use_narrative=use_narrative,
        split="validation",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, num_workers=0, seed=7,
    )

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    if is_vanilla:
        # Eval-only path. Skip any prior jsonl and run a single final-val pass.
        if jsonl_path.exists():
            jsonl_path.unlink()
        t_start = time.time()
        final_val = run_val(model, val_dl, device, val_batches, window_size)
        elapsed = time.time() - t_start
        with open(jsonl_path, "w") as fp:
            fp.write(json.dumps({
                "phase": "val", "step": 0, "variant": variant,
                "final": True, "eval_only": True, **final_val,
            }) + "\n")
        print(f"  [eval-only] vanilla_llama val_recon={final_val['val_loss_recon']:.4f}  "
              f"top1={final_val['val_top1_acc']*100:.1f}%  ({elapsed:.1f}s)", flush=True)
        summary = {
            "variant": variant,
            "trainable_params": n_trainable,
            "n_steps": 0,
            "elapsed_s": elapsed,
            "final_val_loss_recon": final_val["val_loss_recon"],
            "final_val_top1_acc": final_val["val_top1_acc"],
            "final_val_per_family": final_val["val_per_family"],
            "eval_only": True,
        }
        del model
        torch.cuda.empty_cache()
        return summary

    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model_state_dict"], strict=False)
        opt.load_state_dict(sd["optimizer_state_dict"])
        start_step = int(sd.get("step", 0)) + 1
        print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1}")
    else:
        if jsonl_path.exists():
            jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = start_step, t_start

    # Resume: use a local step counter rather than skipping batches from the
    # iterator. The dataloader is seeded but stochastic per worker; replaying
    # start_step batches is wasted compute and the resumed run sees a
    # different stream than the original anyway. Just resume the optimizer
    # state and start consuming fresh batches at start_step.
    step = start_step
    last_completed = start_step - 1  # for the final save (= last step whose body finished)
    for batch in train_dl:
        if step >= n_steps:
            break

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = to_device(batch, device)

        opt.zero_grad()
        out = model.compute_qa_loss(batch, window_size=window_size)
        loss = out["loss"]
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.trainable_parameters(), cfg.grad_clip)
        opt.step()

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"]),
            "loss_recon": float(out["loss_recon"]),
            "loss_aux": float(out["loss_aux"]),
            "top1_acc": float(out["top1_acc"]),
            "n_content_positions": int(out["n_content_positions"]),
            "grad_norm": float(gn),
            "lr": lr,
            "memory_M": out["memory_shape"][1],
        }
        # Splat-variant sublosses (only present when variant == splat_baseline)
        for key in ("splat_aux", "splat_L_pin", "splat_L_prop",
                    "splat_L_adj", "splat_L_sat"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Graph-variant sublosses (only present when variant == graph_baseline)
        for key in ("graph_aux", "graph_L_connect", "graph_L_adjust",
                    "graph_saliency_mean", "graph_endpoint_reuse"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            # Display the variant-specific aux that's actually contributing
            # to the loss: graph_aux for graph_baseline, splat_aux for splat,
            # plain loss_aux (load_balance+orth+z) for everything else.
            if "graph_aux" in out and out["graph_aux"] is not None:
                aux_display = float(out["graph_aux"])
                aux_tag = "g_aux"
            elif "splat_aux" in out and out["splat_aux"] is not None:
                aux_display = float(out["splat_aux"])
                aux_tag = "s_aux"
            else:
                aux_display = float(out["loss_aux"])
                aux_tag = "aux"
            print(f"  step {step:6d}/{n_steps}  recon={float(out['loss_recon']):.4f}  "
                  f"top1={float(out['top1_acc'])*100:5.1f}%  "
                  f"{aux_tag}={aux_display:.3f}  "
                  f"gnorm={float(gn):6.2f}  lr={lr:.2e}  ({sps:.1f} step/s)",
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_dl, device, val_batches, window_size)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            print(f"    [val @ {step}]  recon={vm['val_loss_recon']:.4f}  "
                  f"top1={vm['val_top1_acc']*100:.1f}%",
                  flush=True)

        if step > 0 and step % save_every == 0:
            # `step` here is the last completed step. Resume reads N → starts at N+1.
            save_checkpoint(model, opt, step, ckpt_path)

        last_completed = step
        step += 1

    # Final save: `step` has been incremented past the last completed iter
    # (or equals start_step if the loop never ran). Persist last_completed so
    # resume's `+ 1` lands on the correct next step.
    save_checkpoint(model, opt, last_completed, ckpt_path)
    final_val = run_val(model, val_dl, device, val_batches, window_size)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_recon={final_val['val_loss_recon']:.4f} "
          f"top1={final_val['val_top1_acc']*100:.1f}%", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "final_val_loss_recon": final_val["val_loss_recon"],
        "final_val_top1_acc": final_val["val_top1_acc"],
        "final_val_per_family": final_val["val_per_family"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=[
        "flat_baseline", "continuous_baseline", "memorizing_baseline",
        "recurrent_baseline", "plastic_baseline", "splat_baseline",
        "graph_baseline",
        "vanilla_llama",
    ])
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=10)
    ap.add_argument("--chunk-size", type=int, default=4096)
    ap.add_argument("--window-size", type=int, default=1024)
    ap.add_argument("--passages-per-chunk", type=int, default=0,
                    help="composite_v1 passages sampled per chunk. 0 = auto: "
                         "scales with chunk_size (~75 per 1024 tokens). "
                         "Manual override accepted as positive int.")
    ap.add_argument("--b-diversity-scale", type=float, default=50.0)
    ap.add_argument("--mt-diversity-scale", type=float, default=50.0)
    ap.add_argument("--out-tag", type=str, default="v1h")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-hotpot", action="store_true",
                    help="Disable HotpotQA source (default: enabled)")
    ap.add_argument("--narrative", action="store_true",
                    help="Enable NarrativeQA source (default: DISABLED — most "
                         "examples fall into random-window label noise because "
                         "documents are typically ~75k tokens and the answer "
                         "isn't tokenizer-matched. Re-enable only after building "
                         "an evidence-anchored cached pipeline.)")
    ap.add_argument("--mix-weights", nargs=3, type=float, default=[0.7, 0.3, 0.0],
                    metavar=("COMPOSITE", "HOTPOT", "NARRATIVE"),
                    help="Sampling weights for the three sources. Default has "
                         "narrative at 0 since --narrative is disabled by default.")
    args = ap.parse_args()

    if "v21" in args.variants:
        raise SystemExit("v21 is not supported in v1h yet.")

    # Flag/weight consistency. The flags toggle source *availability*; the
    # weights control *sampling*. A source with weight=0 is never sampled,
    # so enabling the flag without bumping the weight is a no-op that
    # silently loads ~570MB of data (HotpotQA) or downloads NarrativeQA
    # while contributing nothing. Surface this immediately.
    if args.narrative and args.mix_weights[2] <= 0:
        raise SystemExit(
            "--narrative is set but mix_weights[2] (NarrativeQA) is 0. "
            "Either drop --narrative or pass --mix-weights with a positive "
            "third value (e.g. --mix-weights 0.5 0.3 0.2)."
        )
    if (not args.no_hotpot) and args.mix_weights[1] <= 0:
        # HotpotQA is on by default; if user explicitly zeros the weight
        # they likely meant to disable the source entirely.
        raise SystemExit(
            "HotpotQA is enabled but mix_weights[1] is 0. Either pass "
            "--no-hotpot or set --mix-weights with a positive second value."
        )
    if args.mix_weights[0] <= 0:
        raise SystemExit(
            "Composite (mix_weights[0]) is 0. composite_v1 is the primary "
            "source and cannot be disabled."
        )

    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        max_steps=args.steps,
        warmup_steps=500,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=36,
        edge_token_packing="fused",
        b_diversity_scale=args.b_diversity_scale,
        mt_diversity_scale=args.mt_diversity_scale,
        d_mamba=768,
    )

    # Auto-scale composite passages_per_chunk with chunk_size if user passed 0.
    # composite_v1 passages average ~13 tokens; we target ~75 passages per
    # 1024 chunk tokens so the chunk fills to ~95% even after rejecting
    # over-long candidates.
    if args.passages_per_chunk <= 0:
        args.passages_per_chunk = max(75, (args.chunk_size // 1024) * 75)
        print(f"[auto] composite passages_per_chunk = {args.passages_per_chunk} "
              f"(scaled for chunk_size={args.chunk_size})")

    print(f"v1h config: chunk={args.chunk_size}, window={args.window_size}, "
          f"passages_per_chunk={args.passages_per_chunk}")
    print(f"Bottleneck (baselines): {cfg.n_flat_codes} × {cfg.d_continuous} "
          f"= {cfg.n_flat_codes * cfg.d_continuous} floats")
    print(f"Steps: {args.steps}, batch={cfg.batch_size}")

    print(f"\nLoading tokenizer {cfg.llama_model}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.llama_model)
    print("Loading Llama (shared across variants, frozen)...")
    llama, _ = load_frozen_llama(cfg.llama_model, dtype=torch.bfloat16)

    summaries = []
    for variant in args.variants:
        out_dir = REPO / f"outputs/repr_learning/{args.out_tag}_{variant}"
        out_dir.mkdir(parents=True, exist_ok=True)
        s = train_one_variant(
            variant=variant, llama=llama, tokenizer=tokenizer, cfg=cfg,
            n_steps=args.steps, log_every=args.log_every,
            val_every=args.val_every, save_every=args.save_every,
            val_batches=args.val_batches, out_dir=out_dir,
            chunk_size=args.chunk_size, window_size=args.window_size,
            passages_per_chunk=args.passages_per_chunk,
            resume=args.resume,
            use_hotpot=not args.no_hotpot,
            use_narrative=args.narrative,
            mix_weights=tuple(args.mix_weights),
        )
        summaries.append(s)

    print("\n" + "=" * 78)
    print("v1h SUMMARY")
    print("=" * 78)
    print(f"  {'variant':<25}{'params':>12}{'final_recon':>13}{'top1':>8}{'time(min)':>12}")
    print("  " + "-" * 70)
    for s in summaries:
        print(f"  {s['variant']:<25}{s['trainable_params']:>12,}"
              f"{s['final_val_loss_recon']:>13.4f}"
              f"{s['final_val_top1_acc']*100:>7.1f}%"
              f"{s['elapsed_s']/60:>12.1f}")

    summary_path = REPO / f"outputs/repr_learning/{args.out_tag}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nWrote {summary_path}")


if __name__ == "__main__":
    main()
