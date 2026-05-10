"""Wave 1 entry point — long-doc TF NTP pretraining (plan §4.5).

Reads pre-tokenized parquet from preprocess_longdoc.py (and optionally
synthesize_needle.py), packs into D*T_window chunks, runs cross-window
TBPTT with Phase1Trainer.

Features:
  - LR warmup + cosine decay
  - Gradient clipping
  - Checkpoint save / resume (model + optimizer + scheduler + RNG state)
  - Per-step metrics (loss, grad_norm, LR)

Usage:
    python scripts/train_wave1.py \\
        --data-paths \\
            data/wave1/fineweb_edu.parquet \\
            data/wave1/wikipedia_en.parquet \\
            data/wave1/slimpajama_6b.parquet \\
            data/wave1/needle.parquet \\
        --batch-size 1 --num-steps 1000 \\
        --checkpoint-out outputs/wave1/ckpt.pt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.trajectory_memory.config import TrajMemConfig
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.integrated_lm import IntegratedLM
from src.trajectory_memory.training.metrics import (
    grad_norms_by_component,
    surprise_stats,
    trajectory_diversity_stats,
    vram_stats,
)
from src.trajectory_memory.training.plotting import (
    save_training_plots,
    dump_history_json,
)
from src.trajectory_memory.training import (
    Phase1Trainer,
    WarmupCosineScheduler,
    build_optimizer,
    capture_rng_state,
    load_checkpoint,
    save_checkpoint,
)
from src.trajectory_memory.training.loaders import (
    BatchedLongDocDataset, LongDocDataset,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-paths", nargs="+", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Multi-stream BS (Tier 4 #13). Each batch slot "
                         "is an independent doc with its own state lifecycle. "
                         "BS=4 is bench-optimal on RTX 4090 (15 GB peak with "
                         "KV cache); BS=8 OOMs. BS=1 falls back to the "
                         "single-stream LongDocDataset.")
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--warmup-steps", type=int, default=100)
    # Tier 2 #8 — peak LR halved from 3e-4 → 1.5e-4 for memory params.
    # The 3e-4 peak appeared to over-shoot in the prior 10k-step run
    # (val regressed mid-training as LR was decaying through the high
    # range). 1.5e-4 with longer warmup gives a more stable trajectory.
    ap.add_argument("--lr-memory", type=float, default=1.5e-4)
    ap.add_argument("--lr-adapter", type=float, default=5e-5)
    ap.add_argument("--lr-min-ratio", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--config-tier", choices=["small", "medium", "large"],
                    default="medium")
    ap.add_argument("--model-name", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoint-in", type=Path, default=None)
    ap.add_argument("--warm-start", action="store_true",
                    help="Load only model weights from --checkpoint-in; "
                         "do NOT restore optimizer state, scheduler state, "
                         "or step count. Use when starting a NEW wave from "
                         "a previous wave's checkpoint (default behavior is "
                         "full resume — appropriate only when continuing the "
                         "same wave's training).")
    ap.add_argument("--checkpoint-out", type=Path, default=None)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--val-data-paths", nargs="+", type=Path, default=None,
                    help="held-out val parquets (e.g., needle.val.parquet for "
                         "memory-bridging probe). If set, eval at each save.")
    ap.add_argument("--val-batches", type=int, default=100,
                    help="number of val batches to average per eval pass. "
                         "Tier 2 #6 — bumped from 20 to 100 because 20-batch "
                         "vals had ±0.5 noise band, which is larger than any "
                         "real signal we'd expect over a single save interval.")
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="Disable torch.compile (default ON). Compile gives "
                         "~28% speedup at BS=2 with a ~2 min cold-start; "
                         "disable for fast smoke iteration where startup time "
                         "matters more than steady-state speed.")
    ap.add_argument("--no-kv-cache", dest="use_kv_cache", action="store_false",
                    help="Disable sliding KV cache (default ON). KV cache "
                         "gives ~1.79× speedup on Phase 1 by skipping the "
                         "rolling LM buffer re-encode per window. Disable "
                         "only to benchmark the rolling-buffer fallback path.")
    ap.set_defaults(compile=True, use_kv_cache=True)
    ap.add_argument("--plot-path", type=Path, default=None,
                    help="If set, save a multi-panel diagnostic plot here "
                         "every --plot-every-seconds. PNG; overwritten in "
                         "place. Companion .json dump is written next to it.")
    ap.add_argument("--plot-every-seconds", type=float, default=180.0,
                    help="Seconds between plot refreshes (default 180 = 3 min).")
    args = ap.parse_args()

    # Allow TF32 for fp32 matmul (memory params, bridge, lm_head). On Ampere
    # and later this trades trivial precision for a meaningful speedup; the
    # bf16 backbone is unaffected.
    torch.set_float32_matmul_precision("high")

    cfg = getattr(TrajMemConfig, args.config_tier)()
    print(f"Config tier: {args.config_tier}")
    print(f"  N={cfg.N}, J={cfg.J}, K_read={cfg.K_read}, D={cfg.D}, T_window={cfg.T_window}")

    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id

    model = IntegratedLM(cfg, model_name=args.model_name, attach_lm=True).to(args.device)
    # NOTE: HF Llama's gradient_checkpointing IS INCOMPATIBLE with
    # use_cache=True / past_key_values. Enabling it forces use_cache=False
    # silently and `past_key_values` returns None — defeating our KV cache
    # entirely. (Verified: HF prints "use_cache=True is incompatible with
    # gradient checkpointing. Setting use_cache=False.")
    # We pick KV cache (1.79× speedup, 5 GB less mem in production) over
    # gradient checkpointing. With BS=1 (W1's assertion), activation memory
    # isn't the bottleneck. If config-tier=large or BS>1 ever needs it,
    # add `--no-kv-cache --grad-checkpoint` as a separate path.

    if args.compile:
        # dynamic=True: forward_window's lm_input_ids varies in length as the
        # rolling buffer grows (256, 512, 768, ..., up to effective_lm_context).
        # With dynamic=False we recompile per shape and hit dynamo's
        # recompile_limit (8) by chunk 3. dynamic=True trades 6% per-step
        # speed for stable performance across chunks. See
        # `scripts/experiment_compile_dynamic.py`.
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=True,
        )
        print("Compiled model.forward_window (cold-start on first step ~1-3 min).")

    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    trainer = Phase1Trainer(
        model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        pad_token_id=tokenizer.pad_token_id,
        use_kv_cache=args.use_kv_cache,
        prior_loss_weight=0.0,  # W1 has no prior/response distinction
    )
    if args.use_kv_cache:
        print("KV cache enabled — Llama re-encodes only new T_window tokens per window.")

    if args.checkpoint_in is not None:
        if args.warm_start:
            # Warm-start: load model weights only. Optimizer/scheduler/step
            # stay fresh — appropriate when starting a new wave from a
            # previous wave's checkpoint (the LR schedule for THIS wave
            # should run from step 0).
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=None, scheduler=None,
                map_location=args.device,
            )
            print(f"Warm-started model from {args.checkpoint_in} "
                  f"(optimizer/scheduler/step reset to fresh)")
        else:
            # Full resume: restore everything.
            ckpt = load_checkpoint(
                args.checkpoint_in, model=model,
                optimizer=optimizer, scheduler=scheduler,
                map_location=args.device,
            )
            trainer.load_state_dict({"step_count": ckpt.get("step", 0)})
            # Restore RNG state for reproducibility on resume (was saved
            # but never restored before).
            from src.trajectory_memory.training.checkpoint import restore_rng_state
            if "rng_state" in ckpt:
                restore_rng_state(ckpt["rng_state"])
            print(f"Resumed from {args.checkpoint_in} at step {ckpt.get('step')} "
                  f"(optimizer/scheduler/RNG restored)")

    # Multi-stream batching landed (Tier 4 #13). BS > 1 uses
    # `BatchedLongDocDataset` which maintains BS parallel slots, each
    # with its own current-doc + chunk position, sharing a single
    # source-mix pool. The trainer applies per-slot state reset using
    # `torch.where` on the `is_doc_start_per_slot` mask (see the train
    # loop below).
    #
    # KV cache + multi-stream uses LOCKSTEP RESET: when ANY slot's doc
    # boundary fires, reset the WHOLE batched cache. This costs the
    # other slots their cache for a few windows (until refilled), but
    # is much simpler than per-slot cache lifecycle (which would need
    # custom attention_mask + per-slot RoPE positions). At BS=4, ~24%
    # of steps trigger a whole-cache reset; other 76% get full KV
    # cache benefit. Net throughput is well above rolling-buffer mode
    # which OOMs at BS=4.

    if args.batch_size == 1:
        dataset = LongDocDataset(
            args.data_paths,
            chunk_tokens=cfg.D * cfg.T_window,
            pad_id=pad_id, drop_short=False,
        )
    else:
        dataset = BatchedLongDocDataset(
            args.data_paths,
            batch_size=args.batch_size,
            chunk_tokens=cfg.D * cfg.T_window,
            pad_id=pad_id, drop_short=False,
        )
        print(f"Multi-stream BS={args.batch_size} dataset enabled.")
    # B6 fix — on resume, advance dataset epoch counter by step_count so
    # resumed training doesn't replay the head of the shuffle order. Each
    # epoch's shuffle uses seed + epoch (B2 fix), so bumping epoch gives a
    # fresh shuffle that doesn't overlap the pre-resume one. Approximate —
    # we don't truly seek to step N within an epoch — but eliminates the
    # head-overtraining/tail-undertraining pathology.
    if trainer.step_count > 0:
        dataset._epoch = trainer.step_count
        print(f"Dataset epoch advanced to {trainer.step_count} for resume "
              f"(avoids replaying head of shuffle order)")

    # Per-source val datasets — each parquet path gets its own dataset
    # and its own loss series. Lets us plot the needle val loss
    # specifically (memory-bridging probe) alongside other sources.
    val_datasets: dict[str, LongDocDataset] = {}
    if args.val_data_paths:
        for p in args.val_data_paths:
            # Source label = stem without trailing ".val" if present.
            label = Path(p).stem.replace(".val", "")
            val_datasets[label] = LongDocDataset(
                [p],
                chunk_tokens=cfg.D * cfg.T_window,
                pad_id=pad_id, drop_short=False,
            )
        print(f"Validation: {len(val_datasets)} source(s) "
              f"({', '.join(val_datasets.keys())}), "
              f"{args.val_batches} batches per eval per source")

    def run_val_per_source() -> dict[str, float]:
        """Returns {source_label: mean_val_loss}.

        N9 fix: state THREADS across chunks of the same val doc, just like
        training. Without this, val on long docs (especially needle-haystack
        with planted facts >2K tokens away) couldn't measure cross-chunk
        memory ability — eval would invalidate the architecture's whole point.
        Reset on `is_doc_start` (matches training behavior).
        """
        out: dict[str, float] = {}
        for label, vds in val_datasets.items():
            losses_v: list[float] = []
            v_prev_states = None
            v_prev_hiddens = None
            v_prev_lm_ctx = None
            v_past_kv = None
            v_cache_abs = 0
            for i, item in enumerate(vds):
                if i >= args.val_batches:
                    break
                if item.is_doc_start:
                    v_prev_states = None
                    v_prev_hiddens = None
                    v_prev_lm_ctx = None
                    v_past_kv = None
                    v_cache_abs = 0
                chunk_v = item.input_ids.unsqueeze(0).to(args.device)
                valid_mask_v = item.valid_mask.unsqueeze(0).to(args.device)
                target_mask_v = valid_mask_v.view(1, cfg.D, cfg.T_window)
                ev = trainer.eval_wave1(
                    chunk_v,
                    prev_states=v_prev_states,
                    prev_window_hiddens=v_prev_hiddens,
                    prev_lm_context=v_prev_lm_ctx,
                    target_mask=target_mask_v,
                    past_key_values=v_past_kv,
                    cache_abs_pos=v_cache_abs,
                )
                losses_v.append(ev["loss"])
                v_prev_states = ev["final_states"]
                v_prev_hiddens = ev["final_hiddens"]
                v_prev_lm_ctx = ev["final_lm_context"]
                v_past_kv = ev["final_past_key_values"]
                v_cache_abs = ev["final_cache_abs_pos"]
            out[label] = sum(losses_v) / max(len(losses_v), 1)
        return out

    print(f"Starting Wave 1 training: {args.num_steps} steps "
          f"(starting from step {trainer.step_count})")
    losses: list = []
    t_start = time.time()
    last_plot_t = time.time()
    # Best-checkpoint tracking — `--checkpoint-out` is a rolling save (latest
    # weights), but the model can have a high-variance val trajectory and we
    # don't want to lose the best intermediate. `ckpt.best.pt` is rewritten
    # only when the val score improves. Score = first val source's loss
    # (typically `needle` since it goes first in --val-data-paths).
    best_val_loss: float = float("inf")
    best_ckpt_path = (
        args.checkpoint_out.with_name(args.checkpoint_out.stem + ".best.pt")
        if args.checkpoint_out is not None else None
    )
    # Live history dict — populated each step, plotted every plot-every-seconds.
    history: dict = {
        "step": [],
        "loss": [],
        "grad_norm": [],
        "lr": [],
        "surprise_mean": [],
        "surprise_std": [],
        "tok_per_sec": [],
        "vram_peak_gb": [],
        "val_step": [],
        "val_loss": {},  # {source: [vals indexed by val_step]}
        # Per-source train loss series. Updated each step from item.source.
        # Lets us see whether needle loss is dropping faster than fineweb,
        # which is the actual diagnostic for memory-bridging behavior.
        "loss_by_source": {},  # {source: [(step, loss), ...]}
    }
    # Per-component grad-norm series initialized lazily once we see them.
    # Cross-chunk state for the single batch slot.
    prev_states = None
    prev_window_hiddens = None
    prev_lm_context = None
    past_kv = None  # KV-cache mode only: carries across chunks of the same doc
    cache_abs_pos = 0  # N1: absolute position counter for RoPE correctness
    last_step_t = time.time()

    while trainer.step_count < args.num_steps:
        for item in dataset:
            if trainer.step_count >= args.num_steps:
                break

            # Branch on single-stream vs multi-stream batch shape.
            if args.batch_size == 1:
                # ── Single-stream path ───────────────────────────────
                if item.is_doc_start:
                    prev_states = None
                    prev_window_hiddens = None
                    prev_lm_context = None
                    past_kv = None
                    cache_abs_pos = 0
                chunk = item.input_ids.unsqueeze(0).to(args.device)         # [1, T]
                valid_mask = item.valid_mask.unsqueeze(0).to(args.device)   # [1, T]
                target_mask = valid_mask.view(1, cfg.D, cfg.T_window)
                step_sources = [getattr(item, "source", "") or "unknown"]
            else:
                # ── Multi-stream path (BS > 1) ───────────────────────
                # Per-slot reset: where is_doc_start_per_slot[i] is True,
                # zero out slot i's state. `prev_states` is naturally
                # batched [BS, N, D]; we use torch.where to selectively
                # reset only the slots that need it. The shared
                # rolling-buffer prev_lm_context is reset whenever ANY
                # slot needs reset (per-slot lm_buffer would require
                # custom attention_mask plumbing — deferred).
                BS_chunk = item.input_ids.shape[0]
                is_start = item.is_doc_start_per_slot.to(args.device)  # [BS]

                # Initialize state on very first iteration.
                if prev_states is None:
                    prev_states = model.manifold.reset_states(
                        batch_size=BS_chunk,
                    ).to(args.device)

                if is_start.any():
                    fresh_states = model.manifold.reset_states(
                        batch_size=BS_chunk,
                    ).to(args.device)
                    prev_states = torch.where(
                        is_start[:, None, None], fresh_states, prev_states,
                    )
                    if prev_window_hiddens is not None:
                        prev_window_hiddens = torch.where(
                            is_start[:, None, None],
                            torch.zeros_like(prev_window_hiddens),
                            prev_window_hiddens,
                        )
                    # Whole-buffer reset on any slot reset (simple +
                    # correct; non-resetting slots will rebuild their
                    # buffer over the next ~8 windows).
                    prev_lm_context = None
                    # KV cache disabled in multi-stream mode (forced
                    # above); cache_abs_pos stays 0.
                    past_kv = None
                    cache_abs_pos = 0

                chunk = item.input_ids.to(args.device)            # [BS, T]
                valid_mask = item.valid_mask.to(args.device)      # [BS, T]
                target_mask = valid_mask.view(BS_chunk, cfg.D, cfg.T_window)
                step_sources = list(item.sources)

            metrics = trainer.step_wave1(
                chunk,
                prev_states=prev_states,
                prev_window_hiddens=prev_window_hiddens,
                prev_lm_context=prev_lm_context,
                target_mask=target_mask,
                past_key_values=past_kv,
                cache_abs_pos=cache_abs_pos,
            )
            prev_states = metrics.final_states
            prev_window_hiddens = metrics.final_hiddens
            prev_lm_context = metrics.final_lm_context
            past_kv = metrics.final_past_key_values
            cache_abs_pos = metrics.final_cache_abs_pos
            losses.append(metrics.loss)

            # Per-source train loss tracking. In multi-stream mode the
            # step's loss is averaged across all BS slots' sources, so
            # attribute that single loss value to each contributing
            # source (lossy but informative for trend tracking).
            for src_label in step_sources:
                history["loss_by_source"].setdefault(src_label, []).append(
                    (trainer.step_count, metrics.loss),
                )

            # S5 fix — NaN-loss kill switch + grad-spike alert.
            import math, sys
            if not math.isfinite(metrics.loss):
                print(f"FATAL: non-finite loss ({metrics.loss}) at step "
                      f"{trainer.step_count}. Aborting.", file=sys.stderr)
                sys.exit(1)
            if not math.isfinite(metrics.grad_norm):
                print(f"FATAL: non-finite grad_norm ({metrics.grad_norm}) at "
                      f"step {trainer.step_count}. Aborting.", file=sys.stderr)
                sys.exit(1)
            # Grad-spike alert: warn if grad_norm > 5× recent median (last 100).
            recent_gn = history.get("grad_norm", [])[-100:]
            if len(recent_gn) >= 20:
                med = sorted(recent_gn)[len(recent_gn) // 2]
                if med > 0 and metrics.grad_norm > 5 * med:
                    print(f"  WARN step {trainer.step_count}: grad_norm "
                          f"{metrics.grad_norm:.2f} = {metrics.grad_norm/med:.1f}× "
                          f"recent median {med:.2f}")

            # ── Per-step metric collection for live monitoring ─────
            step = trainer.step_count
            now = time.time()
            history["step"].append(step)
            history["loss"].append(metrics.loss)
            history["grad_norm"].append(metrics.grad_norm)
            history["lr"].append(list(metrics.lr))
            # B8 — inject SNR (memory-contribution diagnostic).
            inj = metrics.inject_norm
            hid = metrics.hidden_norm
            history.setdefault("inject_norm", []).append(inj)
            history.setdefault("hidden_norm", []).append(hid)
            history.setdefault("inject_snr", []).append(inj / max(hid, 1e-9))
            # B9 — trajectory diversity (read + write paths).
            if metrics.read_visited_ids is not None:
                rs = trajectory_diversity_stats(metrics.read_visited_ids, cfg.N)
                history.setdefault("read_unique_frac", []).append(rs["unique_frac"])
                history.setdefault("read_self_overlap", []).append(rs["self_overlap_rate"])
            if metrics.write_visited_ids is not None:
                ws = trajectory_diversity_stats(metrics.write_visited_ids, cfg.N)
                history.setdefault("write_unique_frac", []).append(ws["unique_frac"])
                history.setdefault("write_self_overlap", []).append(ws["self_overlap_rate"])
            # Surprise distribution (per-window NTP CE — should drop as
            # memory learns to bridge the LM cap).
            if metrics.surprise_history is not None:
                ss = surprise_stats(metrics.surprise_history)
                history["surprise_mean"].append(ss["mean"])
                history["surprise_std"].append(ss["std"])
            else:
                history["surprise_mean"].append(0.0)
                history["surprise_std"].append(0.0)
            # Per-component grad norms (safe: grads still populated from
            # this step; zero_grad in the NEXT step will clear).
            try:
                comp_norms = grad_norms_by_component(model)
                for comp, val in comp_norms.items():
                    history.setdefault(f"grad_norm_{comp}", []).append(val)
            except Exception:
                pass
            # Throughput (tok/s for this step).
            # Total tokens this step = BS × chunk_tokens (was chunk.shape[1]
            # only — bug for BS>1 since it ignored the batch dim).
            chunk_tokens_total = chunk.shape[0] * chunk.shape[1]
            dt = max(now - last_step_t, 1e-6)
            history["tok_per_sec"].append(chunk_tokens_total / dt)
            last_step_t = now
            # VRAM peak.
            v = vram_stats()
            history["vram_peak_gb"].append(v["peak_gb"])

            step = trainer.step_count
            if step % args.log_every == 0:
                avg = sum(losses[-args.log_every:]) / max(len(losses[-args.log_every:]), 1)
                elapsed = time.time() - t_start
                lrs = " ".join(f"{lr:.2e}" for lr in metrics.lr)
                print(f"  step {step:>5}  loss={metrics.loss:.4f}  avg10={avg:.4f}  "
                      f"grad_norm={metrics.grad_norm:.2f}  lr=[{lrs}]  "
                      f"({elapsed/max(step, 1):.2f}s/step)")

            if step > 0 and step % args.save_every == 0:
                cur_val = None
                if val_datasets:
                    per_source = run_val_per_source()
                    history["val_step"].append(step)
                    for src, v in per_source.items():
                        history["val_loss"].setdefault(src, []).append(v)
                    pretty = "  ".join(f"{s}={v:.3f}" for s, v in per_source.items())
                    print(f"  step {step:>5}  val: {pretty}")
                    # Best-val score = first source's loss (typically `needle`,
                    # the memory-bridging probe). Falls back to mean if needed.
                    cur_val = next(iter(per_source.values()), None)
                if args.checkpoint_out is not None:
                    save_checkpoint(
                        args.checkpoint_out,
                        model=model, optimizer=optimizer, scheduler=scheduler,
                        step=step,
                        rng_state=capture_rng_state(),
                        extra={"config": cfg.__dict__, "losses": losses,
                               "best_val_loss": best_val_loss},
                    )
                    print(f"  saved checkpoint to {args.checkpoint_out} at step {step}")
                    # Best-checkpoint tracking — overwrite ckpt.best.pt only
                    # when val score improves. Lets us recover the best
                    # intermediate weights even if late training degrades them.
                    if cur_val is not None and cur_val < best_val_loss:
                        best_val_loss = cur_val
                        save_checkpoint(
                            best_ckpt_path,
                            model=model, optimizer=optimizer, scheduler=scheduler,
                            step=step,
                            rng_state=capture_rng_state(),
                            extra={"config": cfg.__dict__, "losses": losses,
                                   "best_val_loss": best_val_loss,
                                   "best_val_source": next(iter(per_source.keys()))},
                        )
                        print(f"  ⭐ new best val={best_val_loss:.4f} → {best_ckpt_path}")

            # ── Live plot refresh (time-based, default 3 min) ─────
            if args.plot_path is not None:
                if now - last_plot_t > args.plot_every_seconds:
                    save_training_plots(history, args.plot_path)
                    json_path = args.plot_path.with_suffix(".json")
                    dump_history_json(history, json_path)
                    last_plot_t = now
                    print(f"  step {step:>5}  plot saved to {args.plot_path} "
                          f"(history: {json_path})")

    if args.checkpoint_out is not None:
        save_checkpoint(
            args.checkpoint_out,
            model=model, optimizer=optimizer, scheduler=scheduler,
            step=trainer.step_count,
            rng_state=capture_rng_state(),
            extra={"config": cfg.__dict__, "losses": losses},
        )
        print(f"Final checkpoint saved to {args.checkpoint_out}")


if __name__ == "__main__":
    main()
