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
    concept_state_drift,
    effective_lr_by_component,
    grad_norms_by_component,
    logit_stats,
    param_norms_by_component,
    routing_entropy,
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
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Multi-stream BS. Each batch slot is an "
                         "independent doc with its own state lifecycle. "
                         "BS=8 with --compile is bench-optimal on RTX 4090 "
                         "(~12 GB peak, 27k tok/s). BS=12 fits but compile "
                         "regression on that shape; BS=16 OOMs. BS=1 falls "
                         "back to single-stream LongDocDataset.")
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
    ap.set_defaults(compile=True)
    ap.add_argument("--plot-path", type=Path, default=None,
                    help="If set, save a multi-panel diagnostic plot here "
                         "every --plot-every-seconds. PNG; overwritten in "
                         "place. Companion .json dump is written next to it.")
    ap.add_argument("--plot-every-seconds", type=float, default=180.0,
                    help="Seconds between plot refreshes (default 180 = 3 min).")
    # Routing-collapse mitigations (canonical Switch-Transformer / ST-MoE /
    # VQ-VAE values). The prior Wave 1 run plateaued because routing
    # saturated → entry_proj gradient died → manifold stuck. These four
    # knobs implement the layered defense the literature converged on.
    ap.add_argument("--load-balance-coef", type=float, default=1e-2,
                    help="Switch-Transformer aux-loss coefficient. 0 disables.")
    ap.add_argument("--z-loss-coef", type=float, default=1e-3,
                    help="ST-MoE router z-loss coefficient. 0 disables.")
    ap.add_argument("--revive-every", type=int, default=500,
                    help="VQ-VAE dead-code revival cadence (steps). 0 disables.")
    ap.add_argument("--revive-threshold", type=float, default=1e-5,
                    help="usage_ema below this counts a concept as dead.")
    ap.add_argument("--tau-init", type=float, default=1.0,
                    help="Gumbel temperature at step 0.")
    ap.add_argument("--tau-floor", type=float, default=0.5,
                    help="Gumbel temperature floor (saturation guard).")
    ap.add_argument("--tau-decay-rate", type=float, default=1e-4,
                    help="τ = max(floor, init·exp(-rate·step)).")
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
    # NOTE: HF Llama's gradient_checkpointing is incompatible with use_cache=True
    # (silently sets use_cache=False, defeating our KV cache). We pick KV
    # cache (1.79× speedup, ~3 GB less mem) over checkpointing.

    if args.compile:
        # dynamic=False: KV-cache mode (default) means every forward_window
        # call passes lm_input_ids of fixed length T_window=256 — no shape
        # variation, so no recompile cliff. dynamic=True was needed when
        # we used rolling-buffer mode (lm_input_ids grew 256/512/.../2048
        # across windows of a chunk), but with the KV cache restored
        # (2026-05-11) the rolling-buffer path is only the fallback.
        # dynamic=True also triggers an AOT autograd partitioner bug with
        # our combination of bf16 autocast + activation checkpointing on
        # trajectory generators: `AssertionError: Node add_NNNN was
        # invalid, but is output` from `min_cut_rematerialization_partition`.
        # dynamic=False avoids the bug and is faster.
        model.forward_window = torch.compile(
            model.forward_window, mode="default", dynamic=False,
        )
        print("Compiled model.forward_window (cold-start on first step ~1-3 min).")

    optimizer = build_optimizer(model, lr_memory=args.lr_memory, lr_adapter=args.lr_adapter)
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps=args.warmup_steps,
        total_steps=args.num_steps,
        lr_min_ratio=args.lr_min_ratio,
    )
    # use_kv_cache=True with NO doc-boundary resets (Transformer-XL style;
    # see the multi-stream branch above). KV cache trims to
    # effective_lm_context=2048 naturally; cross-doc attention is permitted
    # and self-cleans as old-doc K/V drops out of the cache. RoPE
    # shift-equivariance means cache_abs_pos can grow unboundedly without
    # affecting attention scores — only relative positions (bounded by
    # the cache cap) matter.
    trainer = Phase1Trainer(
        model, optimizer, scheduler=scheduler, grad_clip=args.grad_clip,
        pad_token_id=tokenizer.pad_token_id,
        prior_loss_weight=0.0,  # W1 has no prior/response distinction
        load_balance_coef=args.load_balance_coef,
        z_loss_coef=args.z_loss_coef,
        revive_every=args.revive_every,
        revive_threshold=args.revive_threshold,
        tau_init=args.tau_init,
        tau_floor=args.tau_floor,
        tau_decay_rate=args.tau_decay_rate,
    )

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
    # Phase 1 uses rolling-buffer Llama (no KV cache), so multi-stream
    # batching just resets `prev_states` per-slot via torch.where. The
    # earlier per-slot KV cache machinery (and the lockstep wipe hack
    # that preceded it) is gone — rolling buffer carries no cross-window
    # state that needs slot-aware lifecycle.

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

    def run_answer_only_val() -> dict[str, float]:
        """Second val pass — only computes loss on ANSWER-span tokens
        (mask >= answer_span_weight threshold). The clean memory-probe
        diagnostic: directly measures whether memory retrieves the
        planted answer, without dilution by the 30K filler tokens.
        Only runs on sources with answer-span metadata (i.e., needle).

        Cost: ~30s per save (one extra val pass through needle val).
        """
        out: dict[str, float] = {}
        for label, vds in val_datasets.items():
            # Only needle has answer-span metadata; skip others.
            # We detect by checking if any val_mask value > 50 (the
            # answer_span_weight default is 100; filler is 1.0).
            losses_v: list[float] = []
            n_answer_chunks = 0
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
                # Build BINARY answer-only mask: 1.0 where answer span,
                # 0.0 elsewhere. Skip chunk entirely if no answer tokens.
                answer_only_mask = (item.valid_mask >= 50.0).float().unsqueeze(0).to(args.device)
                target_mask_v = answer_only_mask.view(1, cfg.D, cfg.T_window)
                ev = trainer.eval_wave1(
                    chunk_v,
                    prev_states=v_prev_states,
                    prev_window_hiddens=v_prev_hiddens,
                    prev_lm_context=v_prev_lm_ctx,
                    target_mask=target_mask_v,
                    past_key_values=v_past_kv,
                    cache_abs_pos=v_cache_abs,
                )
                if answer_only_mask.sum() > 0:
                    losses_v.append(ev["loss"])
                    n_answer_chunks += 1
                v_prev_states = ev["final_states"]
                v_prev_hiddens = ev["final_hiddens"]
                v_prev_lm_ctx = ev["final_lm_context"]
                v_past_kv = ev["final_past_key_values"]
                v_cache_abs = ev["final_cache_abs_pos"]
            if losses_v:
                out[label] = sum(losses_v) / len(losses_v)
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
    # Cross-chunk state.
    prev_states = None
    prev_window_hiddens = None
    prev_lm_context = None
    past_kv = None
    cache_abs_pos = 0
    last_step_t = time.time()

    while trainer.step_count < args.num_steps:
        for item in dataset:
            if trainer.step_count >= args.num_steps:
                break

            # Transformer-XL / RMT-style continuous training: we do NOT
            # reset any per-row state at doc boundaries. Each batch row
            # is treated as a continuous corpus stream — the trajectory-
            # memory's write_module gradually overwrites stale concepts
            # as new-doc tokens are processed (same role as TXL's mems).
            # KV cache trims naturally to effective_lm_context; cross-doc
            # attention contamination is "limited impact" per Llama 3's
            # finding (§3.4 of arXiv:2407.21783). RoPE position values can
            # grow unboundedly without breaking attention due to shift-
            # equivariance: only RELATIVE positions matter, and the cache
            # trim keeps the relative-position span bounded by cap=2048.
            # Eliminates the lockstep wipe entirely.
            if args.batch_size == 1:
                # ── Single-stream path ───────────────────────────────
                chunk = item.input_ids.unsqueeze(0).to(args.device)         # [1, T]
                valid_mask = item.valid_mask.unsqueeze(0).to(args.device)   # [1, T]
                target_mask = valid_mask.view(1, cfg.D, cfg.T_window)
                step_sources = [getattr(item, "source", "") or "unknown"]
            else:
                # ── Multi-stream path (BS > 1) ───────────────────────
                BS_chunk = item.input_ids.shape[0]
                if prev_states is None:
                    # First chunk of training only; subsequent chunks
                    # carry state via the loop's prev_states = metrics.final_states.
                    prev_states = model.manifold.reset_states(
                        batch_size=BS_chunk,
                    ).to(args.device)
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
            # Per-source surprise (writer's CE input). Surprise is computed
            # per-window inside forward_window; metrics.surprise_history is
            # [BS, D] when available. Average across windows then attribute
            # per-slot to the slot's source. Lets us see if the writer's
            # surprise drops faster on needle docs (memory active) vs
            # generic LM docs.
            history.setdefault("surprise_by_source", {})
            sh = metrics.surprise_history
            if sh is not None and sh.dim() == 2:
                # Per-slot mean surprise across the chunk's D windows.
                slot_means = sh.float().mean(dim=1).tolist()
                for slot_idx, src_label in enumerate(step_sources):
                    if slot_idx < len(slot_means):
                        history["surprise_by_source"].setdefault(
                            src_label, [],
                        ).append((trainer.step_count, slot_means[slot_idx]))

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
                # Routing entropy (sharpest collapse detector). Healthy
                # for N=4096 with 4 trajectories × K=8 reads per step =
                # 32 visits/step: entropy should be log(min(32, 4096))
                # ≈ 3.5 nats early; rises toward log(4096)≈8.3 as
                # routing diversifies across many concepts. If it drops
                # toward log(4-8)≈1-2, routing is collapsing.
                history.setdefault("read_routing_entropy", []).append(
                    routing_entropy(metrics.read_visited_ids, cfg.N),
                )
            if metrics.write_visited_ids is not None:
                ws = trajectory_diversity_stats(metrics.write_visited_ids, cfg.N)
                history.setdefault("write_unique_frac", []).append(ws["unique_frac"])
                history.setdefault("write_self_overlap", []).append(ws["self_overlap_rate"])
                history.setdefault("write_routing_entropy", []).append(
                    routing_entropy(metrics.write_visited_ids, cfg.N),
                )
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
                # Per-component param norms (param-explosion detector).
                comp_params = param_norms_by_component(model)
                for comp, val in comp_params.items():
                    history.setdefault(f"param_norm_{comp}", []).append(val)
                # Effective LR per component = lr · ||grad|| / ||param||.
                # Sanity range 1e-5 to 1e-3; >1e-2 unstable; <1e-7 stuck.
                eff_lrs = effective_lr_by_component(
                    comp_norms, comp_params, list(metrics.lr),
                )
                for comp, val in eff_lrs.items():
                    history.setdefault(f"eff_lr_{comp}", []).append(val)
            except Exception:
                pass
            # Concept-state drift (manifold runaway detector).
            try:
                if metrics.final_states is not None:
                    drift = concept_state_drift(
                        metrics.final_states, model.manifold.state_init,
                    )
                    history.setdefault("concept_drift_mean", []).append(drift["mean_drift"])
                    history.setdefault("concept_drift_max", []).append(drift["max_drift"])
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
                # Memory-health one-liner: routing diversity + write_module
                # grad norm. If read/write_unique_frac collapses toward 0
                # or write_module grad norm flatlines at 0, memory is
                # bottlenecked.
                ruf = history.get("read_unique_frac", [None])[-1]
                wuf = history.get("write_unique_frac", [None])[-1]
                w_gn = history.get("grad_norm_write_module", [None])[-1]
                rent = history.get("read_routing_entropy", [None])[-1]
                mem_str = (
                    f" r_uf={ruf:.3f}" if ruf is not None else ""
                ) + (
                    f" w_uf={wuf:.3f}" if wuf is not None else ""
                ) + (
                    f" r_ent={rent:.2f}" if rent is not None else ""
                ) + (
                    f" w_gn={w_gn:.2e}" if w_gn is not None else ""
                )
                print(f"  step {step:>5}  loss={metrics.loss:.4f}  avg10={avg:.4f}  "
                      f"grad_norm={metrics.grad_norm:.2f}  lr=[{lrs}]  "
                      f"({elapsed/max(step, 1):.2f}s/step){mem_str}")

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
                    # Answer-only val (the clean memory-probe diagnostic).
                    # Second val pass with binary answer-only mask.
                    answer_only = run_answer_only_val()
                    for src, v in answer_only.items():
                        history.setdefault("val_answer_loss", {}).setdefault(
                            src, [],
                        ).append(v)
                    if answer_only:
                        ans_pretty = "  ".join(
                            f"{s}={v:.3f}" for s, v in answer_only.items()
                        )
                        print(f"  step {step:>5}  val ANSWER-ONLY: {ans_pretty}")
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
