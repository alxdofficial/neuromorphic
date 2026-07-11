"""The live mixed round-robin trainer loop and the max-BS VRAM probe.

Extracted from ``scripts/train/train.py`` (harness reorg phase 2). ``train_mixed_variant`` routes
task_mode via ``mixes.task_mode(task)``. The retired composite-QA single-task loop
(``train_one_variant``) has been removed along with the composite dataset.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.memory.config import ReprConfig
from src.memory.data.common import collate_qa
from src.memory.model import ReprLearningModel
from src.memory.data import mixes

from .utils import lr_at_step, to_device
from .eval import run_mixed_val
from .checkpoint import save_checkpoint
from .objectives import _grad_cached_objective_step, _coding_rate, _behavioral_kl_step
from .data_mix import make_mixed_train_dataloaders, make_mixed_val_sets


def train_mixed_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, window_size: int,
    mixed_tasks: tuple, mixed_ctx: int, mixed_M: int,
    babi_tasks: tuple, predict_len: int, mae_src_tok: str,
    resume: bool = False, train_seed: int = 42,
) -> dict:
    """ONE model per architecture, trained on an EQUAL round-robin of mixed_tasks,
    evaluated PER-TASK. Homogeneous batches; tasks alternate step-to-step. Before
    each compute_loss the model's task_mode is set from the batch's task so MAE
    routes to the infill path and babi/continuation to the generic QA path.

    Per-task val rows are logged to the run jsonl, each tagged with `task` + `step`;
    per-task best_step/best_metric are tracked independently."""
    device = "cuda"
    # Per-arm cfg adjustments so ONE sweep command is correct for every arm (cfg is shared across the
    # variant loop). Titans' inner test-time-autograd (create_graph) conflicts with the outer streaming
    # activation-checkpoint → force stream-ckpt OFF for titans only (was a manual --no-grad-ckpt-stream).
    if variant == "titans_baseline" and getattr(cfg, "grad_checkpoint_stream", False):
        import copy as _copy
        # copy.copy (NOT dataclasses.replace): replace() rebuilds from DECLARED fields only, silently
        # dropping the ~20 dynamically-attached cfg attrs (kl_coef, titans_mem_hidden, per-arm ranks) —
        # the same drop-dynamic-attrs bug as audit #3. copy.copy preserves __dict__ (all attrs).
        cfg = _copy.copy(cfg)
        cfg.grad_checkpoint_stream = False
        print("[titans] stream activation-checkpoint auto-disabled (create_graph incompatibility)")
    llama_arg = None if cfg.use_llama_lora else llama
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps, "
          f"MIXED={'+'.join(mixed_tasks)})")
    print(f"{'='*78}")
    if n_trainable == 0:
        raise SystemExit(
            f"mixed training requires trainable params; {variant} has none "
            f"(vanilla floor/ceiling are single-task eval-only references).")

    opt = torch.optim.AdamW(
        model.trainable_parameters(), lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        fused=torch.cuda.is_available())

    # streaming-write retention placement for condrecon_bio (queried pair pinned to a window);
    # None = off (any window). window_size ties the placement to the encoder's chunking.
    _bio_qw = getattr(cfg, "cond_recon_bio_query_window", None)
    train_dls = make_mixed_train_dataloaders(
        mixed_tasks, tokenizer, cfg, ctx_len=mixed_ctx, m_slots=mixed_M,
        mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
        train_seed=train_seed, window_size=window_size, bio_query_window=_bio_qw)
    print(f"  Materializing fixed per-task val sets ({val_batches} batches each)...")
    val_sets = make_mixed_val_sets(
        mixed_tasks, tokenizer, cfg, val_batches, ctx_len=mixed_ctx, m_slots=mixed_M,
        mae_src_tok=mae_src_tok, babi_tasks=babi_tasks, predict_len=predict_len,
        window_size=window_size, bio_query_window=_bio_qw)
    print(f"  Val sets ready: {{ {', '.join(f'{t}:{len(val_sets[t])}' for t in mixed_tasks)} }}")

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    best_ckpt_path = out_dir / f"ckpts/{variant}.best.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    # Per-task best tracking — each task plateaus on its own schedule, so we keep
    # an independent best_step/best_metric per task (the primary metric is val_loss).
    best = {t: {"metric": float("inf"), "step": -1} for t in mixed_tasks}
    # Aggregate (mean val_loss across tasks) best — the SINGLE selection metric backed by a saved
    # .best.pt. One model serves all tasks, so a per-task best can't each own weights; the aggregate
    # is the principled selection criterion (per-task best_metric stays as informational telemetry).
    agg_best = {"metric": float("inf"), "step": -1}
    # early-stop config (from cfg; the CLI flags were previously parsed but never consumed). Stop when
    # `patience` consecutive val-evals show no real (> min_delta) agg_val improvement, past min_step.
    _es_patience = int(getattr(cfg, "patience", 0) or 0)
    _es_min_delta = float(getattr(cfg, "early_stop_min_delta", 0.01))
    _es_min_step = int(getattr(cfg, "min_step_for_stop", 3000))
    _no_improve = 0

    # ── resume: restore model + optimizer + step + best tracking from .last.pt ──
    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        _meta = sd.get("metadata", {}) or {}
        _ck_cfg = _meta.get("cfg_dict", {}) or {}
        _ck_all = _meta.get("cfg_all", {}) or {}          # includes dynamically-set fields (checkpoint.py)
        # Drift check: backbone/task_mode (weights-invalidating) PLUS the experiment-critical fields the
        # audit flagged — objective + KL coefficients + per-arm adapter ranks/sizes. Resuming across any of
        # these silently changes what is being trained (or rebuilds the arm with wrong capacity). cfg_all
        # is checked first so the dynamically-attached ranks (not in cfg_dict) are caught.
        _crit = ("backbone_model", "objective_mode", "kl_coef", "kl_ce_coef", "kl_temp",
                 "memoryllm_lora_rank", "titans_mem_hidden", "gisting_lora_rank", "icae_lora_rank",
                 "slotgraph_lora_rank", "slotgraph_d_edge", "slotgraph_write_layers")
        for _field in ("backbone_model", "task_mode", *_crit):
            _now = (model.cfg.llama_model if _field == "backbone_model"
                    else getattr(model.cfg, _field, None))
            _was = _meta.get(_field, _ck_all.get(_field, _ck_cfg.get(_field)))
            if _was is not None and _now is not None and _was != _now:
                raise RuntimeError(
                    f"[resume] {_field} drift: checkpoint has {_was!r} but this run is {_now!r}. "
                    f"Refusing to resume across a changed objective/backbone/capacity — use a fresh --out-tag.")
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower() and not k.startswith("encoder.base.")
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(f"[resume] checkpoint/arch mismatch — would leave params random: {_bad[:12]}")
        opt.load_state_dict(sd["optimizer_state_dict"])
        start_step = int(sd.get("step", 0)) + 1
        if "mixed_best" in sd:
            best = {t: dict(sd["mixed_best"].get(t, {"metric": float("inf"), "step": -1}))
                    for t in mixed_tasks}
        if "mixed_agg_best" in sd:
            agg_best = dict(sd["mixed_agg_best"])
        print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
              f"(agg_best={agg_best['metric']:.4f} @ {agg_best['step']})")
    elif jsonl_path.exists() and not resume:
        jsonl_path.unlink()
    jsonl_fp = open(jsonl_path, "a", buffering=1)

    t_start = time.time()
    last_print_step, last_print_time = start_step, t_start
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # One persistent iterator per task; round-robin pulls the next task in rotation.
    iters = {t: iter(dl) for t, dl in train_dls.items()}

    def next_batch(t):
        try:
            return next(iters[t])
        except StopIteration:                # streaming loaders are ~infinite; re-arm defensively
            iters[t] = iter(train_dls[t])
            return next(iters[t])

    _mode_banner = str(getattr(cfg, "objective_mode", "plain"))
    if _mode_banner != "plain" or float(getattr(cfg, "contrastive_shuf_coef", 0.0)) > 0:
        # InfoNCE is ONLY the contrastive GradCache path — behavioral_kl applies CE+KL
        # only (no InfoNCE), so advertise the KL terms for it instead (the old banner mislabeled it).
        _is_nce = _mode_banner == "contrastive"
        print(f"[objective] mode={_mode_banner}"
              + (f"  InfoNCE coef={cfg.objective_coef} (in-batch, all B-1 negatives)" if _is_nce else "")
              + (f"  KL(teacher‖student) coef={cfg.kl_coef} + CE coef={cfg.kl_ce_coef}, temp={cfg.kl_temp}"
                 if _mode_banner == "behavioral_kl" else "")
              + (f"  legacy softplus coef={cfg.contrastive_shuf_coef}"
                 if float(getattr(cfg, 'contrastive_shuf_coef', 0.0)) > 0 else ""),
              flush=True)
    rotation = []   # diagnostic: the realized task sequence (capped for the smoke print)
    for step in range(start_step, n_steps):
        task = mixed_tasks[step % len(mixed_tasks)]   # equal round-robin
        if len(rotation) < 24:
            rotation.append(task)
        model.task_mode = mixes.task_mode(task)       # per-batch dispatch (E/D)
        batch = next_batch(task)

        lr = lr_at_step(step, n_steps, cfg.learning_rate, cfg.warmup_steps)
        for g in opt.param_groups:
            g["lr"] = lr

        batch = to_device(batch, device)
        opt.zero_grad(set_to_none=True)
        _mode = str(getattr(cfg, "objective_mode", "plain"))
        _legacy_coef = float(getattr(cfg, "contrastive_shuf_coef", 0.0))
        _obj_extras = {}
        if _mode == "behavioral_kl":
            # context distillation: teacher (full-context) ‖ student (memory) KL on answer spans.
            # backward runs INSIDE (two frozen-LM forwards). obj_kl logged every step.
            out, loss, _obj_extras = _behavioral_kl_step(model, batch, cfg, window_size)
            _needs_backward = False
        elif _mode == "contrastive":
            # in-batch InfoNCE: backward runs INSIDE (GradCache memory cut) — no
            # loss.backward() below. Loud by construction: obj_nce is logged every step.
            out, loss, _obj_extras = _grad_cached_objective_step(model, batch, cfg, window_size)
            _needs_backward = False
        elif _legacy_coef > 0:
            # legacy 1-negative softplus hinge — NOW SUPPORTED HERE (2026-07-02: this trainer
            # used to silently IGNORE the coef; the "sg3_contrast turning point" never ran).
            # REAL and SHUF share the mask via RNG restore (mask-difficulty noise cancels).
            _rng = torch.cuda.get_rng_state()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.compute_loss(batch, window_size=window_size)
            torch.cuda.set_rng_state(_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_shuf = model.compute_loss(batch, window_size=window_size, shuffle_memory=True)
            contrast = torch.nn.functional.softplus(out["loss"] - out_shuf["loss"])
            loss = out["loss"] + _legacy_coef * contrast
            _obj_extras = {"obj_softplus": float(contrast)}
            _needs_backward = True
        else:
            _rank_c = float(getattr(cfg, "rank_reward_coef", 0.0))
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model.compute_loss(batch, window_size=window_size, return_memory=(_rank_c > 0))
            loss = out["loss"]
            if _rank_c > 0 and out.get("_memory") is not None:
                # MCR² coding-rate REWARD on the WITHIN-example memory (2026-07-03 diagnosis fix +
                # a-vs-d discriminator): maximize R_i = ½·logdet(I + (d/(M·ε²))·ZᵀZ) per example on
                # unit-normed tokens → charge the objective for rank so the emitted memory can't
                # collapse to a rank-2 blur. In fp32 (logdet is ill-conditioned under bf16).
                with torch.amp.autocast("cuda", enabled=False):
                    _R = _coding_rate(out["_memory"].float(), float(getattr(cfg, "rank_reward_eps", 0.5)))
                loss = loss - _rank_c * _R
                _obj_extras = {"obj_rank_R": float(_R.detach())}
            _needs_backward = True
        if not torch.isfinite(loss):
            # Skip a rare non-finite batch instead of dying (grads are still zeroed from above, so the
            # model is NOT poisoned). A SYSTEMATIC divergence would skip every step → visible in the log.
            print(f"  [step {step}] non-finite loss = {float(loss.detach())} (task={task}) — skipping batch")
            opt.zero_grad(set_to_none=True)
            continue
        # Debug: from --anomaly-from onward, run backward under anomaly detection so the FIRST non-finite
        # gradient halts with a traceback to the exact forward op that produced it (catches backward NaNs
        # that the forward-loss skip-guard above can't see).
        import contextlib as _ctxlib
        _afrom = int(getattr(cfg, "anomaly_from", -1))
        _anom = (torch.autograd.detect_anomaly()
                 if (_afrom >= 0 and step >= _afrom) else _ctxlib.nullcontext())
        with _anom:
            if _needs_backward:
                loss.backward()
        _mem_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("decoder.llama.")]
        _lora_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("decoder.llama.")]
        gn_mem = torch.nn.utils.clip_grad_norm_(_mem_params, cfg.grad_clip)
        gn_lora = torch.nn.utils.clip_grad_norm_(_lora_params, cfg.grad_clip)
        gn = (gn_mem ** 2 + gn_lora ** 2) ** 0.5
        # Skip the optimizer step on a non-finite GRADIENT (the forward-loss check above misses backward
        # NaNs). This is the standard bf16/AMP handling — PyTorch's GradScaler skips inf/nan-grad steps —
        # and is the right fix for a STOCHASTIC mixed-precision NaN (a rare borderline overflow that fires
        # on different steps each run and vanishes under anomaly mode), vs poisoning the model.
        if not (torch.isfinite(gn_mem) and torch.isfinite(gn_lora)):
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            short = [n.replace("encoder.", "enc.").replace("decoder.llama", "dec") for n in bad
                     if "base.model.layers" not in n][:14]   # drop the 60 frozen-base LoRA names
            print(f"  [step {step}] non-finite GRADIENT (task={task}) — culprit params: {short} "
                  f"(+{len(bad)-len(short)} more) — skipping opt.step")
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()

        # Per-step TRAIN row, tagged with the task that produced this batch.
        train_row = {
            "step": step, "variant": variant, "task": task, "phase": "train",
            "loss": float(out["loss"].detach()),
            "loss_recon": float(out["loss_recon"]),
            "top1_acc": float(out["top1_acc"]),
            "grad_norm": float(gn),
            # split norms (review fix): a combined norm hides memory-side gradient starvation —
            # the #1 historical failure mode — in the trainer the objective modes run in
            "grad_norm_memory": float(gn_mem),
            "grad_norm_lora": float(gn_lora),
            "memory_M": out["memory_shape"][1],
            "lr": lr,
        }
        # arm collapse/health canaries at train frequency (biomem edge/decay/beta/sat/mem_effrank/…, etc.)
        for _k, _v in out.items():
            if _v is None or not _k.startswith(("graph_", "biomem_", "slotgraph_", "vqicae_")):
                continue    # "graph_" added for parity with eval.py/val-row forwarding (graph_baseline canaries)
            if isinstance(_v, (int, float)):
                train_row[_k] = float(_v)
            elif torch.is_tensor(_v) and _v.numel() == 1:
                train_row[_k] = float(_v.detach())
        # objective-mode telemetry (obj_nce/obj_ce_real/obj_grpo_*/obj_softplus) — logged EVERY step
        # so a silently-inert objective is impossible to miss (the sg3_contrast lesson).
        for _k, _v in _obj_extras.items():
            train_row[_k] = float(_v)
        jsonl_fp.write(json.dumps(train_row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            print(f"  step {step:6d}/{n_steps}  [{task:12}]  "
                  f"recon={float(out['loss_recon']):.4f}  "
                  f"top1={float(out['top1_acc'])*100:5.1f}%  "
                  f"M={out['memory_shape'][1]}  gnorm={float(gn):6.2f}  "
                  f"lr={lr:.2e}  ({sps:.1f} step/s)", flush=True)

        if step > 0 and step % val_every == 0:
            per_task = run_mixed_val(model, mixed_tasks, val_sets, device,
                                     val_batches, window_size,
                                     gate_batches=int(getattr(cfg, "mixed_gate_batches", 0)))
            # one PER-TASK val row each, tagged with task + step (E/E).
            parts = []
            for t in mixed_tasks:
                vm = per_task[t]
                row = {"phase": "val", "step": step, "variant": variant, "task": t,
                       "val_loss": vm["val_loss_recon"], "top1": vm["val_top1_acc"]}
                if "val_babi_em" in vm:
                    row["val_babi_em"] = vm["val_babi_em"]
                if "val_cont_early_loss" in vm:
                    row["val_cont_early_loss"] = vm["val_cont_early_loss"]
                # REAL/SHUF/OFF binding gate (example-specificity diagnostic; present when gate_batches>0)
                for _gk in ("val_shuf_minus_real", "val_off_minus_real",
                            "val_loss_recon_shuf", "val_loss_recon_off"):
                    if _gk in vm:
                        row[_gk] = vm[_gk]
                for _k, _v in vm.items():               # graph read+write collapse canaries
                    if _k.startswith(("val_graph_", "val_biomem_", "val_slotgraph", "val_vqicae_")):
                        row[_k] = _v                       # "val_slotgraph" (no trailing _) catches slotgraph/2/3
                # per-task best on val_loss
                if vm["val_loss_recon"] < best[t]["metric"]:
                    best[t]["metric"] = vm["val_loss_recon"]
                    best[t]["step"] = step
                row["best_step"] = best[t]["step"]
                row["best_metric"] = best[t]["metric"]
                jsonl_fp.write(json.dumps(row) + "\n")
                tag = f"{t}={vm['val_loss_recon']:.3f}"
                if "val_babi_em" in vm:
                    tag += f"(EM={vm['val_babi_em']*100:.0f}%)"
                if "val_cont_early_loss" in vm:
                    tag += f"(early={vm['val_cont_early_loss']:.3f})"
                if "val_graph_edge_cos" in vm:          # write/read collapse at a glance
                    tag += (f"[ec={vm['val_graph_edge_cos']:.2f}"
                            f" nu={int(vm.get('val_graph_nodes_used', 0))}"
                            f" isns={vm.get('val_graph_input_sens', 0):.1f}"
                            f" mr={vm.get('val_graph_mem_effrank', 0):.1f}]")
                if "val_vqicae_perplexity" in vm:       # vqicae codebook-usage canary at a glance
                    tag += (f"[ppl={vm['val_vqicae_perplexity']:.0f}"
                            f" act={int(vm.get('val_vqicae_active_codes', 0))}]")
                if "val_slotgraph_edge_frac" in vm:     # slotgraph structure canary at a glance
                    tag += (f"[ef={vm['val_slotgraph_edge_frac']:.2f}"
                            f" se={vm.get('val_slotgraph_src_entropy', 0):.1f}"
                            f" mr={vm.get('val_slotgraph_mem_effrank', 0):.1f}]")
                parts.append(tag)
            print(f"    [val @ {step}]  " + "  ".join(parts), flush=True)
            # Aggregate selection: mean val_loss across tasks → save the single .best.pt.
            agg = sum(per_task[t]["val_loss_recon"] for t in mixed_tasks) / len(mixed_tasks)
            if agg < agg_best["metric"] - _es_min_delta:      # a REAL improvement (past the noise floor)
                agg_best["metric"], agg_best["step"] = agg, step
                _no_improve = 0
                save_checkpoint(model, opt, step, best_ckpt_path,
                                mixed_best=best, mixed_agg_best=agg_best)
                print(f"    [best @ {step}]  agg_val={agg:.4f} → saved {best_ckpt_path.name}", flush=True)
            else:
                # still save a new best if it's the lowest (for checkpoint fidelity), but don't reset patience
                if agg < agg_best["metric"]:
                    agg_best["metric"], agg_best["step"] = agg, step
                    save_checkpoint(model, opt, step, best_ckpt_path,
                                    mixed_best=best, mixed_agg_best=agg_best)
                _no_improve += 1
            # EARLY STOP: patience val-evals with no real improvement, past the warmup-noise floor.
            if (_es_patience > 0 and step >= _es_min_step and _no_improve >= _es_patience):
                print(f"    [early-stop @ {step}]  no agg_val improvement (>{_es_min_delta}) for "
                      f"{_no_improve} val-evals since step {agg_best['step']}; stopping.", flush=True)
                break

        if step > 0 and step % save_every == 0:
            save_checkpoint(model, opt, step, ckpt_path,
                            mixed_best=best, mixed_agg_best=agg_best)

    save_checkpoint(model, opt, max(n_steps - 1, 0), ckpt_path,
                    mixed_best=best, mixed_agg_best=agg_best)
    final = run_mixed_val(model, mixed_tasks, val_sets, device, val_batches, window_size,
                          gate_batches=int(getattr(cfg, "mixed_gate_batches", 0)))
    for t in mixed_tasks:
        vm = final[t]
        row = {"phase": "val", "step": n_steps, "variant": variant, "task": t,
               "final": True, "val_loss": vm["val_loss_recon"], "top1": vm["val_top1_acc"],
               "best_step": best[t]["step"], "best_metric": best[t]["metric"]}
        if "val_babi_em" in vm:
            row["val_babi_em"] = vm["val_babi_em"]
        if "val_cont_early_loss" in vm:
            row["val_cont_early_loss"] = vm["val_cont_early_loss"]
        for _k, _v in vm.items():                       # graph/biomem/slotgraph/arrival collapse canaries
            if _k.startswith(("val_graph_", "val_biomem_", "val_slotgraph", "val_vqicae_")):
                row[_k] = _v                             # "val_slotgraph" (no trailing _) catches slotgraph/2/3
        jsonl_fp.write(json.dumps(row) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
    print(f"  DONE (mixed): {n_steps} steps in {elapsed/60:.1f} min  "
          f"peak_vram={peak_vram_gb:.1f}GB", flush=True)
    print(f"  rotation[:{len(rotation)}] = {rotation}", flush=True)
    summary = {
        "variant": variant, "trainable_params": n_trainable, "n_steps": n_steps,
        "elapsed_s": elapsed, "peak_vram_gb": peak_vram_gb, "mixed": True,
        "mixed_tasks": list(mixed_tasks),
        "per_task_final": {t: {"val_loss": final[t]["val_loss_recon"],
                               "top1": final[t]["val_top1_acc"],
                               "best_step": best[t]["step"],
                               "best_metric": best[t]["metric"]} for t in mixed_tasks},
        "agg_best_metric": agg_best["metric"], "agg_best_step": agg_best["step"],
    }
    del model, opt
    torch.cuda.empty_cache()
    return summary


def probe_bs(variants, llama, tokenizer, cfg, args):
    """Per-arm max-batch-size VRAM probe on the conditioned-reconstruction path (no training, no checkpoints).
    For each arm: push BS up the --probe-bs-list until OOM; report the largest fitting BS,
    its peak VRAM, and throughput. Builds each arm with the SAME production cfg as a real
    run, so the reported max-BS is directly usable as that arm's --batch-size."""
    import time
    from dataclasses import replace as _dc_replace
    from src.memory.data.sources import SOURCE_REGISTRY
    from src.memory.data.tasks import get_task
    from src.memory.data.tasks.base import TaskDataset
    from src.memory.data.schedule import EpisodeSpec
    device = "cuda"
    _pad = cfg.pad_token_id if cfg.pad_token_id is not None else 128_001
    # Rebuild the conditioned-reconstruction sample pool via the 4-layer path (Source × Task × spec).
    source = SOURCE_REGISTRY["bio"](tokenizer, split="train", world_seed=0,
                                    n_facts=args.cond_recon_bio_n_facts)
    task = get_task("reconstruction")
    spec = EpisodeSpec(source="bio", task="reconstruction", total_len=args.chunk_size,
                       n_inputs=args.cond_recon_n_pairs, n_queries=args.cond_recon_n_query)
    ds = TaskDataset(source, task, spec, tokenizer, pad_token_id=_pad)
    pool, it = [], iter(ds)
    for _ in range(max(args.probe_bs_list)):
        pool.append(next(it))

    _TF = ("context_ids", "context_mask", "question_ids", "question_mask",
           "answer_ids", "answer_mask", "answer_content_mask")

    def _batch(bs):
        qb = collate_qa(pool[:bs])
        return _dc_replace(qb, **{f: getattr(qb, f).to(device) for f in _TF})

    rows = []
    for variant in variants:
        llama_arg = None if cfg.use_llama_lora else llama
        try:
            model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
        except torch.cuda.OutOfMemoryError:
            print(f"  [{variant:24}] OOM at model build", flush=True)
            torch.cuda.empty_cache(); rows.append((variant, 0, 0.0, 0.0)); continue
        if model.n_trainable_params() == 0:
            print(f"  [{variant:24}] eval-only — skipped", flush=True)
            del model; torch.cuda.empty_cache(); continue
        opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
        model.train()
        best = (0, 0.0, 0.0)
        for bs in args.probe_bs_list:
            try:
                batch = _batch(bs)
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):                                   # warmup (compile/alloc)
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model.compute_loss(batch, window_size=args.window_size)
                    out["loss"].backward(); opt.step()
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(5):
                    opt.zero_grad(set_to_none=True)
                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        out = model.compute_loss(batch, window_size=args.window_size)
                    out["loss"].backward(); opt.step()
                torch.cuda.synchronize()
                dt = (time.time() - t0) / 5
                peak = torch.cuda.max_memory_allocated() / 2**30
                best = (bs, peak, bs / dt)
                print(f"  [{variant:24}] BS={bs:4d}  peak={peak:5.1f}GB  {bs/dt:6.1f} samp/s  "
                      f"{dt*1e3:6.0f}ms/step", flush=True)
            except torch.cuda.OutOfMemoryError:
                print(f"  [{variant:24}] BS={bs:4d}  OOM", flush=True)
                torch.cuda.empty_cache(); break
        rows.append((variant, *best))
        del model, opt; torch.cuda.empty_cache()

    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"\n==== conditioned-reconstruction max-BS per arm (N={args.cond_recon_n_pairs}, {total:.0f}GB GPU) ====", flush=True)
    print(f"  {'arm':<24}{'maxBS':>7}{'peakGB':>8}{'headGB':>8}{'samp/s':>9}")
    for v, bs, pk, sps in rows:
        print(f"  {v:<24}{bs:>7}{pk:>8.1f}{total-pk:>8.1f}{sps:>9.1f}")
