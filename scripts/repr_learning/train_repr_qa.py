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
def materialize_val_set(val_dl, n_batches: int) -> list:
    """Drain the streaming val_dl ONCE into a fixed list of batches.

    Fixes the val sampling variance bug (#614). Previously each `run_val` call
    advanced the val DataLoader iterator → different random batches per eval →
    0.4-0.7 nat noise between streaming-best and final-eval on the SAME model.
    Materializing the set once means every eval is on the same data → numbers
    become directly comparable and the streaming-best vs final-eval gap closes
    to ~zero (modulo checkpoint differences).

    Memory cost: ~2 MB for 30 batches × 2 examples × 4096 tokens × int64. Trivial.
    """
    fixed = []
    for i, batch in enumerate(val_dl):
        if i >= n_batches:
            break
        fixed.append(batch)
    return fixed


def run_val(model, val_set, device, n_batches: int, window_size: int) -> dict:
    """Eval on a fixed val_set (list of batches from materialize_val_set).

    n_batches caps the iteration in case val_set has more batches than we want
    to spend time on (e.g., subsampling for a fast in-training eval). To use the
    full set, pass n_batches >= len(val_set).
    """
    model.train(False)
    losses, accs, per_fam_stats = [], [], {}
    last_mp_cos: float | None = None  # eval-only telemetry (gated in MP readout)
    # Deterministic-eval seed (audit fix 2026-05-27): graph_v5's chunk-fresh
    # init was sampling fresh noise per call → same model + same batch produced
    # ~0.2 loss variance. Seeding torch RNG per batch makes eval reproducible.
    # In eval mode (model.train(False)) dropout is off so this is safe.
    rng_state = torch.get_rng_state()
    cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    for i, batch in enumerate(val_set):
        if i >= n_batches:
            break
        torch.manual_seed(20260527 + i)  # batch-dependent deterministic seed
        batch = to_device(batch, device)
        # no_grad: huge memory win (~18→4 GiB at B=12). Val never backprops,
        # so the autograd graph was being built for nothing under autocast.
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_qa_loss(batch, window_size=window_size)
        losses.append(float(out["loss_recon"]))
        accs.append(float(out["top1_acc"]))
        # Capture v5.4 oversmoothing canary from the last batch's last MP round.
        # Only populated in eval (MP readout gates the K×K matmul off in train).
        arr = out.get("graph_v5_mp_buf_cross_node_cos_per_round")
        if arr is not None and len(arr) > 0:
            last_mp_cos = float(arr[-1])
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
    # Restore training RNG state so eval doesn't perturb the training stream.
    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)
    n = max(len(losses), 1)
    fam_summary = {f: {"n": v["n"], "mean_loss": v["loss"] / max(v["n"], 1)}
                   for f, v in per_fam_stats.items()}
    result = {
        "val_loss_recon": sum(losses) / n,
        "val_top1_acc": sum(accs) / n,
        "val_n_batches": len(losses),
        "val_per_family": fam_summary,
    }
    if last_mp_cos is not None:
        result["val_graph_v5_mp_buf_cross_node_cos_final"] = last_mp_cos
    return result


def save_checkpoint(model, opt, step, path: Path, **extras):
    """Persist model + opt state. `extras` lets callers stash auxiliary
    tracking fields (best_val_recon / best_val_step) so resume can pick
    them up; without that, resume starts best_val_recon=inf and the very
    first val of the resumed run overwrites the genuine prior best.pt.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    def keep(k: str) -> bool:
        if not k.startswith("decoder.llama."):
            return True
        return "lora_" in k

    payload = {
        "step": step,
        "model_state_dict": {
            k: v for k, v in model.state_dict().items() if keep(k)
        },
        "optimizer_state_dict": opt.state_dict(),
    }
    payload.update(extras)
    torch.save(payload, path)


def train_one_variant(
    variant: str, llama, tokenizer, cfg: ReprConfig,
    n_steps: int, log_every: int, val_every: int, save_every: int,
    val_batches: int, out_dir: Path, chunk_size: int, window_size: int,
    passages_per_chunk: int, resume: bool = False,
    use_hotpot: bool = True, use_narrative: bool = True,
    use_musique: bool = False, use_babilong: bool = False,
    babilong_config: str = "4k",
    mix_weights: tuple = (0.5, 0.25, 0.25, 0.0, 0.0),
    composite_task_weights: Optional[dict[str, float]] = None,
    patience: int = 5,                       # eval-points without improvement → stop
    min_step_for_stop: int = 2000,           # don't stop during warmup-noise era
) -> dict:
    device = "cuda"
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama).to(device)
    n_trainable = model.n_trainable_params()
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    is_vanilla = variant in ("vanilla_llama", "vanilla_full_context")

    # vanilla_* variants have no trainable encoder params:
    #   - vanilla_llama: no memory at all (loss floor reference)
    #   - vanilla_full_context: passes raw context embeddings as memory (loss ceiling)
    # Both just establish reference points for the variant comparison;
    # training them is wasted compute. Treat as eval-only.
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
        use_musique=use_musique, use_babilong=use_babilong,
        babilong_config=babilong_config,
        split="train",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, composite_task_weights=composite_task_weights,
        num_workers=2, seed=42,
    )
    val_dl = make_mixed_qa_dataloader(
        cfg, tokenizer,
        composite_passages_path=COMPOSITE_VAL_P,
        composite_questions_path=COMPOSITE_VAL_Q,
        use_hotpot=use_hotpot, use_narrative=use_narrative,
        use_musique=use_musique, use_babilong=use_babilong,
        babilong_config=babilong_config,
        split="validation",
        chunk_size=chunk_size, passages_per_chunk=passages_per_chunk,
        weights=mix_weights, composite_task_weights=composite_task_weights,
        num_workers=2, seed=7,
    )
    # Fixes #614: drain val_dl ONCE into a fixed list so every run_val call
    # sees the same batches. Without this, in-training val and final-eval got
    # different random draws → 0.4-0.7 nat noise → streaming-best was biased
    # low (cherry-picked from many noisy draws) and final-eval was a one-shot
    # gamble. With a fixed set: streaming val and final eval directly comparable.
    print(f"  Materializing fixed val set ({val_batches} batches)...")
    val_set = materialize_val_set(val_dl, val_batches)
    print(f"  Fixed val set ready: {len(val_set)} batches.")

    jsonl_path = out_dir / f"jsonl/{variant}.jsonl"
    ckpt_path = out_dir / f"ckpts/{variant}.last.pt"
    best_ckpt_path = out_dir / f"ckpts/{variant}.best.pt"
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    # Best-checkpoint tracking. Only consider vals from step >= BEST_MIN_STEP
    # (10% of total steps, min 1000) to filter early-warmup flukes — LR is
    # still ramping and val noise dominates real signal at low steps.
    best_val_recon = float("inf")
    best_val_step = -1
    BEST_MIN_STEP = max(1000, n_steps // 10)
    # Patience-based early stopping (best.pt staleness criterion). Counts
    # eval points since the last best.pt update. Stop only when no new
    # global best has been seen for `patience` consecutive evals past
    # min_step_for_stop. Previous "smoothed-mean" criterion triggered on
    # volatility and could fire on the same step that just produced a new
    # best.pt — invalidated several runs. Best-staleness avoids that
    # pathology entirely: the moment a true improvement lands, the counter
    # resets, so we never stop within `patience * val_every` steps of an
    # actual improvement.
    early_stopped = False
    stopped_at_step = n_steps
    evals_since_best = 0

    if is_vanilla:
        # Eval-only path. Skip any prior jsonl and run a single final-val pass.
        if jsonl_path.exists():
            jsonl_path.unlink()
        t_start = time.time()
        final_val = run_val(model, val_set, device, val_batches, window_size)
        elapsed = time.time() - t_start
        with open(jsonl_path, "w") as fp:
            fp.write(json.dumps({
                "phase": "val", "step": 0, "variant": variant,
                "final": True, "eval_only": True, **final_val,
            }) + "\n")
        print(f"  [eval-only] {variant} val_recon={final_val['val_loss_recon']:.4f}  "
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
        # Restore best-tracking so we don't overwrite the prior best.pt with
        # a worse first-val from the resumed run. Older ckpts didn't store
        # these — fall back to inf and warn so the user knows the prior best
        # is at risk of being clobbered on the next improvement check.
        if "best_val_recon" in sd and "best_val_step" in sd:
            best_val_recon = float(sd["best_val_recon"])
            best_val_step = int(sd["best_val_step"])
            evals_since_best = int(sd.get("evals_since_best", 0))
            print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
                  f"(prior best={best_val_recon:.4f} @ step {best_val_step}, "
                  f"evals_since_best={evals_since_best})")
        else:
            print(f"  [resume] loaded {ckpt_path.name} @ step {start_step - 1} "
                  f"(WARN: no best_val_recon in ckpt — first val will define "
                  f"a new best.pt; prior best.pt may be overwritten)")
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

        opt.zero_grad(set_to_none=True)
        # bf16 autocast for ~40% speed + ~50% activation memory reduction.
        # Encoders have `enabled=False` blocks around numerically-sensitive ops
        # (entropy, normalization) so those remain fp32 even under the outer
        # autocast — the codebase was designed for this pattern.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            "loss_aux": float(out["loss_aux"]),               # load_balance only (unweighted)
            "top1_acc": float(out["top1_acc"]),
            "n_content_positions": int(out["n_content_positions"]),
            "grad_norm": float(gn),
            "lr": lr,
            "memory_M": out["memory_shape"][1],
        }
        # Per-component aux breakdown (unweighted). Lets us see WHICH aux
        # term is exploding when total aux spikes — previously we only had
        # the sum, and a load_balance=1392 spike was indistinguishable from
        # an orth_loss=1392 spike. Both have very different root causes.
        for key in ("loss_orth", "loss_z"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Splat-variant sublosses (only present when variant == splat_baseline)
        for key in ("splat_aux", "splat_L_pin", "splat_L_prop",
                    "splat_L_adj", "splat_L_sat"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Graph-variant telemetry (only present when variant == graph_baseline).
        # v4 has no aux loss; logged keys are gate-distribution + endpoint
        # clustering diagnostics.
        # Whitelisted scalars + a glob for per-window breakdown keys.
        _graph_scalar_keys = (
            "graph_aux", "graph_endpoint_reuse",
            "graph_u_mean", "graph_age_mean", "graph_src_norm",
            "graph_pick_affinity_avg", "graph_gate_mean_avg",
            "graph_frac_anchor_avg", "graph_frac_loadbearer_avg",
            "graph_frac_jumpedship_avg", "graph_frac_selfpick_avg",
            # Specialization + endpoint clustering (audit-2 additions)
            "graph_g_slot_std", "graph_g_slot_range",
            "graph_endpoint_eff_rank", "graph_endpoint_cos_max",
        )
        for key in _graph_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # Graph v5.1 telemetry (only present when variant == graph_v5_baseline).
        # Probes the soft-pointer mechanism: are queries sharp, do edges
        # converge on shared nodes, are roles mixing.
        _graph_v5_scalar_keys = (
            # Per-window write-side
            "graph_v5_node_gate_mean", "graph_v5_edge_gate_mean",
            "graph_v5_edge_pick_affinity", "graph_v5_edge_frac_selfpick",
            "graph_v5_edge_pick_entropy",
            # Chunk-end (averaged) write-side
            "graph_v5_node_gate_mean_avg", "graph_v5_edge_gate_mean_avg",
            "graph_v5_edge_pick_affinity_avg", "graph_v5_edge_frac_selfpick_avg",
            "graph_v5_edge_pick_entropy_avg",
            # Chunk-end read-side: soft pointer sharpness + reuse
            "graph_v5_edge_src_entropy", "graph_v5_edge_dst_entropy",
            "graph_v5_unique_picks_frac", "graph_v5_cross_role_overlap",
            "graph_v5_endpoint_cos_mean", "graph_v5_endpoint_cos_max",
            # v5.3+: learnable soft-pointer temperature (scalar)
            "graph_v5_soft_pointer_temperature",
        )
        for key in _graph_v5_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # v5.4: per-round MP telemetry — log final-round value as a scalar so
        # it shows up in jsonl plotting. Full arrays kept in aux for probes.
        for arr_key, scalar_key in [
            ("graph_v5_mp_buf_norm_per_round",            "graph_v5_mp_buf_norm_final"),
            ("graph_v5_mp_agg_norm_per_round",            "graph_v5_mp_agg_norm_final"),
            ("graph_v5_mp_buf_cross_node_cos_per_round",  "graph_v5_mp_buf_cross_node_cos_final"),
        ]:
            arr = out.get(arr_key)
            if arr is not None and len(arr) > 0:
                row[scalar_key] = float(arr[-1])
        # Per-window breakdown: graph_g_mean_w0..w3, graph_frac_*_w0..w3
        for k, v in out.items():
            if v is None: continue
            if (k.startswith("graph_g_mean_w") or
                (k.startswith("graph_frac_") and k.endswith(("_w0", "_w1", "_w2", "_w3")))):
                row[k] = float(v)
        jsonl_fp.write(json.dumps(row) + "\n")

        if step % log_every == 0:
            now = time.time()
            sps = (step - last_print_step) / max(now - last_print_time, 1e-9)
            last_print_step, last_print_time = step, now
            # Display the variant-specific signal that's actually informative:
            # - splat_aux: weighted aux contributing to loss
            # - graph: u/pick/overwrite signals (no aux loss in v3 — graph_aux
            #   is always 0, so displaying it tells you nothing)
            # - else: plain loss_aux (load_balance+orth+z)
            extra_field = ""
            if variant == "graph_baseline":
                # v4: show gate distribution (anchor/loadbearer/jumpedship)
                # + self-pick rate. Tells you immediately whether g is doing
                # anything (gate≈0.05 + no escalation = stubborn-stuck).
                g = float(out.get("graph_gate_mean_avg", 0.0) or 0.0)
                fa = float(out.get("graph_frac_anchor_avg", 0.0) or 0.0)
                fl = float(out.get("graph_frac_loadbearer_avg", 0.0) or 0.0)
                fj = float(out.get("graph_frac_jumpedship_avg", 0.0) or 0.0)
                fs = float(out.get("graph_frac_selfpick_avg", 0.0) or 0.0)
                extra_field = f"g={g:.3f} a/l/j={fa:.2f}/{fl:.2f}/{fj:.2f} self={fs:.2f}"
                aux_tag, aux_display = "aux", float(out["loss_aux"])
            elif variant == "graph_v5_baseline":
                # v5.1: show node/edge gates + soft-pointer sharpness + reuse.
                # Key signals:
                #   g_n / g_e: node and edge gate means (anchor-init ≈ 0.38 / 0.27)
                #   selfpick: fraction of edges picking their own slot's proposal
                #   src_ent: soft-pointer entropy (low = sharp, high = spread)
                #   uniq: fraction of unique bank entries among edge picks
                #         (low = high reuse, which is the THESIS goal)
                #   xrole: cross-role overlap (slots appearing as both src+dst)
                gn = float(out.get("graph_v5_node_gate_mean_avg", 0.0) or 0.0)
                ge = float(out.get("graph_v5_edge_gate_mean_avg", 0.0) or 0.0)
                se = float(out.get("graph_v5_edge_src_entropy", 0.0) or 0.0)
                uq = float(out.get("graph_v5_unique_picks_frac", 0.0) or 0.0)
                # v5.3+: learnable soft-pointer τ (should drift toward 0 if model
                # finds sharper routing helpful). v5.4+ mp_cos (oversmoothing
                # canary) only computed during eval — see val print line.
                tau = float(out.get("graph_v5_soft_pointer_temperature", 0.0) or 0.0)
                extra_field = (
                    f"g_n={gn:.2f} g_e={ge:.2f} src_ent={se:.2f} uniq={uq:.2f} "
                    f"τ={tau:.3f}"
                )
                aux_tag, aux_display = "aux", float(out["loss_aux"])
            elif "splat_aux" in out and out["splat_aux"] is not None:
                aux_display = float(out["splat_aux"])
                aux_tag = "s_aux"
            else:
                aux_display = float(out["loss_aux"])
                aux_tag = "aux"
            print(f"  step {step:6d}/{n_steps}  recon={float(out['loss_recon']):.4f}  "
                  f"top1={float(out['top1_acc'])*100:5.1f}%  "
                  f"{aux_tag}={aux_display:.3f}  "
                  f"{extra_field + '  ' if extra_field else ''}"
                  f"gnorm={float(gn):6.2f}  lr={lr:.2e}  ({sps:.1f} step/s)",
                  flush=True)

        if step > 0 and step % val_every == 0:
            vm = run_val(model, val_set, device, val_batches, window_size)
            val_row = {"phase": "val", "step": step, "variant": variant, **vm}
            jsonl_fp.write(json.dumps(val_row) + "\n")
            extra = ""
            if "val_graph_v5_mp_buf_cross_node_cos_final" in vm:
                extra = f"  mp_cos={vm['val_graph_v5_mp_buf_cross_node_cos_final']:.3f}"
            print(f"    [val @ {step}]  recon={vm['val_loss_recon']:.4f}  "
                  f"top1={vm['val_top1_acc']*100:.1f}%{extra}",
                  flush=True)
            # Best-checkpoint save: only past the warmup-fluke window.
            # This is also the patience-reset signal — if best.pt updates,
            # the model is still improving; clear the staleness counter.
            improved = (step >= BEST_MIN_STEP
                        and vm["val_loss_recon"] < best_val_recon - 1e-4)
            if improved:
                best_val_recon = vm["val_loss_recon"]
                best_val_step = step
                save_checkpoint(model, opt, step, best_ckpt_path)
                print(f"    [best ckpt @ {step}]  val_recon={best_val_recon:.4f}",
                      flush=True)
                evals_since_best = 0
            else:
                # No improvement. Only count it against patience past
                # min_step_for_stop (warmup eval points don't trigger stops).
                if step >= min_step_for_stop:
                    evals_since_best += 1
                    if patience > 0 and evals_since_best >= patience:
                        print(f"    [early stop @ {step}]  {evals_since_best} "
                              f"evals without best.pt update "
                              f"(best val_recon={best_val_recon:.4f} "
                              f"@ step {best_val_step})", flush=True)
                        early_stopped = True
                        stopped_at_step = step
                        # Audit fix #9: stamp last_completed BEFORE break
                        # so the final .last.pt save records the correct
                        # step (otherwise it'd be one save_every behind).
                        last_completed = step
                        break

        if step > 0 and step % save_every == 0:
            # `step` here is the last completed step. Resume reads N → starts at N+1.
            # Stash best tracking in last.pt so resume preserves prior best.pt
            # rather than overwriting it with a worse first-val.
            save_checkpoint(
                model, opt, step, ckpt_path,
                best_val_recon=best_val_recon,
                best_val_step=best_val_step,
                evals_since_best=evals_since_best,
            )

        last_completed = step
        step += 1

    # Final save: `step` has been incremented past the last completed iter
    # (or equals start_step if the loop never ran). Persist last_completed so
    # resume's `+ 1` lands on the correct next step.
    save_checkpoint(
        model, opt, last_completed, ckpt_path,
        best_val_recon=best_val_recon,
        best_val_step=best_val_step,
        evals_since_best=evals_since_best,
    )

    # Final eval on BEST.PT, not on the last training step's weights.
    # Previously we evaluated whatever weights were in memory at end of
    # training — which is .last.pt, NOT .best.pt. With the patience
    # criterion, .last.pt is by definition a checkpoint that did NOT beat
    # the best. So final-eval was systematically biased high (worse) vs
    # the model's actual peak. Loading best.pt before the final eval gives
    # the trustworthy "what's the best this variant achieved" number.
    if best_ckpt_path.exists():
        sd = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(sd["model_state_dict"], strict=False)
        print(f"  [loaded best.pt @ step {best_val_step} for final eval]", flush=True)
    final_val = run_val(model, val_set, device, val_batches, window_size)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, "evaluated_on_best": best_ckpt_path.exists(),
        "best_step": best_val_step, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    src = f"best.pt @ {best_val_step}" if best_ckpt_path.exists() else f"last weights"
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_recon={final_val['val_loss_recon']:.4f} "
          f"top1={final_val['val_top1_acc']*100:.1f}% ({src})", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "final_val_loss_recon": final_val["val_loss_recon"],
        "final_val_top1_acc": final_val["val_top1_acc"],
        "final_val_per_family": final_val["val_per_family"],
        "best_val_loss_recon": (best_val_recon if best_val_step >= 0
                                else final_val["val_loss_recon"]),
        "best_val_step": best_val_step,
        "early_stopped": early_stopped,
        "stopped_at_step": stopped_at_step,
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
        "vanilla_llama",          # loss floor
        "vanilla_full_context",   # loss ceiling
    ])
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=None,
                    help="Override cfg.learning_rate (default 1e-4). Scale with "
                         "BS — e.g. sqrt rule: 1e-4×sqrt(BS/2) → BS=16 ≈ 2.5e-4.")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--val-batches", type=int, default=32,
                    help="Number of batches in the fixed val set. With the "
                         "composite mix sampling 9 families + 3 external sources, "
                         "10 batches ≈ 1 example per family (high per-family "
                         "noise). 32 batches × BS=2 = 64 examples ≈ ~5 per family. "
                         "Bumped from old default of 10.")
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
                    help="Enable NarrativeQA source (default: DISABLED). "
                         "Uses random window (oracle-centering removed in "
                         "post-audit fix).")
    ap.add_argument("--musique", action="store_true",
                    help="Enable MuSiQue-Ans source (default: DISABLED). "
                         "Contamination-controlled 2-4 hop QA — complements "
                         "HotpotQA by eliminating shortcut reasoning.")
    ap.add_argument("--babilong", action="store_true",
                    help="Enable BABILong source (default: DISABLED). "
                         "Synthetic state-tracking, pre-formatted at the "
                         "config length (4k/8k/16k).")
    ap.add_argument("--babilong-config", type=str, default="auto",
                    help="BABILong length config. 'auto' picks the closest "
                         "config below chunk_size (e.g. 4k for chunk=4096, "
                         "8k for chunk=8192). Manual: 0k, 1k, 2k, 4k, 8k, "
                         "16k, 32k, 64k, 128k.")
    ap.add_argument("--mix-weights", nargs="+", type=float,
                    default=[0.7, 0.3, 0.0, 0.0, 0.0],
                    metavar="W",
                    help="Sampling weights for (composite, hotpot, narrative, "
                         "musique, babilong). Older 3-tuple callers still work; "
                         "missing entries default to 0.")
    ap.add_argument("--composite-task-weights", nargs="+", default=None,
                    metavar="FAMILY:W",
                    help="Per-family weights inside composite_v1 (e.g. "
                         "'biographical:1.0' to train only on biographical, or "
                         "'biographical:2.0 hotpot_qa:1.0' for 2:1 ratio). "
                         "Unlisted families get weight 0 (filtered out). "
                         "Default: None = all 9 families sampled uniformly.")
    ap.add_argument("--patience", type=int, default=5,
                    help="Stop training when best.pt hasn't updated for this "
                         "many consecutive val evals past --min-step-for-stop. "
                         "Best-staleness criterion (was previously smoothed "
                         "rolling mean — that one triggered on volatility "
                         "and could fire on the same step a new best landed). "
                         "0 disables. Default 5 (≈ 2500-step plateau at "
                         "val_every=500).")
    ap.add_argument("--min-step-for-stop", type=int, default=3000,
                    help="Don't trigger early-stop before this step. Skips "
                         "warmup-noise era where val is bouncy. Bumped 2000→"
                         "3000 after tranche 1 v2: flat_baseline was still "
                         "improving past step 5000 when patience fired at 5k. "
                         "Slow learners need more runway before plateau check.")
    args = ap.parse_args()

    if "v21" in args.variants:
        raise SystemExit("v21 is not supported in v1h yet.")

    # Flag/weight consistency. The flags toggle source *availability*; the
    # weights control *sampling*. A source with weight=0 is never sampled,
    # so enabling the flag without bumping the weight is a no-op that
    # silently loads ~570MB of data (HotpotQA) or downloads NarrativeQA
    # while contributing nothing. Surface this immediately.
    # Pad mix_weights to length 5 for the 5-source schema.
    padded_weights = list(args.mix_weights) + [0.0] * (5 - len(args.mix_weights))
    args.mix_weights = padded_weights

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

    # Parse --composite-task-weights "family:weight" pairs into a dict.
    composite_task_weights = None
    if args.composite_task_weights:
        composite_task_weights = {}
        for item in args.composite_task_weights:
            if ":" not in item:
                raise SystemExit(
                    f"--composite-task-weights expects 'family:weight', got {item!r}"
                )
            fam, w = item.split(":", 1)
            composite_task_weights[fam.strip()] = float(w)
        print(f"[composite] per-family weights: {composite_task_weights}")
    if args.musique and args.mix_weights[3] <= 0:
        raise SystemExit(
            "--musique is set but mix_weights[3] (MuSiQue) is 0. Either drop "
            "--musique or pass --mix-weights with a positive fourth value "
            "(e.g. --mix-weights 0.4 0.2 0.2 0.2)."
        )
    if args.babilong and args.mix_weights[4] <= 0:
        raise SystemExit(
            "--babilong is set but mix_weights[4] (BABILong) is 0. Either drop "
            "--babilong or pass --mix-weights with a positive fifth value "
            "(e.g. --mix-weights 0.35 0.15 0.15 0.15 0.2)."
        )

    # Tranche-2 sizing (chunk=8192, M=128, 64× compression):
    #   graph_v5: K_node=128, K_edge=196, K_proposal=196, d_node=d_state=256
    #     → substrate = 128·256 + 196·(2·256+256) = 32,768 + 150,528 = 183,296 floats ≈ 179K
    #   baselines: n_flat_codes=128, d_inner=1398 → 128·1398 = 178,944 floats (matched)
    #   Memory floats reaching Llama: M=128 × d_llama=2048 = 262K → 8192·2048 / 262K = 64×
    cfg = ReprConfig(
        batch_size=args.batch_size,
        fixed_window_size=args.window_size,
        max_window_size=args.chunk_size,
        max_steps=args.steps,
        warmup_steps=500,
        d_node_state=128,
        n_edges=68,
        n_flat_codes=128,                 # was 36 → 128 (M=K_node, 64× compression)
        d_continuous=1398,                # was 725 → 1398 (substrate match)
        d_concept_baseline=1398,
        d_mt_value=1398,
        d_recurrent=1398,
        graph_v5_K_node=128,              # was 32
        graph_v5_K_edge=196,              # was 57
        graph_v5_K_proposal=196,          # was 80
        graph_v5_d_node=256,              # was 128
        graph_v5_d_state=256,             # was 128
        edge_token_packing="fused",
        b_diversity_scale=args.b_diversity_scale,
        mt_diversity_scale=args.mt_diversity_scale,
        d_mamba=768,
        **({"learning_rate": args.lr} if args.lr is not None else {}),
    )

    # Auto-pick BABILong config to match chunk_size (audit fix #10).
    if args.babilong_config == "auto":
        # Map chunk_size to nearest BABILong config at or below it.
        cs = args.chunk_size
        if cs >= 16384:
            args.babilong_config = "16k"
        elif cs >= 8192:
            args.babilong_config = "8k"
        elif cs >= 4096:
            args.babilong_config = "4k"
        elif cs >= 2048:
            args.babilong_config = "2k"
        elif cs >= 1024:
            args.babilong_config = "1k"
        else:
            args.babilong_config = "0k"
        if args.babilong:
            print(f"[auto] babilong_config = {args.babilong_config} "
                  f"(scaled for chunk_size={args.chunk_size})")

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
            use_musique=args.musique,
            use_babilong=args.babilong,
            babilong_config=args.babilong_config,
            mix_weights=tuple(args.mix_weights),
            composite_task_weights=composite_task_weights,
            patience=args.patience,
            min_step_for_stop=args.min_step_for_stop,
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
