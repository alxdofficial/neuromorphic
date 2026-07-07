"""The two live trainer loops (single-task + mixed round-robin) and the max-BS VRAM probe.

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). The ONLY change from
the original bodies: ``train_mixed_variant`` routes task_mode via ``mixes.task_mode(task)`` instead
of the deleted module-level ``MIXED_TASK_MODE`` dict (same value). ``REPO`` / ``COMPOSITE_*`` are
redefined here (their point of use) with the path depth for this module's location; they resolve to
the identical absolute paths.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from src.memory.config import ReprConfig
from src.memory.data.mixed import make_mixed_qa_dataloader
from src.memory.data.common import collate_qa
from src.memory.data.sources.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS
from src.memory.model import ReprLearningModel
from src.memory.data import mixes

from .utils import lr_at_step, to_device, materialize_val_set
from .eval import run_val, run_mixed_val
from .checkpoint import save_checkpoint, _ckpt_metadata, _grad_group_norm
from .objectives import _grad_cached_objective_step, _coding_rate, _behavioral_kl_step
from .data_mix import make_mixed_train_dataloaders, make_mixed_val_sets

# src/memory/training/loops.py → parents[3] == repo root (train.py used parents[2] from
# scripts/train/); the resolved COMPOSITE_* paths are identical.
REPO = Path(__file__).resolve().parents[3]
COMPOSITE_TRAIN_P = REPO / "data/bio/train/passages.jsonl"
COMPOSITE_TRAIN_Q = REPO / "data/bio/train/questions.jsonl"
COMPOSITE_VAL_P   = REPO / "data/bio/val/passages.jsonl"
COMPOSITE_VAL_Q   = REPO / "data/bio/val/questions.jsonl"


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
    min_delta: float = 0.01,                 # min val_recon drop to count as a real improvement
    task: str = "masked_reconstruction",     # "masked_reconstruction" = sentence-pair MAE (active); "qa" = composite-QA mix; "conditioned_reconstruction" = key→value; "continuation" = gist-LM
    cond_recon_n_pairs: int = 64, cond_recon_n_query: int = 1, cond_recon_value_len: int = 1,
    cond_recon_bio_n_facts: int = 3, cond_recon_bio_world_seed: int = 0,
    babi_tasks: tuple = BABI_DEFAULT_TASKS,
    compress_len: int = 2048, predict_len: int = 64,
    mae_src_tok: str = "meta-llama/Llama-3.2-1B",
) -> dict:
    device = "cuda"
    # LoRA wraps Llama's q/v in place (LoRALinear). The base `llama` is shared
    # across variants, so when LoRA is on each variant must get its OWN fresh
    # frozen Llama (pass None → the decoder self-loads one) — otherwise variant N
    # would double-wrap variant N-1's LoRALinear and inherit its trained adapter.
    # LoRA off → keep sharing the single read-only frozen Llama (fast).
    llama_arg = None if cfg.use_llama_lora else llama
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
    model.task_mode = task   # explicit dispatch (MAE→masked_reconstruction path; else generic compute_loss)
    n_trainable = model.n_trainable_params()
    # A variant is eval-only iff it has no trainable params. With LoRA-all on, the
    # two vanilla references DO train (their ~1.7M LoRA) — they become the LoRA'd
    # floor (no context) and ceiling (full context), not frozen reference points.
    # vanilla_full_context is kept FROZEN/eval-only (the ceiling): it re-forwards
    # the full ~8192-token context per question (~40x the decoder tokens of the
    # memory arms), so LoRA-training it OOMs under backward — and a frozen
    # full-context Llama is already a valid upper bound. Its zero-init LoRA, if
    # built, is an identity no-op. vanilla_llama (the floor) still LoRA-trains.
    is_eval_only = (n_trainable == 0) or (
        variant == "vanilla_full_context" and task != "masked_reconstruction")
    print(f"\n{'='*78}")
    print(f"Variant: {variant}  ({n_trainable:,} trainable, {n_steps} steps)")
    print(f"{'='*78}")

    # vanilla_* reference points:
    #   - vanilla_llama: no memory at all (LoRA'd loss FLOOR)
    #   - vanilla_full_context: passes raw context embeddings as memory (LoRA'd CEILING)
    # With LoRA off they have 0 trainable params → eval-only. With LoRA-all on they
    # train their shared ~1.7M adapter exactly like every other arm.
    if not is_eval_only:
        opt = torch.optim.AdamW(
            model.trainable_parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            betas=(0.9, 0.95),
            fused=torch.cuda.is_available(),
        )
    else:
        opt = None

    # qa (composite multi-hop mix) — the only single-task path; the 4-task `mixed`
    # benchmark has its own trainer (train_mixed_variant). Same encoder→memory→
    # prepend→CE path; only the DATA differs.
    train_dl = None if is_eval_only else make_mixed_qa_dataloader(
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
    # 10% of total (min 1000) to skip early-warmup flukes, but never exceed the run
    # itself — short smokes (e.g. 600 steps) must still save a best after warmup [merge #5]
    BEST_MIN_STEP = min(1000, max(val_every, n_steps // 10))
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

    if is_eval_only:
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
        # Save a loadable ckpt so the per-family eval harness can load this
        # eval-only arm (e.g. vanilla_full_context) uniformly — load_variant
        # requires a ckpt. No optimizer state (nothing trained); cfg_dict +
        # mask_embed + zero-init LoRA are all eval needs.
        best_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        _keep = lambda k: (not k.startswith("decoder.llama.")) or ("lora_" in k)
        torch.save({
            "step": 0,
            "model_state_dict": {k: v for k, v in model.state_dict().items() if _keep(k)},
            "metadata": _ckpt_metadata(model),
            "eval_only": True,
        }, best_ckpt_path)
        print(f"  [eval-only] saved ckpt → {best_ckpt_path}", flush=True)
        del model
        torch.cuda.empty_cache()
        return summary

    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        # objective/backbone drift check — a resume that silently switches task or
        # backbone would corrupt the run (the metadata's whole reason for existing).
        _meta = sd.get("metadata", {}) or {}
        _ck_cfg = _meta.get("cfg_dict", {}) or {}
        for _field, _now in (("backbone_model", model.cfg.llama_model),
                             ("task_mode", getattr(model.cfg, "task_mode", None))):
            _was = _meta.get(_field, _ck_cfg.get(_field))
            if _was is not None and _was != _now:
                raise RuntimeError(
                    f"[resume] {_field} drift: checkpoint has {_was!r} but this run "
                    f"is {_now!r}. Resuming would mix objectives/backbones — refusing. "
                    f"Start a fresh --out-tag or match the original setting.")
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower()
                and not k.startswith("encoder.base.")      # frozen own-base (dropped by keep())
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(
                "[resume] checkpoint/arch mismatch — non-Llama missing/unexpected "
                f"keys would silently leave params random: {_bad[:12]}")
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
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()   # per-variant peak-VRAM tracking

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
        _do_contrast = getattr(cfg, "contrastive_shuf_coef", 0.0) > 0
        # REAL and SHUF must see the SAME random mask, else the contrastive term
        # absorbs mask-difficulty noise (the masking is drawn fresh each compute_loss
        # call). Capture the RNG before REAL and restore it before SHUF so SHUF redraws
        # the identical mask — REAL/SHUF then differ only in memory identity. [R1-M3]
        if _do_contrast:
            _rng = torch.cuda.get_rng_state()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_loss(batch, window_size=window_size)
        loss = out["loss"]
        if _do_contrast:
            # contrastive binding pressure: gradient flows through BOTH branches
            # (lower REAL loss AND raise SHUF loss — both require doc-identity in
            # the state to be expressed by the read). Primary CE on REAL keeps
            # the help-on-match direction anchored against poison-on-mismatch.
            torch.cuda.set_rng_state(_rng)            # SHUF redraws the SAME mask as REAL
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out_shuf = model.compute_loss(batch, window_size=window_size,
                                                 shuffle_memory=True)
            contrast = torch.nn.functional.softplus(out["loss"] - out_shuf["loss"])
            loss = loss + cfg.contrastive_shuf_coef * contrast
        if not torch.isfinite(loss):
            print(f"  [step {step}] FATAL: non-finite loss = {float(loss)}")
            break
        loss.backward()
        # Per-module grad-norm split BEFORE clipping: separates the memory
        # mechanism (encoder/graph write+read) from the decoder LoRA so we can
        # SEE if the memory params are gradient-starved relative to the adapter
        # (a global norm hides this — the #1 graph training-failure mode).
        _gn_enc = _grad_group_norm(
            p for n, p in model.named_parameters()
            if p.requires_grad and not n.startswith("decoder.llama."))
        _gn_lora = _grad_group_norm(
            p for n, p in model.named_parameters()
            if p.requires_grad and n.startswith("decoder.llama."))
        # Per-MECHANISM gradient flow (graph variant): is any sub-module being
        # gradient-starved / ignored (the model learning to route around it)?
        # Splits write (obs/blocks/endpoint-heads/edge-head/slots) and read
        # (q_in/blocks/out/gate/tag-role) so the sweep can see WHICH path carries
        # the learning signal. (VQ codebook is EMA — no grad — intentionally absent.)
        graph_gn = {}
        if variant == "graph_baseline":
            _gmech = {
                "p_obs": ("encoder.parser.obs_proj",),
                "p_blocks": ("encoder.parser.blocks",),
                "p_ptr": ("encoder.parser.q_src", "encoder.parser.q_dst", "encoder.parser.bank_key",
                          "encoder.parser.log_temp"),   # incl. the pointer sharpness param
                "p_edge": ("encoder.parser.edge_head",),
                "p_bank": ("encoder.parser.node_bank", "encoder.parser.node_role_avail"),
                "p_slots": ("encoder.parser.init_graph", "encoder.parser.role",
                            "encoder.parser.tag", "encoder.parser.part"),
                # Perceiver read: latent queries + role/tag KV features + cross/self
                # blocks (each incl. its pre-LN) + FFN(+final norm) + out projection.
                "r_queries": ("encoder.reader.queries",),
                "r_kv": ("encoder.reader.role_emb", "encoder.reader.tag"),
                "r_cross": tuple(p for i in range(cfg.graph_read_layers)
                                 for p in (f"encoder.reader.blocks.{i}.cross",
                                           f"encoder.reader.blocks.{i}.cn")),
                "r_self": tuple(p for i in range(cfg.graph_read_layers)
                                for p in (f"encoder.reader.blocks.{i}.slf",
                                          f"encoder.reader.blocks.{i}.sn")),
                "r_ffn": tuple(p for i in range(cfg.graph_read_layers)
                               for p in (f"encoder.reader.blocks.{i}.ff",
                                         f"encoder.reader.blocks.{i}.fn")) + ("encoder.reader.norm",),
                "r_out": ("encoder.reader.out",),
            }
            for _lab, _prefs in _gmech.items():
                graph_gn[f"graph_gn_{_lab}"] = _grad_group_norm(
                    p for n, p in model.named_parameters()
                    if p.requires_grad and any(n.startswith(x) for x in _prefs))
        # PER-GROUP clipping (audit fix): a single global clip is dominated by the
        # LoRA group (>99% of the norm), silently shrinking the memory side's step
        # whenever LoRA spikes. Clip memory and LoRA separately at the same bound.
        _mem_params = [p for n, p in model.named_parameters()
                       if p.requires_grad and not n.startswith("decoder.llama.")]
        _lora_params = [p for n, p in model.named_parameters()
                        if p.requires_grad and n.startswith("decoder.llama.")]
        gn_mem = torch.nn.utils.clip_grad_norm_(_mem_params, cfg.grad_clip)
        gn_lora = torch.nn.utils.clip_grad_norm_(_lora_params, cfg.grad_clip)
        gn = (gn_mem ** 2 + gn_lora ** 2) ** 0.5
        opt.step()

        row = {
            "step": step,
            "variant": variant,
            "loss": float(out["loss"].detach()),
            "loss_recon": float(out["loss_recon"]),
            "loss_aux": float(out["loss_aux"]),               # load_balance only (unweighted)
            "top1_acc": float(out["top1_acc"]),
            "n_content_positions": int(out["n_content_positions"]),
            "grad_norm": float(gn),
            "grad_norm_memory": _gn_enc,    # encoder/graph write+read params
            "grad_norm_lora": _gn_lora,     # decoder rank-16 q/v adapter
            "lr": lr,
            "memory_M": out["memory_shape"][1],
        }
        # flat_baseline codebook health (codes_active = #live codes).
        for key in ("codes_active", "routing_entropy"):
            if key in out and out[key] is not None:
                row[key] = float(out[key])
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
        # The relational parser emits the selection/vocabulary/relation/read canaries
        # (pointer entropy, distinct nodes used, eff-ranks, read-gate amplitude).
        _graph_scalar_keys = (
            # relational-parser canaries (selection → vocabulary → relation → read)
            "graph_ptr_entropy", "graph_nodes_used", "graph_bank_effrank",
            "graph_edge_effrank", "graph_read_effrank", "graph_read_gate",
        )
        for key in _graph_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        row.update(graph_gn)         # per-mechanism gradient flow (graph variant)
        # soft_pointer_graph telemetry (variant == soft_pointer_graph_baseline).
        # Write-side gate means + fact norm are emitted every step; the read-side
        # health probes (rezero/state_effect/collapse/entropy/active_frac) are
        # eval-only (computed in finalize_memory when not self.training), so on
        # train steps only the first three are present — `if key in out` handles it.
        _spg_scalar_keys = (
            "spg_node_gate_mean_avg", "spg_edge_gate_mean_avg",
            "spg_fact_norm",
            "spg_rezero_scale_eff", "spg_state_effect",
            "spg_node_collapse_cos",
            "spg_read_src_entropy", "spg_read_dst_entropy",
            "spg_node_active_frac",
        )
        for key in _spg_scalar_keys:
            if key in out and out[key] is not None:
                row[key] = float(out[key])
        # hlvocab telemetry (variant == hlvocab_baseline) — emitted every step.
        for key, value in out.items():
            if not key.startswith("hlvocab_") or value is None:
                continue
            if isinstance(value, (int, float)):
                row[key] = float(value)
            elif torch.is_tensor(value) and value.numel() == 1:
                row[key] = float(value.detach())
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
            if variant == "soft_pointer_graph_baseline":
                # show node/edge gate means (anchor-init ≈ 0.38 / 0.27).
                g_n = float(out.get("spg_node_gate_mean_avg", 0.0) or 0.0)
                g_e = float(out.get("spg_edge_gate_mean_avg", 0.0) or 0.0)
                extra_field = f"g_n={g_n:.2f} g_e={g_e:.2f}"
                aux_tag, aux_display = "aux", float(out["loss_aux"])
            elif variant == "graph_baseline":
                # live collapse canaries: nodes used, pointer entropy, read eff-rank,
                # read gate. Watch nodes→1 (vocabulary collapse), ent high (pointer not
                # sharpening), rd_er→1 (membership-not-binding), gate→0 (read ignored).
                extra_field = (f"nodes={float(out.get('graph_nodes_used',0)):.0f} "
                               f"ent={float(out.get('graph_ptr_entropy',0)):.2f} "
                               f"rd_er={float(out.get('graph_read_effrank',0)):.1f} "
                               f"gate={float(out.get('graph_read_gate',0)):.3f}")
                aux_tag, aux_display = "ent", float(out.get("graph_ptr_entropy", 0.0))
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
            extra_parts = []
            if "val_off_minus_real" in vm:
                extra_parts.append(f"OFF-REAL={vm['val_off_minus_real']:+.3f}")
            if "val_shuf_minus_real" in vm:
                extra_parts.append(f"SHUF-REAL={vm['val_shuf_minus_real']:+.3f}")
            if "val_babi_em" in vm:
                extra_parts.append(f"babi_EM={vm['val_babi_em']*100:.1f}%")
            extra = ("  " + "  ".join(extra_parts)) if extra_parts else ""
            print(f"    [val @ {step}]  recon={vm['val_loss_recon']:.4f}  "
                  f"top1={vm['val_top1_acc']*100:.1f}%{extra}",
                  flush=True)
            # Best-checkpoint save: only past the warmup-fluke window.
            # This is also the patience-reset signal — if best.pt updates,
            # the model is still improving; clear the staleness counter.
            improved = (step >= BEST_MIN_STEP
                        and vm["val_loss_recon"] < best_val_recon - min_delta)
            if improved:
                best_val_recon = vm["val_loss_recon"]
                best_val_step = step
                evals_since_best = 0
                # best.pt carries its own best-tracking metadata so a --resume from
                # best.pt restores it (resume normally uses last.pt, which already
                # stashes these; this keeps best.pt self-describing too).
                save_checkpoint(
                    model, opt, step, best_ckpt_path,
                    best_val_recon=best_val_recon, best_val_step=best_val_step,
                    evals_since_best=evals_since_best,
                )
                print(f"    [best ckpt @ {step}]  val_recon={best_val_recon:.4f}",
                      flush=True)
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
        _res = model.load_state_dict(sd["model_state_dict"], strict=False)
        _bad = [k for k in (_res.missing_keys + _res.unexpected_keys)
                if "llama" not in k.lower()
                and not k.startswith("encoder.base.")      # frozen own-base (dropped by keep())
                and not k.startswith("decoder.llama.")]
        if _bad:
            raise RuntimeError(f"[final-best] checkpoint/arch mismatch — non-Llama keys: {_bad[:12]}")
        print(f"  [loaded best.pt @ step {best_val_step} for final eval]", flush=True)
    final_val = run_val(model, val_set, device, val_batches, window_size)
    jsonl_fp.write(json.dumps({
        "phase": "val", "step": step, "variant": variant,
        "final": True, "evaluated_on_best": best_ckpt_path.exists(),
        "best_step": best_val_step, **final_val,
    }) + "\n")
    jsonl_fp.close()

    elapsed = time.time() - t_start
    peak_vram_gb = (torch.cuda.max_memory_allocated() / 1e9
                    if torch.cuda.is_available() else 0.0)
    src = f"best.pt @ {best_val_step}" if best_ckpt_path.exists() else f"last weights"
    print(f"  DONE: {step} steps in {elapsed/60:.1f} min  "
          f"final val_recon={final_val['val_loss_recon']:.4f} "
          f"top1={final_val['val_top1_acc']*100:.1f}% ({src})  "
          f"peak_vram={peak_vram_gb:.1f}GB", flush=True)

    summary = {
        "variant": variant,
        "trainable_params": n_trainable,
        "n_steps": step,
        "elapsed_s": elapsed,
        "peak_vram_gb": peak_vram_gb,
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

    Distinct from train_one_variant (single-task) so the single-task path stays
    untouched. Per-task val rows are logged to the run jsonl, each tagged with
    `task` + `step`; per-task best_step/best_metric are tracked independently."""
    device = "cuda"
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

    # ── resume: restore model + optimizer + step + best tracking from .last.pt ──
    start_step = 0
    if resume and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location=device, weights_only=False)
        _meta = sd.get("metadata", {}) or {}
        _ck_cfg = _meta.get("cfg_dict", {}) or {}
        for _field, _now in (("backbone_model", model.cfg.llama_model),
                             ("task_mode", getattr(model.cfg, "task_mode", None))):
            _was = _meta.get(_field, _ck_cfg.get(_field))
            if _was is not None and _was != _now:
                raise RuntimeError(
                    f"[resume] {_field} drift: checkpoint has {_was!r} but this run is {_now!r}. "
                    f"Refusing to mix objectives/backbones — use a fresh --out-tag.")
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
        print(f"[objective] mode={_mode_banner}"
              + (f"  InfoNCE coef={cfg.objective_coef} ({'in-batch, all B-1 negatives'})" if _mode_banner != 'plain' else "")
              + (f"  GRPO G={cfg.grpo_samples} coef={cfg.grpo_coef} (router-only REINFORCE)"
                 if _mode_banner == "trajectory" else "")
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
        elif _mode in ("contrastive", "trajectory"):
            # in-batch InfoNCE (+GRPO): backward runs INSIDE (GradCache memory cut) — no
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
            if _v is None or not _k.startswith(("biomem_", "slotgraph_", "slotgraph2_", "slotgraph3_", "vqicae_")):
                continue
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
            if agg < agg_best["metric"]:
                agg_best["metric"], agg_best["step"] = agg, step
                save_checkpoint(model, opt, step, best_ckpt_path,
                                mixed_best=best, mixed_agg_best=agg_best)
                print(f"    [best @ {step}]  agg_val={agg:.4f} → saved {best_ckpt_path.name}", flush=True)

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
