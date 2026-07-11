"""Post-training band + binding-gate eval for the mixed4k_bio campaign.

For every TRAINED variant (graph + 4 baselines) × every mixed task:
  - REAL      : val loss with memory ON
  - OFF       : val loss with memory ZEROED (per-variant no-memory floor)
  - SHUF      : val loss with memory rolled across the batch (example-specificity probe)
  - OFF-REAL  : >0 ⇒ the memory is USED (helps vs no memory)
  - SHUF-REAL : >0 ⇒ the memory is EXAMPLE-SPECIFIC (not a generic prior)
  - babi EM   : exact-match on the bAbI task

Plus a SHARED band per task from two fresh eval-only references (untrained,
identity-LoRA → clean reference points):
  - FLOOR = vanilla_llama          (no memory at all)
  - CEIL  = vanilla_full_context   (decoder sees the full uncompressed context)
  - %band = (FLOOR - REAL) / (FLOOR - CEIL) * 100   (0 = no better than no-memory,
            100 = as good as seeing the full context)

CRITICAL: every TRAINED variant's config is rebuilt from its checkpoint's
metadata.cfg_dict (NOT module defaults), so the graph's saved shapes
(N=2048/E=128/M=32/d_graph=256/W3R2) match the state_dict. (The old diagnostic
scripts rebuilt from defaults → shape-mismatch crash.)

Usage:
  python scripts/diagnostics/mixed/mixed_band_gate_eval.py [--val-batches 32] [--out-tag mixed4k_bio]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from transformers import AutoTokenizer

from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.training import make_mixed_val_sets, run_val, _continuation_early_loss
from src.memory.data.mixes import TASK_MODE, DEFAULT_TRAIN_MIX, DEFAULT_MIXED_M
from src.memory.data.sources.babi import DEFAULT_TASKS as BABI_DEFAULT_TASKS

TRAINED_VARIANTS = [
    # active trainable cohort (2026-07-11); missing ckpts are skipped gracefully
    "icae_baseline", "autocompressor_baseline", "titans_baseline",
    "gisting_baseline", "memoryllm_baseline", "slotgraph_baseline",
]
# Campaign launch constants (must match the run that produced the checkpoints — the current 2026-07 config).
MIXED_CTX = 2048            # was 1024 (stale)
MIXED_M = DEFAULT_MIXED_M   # follows the mixed default (mixes.DEFAULT_MIXED_M = 96)
WINDOW_SIZE = 256           # 8 streaming windows at ctx 2048 (was 1024 = single-shot, stale)
PREDICT_LEN = 64
MAE_SRC_TOK = "meta-llama/Llama-3.2-1B"


def _ckpt_path(out_tag: str, variant: str) -> Path:
    base = REPO / f"outputs/memory/{out_tag}_{variant}/ckpts"
    best = base / f"{variant}.best.pt"
    return best if best.exists() else base / f"{variant}.last.pt"   # prefer the early-stop BEST


def _load_cfg_from_ckpt(ckpt: Path) -> tuple[ReprConfig, dict]:
    import dataclasses
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    valid = {f.name for f in dataclasses.fields(ReprConfig)}     # drop fields removed since training
    # cfg_all captures the DYNAMICALLY-attached attrs (ranks, objective_mode, kl_coef, …) that
    # dataclasses.asdict drops from cfg_dict — required to rebuild the exact trained shapes (autocompressor
    # r52 / memoryllm r39 / titans h4650), else the rebuilt model shape-mismatches the state_dict.
    cfg_src = sd["metadata"].get("cfg_all") or sd["metadata"]["cfg_dict"]
    cfg = ReprConfig(**{k: v for k, v in cfg_src.items() if k in valid})
    # cfg_all also holds the ~20 DYNAMICALLY-attached fields cli.py sets (titans_mem_hidden=4650,
    # memoryllm_lora_rank=39, gisting_*, kl_coef, …) that are NOT ReprConfig dataclass fields, so the
    # constructor above cannot accept them. Restore them via setattr — WITHOUT this the encoder falls
    # back to its getattr default (titans h=4864, memoryllm r=46) and load_state_dict RAISES on the
    # resulting shape mismatch (strict=False still errors on shape), crashing the eval of titans/memoryllm.
    for k, v in cfg_src.items():
        if k not in valid:
            setattr(cfg, k, v)
    return cfg, sd


def _eval_variant(variant, cfg, state_dict, tokenizer, val_sets, tasks,
                  device, gate: bool):
    """Build `variant` from cfg, optionally load weights, eval per task.

    gate=True  → trained variants: REAL/OFF/SHUF over ALL val batches.
    gate=False → reference variants (vanilla_*): REAL only (the band endpoints).
    """
    llama_arg = None if cfg.use_llama_lora else None  # always self-load a fresh frozen base
    model = ReprLearningModel(cfg, variant=variant, llama_model=llama_arg).to(device)
    if state_dict is not None:
        res = model.load_state_dict(state_dict, strict=False)
        # Only the frozen Llama backbone should be "missing" (it's reloaded, not saved). Any OTHER missing
        # key = a trainable param that never loaded (shape/config mismatch → silently zero-init → invalid
        # eval), so surface it. Unexpected keys are equally worth flagging.
        _frozen = ("decoder.llama.", "encoder.base.")
        missing = [k for k in res.missing_keys if not any(k.startswith(p) for p in _frozen)]
        if missing:
            print(f"  [WARN] {variant}: {len(missing)} TRAINABLE keys MISSING from ckpt "
                  f"(config mismatch → zero-init → results INVALID; e.g. {missing[:2]})")
        if res.unexpected_keys:
            print(f"  [warn] {variant}: {len(res.unexpected_keys)} unexpected keys "
                  f"(e.g. {res.unexpected_keys[:2]})")
    model.train(False)

    out = {}
    n = len(next(iter(val_sets.values())))   # batches per task (uniform)
    for t in tasks:
        model.task_mode = TASK_MODE[t]
        gb = n if gate else 0
        vm = run_val(model, val_sets[t], device, n_batches=n,
                     window_size=WINDOW_SIZE, gate_batches=gb)
        rec = {
            "real": vm["val_loss_recon"],
            "top1": vm["val_top1_acc"],
        }
        if "val_loss_recon_off" in vm:
            rec["off"] = vm["val_loss_recon_off"]
            rec["off_minus_real"] = vm["val_off_minus_real"]
        if "val_loss_recon_shuf" in vm:
            rec["shuf"] = vm["val_loss_recon_shuf"]
            rec["shuf_minus_real"] = vm["val_shuf_minus_real"]
        if "val_babi_em" in vm:
            rec["babi_em"] = vm["val_babi_em"]
        if t == "continuation":
            rec["early"] = _continuation_early_loss(
                model, val_sets[t], device, n, WINDOW_SIZE)
        out[t] = rec

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-tag", default="mixed4k_bio")
    ap.add_argument("--val-batches", type=int, default=32)
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_TRAIN_MIX))
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    device = "cuda"
    tasks = args.tasks

    # A representative cfg (for tokenizer + val-set batch_size/pad) — shared across
    # variants. Load from the graph ckpt; fall back to the first available.
    base_cfg = None
    for v in TRAINED_VARIANTS:
        p = _ckpt_path(args.out_tag, v)
        if p.exists():
            base_cfg, _ = _load_cfg_from_ckpt(p)
            break
    if base_cfg is None:
        raise SystemExit(f"no checkpoints found under outputs/memory/{args.out_tag}_*")

    tokenizer = AutoTokenizer.from_pretrained(base_cfg.llama_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[band+gate] building per-task val sets ({args.val_batches} batches each): {tasks}")
    val_sets = make_mixed_val_sets(
        tasks, tokenizer, base_cfg, args.val_batches, ctx_len=MIXED_CTX,
        m_slots=MIXED_M, mae_src_tok=MAE_SRC_TOK, babi_tasks=BABI_DEFAULT_TASKS,
        predict_len=PREDICT_LEN)
    print(f"  val sets ready: {{ {', '.join(f'{t}:{len(val_sets[t])}' for t in tasks)} }}")

    results = {}

    # --- shared band endpoints (fresh, eval-only references) ---
    for ref in ("vanilla_llama", "vanilla_full_context"):
        print(f"\n[ref] {ref} ...", flush=True)
        results[ref] = _eval_variant(ref, base_cfg, None, tokenizer, val_sets,
                                     tasks, device, gate=False)

    # --- trained variants (per-variant cfg from ckpt) + binding gate ---
    for v in TRAINED_VARIANTS:
        p = _ckpt_path(args.out_tag, v)
        if not p.exists():
            print(f"\n[skip] {v}: no checkpoint at {p}")
            continue
        print(f"\n[eval] {v} (cfg from ckpt) ...", flush=True)
        cfg, sd = _load_cfg_from_ckpt(p)
        results[v] = _eval_variant(v, cfg, sd["model_state_dict"], tokenizer,
                                   val_sets, tasks, device, gate=True)

    # --- H2O: training-free KV-eviction reference (no checkpoint; still a memory method → gate it) ---
    print(f"\n[eval] h2o_baseline (training-free) ...", flush=True)
    results["h2o_baseline"] = _eval_variant("h2o_baseline", base_cfg, None, tokenizer,
                                            val_sets, tasks, device, gate=True)

    _print_tables(results, tasks)

    out_json = args.out_json or str(REPO / f"outputs/memory/{args.out_tag}_band_gate.json")
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[band+gate] wrote {out_json}")


def _print_tables(results, tasks):
    display = [v for v in TRAINED_VARIANTS + ["h2o_baseline"] if v in results]
    for t in tasks:
        floor = results.get("vanilla_llama", {}).get(t, {}).get("real")
        ceil = results.get("vanilla_full_context", {}).get(t, {}).get("real")
        band = (floor - ceil) if (floor is not None and ceil is not None) else None
        # A valid band needs full-context to actually BEAT no-memory. On reconstruct/MAE the memory can hurt
        # (full-ctx ≥ no-mem → band ≤ 0), so %band would divide by a non-positive number → meaningless.
        band_ok = band is not None and band > 1e-3
        print(f"\n{'='*92}")
        if band is not None:
            flag = "" if band_ok else "  ⚠ INVALID (full-ctx not a ceiling → %band n/a)"
            print(f"TASK: {t}   FLOOR(no-mem)={floor:.3f}  CEIL(full-ctx)={ceil:.3f}  band={band:.3f}{flag}")
        else:
            print(f"TASK: {t}")
        print(f"{'='*92}")
        hdr = f"{'variant':<26} {'REAL':>7} {'OFF':>7} {'OFF-REAL':>9} {'SHUF-REAL':>10} {'%band':>7}"
        if t == "babi":
            hdr += f" {'TF-EM':>6}"      # teacher-forced span match, NOT autoregressive exact-match
        print(hdr)
        print("-" * len(hdr))
        for v in display:
            r = results.get(v, {}).get(t)
            if r is None:
                continue
            real = r["real"]
            off = r.get("off", float("nan"))
            offmr = r.get("off_minus_real", float("nan"))
            shufmr = r.get("shuf_minus_real", float("nan"))
            pct = f"{100.0 * (floor - real) / band:>6.1f}%" if band_ok else f"{'n/a':>7}"
            line = (f"{v:<26} {real:>7.3f} {off:>7.3f} {offmr:>+9.3f} "
                    f"{shufmr:>+10.3f} {pct}")
            if t == "babi":
                line += f" {100*r.get('babi_em', float('nan')):>5.1f}%"
            print(line)


if __name__ == "__main__":
    main()
