"""Smoke + timing harness: every active baseline × every objective on REAL data.

Phase A (correctness): all 7 active variants × 5 objectives at SMALL real contexts —
one forward+backward, assert finite loss AND a finite trainable grad. Fast; catches
shape/device/dtype/integration regressions (incl. the just-fixed beacon mask, ccm carry,
graph canaries, continuation tokenizer, bio resample).

Phase B (timing): per objective at the REAL training context/M, measure mean fwd+bwd+opt
step time for each variant on the 4090 → steps-for-~1hr-cohort + a batch-size suggestion.

  python scripts/diagnostics/smoke_all_objectives.py            # both phases
  python scripts/diagnostics/smoke_all_objectives.py --phase a  # correctness only
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from transformers import AutoTokenizer, AutoConfig

from src.memory.config import ReprConfig
from src.memory.common import resolve_special_ids
from src.memory.model import ReprLearningModel
from scripts.train.train import (
    to_device, COMPOSITE_TRAIN_P, COMPOSITE_TRAIN_Q,
)

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
SRC_TOK = "meta-llama/Llama-3.2-1B"          # FineWeb parquet/text_cache source tokenizer
DEV = "cuda"
COHORT = ["vanilla_llama", "vanilla_full_context", "graph_baseline",
          "icae_baseline", "ccm_baseline", "autocompressor_baseline", "beacon_baseline"]
OBJECTIVES = ["masked_reconstruction", "conditioned_reconstruction",
              "conditioned_reconstruction_bio", "continuation", "qa"]

_TOK = AutoTokenizer.from_pretrained(BACKBONE)
_HID = AutoConfig.from_pretrained(BACKBONE).hidden_size


def make_cfg(M: int):
    c = ReprConfig()
    c.llama_model = BACKBONE
    c.d_llama = _HID
    c.pad_token_id, c.sep_token_id = resolve_special_ids(_TOK)
    c.n_flat_codes = M                        # compressor slot count (icae/ccm/ac n_slots=0 → this)
    c.graph_n_edges = max(8, min(M, 64))      # graph edges ≈ M (capacity-relative)
    return c


def build_batch(obj, *, ctx, batch_size, compress=None, predict=None):
    """Build ONE real batch for `obj`. Returns (batch, task_mode)."""
    if obj == "masked_reconstruction":
        from src.memory.data_masked_reconstruction import make_sentence_dataloader
        dl = make_sentence_dataloader(_TOK, batch_size=batch_size, src_tokenizer_name=SRC_TOK,
                                      split="val", max_len=min(ctx, 128), seed=0,
                                      pad_token_id=resolve_special_ids(_TOK)[0] or 0, num_workers=0)
    elif obj == "conditioned_reconstruction":
        from src.memory.data_conditioned_reconstruction import make_conditioned_reconstruction_dataloader
        dl = make_conditioned_reconstruction_dataloader(
            _TOK, context_len=ctx, batch_size=batch_size, n_pairs=min(64, ctx // 12),
            n_query=1, value_len=1, split="train", seed=0,
            pad_token_id=resolve_special_ids(_TOK)[0], num_workers=0)
    elif obj == "conditioned_reconstruction_bio":
        from src.memory.data_conditioned_reconstruction_bio import make_conditioned_reconstruction_bio_dataloader
        dl = make_conditioned_reconstruction_bio_dataloader(
            _TOK, context_len=ctx, batch_size=batch_size, n_pairs=16, n_query=1, n_facts=3,
            split="train", world_seed=0, stream_seed=0,
            pad_token_id=resolve_special_ids(_TOK)[0], num_workers=0)
    elif obj == "continuation":
        from src.memory.data_continuation import make_continuation_dataloader
        dl = make_continuation_dataloader(
            _TOK, batch_size=batch_size, compress_len=compress, predict_len=predict,
            split="validation", seed=0, src_tokenizer_name=SRC_TOK,
            pad_token_id=resolve_special_ids(_TOK)[0], num_workers=0)
    elif obj == "qa":
        from src.memory.data_qa import make_mixed_qa_dataloader
        cfg = make_cfg(16)
        dl = make_mixed_qa_dataloader(
            cfg, _TOK, composite_passages_path=COMPOSITE_TRAIN_P,
            composite_questions_path=COMPOSITE_TRAIN_Q, use_hotpot=False, use_narrative=False,
            split="train", chunk_size=ctx, passages_per_chunk=max(8, ctx // 64),
            weights=(1.0, 0, 0, 0, 0), num_workers=0, seed=0, batch_size=batch_size)
    else:
        raise ValueError(obj)
    batch = next(iter(dl))
    task_mode = "masked_reconstruction" if obj == "masked_reconstruction" else obj
    return to_device(batch, DEV), task_mode


def one_step(model, batch, opt=None):
    """fwd + bwd (+ opt step). Returns (loss_float, grad_finite)."""
    if opt is not None:
        opt.zero_grad(set_to_none=True)
    else:
        model.zero_grad(set_to_none=True)
    out = model.compute_loss(batch)
    loss = out["loss"]
    loss.backward()
    g_ok = True
    seen = False
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            seen = True
            if not torch.isfinite(p.grad).all():
                g_ok = False
                break
    if opt is not None:
        opt.step()
    return float(loss.detach()), (g_ok and seen)


# ───────────────────────────── Phase A: correctness ─────────────────────────────
def phase_a():
    print("\n" + "=" * 72 + "\nPHASE A — correctness (small real contexts, fwd+bwd, finite loss+grad)\n" + "=" * 72)
    small = {  # small but real contexts for fast crash-catching
        "masked_reconstruction": dict(ctx=128, batch_size=4),
        "conditioned_reconstruction": dict(ctx=256, batch_size=4),
        "conditioned_reconstruction_bio": dict(ctx=256, batch_size=4),
        "continuation": dict(ctx=256, batch_size=4, compress=192, predict=64),
        "qa": dict(ctx=512, batch_size=2),
    }
    batches = {}
    for obj, kw in small.items():
        try:
            batches[obj] = build_batch(obj, **kw)
            print(f"  built batch: {obj}")
        except Exception as e:
            print(f"  [DATA-FAIL] {obj}: {e}")
            batches[obj] = None
    results = {}
    for v in COHORT:
        cfg = make_cfg(16)
        model = ReprLearningModel(cfg, variant=v).to(DEV)
        for obj in OBJECTIVES:
            if batches.get(obj) is None:
                results[(v, obj)] = "skip(no-data)"
                continue
            batch, tmode = batches[obj]
            model.task_mode = tmode
            try:
                loss, g_ok = one_step(model, batch)
                fin = (loss == loss) and abs(loss) != float("inf")
                results[(v, obj)] = f"loss={loss:.3f} grad={'ok' if g_ok else 'BAD'}" \
                    if (fin and g_ok) else f"FAIL(loss={loss} grad={g_ok})"
            except Exception as e:
                results[(v, obj)] = "EXC: " + str(e).splitlines()[0][:60]
                traceback.print_exc()
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
    # table
    print(f"\n{'variant':<26}" + "".join(f"{o[:14]:<16}" for o in OBJECTIVES))
    npass = 0
    for v in COHORT:
        row = f"{v:<26}"
        for obj in OBJECTIVES:
            r = results[(v, obj)]
            ok = r.startswith("loss=")
            npass += ok
            row += f"{('OK' if ok else r)[:15]:<16}"
        print(row)
    total = sum(1 for k in results if not results[k].startswith("skip"))
    print(f"\nPhase A: {npass}/{total} (variant×objective) combos passed")
    return npass == total


# ───────────────────────────── Phase B: timing ─────────────────────────────
def phase_b():
    print("\n" + "=" * 72 + "\nPHASE B — timing at REAL contexts (mean fwd+bwd+opt step, eager)\n" + "=" * 72)
    # (ctx, M, batch_size) per objective — REAL training settings (train.py overrides).
    real = {
        "masked_reconstruction":          dict(ctx=128,  M=16,  batch_size=32),
        "conditioned_reconstruction":     dict(ctx=1024, M=35,  batch_size=8),
        "conditioned_reconstruction_bio": dict(ctx=1024, M=35,  batch_size=8),
        "continuation":                   dict(ctx=1024, M=35,  batch_size=8, compress=1024, predict=512),
        "qa":                             dict(ctx=8192, M=273, batch_size=1),
    }
    NWARM, NTIME = 1, 4
    summary = {}
    for obj in OBJECTIVES:
        kw = dict(real[obj]); M = kw.pop("M"); bs = kw["batch_size"]
        print(f"\n--- {obj}  (ctx={kw.get('ctx')}, M={M}, batch_size={bs}) ---")
        # build batch once (reduce BS on OOM)
        batch = tmode = None
        for try_bs in (bs, max(1, bs // 2), 1):
            try:
                kw["batch_size"] = try_bs
                batch, tmode = build_batch(obj, **kw)
                bs = try_bs
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache(); continue
                raise
        if batch is None:
            print(f"  [SKIP] could not build batch"); continue
        per_obj = {}
        for v in COHORT:
            cfg = make_cfg(M)
            try:
                model = ReprLearningModel(cfg, variant=v).to(DEV)
                model.task_mode = tmode
                opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
                for _ in range(NWARM):
                    one_step(model, batch, opt)
                torch.cuda.synchronize(); t0 = time.time()
                for _ in range(NTIME):
                    one_step(model, batch, opt)
                torch.cuda.synchronize()
                dt = (time.time() - t0) / NTIME
                per_obj[v] = dt
                print(f"  {v:<26} {dt*1000:8.1f} ms/step")
            except RuntimeError as e:
                per_obj[v] = None
                print(f"  {v:<26} {'OOM/ERR':>8}  ({str(e).splitlines()[0][:40]})")
            finally:
                del model
                import gc; gc.collect(); torch.cuda.empty_cache()
        summary[obj] = (bs, per_obj)
    # steps-for-1hr (cohort run sequentially → total = steps × Σ per-step)
    print("\n" + "=" * 72 + "\nSTEPS for ~1 HOUR (whole cohort, sequential) per objective\n" + "=" * 72)
    print(f"{'objective':<32}{'batch':>6}{'Σ step/cohort':>16}{'steps/1hr':>12}")
    for obj in OBJECTIVES:
        if obj not in summary:
            continue
        bs, per_obj = summary[obj]
        ts = [t for t in per_obj.values() if t is not None]
        if not ts:
            print(f"{obj:<32}{bs:>6}{'n/a':>16}{'n/a':>12}"); continue
        cohort_step = sum(ts)              # one step across all variants
        steps = int(3600 / cohort_step)
        print(f"{obj:<32}{bs:>6}{cohort_step:>14.2f}s{steps:>12,}")
    print("\n(eager timing; torch.compile would give headroom. 'steps/1hr' = step budget so "
          "training ALL cohort variants sequentially on that objective ≈ 1 hour.)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["a", "b", "both"], default="both")
    args = ap.parse_args()
    a_ok = True
    if args.phase in ("a", "both"):
        a_ok = phase_a()
    if args.phase in ("b", "both"):
        phase_b()
    sys.exit(0 if a_ok else 1)
