"""Debug sweep for the new arm (slotgraph) — verify intent + no math/explosion errors.

For each variant × {mae, babi}:
  - REAL data, REAL bf16 path, BOTH compute paths (mae→masked_recon, babi→generic).
  - finite loss; memory finite + bounded magnitude (no exploding |x|); intent canaries.
  - gradient flow: every trainable submodule gets FINITE, NON-zero gradient (no starvation, no
    inf/nan, no absurd magnitude); global grad norm reported.

Usage: python scripts/diagnostics/cohort/debug_sweep_new_models.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer, AutoConfig
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import resolve_special_ids
from src.memory.training import make_mixed_val_sets, to_device
from src.memory.data.mixes import TASK_MODE

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
DEV = "cuda"
EXPLODE = 1e3       # |memory| above this ⇒ flag as exploding
GRAD_HI = 1e5       # per-group grad norm above this ⇒ flag (possible explosion)


def apply_capacity(cfg, variant):
    cfg.use_llama_lora = True
    cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    if variant == "slotgraph_baseline":
        cfg.slotgraph_n_nodes = 96
        cfg.slotgraph_lora_rank = 84; cfg.slotgraph_lora_alpha = 168
        cfg.slotgraph_write_layers = 6


def build(variant):
    cfg = ReprConfig(use_llama_lora=True, batch_size=4)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"
    apply_capacity(cfg, variant)
    m = ReprLearningModel(cfg, variant=variant, llama_model=None).to(DEV)
    m.train(True)
    return m, tok, cfg


def group(name):
    # top-level trainable submodule (after "encoder."/"decoder.")
    p = name.split(".")
    if name.startswith("encoder."):
        return "enc." + p[1]
    if "lora" in name.lower():
        return "decoder_LoRA"
    return p[0]


def run_variant(variant):
    print(f"\n{'#'*74}\n# {variant}\n{'#'*74}")
    m, tok, cfg = build(variant)
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"params={tot:,} ({tot/1e6:.2f}M)  | cohort 6.91-7.01M")
    tasks = ["mae", "babi"]
    vs = make_mixed_val_sets(tasks, tok, cfg, 1, ctx_len=1024, m_slots=cfg.slotgraph_n_nodes,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)
    enc = m.encoder
    ok = True
    for t in tasks:
        m.task_mode = TASK_MODE[t]
        b = to_device(vs[t][0], DEV)
        embed = m.decoder.llama.get_input_embeddings()
        with torch.no_grad():
            ctx = embed(b.context_ids); enc_in = m._encode_for_memory(ctx, b.context_mask)
        st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
        st, _ = enc.streaming_write(st, enc_in, b.context_mask)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            mem, aux = enc.finalize_memory(st)
        finite = bool(torch.isfinite(mem).all())
        mx = float(mem.abs().max()); nrm = float(mem.float().norm(dim=-1).mean())
        flag = "" if (finite and mx < EXPLODE) else "  <<< BAD"
        if not (finite and mx < EXPLODE):
            ok = False
        print(f"\n--- {t} ({TASK_MODE[t]}) ---")
        print(f"  memory: finite={finite} |max|={mx:.2f} per-tok-norm={nrm:.2f}{flag}")
        cans = {k: float(v) for k, v in aux.items() if torch.is_tensor(v) and v.numel() == 1}
        print(f"  canaries: " + "  ".join(f"{k.replace('slotgraph_','')}={v:.3f}"
                                          for k, v in sorted(cans.items())))
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = (m.compute_masked_reconstruction_loss(b)
                   if TASK_MODE[t] == "masked_reconstruction"
                   else m.compute_loss(b, window_size=1024))
        loss = out["loss"]; recon = out.get("loss_recon", loss)
        lf = bool(torch.isfinite(loss))
        if not lf:
            ok = False
        print(f"  loss={float(recon):.3f}  total={float(loss):.3f}  finite={lf}{'' if lf else '  <<< BAD'}")

    # ── gradient flow (mae backward) ──
    print(f"\n  === gradient flow (real mae loss → backward) ===")
    m.zero_grad(set_to_none=True)
    m.task_mode = "masked_reconstruction"
    b = to_device(vs["mae"][0], DEV)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = m.compute_masked_reconstruction_loss(b)
    out["loss"].backward()
    groups = {}
    nonfinite = []
    for n, p in m.named_parameters():
        if not p.requires_grad:
            continue
        g = p.grad
        if g is None:
            groups.setdefault(group(n), [0.0, 0]); groups[group(n)][1] += 1; continue
        if not torch.isfinite(g).all():
            nonfinite.append(n)
        gn = float(g.float().norm())
        gr = groups.setdefault(group(n), [0.0, 0])
        gr[0] += gn; gr[1] += 1
    glob = sum(float(p.grad.float().norm())**2 for p in m.parameters()
               if p.requires_grad and p.grad is not None) ** 0.5
    for gname in sorted(groups):
        gn, n = groups[gname]
        flag = "  <-- STARVED" if gn < 1e-12 else ("  <-- HUGE" if gn > GRAD_HI else "")
        if gn < 1e-12 or gn > GRAD_HI:
            ok = False
        print(f"    {gname:22} grad={gn:.2e} (n={n}){flag}")
    print(f"    {'GLOBAL':22} grad={glob:.2e}")
    if nonfinite:
        ok = False
        print(f"    NON-FINITE grads in: {nonfinite[:6]}")
    print(f"\n  {variant}: {'ALL CHECKS PASS ✓' if ok else 'FAILED ✗'}")
    del m; torch.cuda.empty_cache()
    return ok


def main():
    results = {v: run_variant(v) for v in ("slotgraph_baseline",)}
    print(f"\n{'='*74}")
    for v, r in results.items():
        print(f"  {v:24} {'PASS ✓' if r else 'FAIL ✗'}")


if __name__ == "__main__":
    main()
