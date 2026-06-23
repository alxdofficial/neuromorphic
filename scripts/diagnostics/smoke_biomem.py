"""biomem smoke — verify the gated fast-Hebbian grid in the REAL mixed bf16 path on REAL data.

biomem is the fast-weights arm (memory = per-example fast edges W, written by a gated delta rule,
read query-conditioned via a decoder-layer hook — NO prepend). This checks, in the actual path:

  - matched param count (cohort ~6.9M) + capacity (W = 2 * C * K^2 floats / example = the read budget)
  - both compute paths finite: mae → masked_reconstruction; babi → generic compute_loss
  - the conditioned-read hook is exercised (loss goes through write → finalize(stash W) → read-fuse)
  - gradient reaches EVERY component — esp. the WRITE side (in_proj/regulator/cond/eta/leak): the
    read propagates the query through W(write), so write-grad starvation (the wall that killed prior
    arms) would show as a starved write side here. eta0>0 + read_gate=0.1 are the cold-start guards.
  - write canaries: edge_absmean, edge/state saturation, leak, eta
"""
from __future__ import annotations
import sys
from pathlib import Path
import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from transformers import AutoTokenizer, AutoConfig
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import resolve_special_ids
from scripts.train.train import make_mixed_val_sets, to_device, MIXED_TASK_MODE

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
DEV = "cuda"

# component-name → (group, prefixes). WRITE = write-side (must NOT be starved); READ = read-side.
GROUPS = {
    "WRITE in_proj":   ("encoder.in_proj",),
    "WRITE regulator": ("encoder.regulator",),
    "WRITE cond":      ("encoder.cond",),
    "WRITE eta_raw":   ("encoder.eta_raw",),
    "WRITE leak_raw":  ("encoder.leak_raw",),
    "READ query_proj": ("encoder.read_query_proj",),
    "READ fuse_proj":  ("encoder.read_fuse_proj",),
    "READ gate":       ("encoder.read_gate",),
    "READ readout":    ("encoder.readout",),
    "decoder_LoRA":    ("decoder.llama", "lora"),
}


def build():
    cfg = ReprConfig(use_llama_lora=True, batch_size=4)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    m = ReprLearningModel(cfg, variant="biomem_baseline", llama_model=None).to(DEV)
    m.train(True)
    return m, tok, cfg


def grad_norm(m, prefixes):
    tot = 0.0
    for n, p in m.named_parameters():
        if p.grad is None:
            continue
        if all((k in n.lower()) if k == "lora" else n.startswith(k) for k in prefixes):
            tot += float(p.grad.float().norm()) ** 2
    return tot ** 0.5


def main():
    m, tok, cfg = build()
    enc = m.encoder
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    Wfloats = 2 * cfg.biomem_n_cols * cfg.biomem_k * cfg.biomem_k * 0 + \
        enc.n_pairs * cfg.biomem_n_cols * cfg.biomem_k * cfg.biomem_k
    print(f"\n{'='*74}\nbiomem smoke (matched mixed config, REAL bf16 path)\n{'='*74}")
    print(f"params = {tot:,} ({tot/1e6:.2f}M)   | cohort target ≈ icae 6.93M")
    print(f"capacity: W = n_pairs*C*K^2 = {enc.n_pairs}*{cfg.biomem_n_cols}*{cfg.biomem_k}^2 "
          f"= {Wfloats:,} floats/example  (cohort read budget = 32*576 = 18,432)")

    tasks = ["mae", "babi", "continuation", "condrecon_bio"]
    print(f"\nLoading real mixed val sets {tasks}...")
    vs = make_mixed_val_sets(tasks, tok, cfg, 2, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    # ── 1+2. write canaries + finite forward (standalone write→finalize) ──
    embed = m.decoder.llama.get_input_embeddings()
    for t in tasks:
        m.task_mode = MIXED_TASK_MODE[t]
        b = to_device(vs[t][0], DEV)
        with torch.no_grad():
            ctx = embed(b.context_ids)
            enc_in = m._encode_for_memory(ctx, b.context_mask)   # biomem ingests LM final hidden
            st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
            st, _ = enc.streaming_write(st, enc_in, b.context_mask)
            mem, aux = enc.finalize_memory(st)
        print(f"\n=== {t} ({MIXED_TASK_MODE[t]}) ===")
        print(f"  prepend mem shape = {tuple(mem.shape)} (M=0 expected — read is query-conditioned)")
        print(f"  edges finite = {bool(torch.isfinite(enc._read_W).all())}  "
              f"edge_absmean={float(aux['biomem_edge_absmean']):.4f}  "
              f"edge_satfrac={float(aux['biomem_edge_satfrac']):.3f}  "
              f"state_satfrac={float(aux['biomem_state_satfrac']):.3f}")
        print(f"  leak={float(aux['biomem_leak']):.4f}  eta={float(aux['biomem_eta']):.4f}")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = (m.compute_masked_reconstruction_loss(b)
                   if MIXED_TASK_MODE[t] == "masked_reconstruction"
                   else m.compute_loss(b, window_size=1024))
        loss = out.get("loss_recon", out.get("loss"))
        top1 = out.get("top1_acc", out.get("top1", torch.zeros(())))
        print(f"  REAL {MIXED_TASK_MODE[t]} loss={float(loss):.3f}  finite={bool(torch.isfinite(loss))}  "
              f"top1={float(top1):.3f}")

    # ── 3. gradient flow (real mae loss → backward): WRITE side must not be starved ──
    print(f"\n=== gradient flow (real mae loss → backward) ===")
    m.zero_grad(set_to_none=True)
    m.task_mode = "masked_reconstruction"
    b = to_device(vs["mae"][0], DEV)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = m.compute_masked_reconstruction_loss(b)
    out["loss"].backward()
    g = {grp: grad_norm(m, pref) for grp, pref in GROUPS.items()}
    starved = []
    for grp in GROUPS:
        npar = sum(1 for n, p in m.named_parameters()
                   if all((k in n.lower()) if k == "lora" else n.startswith(k) for k in GROUPS[grp]))
        flag = ""
        if npar > 0 and g[grp] < 1e-12:
            flag = "  <-- STARVED"; starved.append(grp)
        print(f"  {grp:16} grad = {g[grp]:.3e}  (n={npar}){flag}")
    write_tot = sum(g[k] for k in GROUPS if k.startswith("WRITE"))
    read_tot = sum(g[k] for k in GROUPS if k.startswith("READ"))
    print(f"\n  WRITE total = {write_tot:.3e}   READ total = {read_tot:.3e}")
    print(f"  {'WRITE side STARVED ✗ (the wall)' if write_tot < 1e-9 else 'WRITE side gets gradient ✓ (no starvation)'}")
    print(f"\n{'ALL components received gradient ✓' if not starved else 'STARVED: ' + ', '.join(starved) + ' ✗'}")


if __name__ == "__main__":
    main()
