"""biomem smoke — verify the CHUNK-PARALLEL gated-delta grid in the REAL mixed bf16 path on REAL data.

biomem v3: memory = per-example fast edges W in n_pairs STACKED synaptic column-layers, written by a
chunk-parallel gated-delta scan (fla) with a learned per-neuron threshold between layers; read by PREPEND
(M seeds propagate through W → prepend; the LM's attention addresses them; refreshed per decoder layer).
This checks, in the actual path:

  - matched param count (cohort ~6.9M) + read budget (M*d = 32*576); W is uncounted per-example state
  - all 4 mixed tasks finite: mae → masked_reconstruction; babi/cont/condrecon → generic compute_loss
  - the chunk scan + prepend + per-layer refresh run end-to-end (forward finite, mem_effrank not collapsed)
  - gradient reaches EVERY component — esp. the WRITE side (in_proj/V_w/beta/decay_proj/theta): the read
    propagates the seeds through W(write), so write-grad starvation (the wall that killed prior arms)
    would show as a starved write side here.
  - canaries: edge_absmean, decay (mean α), beta (mean write rate), surprise (mean −logp/lnV), mem_effrank
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

# component-name → prefixes. WRITE = write-side (must NOT be starved); READ = prepend-read side.
GROUPS = {
    "WRITE in_proj":   ("encoder.in_proj",),
    "WRITE V_w":       ("encoder.V_w",),
    "WRITE beta":      ("encoder.beta_",),
    "WRITE decay_proj":("encoder.decay_proj",),
    "WRITE theta":     ("encoder.theta",),
    "WRITE mem_decay": ("encoder.mem_decay_raw",),
    "READ seeds":      ("encoder.read_seeds",),
    "READ readout":    ("encoder.readout",),
    "READ out_norm":   ("encoder.out_norm",),
    "REFRESH read_in": ("encoder.read_in",),
    "REFRESH gate":    ("encoder.refresh_gate",),
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
    Wfloats = enc.n_pairs * cfg.biomem_n_cols * cfg.biomem_k * cfg.biomem_k
    print(f"\n{'='*74}\nbiomem smoke (matched mixed config, REAL bf16 path)\n{'='*74}")
    print(f"params = {tot:,} ({tot/1e6:.2f}M)   | cohort target ≈ icae 6.93M  (chunk-parallel, n_pairs={enc.n_pairs})")
    print(f"read budget = M*d = {enc.M}*{cfg.d_llama} = {enc.M * cfg.d_llama:,} floats (prepend; matched to cohort)")
    print(f"internal state W = n_pairs*C*K^2 = {enc.n_pairs}*{cfg.biomem_n_cols}*{cfg.biomem_k}^2 "
          f"= {Wfloats:,} floats/example (the MECHANISM — uncounted, like a KV cache)")

    tasks = ["mae", "babi", "continuation", "condrecon_bio"]
    print(f"\nLoading real mixed val sets {tasks}...")
    vs = make_mixed_val_sets(tasks, tok, cfg, 2, ctx_len=1024, m_slots=32,
                             mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    # ── 1+2. write canaries + finite forward (standalone write→finalize) ──
    embed = m.decoder.llama.get_input_embeddings()
    for t in tasks:
        m.task_mode = TASK_MODE[t]
        b = to_device(vs[t][0], DEV)
        with torch.no_grad():
            ctx = embed(b.context_ids)
            enc_in = m._encode_for_memory(ctx, b.context_mask)   # biomem ingests LM final hidden
            sur = m._token_surprise(enc_in, b.context_ids, b.context_mask)   # frozen-LM next-token surprise
            st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
            st, _ = enc.streaming_write(st, enc_in, b.context_mask, surprise=sur)
            mem, aux = enc.finalize_memory(st)
        print(f"\n=== {t} ({TASK_MODE[t]}) ===")
        print(f"  prepend mem shape = {tuple(mem.shape)} (M=32 prepend tokens expected) "
              f"finite={bool(torch.isfinite(mem).all())}  mem_effrank={float(aux.get('biomem_mem_effrank', 0)):.2f}/{cfg.d_llama}")
        print(f"  edges finite = {bool(torch.isfinite(enc._read_W).all())}  "
              f"edge_absmean={float(aux['biomem_edge_absmean']):.4f}  "
              f"W shape={tuple(enc._read_W.shape)}")
        print(f"  decay(mean α)={float(aux['biomem_decay']):.4f}  beta(mean)={float(aux['biomem_beta']):.4f}  "
              f"membrane(mean λ)={float(aux.get('biomem_mem_decay', 0)):.4f}  "
              f"surprise(mean −logp/lnV)={float(aux['biomem_surprise']):.4f}")
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = (m.compute_masked_reconstruction_loss(b)
                   if TASK_MODE[t] == "masked_reconstruction"
                   else m.compute_loss(b, window_size=1024))
        loss = out.get("loss_recon", out.get("loss"))
        top1 = out.get("top1_acc", out.get("top1", torch.zeros(())))
        print(f"  REAL {TASK_MODE[t]} loss={float(loss):.3f}  finite={bool(torch.isfinite(loss))}  "
              f"top1={float(top1):.3f}")

    # ── 3. gradient flow (real mae loss → backward): WRITE side must not be starved ──
    print(f"\n=== gradient flow (real mae loss → backward) ===")
    # refresh_gate now inits to 1e-3 (tiny, ~ReZero) so read_in gets gradient from step 1 at the DEFAULT
    # init (no override) — the starvation the review flagged is gone; verify it directly below.
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
