"""slotgraph multi-hop message-passing READ — smoke + gradient/movement probe.

The MP read makes the prepended memory a FUNCTION of the predicted topology, so the edge/role heads
should finally get real loss gradient (the plain-prepend read left them inert — see
docs/slotgraph_diagnostics.md). This script verifies, in the REAL bf16 mixed path on REAL data:

  1. forward is finite, memory is still M=32 vectors (same prepend budget as icae);
  2. the MP canaries (hops, gate, delta=1-cos(pre,post)) are sane and non-inert;
  3. per-group gradient norms — TOPOLOGY heads (role/src/dst) + MP modules (msg/update/gate) vs the
     CONTENT path + LoRA — to see the read delivers gradient where the old read starved it;
  4. A/B: mp_read ON vs OFF (same seed/data) → does the read LIFT the topology-head gradient?
  5. parameter MOVEMENT over a few Adam steps → do the topology/MP params actually move (not just
     receive a one-shot gradient)?

Usage: python scripts/diagnostics/smoke_slotgraph_mpread.py
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

# component-name → (group, prefixes). TOPO = topology-bearing; MP = message-passing read modules.
GROUPS = {
    "TOPO src_head":   ("encoder.src_head",),
    "TOPO dst_head":   ("encoder.dst_head",),
    "TOPO log_temp":   ("encoder.log_temp",),
    "MP   msg":        ("encoder.msg",),
    "MP   update":     ("encoder.update",),
    "CONT slot_init":  ("encoder.slot_init",),
    "CONT norm":       ("encoder.norm",),
    "encoder_LoRA":    ("encoder.base", "lora"),
    "decoder_LoRA":    ("decoder.llama", "lora"),
}


def build(structure=True, seed=0):
    torch.manual_seed(seed)
    cfg = ReprConfig(use_llama_lora=True, batch_size=4)
    bc = AutoConfig.from_pretrained(BACKBONE)
    cfg.llama_model = BACKBONE; cfg.d_llama = bc.hidden_size; cfg.llama_vocab_size = bc.vocab_size
    tok = AutoTokenizer.from_pretrained(BACKBONE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg.pad_token_id, cfg.sep_token_id = resolve_special_ids(tok)
    cfg.task_mode = "mixed"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.slotgraph_n_slots = 32
    cfg.slotgraph_lora_rank = 85; cfg.slotgraph_lora_alpha = 170
    cfg.slotgraph_use_structure = structure
    m = ReprLearningModel(cfg, variant="slotgraph_baseline", llama_model=None).to(DEV)
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


def backward_once(m, batch, task):
    m.zero_grad(set_to_none=True)
    m.task_mode = MIXED_TASK_MODE[task]
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = (m.compute_masked_reconstruction_loss(batch) if task == "mae"
               else m.compute_loss(batch, window_size=1024))
    out["loss"].backward()
    return float(out["loss"].detach())


def main():
    m, tok, cfg = build(structure=True)
    enc = m.encoder
    tot = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"\n{'='*74}\nslotgraph MP-READ smoke (matched mixed config, REAL bf16 path)\n{'='*74}")
    print(f"params = {tot:,} ({tot/1e6:.2f}M)   | cohort target ≈ icae (rebalance LoRA if off)")

    # ALL 4 mixed tasks (condrecon_bio + continuation matter: a concentrated-graph condrecon_bio batch
    # is what overflowed the un-normalized MP aggregation → NaN; check every task's forward is finite).
    vs = make_mixed_val_sets(["mae", "babi", "continuation", "condrecon_bio"], tok, cfg, 2,
                             ctx_len=1024, m_slots=32, mae_src_tok="meta-llama/Llama-3.2-1B",
                             babi_tasks=(1, 2, 3, 7, 8, 11, 12, 13, 14), predict_len=64)

    # ── 1+2. forward: shape, finiteness, MP canaries ──
    embed = m.decoder.llama.get_input_embeddings()
    for t in ("mae", "babi", "continuation", "condrecon_bio"):
        m.task_mode = MIXED_TASK_MODE[t]
        b = to_device(vs[t][0], DEV)
        with torch.no_grad():
            ctx = embed(b.context_ids)
            st = enc.init_streaming_state(ctx.shape[0], DEV, ctx.dtype)
            st, _ = enc.streaming_write(st, ctx, b.context_mask)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                mem, aux = enc.finalize_memory(st)
        print(f"\n=== {t} ({MIXED_TASK_MODE[t]}) ===")
        print(f"  memory shape = {tuple(mem.shape)}  (M should be 32 — same prepend budget as icae)")
        print(f"  finite = {bool(torch.isfinite(mem).all())}")
        print(f"  MP read: hops={int(aux['slotgraph_mp_hops'])}  "
              f"delta(1-cos pre→post)={float(aux['slotgraph_mp_delta']):.3f}  "
              f"({'INERT (≈0)' if float(aux['slotgraph_mp_delta'])<1e-3 else 'read uses the graph'})")
        print(f"  struct: edge_frac={float(aux['slotgraph_edge_frac']):.3f}  "
              f"invalid_edge_frac={float(aux['slotgraph_invalid_edge_frac']):.3f} (should be ≈0 w/ edges→nodes mask)  "
              f"src_ent={float(aux['slotgraph_src_entropy']):.2f}/{torch.log(torch.tensor(32.)):.2f}  "
              f"mem_effrank={float(aux['slotgraph_mem_effrank']):.2f}/{cfg.d_llama}")

    # ── 3. per-group gradient norms (real mae backward) ──
    print(f"\n=== gradient flow: MP read ON (real mae loss → backward) ===")
    loss = backward_once(m, to_device(vs["mae"][0], DEV), "mae")
    print(f"  loss={loss:.3f}")
    on = {g: grad_norm(m, pref) for g, pref in GROUPS.items()}
    for g in GROUPS:
        print(f"  {g:16} grad = {on[g]:.3e}")
    topo = sum(on[g] for g in GROUPS if g.startswith("TOPO"))
    mp = sum(on[g] for g in GROUPS if g.startswith("MP"))
    cont = on["CONT slot_init"] + on["CONT norm"] + on["encoder_LoRA"] + on["decoder_LoRA"]
    print(f"\n  TOPO total={topo:.3e}   MP total={mp:.3e}   CONTENT+LoRA={cont:.3e}")
    print(f"  topo/content ratio = {topo/max(cont,1e-12):.4f}  "
          f"({'starved' if topo < 0.01*cont else 'real signal'})")

    # ── 4. A/B vs plain-prepend read (same seed) ──
    print(f"\n=== A/B: does the MP read LIFT the endpoint-head gradient? (same seed/data) ===")
    m_off, _, _ = build(structure=False, seed=0)
    backward_once(m_off, to_device(vs["mae"][0], DEV), "mae")
    for g in ("TOPO src_head", "TOPO dst_head"):
        off = grad_norm(m_off, GROUPS[g])
        ratio = on[g] / max(off, 1e-12)
        print(f"  {g:16} OFF={off:.3e}  ON={on[g]:.3e}  ×{ratio:.1f}")

    # ── 5. parameter MOVEMENT over a few Adam steps ──
    print(f"\n=== parameter movement over 8 Adam steps (do topology/MP params move?) ===")
    m.train(True)
    init = {n: p.detach().clone() for n, p in m.named_parameters() if p.requires_grad}
    opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-4)
    order = [("mae", vs["mae"][i % len(vs["mae"])]) for i in range(4)] + \
            [("babi", vs["babi"][i % len(vs["babi"])]) for i in range(4)]
    for i, (t, b) in enumerate(order):
        l = backward_once(m, to_device(b, DEV), t)
        opt.step()
    def moved(prefixes):
        num = den = 0.0
        for n, p in m.named_parameters():
            if not p.requires_grad:
                continue
            if all((k in n.lower()) if k == "lora" else n.startswith(k) for k in prefixes):
                num += float((p.detach() - init[n]).norm()) ** 2
                den += float(init[n].norm()) ** 2
        return (num ** 0.5), (num ** 0.5) / max(den ** 0.5, 1e-12)
    for g in GROUPS:
        absmv, relmv = moved(GROUPS[g])
        print(f"  {g:16} |Δ|={absmv:.3e}   rel|Δ|={relmv:.2e}  "
              f"{'<-- did NOT move' if absmv < 1e-9 else ''}")
    print(f"\nDone. Topology heads + MP modules should show real gradient (3), a lift vs OFF (4), "
          f"and nonzero movement (5).")


if __name__ == "__main__":
    main()
