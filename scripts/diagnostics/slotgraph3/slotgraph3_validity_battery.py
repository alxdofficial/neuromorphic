"""slotgraph3 validity battery — dynamic numerical checks for the post-K=16 machinery.

Run before trusting results from new configs. Checks (GPU, small-B, safe alongside training):
  1. GLOBAL ST forward exactness      output tokens == hard gather of top-E (leak backward-only)
  2. chunk-checkpoint grad equivalence grads identical with/without checkpointing (the OOM fix)
  3. floor selection sanity           flat_i has no duplicates; all K sources covered
  4. matrix _rel_input consistency    same (src,dst) pair → same rel input via _token_block vs chunk path
  5. full-combo custom arm            K=128 × global × matrix × no-write-expand fwd/bwd finite (never co-run)
  6. save→load round trip             state_dict reload reproduces the memory bit-for-bit
  7. idle-row freeze at K=128         inactive batch rows keep their latents

Usage: python scripts/diagnostics/slotgraph3/slotgraph3_validity_battery.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.memory.config import ReprConfig
from src.memory.models.slotgraph3 import SlotGraph3Encoder

DEV = "cuda"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


def build(write_mode="lm", write_expand=False, edge_state="matrix"):
    cfg = ReprConfig()
    cfg.llama_model = "HuggingFaceTB/SmolLM2-135M"; cfg.d_llama = 576
    cfg.slotgraph3_n_nodes = 128; cfg.slotgraph3_d_key = 128
    cfg.slotgraph3_gate_ids = True; cfg.slotgraph3_st_leak = True
    cfg.slotgraph3_edge_budget = 384; cfg.slotgraph3_route_key = "node"
    cfg.slotgraph3_edge_state = edge_state
    cfg.slotgraph3_write_expand = write_expand
    cfg.slotgraph3_write = write_mode
    cfg.slotgraph3_read = "edges"    # the battery probes the LEGACY expansion machinery (φ, node_id,
                                     # id_scale, role) — those exist only in edges mode since the
                                     # 2026-07-03 conditional-param cleanup (raw default lacks them)
    return SlotGraph3Encoder(cfg).to(DEV)


def encode(enc, emb, mask):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        st = enc.init_streaming_state(emb.shape[0], emb.device, emb.dtype)
        st, _ = enc.streaming_write(st, emb, mask)
        return enc.finalize_memory(st)


def main():
    torch.manual_seed(0)
    enc = build()
    B, K, d, E = 2, 128, 576, 384
    nl = (enc.node_lat_init.float().unsqueeze(0).expand(B, -1, -1) + 0.3 * torch.randn(B, K, d, device=DEV)).detach()
    el = (enc.edge_lat_init.float().unsqueeze(0).expand(B, -1, -1) + 0.3 * torch.randn(B, K, d, device=DEV)).detach()
    nid = torch.nn.functional.normalize(enc.node_id.float(), dim=-1).detach()
    role, id_scale = enc.role.float().detach(), enc.id_scale.float().detach()

    print("1. global ST forward exactness")
    with torch.no_grad():
        A = enc._route(nl)
        out, keep, _, topv = enc._expand_global(nl, el, nid, id_scale, role, A)
        flat = A.reshape(B, K * K)
        f1 = A.argmax(-1)
        floor_i = torch.arange(K, device=DEV).unsqueeze(0) * K + f1
        rest_i = flat.scatter(1, floor_i, -1.0).topk(E - K, dim=-1).indices
        flat_i = torch.cat([floor_i, rest_i], 1)
        tv = flat.gather(1, flat_i)
        H = enc._token_block(nl, el, nid, id_scale, role, tv, flat_i // K, flat_i % K)
        diff = (out - H).abs().max().item()
    check("forward == hard gather", diff < 1e-5, f"max diff {diff:.2e}")

    print("2. chunk-checkpoint gradient equivalence")
    # routing computed ONCE in eval mode and shared by both runs: train mode now adds Gumbel
    # exploration noise to _route (2026-07-03), so recomputing A per-call would compare grads
    # through two DIFFERENT wirings — the test isolates checkpointing, so A must be held fixed.
    enc.eval()
    with torch.no_grad():
        A_fixed = enc._route(nl)

    def grads(train_mode):
        enc.zero_grad(set_to_none=True)
        enc.train(train_mode)                                   # checkpointing active only in train
        el2 = el.clone().requires_grad_(True)
        A2 = A_fixed.clone()
        out, *_ = enc._expand_global(nl, el2, nid, id_scale, role, A2)
        torch.manual_seed(7)
        (out.float() * torch.randn_like(out).float()).sum().backward()
        return el2.grad.clone(), enc.phi[0].weight.grad.clone()
    g_ck, gp_ck = grads(True)
    g_no, gp_no = grads(False)
    rel = (g_ck - g_no).norm() / g_no.norm().clamp_min(1e-12)
    relp = (gp_ck - gp_no).norm() / gp_no.norm().clamp_min(1e-12)
    check("ckpt grads == no-ckpt grads", rel < 1e-3 and relp < 1e-3, f"rel diff el {rel:.2e}, phi {relp:.2e}")
    enc.train(False)

    print("3. floor selection sanity")
    dup = max(int(flat_i[b].unique().numel()) for b in range(B))
    cov = min(int((flat_i[b] // K).unique().numel()) for b in range(B))
    check("no duplicate edges", dup == E, f"unique {dup}/{E}")
    check("all sources covered", cov == K, f"coverage {cov}/{K}")

    print("4. matrix _rel_input cross-site consistency")
    with torch.no_grad():
        src_sel, dst_idx = flat_i // K, flat_i % K
        er_g = torch.gather(el, 1, src_sel.unsqueeze(-1).expand(B, E, d))
        dst_g = torch.gather(nl, 1, dst_idx.unsqueeze(-1).expand(B, E, d))
        rel_site1 = enc._rel_input(er_g, dst_g)                          # _token_block path
        c0, c1 = 0, 32                                                   # chunk path for sources 0..31
        er_c = el[:, c0:c1].unsqueeze(2).expand(B, 32, K, d)
        dst_c = nl.unsqueeze(1).expand(B, 32, K, d)
        rel_site2 = enc._rel_input(er_c, dst_c)                          # chunk_fn path
        m = src_sel < 32
        errs = []
        for b in range(B):
            idx = m[b].nonzero(as_tuple=True)[0][:50]
            for i in idx:
                s, t = int(src_sel[b, i]), int(dst_idx[b, i])
                errs.append((rel_site1[b, i] - rel_site2[b, s, t]).abs().max().item())
    check("same pair → same rel across sites", max(errs) < 1e-4, f"max {max(errs):.2e} over {len(errs)} pairs")

    print("5. full-combo custom arm (K=128 × global × matrix × no-write-expand)")
    del enc; torch.cuda.empty_cache()
    enc_c = build(write_mode="custom")
    emb = torch.randn(B, 600, d, device=DEV, dtype=torch.bfloat16) * 0.5
    mask = torch.ones(B, 600, device=DEV, dtype=torch.long)
    enc_c.train(True)
    mem, aux = encode(enc_c, emb, mask)
    (mem.float() * torch.randn_like(mem).float()).sum().backward()
    g = lambda p: 0.0 if p.grad is None else p.grad.norm().item()
    ok = torch.isfinite(mem).all().item() and mem.shape == (B, enc_c.M, d) \
        and g(enc_c.rel_key.weight) > 0 and g(enc_c.blocks[0].o.weight) > 0   # M includes boundary rows
    check("custom full-combo fwd/bwd", ok,
          f"finite mem{tuple(mem.shape)}, rel_key.g={g(enc_c.rel_key.weight):.3f}, blk0.o.g={g(enc_c.blocks[0].o.weight):.3f}")

    print("6. save→load round trip (custom full-combo)")
    sd = {k: v for k, v in enc_c.state_dict().items() if not k.startswith("base.")}
    enc_c2 = build(write_mode="custom")
    missing = enc_c2.load_state_dict(sd, strict=False)
    bad = [k for k in missing.missing_keys if not k.startswith("base.")]
    enc_c.train(False); enc_c2.train(False)
    with torch.no_grad():
        m1, _ = encode(enc_c, emb, mask)
        m2, _ = encode(enc_c2, emb, mask)
    check("reload reproduces memory", bool((m1 == m2).all()) and not bad,
          f"missing-nonbase={bad[:3]}, equal={bool((m1 == m2).all())}")

    print("7. idle-row freeze at K=128")
    mask2 = mask.clone(); mask2[1, 300:] = 0    # row 1: w0 active, w1 partial (tok 256-299), w2 fully idle
    enc_c.train(True)
    enc_c._trace = []
    mem3, _ = encode(enc_c, emb, mask2)
    tr = enc_c._trace; enc_c._trace = None
    frozen = bool(torch.equal(tr[2][1][1], tr[1][1][1])) if len(tr) > 2 else False   # w2 is the idle window
    moved_w1 = not bool(torch.equal(tr[1][1][1], tr[0][1][1]))                       # w1 partial → SHOULD move
    check("idle row keeps latents (w2) & partial row updates (w1)",
          frozen and moved_w1 and torch.isfinite(mem3).all().item(),
          f"windows={len(tr)}, w2 frozen={frozen}, w1 moved={moved_w1}")

    print(f"\n{'='*70}\nSUMMARY: {sum(1 for _, ok, _ in results if ok)}/{len(results)} PASS")
    for n, ok, det in results:
        if not ok:
            print(f"  FAIL: {n} — {det}")


if __name__ == "__main__":
    main()
