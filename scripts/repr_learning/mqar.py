"""MQAR gate — does each memory mechanism's K/V form support associative recall?

Stream N (key, value) single-token pairs (real Llama vocab tokens, RANDOM pairings), then query by a
key and retrieve its value. Llama's FROZEN embedding table vectorizes tokens; a small custom read head
(NOT Llama) reads the memory. Metric: retrieval accuracy, REAL vs SHUF, swept over N (capacity curve).

Each mechanism is implemented as its minimal faithful K/V primitive (the literature's MQAR style):
  identity  — literal (key_emb, value_emb) dict, M=N            POSITIVE CONTROL (must hit ~1.0)
  mt        — native (k,v) bank, ratio-1 (M=N)                  kNN reference
  graph     — un-fused edges: key-head(src) + value-head(dst), soft-routed to M nodes   [the un-fuse]
  vq        — DKVB: KEY codebook (addresses) + per-example VALUE slots written by code assignment
  mamba     — associative matrix S += phi(k) (x) v, read q.S    (linear-attention, its native read)
  slot      — canonical slot-attention -> M slots, keys==values  UN-SPLIT NEGATIVE CONTROL (no precedent)

`produces_kv` stores expose (mem_keys, mem_values) read by a SHARED QK-norm cross-attention head (co-adapted
reader, per the literature). Mamba does its own linear-attention read. Run:
  .venv/bin/python scripts/repr_learning/mqar.py --debug-sweep            # correctness, no training
  .venv/bin/python scripts/repr_learning/mqar.py --store graph --n-pairs 16
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from scripts.repr_learning.train_stage_a import stage_a_cfg

DEV = "cuda"
D_MEM = 128          # internal memory dim for the compressing mechanisms
M_SLOTS = 64         # fixed memory budget (slots) for graph/vq/slot; N>M forces compression


def load_embed():
    cfg = stage_a_cfg("nc8")
    m = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
    emb = m.get_input_embeddings().to(DEV)
    emb.weight.requires_grad_(False)
    del m
    torch.cuda.empty_cache()
    return emb, emb.weight.shape[1]


class MQAR:
    def __init__(self, n_pairs, bs, n_keys=512, n_vals=256, key_lo=10_000, val_lo=60_000, seed=0):
        self.N, self.bs, self.V = n_pairs, bs, n_vals
        self.key_pool = torch.arange(key_lo, key_lo + n_keys)
        self.val_pool = torch.arange(val_lo, val_lo + n_vals)
        self.g = torch.Generator().manual_seed(seed)

    def batch(self):
        B, N = self.bs, self.N
        ki = torch.stack([torch.randperm(len(self.key_pool), generator=self.g)[:N] for _ in range(B)])
        keys = self.key_pool[ki]
        vidx = torch.randint(self.V, (B, N), generator=self.g)
        vals = self.val_pool[vidx]
        qj = torch.randint(N, (B,), generator=self.g)
        bi = torch.arange(B)
        return keys.to(DEV), vals.to(DEV), keys[bi, qj].to(DEV), vidx[bi, qj].to(DEV)


# ─────────────────────────── shared read head ───────────────────────────
class ReadHead(nn.Module):
    """query -> QK-norm cross-attention over memory KEYS -> VALUES -> value-token logits. Co-adapted reader
    shared by all produces_kv stores. QK-norm fights Llama-embedding anisotropy."""
    def __init__(self, d_q, d_kv, V, hid=256, heads=4, layers=2):
        super().__init__()
        self.qp = nn.Linear(d_q, hid)
        self.kp = nn.Linear(d_kv, hid)
        self.vp = nn.Linear(d_kv, hid)
        self.heads = heads
        self.logit_scale = nn.Parameter(torch.tensor(2.3))
        self.blocks = nn.ModuleList([nn.ModuleDict({
            "ln": nn.LayerNorm(hid),
            "ffn": nn.Sequential(nn.Linear(hid, 2 * hid), nn.GELU(), nn.Linear(2 * hid, hid)),
            "ln2": nn.LayerNorm(hid),
        }) for _ in range(layers)])
        self.head = nn.Sequential(nn.LayerNorm(hid), nn.Linear(hid, V))

    def forward(self, query, mem_keys, mem_values):
        B, M, _ = mem_keys.shape
        H = self.heads
        q = self.qp(query).view(B, H, 1, -1)
        k = self.kp(mem_keys).view(B, M, H, -1).transpose(1, 2)
        v = self.vp(mem_values).view(B, M, H, -1).transpose(1, 2)
        scale = self.logit_scale.exp().clamp(max=100.0)
        r = None
        for blk in self.blocks:
            qn, kn = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
            att = (qn @ kn.transpose(-1, -2)) * scale
            ctx = (att.softmax(-1) @ v).transpose(1, 2).reshape(B, -1)
            r = ctx if r is None else r + ctx
            r = blk["ln"](r)
            r = blk["ln2"](r + blk["ffn"](r))
            q = r.view(B, H, 1, -1)
        return self.head(r)


# ─────────────────────────── mechanism stores ───────────────────────────
class IdentityStore(nn.Module):
    """Positive control: literal (key_emb, value_emb) dict, M=N."""
    produces_kv = True
    def __init__(self, d):
        super().__init__()
        self.d_out = d
    def forward(self, ke, ve):
        return ke, ve


class MTStore(nn.Module):
    """Memorizing-Transformer: native (k,v) bank, ratio-1 (M=N). Learned projections, kNN-style read."""
    produces_kv = True
    def __init__(self, d):
        super().__init__()
        self.kp = nn.Linear(d, D_MEM)
        self.vp = nn.Linear(d, D_MEM)
        self.d_out = D_MEM
    def forward(self, ke, ve):
        return self.kp(ke), self.vp(ve)


class GraphStore(nn.Module):
    """Un-fused edges: key-head(src=key) + value-head(dst=value), soft-routed to M node slots by the key."""
    produces_kv = True
    def __init__(self, d, M=M_SLOTS, dm=D_MEM, persist_keys=True):
        super().__init__()
        self.kh = nn.Sequential(nn.Linear(d, dm), nn.GELU(), nn.Linear(dm, dm))
        self.vh = nn.Sequential(nn.Linear(d, dm), nn.GELU(), nn.Linear(dm, dm))
        self.node_q = nn.Parameter(torch.randn(M, dm))                     # persistent node addresses (the codebook)
        self.route_scale = nn.Parameter(torch.tensor(2.3))                 # sharp cosine-routing temp (ln 10)
        self.persist_keys = persist_keys
        self.M, self.d_out = M, dm
    def forward(self, ke, ve):
        kk, vv = self.kh(ke), self.vh(ve)                                  # [B,N,dm]
        kn = F.normalize(kk, dim=-1)
        qn = F.normalize(self.node_q, dim=-1)
        route = torch.softmax((kn @ qn.t()) * self.route_scale.exp().clamp(max=100.0), dim=-1)  # [B,N,M]
        w = route.transpose(1, 2)                                          # [B,M,N]
        denom = w.sum(-1, keepdim=True).clamp_min(1e-6)
        val_slots = (w @ vv) / denom                                       # [B,M,dm] scatter-mean values
        if self.persist_keys:
            # PERSISTENT addresses (DKVB/VQ form): keys = the query-independent node codebook. Scatter-mean keys
            # DRIFT to the global mean under overload -> routing collapses to one attractor -> cliff (modern-Hopfield).
            key_slots = self.node_q.unsqueeze(0).expand(ke.shape[0], -1, -1)
        else:
            key_slots = (w @ kk) / denom                                   # drifting addresses (cliffs) — for the A/B
        return key_slots, val_slots


class VQStore(nn.Module):
    """DKVB: a KEY codebook (addresses; frozen by default) + per-example VALUE slots written by soft code
    assignment. Read: query -> nearest key-code -> its written value slot."""
    produces_kv = True
    def __init__(self, d, M=M_SLOTS, dm=D_MEM, freeze_keys=False):
        super().__init__()
        self.kp = nn.Linear(d, dm)
        self.vh = nn.Linear(d, dm)
        # codebook = the M KEY addresses. Learnable (decoupling/freezing is the deferred ablation knob) + sharp
        # COSINE assignment (parity with GraphStore) — a frozen random codebook leaves most codes unused -> collisions.
        self.codebook = nn.Parameter(torch.randn(M, dm), requires_grad=not freeze_keys)
        self.assign_scale = nn.Parameter(torch.tensor(2.3))                # learnable assignment temp (ln 10)
        self.M, self.d_out = M, dm
    def forward(self, ke, ve):
        kk, vv = self.kp(ke), self.vh(ve)
        cn = F.normalize(self.codebook, dim=-1)
        assign = torch.softmax(F.normalize(kk, dim=-1) @ cn.t() * self.assign_scale.exp().clamp(max=100.0), dim=-1)
        w = assign.transpose(1, 2)
        denom = w.sum(-1, keepdim=True).clamp_min(1e-6)
        val_slots = (w @ vv) / denom                                       # [B,M,dm] values at codes
        keys = self.codebook.unsqueeze(0).expand(ke.shape[0], -1, -1)      # [B,M,dm] addresses
        return keys, val_slots


class SlotStore(nn.Module):
    """UN-SPLIT negative control: canonical slot-attention over the interleaved pair tokens -> M slots;
    keys == values (the read head must address the raw, un-split slots). No KV precedent — expected to lag."""
    produces_kv = True
    def __init__(self, d, M=M_SLOTS, dm=D_MEM, iters=3):
        super().__init__()
        self.proj = nn.Linear(d, dm)
        self.mu = nn.Parameter(torch.randn(1, 1, dm) * 0.02)
        self.logsig = nn.Parameter(torch.zeros(1, 1, dm))
        self.to_q = nn.Linear(dm, dm)
        self.to_k = nn.Linear(dm, dm)
        self.to_v = nn.Linear(dm, dm)
        self.gru = nn.GRUCell(dm, dm)
        self.mlp = nn.Sequential(nn.LayerNorm(dm), nn.Linear(dm, dm), nn.GELU(), nn.Linear(dm, dm))
        self.ln_in = nn.LayerNorm(dm)
        self.ln_s = nn.LayerNorm(dm)
        self.M, self.iters, self.d_out = M, iters, dm
    def forward(self, ke, ve):
        B, N, _ = ke.shape
        seq = torch.stack([ke, ve], dim=2).reshape(B, 2 * N, ke.shape[-1])  # interleave [k1,v1,k2,v2,...]
        x = self.ln_in(self.proj(seq))
        slots = self.mu + self.logsig.exp() * torch.randn(B, self.M, x.shape[-1], device=x.device)
        k, v = self.to_k(x), self.to_v(x)
        for _ in range(self.iters):
            q = self.to_q(self.ln_s(slots))
            att = torch.softmax(q @ k.transpose(1, 2) / (k.shape[-1] ** 0.5), dim=1)   # softmax over SLOTS
            att = att / att.sum(-1, keepdim=True).clamp_min(1e-6)
            upd = att @ v
            slots = self.gru(upd.reshape(-1, upd.shape[-1]), slots.reshape(-1, slots.shape[-1])).view(B, self.M, -1)
            slots = slots + self.mlp(slots)
        return slots, slots                                               # un-split: key == value


class MambaStore(nn.Module):
    """Mamba/SSM associative memory with the DELTA RULE (DeltaNet, Schlag 2102.11174 / Yang 2406.06484):
    S <- S + beta*(v - S phi(k)) phi(k)^T removes the old binding before writing, so keys can be REWRITTEN
    instead of colliding (the additive S+=phi(k)(x)v stalls once #pairs approaches state rank). Own linear read."""
    produces_kv = False
    def __init__(self, d, V, dk=D_MEM, dv=D_MEM, read_hid=768):
        super().__init__()
        self.kp = nn.Linear(d, dk)
        self.qp = nn.Linear(d, dk)
        self.vp = nn.Linear(d, dv)
        self.beta = nn.Linear(d, 1)                                       # data-dependent write strength (DeltaNet)
        # decode head param-matched (~1.7M total) to the cross-attn read head the other stores get — mamba's
        # q.S associative read is parameter-free, so fairness needs comparable DECODE capacity (matrix unchanged).
        self.head = nn.Sequential(
            nn.LayerNorm(dv),
            nn.Linear(dv, read_hid), nn.GELU(),
            nn.Linear(read_hid, read_hid), nn.GELU(),
            nn.Linear(read_hid, V),
        )
    @staticmethod
    def _feat(x):
        return F.elu(x) + 1
    def forward(self, ke, ve, qe, shuffle=False):
        k = F.normalize(self._feat(self.kp(ke)), dim=-1)                 # [B,N,dk] unit keys -> stable delta rule
        v = self.vp(ve)                                                  # [B,N,dv]
        beta = torch.sigmoid(self.beta(ke)).squeeze(-1)                  # [B,N] write gate
        B, N, _ = k.shape
        S = k.new_zeros(B, v.shape[-1], k.shape[-1])                     # [B,dv,dk]
        for n in range(N):                                              # small N; sequential delta scan
            kn, vn = k[:, n], v[:, n]
            pred = torch.einsum("bvk,bk->bv", S, kn)                     # current read of kn
            S = S + beta[:, n, None, None] * torch.einsum("bv,bk->bvk", vn - pred, kn)
        if shuffle:
            S = S.roll(1, 0)
        q = F.normalize(self._feat(self.qp(qe)), dim=-1)                # [B,dk]
        return self.head(torch.einsum("bvk,bk->bv", S, q))              # [B,V]


BUDGET = 16384   # matched per-example memory floats (honest bottleneck); M derived per mechanism, not magic.


def build_store(name, d, V):
    if name == "identity":
        return IdentityStore(d)                            # unbounded ceiling (2*N*d_llama)
    if name == "mt":
        return MTStore(d)                                  # unbounded ratio-1 ref (2*N*D_MEM)
    if name == "graph":
        return GraphStore(d, M=BUDGET // (2 * D_MEM))      # persistent keys (graceful) -> 64 slots
    if name == "graph_drift":
        return GraphStore(d, M=BUDGET // (2 * D_MEM), persist_keys=False)  # scatter-mean keys (cliffs) — A/B only
    if name == "vq":
        return VQStore(d, M=BUDGET // D_MEM)               # values only (codebook amortized) -> 128 codes
    if name == "slot":
        return SlotStore(d, M=BUDGET // D_MEM)             # slots reused as K=V -> 128 slots
    if name == "mamba":
        return MambaStore(d, V)                            # dk*dv matrix = 128*128 = 16384
    raise ValueError(name)


STORES = ["identity", "mt", "graph", "vq", "slot", "mamba"]


def _logits(store, head, ke, ve, qe, shuffle=False):
    if store.produces_kv:
        mk, mv = store(ke, ve)
        if shuffle:
            mk, mv = mk.roll(1, 0), mv.roll(1, 0)
        return head(qe, mk, mv)
    return store(ke, ve, qe, shuffle=shuffle)


def run(name, n_pairs=16, steps=2000, bs=64, lr=3e-4, emb=None, d=None):
    if emb is None:
        emb, d = load_embed()
    data, vdat = MQAR(n_pairs, bs, seed=1), MQAR(n_pairs, bs, seed=999)
    store = build_store(name, d, data.V).to(DEV)
    head = ReadHead(d, store.d_out, data.V).to(DEV) if store.produces_kv else None
    params = list(store.parameters()) + (list(head.parameters()) if head else [])
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    # LR warmup (Slot Attention REQUIRES it — Locatello 2020) + grad clip; standard for ALL stores, fair to all.
    warmup = max(1, steps // 20)                                            # 5% of steps (scales with budget)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    def evaluate():
        store.eval()
        if head:
            head.eval()
        real = shuf = 0.0
        with torch.no_grad():
            for _ in range(8):
                keys, vals, q, tgt = vdat.batch()
                ke, ve, qe = emb(keys).float(), emb(vals).float(), emb(q).float()
                real += (_logits(store, head, ke, ve, qe).argmax(-1) == tgt).float().mean().item()
                shuf += (_logits(store, head, ke, ve, qe, shuffle=True).argmax(-1) == tgt).float().mean().item()
        store.train()
        if head:
            head.train()
        return real / 8, shuf / 8

    best_r, best_s = 0.0, 1.0                                              # fair best-checkpoint metric
    for step in range(steps):
        keys, vals, q, tgt = data.batch()
        ke, ve, qe = emb(keys).float(), emb(vals).float(), emb(q).float()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(_logits(store, head, ke, ve, qe), tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)                         # standard (matches encoder trainer)
        opt.step()
        sched.step()
        if step % 250 == 0:
            r, s = evaluate()
            if r > best_r:
                best_r, best_s = r, s
            print(f"  step {step:4d}  loss {loss.item():.3f}  REAL {r:.3f}  SHUF {s:.3f}  (chance {1/data.V:.3f})",
                  flush=True)
    r, s = evaluate()
    if r > best_r:
        best_r, best_s = r, s
    print(f"FINAL  {name}  N={n_pairs}  best REAL {best_r:.3f} SHUF {best_s:.3f}  (final {r:.3f}/{s:.3f}, "
          f"chance {1/data.V:.3f})", flush=True)
    return best_r, best_s


def debug_sweep():
    """Correctness check on every store's K/V form — NO training. forward+backward finite, grads flow,
    shapes hold, and REAL != SHUF at init (memory is actually used)."""
    emb, d = load_embed()
    data = MQAR(16, 8, seed=1)
    keys, vals, q, tgt = data.batch()
    ke, ve, qe = emb(keys).float(), emb(vals).float(), emb(q).float()
    ok_all = True
    for name in STORES:
        store = build_store(name, d, data.V).to(DEV)
        head = ReadHead(d, store.d_out, data.V).to(DEV) if store.produces_kv else None
        params = list(store.parameters()) + (list(head.parameters()) if head else [])
        try:
            lg = _logits(store, head, ke, ve, qe)
            shp_ok = lg.shape == (8, data.V) and torch.isfinite(lg).all().item()
            F.cross_entropy(lg, tgt).backward()
            dead = [n for n, p in store.named_parameters() if p.requires_grad and (p.grad is None or float(p.grad.abs().sum()) == 0)]
            fin = all(torch.isfinite(p.grad).all() for _, p in store.named_parameters() if p.requires_grad and p.grad is not None)
            with torch.no_grad():
                lr_ = _logits(store, head, ke, ve, qe)
                ls_ = _logits(store, head, ke, ve, qe, shuffle=True)
            uses_mem = (lr_ - ls_).abs().max().item() > 1e-4
            mshape = "" if not store.produces_kv else f" memKV [B,{store(ke,ve)[0].shape[1]},{store.d_out}]"
            ok = shp_ok and not dead and fin and uses_mem
            ok_all &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:9s} logits{tuple(lg.shape)} finite={shp_ok} "
                  f"dead={len(dead)} gradfin={fin} REAL!=SHUF={uses_mem}{mshape}", flush=True)
            if dead:
                print(f"           dead params: {dead[:6]}", flush=True)
        except Exception as e:
            ok_all = False
            import traceback
            print(f"  [FAIL] {name:9s} EXCEPTION {repr(e)[:160]}", flush=True)
            traceback.print_exc()
        del store, head
        torch.cuda.empty_cache()
    print(f"\n{'ALL STORES PASS' if ok_all else 'SOME STORES FAILED'}", flush=True)


def smoke(steps=600, n_pairs=16):
    """Short end-to-end train of every store at N<=M (no compression) — confirms each mechanism LEARNS the
    binding (REAL climbs over SHUF), not just runs. Pre-flight before the full capacity sweep."""
    emb, d = load_embed()
    print(f"[smoke] N={n_pairs} (<= M={M_SLOTS}, no compression), {steps} steps/store", flush=True)
    summary = {}
    for name in STORES:
        print(f"\n--- {name} ---", flush=True)
        r, s = run(name, n_pairs=n_pairs, steps=steps, emb=emb, d=d)
        summary[name] = (r, s)
    print("\n================ SMOKE SUMMARY (REAL / SHUF, chance≈0.004) ================", flush=True)
    for name, (r, s) in summary.items():
        print(f"  {name:9s} REAL {r:.3f}  SHUF {s:.3f}  {'OK' if r > 0.5 and s < 0.05 else 'CHECK'}", flush=True)


def sweep(steps=2000, n_list=(8, 16, 32, 64, 96, 128, 256), stores=None):
    """The capacity curve: every store across N, fixed 16k-float budget, best-checkpoint REAL vs SHUF.
    The N>M (=64/128) tail is where the fixed-footprint mechanisms separate from the unbounded ceilings."""
    stores = stores or STORES
    emb, d = load_embed()
    grid = {}
    for name in stores:
        for N in n_list:
            print(f"\n=== {name}  N={N} ===", flush=True)
            grid[(name, N)] = run(name, n_pairs=N, steps=steps, emb=emb, d=d)
    print("\n\n================ MQAR CAPACITY — best REAL (SHUF) — chance≈0.004 ================", flush=True)
    print("store        " + "".join(f"N={n:<9}" for n in n_list), flush=True)
    for name in stores:
        row = f"{name:12s} "
        for n in n_list:
            r, s = grid[(name, n)]
            row += f"{r:.2f}({s:.2f})".ljust(11)
        print(row, flush=True)


def audit():
    """Fairness audit: TRAINABLE params (write vs read) + per-example memory floats, per store. The honest
    bottleneck has TWO axes — memory footprint (matched 16k) AND model/learning capacity (params)."""
    emb, d = load_embed()
    V = 256

    def mem_floats(name):
        if name == "identity":
            return "2·N·2048  (grows w/ N)"
        if name == "mt":
            return "2·N·128   (grows w/ N)"
        if name == "graph":
            return f"{2 * (BUDGET // (2 * D_MEM)) * D_MEM:,} (keys+values, M=64)"
        if name in ("vq", "slot"):
            return f"{(BUDGET // D_MEM) * D_MEM:,} (M={BUDGET // D_MEM})"
        return f"{D_MEM * D_MEM:,} (matrix 128x128)"

    print(f"{'store':10s} {'write/own':>11s} {'read_head':>10s} {'TOTAL trainable':>16s}   per-example mem floats", flush=True)
    for name in STORES:
        store = build_store(name, d, V)
        wp = sum(p.numel() for p in store.parameters() if p.requires_grad)
        rp = sum(p.numel() for p in ReadHead(d, store.d_out, V).parameters()) if store.produces_kv else 0
        note = "  (read inside 'write/own')" if not store.produces_kv else ""
        print(f"{name:10s} {wp:>11,d} {rp:>10,d} {wp + rp:>16,d}   {mem_floats(name)}{note}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="identity", choices=STORES)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--debug-sweep", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--graph-ab", action="store_true")
    args = ap.parse_args()
    if args.audit:
        print("[mqar] FAIRNESS AUDIT — trainable params + memory floats", flush=True)
        audit()
        return
    if args.graph_ab:
        print("[mqar] GRAPH A/B — drift (scatter-mean keys) vs persist (codebook keys) vs vq, across N", flush=True)
        sweep(n_list=(32, 64, 96, 128, 256), stores=["graph_drift", "graph", "vq"])
        return
    if args.sweep:
        print("[mqar] CAPACITY SWEEP — all stores x N, matched 16k budget, single-query", flush=True)
        sweep(steps=args.steps if args.steps != 2000 else 2000)
        return
    if args.debug_sweep:
        print("[mqar] DEBUG SWEEP — correctness of all K/V stores, no training", flush=True)
        debug_sweep()
        return
    if args.smoke:
        smoke(steps=args.steps if args.steps != 2000 else 600, n_pairs=args.n_pairs)
        return
    print(f"[mqar] store={args.store} N={args.n_pairs}", flush=True)
    run(args.store, n_pairs=args.n_pairs, steps=args.steps)


if __name__ == "__main__":
    main()
