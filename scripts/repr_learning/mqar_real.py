"""MQAR gate on the REAL encoders — minimal modification, so results predict Stage-1B/2 behavior.

We take the ACTUAL encoders (graph_v6, vqvae, slot, mt, mamba — already param/capacity-matched in the
v2.1 lineage), feed them the (key,value) pair-sequence through their real streaming_write -> finalize_memory,
and read with a small head. The ONLY thing stripped is the Llama *transformer* (the host reader); Llama's
embedding *table* is kept to vectorize tokens. Nothing about the encoders is changed.

Reuses the MQAR data + read head + embedding loader from mqar.py.
  .venv/bin/python scripts/repr_learning/mqar_real.py --smoke
  .venv/bin/python scripts/repr_learning/mqar_real.py --variant graph_v6 --n-pairs 16
"""
import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
import torch.nn.functional as F

from scripts.repr_learning.mqar import MQAR, ReadHead, load_embed
from scripts.repr_learning.train_stage_a import stage_a_cfg
from src.repr_learning.encoder import (
    GraphV6BaselineEncoder, VQVAEBaselineEncoder, SlotAttentionBaselineEncoder,
    MemorizingTransformerBaselineEncoder, MambaBaselineEncoder,
)

DEV = "cuda"
ENC = {"graph_v6": GraphV6BaselineEncoder, "vqvae": VQVAEBaselineEncoder, "slot": SlotAttentionBaselineEncoder,
       "mt": MemorizingTransformerBaselineEncoder, "mamba": MambaBaselineEncoder}
VARIANTS = list(ENC)
D_ENC = {"vqvae": 1600, "mt": 1536}


GRAPH_OVERRIDES = {}   # set by CLI (--kv-split / --no-persist) to A/B graph_v6 baseline vs split
VQVAE_SPLIT = False    # set by CLI (--kv-split) to enable the vqvae DKVB split too
MAMBA_DELTA = False    # set by CLI (--kv-split) to enable the mamba DeltaNet delta-rule


def cfg_for(v):
    cfg = replace(stage_a_cfg("nc8"), graph_v6_d_updater=384, graph_v6_updater_layers=3,
                  graph_v6_read_ffn_mult=1, d_enc=D_ENC.get(v, 1408), d_mamba=1408)
    if v == "graph_v6" and GRAPH_OVERRIDES:
        cfg = replace(cfg, **GRAPH_OVERRIDES)
    if v == "vqvae" and VQVAE_SPLIT:
        cfg = replace(cfg, vqvae_kv_split=True)
    if v == "mamba" and MAMBA_DELTA:
        cfg = replace(cfg, mamba_delta_rule=True)
    return cfg


def encode_pairs(encoder, emb, keys, vals):
    """Feed the interleaved (k1,v1,k2,v2,...) pair-sequence into the REAL encoder's streaming write."""
    B, N = keys.shape
    seq = torch.stack([keys, vals], dim=2).reshape(B, 2 * N)            # [B, 2N]
    te = emb(seq).float()                                              # Llama embedding table only
    mask = torch.ones(B, 2 * N, dtype=torch.bool, device=te.device)
    st = encoder.init_streaming_state(B, te.device, te.dtype)
    st, _ = encoder.streaming_write(st, te, mask, chunk_offset=0)
    return encoder.finalize_memory(st)                                 # (mem [B,M,d_llama], aux)


def mem_kv(mem, aux):
    """(mem_keys, mem_values) for the shared read head.
    Priority: explicit per-encoder split (*_kv: graph_v6/vqvae/mamba) > mt_bank (native) > fused."""
    for k in ("graph_v6_kv", "vqvae_kv", "mamba_kv"):
        kv = aux.get(k)
        if kv is not None:
            return kv["keys"].float(), kv["values"].float()
    if aux.get("mt_bank") is not None:
        bank = aux["mt_bank"]
        return bank.get("keys", bank["values"]).float(), bank["values"].float()
    return mem.float(), mem.float()


def run(name, n_pairs=16, steps=2000, bs=64, lr=3e-4, emb=None, d=None, bf16=True, use_compile=True, dynamic=False):
    if emb is None:
        emb, d = load_embed()
    cfg = cfg_for(name)
    enc = ENC[name](cfg).to(DEV)
    enc.train()
    data, vdat = MQAR(n_pairs, bs, seed=1), MQAR(n_pairs, bs, seed=999)
    head = ReadHead(d, d, data.V).to(DEV)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warmup))

    def _encode(keys, vals):
        mem, aux = encode_pairs(enc, emb, keys, vals)
        return mem_kv(mem, aux)
    # torch.compile fuses the encoder's launch-bound norm/elementwise/copy storm (~1.69x measured).
    # dynamic=False for the fixed-shape gate (compile once); dynamic=True for var-len Stage-A.
    encode_fn = torch.compile(_encode, dynamic=dynamic) if use_compile else _encode

    def logits(keys, vals, q, shuffle=False):
        # bf16 autocast on the COMPILED encoder forward: matmuls run bf16, autocast keeps norms/
        # softmax/attention-scores fp32, model PARAMS stay fp32 (master weights). CE in fp32 at call site.
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            mk, mv = encode_fn(keys, vals)
            if shuffle:
                mk, mv = mk.roll(1, 0), mv.roll(1, 0)
            return head(emb(q).float(), mk, mv)

    def evaluate():
        enc.eval()
        head.eval()
        real = shuf = 0.0
        with torch.no_grad():
            for _ in range(8):
                keys, vals, q, tgt = vdat.batch()
                real += (logits(keys, vals, q).argmax(-1) == tgt).float().mean().item()
                shuf += (logits(keys, vals, q, shuffle=True).argmax(-1) == tgt).float().mean().item()
        enc.train()
        head.train()
        return real / 8, shuf / 8

    n_params = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    print(f"  [{name}] encoder trainable params: {n_params:,}", flush=True)
    best_r, best_s = 0.0, 1.0
    for step in range(steps):
        keys, vals, q, tgt = data.batch()
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits(keys, vals, q).float(), tgt)   # fp32 loss (autocast best practice)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 250 == 0:
            r, s = evaluate()
            if r > best_r:
                best_r, best_s = r, s
            print(f"  {name} step {step:4d} loss {loss.item():.3f} REAL {r:.3f} SHUF {s:.3f}", flush=True)
    r, s = evaluate()
    if r > best_r:
        best_r, best_s = r, s
    print(f"FINAL {name} N={n_pairs} best REAL {best_r:.3f} SHUF {best_s:.3f} (chance {1/vdat.V:.3f})", flush=True)
    return best_r, best_s


def debug_sweep():
    """Confirm the MQAR objective is correctly wired to EVERY real encoder (no training): pairs flow through
    the real streaming_write -> finalize_memory, the head reads it, gradients reach the encoder's WRITE params,
    and REAL != SHUF (the memory is actually used). This validates 'as intended' before any training."""
    emb, d = load_embed()
    data = MQAR(16, 8, seed=1)
    keys, vals, q, tgt = data.batch()
    qe = emb(q).float()
    ok_all = True
    for name in VARIANTS:
        try:
            enc = ENC[name](cfg_for(name)).to(DEV)
            enc.train()
            head = ReadHead(d, d, data.V).to(DEV)
            mem, aux = encode_pairs(enc, emb, keys, vals)
            mk, mv = mem_kv(mem, aux)
            lg = head(qe, mk, mv)
            shp_ok = lg.shape == (8, data.V) and torch.isfinite(lg).all().item()
            F.cross_entropy(lg, tgt).backward()
            ep = [(n, p) for n, p in enc.named_parameters() if p.requires_grad]
            got = [n for n, p in ep if p.grad is not None and float(p.grad.abs().sum()) > 0]
            dead = [n for n, p in ep if p.grad is None or float(p.grad.abs().sum()) == 0]
            fin = all(torch.isfinite(p.grad).all() for _, p in ep if p.grad is not None)
            with torch.no_grad():
                uses = (head(qe, mk, mv) - head(qe, mk.roll(1, 0), mv.roll(1, 0))).abs().max().item() > 1e-4
            npar = sum(p.numel() for _, p in ep)
            mshape = tuple(mem.shape) if mem.numel() > 0 else f"empty+bank{tuple(mv.shape)}"
            # PASS = wiring sound: finite logits, finite grads, encoder write IS trained (grads flow), memory used.
            ok = shp_ok and fin and len(got) > 0 and uses
            ok_all &= ok
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:9s} {npar/1e6:5.1f}M  mem {mshape}  "
                  f"grad->{len(got)}/{len(ep)} tensors  finite={shp_ok and fin}  REAL!=SHUF={uses}", flush=True)
            if dead:
                print(f"           ({len(dead)} param tensors got no grad — e.g. {dead[:4]})", flush=True)
            del enc, head
            torch.cuda.empty_cache()
        except Exception as e:
            ok_all = False
            import traceback
            print(f"  [FAIL] {name:9s} EXCEPTION {repr(e)[:180]}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()
    print(f"\n{'ALL REAL ENCODERS WIRED TO MQAR OK' if ok_all else 'SOME ENCODERS FAILED — see above'}", flush=True)


def profile():
    """Stage-1 (gate) profiler: per encoder, fwd+bwd step time + throughput vs batch size (no Llama, so
    bigger BS should amortize the encoder). Finds the fastest training config. fp32 vs bf16-autocast."""
    import time
    emb, d = load_embed()
    for name in VARIANTS:
        print(f"\n#### {name}", flush=True)
        enc = ENC[name](cfg_for(name)).to(DEV)
        enc.train()
        head = ReadHead(d, d, 256).to(DEV)
        opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=1e-4)

        def step(bs, bf16):
            data = MQAR(8, bs, seed=1)
            k, v, q, t = data.batch()
            opt.zero_grad(set_to_none=True)
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if bf16 else torch.autocast("cuda", enabled=False)
            with ctx:
                mem, aux = encode_pairs(enc, emb, k, v)
                mk, mv = mem_kv(mem, aux)
                loss = F.cross_entropy(head(emb(q).float(), mk, mv), t)
            loss.backward()
            opt.step()

        for bf16 in (False, True):
            best = (0, 0.0)
            tag = "bf16" if bf16 else "fp32"
            for bs in (16, 32, 64, 128, 256):
                try:
                    for _ in range(3):
                        step(bs, bf16)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    for _ in range(10):
                        step(bs, bf16)
                    torch.cuda.synchronize()
                    dt = (time.time() - t0) / 10
                    thr = bs / dt
                    if thr > best[1]:
                        best = (bs, thr)
                    print(f"  [{tag}] BS={bs:4d}  step {dt*1000:6.1f}ms  thr {thr:6.0f} samp/s", flush=True)
                except torch.cuda.OutOfMemoryError:
                    print(f"  [{tag}] BS={bs:4d}  OOM", flush=True)
                    torch.cuda.empty_cache()
                    break
            print(f"  [{tag}] BEST: BS={best[0]} @ {best[1]:.0f} samp/s", flush=True)
        del enc, head, opt
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="graph_v6", choices=VARIANTS)
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--debug-sweep", action="store_true")
    ap.add_argument("--profile", action="store_true", help="stage-1 gate: step time + throughput vs BS, fp32 vs bf16")
    ap.add_argument("--kv-split", action="store_true", help="graph_v6: split fact-tokens into key+value")
    ap.add_argument("--no-persist", action="store_true", help="graph_v6: ablation — key from drifting state not node_id")
    args = ap.parse_args()
    if args.kv_split:
        global VQVAE_SPLIT, MAMBA_DELTA
        GRAPH_OVERRIDES["graph_v6_kv_split"] = True
        GRAPH_OVERRIDES["graph_v6_persist_keys"] = not args.no_persist
        VQVAE_SPLIT = True   # enable the vqvae DKVB split too
        MAMBA_DELTA = True   # enable the mamba DeltaNet delta-rule too
        print(f"[mqar_real] K/V split ON (graph_v6 persist_keys={not args.no_persist}, vqvae DKVB, mamba delta)", flush=True)
    if args.profile:
        print("[mqar_real] STAGE-1 PROFILE — step time + throughput vs BS (fp32 vs bf16), per encoder", flush=True)
        profile()
        return
    if args.debug_sweep:
        print("[mqar_real] DEBUG SWEEP — confirm MQAR objective wired to all real encoders (no training)", flush=True)
        debug_sweep()
        return
    if args.smoke:
        emb, d = load_embed()
        steps = args.steps if args.steps != 2000 else 3000
        print(f"[mqar_real] PROPER SMOKE — all real encoders, N={args.n_pairs}, {steps} steps "
              f"(do they LEARN: REAL climbs above SHUF?)", flush=True)
        summary = {}
        for v in VARIANTS:
            print(f"\n--- {v} (REAL encoder) ---", flush=True)
            try:
                summary[v] = run(v, n_pairs=args.n_pairs, steps=steps, emb=emb, d=d)
            except Exception as e:
                import traceback
                summary[v] = (float("nan"), float("nan"))
                print(f"  [{v}] FAILED: {repr(e)[:200]}", flush=True)
                traceback.print_exc()
        print(f"\n========= SMOKE SUMMARY (best REAL / SHUF, chance≈{1/256:.3f}) =========", flush=True)
        for v, (r, s) in summary.items():
            verdict = "LEARNS" if r > 0.3 and s < 0.05 else ("at chance" if r < 0.05 else "partial")
            print(f"  {v:9s} REAL {r:.3f}  SHUF {s:.3f}   {verdict}", flush=True)
        return
    run(args.variant, n_pairs=args.n_pairs, steps=args.steps)


if __name__ == "__main__":
    main()
