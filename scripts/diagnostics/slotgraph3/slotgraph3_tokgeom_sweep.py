"""slotgraph3 tokenization-geometry + objective-mode debug sweep (2026-07-02 batch).

Validates the concat-project/raw/boundary/bidir tokenization AND the three objective modes
BEFORE any training run is trusted. Numerical checks run in the REAL path (bf16 autocast,
ReprLearningModel.compute_loss, both MAE and QA dispatches). The companion real-data smoke
(3-step train.py per mode) is launched by the calling session, not here.

Checks:
  1  build (raw+matrix+boundary, lm4 arm): expected M, param count
  2  finalize shapes: boundary rows, memory_mask, latents stash
  3  id-binding geometry at init: matching-id inner product > non-matching
  4  MAE fwd/bwd finite + grads reach {tok_proj, type_embed, boundary, rel_key, q_route, head_node}
  5  QA fwd/bwd finite (babi-shaped batch)
  6  loss_per_example (grad) mean == loss_recon, both paths
  7  roll equivalence: encoder_only + memory_override(shuffle_roll=1) == legacy shuffle_memory
  8  bidir mask: unit truth table + live effect on the loss
  9  GradCache exactness: analytic-W accumulated grads == naive autograd of CE+coef·nce
 10  trajectory sampling: shapes, router-only logp grads, rollouts differ, GRPO step finite
 11  custom arm builds + MAE fwd/bwd finite
 12  edges legacy mode builds + fwd finite (backcompat)

Usage: .venv/bin/python scripts/diagnostics/slotgraph3/slotgraph3_tokgeom_sweep.py
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.memory.config import ReprConfig  # noqa: E402
from src.memory.model import ReprLearningModel  # noqa: E402

DEV = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"
VOCAB = 49152
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def mk_cfg(**over):
    cfg = ReprConfig()
    cfg.llama_model = BACKBONE
    cfg.d_llama = 576
    cfg.pad_token_id = 0
    cfg.use_llama_lora = True          # the mixed protocol trains decoder LoRA (real-path fidelity)
    cfg.slotgraph3_n_nodes = 16
    cfg.slotgraph3_gate_ids = True
    cfg.slotgraph3_st_leak = True
    cfg.slotgraph3_route_key = "node"
    cfg.slotgraph3_edge_state = "matrix"
    cfg.slotgraph3_read = "raw"
    cfg.slotgraph3_write = "lm"
    cfg.slotgraph3_write_layers = 4
    cfg.slotgraph3_boundary_tokens = True
    for k, v in over.items():
        setattr(cfg, k, v)
    return cfg


def mk_model(**over):
    return ReprLearningModel(mk_cfg(**over), variant="slotgraph3_baseline",
                             llama_model=None).to(DEV)


def mae_batch(B=4, T=128, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1, VOCAB, (B, T), generator=g).to(DEV)
    return types.SimpleNamespace(context_ids=ids,
                                 context_mask=torch.ones(B, T, dtype=torch.bool, device=DEV),
                                 k_slots=None)


def qa_batch(B=4, T=128, Tq=12, Ta=6, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return types.SimpleNamespace(
        context_ids=torch.randint(1, VOCAB, (B, T), generator=g).to(DEV),
        context_mask=torch.ones(B, T, dtype=torch.bool, device=DEV),
        question_ids=torch.randint(1, VOCAB, (B, Tq), generator=g).to(DEV),
        question_mask=torch.ones(B, Tq, dtype=torch.bool, device=DEV),
        answer_ids=torch.randint(1, VOCAB, (B, Ta), generator=g).to(DEV),
        answer_mask=torch.ones(B, Ta, dtype=torch.bool, device=DEV),
        answer_content_mask=torch.ones(B, Ta, dtype=torch.bool, device=DEV),
    )


def run_loss(model, batch, mode, **kw):
    model.task_mode = mode
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        return model.compute_loss(batch, window_size=1024, **kw)


def main():
    torch.manual_seed(0)

    # ── 1+2: build + shapes ─────────────────────────────────────────────────────
    print("1. build raw+matrix+boundary (lm4 arm)")
    model = mk_model()
    enc = model.encoder
    K, topk = enc.K, enc.read_topk
    exp_M = 2 + K + K * topk
    check("expected M", enc.M == exp_M, f"M={enc.M} (2 boundary + {K} nodes + {K}×{topk} edges)")
    n_par = model.n_trainable_params()
    check("param count in matched band", 5e6 < n_par < 9e6, f"{n_par/1e6:.2f}M trainable")

    print("2. finalize shapes: boundary, mask, latents")
    b = mae_batch()
    out = run_loss(model, b, "masked_reconstruction", return_memory=True)
    mem, aux = out["_memory"], out["_mem_aux"]
    check("memory shape", tuple(mem.shape) == (4, exp_M, 576), f"{tuple(mem.shape)}")
    same_start = (mem[:, 0] - mem[0, 0]).abs().max().item()
    same_end = (mem[:, -1] - mem[0, -1]).abs().max().item()
    check("boundary rows batch-constant", same_start < 1e-5 and same_end < 1e-5,
          f"start Δ{same_start:.1e} end Δ{same_end:.1e}")
    mm = aux["memory_mask"]
    check("memory_mask shape + boundary=True", tuple(mm.shape) == (4, exp_M)
          and bool(mm[:, 0].all()) and bool(mm[:, -1].all()))
    check("latents stashed", isinstance(aux.get("latents"), tuple)
          and aux["latents"][0].shape == (4, K, 576))

    # ── 3: id-binding geometry at init ─────────────────────────────────────────
    print("3. id-binding geometry (concat-project preserves id matching)")
    with torch.no_grad():
        idh = enc.id_half.float()
        zero = torch.zeros(1, 576, device=DEV)
        n_toks = torch.stack([enc._tok(zero[0:1], idh[i:i+1], idh[i:i+1], 0).squeeze(0)
                              for i in range(K)])                       # node tokens, content-free
        e_ij = enc._tok(zero[0:1], idh[3:4], idh[7:8], 2).squeeze(0)    # edge 3→7
        sims = F.cosine_similarity(e_ij.unsqueeze(0), n_toks, dim=-1)
        top2 = sims.topk(2).indices.tolist()
    check("edge(3→7) most similar to nodes {3,7}", set(top2) == {3, 7},
          f"top2={top2}, sims 3/7 = {sims[3]:.3f}/{sims[7]:.3f} vs max-other "
          f"{sims[[i for i in range(K) if i not in (3, 7)]].max():.3f}")

    # ── 4+5+6: fwd/bwd both paths, grads, per-example ──────────────────────────
    print("4. MAE fwd/bwd + grad reach")
    model.zero_grad(set_to_none=True)
    out = run_loss(model, b, "masked_reconstruction")
    out["loss"].backward()
    check("MAE loss finite", torch.isfinite(out["loss"]).item(), f"{float(out['loss']):.4f}")
    targets = {"tok_proj": enc.tok_proj.weight, "type_embed": enc.type_embed,
               "boundary": enc.boundary, "rel_key": enc.rel_key.weight,
               "q_route": enc.q_route[0].weight, "head_node": enc.head_node.weight}
    for nm, p in targets.items():
        gn = 0.0 if p.grad is None else float(p.grad.norm())
        check(f"grad reaches {nm}", gn > 0, f"|g|={gn:.2e}")
    # NOTE: mean(loss_per_example) ≠ loss_recon at B>1 when masked-token counts differ per row
    # (mean-of-means vs global token mean) — BY DESIGN; per-example mean-NLL is the InfoNCE score.
    # Exactness is checked at B=1 where the two estimators coincide.
    out1 = run_loss(model, mae_batch(B=1, T=96, seed=3), "masked_reconstruction")
    check("MAE loss_per_example == loss_recon at B=1",
          abs(float(out1["loss_per_example"][0]) - float(out1["loss_recon"])) < 2e-3,
          f"{float(out1['loss_per_example'][0]):.4f} vs {float(out1['loss_recon']):.4f}")

    print("5. QA fwd/bwd")
    qb = qa_batch()
    model.zero_grad(set_to_none=True)
    outq = run_loss(model, qb, "qa")
    outq["loss"].backward()
    check("QA loss finite + backward", torch.isfinite(outq["loss"]).item(), f"{float(outq['loss']):.4f}")
    check("QA loss_per_example.mean == loss_recon",
          abs(float(outq["loss_per_example"].mean()) - float(outq["loss_recon"])) < 2e-3,
          f"{float(outq['loss_per_example'].mean()):.4f} vs {float(outq['loss_recon']):.4f}")
    check("QA loss_per_example has grad", outq["loss_per_example"].requires_grad)

    # ── 7: roll equivalence (override path == legacy SHUF path) ────────────────
    print("7. roll equivalence: encoder_only + memory_override(roll=1) == shuffle_memory")
    model.eval()
    with torch.no_grad():
        for mode, bb in (("masked_reconstruction", b), ("qa", qb)):
            rng = torch.cuda.get_rng_state()
            enc_out = run_loss(model, bb, mode, encoder_only=True)
            ov = (enc_out["_memory"], enc_out["_mem_aux"])
            torch.cuda.set_rng_state(rng)
            l_ov = run_loss(model, bb, mode, memory_override=ov,
                            shuffle_memory=True, shuffle_roll=1)["loss_recon"]
            torch.cuda.set_rng_state(rng)
            l_legacy = run_loss(model, bb, mode, shuffle_memory=True)["loss_recon"]
            d = abs(float(l_ov) - float(l_legacy))
            check(f"roll-equivalence [{mode}]", d < 2e-3, f"Δ={d:.2e}")
    model.train()

    # ── 8: bidirectional memory mask ────────────────────────────────────────────
    print("8. bidir mask: truth table + live effect")
    attn2d = torch.ones(1, 8, dtype=torch.long, device=DEV); attn2d[0, 7] = 0   # last col invalid
    m4 = model._bidir_prepend_mask(attn2d, 3, torch.float32)[0, 0]
    ok = ((m4[0, 2] == 0)                       # memory row 0 sees LATER memory col 2 (bidir)
          and (m4[4, 5] < -1e30)                # text row 4 cannot see future text col 5 (causal)
          and (m4[4, 1] == 0)                   # text sees memory
          and (m4[6, 7] < -1e30))               # invalid col blocked
    check("mask truth table", bool(ok))
    with torch.no_grad():
        rng = torch.cuda.get_rng_state()
        l_off = float(run_loss(model, b, "masked_reconstruction")["loss_recon"])
        model.cfg.bidir_mem_attn = True
        torch.cuda.set_rng_state(rng)
        l_on = float(run_loss(model, b, "masked_reconstruction")["loss_recon"])
        model.cfg.bidir_mem_attn = False
    check("bidir mask is live (loss changes)", abs(l_on - l_off) > 1e-5,
          f"off={l_off:.4f} on={l_on:.4f}")

    # ── 9: GradCache exactness ──────────────────────────────────────────────────
    print("9. GradCache exactness (analytic-W vs naive autograd)")
    from src.memory.training import _grad_cached_objective_step  # noqa: E402
    small = mae_batch(B=3, T=64, seed=7)
    model.task_mode = "masked_reconstruction"
    model.cfg.objective_mode = "contrastive"
    model.cfg.objective_coef = 0.5
    # naive: single autograd graph over all rolls
    model.zero_grad(set_to_none=True)
    rng = torch.cuda.get_rng_state()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        e_out = model.compute_loss(small, window_size=1024, encoder_only=True)
    mem_n, aux_n = e_out["_memory"], e_out["_mem_aux"]
    rows = []
    for r in range(3):
        torch.cuda.set_rng_state(rng)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.compute_loss(small, window_size=1024, memory_override=(mem_n, aux_n),
                                   shuffle_memory=(r > 0), shuffle_roll=r)
        rows.append(o["loss_per_example"].float())
    S = torch.stack(rows, 0)

    def _std_objective(S_):
        # the STANDARDIZED objective (mirrors train._infonce_logits_weights, valid=None —
        # MAE batches have no answers): detached per-example stats, inv_temp=1
        with torch.no_grad():
            mu_ = S_.detach().mean(0, keepdim=True)
            sd_ = S_.detach().std(0, keepdim=True).clamp_min(1e-4)
        z_ = -(S_ - mu_) / sd_
        return S_[0].mean() + 0.5 * (-(torch.log_softmax(z_, dim=0)[0]).mean())

    naive = _std_objective(S)
    naive.backward()
    g_naive = enc.tok_proj.weight.grad.detach().clone()
    # GradCache path (same RNG stream start)
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    _grad_cached_objective_step(model, small, model.cfg, 1024)
    # regenerate the naive grads under the SAME seed to compare like-for-like
    g_cache = enc.tok_proj.weight.grad.detach().clone()
    model.zero_grad(set_to_none=True)
    torch.manual_seed(123); torch.cuda.manual_seed_all(123)
    rng2 = torch.cuda.get_rng_state()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        e_out2 = model.compute_loss(small, window_size=1024, encoder_only=True)
    rows2 = []
    for r in range(3):
        torch.cuda.set_rng_state(rng2)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.compute_loss(small, window_size=1024,
                                   memory_override=(e_out2["_memory"], e_out2["_mem_aux"]),
                                   shuffle_memory=(r > 0), shuffle_roll=r)
        rows2.append(o["loss_per_example"].float())
    S2 = torch.stack(rows2, 0)
    naive2 = _std_objective(S2)
    naive2.backward()
    g_naive2 = enc.tok_proj.weight.grad.detach().clone()
    cos = F.cosine_similarity(g_cache.flatten(), g_naive2.flatten(), dim=0).item()
    rel = ((g_cache - g_naive2).norm() / g_naive2.norm().clamp_min(1e-12)).item()
    check("GradCache grad == naive grad", cos > 0.995 and rel < 0.1,
          f"cos={cos:.5f} relΔ={rel:.3f} (bf16 tolerance)")
    del g_naive  # (first naive pass used an uncontrolled seed — comparison uses the seeded pair)

    # ── 10: trajectory / GRPO ───────────────────────────────────────────────────
    print("10. trajectory sampling + GRPO step")
    nl, el = aux_n["latents"]
    rollouts, route_H = enc.sample_read_expansion(nl.detach(), el.detach(), 3)
    m0, k0, lp0 = rollouts[0]
    check("rollout memory shape", tuple(m0.shape) == (3, exp_M, 576), f"{tuple(m0.shape)}")
    check("logp finite + grad-capable", torch.isfinite(lp0).all().item() and lp0.requires_grad)
    check("router entropy in (0, ln K) + grad-capable",
          0.0 < float(route_H) < math.log(16) and route_H.requires_grad, f"H={float(route_H):.3f}")
    diffs = (rollouts[0][0] - rollouts[1][0]).abs().max().item()
    check("rollouts differ (sampling live)", diffs > 1e-6, f"maxΔ={diffs:.2e}")
    # PL formula correctness: on a tiny routing (4 dests, k=2), enumerate ALL ordered pick
    # sequences under the SAME running remaining-set formula — total probability must be 1.
    # (This is what the replaced independent-softmax surrogate fails: its "probabilities"
    # over sequences do not normalize.)
    import itertools
    with torch.no_grad():
        sc_t = torch.randn(4, device=DEV)
        tot, tot_surro = 0.0, 0.0
        for seq in itertools.permutations(range(4), 2):
            remain = torch.zeros(4, device=DEV)
            lp, lp_s = 0.0, 0.0
            for j in seq:
                lp += float(sc_t[j] - torch.logsumexp(sc_t + remain, dim=0))
                lp_s += float(sc_t[j] - torch.logsumexp(sc_t, dim=0))
                remain[j] = float("-inf")
            tot += math.exp(lp); tot_surro += math.exp(lp_s)
    check("PL log-prob normalizes (Σp over sequences == 1; surrogate does NOT)",
          abs(tot - 1.0) < 1e-5 and abs(tot_surro - 1.0) > 0.05,
          f"PL Σp={tot:.6f}, surrogate Σp={tot_surro:.4f}")
    model.zero_grad(set_to_none=True)
    lp0.sum().backward()
    gq = float(enc.q_route[0].weight.grad.norm()) if enc.q_route[0].weight.grad is not None else 0.0
    gt = float(enc.tok_proj.weight.grad.norm()) if enc.tok_proj.weight.grad is not None else 0.0
    check("logp grads reach ROUTER only", gq > 0 and gt == 0.0, f"|g_route|={gq:.2e} |g_tok|={gt:.2e}")
    model.zero_grad(set_to_none=True)
    model.cfg.objective_mode = "trajectory"
    model.cfg.grpo_samples = 2
    _, loss_t, extras = _grad_cached_objective_step(model, small, model.cfg, 1024)
    check("trajectory step finite", torch.isfinite(loss_t).item(),
          f"loss={float(loss_t):.4f} nce={extras['obj_nce']:.4f} pol={extras['obj_grpo_policy']:.4f} "
          f"R={extras['obj_grpo_reward']:.4f}±{extras['obj_grpo_reward_std']:.4f}")
    model.cfg.objective_mode = "plain"

    # ── 11+12: custom arm + edges legacy backcompat ────────────────────────────
    print("11. custom arm (raw+matrix+boundary)")
    del model
    torch.cuda.empty_cache()
    mc = mk_model(slotgraph3_write="custom")
    outc = run_loss(mc, mae_batch(), "masked_reconstruction")
    outc["loss"].backward()
    check("custom arm fwd/bwd finite", torch.isfinite(outc["loss"]).item(), f"{float(outc['loss']):.4f}")
    del mc
    torch.cuda.empty_cache()

    print("12. edges legacy mode backcompat")
    ml = mk_model(slotgraph3_read="edges", slotgraph3_boundary_tokens=False)
    outl = run_loss(ml, mae_batch(), "masked_reconstruction")
    check("edges legacy fwd finite", torch.isfinite(outl["loss"]).item(),
          f"loss={float(outl['loss']):.4f} M={ml.encoder.M}")

    # ── 13: write-audit T1/T2 (2026-07-03) ─────────────────────────────────────
    print("13. write-audit repairs: routing, init noise, gate, splice scale, legacy")
    m13 = mk_model()
    e13 = m13.encoder
    with torch.no_grad():
        nl13 = (e13.node_lat_init.float().unsqueeze(0).expand(2, -1, -1)
                + 0.3 * torch.randn(2, 16, 576, device=DEV))
        e13.eval()
        A = e13._route(nl13)
    offdiag = ~torch.eye(16, dtype=torch.bool, device=DEV)
    check("softmax routing dense — no dead support (off-diagonal; self-loops masked by design)",
          bool((A[:, offdiag].reshape(2, -1) > 0).all().item())
          and abs(float(A.sum(-1).mean()) - 1.0) < 1e-4,
          f"offdiag min={float(A[:, offdiag].min()):.2e}")
    e13.route_act = "sparsemax"
    with torch.no_grad():
        A2 = e13._route(nl13)
    check("sparsemax legacy path alive", bool((A2 >= 0).all().item())
          and bool((A2 == 0).any().item()), "exact zeros present")
    e13.route_act = "softmax"
    # train-time Gumbel noise: two train-mode routings differ, eval deterministic
    e13.train()
    with torch.no_grad():
        d_tr = float((e13._route(nl13) - e13._route(nl13)).abs().max())
    e13.eval()
    with torch.no_grad():
        d_ev = float((e13._route(nl13) - e13._route(nl13)).abs().max())
    check("route noise train-only", d_tr > 1e-6 and d_ev == 0.0, f"train Δ={d_tr:.2e} eval Δ={d_ev:.2e}")
    # init noise: train-mode memories differ across forwards, eval identical
    b13 = mae_batch(B=2, T=96, seed=11)
    m13.train()
    o_a = run_loss(m13, b13, "masked_reconstruction", return_memory=True)
    o_b = run_loss(m13, b13, "masked_reconstruction", return_memory=True)
    d_mem_tr = float((o_a["_memory"] - o_b["_memory"]).abs().max())
    m13.eval()
    with torch.no_grad():
        rng13 = torch.cuda.get_rng_state()
        o_c = run_loss(m13, b13, "masked_reconstruction", return_memory=True)
        torch.cuda.set_rng_state(rng13)
        o_d = run_loss(m13, b13, "masked_reconstruction", return_memory=True)
    d_mem_ev = float((o_c["_memory"] - o_d["_memory"]).abs().max())
    check("init-noise sampling train-only", d_mem_tr > 1e-6 and d_mem_ev == 0.0,
          f"train Δ={d_mem_tr:.2e} eval Δ={d_mem_ev:.2e}")
    # per-slot gate near sigmoid(1.5)=0.82 at init (zero weight → bias-dominated)
    g0 = torch.sigmoid(torch.tensor(1.5)).item()
    check("write gate initialized OPEN", abs(float(torch.sigmoid(e13.gate_head.bias)) - g0) < 1e-4,
          f"gate={float(torch.sigmoid(e13.gate_head.bias)):.3f} (target {g0:.3f}; old throttle was 0.23)")
    del m13, o_a, o_b, o_c, o_d
    torch.cuda.empty_cache()

    print("14. full-LEGACY write backcompat (additive + sparsemax + no noise/match/bb)")
    mleg = mk_model(slotgraph3_write_update="additive", slotgraph3_route_act="sparsemax",
                    slotgraph3_init_noise=False, slotgraph3_write_norm_match=False,
                    slotgraph3_write_boundary_bidir=False, slotgraph3_edge_write="slot")
    outleg = run_loss(mleg, mae_batch(B=2, T=96, seed=5), "masked_reconstruction")
    outleg["loss"].backward()
    check("legacy write fwd/bwd finite", torch.isfinite(outleg["loss"]).item(),
          f"loss={float(outleg['loss']):.4f}")
    check("legacy beta now write-open", abs(float(torch.sigmoid(mleg.encoder.beta_node)) - 0.8176) < 1e-3,
          f"gate={float(torch.sigmoid(mleg.encoder.beta_node)):.3f}")
    del mleg
    torch.cuda.empty_cache()

    # ── 15: T3 assoc edge write (keyed delta rule) ──────────────────────────────
    print("15. T3 assoc edge write: exact/targeted/roundtrip + integration")
    m15 = mk_model()
    e15 = m15.encoder
    r = e15.r
    check("assoc is default + decay init", e15.edge_write == "assoc"
          and abs(float(torch.sigmoid(e15.decay_head.bias)) - 0.125) < 1e-4,
          f"alpha0={float(torch.sigmoid(e15.decay_head.bias)):.4f}")
    with torch.no_grad():
        M = torch.randn(1, 1, r, r, device=DEV)
        kk1 = F.normalize(torch.randn(1, 1, r, device=DEV), dim=-1)
        vv = torch.randn(1, 1, r, device=DEV)
        M2 = M + torch.einsum("bki,bkj->bkij", vv - torch.einsum("bkij,bkj->bki", M, kk1), kk1)
        e_w = float((torch.einsum("bkij,bkj->bki", M2, kk1) - vv).abs().max())
        kk2 = torch.randn(1, 1, r, device=DEV)
        kk2 = F.normalize(kk2 - (kk2 * kk1).sum(-1, keepdim=True) * kk1, dim=-1)
        e_o = float((torch.einsum("bkij,bkj->bki", M2, kk2)
                     - torch.einsum("bkij,bkj->bki", M, kk2)).abs().max())
    check("exact-write M'k == v", e_w < 1e-5, f"|err|={e_w:.2e}")
    check("targeted — orthogonal address untouched", e_o < 1e-5, f"|err|={e_o:.2e}")
    with torch.no_grad():                                    # write→read via the REAL read path
        nl15 = e15.node_lat_init.float().unsqueeze(0) + 0.5 * torch.randn(1, 16, 576, device=DEV)
        kj = F.normalize(e15.rel_key(nl15[:, 7]), dim=-1)
        vc = torch.randn(1, r, device=DEV)
        el15 = torch.einsum("bi,bj->bij", vc, kj).reshape(1, 1, 576).expand(1, 16, 576).contiguous()
        rt = float((e15._rel_input(el15[:, 3:4], nl15[:, 7:8]) - e15.rel_up(vc)).abs().max())
    check("read-path roundtrip (write addr == read addr)", rt < 1e-5, f"|err|={rt:.2e}")
    # LIVE-formula check (not the idealized g=1/α=0 primitive): iterate the ACTUAL update with the
    # init gate/decay and one-hot routing — retrieval must converge to the fixed point g/(g+α)·v,
    # and an ORTHOGONAL (unwritten) address must decay by exactly (1−α)^n.
    with torch.no_grad():
        g0 = float(torch.sigmoid(e15.gate_head.bias))
        a0 = float(torch.sigmoid(e15.decay_head.bias))
        Ml = torch.randn(r, r, device=DEV)
        k1l = F.normalize(torch.randn(r, device=DEV), dim=-1)
        k2l = torch.randn(r, device=DEV)
        k2l = F.normalize(k2l - (k2l @ k1l) * k1l, dim=-1)
        vl = torch.randn(r, device=DEV)
        perp0 = float((Ml @ k2l).norm())
        n_it = 12
        for _ in range(n_it):
            Ml = (1 - a0) * Ml + g0 * torch.outer(vl - Ml @ k1l, k1l)
        fp = g0 / (g0 + a0)
        e_fp = float((Ml @ k1l - fp * vl).norm() / (fp * vl).norm().clamp_min(1e-9))
        dec = float((Ml @ k2l).norm()) / max(perp0, 1e-9)
        e_dec = abs(dec - (1 - a0) ** n_it) / (1 - a0) ** n_it
    check("LIVE formula: retrieval → g/(g+α)·v fixed point", e_fp < 0.02,
          f"g={g0:.3f} α={a0:.3f} fp={fp:.3f}·v, relerr={e_fp:.2e}")
    check("LIVE formula: orthogonal address decays (1−α)^n", e_dec < 0.05,
          f"ratio={dec:.4f} vs {(1-a0)**n_it:.4f}")
    # integration: full fwd/bwd, grads reach the new heads, telemetry emitted
    m15.zero_grad(set_to_none=True)
    out15 = run_loss(m15, mae_batch(B=2, T=200, seed=9), "masked_reconstruction", return_memory=True)
    out15["loss"].backward()
    grv = float(e15.rel_val.weight.grad.norm()) if e15.rel_val.weight.grad is not None else 0.0
    grd = float(e15.decay_head.bias.grad.norm()) if e15.decay_head.bias.grad is not None else 0.0
    check("grads reach rel_val + decay_head", grv > 0 and grd > 0, f"|g_val|={grv:.2e} |g_decay|={grd:.2e}")
    check("assoc telemetry emitted", "slotgraph3_edge_alpha" in out15["_mem_aux"]
          and "slotgraph3_edge_keycos" in out15["_mem_aux"],
          f"alpha={out15['_mem_aux'].get('slotgraph3_edge_alpha'):.3f} "
          f"keycos={out15['_mem_aux'].get('slotgraph3_edge_keycos'):.3f}")
    check("no head_edge in assoc mode (param honesty)", not hasattr(e15, "head_edge"))
    del m15, out15
    torch.cuda.empty_cache()
    mslot = mk_model(slotgraph3_edge_write="slot")
    outs = run_loss(mslot, mae_batch(B=2, T=96, seed=4), "masked_reconstruction")
    check("legacy edge_write=slot alive", torch.isfinite(outs["loss"]).item(),
          f"loss={float(outs['loss']):.4f}")
    del mslot
    torch.cuda.empty_cache()

    # ── summary ────────────────────────────────────────────────────────────────
    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*60}\n{n_ok}/{len(results)} checks passed")
    if n_ok < len(results):
        for nm, ok, det in results:
            if not ok:
                print(f"  FAILED: {nm} — {det}")
        sys.exit(1)


if __name__ == "__main__":
    main()
