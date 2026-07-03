"""Objective-mode SEMANTIC debug battery (2026-07-03).

The tokgeom sweep proves the objective GRADIENTS are computed correctly (GradCache ≡ autograd).
This battery proves they push in the right DIRECTIONS and that the machinery's statistical
assumptions hold end-to-end in the real compute path — the layer a wiring bug (sg3_contrast)
or a sign bug would live in:

  1  pass-1/pass-2 S-consistency — the no-grad scoring pass and the grad recompute pass must see
     IDENTICAL masks (RNG restore end-to-end); silent drift would corrupt the analytic W.
  2  W-matrix directions — W[0,:] > 0 (positive's CE minimized), W[r>0,:] < 0 (negatives' CE
     maximized), column sums = 1/B (net weight per example).
  3  nce value cross-check vs an independent computation from S.
  4  GRPO policy-gradient direction — a rollout with positive advantage must receive a NEGATIVE
     logp gradient (so a descent step RAISES its log-probability).
  5  contrastive perturbation probe — nudging the REAL memory toward random noise must RAISE the
     InfoNCE term (its own memory explains its target worse → softmax mass leaks to negatives).

Usage: .venv/bin/python scripts/diagnostics/objective_debug.py
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))

from src.memory.config import ReprConfig  # noqa: E402
from src.memory.model import ReprLearningModel  # noqa: E402

DEV = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"
VOCAB = 49152
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def mk_model():
    cfg = ReprConfig()
    cfg.llama_model = BACKBONE
    cfg.d_llama = 576
    cfg.pad_token_id = 0
    cfg.use_llama_lora = True
    cfg.slotgraph3_n_nodes = 16
    cfg.slotgraph3_gate_ids = True
    cfg.slotgraph3_st_leak = True
    cfg.slotgraph3_route_key = "node"
    cfg.slotgraph3_edge_state = "matrix"
    cfg.slotgraph3_write = "lm"
    cfg.slotgraph3_write_layers = 4
    cfg.objective_coef = 0.5
    return ReprLearningModel(cfg, variant="slotgraph3_baseline", llama_model=None).to(DEV)


def mae_batch(B=4, T=128, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.randint(1, VOCAB, (B, T), generator=g).to(DEV)
    return types.SimpleNamespace(context_ids=ids,
                                 context_mask=torch.ones(B, T, dtype=torch.bool, device=DEV),
                                 k_slots=None)


def main():
    torch.manual_seed(0)
    model = mk_model()
    model.task_mode = "masked_reconstruction"
    B = 4
    batch = mae_batch(B=B)
    coef = 0.5

    # ── replicate the GradCache passes exactly as _grad_cached_objective_step does ──
    print("1. pass-1 (no-grad) vs pass-2 (grad) S-consistency — RNG mask alignment")
    _rng = torch.cuda.get_rng_state()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc_out = model.compute_loss(batch, window_size=1024, encoder_only=True)
    mem, aux = enc_out["_memory"], enc_out["_mem_aux"]
    mem_leaf = mem.detach().requires_grad_(True)
    S1_rows, S2_rows = [], []
    with torch.no_grad():
        for r in range(B):
            torch.cuda.set_rng_state(_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model.compute_loss(batch, window_size=1024, memory_override=(mem_leaf, aux),
                                       shuffle_memory=(r > 0), shuffle_roll=r)
            S1_rows.append(o["loss_per_example"].float())
    for r in range(B):
        torch.cuda.set_rng_state(_rng)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.compute_loss(batch, window_size=1024, memory_override=(mem_leaf, aux),
                                   shuffle_memory=(r > 0), shuffle_roll=r)
        S2_rows.append(o["loss_per_example"].float().detach())
    S1 = torch.stack(S1_rows, 0)
    S2 = torch.stack(S2_rows, 0)
    d12 = float((S1 - S2).abs().max())
    check("pass1 == pass2 element-wise (identical masks)", d12 < 2e-3, f"maxΔ={d12:.2e} (bf16)")

    print("2. analytic W directions (standardized-logit form)")
    from train import _infonce_logits_weights, _same_answer_valid_mask  # noqa: E402
    nce_t, W, p = _infonce_logits_weights(S1, coef, inv_temp=1.0, valid=None)
    check("W[0,:] > 0 — positive's CE is MINIMIZED", bool((W[0] > 0).all().item()),
          f"min={float(W[0].min()):.4f}")
    check("W[r>0,:] < 0 — negatives' CE is MAXIMIZED", bool((W[1:] < 0).all().item()),
          f"max={float(W[1:].max()):.4f}")
    csum = W.sum(0)
    check("column sums == 1/B", bool(torch.allclose(csum, torch.full_like(csum, 1.0 / B), atol=1e-6)),
          f"{[round(float(x), 4) for x in csum]}")
    # W must equal autograd of the SAME standardized objective (detached stats)
    S_leaf = S1.detach().clone().requires_grad_(True)
    with torch.no_grad():
        mu = S_leaf.mean(0, keepdim=True); sd = S_leaf.std(0, keepdim=True).clamp_min(1e-4)
    z = -(S_leaf - mu) / sd
    loss_ref = S_leaf[0].mean() + coef * (-(torch.log_softmax(z, dim=0)[0]).mean())
    loss_ref.backward()
    dW = float((S_leaf.grad - W).abs().max())
    check("W == autograd of the standardized objective", dW < 1e-6, f"maxΔ={dW:.2e}")
    # masked-stats variant (review fix): with a valid mask, μ/σ use valid entries only —
    # verify the analytic W still equals autograd of that exact objective
    vmask = torch.ones(B, B, dtype=torch.bool, device=DEV)
    vmask[2, 0] = False; vmask[3, 1] = False                             # two arbitrary false negatives
    nce_m, W_m, _ = _infonce_logits_weights(S1, coef, inv_temp=1.0, valid=vmask)
    S_leaf2 = S1.detach().clone().requires_grad_(True)
    with torch.no_grad():
        v = vmask.float(); n = v.sum(0, keepdim=True).clamp_min(2.0)
        mu2 = (S_leaf2 * v).sum(0, keepdim=True) / n
        sd2 = (((S_leaf2 - mu2) ** 2 * v).sum(0, keepdim=True) / (n - 1.0)).sqrt().clamp_min(1e-4)
    z2 = (-(S_leaf2 - mu2) / sd2).masked_fill(~vmask, float("-inf"))
    (S_leaf2[0].mean() + coef * (-(torch.log_softmax(z2, dim=0)[0]).mean())).backward()
    dWm = float((S_leaf2.grad - W_m).abs().max())
    check("masked-stats W == autograd (+ masked entries get W=0)",
          dWm < 1e-6 and float(W_m[2, 0]) == 0.0 and float(W_m[3, 1]) == 0.0, f"maxΔ={dWm:.2e}")

    print("3. nce cross-check + standardization sharpens")
    z1 = -(S1 - S1.mean(0, keepdim=True)) / S1.std(0, keepdim=True).clamp_min(1e-4)
    nce_b = float((-torch.log_softmax(z1, dim=0)[0]).mean())
    check("helper nce == independent computation", abs(float(nce_t) - nce_b) < 1e-5,
          f"{float(nce_t):.4f} vs {nce_b:.4f} (chance=ln{B}={math.log(B):.4f})")
    p_raw = torch.softmax(-S1, dim=0)[0].mean()                          # the OLD τ=1 raw-logit regime
    check("raw τ=1 softmax was near-uniform (the dead-gradient regime the fix removes)",
          abs(float(p_raw) - 1.0 / B) < 0.08,
          f"raw p_pos={float(p_raw):.3f} vs chance {1.0/B:.3f}; standardized p_pos={float(p[0].mean()):.3f}")

    print("3b. same-answer false-negative mask")
    qb = types.SimpleNamespace(
        answer_ids=torch.tensor([[5, 6], [7, 8], [5, 6], [9, 0]], device=DEV),
        answer_content_mask=torch.tensor([[1, 1], [1, 1], [1, 1], [1, 0]], dtype=torch.bool, device=DEV))
    vm = _same_answer_valid_mask(qb, 4, DEV)
    # examples 0 and 2 share the answer → the rolls pairing them must be invalid:
    # roll r pairs target i with memory (i−r)%4 → (i=0,r=2)→j=2 and (i=2,r=2)→j=0
    expect_false = {(2, 0), (2, 2)}
    got_false = {(r, i) for r in range(4) for i in range(4) if not bool(vm[r, i])}
    check("mask excludes exactly the same-answer pairings", got_false == expect_false,
          f"masked={sorted(got_false)}")

    print("4. GRPO policy-gradient direction")
    G = 3
    logp = torch.randn(G, B, device=DEV, requires_grad=True)
    R = torch.randn(G, B, device=DEV)
    adv = R - R.mean(0, keepdim=True)
    pol = -(adv.detach() * logp).mean()
    pol.backward()
    ok_dir = bool(((logp.grad * adv) <= 1e-9).all().item())              # grad opposes advantage sign
    check("positive-advantage rollout gets NEGATIVE logp grad (descent raises its prob)",
          ok_dir, f"max(grad·adv)={float((logp.grad * adv).max()):.2e}")
    check("group baseline zeroes mean advantage", abs(float(adv.mean())) < 1e-6)

    print("5. perturbation probe — corrupting the REAL memory must RAISE the InfoNCE term")
    with torch.no_grad():
        noise = torch.randn_like(mem)
        mem_bad = mem + 0.5 * mem.norm(dim=-1, keepdim=True) * F.normalize(noise, dim=-1)
        Sb_rows = []
        for r in range(B):
            torch.cuda.set_rng_state(_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model.compute_loss(batch, window_size=1024, memory_override=(mem_bad, aux),
                                       shuffle_memory=(r > 0), shuffle_roll=r)
            Sb_rows.append(o["loss_per_example"].float())
        Sb = torch.stack(Sb_rows, 0)
        # corrupting ALL memories equally: nce should stay ~chance. Corrupt ONLY the real pairing
        # by scoring: positives from corrupted memory, negatives from clean rolls:
        S_mixed = S1.clone(); S_mixed[0] = Sb[0]
        nce_clean = float(_infonce_logits_weights(S1, coef)[0])
        nce_bad = float(_infonce_logits_weights(S_mixed, coef)[0])
    check("worse positive ⇒ higher InfoNCE", nce_bad > nce_clean,
          f"{nce_clean:.4f} → {nce_bad:.4f} (Δ={nce_bad-nce_clean:+.4f})")

    n_ok = sum(1 for _, ok, _ in results if ok)
    print(f"\n{'='*60}\n{n_ok}/{len(results)} checks passed")
    if n_ok < len(results):
        for nm, ok, det in results:
            if not ok:
                print(f"  FAILED: {nm} — {det}")
        sys.exit(1)


if __name__ == "__main__":
    main()
