"""Objective ladder: CE / in-batch InfoNCE / MCR² coding-rate / GRPO trajectory.

Extracted verbatim from ``scripts/train/train.py`` (harness reorg phase 2). No logic changes.
"""
from __future__ import annotations

import torch


def _infonce_logits_weights(S, coef, inv_temp=1.0, valid=None):
    """Row-standardized InfoNCE over the memory-roll score matrix + the analytic dL/dS.

    S[r, i] = CE(target_i | memory_{i−r}); roll 0 = the positive. 2026-07-03 math audit:
    (a) RAW mean-NLL margins are O(0.1–0.3) — at τ=1 over B=8 the softmax is nearly uniform
        (p_pos≈0.16 vs chance 0.125) → dead gradient. Fix: per-example STANDARDIZATION over
        rolls (detached μ_i, σ_i — the stats are normalizers, not optimization targets), then
        a tunable inverse temperature. z[r,i] = −(S[r,i] − μ_i)/σ_i · inv_temp.
    (b) valid[r,i]=False excludes FALSE NEGATIVES from example i's denominator (bAbI batch
        mates sharing the gold answer legitimately explain each other's targets — penalizing
        them teaches the encoder to make same-answer memories dissimilar, corrupting binding;
        SupCon same-label-is-not-a-negative rule). Masked entries get z=−inf ⇒ p=0 ⇒ W=0.
    Returns (nce, W, p) with W[r,i] = δ_{r,0}/B + coef·(δ_{r,0} − p[r,i])·inv_temp/(σ_i·B) —
    the exact gradient of  mean_i S[0,i] + coef·nce  wrt S under detached stats.
    """
    B = S.shape[0]
    if valid is not None:
        # standardization stats over VALID entries only (review fix): a masked same-answer false
        # negative has legitimately LOW CE — including it would drag μ down and inflate σ,
        # distorting the gradient scale of the entries that remain in the softmax.
        v = valid.float()
        n = v.sum(dim=0, keepdim=True).clamp_min(2.0)
        mu = (S * v).sum(dim=0, keepdim=True) / n
        sd = (((S - mu) ** 2 * v).sum(dim=0, keepdim=True) / (n - 1.0)).sqrt().clamp_min(1e-4)
    else:
        mu = S.mean(dim=0, keepdim=True)
        sd = S.std(dim=0, keepdim=True).clamp_min(1e-4)
    z = -(S - mu) / sd * float(inv_temp)
    if valid is not None:
        z = z.masked_fill(~valid, float("-inf"))
    logp = torch.log_softmax(z, dim=0)
    nce = -logp[0].mean()
    p = logp.exp()
    delta = torch.zeros_like(S)
    delta[0] = 1.0
    W = delta / B + coef * (delta - p) * (float(inv_temp) / sd) / B
    return nce, W, p


def _same_answer_valid_mask(batch, B, device):
    """[B_roll, B_ex] bool: False where roll r pairs example i with a memory whose gold answer
    equals example i's (a false negative — bAbI's tiny answer vocabulary makes these common).
    None for batches without answers (MAE/continuation passages: false-negative risk ~nil)."""
    a_ids = getattr(batch, "answer_ids", None)
    a_cm = getattr(batch, "answer_content_mask", None)
    if a_ids is None or a_cm is None:
        return None
    a_norm = a_ids.masked_fill(~a_cm.bool(), -1)
    eq = (a_norm.unsqueeze(0) == a_norm.unsqueeze(1)).all(-1)            # [B,B] same-answer
    idx = torch.arange(B, device=device)
    j_of = (idx.view(1, B) - idx.view(B, 1)) % B                         # [r,i] → memory's example j
    valid = ~eq[idx.view(1, B).expand(B, B), j_of]
    valid[0] = True                                                      # the positive always counts
    return valid


def _coding_rate(memory, eps2=0.5):
    """MCR² coding rate (Yu et al. 2020) of the WITHIN-example memory, averaged over the batch.
    memory [B,M,d] → per example, unit-norm the M tokens (scale-robust), then
    R_i = ½·logdet(I_d + (d/(M·eps2))·ZᵀZ). Rank-1 Z → R≈0; full-rank Z → R large. Returned as a
    scalar to MAXIMIZE (the caller subtracts rank_reward_coef·R from the loss)."""
    B, M, d = memory.shape
    Z = memory / memory.norm(dim=-1, keepdim=True).clamp_min(1e-6)      # [B,M,d] unit tokens
    G = torch.einsum("bmd,bnd->bmn", Z, Z)                              # [B,M,M] Gram (M<d ⇒ cheaper, same logdet)
    scale = d / (M * eps2)
    eye = torch.eye(M, device=Z.device, dtype=Z.dtype).unsqueeze(0)
    R = 0.5 * torch.logdet(eye + scale * G)                            # [B]  logdet(I+αZZᵀ)=logdet(I+αZᵀZ)
    return R.mean()


def _grad_cached_objective_step(model, batch, cfg, window_size):
    """contrastive/trajectory objective step (train_mixed_variant — the trainer that actually
    runs mixed; the 2026-07-02 lesson: the legacy coef lived only in train_one_variant and was
    silently ignored here).

    In-batch InfoNCE with a GradCache-style cut at the memory tensor, so peak activations stay
    at ONE decoder branch + one encoder graph (B+1 live decoder graphs would OOM at BS8):
      pass 0: encoder ONCE (graph kept; decode skipped).
      pass 1 (no_grad): decode the B memory rolls → S[r,i] = CE(target_i | memory_{i−r});
              roll 0 = REAL. InfoNCE = CE over logits −S per example (positive = roll 0).
      pass 2 (grad, one roll at a time): recompute roll r, backward(Σ_i W[r,i]·CE_r[i]) with
              W = the ANALYTIC d(CE_real + coef·InfoNCE)/dS — decoder-side grads accumulate
              directly, memory grads accumulate on a detached leaf; each branch is freed
              before the next (passes 1/2 share RNG → identical masks → identical S).
      pass 3: memory.backward(gradient=leaf.grad) — encoder backward ONCE.
    trajectory mode adds GRPO on the discrete read: G Gumbel-top-k sampled expansions,
    reward = per-example binding advantage (CE_shuf − CE_shuf-free real, no-grad reads),
    group-relative advantage × logprob REINFORCE on the ROUTER only (hybrid estimator).

    BACKWARD RUNS INSIDE — the caller must NOT call loss.backward(). Returns
    (out_real, total_loss_detached, extras_for_the_train_row)."""
    F = torch.nn.functional
    _rng = torch.cuda.get_rng_state()
    B = batch.context_ids.shape[0]
    coef = float(getattr(cfg, "objective_coef", 0.5))
    # ── pass 0: encoder only, keep the graph ──
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        enc_out = model.compute_loss(batch, window_size=window_size, encoder_only=True)
    mem, aux = enc_out["_memory"], enc_out["_mem_aux"]
    mem_leaf = mem.detach().requires_grad_(True)
    # ── pass 1: no-grad score matrix S[r, i] ──
    S_rows = []
    with torch.no_grad():
        for r in range(B):
            torch.cuda.set_rng_state(_rng)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model.compute_loss(batch, window_size=window_size,
                                       memory_override=(mem_leaf, aux),
                                       shuffle_memory=(r > 0), shuffle_roll=r)
            S_rows.append(o["loss_per_example"].float())
    S = torch.stack(S_rows, dim=0)                     # [B_roll, B_ex]
    # NOTE: obj_ce_real is the ROW-mean (each example weighted equally) — the plain mode's
    # loss_recon is the TOKEN-mean; both are logged (train_row carries out_real's token-mean
    # loss_recon), so cross-mode comparisons should use loss_recon, not obj_ce_real.
    ce_real = S[0].mean()
    with torch.no_grad():
        valid = _same_answer_valid_mask(batch, B, S.device)
        nce, W, _p = _infonce_logits_weights(
            S, coef, inv_temp=float(getattr(cfg, "objective_inv_temp", 1.0)), valid=valid)
    # ── pass 2: grad, one roll at a time (surrogate = Σ W·CE reproduces the exact gradient) ──
    out_real = None
    for r in range(B):
        torch.cuda.set_rng_state(_rng)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = model.compute_loss(batch, window_size=window_size,
                                   memory_override=(mem_leaf, aux),
                                   shuffle_memory=(r > 0), shuffle_roll=r)
        (W[r].to(o["loss_per_example"].dtype) * o["loss_per_example"]).sum().backward()
        if r == 0:
            out_real = {k: v for k, v in o.items() if not torch.is_tensor(v) or not v.requires_grad}
    # ── pass 3: encoder backward once, through the accumulated memory gradient ──
    if mem_leaf.grad is not None:
        mem.backward(gradient=mem_leaf.grad.to(mem.dtype))
    extras = {"obj_nce": float(nce), "obj_ce_real": float(ce_real)}
    # ── trajectory: GRPO on the discrete read expansion (router-only REINFORCE) ──
    if str(getattr(cfg, "objective_mode", "plain")) == "trajectory":
        lat = aux.get("latents")
        if lat is None or not hasattr(model.encoder, "sample_read_expansion"):
            raise ValueError("objective_mode=trajectory needs a slotgraph3 encoder "
                             "(finalize aux['latents'] + sample_read_expansion)")
        nl, el = lat
        rollouts, route_H = model.encoder.sample_read_expansion(
            nl.detach(), el.detach(), int(getattr(cfg, "grpo_samples", 4)))
        rewards, logps = [], []
        for mem_g, keep_g, logp_g in rollouts:
            aux_g = {"memory_mask": keep_g}
            with torch.no_grad():
                torch.cuda.set_rng_state(_rng)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    o_real = model.compute_loss(batch, window_size=window_size,
                                                memory_override=(mem_g, aux_g))
            # reward = −CE_real (2026-07-03 audit: the old CE_shuf−CE_real reward had an
            # UNBOUNDED poison lever — raise CE_shuf instead of lowering CE_real; the group
            # baseline already strips per-example difficulty, so the shuf term was redundant
            # for its purpose. Also halves the reward reads.)
            rewards.append((-o_real["loss_per_example"]).float())
            logps.append(logp_g)
        R = torch.stack(rewards, dim=0)                # [G, B] −CE_real per rollout
        adv = R - R.mean(dim=0, keepdim=True)          # group baseline = (G−1)/G × RLOO (unbiased dir)
        # per-DECISION scale: logp sums K×topk ≈ 128 PL terms (summing is the correct joint
        # log-prob — per-sample averaging would be the Dr.GRPO length bias; dividing by a
        # CONSTANT is a pure coefficient, direction-identical) so grpo_coef=1.0 stays sane.
        n_dec = max(1, model.encoder.K * model.encoder.read_topk)
        pol = -(adv.detach() * torch.stack(logps, dim=0)).mean() / n_dec
        # entropy BONUS on the router distribution — THE load-bearing regularizer for policy
        # gradients on latent structure (entropy collapse = the documented GRPO failure mode;
        # Gumbel noise alone stops exploring once logits sharpen).
        ent_coef = float(getattr(cfg, "grpo_entropy_coef", 0.01))
        (float(getattr(cfg, "grpo_coef", 1.0)) * pol - ent_coef * route_H).backward()
        extras["obj_grpo_policy"] = float(pol)
        extras["obj_grpo_reward"] = float(R.mean())
        extras["obj_grpo_reward_std"] = float(R.std())
        extras["obj_route_entropy"] = float(route_H)
    total = float(ce_real) + coef * float(nce)
    out_real["loss"] = torch.tensor(total, device=S.device)        # detached (backward already done)
    out_real["loss_recon"] = ce_real
    return out_real, out_real["loss"], extras
