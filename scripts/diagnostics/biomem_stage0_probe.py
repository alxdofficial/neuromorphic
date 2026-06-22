"""STAGE 0 — feasibility probe for the gated fast-Hebbian memory.

The load-bearing bet of the whole bio-memory arm: memory lives in synaptic STATE
(a fast weight matrix W updated by a *gated* Hebbian rule per input), and both
write and read are signal propagation. Before building the cortical-column grid +
LM integration, we isolate the ONE assumption that everything rests on:

  Can a single gated fast-Hebbian layer BIND a handful of (key, value) vector
  pairs and RECALL the value when re-queried with the key?

Setup (NO grid, NO LM — one W, one regulator gate):
  * Per example draw N random key/value pairs (random unit vectors in R^d).
  * Fast weight W in R^{d x d} resets to 0 each example.
  * WRITE pair i: propagate key_i through W (read-then-write), form the Hebbian
    outer-product proposal dW = outer(s_pre, err_post), and let a LEARNED
    regulator gate g in [-1,1] decide how much of dW to commit (apply / freeze /
    reverse). W <- clamp(W + g * dW, -1, 1).
  * QUERY key_j: s_out = hardtanh(W @ s_key_j - theta); decode -> recovered value.
  * Score = mean cosine(recovered, true value) over the N pairs, averaged over a
    batch of examples. Also report a discrimination metric: is the recalled value
    closer to its OWN target than to the other N-1 distractors (recall@1)?

We TRAIN (by backprop through the whole write+read rollout) the small learned
objects: key/value encoders, the regulator MLP, the readout decoder, and the leak
lambda. theta thresholds are random-fixed (per the spec). We sweep N in {4,8,16,32}.

Verdict: if it CANNOT recall ~8 pairs after training, STOP (the grid won't work).
If it can, proceed to the grid.
"""
from __future__ import annotations
import sys, os, math, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
import torch.nn as nn
import torch.nn.functional as F


def hardtanh(x):
    return torch.clamp(x, -1.0, 1.0)


class GatedHebbProbe(nn.Module):
    """A single gated fast-Hebbian binding layer + the small learned wrappers.

    d      : substrate width (the fast weight is d x d)
    The fast weight W is per-example STATE (reset to 0); the only PARAMETERS are
    the key/value encoders, the regulator MLP, the readout, and the leak scalar.
    theta is a random-FIXED per-neuron threshold buffer (not learned).
    """

    def __init__(self, d_io: int, d: int, reg_hidden: int = 32):
        super().__init__()
        self.d = d
        # encoders: map the io vectors into the substrate (1/sqrt(fan_in) init)
        self.key_enc = nn.Linear(d_io, d)
        self.val_enc = nn.Linear(d_io, d)
        self.readout = nn.Linear(d, d_io)           # substrate activity -> io value
        # leak lambda: learned scalar in (0,1) via sigmoid of a raw param
        self.leak_raw = nn.Parameter(torch.tensor(-2.0))   # sigmoid(-2) ~= 0.12
        # regulator: per-edge MLP on [dW_ij, s_pre_i, s_post_j] -> g_ij in [-1,1].
        # (Stage 0 has a single location, so no cond vector; the grid adds cond_l.)
        self.reg = nn.Sequential(
            nn.Linear(3, reg_hidden), nn.GELU(),
            nn.Linear(reg_hidden, 1),
        )
        # random-FIXED per-neuron threshold (drawn once, registered as a buffer)
        self.register_buffer("theta", 0.1 * torch.randn(d))

    @property
    def leak(self):
        return torch.sigmoid(self.leak_raw)

    def _gate(self, dW, s_pre, s_post):
        # dW: [B,d,d]  s_pre: [B,d]  s_post: [B,d]
        B, d, _ = dW.shape
        feat = torch.stack([
            dW,
            s_pre[:, :, None].expand(B, d, d),
            s_post[:, None, :].expand(B, d, d),
        ], dim=-1)                                   # [B,d,d,3]
        g = torch.tanh(self.reg(feat).squeeze(-1))   # [B,d,d] in [-1,1]
        return g

    def write_read(self, keys, vals):
        """keys, vals: [B,N,d_io]. Returns recovered values [B,N,d_io]."""
        B, N, _ = keys.shape
        d = self.d
        s_keys = hardtanh(self.key_enc(keys))        # [B,N,d]
        s_vals = hardtanh(self.val_enc(vals))        # [B,N,d] target post-activity
        W = torch.zeros(B, d, d, device=keys.device, dtype=keys.dtype)
        leak = self.leak
        # ---- WRITE: bind each pair via the gated-Hebbian rule ----
        for i in range(N):
            s_pre = s_keys[:, i]                      # [B,d]
            # current readout of the key through W (the "post" the substrate produces)
            inp = torch.einsum("bij,bi->bj", W, s_pre)   # [B,d]
            s_post_cur = hardtanh(inp - self.theta)
            # Hebbian proposal toward the TARGET value post-activity (delta-rule flavor:
            # pre x (target - current) so already-bound pairs propose ~0 update).
            s_post_tgt = s_vals[:, i]
            err = s_post_tgt - s_post_cur                # [B,d]
            dW = torch.einsum("bi,bj->bij", s_pre, err)  # [B,d,d] Hebbian outer product
            dW = dW - leak * W                            # leak term
            g = self._gate(dW, s_pre, s_post_cur)        # [B,d,d] gate in [-1,1]
            W = torch.clamp(W + g * dW, -1.0, 1.0)
        # ---- READ: query each key, decode the recovered value ----
        recovered = []
        for j in range(N):
            s_pre = s_keys[:, j]
            inp = torch.einsum("bij,bi->bj", W, s_pre)
            s_out = hardtanh(inp - self.theta)
            recovered.append(self.readout(s_out))        # [B,d_io]
        return torch.stack(recovered, dim=1)             # [B,N,d_io]


def make_batch(B, N, d_io, device):
    keys = F.normalize(torch.randn(B, N, d_io, device=device), dim=-1)
    vals = F.normalize(torch.randn(B, N, d_io, device=device), dim=-1)
    return keys, vals


def cosine_recall(recovered, vals):
    """mean cosine(recovered_i, true_i)."""
    r = F.normalize(recovered, dim=-1)
    v = F.normalize(vals, dim=-1)
    return (r * v).sum(-1).mean()


def recall_at_1(recovered, vals):
    """fraction where recovered_i is closest (cosine) to its OWN target among all N."""
    r = F.normalize(recovered, dim=-1)                  # [B,N,d]
    v = F.normalize(vals, dim=-1)                       # [B,N,d]
    sim = torch.einsum("bnd,bmd->bnm", r, v)           # [B,N,N] recovered n vs target m
    pred = sim.argmax(-1)                               # [B,N]
    tgt = torch.arange(vals.shape[1], device=vals.device)[None].expand_as(pred)
    return (pred == tgt).float().mean()


def run_n(N, d_io=32, d=64, steps=1500, B=64, lr=3e-3, device="cuda", seed=0):
    torch.manual_seed(seed)
    model = GatedHebbProbe(d_io=d_io, d=d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        keys, vals = make_batch(B, N, d_io, device)
        recovered = model.write_read(keys, vals)
        # train objective: maximize cosine recall (= minimize 1 - cosine)
        loss = 1.0 - cosine_recall(recovered, vals)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
    # eval on fresh data
    model.eval()
    with torch.no_grad():
        keys, vals = make_batch(256, N, d_io, device)
        rec = model.write_read(keys, vals)
        cos = cosine_recall(rec, vals).item()
        r1 = recall_at_1(rec, vals).item()
        leak = model.leak.item()
    return cos, r1, leak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--d", type=int, default=64, help="substrate width (W is dxd)")
    ap.add_argument("--d-io", type=int, default=32, help="key/value vector dim")
    ap.add_argument("--Ns", type=int, nargs="+", default=[4, 8, 16, 32])
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"STAGE 0 — gated fast-Hebbian recall probe (device={device}, "
          f"d_io={args.d_io}, substrate d={args.d}, steps={args.steps})")
    print(f"{'N':>4} | {'cos_recall':>10} | {'recall@1':>9} | {'leak':>6}")
    print("-" * 40)
    results = {}
    for N in args.Ns:
        cos, r1, leak = run_n(N, d_io=args.d_io, d=args.d, steps=args.steps, device=device)
        results[N] = (cos, r1)
        print(f"{N:>4} | {cos:>10.3f} | {r1:>9.3f} | {leak:>6.3f}")
    print("-" * 40)
    # verdict at N=8 (the spec's go/no-go threshold)
    cos8, r18 = results.get(8, (0.0, 0.0))
    verdict = "PROMISING — proceed to the grid" if (r18 > 0.6 or cos8 > 0.5) else \
              "WEAK — gated Hebbian cannot bind ~8 pairs; reconsider before building the grid"
    print(f"\nVERDICT (N=8): cos={cos8:.3f}, recall@1={r18:.3f}  ->  {verdict}")


if __name__ == "__main__":
    main()
