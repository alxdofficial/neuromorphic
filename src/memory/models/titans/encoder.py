"""Titans-INSPIRED neural long-term memory (after Behrouz et al. 2024, arXiv:2501.00663) — a memory
that LEARNS TO MEMORIZE AT TEST TIME. The memory is a DEEP MLP M whose weights are updated per window
by a GRADIENT STEP on the associative loss ‖M(kₜ)−vₜ‖² with data-dependent forget (α), momentum (η)
and learning-rate (θ) gates; the OUTER training loop backprops THROUGH those inner updates
(create_graph) — that is Titans' "learning to learn" signal and the whole point of the arm. This WRITE
is faithful to Titans' neural-memory core.

Read = a SIMPLIFIED prepend readout, NOT faithful MAC (Memory-as-Context): N_p persistent tokens
(input-independent, Eq.19) ‖ M_q memory-readout tokens = learned query seeds run through the FINAL
(test-time-adapted) memory MLP, then prepended. Published MAC instead has the CURRENT segment query
the memory, local attention over persistent+current context, an update from the attention output, and
a post-update gated read — none of that is here. So the arm is faithful to Titans' test-time-gradient
WRITE but its read is a static prepend → call it "Titans-inspired neural memory," not MAC.
ASTERISKS: test-time gradient write (not feed-forward) + create_graph meta-cost; simplified non-MAC
read; no frozen base copy (writes on raw token embeds); persistent state = the memory-MLP weights.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...common import _NormMatch
from ...config import ReprConfig


class TitansEncoder(nn.Module):
    # Simplified prepend read (NOT faithful MAC); write on raw embeds; surprise = assoc-loss gradient.
    is_conditioned_read = False
    wants_surprise = False

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_llama
        self.d = d
        self.dM = int(getattr(cfg, "titans_d_mem", 0) or d)          # d_M = d so the readout lands in decoder space
        self.h = int(getattr(cfg, "titans_mem_hidden", 4864))         # MLP hidden (sized to ~7M)
        self.Np = int(getattr(cfg, "titans_n_persistent", 16))
        self.Mq = int(getattr(cfg, "titans_n_read_seeds", 0) or (cfg.n_flat_codes - self.Np))
        dM = self.dM
        self.W_K = nn.Linear(d, dM, bias=False)
        self.W_V = nn.Linear(d, dM, bias=False)
        # memory MLP INIT (meta-learned): M(x) = W2·SiLU(W1·x + b1) + b2. W2 zero-init → M≈0 at start
        # (stable: first window's loss = ‖v‖², grads flow, memory starts writing).
        self.M0_W1 = nn.Parameter(torch.empty(self.h, dM)); nn.init.kaiming_uniform_(self.M0_W1, a=5 ** 0.5)
        self.M0_b1 = nn.Parameter(torch.zeros(self.h))
        self.M0_W2 = nn.Parameter(torch.zeros(dM, self.h))
        self.M0_b2 = nn.Parameter(torch.zeros(dM))
        # data-dependent gates from the window-mean key: α forget, η momentum, θ LR (θ starts small).
        self.g_alpha = nn.Linear(dM, 1)
        self.g_eta = nn.Linear(dM, 1)
        self.g_theta = nn.Linear(dM, 1)
        with torch.no_grad():
            self.g_theta.bias.fill_(-3.0)                             # θ≈0.047 at init (gentle inner LR)
            self.g_alpha.bias.fill_(-2.0)                             # α≈0.12 (slow forget)
        self.persistent = nn.Parameter(torch.randn(self.Np, d) * 0.02)
        self.read_seeds = nn.Parameter(torch.randn(self.Mq, dM) * 0.02)
        self.out_norm = _NormMatch(d)                                 # calibrated to embed norm in ReprModel.__init__
        print(f"[Titans] deep-MLP memory h={self.h}, d_M={dM}; read = {self.Np} persistent + "
              f"{self.Mq} readout (prepend); test-time autograd write (L_M=2)")

    def init_streaming_state(self, batch_size, device, dtype):
        B = batch_size
        st = {}
        # enable_grad so the per-example weights REQUIRE grad even under eval's no_grad (the inner
        # autograd.grad needs it). Derived from M0 (not detached), so the OUTER loop still trains the
        # meta-learned memory-MLP init through the readout.
        with torch.enable_grad():
            for nm, p in (("W1", self.M0_W1), ("b1", self.M0_b1),
                          ("W2", self.M0_W2), ("b2", self.M0_b2)):
                w = p.to(device, dtype).unsqueeze(0).expand(B, *p.shape).contiguous()
                st[nm] = w
                st["S_" + nm] = torch.zeros_like(w)                  # momentum buffers (constants)
        return st

    def _mem_forward(self, W1, b1, W2, b2, x):
        """Batched per-example deep memory: x [B,T,dM] → [B,T,dM]."""
        z = torch.einsum("bhd,btd->bth", W1, x) + b1.unsqueeze(1)
        return torch.einsum("bdh,bth->btd", W2, F.silu(z)) + b2.unsqueeze(1)

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        B, W, d = token_embeds.shape
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        active = attention_mask.bool().any(dim=1)   # [B]; drives _freeze_inactive below
        # (No `if not active.any(): return` early-out: it would force a per-window GPU→CPU sync ×8/step.
        #  The all-pad-batch case is handled bit-identically by _freeze_inactive — loss=0 → grads=0 →
        #  updates frozen back to the old weights — so the shortcut only saved rare wasted compute at
        #  the cost of a guaranteed sync every window. Removed.)
        m = attention_mask.to(token_embeds.dtype).unsqueeze(-1)
        x = token_embeds * m
        # enable_grad: the test-time memory update must run even under eval's no_grad (Titans updates
        # memory at TEST time — that is the point). create_graph=training makes the OUTER loop backprop
        # through the inner step in train, first-order (no meta-graph) in eval.
        # enable_grad wraps the ENTIRE update: the test-time memory step must run under eval's no_grad
        # too (Titans updates at TEST time), AND the UPDATED weights must keep requiring grad so the
        # next window's inner autograd.grad works. create_graph=training → the OUTER loop backprops
        # through the inner step in train; first-order (no meta-graph) in eval.
        with torch.enable_grad():
            k = self.W_K(x); v = self.W_V(x)                         # [B,W,dM]
            W1, b1, W2, b2 = state["W1"], state["b1"], state["W2"], state["b2"]
            pred = self._mem_forward(W1, b1, W2, b2, k)
            err = ((pred - v) ** 2).sum(-1) * attention_mask.to(pred.dtype)   # [B,W], pad-masked
            # PER-EXAMPLE mean, then SUM over examples (not a single batch-wide mean): each example's
            # fast-weight gradient must be normalized by ITS OWN token count, not the batch total. Dividing
            # by the whole-batch token sum made example b's update scale ~1/B (a B=2 duplicate halved it —
            # a batch-dependent memory update). Summing per-example means keeps ∂loss/∂W1[b] a function of
            # example b alone. (Still a per-token mean → the 256× inner-grad blowup stays fixed; identical
            # at B=1, corrects the silent B>1 under-update.)
            _tok = attention_mask.to(pred.dtype).sum(1).clamp_min(1.0)         # [B] valid tokens per example
            loss = (err.sum(1) / _tok).sum()
            # retain_graph defaults to create_graph: in train (create_graph=True) the graph the outer
            # backward needs is kept; in eval it's freed right after this grad call (frees per-window
            # loss-graph buffers). Was an explicit retain_graph=True → held eval buffers needlessly.
            g1, gb1, g2, gb2 = torch.autograd.grad(
                loss, [W1, b1, W2, b2], create_graph=self.training)
            # data-dependent gates from the window-mean key
            km = (k * m).sum(1) / m.sum(1).clamp_min(1e-3)           # [B,dM]
            alpha = torch.sigmoid(self.g_alpha(km))
            eta = torch.sigmoid(self.g_eta(km))
            theta = torch.sigmoid(self.g_theta(km))
            def _upd(Wt, St, g):
                nd = Wt.dim(); shp = (B,) + (1,) * (nd - 1)
                St = eta.view(shp) * St - theta.view(shp) * g        # momentum-ed surprise
                return (1.0 - alpha.view(shp)) * Wt + St, St        # forget + write
            nW1, nS1 = _upd(W1, state["S_W1"], g1)
            nb1, nSb1 = _upd(b1, state["S_b1"], gb1)
            nW2, nS2 = _upd(W2, state["S_W2"], g2)
            nb2, nSb2 = _upd(b2, state["S_b2"], gb2)
            def _freeze_inactive(new, old):
                shp = (B,) + (1,) * (new.dim() - 1)
                return torch.where(active.view(shp), new, old)
            nW1 = _freeze_inactive(nW1, W1)
            nS1 = _freeze_inactive(nS1, state["S_W1"])
            nb1 = _freeze_inactive(nb1, b1)
            nSb1 = _freeze_inactive(nSb1, state["S_b1"])
            nW2 = _freeze_inactive(nW2, W2)
            nS2 = _freeze_inactive(nS2, state["S_W2"])
            nb2 = _freeze_inactive(nb2, b2)
            nSb2 = _freeze_inactive(nSb2, state["S_b2"])
        return {"W1": nW1, "b1": nb1, "W2": nW2, "b2": nb2,
                "S_W1": nS1, "S_b1": nSb1, "S_W2": nS2, "S_b2": nSb2}, {}

    def finalize_memory(self, state):
        W1, b1, W2, b2 = state["W1"], state["b1"], state["W2"], state["b2"]
        B = W1.shape[0]
        seeds = self.read_seeds.to(W1.dtype).unsqueeze(0).expand(B, self.Mq, self.dM)
        readout = self._mem_forward(W1, b1, W2, b2, seeds)          # [B, Mq, d]  (example-specific)
        readout = self.out_norm(readout.float()).to(W1.dtype)
        persist = self.persistent.to(W1.dtype).unsqueeze(0).expand(B, self.Np, self.d)
        mem = torch.cat([persist, readout], dim=1)                 # [B, Np+Mq, d] prepend
        return mem, {}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        B = token_embeds.shape[0]
        st = self.init_streaming_state(B, token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
