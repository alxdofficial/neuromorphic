"""slotgraph — THE graph memory (docs/slotgraph_design.md).

96 node slots, NO edge tokens. ONE shared frozen LM (SmolLM2-135M) with a write-side LoRA does the write by
HARVESTING its per-layer attention; the read is the same base with a read-side LoRA (the shared decoder
adapter). A persistent per-edge state E[i,j] (d_e-vec, dense N×N, init ZERO) rides the attention VALUE path
as an additive residual: out_i += U·(Σ_j a_ij·E[i,j]). E is accumulated within a window from how the LM's
per-layer node-block attention EVOLVES across depth (a learnable inter-layer diff+product feature), and
committed once per window by an ERROR-CORRECTING, per-edge-GATED, EntNet-BOUNDED write. Read = prepend+bidir;
E shapes the node tokens (graph-conv blend + learned-salience pointers), never tokenized.

This is a harvest-and-correct realization of Relational Attention (Diao & Loynd 2023) on a frozen+LoRA LM:
run the LM over [text ; nodes] with output_attentions (eager; cheap at ~352 tokens), read the node-block
scores a_ij, and add the edge residual to the node hiddens via per-layer forward hooks — the LM's own output
Σ_j a_ij v_j PLUS U·(Σ_j a_ij E[i,j]), the additive decomposition that needs no custom kernel (§2).

Carry-forward fixes from slotgraph4 / the write-audit (docs §4, §5, §9): error-correcting delta commit (not
raw EMA), content-gated per-edge retention (idle edges freeze), EntNet post-write norm (bounds ‖E‖),
learned salience read selector (‖E‖ is degenerate under unit-norm), write-gate-open init (avoid the E=0∧U=0
double-zero deadlock), per-forward Gaussian slot init, prepend+bidir read (forced in model.py).
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


def _participation_ratio(x: Tensor) -> Tensor:
    """Effective rank (participation ratio) of the rows of x:[N,d] — a 0-dim GPU tensor (no host sync)."""
    if x.shape[0] < 2:
        return x.new_zeros(())
    xc = x.float() - x.float().mean(0, keepdim=True); C = xc.t() @ xc
    return C.diagonal().sum() ** 2 / (C * C).sum().clamp_min(1e-12)


class SlotGraphEncoder(nn.Module):
    is_conditioned_read = False              # READ = PREPEND the node-centric (+ pointer) tokens
    reads_per_layer_kv = False               # relational read needs intra-memory attention → prepend, not KV
    wants_surprise = False

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama, apply_lora_to_llama
        d = cfg.d_llama; self.d = d
        self.N = int(cfg.slotgraph_n_nodes)
        self.de = int(cfg.slotgraph_d_edge)
        self.window = int(cfg.slotgraph_window)
        self.dk = int(cfg.slotgraph_d_key)
        self.read_topk = int(cfg.slotgraph_read_topk)
        self.read_hops = int(cfg.slotgraph_read_hops)
        self.init_noise = bool(cfg.slotgraph_init_noise)
        self.pair_gap = max(1, int(cfg.slotgraph_layer_pair_gap) or 1)     # inter-layer op: l vs l+gap
        N, de = self.N, self.de
        h = d // 2
        if N > h:
            raise ValueError(f"slotgraph needs N ≤ d/2 orthonormal ids (N={N}, d/2={h})")

        # ── shared LM: EAGER attention (harvest needs output_attentions) + a WRITE-side LoRA ──
        # (The READ-side LoRA is the shared FrozenLlamaDecoder adapter in model.py — the "two LoRAs".)
        base, _ = load_frozen_llama(cfg.llama_model, attn_implementation="eager")
        for p in base.parameters():
            p.requires_grad_(False)
        self._write_lora_n = apply_lora_to_llama(
            base, rank=int(cfg.slotgraph_lora_rank), alpha=int(cfg.slotgraph_lora_alpha),
            target_names=tuple(cfg.llama_lora_target_names))
        base.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.base = base
        _bc = base.config
        self.L = _bc.num_hidden_layers
        self.n_heads = _bc.num_attention_heads
        # which layers to harvest: 0 = all; N>0 = last-N (fewer = cheaper, later = more semantic)
        self.write_layers = int(cfg.slotgraph_write_layers)
        embed = base.get_input_embeddings()
        with torch.no_grad():
            mean_vec = embed.weight.float().mean(0); emb_std = embed.weight.float().std().item()
            emb_norm = embed.weight.float().norm(dim=-1).mean().item()

        # ── frozen orthonormal ids (anti-collapse basis / reusable entity keys) + token former ──
        idh = torch.empty(N, h); nn.init.orthogonal_(idh)
        self.register_buffer("id_half", F.normalize(idh, dim=-1), persistent=True)
        self.type_embed = nn.Parameter(torch.empty(2, h))                  # [node, pointer]
        nn.init.orthogonal_(self.type_embed)
        with torch.no_grad():
            self.type_embed.copy_(F.normalize(self.type_embed, dim=-1))
        self.tok_proj = nn.Linear(d + 3 * h, d)                            # [content ‖ id_a ‖ id_b ‖ type] → d

        # ── node slots (free latents; per-forward noise breaks symmetry) ──
        self.node_init = nn.Parameter(mean_vec.view(1, d).repeat(N, 1) + emb_std * torch.randn(N, d))
        if self.init_noise:
            self.node_logsig = nn.Parameter(torch.full((d,), math.log(max(emb_std, 1e-4))))

        # ── edge-feature machinery (harvest a_ij → per-edge vector) ──
        # φ endpoint content feature, FACTORED as φ_i(x_i) + φ_j(x_j) instead of φ([x_i‖x_j]). This is the
        # memory fix: the joint form materializes a [B,N,N,2d] tensor (~254MB at B=2) — the factored form
        # projects the CHEAP [B,N,d] hiddens and only the final [B,N,N,d_e] (7MB) ever exists. The lost
        # cross-endpoint interaction term is already supplied by the a_ij gate + the inter-layer op, so it
        # is a negligible expressiveness change. LN per-endpoint (raw LM hiddens have large layer-varying
        # RMS — un-normalized, the edge feature is driven by hidden magnitude → gnorm explosion).
        self.phi_ln = nn.LayerNorm(d)
        self.phi_i = nn.Linear(d, de)
        self.phi_j = nn.Linear(d, de)
        # learnable inter-layer operator W·[ (e^{l+gap}-e^l) ‖ (e^l ⊙ e^{l+gap}) ‖ e^{l+gap} ] → d_e (fixed W).
        self.op_W = nn.Linear(3 * de, de)
        # value-path injection U: edge aggregate (Σ_j a_ij E[i,j]) → d, added to node hiddens. ZERO-init
        # (ReZero). Applied AFTER the aggregate (linearity: Σ a·U(E) = U(Σ a·E)) so we never lift the
        # per-pair edge to d — the [B,N,N,d] E_lift (~127MB) is gone; only [B,N,d_e] aggregate exists.
        self.U_ln = nn.LayerNorm(de)
        self.U = nn.Linear(de, d, bias=False)
        nn.init.zeros_(self.U.weight)
        # bounded residual output: a per-channel scale (tanh-gated) so the injection into the LM stream
        # stays bounded even as U trains — the loop-gain cap that stops the gnorm feedback blowup. Init
        # SMALL but NONZERO (not zero): with U also zero-init, a zero resid_scale would be a double-zero
        # deadlock (∂L/∂U = ∂L/∂resid_scale = 0 — the design-review P8 trap). At small init the injection
        # is ~0 (U=0 → tanh(0)=0) but U earns gradient from step 1, then resid_scale bounds the magnitude.
        self.resid_scale = nn.Parameter(torch.full((d,), 0.1))
        # ── per-edge gated, error-correcting commit (§4) — gates ALSO factored per-endpoint (memory) ──
        self.gate_a_i = nn.Linear(d, de); self.gate_a_j = nn.Linear(d, de)  # retain α ← φ_ln(x_i)+φ_ln(x_j)
        self.gate_b_i = nn.Linear(d, de); self.gate_b_j = nn.Linear(d, de)  # write  β
        for _g in (self.gate_a_i, self.gate_a_j, self.gate_b_i, self.gate_b_j):
            nn.init.zeros_(_g.weight); nn.init.zeros_(_g.bias)             # start uniform; biases added below
        self.gate_a_bias = nn.Parameter(torch.full((de,), 2.0))           # α≈0.88 retain
        self.gate_b_bias = nn.Parameter(torch.full((de,), 1.5))           # β≈0.82 write-OPEN
        self.rd_k = nn.Linear(de, de, bias=False)                         # read(E) for the error-correcting Δ
        self.edge_norm = _NormMatch(de)                                   # EntNet post-write norm (bounds ‖E‖)
        with torch.no_grad():
            self.edge_norm.scale.data.fill_(1.0)
        # cross-window BPTT truncation: detach E every K window commits (K=1 = per-window state, the design
        # doc's stated propose→commit granularity; K>1 = keep some cross-window credit; 0 = never detach =
        # full 8-deep differentiable recurrence, the gnorm-compounding + memory-accumulating original).
        self.bptt_detach_every = int(getattr(cfg, "slotgraph_bptt_detach_every", 1))

        # ── read heads ──
        self.read_q = nn.Linear(d, self.dk)                               # node-centric pool query
        self.read_k = nn.Linear(d, self.dk)
        self.edge_up = nn.Linear(de, d)                                   # lift edge state → d for the blend/pointers
        self.edge_sal = nn.Linear(de, 1)                                  # learned salience (NOT ‖E‖) for pointer top-k
        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(emb_norm)

        self.M = N + (self.read_topk if self.read_topk > 0 else 0)        # prepend tokens (node-centric + pointers)
        self.force_no_edges = False                                       # eval canary: zero E (edges decorative?)

        hv = f"last-{self.write_layers}" if self.write_layers > 0 else "all"
        print(f"[slotgraph] {N} nodes, NO edge tokens; shared LM + write-LoRA r{cfg.slotgraph_lora_rank} "
              f"({self._write_lora_n} linears); harvest {hv} layers' attention → value-path edge residual "
              f"(d_e={de}, dense {N}×{N}); error-correcting per-edge-gated EntNet-bounded commit; "
              f"prepend+bidir read ({self.M} tokens: {N} node-centric"
              f"{f' + {self.read_topk} pointers' if self.read_topk>0 else ''})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)                                            # base is frozen (LoRA adapts it)
        return self

    # ── streaming interface: accumulate raw embeds; window internally in finalize ──
    def init_streaming_state(self, batch_size, device, dtype):
        return {"emb": torch.zeros(batch_size, 0, self.d, device=device, dtype=dtype),
                "mask": torch.zeros(batch_size, 0, device=device, dtype=torch.bool)}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        return {"emb": torch.cat([state["emb"], token_embeds], dim=1),
                "mask": torch.cat([state["mask"], attention_mask.bool()], dim=1)}, {}

    def _tok(self, content, id_a, id_b, type_idx):
        t = self.type_embed[type_idx].expand(*content.shape[:-1], -1)
        return self.tok_proj(torch.cat([content, id_a, id_b, t], dim=-1))

    def _node_tokens(self, X, B):
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
        return self._tok(X, idh, idh, 0)                                  # node = [X ‖ id ‖ id ‖ node-type]

    def _harvest_window(self, we, wm, X, E, active):
        """ONE LoRA'd LM forward over [text ; node-tokens] with output_attentions. Harvest the per-layer
        node-block attention a_ij, build the within-window inter-layer edge trace S, and inject the value-path
        edge residual U·(Σ_j a_ij E[i,j]) into the node hiddens each layer via forward hooks. Returns the
        final node hiddens (for gates/read) + the scratch trace S:[B,N,N,d_e]."""
        B, T, d = we.shape
        N, de = self.N, self.de
        node_tok = self._node_tokens(X, B)                               # [B,N,d]
        seq = torch.cat([we, node_tok.to(we.dtype)], dim=1)             # [B, T+N, d]
        # idle rows keep ≥1 valid text key (position 0) so an all-pad row can't NaN the softmax.
        wm_enc = wm.clone(); wm_enc[:, 0] |= ~active
        attn_mask = torch.cat([wm_enc, torch.ones(B, N, device=we.device, dtype=torch.bool)], dim=1)

        # per-layer state captured by the hooks. E is kept at d_e (NOT lifted to [B,N,N,d]); U is applied
        # AFTER the a-weighted aggregate (linearity), so the only [N,N,·] tensor is the d_e-wide edge state.
        E_de = self.U_ln(E.reshape(B, N * N, de)).reshape(B, N, N, de)     # [B,N,N,d_e] LN'd edge state (7MB)
        harvest_layers = (set(range(self.L - self.write_layers, self.L)) if self.write_layers > 0
                          else set(range(self.L)))
        store = {"S": we.new_zeros(B, N, N, de), "prev_e": None, "prev_l": None, "last_h": None}
        Toff = T                                                          # node positions start at T

        def _mk_hook(li):
            def hook(module, inp, out):
                hs = out[0] if isinstance(out, tuple) else out            # [B, T+N, d] layer output
                aw = out[1] if (isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1])) else None
                nb = hs[:, Toff:Toff + N]                                 # node hiddens this layer [B,N,d]
                if aw is not None:
                    # node→node attention block (queries=nodes, keys=nodes), mean over heads → a_ij.
                    a_nn = aw[:, :, Toff:Toff + N, Toff:Toff + N].mean(1)  # [B,N,N]
                    # NORMALIZE over the key/node axis: the full softmax was over [text;nodes], so the
                    # sliced node sub-block does NOT sum to 1 (most mass went to text). Renormalizing makes
                    # the residual a proper CONVEX combination of neighbor edge-states — bounded by ‖E‖
                    # (this is both the gnorm fix AND more faithful to "weighted combination of edges").
                    a_nn = a_nn / a_nn.sum(-1, keepdim=True).clamp_min(1e-6)
                    # value-path residual: fold U AFTER the aggregate (Σ a·U(E) = U(Σ a·E)) — never lift
                    # per-pair to d. agg:[B,N,d_e] tiny; resid bounded by tanh + zero-init per-channel scale.
                    agg = torch.einsum("bij,bije->bie", a_nn, E_de)       # [B,N,d_e]
                    resid = torch.tanh(self.U(agg)) * self.resid_scale    # [B,N,d]  (0 at init: ReZero + scale)
                    hs = hs.clone()
                    hs[:, Toff:Toff + N] = nb + resid
                    nb = hs[:, Toff:Toff + N]
                    if li in harvest_layers:
                        # per-edge content feature this layer = a_ij · (φ_i(x_i) + φ_j(x_j)) — FACTORED so
                        # no [B,N,N,2d] intermediate is materialized (only the [B,N,N,d_e] result exists).
                        nbl = self.phi_ln(nb)
                        ei = self.phi_i(nbl).unsqueeze(2)                 # [B,N,1,d_e]
                        ej = self.phi_j(nbl).unsqueeze(1)                 # [B,1,N,d_e]
                        e_l = a_nn.unsqueeze(-1) * (ei + ej)              # [B,N,N,d_e]
                        if store["prev_e"] is not None and (li - store["prev_l"]) >= self.pair_gap:
                            pe = store["prev_e"]
                            feat = self.op_W(torch.cat([e_l - pe, pe * e_l, e_l], dim=-1))  # learnable inter-layer op
                            store["S"] = store["S"] + feat
                            store["prev_e"] = e_l; store["prev_l"] = li
                        elif store["prev_e"] is None:
                            store["prev_e"] = e_l; store["prev_l"] = li
                store["last_h"] = nb
                # SUM-AND-DROP: return None for the attention slot so the full per-layer [B,H,T+N,T+N]
                # matrix is freed immediately (never retained in out.attentions across all L layers — the
                # OOM if kept). We already extracted the small a_nn block above.
                if isinstance(out, tuple):
                    return (hs, None) + tuple(out[2:])
                return hs
            return hook

        handles = [self.base.model.layers[li].self_attn.register_forward_hook(_mk_hook(li))
                   for li in range(self.L)]
        try:
            # NO outer activation-checkpoint here: the forward-HOOKS mutate captured state (store["S"],
            # prev_e) and inject the value residual, so the number of saved tensors differs between the
            # original pass and a checkpoint recompute (CheckpointError). The base LM already has HF
            # per-layer gradient-checkpointing enabled internally, so activation memory stays bounded.
            self.base.model(inputs_embeds=seq, attention_mask=attn_mask.long(),
                            output_attentions=True, use_cache=False)
        finally:
            for hh in handles:
                hh.remove()
        # normalize the trace by the number of harvested layer-PAIRS (S summed one feat per pair) so its
        # scale is invariant to how many layers we harvest — keeps the commit + gnorm well-conditioned.
        n_pairs = max(1, len(harvest_layers) - 1)
        return store["last_h"], store["S"] / n_pairs                       # final node hiddens, mean-scaled trace

    def _commit(self, X, E, node_h, S, active, E0):
        """Error-correcting, per-edge-gated, EntNet-bounded commit (§4).
        ΔE = S − read(E); E ← norm( α⊙E + β⊙ΔE ); idle rows (active=False) freeze.
        Gates FACTORED per-endpoint (α_ij = σ(gate_i(x_i)+gate_j(x_j)+bias)) so no [B,N,N,2d] is formed."""
        B, N, de = E.shape[0], self.N, self.de
        nbl = self.phi_ln(node_h)                                         # [B,N,d] LN'd once
        alpha = torch.sigmoid(self.gate_a_i(nbl).unsqueeze(2) + self.gate_a_j(nbl).unsqueeze(1)
                              + self.gate_a_bias)                          # [B,N,N,de] per-edge retain
        beta = torch.sigmoid(self.gate_b_i(nbl).unsqueeze(2) + self.gate_b_j(nbl).unsqueeze(1)
                             + self.gate_b_bias)                           # per-edge write
        dE = S - self.rd_k(E)                                            # error-correcting (retrieve, write residual)
        E_new = self.edge_norm((alpha * E + beta * dE).reshape(B, N * N, de)).reshape(B, N, N, de)
        if not bool(active.all()):
            E_new = torch.where(active[:, None, None, None], E_new, E0)
        return E_new

    def finalize_memory(self, state):
        emb, mask = state["emb"], state["mask"]
        B, T, d = emb.shape
        if T == 0:
            raise ValueError("slotgraph.finalize_memory: empty context (T=0)")
        N, de = self.N, self.de
        X = self.node_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.init_noise and self.training:
            X = X + self.node_logsig.float().exp() * torch.randn_like(X)
        E = torch.zeros(B, N, N, de, device=emb.device, dtype=torch.float32)   # persistent edge state, init ZERO

        win_metrics = []
        wi = 0
        for w in range(0, T, self.window):
            wm = mask[:, w:w + self.window].bool()
            active = wm.any(dim=1)
            if not bool(active.any()):
                continue
            # TRUNCATED BPTT: detach the INCOMING persistent state every K windows so this window's harvest
            # reads a detached prior E/X — the differentiable recurrence depth is capped (the gnorm-compounding
            # + memory-accumulating fix). Detaching at the START (not end) means THIS window's commit stays
            # fully differentiable into the read; only the cross-window chain is cut. K=1 = per-window state
            # (design-doc propose→commit granularity); gradient still flows RICHLY within the window.
            if self.bptt_detach_every and wi > 0 and (wi % self.bptt_detach_every == 0) and self.training:
                E = E.detach(); X = X.detach()
            wi += 1
            we = emb[:, w:w + self.window]
            E0 = E
            node_h, S = self._harvest_window(we, wm, X, E, active)         # harvest a_ij → trace S (grad)
            with torch.autocast("cuda", enabled=False):
                node_h = node_h.float(); S = S.float()
                E = self._commit(X, E, node_h, S, active, E0)             # error-correcting per-edge commit
                # the node CONTENT also advances: nodes carry the harvested hiddens forward (fast timescale).
                X = torch.where(active[:, None, None], node_h, X)
                win_metrics.append((_participation_ratio(E.reshape(-1, de)).detach(),
                                    E.norm(dim=-1).mean().detach()))

        with torch.autocast("cuda", enabled=False):
            memory, keep_read = self._build_read(X, E, B)
        aux = self._canaries(memory, X, E, win_metrics, emb.device)
        aux["memory_mask"] = keep_read
        return memory, aux

    def _build_read(self, X, E, B):
        """Compress (N nodes + N×N edges) → M prepend tokens. E is an OPERATOR that shapes the read:
          · node-centric: each node ← attention-pool of [X_i ‖ its E-weighted neighbor blend].
          · top-k pointers (optional): strongest edges by a LEARNED salience head (NOT ‖E‖ — degenerate
            under the unit-norm commit), as pure id-pointer tokens."""
        N, de = self.N, self.de
        if self.force_no_edges:
            E = torch.zeros_like(E)
        E_lift = self.edge_up(E.reshape(B, N * N, de)).reshape(B, N, N, self.d)   # [B,N,N,d]
        # graph-conv neighbor blend: node i ← Σ_j softmax_j(read_q(X_i)·read_k(E_lift[i,j])) · E_lift[i,j]
        q = self.read_q(X)                                               # [B,N,dk]
        kk = self.read_k(E_lift)                                         # [B,N,N,dk]
        pa = torch.softmax(torch.einsum("bnd,bnjd->bnj", q, kk) / math.sqrt(self.dk), dim=-1)   # [B,N,N]
        blend = torch.einsum("bnj,bnjd->bnd", pa, E_lift)               # [B,N,d]
        if self.read_hops >= 2:                                          # 2-hop reachability (E² via a second blend)
            kk2 = self.read_k(E_lift)
            pa2 = torch.softmax(torch.einsum("bnd,bnjd->bnj", self.read_q(blend), kk2) / math.sqrt(self.dk), dim=-1)
            blend = blend + torch.einsum("bnj,bnjd->bnd", pa2, E_lift)
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
        node_read = self._tok(X + blend, idh, idh, 0)                    # node-centric token
        toks = [node_read]
        keep = [torch.ones(B, N, dtype=torch.bool, device=X.device)]
        if self.read_topk > 0:
            sal = self.edge_sal(E.reshape(B, N * N, de)).squeeze(-1)     # [B,N*N] LEARNED salience (not ‖E‖)
            topv, topi = sal.topk(min(self.read_topk, N * N), dim=-1)
            gate = torch.sigmoid(topv).unsqueeze(-1)
            src = torch.div(topi, N, rounding_mode="floor"); dst = topi % N
            idh_flat = self.id_half
            sel_lift = torch.gather(E_lift.reshape(B, N * N, self.d), 1, topi.unsqueeze(-1).expand(-1, -1, self.d))
            edge_read = gate * self._tok(sel_lift, idh_flat[src], idh_flat[dst], 1)
            toks.append(edge_read)
            keep.append(torch.ones(B, topi.shape[1], dtype=torch.bool, device=X.device))
        memory = self.norm(torch.cat(toks, dim=1))
        return memory, torch.cat(keep, dim=1)

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)

    @torch.no_grad()
    def _canaries(self, memory, X, E, win_metrics, device):
        B = memory.shape[0]
        E_flat = E.reshape(B, -1, self.de)

        def _within_cos(x):
            S = x.shape[1]
            if S < 2:
                return x.new_zeros(())
            xn = F.normalize(x, dim=-1); cos = xn @ xn.transpose(-1, -2)
            off = cos.sum((-1, -2)) - cos.diagonal(dim1=-2, dim2=-1).sum(-1)
            return (off / (S * (S - 1))).mean()

        def _pr_batched(x):                                              # mean per-example participation ratio
            if x.shape[1] < 2:
                return x.new_zeros(())
            xc = x.float() - x.float().mean(1, keepdim=True)
            C = torch.einsum("bmd,bme->bde", xc, xc)
            tr = C.diagonal(dim1=-2, dim2=-1).sum(-1)
            return (tr ** 2 / (C * C).sum((-1, -2)).clamp_min(1e-12)).mean()

        def _pr_gram_bb(x):                                              # PR across the B rows of x:[B,F]
            # gram over the SMALL (B) axis so F can be huge (E is N²·d_e-dim per example).
            xc = x - x.mean(0, keepdim=True)
            C = xc @ xc.t()                                             # [B,B]
            return C.diagonal().sum() ** 2 / (C * C).sum().clamp_min(1e-12)

        aux = {
            "slotgraph_mem_effrank": _participation_ratio(memory.reshape(-1, self.d)),
            "slotgraph_node_effrank": _participation_ratio(X.reshape(-1, self.d)),
            "slotgraph_edge_effrank": _participation_ratio(E_flat.reshape(-1, self.de)),
            "slotgraph_mem_effrank_perex": _pr_batched(memory),
            "slotgraph_node_cos": _within_cos(X),
            # LEADING canary (input-dependence): cross-example participation ratio of the edge state —
            # does E vary example-to-example (bind) or stay generic (the routing_diversity analog)?
            # Computed via the B×B gram (E flattened per example is N²·d_e-dim — never form the huge gram).
            "slotgraph_edge_inputdep": (_pr_gram_bb(E.reshape(B, -1).float()) if B > 1
                                        else E.new_zeros(())),
            "slotgraph_edge_norm": E.norm(dim=-1).mean(),
        }
        if win_metrics:                                                  # per-window collapse trace (last window)
            aux["slotgraph_edge_effrank_final"] = win_metrics[-1][0]
            aux["slotgraph_edge_norm_final"] = win_metrics[-1][1]
        return aux
