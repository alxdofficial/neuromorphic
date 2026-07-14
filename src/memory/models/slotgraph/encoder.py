"""slotgraph — THE graph memory (docs/design/slotgraph_design.md).

96 node slots, NO edge tokens. The write runs a frozen SmolLM2-135M — the ENCODER's OWN copy — with a
write-side LoRA, HARVESTING its per-layer attention; the read decodes over the DECODER's frozen copy of the
same LM with the read-side LoRA. (TWO frozen copies, identical weights, SEPARATE LoRAs — ICAE option-A so the
adapters can't collide; collapsing to a single shared base is a possible VRAM optimization, not the current
impl.) Each dense directed pair has TWO persistent states: a unit relation vector R[i,j] (semantic TYPE)
and a scalar confidence C[i,j] in [0,1] (edge existence/strength), both init ZERO and updated from the input.
Their product C·R rides the attention VALUE path as an additive residual:
out_i += U·(Σ_j a_ij·C[i,j]R[i,j]). The within-window write comes from how the LM's per-layer node-block
attention EVOLVES across depth (a learnable inter-layer diff+product feature); R and C commit once per window
with separate data-dependent gates. Read = prepend+bidir; C·R shapes graph messages and edge pointers.

This is a harvest-and-correct realization of Relational Attention (Diao & Loynd 2023) on a frozen+LoRA LM:
run the LM over [text ; nodes] with output_attentions (eager; cheap at ~352 tokens), read the node-block
scores a_ij, and add the edge residual to the node hiddens via per-layer forward hooks — the LM's own output
Σ_j a_ij v_j PLUS U·(Σ_j a_ij C[i,j]R[i,j]), the additive decomposition that needs no custom kernel (§2).

Carry-forward fixes from slotgraph4 / the write-audit (docs §4, §5, §9): a bounded delta-style semantic
commit, content-gated retention, explicit confidence separate from semantic direction, write-gate-open init
(avoid the R=0∧U=0 double-zero deadlock), per-forward Gaussian slot init, and prepend+bidir read (forced in
model.py). This is not a key-addressed DeltaNet update and makes no literal STDP claim.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as _ckpt
from torch import Tensor
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv

from ...common import _NormMatch
from ...config import ReprConfig


def _apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """RoPE on x:[B,H,L,hd] with cos/sin:[B,L,hd] — matches HF apply_rotary_pos_emb (unsqueeze_dim=1).
    Used by the flash-harvest path to re-derive the node-query q/k the fused forward doesn't hand back."""
    cos = cos.unsqueeze(1); sin = sin.unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


def _participation_ratio(x: Tensor) -> Tensor:
    """Effective rank (participation ratio) of the rows of x:[N,d] — a 0-dim GPU tensor (no host sync)."""
    if x.shape[0] < 2:
        return x.new_zeros(())
    xc = x.float() - x.float().mean(0, keepdim=True); C = xc.t() @ xc
    return C.diagonal().sum() ** 2 / (C * C).sum().clamp_min(1e-12)


def _confidence_update(conf: Tensor, observation: Tensor, retain: Tensor, write: Tensor) -> Tensor:
    """Bounded evidence accumulator: decay retained confidence, then fill free capacity from new evidence."""
    retained = retain * conf
    return (retained + (1.0 - retained) * write * observation).clamp(0.0, 1.0)


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
        self.flash_harvest = bool(getattr(cfg, "slotgraph_flash_harvest", True))
        N, de = self.N, self.de
        h = d // 2
        if N > h:
            raise ValueError(f"slotgraph needs N ≤ d/2 orthonormal ids (N={N}, d/2={h})")

        # ── shared LM + a WRITE-side LoRA. flash_harvest → SDPA forward (no [S,S] matrix); the hook
        # recomputes only the [N,S] node-query attention. Else EAGER (returns the full matrix to read). ──
        # (The READ-side LoRA is the shared FrozenLlamaDecoder adapter in model.py — the "two LoRAs".)
        base, _ = load_frozen_llama(cfg.llama_model,
                                    attn_implementation="sdpa" if self.flash_harvest else "eager")
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
        self.n_kv = getattr(_bc, "num_key_value_heads", None) or _bc.num_attention_heads
        self.head_dim = getattr(_bc, "head_dim", None) or (d // self.n_heads)
        # v1 read-mechanism experiment: per-layer-KV read (vs the class default prepend+bidir). Instance
        # attribute shadows the class `reads_per_layer_kv=False` so model.py routes this arm through the
        # shared prefix-cache path (_prefix_kv_forward) when the flag is set.
        self.reads_per_layer_kv = bool(getattr(cfg, "slotgraph_kv_read", False))
        if self.reads_per_layer_kv:
            # Node states run ~14× the embedding scale (|X|≈46 vs ~3.2). The prepend read tames this with
            # `self.norm` after the message pass; the KV read feeds the LM directly, so an un-normalized X
            # blows the tok_proj/node grads to ~1e8 (memoryllm hit the identical "unnormalized→huge KV→
            # gnorm in the millions"). NormMatch the node state to the backbone token scale before the read.
            self.kv_in_norm = _NormMatch(d)
            with torch.no_grad():
                self.kv_in_norm.scale.data.fill_(
                    base.get_input_embeddings().weight.float().norm(dim=-1).mean().item())
        # OPTION B (faithful live read): read = PREPEND node tokens whose node↔node attention is edge-modulated
        # live inside the DECODER's own last-K self-attention (model.py installs the hook + a bidir mem mask,
        # and calls self._edge_resid). The edge state R/C is re-injected fresh from the store at each modulated
        # layer, so the relational structure is never smeared (unlike the KV read's frozen KV or the old
        # prepend's baked-once message-pass). ≈ prepend cost, no materialize pass.
        self.live_read = bool(getattr(cfg, "slotgraph_live_read", False))
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
        if bool(getattr(cfg, "slotgraph_diverse_node_init", False)):
            # DIVERSE on-manifold init: N distinct token embeddings → high-rank, low pairwise cosine, but still
            # on the frozen LM's token manifold (unlike orthogonal noise, which is off-manifold). The nodes are
            # re-attended over themselves ~270 times (30 layers × 8 write + 1 read) — a low-pass smear — so
            # starting UNCOLLAPSED lets the write preserve distinct content instead of having to create it
            # against oversmoothing. Fixed generator → reproducible slot assignment.
            W = embed.weight.detach().float()
            gen = torch.Generator().manual_seed(0)
            idx = torch.randperm(W.shape[0], generator=gen)[:N]
            self.node_init = nn.Parameter(W[idx].clone())
        else:
            self.node_init = nn.Parameter(mean_vec.view(1, d).repeat(N, 1) + emb_std * torch.randn(N, d))
        if self.init_noise:
            self.node_logsig = nn.Parameter(torch.full((d,), math.log(max(emb_std, 1e-4))))
        # per-channel GATED residual for the cross-window node carry (finding-2: gated update, not raw replace)
        # X ← (1-g)⊙X + g⊙node_h; init g≈0.88 (adopt new content early) but retention is learnable per channel.
        self.node_gate = nn.Parameter(torch.full((d,), 2.0))

        # ── edge-feature machinery (harvest a_ij → relation observation + confidence evidence) ──
        # φ endpoint content feature, FACTORED as φ_i(x_i) + φ_j(x_j) instead of φ([x_i‖x_j]). This is the
        # memory fix: the joint form materializes a [B,N,N,2d] tensor (~254MB at B=2) — the factored form
        # projects the CHEAP [B,N,d] hiddens and only the final [B,N,N,d_e] (7MB) ever exists. The lost
        # cross-endpoint interaction term is already supplied by the a_ij gate + the inter-layer op, so it
        # is a negligible expressiveness change. LN per-endpoint (raw LM hiddens have large layer-varying
        # RMS — un-normalized, the edge feature is driven by hidden magnitude → gnorm explosion).
        self.phi_ln = nn.LayerNorm(d)
        self.phi_i = nn.Linear(d, de)
        self.phi_j = nn.Linear(d, de)
        # learnable inter-layer operator W·[ (e^{l+gap}-e^l) ‖ (e^l ⊙ e^{l+gap}) ‖ e^{l+gap} ] → d_e.
        # No bias: a relation proposal must be supported by pair evidence, not a shared all-edge constant.
        self.op_W = nn.Linear(3 * de, de, bias=False)
        # A separate learned evidence head observes the SAME relation feature plus absolute attention
        # statistics. Its sigmoid output is the current confidence observation in [0,1]. Biasing it low makes
        # an unseen edge weak without closing the gradient path; random weights preserve input-dependence.
        self.conf_W = nn.Linear(3 * de, 1)
        self.conf_attn_W = nn.Linear(4, 1, bias=False)                    # [raw_l, raw_prev, mass_l, mass_prev]
        # LEARNED aggregation over the harvested layer-PAIRS: replace the uniform mean (1/n_pairs) of the
        # per-pair relation/confidence proposals with a softmax-weighted (convex) combination, so the model
        # can down-weight noisy layers before the SINGLE error-correcting commit. One logit per layer; init 0
        # → uniform softmax → exactly the previous mean AT INIT (behavior only departs from uniform if it
        # helps, so this is a strictly safe, init-neutral change for both the prepend and KV arms).
        self.layer_agg_logits = nn.Parameter(torch.zeros(_bc.num_hidden_layers))
        nn.init.constant_(self.conf_W.bias, -2.0)                         # initial observation ≈0.12
        # value-path injection U: edge aggregate (Σ_j a_ij C[i,j]R[i,j]) → d, added to node hiddens. Applied
        # after the aggregate, so confidence-scaled pair state is never lifted per edge; the [B,N,N,d] tensor
        # is avoided and only the [B,N,d_e] aggregate reaches U.
        self.U_ln = nn.LayerNorm(de)
        self.U = nn.Linear(de, d, bias=False)
        # NONZERO-init (std = 1/√fan_in), NOT ReZero-zero. The gradient the edges receive is ∂(inject)/∂edge
        # ∝ U, so a ZERO U closes the ONLY door by which the read returns gradient to the edges / harvest
        # heads — the arm can't learn and the edges go inert (the prepend arm hid this via edge_up; the KV
        # read has no such crutch). Init stability does NOT need U=0: the edge confidence C starts at 0 →
        # agg=0 → tanh(U·0)=0 for ANY U, so the injection is EXACTLY 0 at init and the frozen LM runs clean
        # regardless. resid_scale (below) is the magnitude gate; U carries direction + the edges' gradient.
        nn.init.normal_(self.U.weight, std=de ** -0.5)
        # bounded residual output: a per-channel scale (tanh-gated) so the injection into the LM stream
        # stays bounded even as U trains — the loop-gain cap that stops the gnorm feedback blowup. SMALL but
        # NONZERO: ∂(inject)/∂edge ∝ resid_scale·U, so a zero scale would re-close the edge-gradient door
        # (the design-review P8 trap) even with U nonzero. 0.1 keeps the injection small early, then bounds
        # magnitude as the edges grow.
        self.resid_scale = nn.Parameter(torch.full((d,), 0.1))
        # ── relation-vector commit — gates factored per endpoint (memory-efficient) ──
        self.gate_a_i = nn.Linear(d, de); self.gate_a_j = nn.Linear(d, de)  # retain α ← φ_ln(x_i)+φ_ln(x_j)
        self.gate_b_i = nn.Linear(d, de); self.gate_b_j = nn.Linear(d, de)  # write  β
        for _g in (self.gate_a_i, self.gate_a_j, self.gate_b_i, self.gate_b_j):
            nn.init.zeros_(_g.weight); nn.init.zeros_(_g.bias)             # start uniform; biases added below
        self.gate_a_bias = nn.Parameter(torch.full((de,), 2.0))           # α≈0.88 retain
        self.gate_b_bias = nn.Parameter(torch.full((de,), 1.5))           # β≈0.82 write-OPEN
        self.rd_k = nn.Linear(de, de, bias=False)                         # read(R) for the delta-style residual
        nn.init.eye_(self.rd_k.weight)                                    # begin as an interpretable S-R correction
        self.edge_norm = nn.LayerNorm(de)                                # relation direction is unit-normalized below

        # Confidence gets its own scalar, input-dependent retention/write gates. The observation itself is
        # pair-specific and input-dependent; these endpoint gates learn when to preserve or trust it.
        self.conf_gate_a_i = nn.Linear(d, 1); self.conf_gate_a_j = nn.Linear(d, 1)
        self.conf_gate_b_i = nn.Linear(d, 1); self.conf_gate_b_j = nn.Linear(d, 1)
        for _g in (self.conf_gate_a_i, self.conf_gate_a_j, self.conf_gate_b_i, self.conf_gate_b_j):
            nn.init.zeros_(_g.weight); nn.init.zeros_(_g.bias)
        self.conf_gate_a_bias = nn.Parameter(torch.tensor(3.0))          # idle retain ≈0.95
        self.conf_gate_b_bias = nn.Parameter(torch.tensor(1.5))          # write ≈0.82
        self.conf_gate_a_obs = nn.Parameter(torch.tensor(-1.0))          # evidence opens replacement
        self.conf_gate_b_obs = nn.Parameter(torch.tensor(1.0))           # evidence opens writing
        # Cross-window BPTT: K=1 detaches each window, K>1 keeps a bounded span, and 0 keeps the full
        # eight-window differentiable recurrence (the default needed for delayed-retention credit).
        self.bptt_detach_every = int(getattr(cfg, "slotgraph_bptt_detach_every", 0))
        self.inject_harvest_only = bool(getattr(cfg, "slotgraph_inject_harvest_only", False))  # PERF (see config)

        # ── read heads ──
        self.read_q = nn.Linear(d, self.dk)                               # node-centric pool query
        self.read_k = nn.Linear(d, self.dk)                               # edge key (routing by C times R)
        self.read_kx = nn.Linear(d, self.dk)                              # NEIGHBOR-CONTENT key (message passing)
        # Bias-free is load-bearing: C=0 must produce exactly zero edge content for the edge-off control.
        self.edge_up = nn.Linear(de, d, bias=False)                       # lift C·R → d for blend/pointers
        self.edge_sal = nn.Linear(de, 1)                                  # relation salience; log(C) supplies strength
        self.norm = _NormMatch(d)
        with torch.no_grad():
            self.norm.scale.data.fill_(emb_norm)

        self.M = N + (self.read_topk if self.read_topk > 0 else 0)        # prepend tokens (node-centric + pointers)
        self.force_no_edges = False                                       # eval canary: C=0 (content-only graph read)

        hv = f"last-{self.write_layers}" if self.write_layers > 0 else "all"
        print(f"[slotgraph] {N} nodes, NO edge tokens; shared LM + write-LoRA r{cfg.slotgraph_lora_rank} "
              f"({self._write_lora_n} linears); harvest {hv} layers' attention → value-path edge residual "
              f"(d_e={de}, dense {N}×{N}); unit relation + dynamic scalar confidence commit; "
              f"prepend+bidir read ({self.M} tokens: {N} node-centric"
              f"{f' + {self.read_topk} pointers' if self.read_topk>0 else ''})")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)                                            # base is frozen (LoRA adapts it)
        return self

    # ── streaming interface: harvest INCREMENTALLY per window, carrying the persistent graph state R/C/X.
    # The trainer already loops streaming_write per window AND wraps each call in an activation checkpoint
    # (compute_loss), so doing the harvest here (not buffering for finalize) means slotgraph inherits that
    # per-window checkpoint like every other arm → holds ONE window of activations, not all 8. It also makes
    # continuation cheap: a snapshot at boundary b just reads the state already advanced to b (§ finding #4/#2).
    def init_streaming_state(self, batch_size, device, dtype):
        del dtype
        B, N, de = batch_size, self.N, self.de
        X = self.node_init.float().unsqueeze(0).expand(B, -1, -1).contiguous()
        if self.init_noise and self.training:
            X = X + self.node_logsig.float().exp() * torch.randn_like(X)
        return {"X": X,
                "R": torch.zeros(B, N, N, de, device=device, dtype=torch.float32),   # relation dir, init 0
                "C": torch.zeros(B, N, N, 1, device=device, dtype=torch.float32),     # confidence, init 0
                "wi": 0, "seen": False, "win_metrics": []}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0, **extra):
        """Advance R/C/X by this chunk's window(s): harvest → error-correcting commit → gated node carry.
        Usually one window/call (multi-window path; the trainer checkpoints the call). A single-call path
        (MAE) that hands the whole context is sub-windowed here, and each sub-window's harvest is
        checkpointed internally (n_sub>1) since that path has no outer checkpoint — bounds memory either way."""
        del chunk_offset, extra
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device, dtype=torch.bool)
        de = self.de
        X, R, C, wi, seen = state["X"], state["R"], state["C"], state["wi"], state["seen"]
        win_metrics = list(state["win_metrics"])              # copy so continuation snapshots don't alias
        T = token_embeds.shape[1]
        n_sub = (T + self.window - 1) // self.window
        ckpt_sub = (n_sub > 1 and bool(getattr(self.cfg, "grad_checkpoint_stream", True))
                    and self.training and torch.is_grad_enabled())
        for s in range(0, T, self.window):
            we = token_embeds[:, s:s + self.window]
            wm = attention_mask[:, s:s + self.window].bool()
            active = wm.any(dim=1)
            if not bool(active.any()):
                continue
            # TRUNCATED BPTT: detach the INCOMING state every K committed windows (0 = full BPTT). Detaching
            # at the START keeps THIS window differentiable into the read; only the cross-window chain is cut.
            if self.bptt_detach_every and wi > 0 and (wi % self.bptt_detach_every == 0) and self.training:
                R = R.detach(); C = C.detach(); X = X.detach()
            wi += 1; seen = True
            R0, C0 = R, C
            if ckpt_sub:
                node_h, S, observation = _ckpt.checkpoint(
                    self._harvest_window, we, wm, X, R, C, active, use_reentrant=False)
            else:
                node_h, S, observation = self._harvest_window(we, wm, X, R, C, active)
            with torch.autocast("cuda", enabled=False):
                node_h = node_h.float(); S = S.float(); observation = observation.float()
                R, C = self._commit(X, R, C, node_h, S, observation, active, R0, C0)
                # node CONTENT advances via a GATED residual (retain identity, not hard replace); idle freeze.
                g = torch.sigmoid(self.node_gate).to(node_h.dtype)
                X_upd = (1.0 - g) * X + g * node_h
                X = torch.where(active[:, None, None], X_upd, X)
                effective = C * R
                win_metrics.append((_participation_ratio(effective.reshape(-1, de)).detach(),
                                    effective.norm(dim=-1).mean().detach(),
                                    C.mean().detach(), C.std().detach()))
        return {"X": X, "R": R, "C": C, "wi": wi, "seen": seen, "win_metrics": win_metrics}, {}

    def _tok(self, content, id_a, id_b, type_idx):
        t = self.type_embed[type_idx].expand(*content.shape[:-1], -1)
        return self.tok_proj(torch.cat([content, id_a, id_b, t], dim=-1))

    def _node_tokens(self, X, B):
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
        return self._tok(X, idh, idh, 0)                                  # node = [X ‖ id ‖ id ‖ node-type]

    def _harvest_window(self, we, wm, X, R, C, active):
        """ONE LoRA'd LM forward over [text ; node-tokens] with output_attentions. Harvest the per-layer
        node-block attention a_ij, build a relation trace S plus scalar confidence observation O, and inject
        U·(Σ_j a_ij C[i,j]R[i,j]) via forward hooks. Returns final node hiddens, S:[B,N,N,d_e], O:[B,N,N,1]."""
        B, T, d = we.shape
        N, de = self.N, self.de
        S = T + N
        node_tok = self._node_tokens(X, B)                               # [B,N,d]
        seq = torch.cat([we, node_tok.to(we.dtype)], dim=1)             # [B, T+N, d]
        # idle rows keep ≥1 valid text key (position 0) so an all-pad row can't NaN the softmax.
        wm_enc = wm.clone(); wm_enc[:, 0] |= ~active                     # [B,T] valid-text key mask

        # ── DENSE-GRAPH write mask (§ finding-1 fix): plain causal made the node-node block LOWER-TRIANGULAR
        # (node i saw only j≤i; node 0 saw no neighbors; the upper half was dead). Here TEXT stays causal, but
        # every NODE attends to ALL valid text + ALL nodes (dense, bidirectional) → a_ij is a real graph.
        # Additive float mask [B,1,S,S] (0 keep, finfo.min block); nodes never leak into text (text-query→node
        # blocked). ──
        dev = we.device
        idx = torch.arange(S, device=dev)
        is_node_q = (idx >= T)[:, None]                                  # [S,1] query is a node
        is_txt_k = (idx < T)[None, :]                                    # [1,S] key is text
        causal_txt = (idx[:, None] >= idx[None, :])                      # [S,S] lower-tri (text-text causal)
        allow = ((is_node_q & torch.ones(1, S, dtype=torch.bool, device=dev))    # node-query: any text OR node
                 | ((~is_node_q) & is_txt_k & causal_txt))                        # text-query: causal text only
        key_ok = torch.cat([wm_enc, torch.ones(B, N, dtype=torch.bool, device=dev)], dim=1)   # [B,S]
        allow = allow[None] & key_ok[:, None, :]                         # AND key validity → [B,S,S]
        neg = torch.finfo(we.dtype).min
        add_mask = torch.where(allow.unsqueeze(1), we.new_zeros(()), we.new_full((), neg))     # [B,1,S,S]
        # UNIFORM node position: text at 0..T-1, ALL nodes at one position (T) → the node block is a SET, not
        # RoPE-ordered, so node-node attention is permutation-symmetric (order-equivariance).
        pos = torch.cat([torch.arange(T, device=dev), torch.full((N,), T, device=dev)])[None].expand(B, S)

        # Per-layer state captured by the hooks. Normalize relation CONTENT first, then apply C; applying LN
        # after C·R would erase the confidence magnitude. U stays after the aggregate, so no [N,N,d] lift.
        R_de = self.U_ln(R.reshape(B, N * N, de)).reshape(B, N, N, de)
        E_de = C * R_de                                                   # effective edge = confidence × relation
        harvest_layers = (set(range(self.L - self.write_layers, self.L)) if self.write_layers > 0
                          else set(range(self.L)))
        store = {"S": we.new_zeros(B, N, N, de),
                 "O": we.new_zeros(B, N, N, 1),
                 "prev_e": None, "prev_raw": None, "prev_mass": None, "prev_l": None}
        Toff = T                                                          # node positions start at T
        # softmax the per-pair aggregation weights over the PAIR-FORMING layers (simulate the hook's pairing
        # once, up front — deterministic given harvest_layers + pair_gap). init logits 0 → uniform = the old
        # 1/n_pairs mean. agg_w[li] scales the proposal harvested AT the pair whose current layer is li.
        _pl, _pv = None, []
        for _li in sorted(harvest_layers):
            if _pl is not None and (_li - _pl) >= self.pair_gap:
                _pv.append(_li); _pl = _li
            elif _pl is None:
                _pl = _li
        if _pv:
            _w = torch.softmax(self.layer_agg_logits[torch.tensor(_pv, device=dev)], dim=0)
            agg_w = {li: _w[i] for i, li in enumerate(_pv)}
        else:
            agg_w = {}

        def _mk_hook(li):
            def hook(module, args, kwargs, out):
                if self.inject_harvest_only and li not in harvest_layers:
                    # PERF: skip the a_ij recompute + value-path injection on non-harvest layers (deferred to
                    # the harvest layers only). Drop the eager attn matrix if present, pass the stream through.
                    if isinstance(out, tuple):
                        return (out[0], None) + tuple(out[2:])
                    return out
                hs = out[0] if isinstance(out, tuple) else out            # [B, T+N, d] attention output
                nb = hs[:, Toff:Toff + N]                                 # node hiddens this layer [B,N,d]
                aw = out[1] if (isinstance(out, tuple) and len(out) > 1 and torch.is_tensor(out[1])) else None
                if aw is not None:
                    # EAGER path: node→node block straight out of the returned [B,H,S,S] matrix.
                    a_raw = aw[:, :, Toff:Toff + N, Toff:Toff + N].mean(1)  # [B,N,N], full-softmax mass
                else:
                    # FLASH path: the fused forward stored no matrix, so recompute ONLY the node-QUERY
                    # attention [B,H,N,S] from this layer's (LoRA'd) q/k + RoPE + GQA + the SAME mask — exactly
                    # what eager would produce, but dropping the ~T text-query rows we never harvest. a_ij is
                    # identical to eager; the [S,S] matrix is never materialized (the B=8 memory fix).
                    hidden = args[0] if args else kwargs["hidden_states"]   # [B,S,d] post-LN input to attn
                    cos, sin = kwargs["position_embeddings"]
                    hd = module.head_dim; Hkv = module.config.num_key_value_heads
                    qn = module.q_proj(hidden[:, Toff:Toff + N]).view(B, N, self.n_heads, hd).transpose(1, 2)
                    ka = module.k_proj(hidden).view(B, S, Hkv, hd).transpose(1, 2)
                    qn = _apply_rope(qn, cos[:, Toff:Toff + N], sin[:, Toff:Toff + N])
                    ka = repeat_kv(_apply_rope(ka, cos, sin), module.num_key_value_groups)    # [B,H,S,hd]
                    scores = torch.matmul(qn, ka.transpose(2, 3)) * module.scaling            # [B,H,N,S]
                    scores = scores + add_mask[:, :, Toff:Toff + N, :]                        # node-query rows
                    a_full = torch.softmax(scores, dim=-1, dtype=torch.float32).to(qn.dtype)  # [B,H,N,S]
                    a_raw = a_full[:, :, :, Toff:Toff + N].mean(1)                            # [B,N,N] node block
                # ── shared (both paths): renormalize over node keys, inject the value-path residual, harvest.
                # a_nn is a proper convex combination over node keys (the full softmax denominator cancels);
                # node_mass keeps the absolute node-vs-text split as CONFIDENCE evidence.
                node_mass = a_raw.sum(-1, keepdim=True)                   # [B,N,1], absolute node attention
                a_nn = a_raw / node_mass.clamp_min(1e-6)                  # [B,N,N], conditional topology
                agg = torch.einsum("bij,bije->bie", a_nn, E_de)          # [B,N,d_e]
                resid = torch.tanh(self.U(agg)) * self.resid_scale       # [B,N,d]  (0 at init: ReZero + scale)
                # PERF (launch-bound): the node block is the LAST N positions (Toff+N == S), and only it
                # changes — so one `cat` reconstructs the stream instead of clone + scatter-assign + re-slice
                # (3 kernels → 1). Numerically identical; the text prefix [:, :Toff] is unchanged.
                nb = nb + resid
                hs = torch.cat([hs[:, :Toff], nb], dim=1)
                if li in harvest_layers:
                    # per-edge content feature this layer = a_ij · (φ_i(x_i) + φ_j(x_j)) — FACTORED so
                    # no [B,N,N,2d] intermediate is materialized (only the [B,N,N,d_e] result exists).
                    nbl = self.phi_ln(nb)
                    ei = self.phi_i(nbl).unsqueeze(2)                     # [B,N,1,d_e]
                    ej = self.phi_j(nbl).unsqueeze(1)                     # [B,1,N,d_e]
                    e_l = a_nn.unsqueeze(-1) * (ei + ej)                  # [B,N,N,d_e]
                    if store["prev_e"] is not None and (li - store["prev_l"]) >= self.pair_gap:
                        pe = store["prev_e"]
                        pair_feat = torch.cat([e_l - pe, pe * e_l, e_l], dim=-1)
                        feat = self.op_W(pair_feat)                        # semantic relation observation
                        # Confidence sees relation dynamics AND the absolute pair/row attention that the
                        # conditional a_nn normalization intentionally removes.
                        stats = torch.stack([
                            a_raw,
                            store["prev_raw"],
                            node_mass.expand(-1, -1, N),
                            store["prev_mass"].expand(-1, -1, N),
                        ], dim=-1)
                        learned_evidence = torch.sigmoid(self.conf_W(pair_feat) + self.conf_attn_W(stats))
                        # Calibrate against uniform node attention: N*sqrt(raw_l*raw_prev)=1 when all
                        # attention goes to nodes uniformly, <1 when text dominates, capped at 1.
                        mass_evidence = (N * (a_raw * store["prev_raw"]).clamp_min(0).sqrt()).clamp_max(1.0)
                        obs = learned_evidence * mass_evidence.unsqueeze(-1)
                        _wl = agg_w[li].to(feat.dtype)                    # learned per-pair aggregation weight
                        store["S"] = store["S"] + _wl * feat
                        store["O"] = store["O"] + _wl * obs
                        store["prev_e"] = e_l; store["prev_l"] = li
                        store["prev_raw"] = a_raw; store["prev_mass"] = node_mass
                    elif store["prev_e"] is None:
                        store["prev_e"] = e_l; store["prev_l"] = li
                        store["prev_raw"] = a_raw; store["prev_mass"] = node_mass
                # drop the attention slot (eager) so the [B,H,S,S] matrix is freed; flash already has None.
                if isinstance(out, tuple):
                    return (hs, None) + tuple(out[2:])
                return hs
            return hook

        handles = [self.base.model.layers[li].self_attn.register_forward_hook(_mk_hook(li), with_kwargs=True)
                   for li in range(self.L)]
        try:
            # NO outer activation-checkpoint here: the forward-HOOKS mutate captured state (store["S"],
            # prev_e) and inject the value residual, so the number of saved tensors differs between the
            # original pass and a checkpoint recompute (CheckpointError). The base LM already has HF
            # per-layer gradient-checkpointing enabled internally, so activation memory stays bounded.
            out = self.base.model(inputs_embeds=seq, attention_mask=add_mask, position_ids=pos,
                                  output_attentions=not self.flash_harvest, use_cache=False)
        finally:
            for hh in handles:
                hh.remove()
        # S/O are now a CONVEX (softmax) combination over the layer-pairs (weights sum to 1), so they're
        # already scale-invariant to how many layers we harvest — no /n_pairs needed (that was the uniform-mean
        # normalizer this learned weighting replaces).
        # node_h = the FULL final hidden (post residual+MLP+final-norm) for the node block — NOT the attention-
        # branch output the hook sees (finding-2 fix: out[0] of self_attn discards the residual stream + MLP,
        # ~1.7 rel-diff from the true hidden). The value-path injection still rode through every layer, so this
        # final hidden already reflects the edge residual.
        node_h = out.last_hidden_state[:, T:T + N]                        # [B,N,d] true node representation
        return node_h, store["S"], store["O"]

    def _commit(self, X, R, C, node_h, S, observation, active, R0, C0):
        """Commit semantic direction R and confidence C with separate data-dependent gates.

        R' = unit_norm(αr⊙R + βr⊙O⊙(S-read(R)))
        C' = confidence_update(C, O, αc, βc). Inactive examples freeze exactly.
        """
        del X
        B, N, de = R.shape[0], self.N, self.de
        nbl = self.phi_ln(node_h)                                         # [B,N,d] LN'd once
        alpha = torch.sigmoid(self.gate_a_i(nbl).unsqueeze(2) + self.gate_a_j(nbl).unsqueeze(1)
                              + self.gate_a_bias)                          # [B,N,N,de] per-edge retain
        beta = torch.sigmoid(self.gate_b_i(nbl).unsqueeze(2) + self.gate_b_j(nbl).unsqueeze(1)
                             + self.gate_b_bias)                           # per-edge write
        dR = S - self.rd_k(R)                                             # delta-style semantic residual
        raw_R = alpha * R + beta * observation * dR
        R_new = self.edge_norm(raw_R.reshape(B, N * N, de)).reshape(B, N, N, de)
        R_new = F.normalize(R_new, dim=-1)                                # direction only; C owns magnitude

        alpha_c = torch.sigmoid(self.conf_gate_a_i(nbl).unsqueeze(2)
                                + self.conf_gate_a_j(nbl).unsqueeze(1)
                                + self.conf_gate_a_bias
                                + self.conf_gate_a_obs * observation)
        beta_c = torch.sigmoid(self.conf_gate_b_i(nbl).unsqueeze(2)
                               + self.conf_gate_b_j(nbl).unsqueeze(1)
                               + self.conf_gate_b_bias
                               + self.conf_gate_b_obs * observation)
        C_new = _confidence_update(C, observation, alpha_c, beta_c)
        if not bool(active.all()):
            row_active = active[:, None, None, None]
            R_new = torch.where(row_active, R_new, R0)
            C_new = torch.where(row_active, C_new, C0)
        return R_new, C_new

    def _materialize_kv(self, X, R, C, B):
        """PER-LAYER-KV read (v1 experiment): run the final graph nodes through the LM node-block — nodes-only,
        dense node↔node attention, with the SAME per-layer C·R value-path injection the write uses — and
        capture each layer's node k_proj/v_proj as the memory. The edges shape the node KV via the injection
        (they never become tokens); the decoder attends to it passively (no intra-memory attention). Returns
        (K, V), each a length-L list of [B, n_kv, N, head_dim], plus an all-valid memory mask."""
        N, de, L = self.N, self.de, self.L
        if self.force_no_edges:
            C = torch.zeros_like(C)                                      # edge-off ablation → content-only KV
        X = self.kv_in_norm(X)                                           # tame node-state magnitude (|X|≈46 → embed scale)
        seq = self._node_tokens(X, B)                                    # [B,N,d] graph nodes → tokens
        dev, dt = seq.device, seq.dtype
        # nodes-only + DENSE: all-zero additive mask overrides HF's default causal so every node sees every
        # node. UNIFORM position 0 → permutation-symmetric set (matches the write's node block); captured k/v
        # are pre-RoPE (position-free), consistent with the other per-layer-KV arms.
        add_mask = seq.new_zeros(B, 1, N, N)
        pos = torch.zeros(B, N, dtype=torch.long, device=dev)
        R_de = self.U_ln(R.reshape(B, N * N, de)).reshape(B, N, N, de)
        E_de = C * R_de                                                  # effective edge = confidence × relation
        Kbuf, Vbuf = [None] * L, [None] * L

        def _mk_hook(li):
            def hook(module, args, kwargs, out):
                hidden = args[0] if args else kwargs["hidden_states"]    # [B,N,d] post-LN layer input
                cos, sin = kwargs["position_embeddings"]
                hd = self.head_dim
                # Explicit projections of the layer input (pre-RoPE) — these ARE the K/V the decoder attends
                # to. Computed from the residual stream arriving at layer li, so they carry the edge injections
                # of all layers below (depth-specialized), exactly like gisting's per-layer capture.
                k = module.k_proj(hidden).view(B, N, self.n_kv, hd)
                v = module.v_proj(hidden).view(B, N, self.n_kv, hd)
                Kbuf[li] = k.permute(0, 2, 1, 3)                         # [B,n_kv,N,hd]
                Vbuf[li] = v.permute(0, 2, 1, 3)
                # node↔node attention a_nn (all keys are nodes → softmax IS the conditional topology), then
                # inject U·(Σ_j a_nn_ij · E_de_ij) into the node hiddens — the write's value-path edge op.
                q = module.q_proj(hidden).view(B, N, self.n_heads, hd).transpose(1, 2)
                qk = _apply_rope(q, cos, sin)
                kk = repeat_kv(_apply_rope(k.transpose(1, 2), cos, sin), module.num_key_value_groups)
                scores = torch.matmul(qk, kk.transpose(2, 3)) * module.scaling          # [B,H,N,N]
                a_nn = torch.softmax(scores.float(), dim=-1).to(qk.dtype).mean(1)        # [B,N,N]
                agg = torch.einsum("bij,bije->bie", a_nn, E_de)                          # [B,N,d_e]
                resid = torch.tanh(self.U(agg)) * self.resid_scale                       # [B,N,d] (0 at init: ReZero)
                hsout = (out[0] if isinstance(out, tuple) else out) + resid
                return (hsout,) + tuple(out[1:]) if isinstance(out, tuple) else hsout
            return hook

        handles = [self.base.model.layers[li].self_attn.register_forward_hook(_mk_hook(li), with_kwargs=True)
                   for li in range(L)]
        try:
            self.base.model(inputs_embeds=seq, attention_mask=add_mask, position_ids=pos, use_cache=False)
        finally:
            for hh in handles:
                hh.remove()
        mm = torch.ones(B, N, device=dev, dtype=torch.float32)
        return Kbuf, Vbuf, mm

    def _effective_edge(self, R, C):
        """E_de = C · U_ln(R) — the confidence-scaled, LN'd relation used by every injection site. Public so
        the model can precompute it ONCE before the live-read decoder hooks (Option B)."""
        B, N, de = R.shape[0], self.N, self.de
        R_de = self.U_ln(R.reshape(B, N * N, de)).reshape(B, N, N, de)
        return C * R_de

    def _edge_resid(self, module, node_hidden, cos, sin, E_de):
        """The write's value-path edge injection, reusable on ANY llama self-attn `module` — the encoder base
        OR the DECODER (Option B live read). Given a layer's node hiddens [B,N,d], this layer's RoPE cos/sin
        for the node positions, and E_de=[B,N,N,d_e], compute the node↔node attention a_nn and return the
        residual U·(Σ_j a_nn_ij·E_de_ij) [B,N,d] to add to the node hiddens. Identical math to _materialize_kv
        (module carries whichever LoRA it was wrapped with — write-LoRA on the encoder, read-LoRA on decoder)."""
        B, N = node_hidden.shape[:2]; hd = self.head_dim
        q = module.q_proj(node_hidden).view(B, N, self.n_heads, hd).transpose(1, 2)
        k = module.k_proj(node_hidden).view(B, N, self.n_kv, hd).transpose(1, 2)
        qk = _apply_rope(q, cos, sin)
        kk = repeat_kv(_apply_rope(k, cos, sin), module.num_key_value_groups)
        scores = torch.matmul(qk, kk.transpose(2, 3)) * module.scaling          # [B,H,N,N]
        a_nn = torch.softmax(scores.float(), dim=-1).to(qk.dtype).mean(1)        # [B,N,N]
        agg = torch.einsum("bij,bije->bie", a_nn, E_de)                          # [B,N,d_e]
        return torch.tanh(self.U(agg)) * self.resid_scale                       # [B,N,d]

    def finalize_memory(self, state):
        # All the per-window harvest/commit already happened in streaming_write (incremental) — finalize is
        # now just the READ over the final graph state. This is why continuation snapshots are cheap: each
        # boundary's state is already advanced, so finalize is O(read), not a prefix re-encode.
        if not state.get("seen", False):
            raise ValueError("slotgraph.finalize_memory: no windows written (empty context)")
        X, R, C = state["X"], state["R"], state["C"]
        B = X.shape[0]
        if self.live_read:                                               # Option B: prepend nodes, edges LIVE
            if self.force_no_edges:
                C = torch.zeros_like(C)                                   # edge-off ablation → content-only read
            # prepend node tokens (scale-matched like the prepend read), NO edge_up message-pass — the edges
            # ride the DECODER's own last-K self-attention via the injection hook (model.py), using edge_R/C.
            node_tok = self.norm(self._node_tokens(X, B))                # [B,N,d]
            aux = self._canaries(node_tok, X, R, C, state["win_metrics"], X.device)
            aux["memory_mask"] = torch.ones(B, self.N, device=X.device, dtype=torch.float32)
            aux["read_mode"] = "live_inject"
            aux["edge_R"] = R
            aux["edge_C"] = C
            return node_tok, aux
        if self.reads_per_layer_kv:                                       # v1 per-layer-KV read
            K, V, mm = self._materialize_kv(X, R, C, B)
            aux = self._canaries(X, X, R, C, state["win_metrics"], X.device)   # canaries on node states X
            aux["memory_mask"] = mm
            aux["past_kv"] = (K, V)
            aux["read_mode"] = "per_layer_kv"
            empty = torch.zeros(B, 0, self.d, device=X.device, dtype=torch.float32)   # M=0 prepend
            return empty, aux
        with torch.autocast("cuda", enabled=False):
            memory, keep_read = self._build_read(X, R, C, B)
        aux = self._canaries(memory, X, R, C, state["win_metrics"], X.device)
        aux["memory_mask"] = keep_read
        return memory, aux

    def _build_read(self, X, R, C, B):
        """Compress (N nodes + N×N edges) → M prepend tokens via a genuine graph MESSAGE PASS (finding-4 fix):
        node i ← Σ_j A_ij·(X_j + E_lift[i,j]), where E_lift=up(C·R), so confidence controls BOTH routing and
        message strength. A_ij also sees the neighbor's CONTENT X_j (the old read pooled only its own edge
        row — no neighbor content, and its 'two-hop' re-
        pooled the same row). Bringing X_j is additive, so no [B,N,N,2d] tensor is materialized.
          · top-k pointers (optional): relation salience + log confidence, as pure id-pointer tokens."""
        N, de = self.N, self.de
        if self.force_no_edges:
            C = torch.zeros_like(C)                                        # content-only graph-read control
        effective = C * R
        E_lift = self.edge_up(effective.reshape(B, N * N, de)).reshape(B, N, N, self.d)  # [B,N,N,d]
        kE = self.read_k(E_lift)                                         # [B,N,N,dk] edge keys (E_lift fixed → cache)

        def _hop(h):
            # A_ij = softmax_j( q_i·(edge_key_ij + content_key_j) ); message m_ij = h_j + E_lift[i,j].
            # blend_i = Σ_j A_ij h_j (message-passed neighbor content) + Σ_j A_ij E_lift[i,j] (edge blend).
            qh = self.read_q(h)                                          # [B,N,dk]
            s_edge = torch.einsum("bnd,bnjd->bnj", qh, kE)              # route by edge
            s_node = torch.einsum("bnd,bjd->bnj", qh, self.read_kx(h))  # route by NEIGHBOR content
            A = torch.softmax((s_edge + s_node) / math.sqrt(self.dk), dim=-1)     # [B,N,N]
            return (torch.einsum("bnj,bjd->bnd", A, h)                  # Σ_j A_ij X_j  (message pass)
                    + torch.einsum("bnj,bnjd->bnd", A, E_lift))         # Σ_j A_ij E_lift[i,j]

        blend = _hop(X)                                                 # 1-hop message pass
        if self.read_hops >= 2:                                          # TRUE 2-hop: reuse hop-1 node states
            blend = blend + _hop(X + blend)                            # info flows i←j←k through effective edges
        idh = self.id_half.unsqueeze(0).expand(B, -1, -1)
        node_read = self._tok(X + blend, idh, idh, 0)                    # node-centric token
        toks = [node_read]
        keep = [torch.ones(B, N, dtype=torch.bool, device=X.device)]
        if self.read_topk > 0:
            rel_sal = self.edge_sal(R.reshape(B, N * N, de)).squeeze(-1)
            conf_log = C.reshape(B, N * N).clamp_min(1e-6).log()
            sal = rel_sal + conf_log                                      # semantic importance + edge strength
            topv, topi = sal.topk(min(self.read_topk, N * N), dim=-1)
            gate = torch.sigmoid(topv).unsqueeze(-1)
            recv = torch.div(topi, N, rounding_mode="floor"); send = topi % N
            idh_flat = self.id_half
            sel_lift = torch.gather(E_lift.reshape(B, N * N, self.d), 1, topi.unsqueeze(-1).expand(-1, -1, self.d))
            edge_read = gate * self._tok(sel_lift, idh_flat[send], idh_flat[recv], 1)  # source j → receiver i
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
    def _canaries(self, memory, X, R, C, win_metrics, device):
        B = memory.shape[0]
        effective = C * R
        E_flat = effective.reshape(B, -1, self.de)
        R_flat = R.reshape(B, -1, self.de)

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
            # Gram over the small B axis so F can be huge (effective edges are N²·d_e per example).
            xc = x - x.mean(0, keepdim=True)
            C = xc @ xc.t()                                             # [B,B]
            return C.diagonal().sum() ** 2 / (C * C).sum().clamp_min(1e-12)

        aux = {
            "slotgraph_mem_effrank": _participation_ratio(memory.reshape(-1, self.d)),
            "slotgraph_node_effrank": _participation_ratio(X.reshape(-1, self.d)),
            "slotgraph_edge_effrank": _participation_ratio(E_flat.reshape(-1, self.de)),
            "slotgraph_relation_effrank": _participation_ratio(R_flat.reshape(-1, self.de)),
            "slotgraph_mem_effrank_perex": _pr_batched(memory),
            "slotgraph_node_cos": _within_cos(X),
            # Leading canaries separate semantic, confidence, and effective-edge input dependence.
            "slotgraph_edge_inputdep": (_pr_gram_bb(effective.reshape(B, -1).float()) if B > 1
                                        else R.new_zeros(())),
            "slotgraph_relation_inputdep": (_pr_gram_bb(R.reshape(B, -1).float()) if B > 1
                                            else R.new_zeros(())),
            "slotgraph_conf_inputdep": (_pr_gram_bb(C.reshape(B, -1).float()) if B > 1
                                        else C.new_zeros(())),
            "slotgraph_edge_norm": effective.norm(dim=-1).mean(),
            "slotgraph_relation_norm": R.norm(dim=-1).mean(),
            "slotgraph_conf_mean": C.mean(),
            "slotgraph_conf_std": C.std(),
            "slotgraph_conf_active_frac": (C > 0.5).float().mean(),
        }
        if win_metrics:                                                  # per-window collapse trace (last window)
            aux["slotgraph_edge_effrank_final"] = win_metrics[-1][0]
            aux["slotgraph_edge_norm_final"] = win_metrics[-1][1]
            aux["slotgraph_conf_mean_final"] = win_metrics[-1][2]
            aux["slotgraph_conf_std_final"] = win_metrics[-1][3]
        return aux
