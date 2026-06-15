"""soft_pointer_graph encoder: soft-pointer graph memory with a no-op-free,
per-token read (docs/graph_v6.md). Substrate primitives live in substrate.py.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...common import _NormMatch
from ...config import ReprConfig


class SoftPointerGraphEncoder(nn.Module):
    """soft_pointer_graph: soft-pointer graph memory with a no-op-free, per-token read.

    See docs/graph_v6.md + src/memory/models/soft_pointer_graph/substrate.py.

    WRITE: chunk-fresh (mu,sigma) node bank + soft-pointer edges, updated per window
      by a unified typed-token transformer (SoftPointerGraphUpdater) with a per-token
      FFN readout + anchor gate. No proposal pool, no competitive write head.
    READ: finalize_memory builds per-edge FACT-TOKENS (directional FiLM-by-state of
      materialized endpoints) and prepends them as [B, K_edge, d_llama] memory.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        self.K_node = cfg.spg_K_node
        self.K_edge = cfg.spg_K_edge
        self.d_node = cfg.spg_d_node
        self.d_state = cfg.spg_d_state
        d_read = cfg.spg_d_read

        from .substrate import (
            SoftPointer,
            SoftPointerGraphUpdater, SoftPointerGraphGate,
            SoftPointerGraphFactBuilder,
        )

        d_up = cfg.spg_d_updater
        self.pin_encoder = nn.Sequential(
            nn.Linear(cfg.d_llama, d_up * 2, bias=False),
            nn.GELU(),
            nn.Linear(d_up * 2, d_up),
            nn.LayerNorm(d_up),
        )
        # chunk-fresh init params (no per-slot trained params)
        s = float(cfg.spg_init_log_sigma)
        self.mu_node = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_node = nn.Parameter(torch.full((self.d_node,), s))
        self.mu_state = nn.Parameter(torch.zeros(self.d_state))
        self.log_sigma_state = nn.Parameter(torch.full((self.d_state,), s))
        self.mu_q = nn.Parameter(torch.zeros(self.d_node))
        self.log_sigma_q = nn.Parameter(torch.full((self.d_node,), s))

        self.updater = SoftPointerGraphUpdater(
            d_node=self.d_node, d_state=self.d_state, d=d_up,
            K_node=self.K_node, K_edge=self.K_edge, d_pin=d_up,
            n_layers=cfg.spg_updater_layers, n_heads=cfg.spg_updater_heads,
        )
        self.node_gate = SoftPointerGraphGate(self.d_node, hidden=64,
                                              init_bias=cfg.spg_node_gate_init_bias)
        self.edge_gate = SoftPointerGraphGate(2 * self.d_node + self.d_state, hidden=64,
                                              init_bias=cfg.spg_edge_gate_init_bias)
        self.read_pointer = SoftPointer(
            d_node=self.d_node, init_temperature=float(cfg.spg_read_temperature),
            kv_split=True,
        )
        self.fact_builder = SoftPointerGraphFactBuilder(
            d_node=self.d_node, d_state=self.d_state, d_read=d_read,
            film_hidden=cfg.spg_film_hidden, mlp_hidden=cfg.spg_builder_mlp_hidden,
        )
        # PREPEND read (parity with every other arm): project the d_read fact tokens
        # to d_llama and prepend; the frozen decoder's own attention does the reading.
        # (The retired per-position cross-attention reader lived here — it left most
        # of its params gradient-dead in the prepend regime.)
        _rm = cfg.spg_read_ffn_mult * d_read
        self.prepend_proj = nn.Sequential(
            nn.Linear(d_read, _rm), nn.GELU(), nn.Linear(_rm, cfg.d_llama))
        # EMAT prepend read: norm-match the projected fact-tokens to Llama's token
        # scale (same as the baselines) so the graph reads through the IDENTICAL
        # prepend interface — no privileged per-position inject hook.
        self.prepend_norm = _NormMatch(cfg.d_llama)

    # ── Streaming interface ──────────────────────────────────────────────
    def init_streaming_state(self, batch_size, device, dtype, seed=None):
        from .substrate import init_soft_pointer_graph_state
        w_dtype = next(self.pin_encoder.parameters()).dtype
        gen = None
        # Deterministic-eval guard: the node/edge/q init noise is symmetry-
        # breaking randomness that SHOULD vary per step during training, but at
        # eval it must be FIXED so metrics are reproducible run-to-run (matches
        # the continuous/MT deterministic-eval-noise convention, v5.4 fix #2).
        # Without this, soft_pointer_graph eval drew from the global RNG (gen=None)
        # and graph init changed between eval runs.
        if seed is None and not self.training:
            seed = 1234
        if seed is not None:
            gen = torch.Generator(device=device)
            gen.manual_seed(int(seed))
        state = init_soft_pointer_graph_state(
            B=batch_size, K_node=self.K_node, K_edge=self.K_edge,
            d_node=self.d_node, d_state=self.d_state,
            mu_node=self.mu_node, log_sigma_node=self.log_sigma_node,
            mu_state=self.mu_state, log_sigma_state=self.log_sigma_state,
            mu_q=self.mu_q, log_sigma_q=self.log_sigma_q,
            device=device, dtype=w_dtype, generator=gen,
        )
        zero = torch.zeros((), device=device, dtype=torch.float32)
        return {**state, "n_windows": 0,
                "node_gate_mean_accum": zero.clone(),
                "edge_gate_mean_accum": zero.clone()}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0):
        from .substrate import _rmsnorm, _sinusoidal_pe
        w_dtype = next(self.pin_encoder.parameters()).dtype
        token_embeds = token_embeds.to(w_dtype)
        if attention_mask is not None and not attention_mask.any():
            return state, {}

        pins = self.pin_encoder(token_embeds)
        T_w = pins.shape[1]
        pe = _sinusoidal_pe(T_w, pins.shape[-1], offset=chunk_offset,
                            device=pins.device, dtype=pins.dtype)
        pins = pins + pe.unsqueeze(0)

        if attention_mask is not None:
            pins_pad_mask = ~attention_mask
            all_pad_rows = pins_pad_mask.all(dim=-1)
            if all_pad_rows.any():
                pins_pad_mask = pins_pad_mask.clone()
                pins_pad_mask[all_pad_rows, 0] = False
            if not pins_pad_mask.any():
                pins_pad_mask = None
            has_real = attention_mask.any(dim=-1)
        else:
            pins_pad_mask = None
            has_real = None

        N_old, q_src_old, q_dst_old, st_old = (
            state["N"], state["q_src"], state["q_dst"], state["state"])
        tgt = self.updater(pins, pins_pad_mask, N_old, q_src_old, q_dst_old, st_old)

        # gate node bank (target = FFN readout)
        g_node = self.node_gate(N_old, tgt["N"])
        N_new = _rmsnorm(N_old + g_node.unsqueeze(-1) * (tgt["N"] - N_old))

        # gate edges (one anchor gate over the concatenated edge fields)
        edge_old = torch.cat([q_src_old, q_dst_old, st_old], dim=-1)
        edge_tgt = torch.cat([tgt["q_src"], tgt["q_dst"], tgt["state"]], dim=-1)
        g_edge = self.edge_gate(edge_old, edge_tgt)                # [B, K_edge]
        ge = g_edge.unsqueeze(-1)
        q_src_new = _rmsnorm(q_src_old + ge * (tgt["q_src"] - q_src_old))
        q_dst_new = _rmsnorm(q_dst_old + ge * (tgt["q_dst"] - q_dst_old))
        st_new = _rmsnorm(st_old + ge * (tgt["state"] - st_old))

        if has_real is not None:
            km = has_real.to(w_dtype).view(-1, 1, 1)
            N_new = N_new * km + N_old * (1 - km)
            q_src_new = q_src_new * km + q_src_old * (1 - km)
            q_dst_new = q_dst_new * km + q_dst_old * (1 - km)
            st_new = st_new * km + st_old * (1 - km)

        with torch.no_grad():
            ngm = g_node.float().mean().to(torch.float32)
            egm = g_edge.float().mean().to(torch.float32)
        new_state = dict(state)
        new_state.update(
            N=N_new, q_src=q_src_new, q_dst=q_dst_new, state=st_new,
            n_windows=state["n_windows"] + 1,
            node_gate_mean_accum=state["node_gate_mean_accum"] + ngm,
            edge_gate_mean_accum=state["edge_gate_mean_accum"] + egm,
        )
        return new_state, {"spg_node_gate_mean": ngm, "spg_edge_gate_mean": egm}

    def _build_facts(self, state, zero_state=False):
        """READ Stage A: materialize endpoints + directional FiLM-by-state fact tokens."""
        N, q_src, q_dst, st = state["N"], state["q_src"], state["q_dst"], state["state"]
        sp_k, sp_v = self.read_pointer.project_kv(N)
        src_ep, _ = self.read_pointer.attend(q_src, sp_k, sp_v)
        dst_ep, _ = self.read_pointer.attend(q_dst, sp_k, sp_v)
        return self.fact_builder(src_ep, dst_ep, st, zero_state=zero_state)

    def finalize_memory(self, state):
        N = state["N"]
        B, device = N.shape[0], N.device
        dtype = next(self.prepend_proj.parameters()).dtype
        zero_state = bool(state.get("zero_state", False))   # finer ablation than zero_memory
        fact_value = self._build_facts(state, zero_state=zero_state)
        # PREPEND read (EMAT-matched, identical to the baselines): the K_edge
        # fact-tokens projected to d_llama + norm-matched, returned as
        # [B, K_edge, d_llama] prepend memory. The old per-position inject hook
        # is retired (model.py) so the graph reads like every other arm and
        # REAL/SHUF/OFF apply to it too. spg_facts stays in aux for telemetry.
        memory = self.prepend_norm(self.prepend_proj(fact_value).float())
        n_w = max(state["n_windows"], 1)
        aux = {
            "load_balance_loss": torch.zeros((), device=device, dtype=dtype),
            # graph_aux=0 (no aux loss) — also flips compute_loss's `graph_aux is not
            # None` guard True so the spg_* telemetry pass-through actually fires.
            "graph_aux": torch.zeros((), device=device, dtype=dtype),
            "spg_facts": {"value": fact_value},
            "spg_node_gate_mean_avg": (state["node_gate_mean_accum"] / n_w).to(torch.float32),
            "spg_edge_gate_mean_avg": (state["edge_gate_mean_accum"] / n_w).to(torch.float32),
            "spg_fact_norm": fact_value.detach().float().norm(dim=-1).mean().to(torch.float32),
        }
        # ── Eval-only health telemetry (no grad, zero train-time cost) ────────
        # Diagnose whether the v6 mechanism is alive: dead read (rezero eff≈0),
        # ignored edge-state (state_effect≈0 violates no-op-free), collapsed node
        # bank (collapse_cos→1), degenerate soft-pointer read (entropy→0 over-sharp
        # or →log K diffuse; active_frac→0 = hub collapse).
        if not self.training:
            with torch.no_grad():
                fact_zero = self._build_facts(state, zero_state=True)
                aux["spg_state_effect"] = (
                    (fact_value - fact_zero).float().norm(dim=-1).mean().to(torch.float32))
                Nf = N.float()
                Nf = Nf / Nf.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                cos = Nf @ Nf.transpose(1, 2)                          # [B, Kn, Kn]
                Kn = N.shape[1]
                offmask = ~torch.eye(Kn, dtype=torch.bool, device=N.device)
                aux["spg_node_collapse_cos"] = cos[:, offmask].mean().to(torch.float32)
                sp_k, sp_v = self.read_pointer.project_kv(N)
                _, a_src = self.read_pointer.attend(state["q_src"], sp_k, sp_v)
                _, a_dst = self.read_pointer.attend(state["q_dst"], sp_k, sp_v)
                def _ent(a):                                           # mean read-pointer entropy
                    p = a.float().clamp_min(1e-9)
                    return (-(p * p.log()).sum(-1)).mean()
                aux["spg_read_src_entropy"] = _ent(a_src).to(torch.float32)
                aux["spg_read_dst_entropy"] = _ent(a_dst).to(torch.float32)
                picks = torch.cat([a_src.argmax(-1), a_dst.argmax(-1)], dim=1)  # [B, 2*K_edge]
                active = torch.zeros(N.shape[0], Kn, dtype=torch.bool, device=N.device)
                active.scatter_(1, picks, True)
                aux["spg_node_active_frac"] = active.float().mean().to(torch.float32)
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        """Non-streaming fallback. soft_pointer_graph reads via the PREPEND path only
        (streaming_write + finalize_memory); the retired per-position inject hook was
        removed. Here we return a prepend projection of the fact-tokens."""
        del mask_positions
        B, device, dtype = token_embeds.shape[0], token_embeds.device, token_embeds.dtype
        state = self.init_streaming_state(B, device, dtype)
        state, _ = self.streaming_write(state, token_embeds, attention_mask)
        fact_value = self._build_facts(state)
        memory = self.prepend_proj(fact_value).to(dtype)          # [B, K_edge, d_llama] prepend read
        aux = {"load_balance_loss": torch.zeros((), device=device, dtype=memory.dtype)}
        return memory, aux
