"""Trainable gated cross-attention READ for the graph_v7 memory baseline.

graph_v7's default "prepend" read hands its memory tokens to the frozen Llama and
relies on self-attention to use them — empirically it doesn't, so the memory
gradient collapses (grad_norm_memory 56→0.15) and the write never learns. This
module gives Llama a DEDICATED, trainable read path: at a few decoder layers,
each decode token cross-attends to the graph's memory vectors and the result is
gated-added into the residual stream, so gradient flows back to the graph
encoder from step 0.

This is graph-only (asymmetric vs the pooled compressors — accepted).

Install approach: a `register_forward_hook` on the chosen `LlamaDecoderLayer`s.
LlamaDecoderLayer.forward returns a bare hidden-state tensor; the hook returns
`output + gate * cross_attn(output, memory)`, modifying the residual stream
without touching the base layer's weights. Removed in `finally` via
`handle.remove()` so the shared Llama is left unmodified across variants — the
same try/finally hook-handle pattern compute_qa_loss already uses.

Why NOT reuse MemInjectLayer: its source no longer exists in the repo (only a
stale .pyc), and its interface (memory_fn / bridge_hidden, no multi-head
attention / fp32-softmax / guaranteed-nonzero gate) does not match this spec.
Why a hook, not a layer wrapper (à la MTLlamaAttention): we ADD to the layer's
output rather than blending inside attention, so a wrapper would need to
delegate the whole LlamaDecoderLayer surface (input_layernorm, self_attn, mlp,
post_attention_layernorm, …) for no benefit. The hook is minimal and exact.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GatedCrossAttnRead(nn.Module):
    """Standard multi-head cross-attention READ, gated-added to a residual stream.

    Query  = Llama hidden states at the host layer [B, T_dec, d_llama].
    Key/Value = the graph's memory tokens [B, M, d_llama].

        attn = softmax(QKᵀ / √d_head) V         (fp32 softmax, autocast disabled)
        out  = o_proj(attn)                       [B, T_dec, d_llama]
        h    = h + gate * out                     (caller adds; gate is a scalar)

    `gate` is a plain learnable scalar nn.Parameter initialized to
    `gate_init` (NONZERO, e.g. 0.1) so gradient flows from step 0 — it is NOT
    tanh-gated-at-0. Set memory via set_memory(mem) before the Llama forward and
    clear_memory() after (in a finally). With no memory set the module is a true
    no-op (returns 0), so the host layer's output equals vanilla Llama.

    The q/k/v/o projections have NO bias (LoRA-style clean linear maps) and are
    the only trainable params here besides the scalar gate.
    """

    def __init__(
        self,
        d_llama: int,
        inner_dim: int = 512,
        n_heads: int = 8,
        gate_init: float = 0.1,
    ):
        super().__init__()
        if inner_dim % n_heads != 0:
            raise ValueError(
                f"inner_dim ({inner_dim}) must be divisible by n_heads ({n_heads})")
        self.d_llama = int(d_llama)
        self.inner_dim = int(inner_dim)
        self.n_heads = int(n_heads)
        self.head_dim = self.inner_dim // self.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_llama, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_llama, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_llama, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, d_llama, bias=False)
        # Plain learnable scalar gate, NONZERO init (load-bearing — must let
        # gradient flow to the graph from step 0). Not tanh-gated.
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

        # Principled init: 1/√fan_in for q/k/v/o (no bare magic constants).
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))

        # Read-time state (set around the decoder forward).
        self._memory: Optional[Tensor] = None   # [B, M, d_llama] or None (OFF → no-op)

    # ---- memory lifecycle ---------------------------------------------------
    def set_memory(self, memory: Tensor) -> None:
        """memory: [B, M, d_llama] — the graph's finalized memory tokens (K/V source)."""
        self._memory = memory

    def clear_memory(self) -> None:
        self._memory = None

    @property
    def has_memory(self) -> bool:
        return self._memory is not None

    # ---- the cross-attention read -------------------------------------------
    def read(self, hidden: Tensor) -> Tensor:
        """hidden (Q): [B, T_dec, d_llama]. Returns gate*attn_out [B, T_dec, d_llama].

        No-op (returns zeros, no grad path through hidden) when no memory set —
        used by the OFF control so the host layer == vanilla Llama numerically.
        """
        if self._memory is None:
            return hidden.new_zeros(hidden.shape)

        mem = self._memory
        B, T, _ = hidden.shape
        M = mem.shape[1]
        H, hd = self.n_heads, self.head_dim

        # q/k/v projections in the autocast dtype, then split heads.
        q = self.q_proj(hidden).view(B, T, H, hd).transpose(1, 2)   # [B,H,T,hd]
        k = self.k_proj(mem.to(hidden.dtype)).view(B, M, H, hd).transpose(1, 2)  # [B,H,M,hd]
        v = self.v_proj(mem.to(hidden.dtype)).view(B, M, H, hd).transpose(1, 2)  # [B,H,M,hd]

        # fp32 softmax under autocast-disable for numerical stability (repo pattern).
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            scores = torch.einsum("bhtd,bhmd->bhtm", q.float(), k.float()) * self.scale
            attn = F.softmax(scores, dim=-1)                        # [B,H,T,M]
            out = torch.einsum("bhtm,bhmd->bhtd", attn, v.float())  # [B,H,T,hd]
        out = out.to(hidden.dtype).transpose(1, 2).reshape(B, T, self.inner_dim)
        out = self.o_proj(out)                                       # [B,T,d_llama]
        return self.gate.to(out.dtype) * out

    def forward(self, hidden: Tensor) -> Tensor:  # pragma: no cover - read() is the API
        return self.read(hidden)


class GraphCrossAttnReader(nn.Module):
    """Holds one GatedCrossAttnRead per host decoder layer + installs/removes hooks.

    All reads share the SAME memory tokens. set_memory / clear_memory broadcast to
    every read module. install_hooks(llama) registers a forward hook on each chosen
    LlamaDecoderLayer that does `output = output + read(output)`; remove_hooks()
    detaches them (call in a finally so the shared Llama is left clean).
    """

    def __init__(
        self,
        d_llama: int,
        layer_indices,
        inner_dim: int = 512,
        n_heads: int = 8,
        gate_init: float = 0.1,
    ):
        super().__init__()
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.reads = nn.ModuleList([
            GatedCrossAttnRead(d_llama, inner_dim, n_heads, gate_init)
            for _ in self.layer_indices
        ])
        self._handles: list = []

    # ---- memory lifecycle (broadcast to all reads) --------------------------
    def set_memory(self, memory: Tensor) -> None:
        for r in self.reads:
            r.set_memory(memory)

    def clear_memory(self) -> None:
        for r in self.reads:
            r.clear_memory()

    # ---- hook install / remove ----------------------------------------------
    def install_hooks(self, llama_model: nn.Module) -> None:
        """Register a forward hook on each host LlamaDecoderLayer.

        The hook adds `read(layer_output)` to the layer's output hidden state.
        LlamaDecoderLayer.forward returns a bare tensor, so the hook returns a
        bare tensor too (returning a non-None value from a forward hook replaces
        the module's output). Raises on double-install.
        """
        if self._handles:
            raise RuntimeError("GraphCrossAttnReader hooks already installed (call remove_hooks first)")
        layers = llama_model.model.layers
        n = len(layers)
        # Validate ALL indices BEFORE registering any hook. Registering then
        # checking the next index would leak already-registered handles if a
        # later index is out of range (it raises mid-loop, before remove_hooks).
        bad = [i for i in self.layer_indices if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"graph_read_layer_indices entries {bad} out of range for {n}-layer Llama")
        for read_mod, layer_idx in zip(self.reads, self.layer_indices):

            def _hook(module, args, output, _read=read_mod):
                # LlamaDecoderLayer returns a bare hidden-state tensor (recent HF);
                # tolerate a tuple form (older HF) by patching element 0.
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + _read.read(hidden)
                    return (hidden, *output[1:])
                return output + _read.read(output)

            self._handles.append(layers[layer_idx].register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class KVGatedCrossAttnRead(nn.Module):
    """K/V-SPLIT gated cross-attention READ for graph_v8 (the CORRECTED columnar v8 substrate).

    Unlike GatedCrossAttnRead (which projects K and V from the SAME memory
    tensor), this module addresses memory by the substrate's node KEYS and
    returns the node VALUES — true associative K/V semantics:

        Q = q_proj(llama_hidden)        [B,T,d_r]   from d_llama
        K = k_proj(node_keys)           [B,N,d_r]   from d_mem  (passage-refined!)
        V = v_proj(node_values)         [B,N,d_r]   from d_mem
        h = h + gate * o_proj(softmax(QKᵀ/√d_head) V)

    set_memory takes the CONCATENATED [B, N, 2*d_mem] tensor (keys ‖ values along
    the last dim) so the harness's single-tensor REAL/SHUF/OFF gate (roll / skip)
    applies verbatim — rolling the concat rolls both halves together.
    """

    def __init__(self, d_llama: int, d_mem: int, inner_dim: int = 928,
                 n_heads: int = 8, gate_init: float = 0.1):
        super().__init__()
        if inner_dim % n_heads != 0:
            raise ValueError(f"inner_dim ({inner_dim}) must divide n_heads ({n_heads})")
        self.d_llama, self.d_mem = int(d_llama), int(d_mem)
        self.inner_dim, self.n_heads = int(inner_dim), int(n_heads)
        self.head_dim = self.inner_dim // self.n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_llama, inner_dim, bias=False)
        self.k_proj = nn.Linear(d_mem, inner_dim, bias=False)
        self.v_proj = nn.Linear(d_mem, inner_dim, bias=False)
        self.o_proj = nn.Linear(inner_dim, d_llama, bias=False)
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))   # NONZERO init (grad from step 0)
        # QK-RMSNorm + learnable per-head log-temperature: with 1/√fan_in q/k init
        # the raw scores have std ~0.03 → softmax over N=1024 is UNIFORM and stays
        # uniform (no gradient to sharpen) — the measured v8 cold-start failure
        # (attn entropy == log N at step 600). Normalizing q,k to unit RMS per
        # head puts scores at O(1) from step 0; the temp is learnable-but-bounded
        # (same fix as the trajectory router's RMSNorm+SDPA switch).
        self.log_temp = nn.Parameter(torch.zeros(n_heads))
        for lin in (self.q_proj, self.k_proj, self.v_proj, self.o_proj):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        self._memory: Optional[Tensor] = None                      # [B,N,2*d_mem] or None (OFF)

    @staticmethod
    def _rms(x: Tensor) -> Tensor:
        return x * x.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(x.dtype)

    def set_memory(self, memory: Tensor) -> None:
        """memory: [B, N, 2*d_mem] — final layer's (keys ‖ values), concatenated."""
        self._memory = memory

    def clear_memory(self) -> None:
        self._memory = None

    @property
    def has_memory(self) -> bool:
        return self._memory is not None

    def read(self, hidden: Tensor) -> Tensor:
        if self._memory is None:
            return hidden.new_zeros(hidden.shape)
        mem = self._memory
        mem_k, mem_v = mem[..., :self.d_mem], mem[..., self.d_mem:]
        B, T, _ = hidden.shape
        N = mem.shape[1]
        H, hd = self.n_heads, self.head_dim
        q = self._rms(self.q_proj(hidden).view(B, T, H, hd)).transpose(1, 2)             # [B,H,T,hd] unit-RMS
        k = self._rms(self.k_proj(mem_k.to(hidden.dtype)).view(B, N, H, hd)).transpose(1, 2)
        v = self.v_proj(mem_v.to(hidden.dtype)).view(B, N, H, hd).transpose(1, 2)        # [B,H,N,hd]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            temp = self.log_temp.clamp(math.log(0.05), math.log(20.0)).exp().view(1, H, 1, 1)
            scores = torch.einsum("bhtd,bhmd->bhtm", q.float(), k.float()) * (self.scale / temp)
            attn = F.softmax(scores, dim=-1)
            out = torch.einsum("bhtm,bhmd->bhtd", attn, v.float())
        out = out.to(hidden.dtype).transpose(1, 2).reshape(B, T, self.inner_dim)
        return self.gate.to(hidden.dtype) * self.o_proj(out)

    def forward(self, hidden: Tensor) -> Tensor:  # pragma: no cover - read() is the API
        return self.read(hidden)


class GraphV8KVReader(nn.Module):
    """One KVGatedCrossAttnRead per host decoder layer + hook install/remove.

    Same lifecycle/hook pattern as GraphCrossAttnReader (set_memory broadcast,
    install_hooks adds `output + read(output)`, remove_hooks in a finally)."""

    def __init__(self, d_llama: int, d_mem: int, layer_indices,
                 inner_dim: int = 928, n_heads: int = 8, gate_init: float = 0.1):
        super().__init__()
        self.layer_indices = tuple(int(i) for i in layer_indices)
        self.reads = nn.ModuleList([
            KVGatedCrossAttnRead(d_llama, d_mem, inner_dim, n_heads, gate_init)
            for _ in self.layer_indices
        ])
        self._handles: list = []

    def set_memory(self, memory: Tensor) -> None:
        for r in self.reads:
            r.set_memory(memory)

    def clear_memory(self) -> None:
        for r in self.reads:
            r.clear_memory()

    def install_hooks(self, llama_model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("GraphV8KVReader hooks already installed (call remove_hooks first)")
        layers = llama_model.model.layers
        n = len(layers)
        bad = [i for i in self.layer_indices if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"graph_v8_reader_layers entries {bad} out of range for {n}-layer Llama")
        for read_mod, layer_idx in zip(self.reads, self.layer_indices):

            def _hook(module, args, output, _read=read_mod):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + _read.read(hidden)
                    return (hidden, *output[1:])
                return output + _read.read(output)

            self._handles.append(layers[layer_idx].register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class GraphV8SymReader(nn.Module):
    """SYMMETRIC-ADDRESSING read for graph_v8 (layer-matched design, 2026-06-10).

    The read of memory layer ℓ reuses THE WRITE'S OWN ROUTER ℓ-1 — the exact
    function that gated the writes into ℓ (substrate.route: shared proj_in[ℓ-1],
    shared temps, source keys of layer ℓ-1). The same text routes the same way at
    write and read BY PARAMETER IDENTITY, so write-location and read-address
    agree from step 0 (our substitute for Beacon's pretrained attention; kills
    the static-hub shortcut — the addressing function is not free to collapse,
    the write depends on it).

    Read point i (hooked at the MATCHED Llama decoder layer, where router i's
    write-side input came from):
        keys_i  = atoms (i=0)  or  K_i state from memory (i>=1)
        attn    = substrate.route(i, hidden, keys_i)          # [B,T,N], fp32
        fetched = attn @ V_{i+1}                              # values of layer i+1
        hidden += gate_i * o_proj_i(v_proj_i(fetched))
    Trainable here: v_proj/o_proj/gate per point. The addressing path trains too,
    but through the SHARED substrate params (read + write gradients both land on
    proj_in/temps/keys).

    set_memory takes the stacked [B, S, N, 2*d_mem] tensor (cat(K_ℓ, V_ℓ) for
    ℓ=1..S) — rolling it on dim 0 (SHUF) swaps addressing keys AND values
    together, exactly like swapping the whole passage's memory.
    """

    def __init__(self, substrate: nn.Module, layer_indices, inner_dim: int = 416):
        super().__init__()
        self.sub = substrate                               # SHARED routers (do NOT clone)
        self.layer_indices = tuple(int(i) for i in layer_indices)
        if len(self.layer_indices) != substrate.depth:
            raise ValueError(
                f"need one matched Llama layer per memory layer: got {len(self.layer_indices)} "
                f"for depth {substrate.depth}")
        d_mem = substrate.config.d_mem
        self.d_mem = d_mem
        self.v_projs = nn.ModuleList(
            [nn.Linear(d_mem, inner_dim, bias=False) for _ in self.layer_indices])
        self.o_projs = nn.ModuleList(
            [nn.Linear(inner_dim, substrate.config.d_model, bias=False) for _ in self.layer_indices])
        self.gates = nn.Parameter(torch.full((len(self.layer_indices),), 0.1))  # NONZERO init
        for lin in (*self.v_projs, *self.o_projs):
            nn.init.normal_(lin.weight, std=1.0 / math.sqrt(lin.in_features))
        self._memory: Optional[Tensor] = None              # [B,S,N,2*d_mem] or None (OFF)
        self._handles: list = []

    def set_memory(self, memory: Tensor) -> None:
        self._memory = memory

    def clear_memory(self) -> None:
        self._memory = None

    def read(self, point: int, hidden: Tensor) -> Tensor:
        """hidden [B,T,d_llama] at the matched layer of memory layer point+1.

        SAME-LAYER K/V pairing (user decision 2026-06-10): the read of memory
        layer ℓ = point+1 addresses with ITS OWN refined keys K_ℓ and fetches its
        own values V_ℓ — the key written WITH a value is its retrieval handle.
        Router index = point+1 (substrate routers: 0=atoms, ℓ=K_ℓ; router depth
        is read-only — L_depth never writes upward)."""
        if self._memory is None:
            return hidden.new_zeros(hidden.shape)
        mem = self._memory
        d = self.d_mem
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            keys = mem[:, point, :, :d].float()                     # K_{point+1}  [B,N,d]
            attn = self.sub.route(point + 1, hidden.float(), keys)  # the K_ℓ router [B,T,N]
            vals = mem[:, point, :, d:].float()                     # V_{point+1}  [B,N,d]
            fetched = torch.einsum("btn,bnd->btd", attn, vals)
            # RMS-normalize the fetched value (the residual-injection analog of the
            # prepend variants' _NormMatch; precedent GraphV7UnbindReader). Without
            # it the injection sits ~2e4x BELOW the Llama residual stream at init
            # (unit-norm values x uniform-attn averaging x 0.1 gate vs stream RMS
            # ~1.8 — measured) and its scale drifts ~20x as routing sharpens.
            # Direction carries the signal; gate/v/o set the magnitude.
            fetched = fetched * fetched.pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt()
        out = self.o_projs[point](self.v_projs[point](fetched.to(hidden.dtype)))
        return self.gates[point].to(hidden.dtype) * out

    def install_hooks(self, llama_model: nn.Module) -> None:
        if self._handles:
            raise RuntimeError("GraphV8SymReader hooks already installed (call remove_hooks first)")
        layers = llama_model.model.layers
        n = len(layers)
        bad = [i for i in self.layer_indices if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"graph_v8_reader_layers entries {bad} out of range for {n}-layer Llama")
        for point, layer_idx in enumerate(self.layer_indices):

            def _hook(module, args, output, _p=point):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + self.read(_p, hidden)
                    return (hidden, *output[1:])
                return output + self.read(_p, output)

            self._handles.append(layers[layer_idx].register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []


class GraphV7UnbindReader(nn.Module):
    """HRR bind-early / unbind-late READ for the graph_v7 associative-memory fix.

    Replaces the cross-attn read in BIND mode. The memory each decode token reads
    is the per-atom HRR-bound content `M[atom]` (a recoverable superposition of
    key⊛value pairs, in d_val). At each host decoder layer, for hidden state
    h [B, T, d_llama]:

        query_key = key_proj(h)                                   # [B,T,d_val]  (SAME key_proj as write)
        w         = softmax( normalize(route_enc(h)) @ normalize(atoms)ᵀ / route_tau )  # [B,T,Kn]
        recovered = Σ_atom  w[...,atom] · hrr_unbind(M[atom], query_key)               # [B,T,d_val]
        out       = W_recover(recovered)                          # [B,T,d_llama]
        h         = h + gate * out                                # gate scalar, NONZERO init

    The write & read KEYS must match, so key_proj / route_enc / atoms / W_recover
    / route_tau are the SHARED substrate parameters (passed at construction) — NOT
    cloned. The only param this module OWNS is the scalar `gate`. set_memory(M)
    installs the bound content [B, Kn, d_val] before the decoder forward;
    clear_memory() removes it (OFF → true no-op == vanilla Llama).
    """

    def __init__(self, substrate: nn.Module, gate_init: float = 0.1):
        super().__init__()
        # SHARED substrate modules (write & read keys must match — do NOT clone).
        self.sub = substrate
        # Plain learnable scalar gate, NONZERO init (load-bearing).
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))
        self._memory: Optional[Tensor] = None   # [B, Kn, d_val] bound content, or None (OFF)

    # ---- memory lifecycle ---------------------------------------------------
    def set_memory(self, memory: Tensor) -> None:
        """memory: [B, Kn, d_val] — the graph's per-atom HRR-bound content."""
        self._memory = memory

    def clear_memory(self) -> None:
        self._memory = None

    @property
    def has_memory(self) -> bool:
        return self._memory is not None

    # ---- the unbind read ----------------------------------------------------
    def read(self, hidden: Tensor) -> Tensor:
        """hidden (Q): [B, T, d_llama]. Returns gate*out [B, T, d_llama].

        No-op (returns zeros, no grad through hidden) when no memory set (OFF).
        """
        if self._memory is None:
            return hidden.new_zeros(hidden.shape)

        from .graph_substrate_v7 import hrr_unbind

        M = self._memory                                # [B, Kn, d_val]
        B, T, _ = hidden.shape
        Kn = M.shape[1]

        # SAME key_proj as the write (keys must match). RMS-normalize the read
        # input the SAME way the write normalizes its (contextualized) input, so
        # query_key lives in the same regime as the write key_t.
        h_norm = self.sub._norm_input(hidden)
        # Unit-norm the query key to MATCH the write's unit-norm key (HRR clean
        # unbind requires a unit key; the write normalizes key_t the same way).
        query_key = F.normalize(self.sub.key_proj(h_norm), dim=-1)   # [B,T,d_val] unit

        # Routing weights: which atoms to read (reuse the write router + atom bank
        # + learnable route temperature). fp32 for the sharp softmax (repo pattern).
        tq = F.normalize(self.sub.route_enc(h_norm), dim=-1)       # [B,T,d_node]
        ak = F.normalize(self.sub.atoms, dim=-1)                   # [Kn,d_node]
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            logits = (torch.einsum('btd,kd->btk', tq.float(), ak.float())
                      / self.sub._route_tau().float())            # [B,T,Kn]
            w = F.softmax(logits, dim=-1)                          # [B,T,Kn]

        # UNBIND-LATE: recover Σ_atom w·unbind(M[atom], query_key). Broadcast the
        # per-token query_key against every atom's bound content (circular
        # correlation), then route-weight and sum over atoms. All in fp32 (the FFT
        # path is fp32 anyway).
        # M_b: [B,1,Kn,d_val]  query_b: [B,T,1,d_val]  → recovered_per_atom [B,T,Kn,d_val]
        d_val = M.shape[-1]
        Mf = M.float().unsqueeze(1)                               # [B,1,Kn,d_val]
        qf = query_key.float().unsqueeze(2)                       # [B,T,1,d_val]
        recovered_per_atom = hrr_unbind(Mf, qf)                   # [B,T,Kn,d_val] (broadcasts)
        recovered = torch.einsum('btk,btkd->btd', w, recovered_per_atom)  # [B,T,d_val]

        # Control the read INJECTION scale: the HRR unbind magnitude scales with
        # the accumulation depth and is largely passage-agnostic, so let only the
        # DIRECTION of `recovered` (the binding signal) drive the read — RMS-norm
        # it to unit, and let W_recover/gate set the magnitude (principled; no bare
        # scale constant). Prevents a giant agnostic bias from dominating the read.
        recovered = recovered * recovered.float().pow(2).mean(-1, keepdim=True).add(1e-6).rsqrt().to(recovered.dtype)
        out = self.sub.W_recover(recovered.to(hidden.dtype))     # [B,T,d_llama]
        return self.gate.to(out.dtype) * out

    def forward(self, hidden: Tensor) -> Tensor:  # pragma: no cover - read() is the API
        return self.read(hidden)

    # ---- hook install / remove (one hook per host layer, same as cross-attn) -
    def install_hooks(self, llama_model: nn.Module, layer_indices) -> None:
        if getattr(self, "_handles", None):
            raise RuntimeError("GraphV7UnbindReader hooks already installed (call remove_hooks first)")
        self._handles = []
        layers = llama_model.model.layers
        n = len(layers)
        idxs = tuple(int(i) for i in layer_indices)
        bad = [i for i in idxs if i < 0 or i >= n]
        if bad:
            raise ValueError(
                f"graph_read_layer_indices entries {bad} out of range for {n}-layer Llama")
        for layer_idx in idxs:

            def _hook(module, args, output, _self=self):
                if isinstance(output, tuple):
                    hidden = output[0]
                    hidden = hidden + _self.read(hidden)
                    return (hidden, *output[1:])
                return output + _self.read(output)

            self._handles.append(layers[layer_idx].register_forward_hook(_hook))

    def remove_hooks(self) -> None:
        for h in getattr(self, "_handles", []):
            h.remove()
        self._handles = []
