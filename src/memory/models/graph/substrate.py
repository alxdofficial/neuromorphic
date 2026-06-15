"""graph model substrate — VQ quantizer, TokenGT writer, custom inject reader.

Design: docs/graph_model.md. The graph's edge ENDPOINTS are discrete VQ codes
(distinct addresses — the fix for the v6/v8/v9 rank-1 read collapse). The TokenGT
writer cross-attends the LLM observation + self-attends the graph (×N) and snaps
node endpoints to codes; the reader cross-attends the graph + causal-self-attends
the decode positions (×M) and injects (RMS-matched, gated) into the frozen LLM.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from vector_quantize_pytorch import VectorQuantize


def _rmsnorm(x: Tensor, eps: float = 1e-6) -> Tensor:
    return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps).to(x.dtype)


@dataclass
class GraphConfig:
    d_llama: int = 576
    d_graph: int = 256
    n_codes: int = 1024
    n_edges: int = 8                 # K — edge budget
    write_layers: int = 3            # N
    read_layers: int = 2             # M
    heads: int = 4
    ffn_mult: int = 4
    vq_decay: float = 0.99           # EMA codebook
    vq_commit: float = 0.25          # commitment loss weight


# VQ = vector-quantize-pytorch's VectorQuantize (EMA codebook + kmeans init + dead-
# code revival — the mature off-the-shelf quantizer; codebook-collapse mitigation
# matters given our history). Built in GraphWriter; returns (quantized, indices, loss).


# ── attention with QK-RMSNorm + learnable temp (the read cold-start fix) ──────
class _Attn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        assert d % heads == 0
        self.h, self.dh = heads, d // heads
        self.q = nn.Linear(d, d, bias=False); self.k = nn.Linear(d, d, bias=False)
        self.v = nn.Linear(d, d, bias=False); self.o = nn.Linear(d, d, bias=False)
        self.log_temp = nn.Parameter(torch.zeros(heads))           # learnable sharpness

    def forward(self, xq: Tensor, xkv: Tensor, kv_mask: Tensor = None,
                causal: bool = False) -> Tensor:
        B, Tq, _ = xq.shape; Tk = xkv.shape[1]
        q = self.q(xq).view(B, Tq, self.h, self.dh).transpose(1, 2)
        k = self.k(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        v = self.v(xkv).view(B, Tk, self.h, self.dh).transpose(1, 2)
        q, k = _rmsnorm(q), _rmsnorm(k)                            # QK-RMSNorm
        temp = self.log_temp.clamp(-3.0, 3.0).exp().view(1, self.h, 1, 1)
        scores = (q @ k.transpose(-1, -2)) * (self.dh ** -0.5) / temp
        if causal:
            cm = torch.triu(torch.ones(Tq, Tk, dtype=torch.bool, device=xq.device), 1)
            scores = scores.masked_fill(cm, float("-inf"))
        if kv_mask is not None:                                    # [B,Tk] True=valid
            scores = scores.masked_fill(~kv_mask[:, None, None, :], float("-inf"))
        a = scores.softmax(-1)
        out = (a @ v).transpose(1, 2).reshape(B, Tq, self.h * self.dh)
        return self.o(out)


class _Block(nn.Module):
    """cross-attend(kv) → self-attend(x, optional causal) → FFN, pre-LN residual."""
    def __init__(self, d: int, heads: int, ffn_mult: int):
        super().__init__()
        self.cn = nn.LayerNorm(d); self.cross = _Attn(d, heads)
        self.sn = nn.LayerNorm(d); self.slf = _Attn(d, heads)
        self.fn = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def forward(self, x: Tensor, kv: Tensor, kv_mask: Tensor = None,
                self_causal: bool = False) -> Tensor:
        x = x + self.cross(self.cn(x), kv, kv_mask=kv_mask)
        xn = self.sn(x)
        x = x + self.slf(xn, xn, causal=self_causal)
        x = x + self.ff(self.fn(x))
        return x


# ── TokenGT writer: builds/updates the graph from the observation ────────────
class GraphWriter(nn.Module):
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        K, d = cfg.n_edges, cfg.d_graph
        self.init_tok = nn.Parameter(torch.randn(3, K, d) / math.sqrt(d))   # src/dst/edge slots
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))          # src/dst/edge role
        self.tag = nn.Parameter(torch.randn(K, d) / math.sqrt(d))           # per-edge instance tag
        self.obs_proj = nn.Linear(cfg.d_llama, d)                           # LLM obs → d_graph (cross-attn KV)
        self.blocks = nn.ModuleList(_Block(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.write_layers))
        self.src_head = nn.Linear(d, d); self.dst_head = nn.Linear(d, d); self.edge_head = nn.Linear(d, d)
        # shared codebook for src+dst endpoints (one vocabulary). kmeans init +
        # dead-code revival (threshold_ema_dead_code) fight codebook collapse.
        self.vq = VectorQuantize(
            dim=d, codebook_size=cfg.n_codes, decay=cfg.vq_decay,
            commitment_weight=cfg.vq_commit, kmeans_init=True, threshold_ema_dead_code=2)

    def forward(self, obs_hiddens: Tensor, obs_mask: Tensor) -> dict:
        B, K, d = obs_hiddens.shape[0], self.cfg.n_edges, self.cfg.d_graph
        kv = self.obs_proj(obs_hiddens.float())                            # [B,L,d]
        base = self.init_tok + self.role[:, None, :] + self.tag[None, :, :]   # [3,K,d]
        x = base.reshape(3 * K, d).unsqueeze(0).expand(B, 3 * K, d).contiguous()
        for blk in self.blocks:
            x = blk(x, kv, kv_mask=obs_mask.bool())
        src_o, dst_o, edge_o = x.view(B, 3, K, d).unbind(1)
        q_src, src_idx, l_s = self.vq(self.src_head(src_o))
        q_dst, dst_idx, l_d = self.vq(self.dst_head(dst_o))
        edge_state = self.edge_head(edge_o)
        return {"src_q": q_src, "dst_q": q_dst, "edge_state": edge_state,
                "src_idx": src_idx, "dst_idx": dst_idx, "vq_loss": l_s + l_d}


# ── custom reader: graph → injected vector for the frozen LLM ─────────────────
class GraphReader(nn.Module):
    def __init__(self, cfg: GraphConfig):
        super().__init__()
        self.cfg = cfg
        d, dl = cfg.d_graph, cfg.d_llama
        self.role = nn.Parameter(torch.randn(3, d) / math.sqrt(d))
        self.tag = nn.Parameter(torch.randn(cfg.n_edges, d) / math.sqrt(d))
        self.q_in = nn.Linear(dl, d)                                       # decode hidden → query
        self.blocks = nn.ModuleList(_Block(d, cfg.heads, cfg.ffn_mult) for _ in range(cfg.read_layers))
        self.out = nn.Linear(d, dl)
        # gate init small-but-NONZERO (tanh(0.1)≈0.1): a 0-init ReZero gate zeros the
        # read output → the reader gets no gradient (the v8c cold-start). ~10% amplitude
        # at init gives the read signal from step 0 (RMS-match keeps it stream-scaled).
        self.gate = nn.Parameter(torch.tensor([0.1]))

    def graph_tokens(self, graph: dict) -> Tensor:
        K = self.cfg.n_edges
        tags = self.tag[None]
        src = graph["src_q"] + self.role[0] + tags
        dst = graph["dst_q"] + self.role[1] + tags
        edg = graph["edge_state"] + self.role[2] + tags
        return torch.stack([src, dst, edg], dim=2).reshape(graph["src_q"].shape[0], 3 * K, self.cfg.d_graph)

    def forward(self, dec_hidden: Tensor, graph: dict) -> Tensor:
        gtok = self.graph_tokens(graph)                                    # [B,3K,d]
        x = self.q_in(dec_hidden.float())                                  # [B,T,d]
        for blk in self.blocks:
            x = blk(x, gtok, self_causal=True)                            # cross graph + causal self
        inj = self.out(x)
        # RMS-match the injection to the residual-stream scale, then gate.
        stream_rms = dec_hidden.float().pow(2).mean(-1, keepdim=True).sqrt()
        inj = _rmsnorm(inj) * stream_rms
        return torch.tanh(self.gate) * inj.to(dec_hidden.dtype)
