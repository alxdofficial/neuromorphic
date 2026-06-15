"""graph encoder — frozen-backbone observation tap + TokenGT writer + reader.

A FROZEN base contextualizes the span (the ICAE/CCM separate-frozen-copy pattern);
one mid layer is tapped as the OBSERVATION. The TokenGT `writer` cross-attends that
observation + self-attends the graph and snaps endpoints to VQ codes → the graph
state. The custom `reader` is NOT prepended — the MAE loss path installs a forward
hook on a mid-late decoder layer that cross-attends the graph and injects (RMS-
matched, gated) into the frozen residual stream. See docs/graph_model.md.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...config import ReprConfig


class GraphEncoder(nn.Module):
    """VQ-codebook graph memory with a TokenGT controller (the current line).

    Unlike the prepend baselines, the read is an INJECT, not a prepend:
    finalize_memory builds the graph state and returns an EMPTY [B,0,d_llama]
    memory (so the harness prepends nothing); the loss path reaches into
    `self.reader` + the graph via a decoder forward-hook. The graph dict and the
    VQ commitment loss ride in the finalize aux.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        from .substrate import GraphConfig, GraphWriter, GraphReader
        base, _ = load_frozen_llama(cfg.llama_model)
        for p_ in base.parameters():
            p_.requires_grad_(False)
        self.base = base                                   # frozen contextualizer
        gcfg = GraphConfig(
            d_llama=cfg.d_llama, d_graph=cfg.graph_d_graph, n_codes=cfg.graph_n_codes,
            n_edges=cfg.graph_n_edges, write_layers=cfg.graph_write_layers,
            read_layers=cfg.graph_read_layers, heads=cfg.graph_heads,
            ffn_mult=cfg.graph_ffn_mult, vq_decay=cfg.graph_vq_decay,
            vq_commit=cfg.graph_vq_commit)
        self.gcfg = gcfg
        self.writer = GraphWriter(gcfg)
        self.reader = GraphReader(gcfg)
        self.obs_tap_layer = cfg.graph_obs_tap_layer       # observation tap
        self.inject_layer = cfg.graph_inject_layer         # reader inject point
        print(f"[graph] VQ-codebook graph: K={gcfg.n_edges} edges, n_codes={gcfg.n_codes}, "
              f"d_graph={gcfg.d_graph}, write×{gcfg.write_layers}/read×{gcfg.read_layers}, "
              f"obs_tap=L{self.obs_tap_layer} inject=L{self.inject_layer}")

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)                             # base ALWAYS frozen/eval
        return self

    def init_streaming_state(self, batch_size: int, device, dtype):
        del batch_size, dtype
        return {"hiddens": None, "mask": None, "device": device}

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0,
                        **extra):
        del chunk_offset, extra
        B, W = token_embeds.shape[:2]
        if attention_mask is None:
            attention_mask = torch.ones(B, W, device=token_embeds.device, dtype=torch.bool)
        with torch.no_grad():                              # frozen base; controller trains on fixed hiddens
            out = self.base.model(inputs_embeds=token_embeds,
                                  attention_mask=attention_mask.long(),
                                  output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.obs_tap_layer + 1]  # [B,W,d_llama]
        if state["hiddens"] is None:
            state["hiddens"] = h.float()
            state["mask"] = attention_mask.bool()
        else:
            state["hiddens"] = torch.cat([state["hiddens"], h.float()], dim=1)
            state["mask"] = torch.cat([state["mask"], attention_mask.bool()], dim=1)
        return state, {}

    def finalize_memory(self, state):
        """Build the graph; return EMPTY prepend memory + the graph in aux.

        The reader injects via a decoder hook (loss path), so there is no prepend.
        aux carries the graph dict and the VQ commitment loss.
        """
        graph = self.writer(state["hiddens"], state["mask"])
        B = state["hiddens"].shape[0]
        empty = torch.zeros(B, 0, self.cfg.d_llama, device=state["hiddens"].device)
        aux = {"graph": graph, "vq_loss": graph["vq_loss"]}
        return empty, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device,
                                       token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
