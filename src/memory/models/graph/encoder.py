"""graph encoder — frozen-backbone observation tap + relational parser + reader.

A FROZEN base contextualizes the span (the ICAE/CCM separate-frozen-copy pattern);
one mid layer is tapped as the OBSERVATION. The `GraphParser` (TokenGT-style) parses
it into E edges over a learnable node bank (pointer-select endpoints + edge states).
The custom `GraphReader` is NOT prepended — the MAE loss path installs a forward hook
on a mid-late decoder layer that binds each edge and injects (RMS-matched, gated) into
the frozen residual stream. See docs/graph_model.md (source of truth).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...config import ReprConfig


class GraphEncoder(nn.Module):
    """Relational-parser graph memory over a learnable node bank (the current line).

    The read is an INJECT, not a prepend: finalize_memory builds the graph state and
    returns an EMPTY [B,0,d_llama] memory (the harness prepends nothing); the loss path
    reaches into `self.reader` + the graph via a decoder forward-hook. The graph dict
    (src_value/dst_value/edge_state + pointer diagnostics) rides in the finalize aux.
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        from .substrate import GraphConfig, GraphParser, GraphReader
        base, _ = load_frozen_llama(cfg.llama_model)
        for p_ in base.parameters():
            p_.requires_grad_(False)
        self.base = base                                   # frozen contextualizer
        gcfg = GraphConfig(
            d_llama=cfg.d_llama, d_graph=cfg.graph_d_graph, n_nodes=cfg.graph_n_nodes,
            n_edges=cfg.graph_n_edges, write_layers=cfg.graph_write_layers,
            read_layers=cfg.graph_read_layers, heads=cfg.graph_heads,
            ffn_mult=cfg.graph_ffn_mult, ptr_logit_temp_init=cfg.graph_ptr_logit_temp_init)
        self.gcfg = gcfg
        self.parser = GraphParser(gcfg)
        self.reader = GraphReader(gcfg)
        # Depth guard: the absolute defaults (obs=6, inject=18) are tuned for SmolLM2-135M's
        # 30 layers (0.20 / 0.60 of depth). On a shallower backbone (e.g. Llama-3.2-1B = 16
        # layers, the config default), inject=18 would index past the stack. Keep the
        # configured taps when they fit; otherwise re-derive depth-relative at the same
        # fractions. Invariant: 0 ≤ obs < inject < n_layers (the read needs ≥1 layer after
        # the inject to consume it; the obs tap reads hidden_states[obs+1] ≤ n_layers).
        n_layers = base.config.num_hidden_layers
        obs_tap, inject = cfg.graph_obs_tap_layer, cfg.graph_inject_layer
        if not (0 <= obs_tap < inject < n_layers):
            obs_tap = max(1, round(0.20 * n_layers))
            inject = min(n_layers - 1, max(obs_tap + 1, round(0.60 * n_layers)))
            print(f"[graph] tap/inject ({cfg.graph_obs_tap_layer}/{cfg.graph_inject_layer}) "
                  f"out of range for {n_layers}-layer backbone → depth-relative {obs_tap}/{inject}")
        self.obs_tap_layer = obs_tap                       # observation tap
        self.inject_layer = inject                         # reader inject point
        print(f"[graph] relational parser: N={gcfg.n_nodes} bank, E={gcfg.n_edges} edges, "
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
        with torch.no_grad():                              # frozen base; parser trains on fixed hiddens
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
        """Parse the observation into the graph — WINDOWED + PERSISTENT: process the
        accumulated hiddens in graph_window-token windows, carrying (and updating) the
        graph across them (the parser ingests the prior graph each window). A short
        input (≤ one window — every MAE sentence) runs once from the fresh init slots
        = a single parse. The reader injects via a decoder hook (loss path), no prepend."""
        hiddens, mask = state["hiddens"], state["mask"]          # [B,T,d_llama], [B,T]
        W = self.cfg.graph_window
        T = hiddens.shape[1]
        graph = None
        for s in range(0, T, W):
            e = min(s + W, T)
            win_mask = mask[:, s:e]
            if not win_mask.any():                               # whole-batch padding → skip
                continue
            new = self.parser(hiddens[:, s:e], win_mask, state=graph)
            if graph is None:
                graph = new
            else:
                # PER-EXAMPLE carry: a row with no real token in THIS window keeps its
                # previous graph unchanged (don't update it from padding). The batch-wide
                # skip above only covers all-padding windows; this handles mixed-length
                # batches where some rows have ended. (MAE = one window → never taken.)
                row_has = win_mask.any(dim=-1)                   # [B] bool
                graph = {k: torch.where(row_has[:, None, None], new[k], graph[k])
                         for k in new}
        if graph is None:                                        # fully-padded batch (degenerate) → one parse
            graph = self.parser(hiddens, mask, state=None)       # avoids a None-deref downstream
        B = hiddens.shape[0]
        empty = torch.zeros(B, 0, self.cfg.d_llama, device=hiddens.device)
        return empty, {"graph": graph}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device,
                                       token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
