"""hlvocab encoder: Compression-by-Vocabulary streaming wrapper.

A FROZEN base contextualizes the span; the substrate (substrate.py) compresses
the contextualized hiddens into presence-ranked node-tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from ...config import ReprConfig


class HLVocabEncoder(nn.Module):
    """hlvocab: Compression-by-Vocabulary (hierarchical_learned_vocab) — see its
    header and docs/compression_model_design.md.

    A FROZEN base contextualizes the full span (the ICAE/CCM separate-frozen-copy
    pattern); the substrate compresses the contextualized hiddens into m_max
    presence-ranked node-tokens. finalize_memory returns [B, m_max, d_llama]; the
    masked_reconstruction harness slices to k = ceil(L/ratio) and prepends. v1 = nodes-only
    (prepend compressor, same decode path as the baselines).
    """

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        from .substrate import HLVocabConfig, HLVocabSubstrate
        base, _ = load_frozen_llama(cfg.llama_model)
        for p_ in base.parameters():
            p_.requires_grad_(False)
        self.base = base                                   # frozen contextualizer
        hlv = HLVocabConfig(
            d_model=cfg.d_llama, d_llama=cfg.d_llama, d_code=cfg.hlvocab_d_code,
            nodes=tuple(cfg.hlvocab_nodes), top_k=cfg.hlvocab_top_k,
            m_max=cfg.hlvocab_m_max, effective_k=cfg.hlvocab_effective_k,
            use_graph=cfg.hlvocab_use_graph, edge_topP=cfg.hlvocab_edge_topP,
            edge_cand=cfg.hlvocab_edge_cand, d_sel=cfg.hlvocab_d_sel,
            sel_layers=cfg.hlvocab_sel_layers, sel_heads=cfg.hlvocab_sel_heads,
            d_read=cfg.hlvocab_d_read, reader_layers=cfg.hlvocab_reader_layers,
            reader_heads=cfg.hlvocab_reader_heads)
        self.sub = HLVocabSubstrate(hlv)
        self.M = hlv.m_max
        # norm-match TARGET = the active backbone's embedding scale, NOT the
        # Llama-era 0.9 default (SmolLM2 embeds have norm ~2.68; 0.9 left memory
        # 3x too quiet and the learnable scalar never grew). In-distribution from
        # step 1; learnable from there.
        with torch.no_grad():
            emb_norm = self.base.get_input_embeddings().weight.float().norm(dim=-1).mean()
        self.sub.token_norm.scale.data.fill_(float(emb_norm))
        print(f"[hlvocab] compression-by-vocabulary: nodes={tuple(cfg.hlvocab_nodes)} "
              f"d_code={cfg.hlvocab_d_code} top_k={cfg.hlvocab_top_k} "
              f"m_max={cfg.hlvocab_m_max} tap=L{cfg.hlvocab_tap_layer} "
              f"norm_match_target={float(emb_norm):.2f}")

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
        with torch.no_grad():                              # frozen base; substrate trains on fixed hiddens
            out = self.base.model(inputs_embeds=token_embeds,
                                  attention_mask=attention_mask.long(),
                                  output_hidden_states=True, use_cache=False)
            h = out.hidden_states[self.cfg.hlvocab_tap_layer + 1]   # [B,W,d_llama]
        if state["hiddens"] is None:
            state["hiddens"] = h.float()
            state["mask"] = attention_mask.bool()
        else:
            state["hiddens"] = torch.cat([state["hiddens"], h.float()], dim=1)
            state["mask"] = torch.cat([state["mask"], attention_mask.bool()], dim=1)
        return state, {}

    def finalize_memory(self, state):
        memory, aux = self.sub(state["hiddens"], state["mask"].float())
        return memory, aux

    def forward(self, token_embeds, attention_mask=None, mask_positions=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask)
        return self.finalize_memory(st)
