"""The KV-graph encoder — the harness entry point.

Per window: run the frozen LM once, parse the window's text, group tokens into particulars, pool their KV,
admit them into one persistent graph, contract to budget, mix, and inject nodes as per-layer cache entries.

Trainable: the mixer, the pooling weight, and the injection projection. The backbone is frozen and the
parser is preprocessing, so the gradient path is short — loss -> frozen decoder -> injected KV -> injector ->
mixer -> stop. No write gate to collapse and no gradient through the parser, which removes most of the ways
this could go unstable.

**One training stage.** There is no read-all warmup: a mixer trained on read-all is free to smear one fact
across co-dependent nodes, which is fine when everything is present and catastrophic under subset retrieval,
so a warmup would spend its whole duration installing the habit `node_dropout` exists to prevent. The
curriculum comes from `set_pressure()` annealing storage and read budgets together, with dropout on from
step 0.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ...config import ReprConfig
from .align import assert_alignment, token_char_offsets
from .build import build_graph
from .ground import KVCapture, pool_hidden, pool_nodes
from .inject import Injector
from .merge import PersistentGraph
from .mixer import GraphMixer


class KVGraphEncoder(nn.Module):
    reads_per_layer_kv = True
    is_conditioned_read = False
    wants_surprise = False

    def __init__(self, cfg: ReprConfig):
        super().__init__()
        self.cfg = cfg
        from ...decoder import load_frozen_llama
        base, tok = load_frozen_llama(cfg.llama_model)
        for p in base.parameters():
            p.requires_grad_(False)
        self.base, self.tok = base, tok
        bc = base.config
        self.L = bc.num_hidden_layers
        self.n_kv = getattr(bc, "num_key_value_heads", None) or bc.num_attention_heads
        self.head_dim = getattr(bc, "head_dim", None) or (bc.hidden_size // bc.num_attention_heads)
        self.d = cfg.d_llama

        g = lambda k, v: getattr(cfg, k, v)  # noqa: E731 - terse config access, all defaults documented below
        self.storage_budget = int(g("kvg_storage_budget", 96))
        self.read_budget = int(g("kvg_read_budget", 32))
        self.n_sinks = int(g("kvg_n_sinks", 4))
        self.recent_windows = int(g("kvg_recent_windows", 2))
        self.node_dropout = float(g("kvg_node_dropout", 0.15))
        self.ppr_alpha = float(g("kvg_ppr_alpha", 0.15))
        self.head_weight = float(g("kvg_head_weight", 2.0))
        # Mid-stack read: entity/coreference information peaks in the middle of the stack, and the
        # in-forward retrieval literature independently lands on ~2/3 depth.
        # 0 in the config means "pick 2L/3": entity/coref information peaks mid-stack.
        self.summary_layer = int(g("kvg_summary_layer", 0)) or max(1, int(self.L * 2 / 3))
        self.ablate_graph = bool(g("kvg_ablate_graph", False))
        self.ablate_relations = bool(g("kvg_ablate_relations", False))
        self.ablate_edge_vec = bool(g("kvg_ablate_edge_vec", False))
        self.pressure = 1.0                                   # 0 = no budget pressure, 1 = terminal budgets

        self.mixer = GraphMixer(self.d, d_mix=int(g("kvg_d_mix", 384)),
                                n_layers=int(g("kvg_mixer_layers", 4)),
                                n_heads=int(g("kvg_mixer_heads", 6)),
                                rank=int(g("kvg_operator_rank", 8)),
                                d_id=int(g("kvg_d_id", 64)))
        self.injector = Injector(self.mixer.d_mix, self.L, self.n_kv, self.head_dim,
                                 rope_mode=str(g("kvg_rope_mode", "compact")),
                                 norm_match=bool(g("kvg_norm_match", True)),
                                 adapter_rank=int(g("kvg_adapter_rank", 24)))
        self.capture = KVCapture(base)
        self._parser = None                                   # lazy: spaCy load is slow and not always needed

        n_train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[kvgraph] storage={self.storage_budget} read={self.read_budget} sinks={self.n_sinks} "
              f"d_mix={self.mixer.d_mix} rope={self.injector.rope_mode} "
              f"summary_layer={self.summary_layer}/{self.L} | trainable={n_train/1e6:.2f}M")

    # ------------------------------------------------------------------ plumbing

    @property
    def parser(self):
        if self._parser is None:
            from .parse import Parser
            self._parser = Parser(spacy_model=getattr(self.cfg, "kvg_spacy_model", "en_core_web_sm"),
                                  use_coref=getattr(self.cfg, "kvg_use_coref", True),
                                  require_coref=getattr(self.cfg, "kvg_require_coref", False))
        return self._parser

    def train(self, mode: bool = True):
        super().train(mode)
        self.base.train(False)
        return self

    def set_pressure(self, p: float) -> None:
        """Anneal budget pressure in [0,1]. `p=0` is unbounded storage and full reads — which is what the
        deleted Stage-0 warmup used to be, except dropout is already on so the smearing habit is never
        learned. ONE knob drives both budgets: annealing them separately produces an early regime where
        storage is loose but reads are tight, i.e. the model reads a tiny slice of a graph it was never
        taught to index."""
        self.pressure = float(max(0.0, min(1.0, p)))

    def _budgets(self) -> tuple[int, int]:
        big = 1 << 20
        s = int(round(big * (1 - self.pressure) + self.storage_budget * self.pressure))
        r = int(round(big * (1 - self.pressure) + self.read_budget * self.pressure))
        return max(s, self.storage_budget), max(r, self.read_budget)

    def init_streaming_state(self, batch_size, device, dtype):
        if batch_size != 1:
            # The graph is a per-sequence object with a data-dependent node count; batching it means ragged
            # graphs, and silently building one sequence's memory for a whole batch would be far worse than
            # refusing. Batch by gradient accumulation instead.
            raise ValueError(f"kvgraph requires batch_size=1 (got {batch_size}); accumulate gradients")
        return {"graph": PersistentGraph(storage_budget=self.storage_budget, n_sinks=self.n_sinks,
                                         recent_windows=self.recent_windows),
                "ids": [], "device": device, "dtype": dtype, "n_windows": 0}

    # ------------------------------------------------------------------ write

    def streaming_write(self, state, token_embeds, attention_mask=None, chunk_offset=0,
                        context_ids=None, **extra):
        """Ingest one window. `context_ids` carries the token ids for THIS window — the parser needs text,
        and the harness hands encoders embeddings. Without ids we cannot parse, so we say so loudly rather
        than silently degrading to an empty graph."""
        del extra
        if context_ids is None:
            raise ValueError(
                "kvgraph.streaming_write needs `context_ids` (the window's token ids) — the parser works on "
                "text and the harness passes embeddings. Pass context_ids through from model.py.")
        if attention_mask is None:
            attention_mask = torch.ones(token_embeds.shape[:2], device=token_embeds.device,
                                        dtype=torch.bool)
        pg: PersistentGraph = state["graph"]
        ids = [int(i) for i, m in zip(context_ids[0].tolist(), attention_mask[0].tolist()) if m]
        if not ids:
            return state, {}

        text, offsets = token_char_offsets(self.tok, ids)
        if state["n_windows"] == 0:
            assert_alignment(text, offsets, len(ids))     # loud on window 0; cheap enough to keep always

        cap = self.capture.run(token_embeds[:, :len(ids)], attention_mask[:, :len(ids)])

        # Sinks: the literal first tokens, unpooled and never evicted. Without them the frozen decoder has
        # nowhere to park the attention mass it cannot place (StreamingLLM).
        if state["n_windows"] == 0:
            for t in range(min(self.n_sinks, len(ids))):
                K = torch.stack([cap["k"][li][0, t] for li in range(self.L)])
                V = torch.stack([cap["v"][li][0, t] for li in range(self.L)])
                pg.add_sink(K, V, label=f"<sink{t}>")

        parse = self.parser.parse(text)
        win, _ = build_graph(parse, offsets)
        if not win.nodes:
            state["n_windows"] += 1
            return state, {}

        node_ids = sorted(win.nodes)
        toks = [win.nodes[i].token_positions for i in node_ids]
        heads = [max((m.head_token for m in win.nodes[i].mentions), default=-1) for i in node_ids]
        K, V = pool_nodes(cap, toks, head_tokens=heads, head_weight=self.head_weight)
        summ = pool_hidden(cap, toks, self.summary_layer)
        # The edge vector is pooled from the tokens that licensed the arc — "merchandise-ness" comes from
        # the actual tokens `sold`/`the farm` in context, with zero trained parameters in v0.
        evecs = {id(e): pool_hidden(cap, [list(e.licensing_tokens)], self.summary_layer)[0]
                 for e in win.edges if e.licensing_tokens}

        pg.admit(win,
                 kv={n: (K[i], V[i]) for i, n in enumerate(node_ids)},
                 summaries={n: summ[i] for i, n in enumerate(node_ids)},
                 edge_vecs=evecs)
        pg.contract_to_budget(self._budgets()[0])
        state["n_windows"] += 1
        state["k_rms"], state["v_rms"] = cap["k_rms"], cap["v_rms"]
        state["last_hidden"] = cap["hidden"][self.summary_layer][0, -1]     # retrieval seed for next window
        return state, {}

    # ------------------------------------------------------------------ read

    def finalize_memory(self, state):
        pg: PersistentGraph = state["graph"]
        dev = state["device"]
        if not pg.g.nodes:
            raise ValueError("kvgraph.finalize_memory: empty graph (no window produced any node)")

        # Retrieval is seeded by the tail of the last ingested window — a non-circular query that streaming
        # supplies for free. A question embedding, when the harness provides one, is strictly better.
        q = state.get("question_embeds")
        seed = q[0].mean(0) if q is not None else state.get(
            "last_hidden", torch.zeros(self.d, device=dev))
        chosen = pg.retrieve(seed, self._budgets()[1], alpha=self.ppr_alpha)
        pg.touch(chosen)

        if self.training and self.node_dropout > 0:
            # Random node dropout, on from step 0. Without it the mixer may smear one fact across several
            # co-dependent nodes — fine under read-all, catastrophic once a subset is retrieved.
            keep = [n for n in chosen
                    if n in pg.sink_ids or torch.rand(()).item() >= self.node_dropout]
            chosen = keep or chosen[: len(pg.sink_ids) + 1]

        pos = {n: i for i, n in enumerate(chosen)}
        Kp = torch.stack([pg.kv[n][0] for n in chosen if n in pg.kv])
        Vp = torch.stack([pg.kv[n][1] for n in chosen if n in pg.kv])
        chosen = [n for n in chosen if n in pg.kv]
        pos = {n: i for i, n in enumerate(chosen)}

        node_vec = torch.stack([pg.summary.get(n, torch.zeros(self.d, device=dev)).to(dev)
                                for n in chosen]).float()
        sub = [e for e in pg.g.edges if e.src in pos and e.dst in pos]
        if sub and not self.ablate_graph:
            src = torch.tensor([pos[e.src] for e in sub], device=dev)
            dst = torch.tensor([pos[e.dst] for e in sub], device=dev)
            rel = torch.tensor([list(type(sub[0].relation)).index(e.relation) for e in sub], device=dev)
            ev = torch.stack([(e.vec if e.vec is not None else torch.zeros(self.d, device=dev)).to(dev)
                              for e in sub]).float()
        else:
            src = dst = rel = torch.zeros(0, dtype=torch.long, device=dev)
            ev = None

        mixed = self.mixer(node_vec, ev, src, dst, rel,
                           ablate_relations=self.ablate_relations,
                           ablate_edge_vec=self.ablate_edge_vec,
                           ablate_graph=self.ablate_graph)
        first = torch.tensor([min((m.token_start for m in pg.g.nodes[n].mentions), default=0)
                              for n in chosen], device=dev)
        Ks, Vs = self.injector(Kp, Vp, mixed, base_model=self.base, first_tokens=first,
                               k_rms=state.get("k_rms"), v_rms=state.get("v_rms"))

        mm = torch.ones(1, len(chosen), device=dev)
        empty = torch.zeros(1, 0, self.d, device=dev)
        return empty, {"past_kv": (Ks, Vs), "memory_mask": mm, "read_mode": "per_layer_kv",
                       "kvgraph_stats": {"n_nodes": len(pg.g.nodes), "n_edges": len(pg.g.edges),
                                         "n_read": len(chosen), "n_evicted": pg.n_evicted,
                                         "n_recalled": pg.n_recalled,
                                         "n_records": sum(len(c) for c in pg.records.values())}}

    def forward(self, token_embeds, attention_mask=None, mask_positions=None, context_ids=None):
        del mask_positions
        st = self.init_streaming_state(token_embeds.shape[0], token_embeds.device, token_embeds.dtype)
        st, _ = self.streaming_write(st, token_embeds, attention_mask, context_ids=context_ids)
        return self.finalize_memory(st)
