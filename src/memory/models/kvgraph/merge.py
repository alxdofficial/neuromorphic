"""Stage 5 — the persistent graph: admission, retrieval, and eviction by contraction.

There is only ever ONE graph. Each window contributes nodes into the same pool, coreference links them where
they denote the same particular, and budget pressure contracts the cold parts. Nothing here combines "graph A
with graph B", because per-window graphs were never objects — just batches of admissions.

Eviction separates POLICY from MECHANISM, and they are ablated independently:

  policy     LRU + protected attention sinks + protected recent window
  mechanism  contract the coldest node into its hottest surviving neighbour

**Policy.** Plain LRU, not an attention-mass EMA. That rule is LRFU (Lee et al. 2001) — it already lost a
head-to-head in Compressive Transformer's own ablation (0.980 BPC vs 0.973), policy sophistication buys ~1.5%
across 6,594 cache traces (SIEVE/S3-FIFO), and attention-based scoring is structurally blind to any fact
nobody has queried yet. Sinks are non-optional: transformers dump large attention onto the first few
positions regardless of content, and our injected prefix would otherwise have nothing playing that role.
The recent window is what protects newly written content, which by construction has zero usage.

**Mechanism.** Contraction, not supernodes. No new node is created, so single-node eviction is immediately
profitable and the entire clustering / gain() / lambda apparatus is unnecessary. Crucially the survivor's own
vector is UNTOUCHED, so nothing is superposed on the routing path — which is what dodges the capacity result
that killed the supernode design (with correlated LM embeddings a bundled key caps out around 9 members).
Only the gist is a superposition, and a degraded gist costs an unnecessary recall, not a wrong answer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .schema import Edge, Graph, Node, NodeKind, Relation


@dataclass
class ArchivedRecord:
    """What a `GRAVESTONE_POINTER` edge points at. In-memory here; a disk backend is a drop-in replacement
    (the only requirement is that `entries` round-trips)."""

    entries: list[tuple[Node, list[Edge]]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


class PersistentGraph:
    """The bounded memory. Holds nodes with their pooled KV, their edges, and the LRU bookkeeping."""

    def __init__(self, *, storage_budget: int = 96, n_sinks: int = 4, recent_windows: int = 2,
                 record_cap: int = 8, pointers_per_survivor: int = 4, link_threshold: float = 0.92):
        self.g = Graph()
        self.storage_budget = storage_budget
        self.n_sinks = n_sinks
        self.recent_windows = recent_windows
        self.record_cap = record_cap
        self.pointers_per_survivor = pointers_per_survivor
        self.link_threshold = link_threshold

        self._next_id = 0
        self._step = 0
        self._window = 0
        self.last_used: dict[int, int] = {}
        self.created_window: dict[int, int] = {}
        self.sink_ids: set[int] = set()
        self.summary: dict[int, torch.Tensor] = {}          # node_id -> [d] mid-stack summary
        self.kv: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}   # node_id -> (K,V) [L,n_kv,hd]
        self.records: dict[int, list[ArchivedRecord]] = {}  # survivor -> its archive chain
        self.n_evicted = 0
        self.n_recalled = 0

    # ------------------------------------------------------------------ admission

    def new_id(self) -> int:
        self._next_id += 1
        return self._next_id - 1

    def add_sink(self, K: torch.Tensor, V: torch.Tensor, label: str) -> int:
        """Register a literal token KV entry as a permanent, unevictable sink at the head of the prefix.

        Not pooled, not merged, not scored. Without these the frozen decoder — which was trained with
        attention sinks always present at the front of the sequence — has nowhere to dump the softmax mass
        it cannot place, and generation degrades in a way that looks like a mixer bug.
        """
        nid = self.new_id()
        node = Node(node_id=nid, kind=NodeKind.ENTITY, label=label)
        node.attrs["sink"] = True
        self.g.add_node(node)
        self.kv[nid] = (K, V)
        self.sink_ids.add(nid)
        self.last_used[nid] = 1 << 30                       # never the LRU victim
        self.created_window[nid] = -1
        return nid

    def admit(self, win: Graph, kv: dict[int, tuple[torch.Tensor, torch.Tensor]],
              summaries: dict[int, torch.Tensor], edge_vecs: dict[int, torch.Tensor] | None = None
              ) -> dict[int, int]:
        """Insert one window's graph, linking entity nodes that corefer with resident ones.

        -> {window_node_id: persistent_node_id}. Linking is deliberately CONSERVATIVE (see
        `link_threshold`): under-merging loses a path, over-merging fuses two individuals and creates a
        FALSE path, and with abstention-aware scoring a wrong answer costs more than a missing one.
        """
        remap: dict[int, int] = {}
        for wid, node in win.nodes.items():
            target = self._find_link(node, summaries.get(wid))
            if target is not None:
                self._absorb_into(target, node, summaries.get(wid))
                remap[wid] = target
                continue
            nid = self.new_id()
            node.node_id = nid
            self.g.add_node(node)
            if wid in kv:
                self.kv[nid] = kv[wid]
            if wid in summaries:
                self.summary[nid] = summaries[wid]
            self.last_used[nid] = self._step
            self.created_window[nid] = self._window
            remap[wid] = nid

        for e in win.edges:
            s, d = remap.get(e.src), remap.get(e.dst)
            if s is None or d is None or s == d:
                continue
            ev = (edge_vecs or {}).get(id(e))
            edge = self.g.add_edge(s, d, e.relation, provenance=e.provenance,
                                   licensing_tokens=e.licensing_tokens)
            edge.vec = ev
        self._window += 1
        return remap

    def _find_link(self, node: Node, summary: torch.Tensor | None) -> int | None:
        """Conservative coreference link into the resident graph. Entities only.

        Events are excluded: two `sell` predicates in one document are usually two different sales, and
        fusing them is the over-merge error. Nominalised event coreference ("the sale" referring back to
        "sold") is a real phenomenon we do NOT handle in v0 — it is the main thing an SRL/AMR upgrade buys.
        """
        if node.kind is not NodeKind.ENTITY:
            return None
        cands = [n for n in self.g.nodes.values()
                 if n.kind is NodeKind.ENTITY and n.node_id not in self.sink_ids]
        exact = [n for n in cands if n.label == node.label]
        if exact and node.label not in _AMBIGUOUS_HEADS:
            return exact[0].node_id
        if summary is None:
            return None
        best, best_sim = None, self.link_threshold
        for n in cands:
            s = self.summary.get(n.node_id)
            if s is None:
                continue
            sim = torch.nn.functional.cosine_similarity(summary.float(), s.float(), dim=-1).item()
            if sim > best_sim:
                best, best_sim = n.node_id, sim
        return best

    def _absorb_into(self, target: int, node: Node, summary: torch.Tensor | None) -> None:
        """Coref merge: identity, so mentions unite and NOTHING is archived.

        The node's KV is deliberately not re-pooled from the new mentions. An entity node is an identity
        KEY; its facts live on its edges, and re-averaging every mention would drift the key toward a
        centroid matching no context — the pool-then-address trap.
        """
        tgt = self.g.nodes[target]
        tgt.mentions.extend(node.mentions)
        for k, v in node.attrs.items():
            tgt.attrs.setdefault(k, v)
        # The summary is deliberately NOT updated. A node is an identity KEY, and blending each new mention
        # in would drift it toward a centroid matching no context — the pool-then-address trap. (An earlier
        # version did a 0.9/0.1 EMA with no principled basis for either constant.)
        self.last_used[target] = self._step

    # ------------------------------------------------------------------ retrieval

    def touch(self, node_ids) -> None:
        self._step += 1
        for n in node_ids:
            if n not in self.sink_ids:
                self.last_used[n] = self._step

    def _adjacency(self, ids: list[int]) -> torch.Tensor:
        """Symmetrised, row-normalised adjacency over `ids`.

        Symmetrised because direction and relation are for COMPUTATION (the mixer's typed operators), not
        for reachability. That is also why no inverse edges are ever stored: with event hubs, two
        co-participants are always exactly 2 hops apart.
        """
        pos = {n: i for i, n in enumerate(ids)}
        A = torch.zeros(len(ids), len(ids))
        for e in self.g.edges:
            i, j = pos.get(e.src), pos.get(e.dst)
            if i is not None and j is not None:
                A[i, j] = A[j, i] = 1.0
        return A / A.sum(1, keepdim=True).clamp(min=1.0)

    def retrieve(self, query: torch.Tensor, read_budget: int, *, alpha: float = 0.15) -> list[int]:
        """Personalised PageRank from query-similar seeds -> top-B node ids (sinks always included).

        `alpha` is the restart probability and the ONLY knob for "how far to traverse": high stays local,
        low diffuses. 0.15 is Brin & Page's original damping, not a guess. It replaces a discrete hop limit
        and handles cycles natively, being a Markov chain.

        Solved EXACTLY as `r = alpha (I - (1-alpha) P^T)^-1 s` rather than by power iteration. An earlier
        version ran 8 iterations, which at alpha=0.15 leaves 0.85^8 = 27% residual error — the ranking was
        being read off an unconverged vector. At a <=few-hundred-node budget the dense solve is microseconds,
        and it removes an iteration count that had no principled value.
        """
        ids = [n for n in self.g.nodes if n not in self.sink_ids]
        if not ids:
            return sorted(self.sink_ids)
        seed = torch.zeros(len(ids))
        for i, n in enumerate(ids):
            s = self.summary.get(n)
            if s is not None:
                seed[i] = torch.nn.functional.cosine_similarity(
                    query.float().flatten(), s.float().flatten(), dim=0).clamp(min=0.0)
        if seed.sum() <= 0:
            seed[:] = 1.0
        seed = seed / seed.sum()

        A = self._adjacency(ids)
        M = torch.eye(len(ids)) - (1 - alpha) * A.t()
        try:
            r = alpha * torch.linalg.solve(M, seed)
        except Exception:                                    # noqa: BLE001 - singular only if A is degenerate
            r = seed.clone()
            for _ in range(64):                              # fallback: iterate to ~1e-5 at alpha=0.15
                r = alpha * seed + (1 - alpha) * (A.t() @ r)

        keep = max(0, read_budget - len(self.sink_ids))
        top = torch.topk(r, min(keep, len(ids))).indices.tolist()
        return sorted(self.sink_ids) + [ids[i] for i in top]

    # ------------------------------------------------------------------ eviction

    def _protected(self) -> set[int]:
        recent = {n for n, w in self.created_window.items()
                  if w >= self._window - self.recent_windows and n in self.g.nodes}
        return self.sink_ids | recent

    def contract_to_budget(self, budget: int | None = None) -> int:
        """Evict down to `budget` by contraction. -> number of nodes removed.

        Coldest-first, three exhaustive cases. There is no fourth case and no refuse-to-evict: the terminal
        node of a fully-evicted component is subject to the same policy, and its now-unreachable record dies
        with it. Holding a resident slot as a handle for a region nothing has ever queried is exactly the
        cache pollution LRU exists to prevent.
        """
        budget = self.storage_budget if budget is None else budget
        removed = 0
        while len(self.g.nodes) > budget:
            protected = self._protected()
            cands = [n for n in self.g.nodes if n not in protected]
            if not cands:
                break                                    # everything left is protected: bounded overrun
            victim = min(cands, key=lambda n: self.last_used.get(n, 0))
            if not self._contract(victim):
                break
            removed += 1
        self.n_evicted += removed
        return removed

    def _neighbours(self, nid: int) -> set[int]:
        return {e.dst for e in self.g.edges if e.src == nid} | {e.src for e in self.g.edges if e.dst == nid}

    def _contract(self, victim: int) -> bool:
        """Case 1/2/3 of the contraction rule. -> False if nothing could be done (caller stops)."""
        nbrs = self._neighbours(victim) & set(self.g.nodes)
        node = self.g.nodes[victim]
        packed = (node, [e for e in self.g.edges if e.src == victim or e.dst == victim])

        if not nbrs:                                     # case 3: orphan. Free, no pointer needed.
            self._forget(victim)
            return True

        survivors = [n for n in nbrs if n in self.g.nodes]
        if survivors:                                    # case 1: contract into the hottest survivor
            host = max(survivors, key=lambda n: self.last_used.get(n, 0))
            self._transfer_edges(victim, host)
            self._archive(host, packed)
            self._forget(victim)
            return True

        # case 2 is unreachable here because `survivors` == `nbrs` by construction; kept explicit so the
        # invariant is visible rather than implied.
        self._forget(victim)
        return True

    def _transfer_edges(self, victim: int, host: int) -> None:
        """The survivor inherits the victim's edges, deduped, retyped GRAVESTONE_POINTER with the original
        relation preserved in the edge vector.

        NOT optional: without it, paths through the victim break and multi-hop severs exactly as memory
        fills — which is literally the M+ failure we measured (0.423 -> 0.286 as context grew).
        """
        existing = self._neighbours(host)
        for e in list(self.g.edges):
            other = e.dst if e.src == victim else (e.src if e.dst == victim else None)
            if other is None or other == host or other not in self.g.nodes:
                continue
            if other in existing:
                continue                                 # dedup: the host already reaches it
            new = self.g.add_edge(host, other, Relation.GRAVESTONE_POINTER,
                                  provenance=f"contracted:{e.relation.value}")
            new.vec = e.vec
            existing.add(other)

    def _archive(self, host: int, packed) -> None:
        chain = self.records.setdefault(host, [])
        if not chain or len(chain[-1]) >= self.record_cap:
            if len(chain) >= self.pointers_per_survivor:
                # Spill: fold the oldest record into a deeper one. Pointer chains are EXACT, so depth is
                # free — unlike nested superposition, whose reliability decays as 1/2^depth.
                chain[1].entries.extend(chain[0].entries)
                chain.pop(0)
            chain.append(ArchivedRecord())
        chain[-1].entries.append(packed)

    def _forget(self, nid: int) -> None:
        self.g.edges = [e for e in self.g.edges if e.src != nid and e.dst != nid]
        self.g.nodes.pop(nid, None)
        self.kv.pop(nid, None)
        self.summary.pop(nid, None)
        self.last_used.pop(nid, None)
        self.created_window.pop(nid, None)

    # ------------------------------------------------------------------ recall

    def recall(self, host: int, budget: int | None = None) -> int:
        """Page a survivor's archived subgraph back in. -> number of nodes restored.

        A cache-line fill. Callers must bound recalls per query, or this degenerates into RAG with extra
        steps.
        """
        chain = self.records.get(host)
        if not chain:
            return 0
        rec = chain.pop()
        restored = 0
        for node, edges in rec.entries:
            self.g.add_node(node)
            self.last_used[node.node_id] = self._step
            self.created_window[node.node_id] = self._window
            for e in edges:
                if e.src in self.g.nodes and e.dst in self.g.nodes:
                    self.g.edges.append(e)
            restored += 1
        if not chain:
            self.records.pop(host, None)
        self.n_recalled += restored
        if budget is not None:
            self.contract_to_budget(budget)
        return restored


#: Head lemmas too generic for exact-label linking to be safe. Merging every "man"/"thing" in a document
#: into one node is the over-merge error, and it is the one that produces wrong answers.
_AMBIGUOUS_HEADS = {"thing", "man", "woman", "person", "people", "one", "time", "day", "way", "place"}
