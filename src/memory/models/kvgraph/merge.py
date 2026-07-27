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
class ArchivedEntry:
    """One contracted node, complete enough to be reinstated.

    The KV and summary travel WITH the node. Archiving structure alone was a real bug: recall would restore
    the topology, `finalize_memory` would skip every restored node for want of a cache entry, and the
    recovery would silently succeed-but-do-nothing.
    """

    node: Node
    edges: list[Edge]
    kv: tuple[torch.Tensor, torch.Tensor] | None = None
    summary: torch.Tensor | None = None


@dataclass
class ArchivedRecord:
    """What a `GRAVESTONE_POINTER` edge points at. In-memory here; a disk backend is a drop-in replacement
    (the only requirement is that `entries` round-trips)."""

    entries: list[ArchivedEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.entries)


def _to_device(x, device):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().to(device, non_blocking=True)
    return tuple(_to_device(i, device) for i in x)


class PersistentGraph:
    """The bounded memory. Holds nodes with their pooled KV, their edges, and the LRU bookkeeping."""

    def __init__(self, *, storage_budget: int = 96, n_sinks: int = 4, recent_windows: int = 2,
                 record_cap: int = 8, pointers_per_survivor: int = 4, link_threshold: float = 0.92,
                 archive_cap: int = 512, archive_device: str = "cpu"):
        self.g = Graph()
        self.storage_budget = storage_budget
        self.n_sinks = n_sinks
        self.recent_windows = recent_windows
        self.record_cap = record_cap
        self.pointers_per_survivor = pointers_per_survivor
        self.link_threshold = link_threshold
        # The archive is the "disk" in recoverable-eviction. Keeping it as live GPU tensors made total KV
        # storage grow with every eviction, so the 96-node resident cap bounded nothing: a probe measured
        # the archive at 5.1x resident after eight short windows. It now moves to CPU and is itself capped,
        # oldest-first, so the memory claim is true rather than aspirational.
        self.archive_cap = archive_cap
        self.archive_device = archive_device
        #: The COMPUTE device the resident graph lives on, learned from the first KV admitted. Recall must
        #: bring archived tensors back onto it — otherwise a recalled node returns on CPU, and the very next
        #: torch.stack over the read set raises. That fires only after an eviction AND a matching query, so
        #: it is invisible to any short run.
        self.device = None
        self.n_archive_dropped = 0

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
        if self.device is None:
            self.device = K.device
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
                if self.device is None:
                    self.device = kv[wid][0].device
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

    @staticmethod
    def _cpu_vec(x: torch.Tensor) -> torch.Tensor:
        """Detach a vector onto CPU float32.

        Retrieval is a dense solve over ~100 nodes — microseconds on CPU, and keeping it there removes all
        device coupling. Mixing a CPU accumulator with CUDA summaries raised only under the trainer, never
        under the CPU-only tests, which is exactly the class of bug that reaches a long run undetected.
        """
        return x.detach().float().flatten().cpu()

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
        deg = A.sum(1, keepdim=True)
        # Dangling (isolated) nodes have an all-zero row, which DESTROYS probability mass instead of
        # conserving it — structurally down-ranking isolated but query-relevant facts. Standard PageRank
        # redistributes their mass; here a self-loop keeps the row stochastic and the node reachable.
        isolated = (deg.squeeze(-1) == 0)
        if isolated.any():
            A = A.clone()
            A[isolated, isolated.nonzero(as_tuple=True)[0]] = 1.0
            deg = A.sum(1, keepdim=True)
        return A / deg.clamp(min=1e-12)

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
        q = self._cpu_vec(query)
        seed = torch.zeros(len(ids))
        for i, n in enumerate(ids):
            s = self.summary.get(n)
            if s is not None:
                sv = self._cpu_vec(s)
                if sv.numel() == q.numel():
                    seed[i] = torch.nn.functional.cosine_similarity(q, sv, dim=0).clamp(min=0.0)
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
        packed = ArchivedEntry(node=node,
                               edges=[e for e in self.g.edges if e.src == victim or e.dst == victim],
                               kv=_to_device(self.kv.get(victim), self.archive_device),
                               summary=_to_device(self.summary.get(victim), self.archive_device))

        survivors = [n for n in nbrs if n in self.g.nodes]
        if survivors:                                    # case 1: contract into the hottest survivor
            host = max(survivors, key=lambda n: self.last_used.get(n, 0))
            self._transfer_edges(victim, host)
            self._archive(host, packed)
            self._inherit_records(victim, host)
            self._forget(victim)
            return True

        # case 3: no surviving neighbour, so nothing can ever point at this node's archive either. Its
        # records must be DROPPED, not orphaned — leaving them keyed to a removed node is memory holding
        # content that is unreachable by construction, which is the stranding bug in a second disguise.
        dropped = self.records.pop(victim, None)
        if dropped:
            self.n_archive_dropped += sum(len(r) for r in dropped)
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
            # The original relation as DATA, not just a provenance string. The docstring promised it was
            # carried in the edge vector; it was only ever in a debug field nothing reads, so
            # `Sale --agent--> Brother` really did degrade to an untyped "something was here".
            new.original_relation = e.relation
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
        self._enforce_archive_cap()

    def _inherit_records(self, victim: int, host: int) -> None:
        """Move the victim's own archive chain to its survivor.

        Without this, contracting a node that ALREADY hosts archives leaves `records[victim]` keyed to a
        node that no longer exists: the archive is unreachable but still resident in memory. A probe found
        44 of 49 archive hosts stranded this way. It also breaks the "pointer chains are exact, so depth is
        free" argument that the emergent-hierarchy claim rests on — a chain that drops its tail is not exact.
        """
        chain = self.records.pop(victim, None)
        if not chain:
            return
        dest = self.records.setdefault(host, [])
        for rec in chain:
            if dest and len(dest[-1]) < self.record_cap:
                dest[-1].entries.extend(rec.entries)
            else:
                dest.append(rec)
        self._enforce_archive_cap()

    def _enforce_archive_cap(self) -> None:
        """Drop the OLDEST archived entries once the total exceeds `archive_cap`, and say how many.

        A cap that silently truncates reads as "everything fit" when it did not, so the count is surfaced
        as a canary rather than swallowed.
        """
        total = sum(len(r) for c in self.records.values() for r in c)
        if total <= self.archive_cap:
            return
        for host in list(self.records):
            for rec in list(self.records[host]):
                while rec.entries and total > self.archive_cap:
                    rec.entries.pop(0)
                    total -= 1
                    self.n_archive_dropped += 1
                if not rec.entries:
                    self.records[host].remove(rec)
            if not self.records[host]:
                self.records.pop(host, None)
            if total <= self.archive_cap:
                return

    def _forget(self, nid: int) -> None:
        # Defensive: no path may leave `records` keyed to a node that is gone. Callers are expected to have
        # already inherited or dropped them; this makes stranding impossible rather than merely unlikely.
        leaked = self.records.pop(nid, None)
        if leaked:
            self.n_archive_dropped += sum(len(r) for r in leaked)
        self.g.edges = [e for e in self.g.edges if e.src != nid and e.dst != nid]
        self.g.nodes.pop(nid, None)
        self.kv.pop(nid, None)
        self.summary.pop(nid, None)
        self.last_used.pop(nid, None)
        self.created_window.pop(nid, None)

    # ------------------------------------------------------------------ recall

    def recall_candidates(self, resident: list[int], query: torch.Tensor,
                          threshold: float) -> list[tuple[int, float]]:
        """Which resident survivors are holding an archive whose GIST matches the query?

        The gist lives on the `GRAVESTONE_POINTER` edge vector, so this is the recall trigger described in
        BUILD.md 8.5: `survivor in top-B AND query . gist > threshold`. Returned hottest-match first so a
        bounded recall budget spends itself on the best candidates.
        """
        out = []
        q = self._cpu_vec(query)
        for nid in resident:
            if nid not in self.records:
                continue
            best = -1.0
            for e in self.g.edges:
                if e.src != nid or e.relation is not Relation.GRAVESTONE_POINTER or e.vec is None:
                    continue
                v = self._cpu_vec(e.vec)
                if v.numel() != q.numel():
                    continue
                best = max(best, float(torch.nn.functional.cosine_similarity(q, v, dim=0)))
            # No gist (an archive whose pointer edge carried no vector) still deserves a chance: fall back
            # to "recall if anything is archived here", which is the correct-but-expensive default rather
            # than silently never recovering.
            score = best if best > -1.0 else threshold
            if score >= threshold:
                out.append((nid, score))
        return sorted(out, key=lambda x: -x[1])

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
        for entry in rec.entries:
            node = entry.node
            self.g.add_node(node)
            if entry.kv is not None:
                self.kv[node.node_id] = _to_device(entry.kv, self.device or entry.kv[0].device)
            if entry.summary is not None:
                self.summary[node.node_id] = _to_device(entry.summary,
                                                        self.device or entry.summary.device)
            # Recalled nodes are marked fresh, so the LRU clock does not immediately evict what a query
            # just asked for. That is the thrash guard.
            self.last_used[node.node_id] = self._step
            self.created_window[node.node_id] = self._window
            for e in entry.edges:
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
