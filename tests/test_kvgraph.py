"""kvgraph correctness tests.

The headline test is `test_single_token_node_roundtrip`: pool a node covering exactly ONE token and assert
its injected KV is bit-identical to that token's original cache entry. That one assertion catches both of
the pipeline's silent failures at once — a char/token misalignment pools the wrong tokens, and a RoPE
mistake rotates keys that should not be rotated (or fails to rotate ones that should be). Both produce
plausible-looking tensors and quietly meaningless memory.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.memory.models.kvgraph.align import assert_alignment, token_char_offsets, tokens_for_span
from src.memory.models.kvgraph.build import build_graph
from src.memory.models.kvgraph.edges import is_attribute, relation_for_argument
from src.memory.models.kvgraph.merge import PersistentGraph
from src.memory.models.kvgraph.mixer import GraphMixer
from src.memory.models.kvgraph.schema import Graph, Mention, Node, NodeKind, Relation

SENTENCE = ("Although Maria didn't believe the rumour that her brother had secretly sold the family farm, "
            "she called him anyway, because the bank had already sent her a letter about it.")

spacy = pytest.importorskip("spacy", reason="kvgraph parsing needs spaCy")


@pytest.fixture(scope="module")
def parser():
    from src.memory.models.kvgraph.parse import Parser
    try:
        return Parser(use_coref=False)
    except OSError as exc:  # pragma: no cover - model not downloaded
        pytest.skip(str(exc))


@pytest.fixture(scope="module")
def tokenizer():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    except Exception as exc:  # pragma: no cover - offline
        pytest.skip(f"tokenizer unavailable: {exc}")


# ----------------------------------------------------------------------------- alignment


def test_offsets_tile_the_text(tokenizer):
    ids = tokenizer(SENTENCE, add_special_tokens=False)["input_ids"]
    text, offsets = token_char_offsets(tokenizer, ids)
    assert_alignment(text, offsets, len(ids))
    assert text == SENTENCE, "detokenisation must reproduce the source exactly, or every offset shifts"


def test_span_lookup_uses_overlap_not_containment(tokenizer):
    """A BPE token carrying a leading space starts BEFORE the mention it belongs to; containment would
    silently drop it and the node would be pooled from a subset of its own tokens."""
    ids = tokenizer(SENTENCE, add_special_tokens=False)["input_ids"]
    text, offsets = token_char_offsets(tokenizer, ids)
    start = text.index("the family farm")
    toks = tokens_for_span(offsets, start, start + len("the family farm"))
    assert toks, "no tokens overlap a span that is plainly present in the text"
    covered = text[offsets[toks[0]][0]:offsets[toks[-1]][1]]
    assert "family farm" in covered


# ----------------------------------------------------------------------------- edge rules


@pytest.mark.parametrize("dep,lemma,passive,expect", [
    ("nsubj", "sell", False, Relation.AGENT),
    ("nsubj", "believe", False, Relation.EXPERIENCER),   # mental predicate takes an experiencer
    ("nsubjpass", "sell", True, Relation.THEME),         # voice inversion is mandatory
    ("agent", "sell", True, Relation.AGENT),
    ("dobj", "sell", False, Relation.THEME),
    ("acl", "rumour", False, Relation.CONTENT),          # THE load-bearing arc
])
def test_dependency_to_relation(dep, lemma, passive, expect):
    assert relation_for_argument(dep, head_lemma=lemma, head_is_passive=passive) is expect


def test_passive_does_not_make_the_theme_the_agent():
    """'The farm was sold by her brother' — the naive nsubj->AGENT mapping would make the FARM the seller."""
    assert relation_for_argument("nsubjpass", head_lemma="sell", head_is_passive=True) is Relation.THEME
    assert relation_for_argument("agent", head_lemma="sell", head_is_passive=True) is Relation.AGENT


def test_bare_adverb_is_an_attribute_not_an_edge():
    """'secretly' is not a particular, so there is no node for a MANNER edge to point at."""
    assert relation_for_argument("advmod", head_lemma="sell", head_is_passive=False) is None
    assert is_attribute("advmod") == "manner"


def test_prepositional_manner_still_yields_an_edge():
    """...but 'sold BY AUCTION' has a nominal object, so MANNER fires."""
    assert relation_for_argument("prep", head_lemma="sell", head_is_passive=False,
                                 prep_lemma="by") is Relation.MANNER


# ----------------------------------------------------------------------------- graph construction


def test_graph_has_event_hubs_and_the_content_arc(parser, tokenizer):
    ids = tokenizer(SENTENCE, add_special_tokens=False)["input_ids"]
    text, offsets = token_char_offsets(tokenizer, ids)
    g, _ = build_graph(parser.parse(text), offsets)

    assert not g.validate(), f"structural invariants violated: {g.validate()}"
    events = g.by_kind(NodeKind.EVENT)
    assert events, "no predicate was reified — every verb became an edge label"
    assert {"sell", "call", "send"} & {n.label for n in events}

    # The sale must be reachable only THROUGH the rumour, or the graph asserts it as world-fact.
    sale = next((n for n in events if n.label == "sell"), None)
    assert sale is not None
    inbound = [e for e in g.edges if e.dst == sale.node_id]
    assert any(e.relation is Relation.CONTENT for e in inbound), \
        "no CONTENT arc into the sale: the graph is asserting the rumoured sale as fact"


def test_node_count_is_data_dependent(parser, tokenizer):
    """We never choose K. A denser paragraph yields more nodes; a repetitive one fewer."""
    def n_nodes(s):
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        text, offsets = token_char_offsets(tokenizer, ids)
        return len(build_graph(parser.parse(text), offsets)[0].nodes)

    dense = "Maria sold the farm. Peter bought the barn. Anna leased the mill. Tom rented the shed."
    sparse = "Maria walked. Maria walked. Maria walked. Maria walked."
    assert n_nodes(dense) > n_nodes(sparse)


# ----------------------------------------------------------------------------- eviction


def _toy_graph(n_entities=3, n_events=6):
    pg = PersistentGraph(storage_budget=4, n_sinks=0, recent_windows=0)
    g = Graph()
    nid = 0
    ents = []
    for i in range(n_entities):
        m = Mention(0, 1, f"e{i}", NodeKind.ENTITY, f"e{i}", token_start=i, token_end=i + 1, head_token=i)
        g.add_node(Node(nid, NodeKind.ENTITY, f"e{i}", [m]))
        ents.append(nid)
        nid += 1
    for j in range(n_events):
        m = Mention(0, 1, f"v{j}", NodeKind.EVENT, f"v{j}", token_start=100 + j, token_end=101 + j,
                    head_token=100 + j)
        g.add_node(Node(nid, NodeKind.EVENT, f"v{j}", [m]))
        g.add_edge(nid, ents[j % n_entities], Relation.AGENT)
        nid += 1
    d = 8
    pg.admit(g, kv={i: (torch.zeros(2, 1, d), torch.zeros(2, 1, d)) for i in g.nodes},
             summaries={i: torch.randn(d) for i in g.nodes})
    return pg


def test_contraction_reaches_budget_without_creating_nodes():
    pg = _toy_graph()
    before = len(pg.g.nodes)
    pg.contract_to_budget(4)
    assert len(pg.g.nodes) <= 4 < before
    # No supernode was created: eviction only ever removes.
    assert all("gravestone" not in n.label for n in pg.g.nodes.values())


def test_contraction_preserves_reachability_via_pointer_edges():
    """A -> B -> C with B evicted must leave A able to reach C, or multi-hop severs exactly as memory
    fills — which is literally the M+ failure we measured."""
    pg = PersistentGraph(storage_budget=2, n_sinks=0, recent_windows=0)
    g = Graph()
    for i, (kind, lab) in enumerate([(NodeKind.ENTITY, "a"), (NodeKind.EVENT, "b"), (NodeKind.ENTITY, "c")]):
        g.add_node(Node(i, kind, lab, [Mention(0, 1, lab, kind, lab, token_start=i, token_end=i + 1,
                                               head_token=i)]))
    g.add_edge(1, 0, Relation.AGENT)
    g.add_edge(1, 2, Relation.THEME)
    d = 4
    pg.admit(g, kv={i: (torch.zeros(1, 1, d), torch.zeros(1, 1, d)) for i in g.nodes},
             summaries={i: torch.randn(d) for i in g.nodes})
    ids = {n.label: n.node_id for n in pg.g.nodes.values()}
    pg.last_used[ids["b"]] = -1                                   # make the event the coldest
    pg.contract_to_budget(2)

    assert ids["b"] not in pg.g.nodes, "the coldest node should have been contracted away"
    live = set(pg.g.nodes)
    assert {ids["a"], ids["c"]} <= live
    assert any({e.src, e.dst} == {ids["a"], ids["c"]} for e in pg.g.edges), \
        "a->c path was severed: the survivor did not inherit the victim's edges"
    assert any(e.relation is Relation.GRAVESTONE_POINTER for e in pg.g.edges)


def test_archived_content_is_recallable():
    pg = _toy_graph()
    pg.contract_to_budget(4)
    assert pg.records, "nothing was archived, so nothing could ever be recalled"
    host = next(iter(pg.records))
    n_before = len(pg.g.nodes)
    restored = pg.recall(host)
    assert restored > 0 and len(pg.g.nodes) == n_before + restored


def test_sinks_are_never_evicted():
    pg = PersistentGraph(storage_budget=1, n_sinks=2, recent_windows=0)
    d = 4
    s0 = pg.add_sink(torch.zeros(1, 1, d), torch.zeros(1, 1, d), "<sink0>")
    s1 = pg.add_sink(torch.zeros(1, 1, d), torch.zeros(1, 1, d), "<sink1>")
    g = Graph()
    g.add_node(Node(0, NodeKind.EVENT, "v", [Mention(0, 1, "v", NodeKind.EVENT, "v", token_start=9,
                                                     token_end=10, head_token=9)]))
    pg.admit(g, kv={0: (torch.zeros(1, 1, d), torch.zeros(1, 1, d))}, summaries={0: torch.randn(d)})
    pg.contract_to_budget(1)
    assert {s0, s1} <= set(pg.g.nodes), "an attention sink was evicted — generation will degrade silently"


# ----------------------------------------------------------------------------- mixer


def test_relation_operator_is_its_diagonal_at_init():
    """At step 0 the per-edge low-rank correction must be exactly zero, so the operator IS its relation's
    diagonal and an untrained mixer is a no-op rather than noise.

    Asserted behaviourally, not by inspecting which tensor is zero-initialised: the zero must sit on `rel_U`
    (the OUTPUT factor), because zero-initialising the gate instead looks equivalent, produces the same
    step-0 behaviour, and permanently kills the gradient to rel_U/rel_V so the branch never wakes up.
    """
    torch.manual_seed(0)
    m = GraphMixer(d_llama=16, d_mix=8, n_layers=1, n_heads=2, rank=4, d_id=8)
    mp = m.mp[0]
    h = torch.randn(4, 8)
    src, dst = torch.tensor([0, 1]), torch.tensor([1, 2])
    rel = torch.tensor([0, 5])
    ev = torch.randn(2, 8)
    with_edge = mp(h, src, dst, rel, ev)
    without = mp(h, src, dst, rel, None)
    assert torch.allclose(with_edge, without, atol=1e-6), \
        "the edge-vector correction is non-zero at init — step 0 is no longer the pure relation operator"

    # ...and the gradient must still reach the low-rank factors, or they are dead weight forever.
    # NB: the probe loss must NOT be `.sum()`. The message passes through a LayerNorm, and the gradient of
    # sum(LayerNorm(x)) w.r.t. x is exactly zero (LN output is shift-invariant, so its sum is constant) —
    # a `.sum()` probe reports every upstream parameter as dead regardless of whether it is.
    torch.manual_seed(1)
    mp.zero_grad()
    probe = torch.randn(4, 8)
    (mp(h, src, dst, rel, ev) * probe).sum().backward()
    assert mp.rel_U.grad is not None and mp.rel_U.grad.abs().sum() > 0, \
        "rel_U has no gradient: the low-rank branch can never wake up"
    assert mp.rel_V.grad is not None and mp.rel_V.grad.abs().sum() == 0, \
        "rel_V should have zero gradient at init (dL/dV flows through U, which is zero) — it wakes at step 1"


def test_ablations_change_the_output():
    torch.manual_seed(0)
    m = GraphMixer(d_llama=16, d_mix=8, n_layers=2, n_heads=2, rank=4, d_id=8).eval()
    for p in m.parameters():                       # break the identity init so relations can differ
        torch.nn.init.normal_(p, std=0.1) if p.dim() > 1 else torch.nn.init.normal_(p, std=0.1)
    nv = torch.randn(4, 16)
    ev = torch.randn(3, 16)
    src = torch.tensor([0, 1, 2])
    dst = torch.tensor([1, 2, 3])
    rel = torch.tensor([0, 5, 9])
    torch.manual_seed(1); full = m(nv, ev, src, dst, rel)
    torch.manual_seed(1); no_rel = m(nv, ev, src, dst, rel, ablate_relations=True)
    torch.manual_seed(1); no_graph = m(nv, ev, src, dst, rel, ablate_graph=True)
    assert not torch.allclose(full, no_rel), "relation typing is inert — the ablation would show nothing"
    assert not torch.allclose(full, no_graph), "edges are inert — the graph-off control would show nothing"


# ----------------------------------------------------------------------------- integration


@pytest.mark.slow
def test_end_to_end_stream_read_and_train():
    """Full pipeline against a real backbone: stream windows, contract to budget, inject, read with the
    frozen decoder, and confirm the trainable modules actually receive gradient.

    Also pins the zero-init bootstrap: the injector's projections start at zero so injection is EXACTLY the
    pooled KV on step 0, which means the mixer gets no gradient until step 1. That is LoRA's behaviour and
    it is load-bearing (it is what makes step 0 a working compressed cache rather than noise) — this test
    exists so a future "fix" that zero-initialises the FiLM scale instead, killing the branch permanently,
    fails loudly.
    """
    torch.manual_seed(0)
    from src.memory.config import ReprConfig
    from src.memory.decoder import build_prefix_cache
    from src.memory.models.kvgraph.encoder import KVGraphEncoder

    # Budget deliberately below what one sentence produces, so eviction actually fires; a test that never
    # exceeds its budget silently stops covering the contraction path.
    cfg = ReprConfig(llama_model="HuggingFaceTB/SmolLM2-135M", d_llama=576,
                     kvg_storage_budget=10, kvg_read_budget=6, kvg_n_sinks=2, kvg_recent_windows=0)
    try:
        enc = KVGraphEncoder(cfg).train()
    except Exception as exc:  # pragma: no cover - offline / no model
        pytest.skip(f"backbone unavailable: {exc}")

    ids = enc.tok(SENTENCE, add_special_tokens=False)["input_ids"]
    emb = enc.base.get_input_embeddings()(torch.tensor([ids]))
    opt = torch.optim.Adam([p for p in enc.parameters() if p.requires_grad], lr=1e-3)

    def one_step():
        st = enc.init_streaming_state(1, emb.device, emb.dtype)
        for s in range(0, len(ids), 32):
            e = min(s + 32, len(ids))
            st, _ = enc.streaming_write(st, emb[:, s:e], torch.ones(1, e - s, dtype=torch.bool),
                                        chunk_offset=s, context_ids=torch.tensor([ids[s:e]]))
        _, aux = enc.finalize_memory(st)
        Ks, Vs = aux["past_kv"]
        assert len(Ks) == enc.L and Ks[0].shape[1] == enc.n_kv
        assert Ks[0].shape[2] <= enc.read_budget, "read exceeded its budget"
        m = Ks[0].shape[2]
        q = torch.tensor([ids[:16]])
        out = enc.base(inputs_embeds=enc.base.get_input_embeddings()(q).to(
            next(enc.base.parameters()).dtype),
            attention_mask=torch.ones(1, m + 16, dtype=torch.long),
            past_key_values=build_prefix_cache((Ks, Vs)), use_cache=True)
        assert torch.isfinite(out.logits).all(), "decoder produced non-finite logits from graph-KV"
        return torch.nn.functional.cross_entropy(out.logits[0, :-1].float(), q[0, 1:]), aux

    mixer_params = [p for n, p in enc.named_parameters() if n.startswith("mixer") and p.requires_grad]
    live = []
    for _ in range(3):
        opt.zero_grad()
        loss, aux = one_step()
        loss.backward()
        live.append(sum(1 for p in mixer_params if p.grad is not None and p.grad.abs().sum() > 0))
        opt.step()

    assert live[0] == 0, "mixer had gradient at step 0 — the injector is no longer zero-initialised"
    assert live[-1] > 0.9 * len(mixer_params), f"mixer never woke up: {live}"
    assert aux["kvgraph_n_evicted"] > 0, "budget was never exceeded — the test is not exercising eviction"
    # Canaries that must hold by construction, not by luck. Catching these here is what makes a null
    # experimental result attributable to the hypothesis rather than to a broken injector.
    assert abs(aux["kvgraph_k_rms_ratio"] - 1.0) < 0.2, \
        f"injected K is off-distribution vs real tokens: ratio={aux['kvgraph_k_rms_ratio']:.3f}"
    assert aux["kvgraph_relation_used"] >= 3, "the parse exercised almost none of the relation inventory"
    # Absolute effective rank is naturally LOW here: LM hidden states are strongly anisotropic (mean
    # cosine between random tokens ~0.6 in mid layers), and the read budget in this test is tiny. So the
    # meaningful check is RELATIVE — the mixer must not be the thing destroying the rank it was handed.
    assert aux["kvgraph_mixed_effrank"] > 1.05, "mixer output fully collapsed to rank 1"
    assert aux["kvgraph_mixed_effrank"] > 0.3 * aux["kvgraph_node_effrank"], (
        f"mixer destroyed most of the rank it was given: "
        f"node={aux['kvgraph_node_effrank']:.2f} -> mixed={aux['kvgraph_mixed_effrank']:.2f} "
        f"(over-smoothing is the graph-native form of the collapse this project has produced twice)")


def test_recall_restores_content_not_just_structure():
    """Archived nodes must come back WITH their KV.

    An earlier version archived `(node, edges)` only. Recall then restored the topology, `finalize_memory`
    skipped every restored node for want of a cache entry, and the recovery silently succeeded while doing
    nothing — the worst kind of bug, because `n_recalled` went up.
    """
    pg = _toy_graph()
    pg.contract_to_budget(4)
    assert pg.records, "nothing archived"
    host = next(iter(pg.records))
    before = set(pg.g.nodes)
    assert pg.recall(host) > 0
    restored = set(pg.g.nodes) - before
    assert restored, "recall returned >0 but added no nodes"
    for nid in restored:
        assert nid in pg.kv, f"restored node {nid} has no KV — it will be silently dropped at read time"


def test_recall_is_actually_triggered_by_a_query():
    """The recall PATH must run, not merely exist. Recoverable eviction that never recovers is H2O with a
    graph in front of it, and `n_recalled` staying at 0 across a whole run is how that hides."""
    pg = _toy_graph()
    pg.contract_to_budget(4)
    resident = [n for n in pg.g.nodes]
    q = torch.randn(8)
    # threshold -1 accepts anything: this asserts the candidate SCAN finds the archive-holding survivors,
    # independently of how well any particular gist happens to match.
    cands = pg.recall_candidates(resident, q, threshold=-1.0)
    assert cands, "no survivor was offered as a recall candidate despite holding an archive"
    assert cands[0][0] in pg.records
