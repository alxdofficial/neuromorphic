"""Health canaries for the KV-graph, emitted every step as `kvgraph_*` keys.

These are not generic logging. Each one watches a failure mode this design is *specifically* exposed to,
and several of them are the difference between a null result that means something and a null result that
means a bug:

**Distribution shift (ranked #1 instability).** `k_rms_ratio` / `v_rms_ratio` compare injected KV against
the layer's real-token RMS. With norm-matching on these should sit at ~1.0; drift means the frozen decoder
is being fed off-distribution keys and every downstream number is suspect.

**Residual discipline.** `inject_delta_frac` is |correction| / |pooled KV|. It must be exactly 0 at step 0
(the zero-init) and should grow slowly. If it runs away, the injector has stopped being a correction to a
working compressed cache and started synthesising one, which is how the distribution-shift failure begins.

**Collapse.** `node_effrank` / `node_cos` on the mixer's output, with `pooled_effrank` alongside so a
collapse can be attributed: if pooled is healthy and mixed is not, the mixer is over-smoothing (the
graph-native form of the representation collapse this project has produced twice). If both are flat, the
problem is upstream in pooling or in the parse.

**Is the typing doing anything?** `operator_div` measures how far the per-relation operators have actually
diverged from each other. It starts at 0 by construction (identity init) and MUST rise: if the relations
stay functionally identical, the edge-typing ablation will read flat for a reason about training rather than
about the hypothesis, and we would misread that as evidence against the thesis. `relation_entropy` catches
the upstream version — a parser that only ever emits two relation types leaves most of the operator bank
untrained regardless.

**Relation vs residual.** `lowrank_frac` is the share of each message coming from the continuous per-edge
correction rather than the discrete relation. This is the two-sided ablation's live readout: if it climbs
toward 1, the edge vector is carrying everything, the relation has gone decorative, and loss-neutrality has
returned wearing a new hat.
"""
from __future__ import annotations

import torch


def participation_ratio(x: torch.Tensor) -> float:
    """Effective rank of the rows of `x` [N, d]. 1.0 == total collapse; higher is more spread."""
    if x.shape[0] < 2:
        return 0.0
    xc = x.float() - x.float().mean(0, keepdim=True)
    C = xc.t() @ xc
    return float(C.diagonal().sum() ** 2 / (C * C).sum().clamp_min(1e-12))


def mean_pairwise_cos(x: torch.Tensor) -> float:
    """Mean off-diagonal cosine between rows. -> 1.0 means every row points the same way."""
    if x.shape[0] < 2:
        return 0.0
    xn = torch.nn.functional.normalize(x.float(), dim=-1)
    C = xn @ xn.t()
    n = x.shape[0]
    return float((C.sum() - C.diagonal().sum()) / (n * (n - 1)))


def operator_divergence(mp_layers) -> dict[str, float]:
    """How far the per-relation operators have actually diverged from one another.

    0 at init by construction (`rel_diag` all ones, `rel_U` zero) and it MUST rise during training. A bank
    whose relations stay identical is indistinguishable from having no typing at all — the ablation would
    then read flat because of training dynamics, not because the hypothesis is wrong, and that is exactly
    the misattribution this canary exists to prevent.
    """
    diag_div, lr_div = [], []
    for mp in mp_layers:
        # CENTRED before comparing. rel_diag starts at all-ones, and the cosine between (1+e1) and (1+e2)
        # is ~1 for any perturbation because the shared component dominates — the uncentred measure read a
        # flat 0 for 40 steps and would have reported "the relations never differentiated" forever.
        D = mp.rel_diag.detach()
        diag_div.append(1.0 - mean_pairwise_cos(D - D.mean(0, keepdim=True)))
        R = mp.rel_U.shape[0]
        U = mp.rel_U.detach().reshape(R, -1)
        U = U - U.mean(0, keepdim=True)
        # An all-zero U (the init) has UNDEFINED pairwise cosine, and normalize() turning 0/0 into 0 would
        # report it as "maximally diverged" — the exact opposite of the truth. Report 0 until it moves.
        lr_div.append(0.0 if float(U.abs().sum()) == 0.0 else 1.0 - mean_pairwise_cos(U))
    return {"kvgraph_operator_div": sum(diag_div) / max(len(diag_div), 1),
            "kvgraph_operator_lowrank_div": sum(lr_div) / max(len(lr_div), 1)}


def relation_stats(rel: torch.Tensor, n_relations: int) -> dict[str, float]:
    """Coverage of the relation inventory on the live subgraph.

    A small closed inventory is the free-bits constraint on the edge channel — but only if the parser
    actually uses it. Two live relation types means most of the operator bank never receives gradient,
    which would silently cap what the typing can express.
    """
    if rel.numel() == 0:
        return {"kvgraph_relation_used": 0.0, "kvgraph_relation_entropy": 0.0}
    counts = torch.bincount(rel, minlength=n_relations).float()
    p = counts / counts.sum()
    nz = p[p > 0]
    return {"kvgraph_relation_used": float((counts > 0).sum()),
            "kvgraph_relation_entropy": float(-(nz * nz.log()).sum())}


def injection_stats(K_pooled: torch.Tensor, V_pooled: torch.Tensor,
                    Ks: list[torch.Tensor], Vs: list[torch.Tensor],
                    k_rms: torch.Tensor | None, v_rms: torch.Tensor | None) -> dict[str, float]:
    """Magnitude sanity on what the frozen decoder actually receives.

    `*_rms_ratio` near 1.0 says the injected cache looks like real tokens to the backbone.
    `inject_delta_frac` near 0 says the injection is still a correction to the pooled KV rather than a
    replacement for it.
    """
    kf = torch.stack([k.float().pow(2).mean() for k in Ks]).sqrt()
    vf = torch.stack([v.float().pow(2).mean() for v in Vs]).sqrt()
    out = {"kvgraph_k_rms": float(kf.mean()), "kvgraph_v_rms": float(vf.mean())}
    if k_rms is not None:
        out["kvgraph_k_rms_ratio"] = float((kf / k_rms.float().clamp_min(1e-6)).mean())
    if v_rms is not None:
        out["kvgraph_v_rms_ratio"] = float((vf / v_rms.float().clamp_min(1e-6)).mean())

    return out


def graph_stats(pg, n_read: int, n_tokens: int) -> dict[str, float]:
    """Shape of the resident graph: is it compressing, is it evicting, is it merging?"""
    from .schema import NodeKind
    nodes = list(pg.g.nodes.values())
    n = max(len(nodes), 1)
    n_ent = sum(1 for x in nodes if x.kind is NodeKind.ENTITY)
    n_mentions = sum(len(x.mentions) for x in nodes)
    return {
        "kvgraph_n_nodes": float(len(nodes)),
        "kvgraph_n_edges": float(len(pg.g.edges)),
        "kvgraph_n_read": float(n_read),
        "kvgraph_edges_per_node": float(len(pg.g.edges)) / n,
        "kvgraph_entity_frac": n_ent / n,
        # Mentions per node — the coreference compression language cannot do. 1.0 means nothing merged.
        "kvgraph_mentions_per_node": n_mentions / n,
        "kvgraph_compression": (n_tokens / n) if n_tokens else 0.0,
        "kvgraph_n_evicted": float(pg.n_evicted),
        "kvgraph_n_recalled": float(pg.n_recalled),
        "kvgraph_n_records": float(sum(len(c) for c in pg.records.values())),
        "kvgraph_n_gravestone_edges": float(sum(
            1 for e in pg.g.edges if e.relation.value == "gravestone_pointer")),
    }
