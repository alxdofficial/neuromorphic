# Exp 1 v5 — `graph_v5_baseline` design

Canonical reference for the v5 graph-structured memory variant. v5
supersedes the v4 lineage (`docs/exp1_graph_baseline.md`) by replacing
per-edge free-vector endpoints with a shared node bank + soft-pointer
edges.

## TL;DR

A shared node bank `N` of `K_node` continuous vectors lives alongside
`K_edge` edges. Edges store *query vectors* `(q_src, q_dst, state)` —
not endpoint vectors. At read time, endpoints materialize by soft-pointer
attention into `N`:

```
α_src = softmax(q_src · N^T / √d_node)
src   = α_src @ N
```

When two edges' queries land on the same `N[k]`, **they share the
underlying node vector by construction** — the substrate represents
cross-edge and cross-role node reuse natively. This was structurally
impossible in v4 (independent src/dst free vectors in disjoint trained
subspaces).

Per-window updates come from a single **HolisticUpdater** that fuses pin
+ node + edge information in one transformer, emitting node updates AND
edge proposals jointly. The node-update head uses Slot Attention
competitive softmax-over-slots, which is load-bearing — without it the
node bank collapses to a shared direction across windows.

## State

Per chunk, sampled fresh from learned `(μ, σ)` distributions
(no per-slot trained identity, no cross-chunk persistence):

| field | shape | what it is |
|---|---|---|
| `N` | `[B, K_node, d_node]` | shared node bank — soft-pointer target |
| `q_src` | `[B, K_edge, d_node]` | per-edge src query (points into N) |
| `q_dst` | `[B, K_edge, d_node]` | per-edge dst query |
| `state` | `[B, K_edge, d_state]` | per-edge relational content |

**Default sizes** (`graph_v5_K_*` config): `K_node=32, K_edge=60,
d_node=d_state=128`. Substrate floats = `K_node·d_node +
K_edge·(2·d_node + d_state) = 4096 + 23040 = 27,136` — within 4% of the
v4.2 bottleneck (26,180) for fair comparison.

`K_node << 2·K_edge` is intentional: 2·K_edge = 120 endpoint queries
must resolve into K_node = 32 distinct nodes, forcing ~3.75 edges per
node on average. This is the structural pressure for graph topology.

## Per-window protocol

```
1. Encode pins (+ sinusoidal PE)
2. HolisticUpdater(pins, N_old, edges_old) → (node_update, edge_proposals)
3. NodeGate blends node_update into N → N_new (RMSNorm'd)
4. Materialize all queries (existing + proposals) against N_new
5. slot_routing_on_endpoints: each edge picks one proposal by cosine
   on materialized endpoints (multiple edges CAN pick the same proposal
   → v4.2-style k-means convergence pressure)
6. EdgeGate per-edge anchor-biased blend → new q_src, q_dst, state
7. RMSNorm + per-row all-pad protection
```

### HolisticUpdater architecture

A single transformer with `[K_node + K_edge]` tokens:
- Cross-attention from tokens to pins (both nodes and edges see text)
- Self-attention over `(nodes ∪ edges)` — nodes see what edges want,
  edges see what nodes look like
- Position-wise FFN
- Per-position embeddings (init std=1.0, not standard 0.02 — see § Three
  load-bearing fixes below)
- AttnBlock building block from v4.2 (QK-Norm + KV-LN + post-attn-LN)

After the transformer stack, two output heads:

**Edge head:** standard projection — emits `(q_src, q_dst, state)`
proposals. Edge specialization comes from downstream routing (multiple
edges can pick the same proposal → convergence).

**Node head: Slot Attention competitive write.** Uses post-transformer
node tokens as queries against pins (separate K, V projections) with
**softmax over slots** (Locatello et al. 2020). Pins compete to be
absorbed by slots; this is what forces node specialization. Without it,
standard cross-attn lets every slot attend to all pins, all slots get
similar updates, bank collapses to rank-1 within a few windows.

### Cross-position whitening

After the transformer stack and before the output heads:
```python
tokens = tokens - tokens.mean(dim=1, keepdim=True)
```
Subtracts the mean across positions per channel — forces the
homogeneous component to zero. Addresses **rank collapse** in pure
attention stacks (Dong et al. 2021 — depth → token representations
collapse to rank-1 subspace exponentially).

## Readout

Directional R-GCN-style projection (carried over from v4.2). When
multiple edges' soft pointers land on the same `N[k]`, the readout
projects that SAME `N[k]` vector role-specifically:

```
h = W_src(endpoint_src) + W_dst(endpoint_dst) + W_state(state)
```

Sharing happens at the substrate (`N`), not at the projection. Plus one
layer of cross-edge self-attention before final projection to d_llama.

## Three load-bearing fixes

These three were discovered through ablation during v5.1 development.
The first v5 attempt without them collapsed within 4 windows
(cross-slot cos = 0.99). Each is necessary; removing any one alone
restores the collapse.

1. **Slot Attention competitive write** for the node-update head.
   softmax-over-slots forces per-slot specialization.
2. **Cross-position whitening** of the transformer output.
   Counteracts rank collapse in deep attention stacks.
3. **Per-position embeddings** at `std=1.0` (not 0.02).
   The token inputs are statistically identical samples from `(μ, σ)`;
   the positional signal needs to be comparable to the content signal
   for the transformer to keep positions distinct.

## Citations

- **Slot Attention** — Locatello et al. 2020, "Object-Centric Learning
  with Slot Attention". Source of the softmax-over-slots competitive
  write trick.
- **Rank collapse** — Dong et al. 2021, "Attention is Not All You Need".
  Identifies the exponential rank-1 collapse in pure attention stacks.
- **σReparam** — Zhai et al. 2023 (Apple), "Stabilizing Transformer
  Training by Preventing Attention Entropy Collapse". More principled
  spectral-norm-based remedy; we use the cheaper whitening alternative.
- **Directional R-GCN readout** — Schlichtkrull et al. 2018, "Modeling
  Relational Data with Graph Convolutional Networks". Source of
  separate W_src/W_dst projections.
- **TPR (role-filler binding)** — Smolensky 1990. Conceptual basis for
  treating src/dst as shared-substrate + role disambiguation, though
  v5.1 ultimately uses directional W_src/W_dst rather than additive role
  embeddings.

## Config defaults (`ReprConfig`)

```python
graph_v5_K_node: int = 32              # shared node slots
graph_v5_K_edge: int = 60              # edges
graph_v5_d_node: int = 128
graph_v5_d_state: int = 128
graph_v5_d_updater: int = 384          # internal token dim
graph_v5_updater_layers: int = 4
graph_v5_updater_n_heads: int = 16
graph_v5_node_gate_init_bias: float = 0.5   # sigmoid(-0.5) ≈ 0.38
graph_v5_edge_gate_init_bias: float = 1.0   # sigmoid(-1) ≈ 0.27
graph_v5_init_log_sigma: float = 0.0        # init σ = 1.0 for N/state/q noise
graph_v5_read_temperature: float = 1.0      # soft-pointer softmax temp
graph_v5_readout_n_heads: int = 4
graph_v5_readout_d_hidden: int = 512
```

## Telemetry surfaced for monitoring

| key | what it measures | healthy range |
|---|---|---|
| `graph_v5_node_gate_mean_avg` | mean node-update gate per window | 0.3–0.6 |
| `graph_v5_edge_gate_mean_avg` | mean edge-update gate per window | 0.1–0.3 (anchor-leaning) |
| `graph_v5_edge_pick_affinity_avg` | mean cosine of edge to picked proposal endpoint | rising over training |
| `graph_v5_edge_frac_selfpick_avg` | edges picking their own slot's proposal | low (0–0.05) is fine |
| `graph_v5_edge_pick_entropy_avg` | entropy of pick_count distribution | lower = concentrated reuse |
| `graph_v5_edge_src_entropy` | soft-pointer entropy per src query | lower = sharper pointing |
| `graph_v5_edge_dst_entropy` | same for dst queries | (current concern: stuck near max — temperature fix needed) |
| `graph_v5_unique_picks_frac` | fraction of distinct bank entries touched by argmax(α_src ∪ α_dst) | high vs K_node = good bank utilization |
| `graph_v5_cross_role_overlap` | fraction of bank entries appearing as both src AND dst | **THE thesis signal** — higher = real cross-role reuse |
| `graph_v5_endpoint_cos_mean` | cross-edge cosine of materialized endpoints | low = endpoints diverse (good), high = soft pointers too flat (current state) |

## Empirical results

See `docs/repr_learning_results.md` § 0–§ 0.1. Summary:

| variant | val_recon | top1 | substrate floats |
|---|---:|---:|---:|
| **v5.4 (current)** | **2.079** | **57%** | **25,984** |
| v5.1-first | 2.057 | 55.9% | 27,136 |
| v4.2 (prior best) | 2.696 | 52.9% | 26,180 |

v5.1-first beats v4.2 by 0.64 val_recon at matched substrate floats; v5.4
closes the loop with a graph-aware readout at slightly tighter budget.
Cross-role overlap reaches 0.38 (12 of 32 bank entries serving both src
and dst roles), which is the load-bearing thesis test and is structurally
impossible in v4.

## Status

- Implementation: `src/repr_learning/graph_substrate_v5.py` +
  `encoder.py:GraphV5BaselineEncoder`.
- Tests: `tests/test_graph_v5.py` (5/5 passing — forward shape,
  backward grad flow, chunk-fresh init, soft-pointer trainable,
  materialize_endpoints unit).
- v5.1-first and v5.4 production training complete; v5.4 is the canonical
  current variant.
- Multi-seed reruns not yet done (single-seed caveat).

## Known concerns / next iterations

1. **Soft-pointer entropy stuck near max** — `edge_src_entropy ≈ 3.42`
   vs max `log(32) = 3.47` means soft pointers are very flat. Edges
   resolve endpoints as near-uniform mixtures of N rather than sharp
   pointers at specific slots. This is the temperature problem:
   `q · N^T / √d_node` gives scores with std ≈ 1, producing nearly
   uniform softmax over 32 entries. Two cheap fixes available:
   - Lower `graph_v5_read_temperature` from 1.0 to 0.3 (sharper softmax)
   - Add an entropy penalty as aux loss
2. **Multi-seed validation** — all current numbers are N=1.
3. **Larger context (tranche 2/3)** — current results at chunk=4096
   only. Behavior at chunk=8192/16384 is unverified.
