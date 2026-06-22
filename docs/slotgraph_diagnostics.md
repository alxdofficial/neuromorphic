# slotgraph diagnostics — the message-passing read + fixed partition makes the graph bind

**Date:** 2026-06-22 · **Model:** `slotgraph` (current graph arm) · **Backbone:** frozen SmolLM2-135M
(d=576) · **Objective:** mixed 4-task (mae / babi / continuation / condrecon_bio), 1024→32, 4000 steps.

## TL;DR
The earlier slotgraph had an **inert** emergent graph — the structure heads sat at random init because a
plain-prepend read never used the topology (see "Background" below). This rebuild fixes that and, for the
first time, **the graph appears to actually bind**:
- **bAbI EM 37.5%** — cohort best (icae 26.2%), and **above** both the old inert slotgraph (35.2%) and the
  id-tag-only baseline (35.5%) → the topology-using read adds ~2 pts beyond what id-tags alone gave, +11 vs icae.
- **Highest SHUF−REAL on all four tasks** — the memory is the *most example-specific* in the cohort (every
  prior graph model had the *lowest*). The qualitative flip from "pools" → "binds."
- **Edges are sharp and valid**: endpoint entropy 54% of max (was ~98% = random), 0% invalid (edges→nodes).
- On the **trained** model the topology heads get strong gradient (topo/content ratio 4.6, was ~0.001 = starved).

It's not a clean victory: the graph **hub-collapses** (only ~6 node slots get used, top-2 absorb 76% of edges)
and memory eff-rank is low (~3.1). And attribution (topology vs the extra MP params) still needs the ablations.

## What changed (the rebuild)
1. **Multi-hop message-passing READ** (`_mp_read`): instead of plain-prepending the slots, run a residual
   GNN over the predicted graph — each node relays its own state + edge-routed neighbour messages
   (`msg=[source-node ; edge-content]`), #hops = the graph's diameter (reachability saturation, capped at 5).
   This makes the prepended memory a *function of the topology*, so the edge heads get real loss gradient.
   Output is still M=32 vectors.
2. **FIXED node/edge partition** (slots 0–15 nodes, 16–31 edges) — role is assigned by position, not
   predicted. A noisy role head made the edges→nodes mask unreliable for most of training; fixing it makes
   "who is a node" 100%-reliable from step 0. More faithful to TokenGT (types given, connectivity learned).
3. **Edges→nodes by construction** — endpoint logits are masked to the node pool *before* the hard pick, so
   every edge is node→node (invalid_edge_frac = 0).
4. **MEAN aggregation** — divide each node's aggregate by its incident-edge count. (Sum aggregation
   overflowed bf16 on a concentrated-graph batch → NaN at step 271 on the first run; mean bounds it.)

## Capacity — matched on BOTH axes (slotgraph is the leanest arm)
Two capacities matter: trainable params (compute budget) and the read-interface floats (memory budget the
decoder sees). slotgraph matches the cohort on both — it is *not* advantaged.

| variant | trainable params | read floats (M × d) |
|---|---|---|
| beacon | 7,006,465 (7.01M) | 32 × 576 = 18,432 |
| vqicae | 6,997,505 (7.00M) † | 32 × 576 = 18,432 |
| autocompressor | 6,950,593 (6.95M) | 32 × 576 = 18,432 |
| icae | 6,932,161 (6.93M) | 32 × 576 = 18,432 |
| ccm | 6,932,161 (6.93M) | 32 × 576 = 18,432 |
| **slotgraph** | **6,909,188 (6.91M)** | **32 × 576 = 18,432** |

The MP modules cost ~1.0M (`msg` 2d×d + `update` d×d, both bias-free); offset by trimming the encoder-LoRA
r104→r85 (−1.09M), so slotgraph lands at the **bottom** of the 6.91–7.01M band with the **identical** 18,432-float
prepend budget (the message passing transforms the 32 slot vectors in place — it adds none). slotgraph's only
buffer is the tiny fixed id table (32×576 ≈ 18K). † vqicae additionally carries a non-trainable EMA codebook
buffer (8192×256 ≈ 2.1M floats) — extra stored capacity the others don't have.

→ slotgraph's bAbI win is **not** a capacity advantage: it has the fewest trainable params and the same read budget.

## Results (4000 steps, REAL loss; bAbI EM + binding gate)

| variant | mae ↓ | **babi EM ↑** | babi SHUF−REAL | cont ↓ | condrec ↓ |
|---|---|---|---|---|---|
| **slotgraph** | **6.540** | **37.5%** | **+0.289** | **3.319** | **2.301** |
| icae | 6.581 | 26.2% | +0.120 | 3.363 | 2.353 |
| vqicae | 6.690 | 23.4% | +0.030 | 3.494 | 2.365 |
| ccm | 6.610 | 22.7% | +0.038 | 3.373 | 2.358 |
| autocompressor | 6.581 | 21.5% | +0.011 | 3.365 | 2.391 |
| beacon | 6.587 | 18.8% | +0.005 | 3.405 | 2.478 |

slotgraph leads the cohort on **every task's REAL loss** and on bAbI EM. SHUF−REAL (shuffle the memory across
the batch; how much does loss rise?) measures how *example-specific* the memory is — slotgraph is highest on
**all four** (mae +0.447, babi +0.289, continuation +0.504, condrecon +0.036) vs icae's +0.330/+0.120/+0.371/+0.029.
Prior graph models were always *lowest* on SHUF−REAL ("pools, never binds"); this one is highest.

## Did the topology form this time? (diagnostics)
**Gradient flow (trained model, `slotgraph_gradflow.py`).** The structure heads are actively trained, not
starved: `src_head` 3.31, `dst_head` 3.39, `msg` 0.56, `update` 1.15 → **topology/content ratio 4.6** (the old
inert model was ~0.001–0.01). The message-passing read is the source of this gradient.

**Structure canaries (`slotgraph_diag.py`, 64 bAbI examples).**

| signal | value | reading |
|---|---|---|
| edge fraction | 0.500 | fixed partition (16 nodes / 16 edges) |
| **endpoint entropy** | **1.875 / 3.466 (54%)** | edges are SHARP/committed (was ~98% = random) |
| **invalid edges** | **0%** | edges→nodes holds by construction |
| node-target usage | **6/32 used; top-2 = 76%** | ⚠️ HUB-COLLAPSE: edges pile onto a few nodes |
| memory eff-rank | 3.18 / 32 | low (structured low-rank, not generic — see SHUF−REAL) |

**One bAbI example — node slots in UMAP, predicted edges as src→dst arrows:**

![slotgraph UMAP + edges](figures/slotgraph_umap_example.png)

A *real* connected subgraph forms (16 valid node→node edges, 0 invalid; node 0 is a hub linking 1/2/3/4) — but
11 of the 16 node slots sit unused. This is a clear step up from the old model (random edges, 75% pointing at
non-node slots), now partial via hub-collapse.

**Histogram panel** — edge-endpoint entropy peaks at ~1.9 (far from the 3.47 random line = sharp); node-target
usage is dominated by slot 0 (~1200 hits); memory eff-rank clusters ~3.0–3.3:

![slotgraph histograms](figures/slotgraph_histograms.png)

## Honest caveats (what's not settled)
1. **Attribution.** The +2 EM over the id-tag baseline is real but modest, and that baseline came from the
   *old* architecture. To prove the win is the *topology* and not just spending ~1.0M params on `msg`/`update`
   instead of LoRA, re-run two ablations under the **new** code at the same 6.91M budget:
   `--slotgraph-no-mp-read` (injection-only, plain prepend) and `--slotgraph-no-structure` (id-tagged ICAE).
2. **Hub-collapse.** Only ~6 node slots are ever targeted (top-2 = 76%); 11 sit idle. The graph works but
   doesn't use its node capacity — a load-balancing / diversity lever is open.
3. **Low memory eff-rank (~3.1).** Possibly mild over-smoothing from the multi-hop relay. The high SHUF−REAL
   says it's a *structured* low-rank (input-dependent binding), not a generic collapsed pool — but worth watching.

## Next levers
- **Attribution ablations** (above) — the prerequisite before claiming the topology earns its keep.
- **Anti-hub-collapse** — encourage edge endpoints to spread across nodes (e.g. a load-balance term or a
  competition over node-targets), so the graph uses more than ~6 nodes.
- **Rank** — if over-smoothing is the cause of the low eff-rank, a GCNII-style initial residual or fewer
  hops would help; the high SHUF−REAL suggests it may not be hurting yet.

## Background — why the previous version was inert
The prior slotgraph predicted node/edge role per slot and used a **plain-prepend read**. Ablation showed the
emergent graph contributed nothing (structure OFF babi 0.355 ≈ ON 0.352): the heads sat at random init, got
~100× less gradient than the content path, and 75% of edges pointed at non-node slots. Diagnosis: the read
never used the topology, so there was no gradient pressure to form it. This rebuild (topology-using read +
fixed partition + validity-by-construction) is the fix, and the diagnostics above are the before→after.

---
*Repro:* `scripts/diagnostics/slotgraph_diag.py` (structure + visuals), `slotgraph_gradflow.py` (per-component
gradient on the trained model), `mixed_band_gate_eval.py` (REAL/SHUF/OFF band + bAbI EM), `smoke_slotgraph_mpread.py`
(fresh-model gradient/movement). Ablations: `train.py … --slotgraph-no-mp-read` / `--slotgraph-no-structure`.
