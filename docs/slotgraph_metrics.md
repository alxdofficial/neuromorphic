# slotgraph metrics — `baseline`

baseline: n=3 seeds. Every selection under two lenses — **decisiveness** (sharp given one input) and **cross-input diversity** (responds to the input = the topology signal). ✓/✗ = moved in the good direction.

## Structural metrics (per task: mae / babi)
| metric | dir | mae | babi |
|---|---|---|---|
| WRITE slot rank | ↑ | 6.452±1.616 | 6.924±2.812 |
| WRITE per-slot rank across inputs | ↑ | 25.647±0.911 | 8.186±3.905 |
| WRITE inter-slot cosine | ↓ | 0.698±0.050 | 0.747±0.017 |
| SELECT endpoint decisiveness | ↑ | 0.690±0.099 | 0.643±0.165 |
| SELECT endpoint cross-input diversity | ↑ | 0.055±0.014 | 0.023±0.017 |
| SELECT node coverage | ↑ | 0.833±0.036 | 0.771±0.036 |
| SELECT node-usage entropy | ↑ | 0.738±0.061 | 0.768±0.087 |
| SELECT per-node cross-sample variance | ↑ | 0.010±0.003 | 0.005±0.000 |
| SELECT edge distinctness | ↑ | 0.858±0.045 | 0.914±0.091 |
| SELECT self-loop frac | →0 | 0.000 | 0.000 |
| SELECT router id-vs-content | ↓ | 0.533±0.018 | 0.546±0.055 |
| SELECT routing-key input-dependence | ↑ | 14.881±2.374 | 6.339±2.467 |
| SELECT selection margin | ↑ | 0.562±0.120 | 0.500±0.222 |
| SELECT routing temperature | · | 0.997±0.003 | 0.997±0.003 |
| MP-READ mp_delta | · | 0.152±0.042 | 0.154±0.051 |
| OUTPUT memory rank | ↑ | 3.022±0.586 | 2.871±1.229 |

## Performance
| metric | dir | value |
|---|---|---|
| babi exact-match | ↑ | 0.328±0.162 |
| mae loss | ↓ | 6.530±0.006 |
| mae SHUF−REAL | ↑ | 0.427±0.034 |
| continuation loss | ↓ | 3.321±0.004 |
| continuation SHUF−REAL | ↑ | 0.484±0.025 |

## Depth profile (effective rank: write per LM-layer, then read per MP-hop)
- **mae** write-layer rank: 29.4 24.6 24.5 24.0 22.8 22.4 21.6 18.0 10.9 11.2 6.5 6.5
  - read-hop rank:   6.5 5.3 3.6 2.5 2.1 1.8
  - write-layer inter-slot cosine: 0.3 0.5 0.6 0.7 0.7 0.7 0.7 0.7 0.7 0.8 0.7 0.7
- **babi** write-layer rank: 29.4 24.6 24.6 23.7 22.6 22.3 21.3 17.5 10.0 10.7 6.9 6.9
  - read-hop rank:   6.9 5.4 3.1 2.1 1.6 1.3
  - write-layer inter-slot cosine: 0.3 0.5 0.6 0.7 0.7 0.7 0.7 0.7 0.7 0.8 0.7 0.7

## Node usage histogram over samples (sorted desc; flat = fixed roles, peaked = hub)
- **mae** baseline: 0.36 0.15 0.10 0.08 0.06 0.06 0.06 0.05 0.03 0.03 0.01 0.01 0.00 0.00 0.00 0.00
- **babi** baseline: 0.31 0.14 0.11 0.08 0.07 0.06 0.05 0.05 0.04 0.04 0.03 0.01 0.00 0.00 0.00 0.00

## Metric glossary (what / how / good direction)
| metric | means | how measured | good |
|---|---|---|---|
| WRITE slot rank | distinct directions among the 32 slots within one input | participation ratio of slot_final (final-layer slot hiddens) over the M slots, per example, mean | ↑ |
| WRITE per-slot rank across inputs | how much each slot position varies with the input (write input-dependence) | mean over slot positions of the participation ratio of slot_final[:,pos] across samples | ↑ |
| WRITE inter-slot cosine | redundancy among the slots (1 = identical) | mean pairwise cosine of the M slot vectors, per example, mean | ↓ |
| SELECT endpoint decisiveness | how peaked each edge's endpoint choice is given one input (1 = one-hot) | 1 − normalized entropy of the soft endpoint distribution, mean over edges | ↑ |
| SELECT endpoint cross-input diversity | does an edge pick different nodes for different inputs — THE topology signal | mean over edges of the normalized entropy of its argmax node pick across samples | ↑ |
| SELECT node coverage | fraction of nodes used as an endpoint somewhere in the batch | fraction of K nodes selected ≥1 time (pooled over samples) | ↑ |
| SELECT node-usage entropy | balance of pooled node usage (1 = uniform, 0 = one hub) | normalized entropy of pooled endpoint counts over the K nodes | ↑ |
| SELECT per-node cross-sample variance | do individual nodes swing in usage with the input (input-selective) vs fixed roles | per node, std across samples of its per-sample usage fraction; mean over nodes | ↑ |
| SELECT edge distinctness | fraction of edges with a distinct (src,dst) pair within an input | mean over samples of #unique (src,dst) / E | ↑ |
| SELECT self-loop frac | edges pointing a node to itself (should be 0 by construction) | fraction of edges with src == dst | →0 |
| SELECT router id-vs-content | fraction of routing key/query magnitude from the FIXED id-stream vs input-dependent content — high = id drowns content (a cause of input-blindness) | ‖id-projection‖/(‖content‖+‖id‖) through the node-key + edge-query heads, mean | ↓ |
| SELECT routing-key input-dependence | do the per-node routing keys vary across inputs, BEFORE the argmax (separates 'keys are fixed' from 'argmax kills variation') | mean over nodes of the participation ratio of the node key across samples | ↑ |
| SELECT selection margin | gap between the top-1 and top-2 endpoint probability (≈0 = fragile near-tie) | mean over edges of (p_top1 − p_top2) of the soft endpoint distribution | ↑ |
| SELECT routing temperature | softmax temperature; tiny → argmax saturates and input-gradient dies | exp(log_temp) | · |
| MP-READ mp_delta | how much the message-passing read changes the output vs a plain prepend | 1 − cos(slot_final, memory), mean over slots | · |
| OUTPUT memory rank | distinct directions in the prepended memory (compare to WRITE slot rank) | participation ratio of the final memory over the M slots, per example, mean | ↑ |
| PERF babi exact-match | babi answer accuracy | exact-match over babi val | ↑ |
| PERF mae loss | masked-reconstruction loss | recon CE on mae val | ↓ |
| PERF mae SHUF−REAL | example-specificity on mae (reliable) | loss(shuffled memory) − loss(real) | ↑ |
| PERF continuation loss | next-token loss | recon CE on continuation val | ↓ |
| PERF continuation SHUF−REAL | example-specificity on continuation (reliable) | loss(shuffled) − loss(real) | ↑ |