# Problem Formulation: Neural Memory Graph for Language Modeling

## What We Are Building

A language model augmented with a persistent, brain-inspired memory graph. The
system has three components that operate at different levels:

1. **Language Model (Cortical Columns)**: A scan-based (linear recurrence) LM with
   16 cortical columns, 7 layers, D=2048. Processes tokens causally. Trained by
   backprop. This is the "fast" system — standard next-token prediction.

2. **Neural Memory Graph**: A network of 1024 neurons with sparse random connectivity
   (96 presynaptic connections each). At every token, neurons receive messages from
   neighbors, integrate them with internal state, and broadcast new messages. 16 "port
   neurons" interface with the LM's cortical columns. This is the "slow" system —
   persistent memory that evolves with every token but is NOT trained by backprop.

3. **Neuromodulator**: An RL-trained policy network (~10M params) that observes every
   neuron's state and adjusts primitives (what neurons broadcast), connection weights
   (routing), and decay (temporal persistence) every 256 tokens. This is the "meta"
   system — it learns HOW to configure the memory graph.

## Current Architecture in Detail

### The LM (What Works Well)

The LM is a standard scan-based sequence model:
- Embedding → 7 scan layers (h[t] = a*h[t-1] + b[t], parallelizable via FLA kernel)
- Per-column predictive coding modules (16 independent PCMs)
- Memory injection: `logits = output_head(H + gate * mem_signals)`
- ~93M trainable parameters
- Throughput: ~85K tok/s without memory (RTX 4090, BS=4)

This part is well-understood and works. The scan layers are fully parallel over T=2048
tokens. No issues here.

### The Memory Graph (Core Design Challenge)

1024 neurons, each with:
- `h` [128 dims] — internal state (persistent across tokens and chunks)
- `prev_messages` [128 dims] — last outgoing message
- `primitives` [128 dims] — modulates outgoing messages (neuromod-controlled)
- `decay` — state persistence (neuromod-controlled)
- 96 connection weights — presynaptic routing (neuromod-controlled)

**Per-token dynamics (sequential, every token):**
```
1. received = A @ prev_messages              # weighted sum of presynaptic messages
   received[:, :16] += cc_signals[:, t]      # port neurons also get LM signal
2. h = decay * h + (1-decay) * received      # temporal integration
3. prev_messages = tanh(h * primitives)       # compute outgoing message
```

The LM reads port neurons' messages at every token as `mem_signals`.

### The Neuromodulator (RL Agent)

Every 256 tokens (8 times per 2048-token chunk):
- Observes all 1024 neurons (387 dims each: primitives, mean activity, decay, entropy, plasticity)
- Outputs per-neuron actions (225 dims: delta_primitives[128] + delta_conn_weights[96] + delta_decay[1])
- Shared 3-layer MLP (hidden=2048, ~10M params) applied to each neuron independently

**RL training**: Per-segment REINFORCE with discounted returns (gamma=0.99) and batch-mean baseline.

---

## Open Problems

### Problem 1: Throughput — Sequential Memory in a Parallel World

**The core tension**: The memory graph's per-token dynamics are inherently sequential.
At each token, every neuron's state depends on its neighbors' messages from the previous
token, which depend on THEIR neighbors from the token before that, and so on. The
nonlinearity (tanh) and inter-neuron coupling (A @ messages) break all parallelization.

**Current cost**: One `torch.bmm([BS, 1024, 1024] × [BS, 1024, 128])` per token.
At BS=4, that's ~2 GFLOP per token × 2048 tokens = ~4 TFLOP per chunk. The sequential
loop adds Python overhead on top. Total memory path: ~100ms per chunk.

**Result**: 42K tok/s with memory vs 85K tok/s without. The memory graph is a 50%
throughput tax. For a research model this is acceptable, but scaling to larger N or
longer T will make it worse (cost scales as O(T × N²)).

**Why this can't be trivially parallelized**:
- The recurrence `h[t] = f(h[t-1], A @ messages[t-1])` has a nonlinear dependency on
  the full graph state at the previous timestep
- If we drop the nonlinearity (tanh) and the inter-neuron coupling (use diagonal
  recurrence instead of matrix), we get a standard linear scan — parallelizable via
  associative scan, but neurons can't communicate at every token
- If we drop inter-neuron coupling but keep nonlinearity, each neuron is an independent
  nonlinear RNN — parallelizable across neurons but still sequential over T per neuron

**Possible future directions**:
- Sub-segment message passing: diagonal scan for K tokens (parallel), then one round of
  message passing, repeat. Message passing happens every K tokens instead of every token.
  Tradeoff: K=1 (current, faithful) vs K=32 (fast, coarser communication).
- Custom CUDA kernel: fuse the per-token loop (bmm + element-wise ops) into a single
  kernel to eliminate Python loop overhead and kernel launch latency.
- CUDA graphs: capture the static computation graph and replay it.
- Smaller N: N=256 with a matrix-valued associative scan would allow true per-token
  parallelization, but reduces graph capacity.

### Problem 2: RL Credit Assignment — Which Actions Mattered?

**The setup**: The neuromodulator takes 8 actions per chunk (one every 256 tokens).
Each action modifies primitives, connection weights, and decay for all 1024 neurons.
The reward is the CE loss on the next 256 tokens.

**What we currently do**:
- Per-segment CE loss as reward (8 rewards per chunk)
- Discounted returns: G_t = r_t + 0.99*r_{t+1} + ... (credit early actions for later gains)
- Batch-mean baseline: baseline_t = mean across batch of G_t (zero parameters)
- Per-step advantages: A_t = G_t - baseline_t

**What works**:
- Each of the 8 actions gets its own advantage (not a single scalar for the whole chunk)
- Discounting credits early structural changes for downstream improvements
- Batch-mean baseline centers advantages at zero per step

**Open questions**:

**Q1: Is the reward signal informative enough?**
The reward is negative CE loss on the NEXT segment's tokens. But the neuromod's action
affects the memory graph's structural configuration (primitives, weights, decay) — it
doesn't directly predict tokens. The mapping from "adjust neuron 500's primitives by
delta" to "segment CE loss decreases by 0.01" is extremely indirect. The neuromod has
to learn that certain structural configurations help the LM predict better, without any
direct gradient signal telling it which configurations are good.

**Q2: Is per-segment reward the right granularity?**
256 tokens per segment. The neuromod adjusts the graph, then the graph processes 256
tokens of text. The CE loss on those 256 tokens is the reward. But:
- The first ~10 tokens after an action barely reflect the structural change (signals
  haven't propagated far through the graph yet)
- The last tokens in the segment fully reflect the changed graph state
- Should we weight later tokens more heavily in the segment reward?

**Q3: Can the neuromod distinguish "my action helped" from "this text was easy"?**
The batch-mean baseline subtracts the average return across batch elements. This helps
when different batch elements see different text. But within a batch element, the
baseline can't distinguish whether low loss came from a good neuromod action or from
inherently predictable text. A learned value function (critic) would help here but
adds parameters and compute.

**Q4: Is the action space right?**
Each action is 225 continuous dims per neuron × 1024 neurons = 230,400 total action
dimensions per step. The policy outputs are sampled from a Gaussian with learned std.
The action is clamped to ±1.0 (L1 normalization bounds the effect). Questions:
- Is 128 dims of delta_primitives too many? Could a lower-rank action suffice?
- Should conn_weight deltas be per-connection (96 dims) or could we use a shared
  scaling factor (1 dim) to reduce the action space?
- The same MLP is applied to all neurons — is the obs sufficient for the MLP to
  differentiate neuron roles?

**Q5: Reward horizon — should discounting go beyond one chunk?**
Currently gamma=0.99 with H=8 steps. The effective horizon is ~100 steps before
discounting to <37%. But the neuromod only sees 8 steps per chunk, then a new chunk
starts (potentially different text, but same graph state). Cross-chunk effects of
neuromod actions are invisible to the current RL setup. A structural change that pays
off 5 chunks later gets zero credit.

### Problem 3: Memory Graph Signal Dynamics — Does It Actually Help?

**The fundamental question**: Does the memory graph provide information to the LM that
the LM's own scan layers can't already capture?

The LM scan layers already have per-layer carries that persist across chunks. They
implement h[t] = a*h[t-1] + b[t] — a linear recurrence that can capture long-range
dependencies. The memory graph adds a SECOND recurrence that's nonlinear and graph-
structured, but communicates with the LM through only 16 port neurons.

**Potential value of the memory graph**:
- Nonlinear dynamics (tanh) can represent patterns that linear scan cannot
- Graph structure gives spatial organization — different neurons can specialize
- The neuromodulator can adapt the graph at inference time (the LM's scan weights are frozen)
- 1024 neurons × 128 dims = 131K state dimensions vs 2048 dims per scan layer

**Potential problems**:
- The 16 port neurons are a severe bottleneck: 2048 dims of LM state compressed to
  16 × 128 = 2048 dims of port messages. The memory graph's internal state (131K dims)
  can only influence the LM through this narrow interface.
- The memory contribution is gated by `mem_gate` (starts at sigmoid(0)=0.5) and added
  to H before the output head. If the LM learns to ignore memory (gate → 0), the
  neuromod gets no reward signal and can't learn.
- The CC signal enters port neurons as a raw additive input. Scale mismatch: CC signals
  are O(1), neighbor messages are O(0.2). Port neurons are dominated by the LM signal.

### Problem 4: Structural Plasticity — Is It Meaningful?

**Current design**: Every 4 segments (~1024 tokens), co-activation-based plasticity runs.
Connections with negative phi (anti-correlated firing patterns) are pruned. New connections
form toward the highest-phi unconnected neuron (80%) or random (20%). Connection weights
are L1-normalized (energy conservation). Vectorized — no Python loop over neurons.

**Remaining concerns**:
- The co-activation EMA (decay=0.995) takes ~200 segments to converge, so early plasticity
  decisions are based on noisy phi estimates
- Graph topology is shared across batch elements — can't specialize per document
- Random exploration (20% of regrowth) may not be enough to discover useful long-range connections
  — it doesn't know WHICH specific connections are useful

### Problem 5: Document Boundary Semantics

**Current behavior**: At document boundaries (EOT token):
- Within a chunk: decay is set to 0 for that token position (h ← received, no carry)
- At chunk boundaries: only h and prev_messages are zeroed; primitives, conn_weights,
  decay, and plasticity metrics persist

**This means**: The graph's structural configuration (what neurons broadcast, how signals
route, how persistent each neuron's memory is) persists across documents. Only the
dynamic state (what neurons are currently "thinking about") resets.

**The design intent**: Primitives and weights represent learned "concept atoms" and
routing patterns. These should accumulate over training and across documents. The
neuromodulator learns to build a graph configuration that's useful across many documents.
At inference, the graph starts with a well-configured structure and quickly adapts its
dynamic state to the new document.

**Open question**: Is this the right split? Should some structural state (conn_weights)
also partially decay at document boundaries to prevent overfitting to recent documents?

---

## Summary of What's Working vs What's Uncertain

| Aspect | Status | Confidence |
|--------|--------|------------|
| LM scan layers | Working, well-understood | High |
| Memory graph per-token dynamics | Implemented, correct | High |
| Throughput | 42K tok/s, 50% overhead | Acceptable |
| Per-segment RL rewards | Implemented | Medium — informative enough? |
| Credit assignment | Per-step advantages | Medium — indirect reward signal |
| Primitives as message modulation | Implemented | Medium — do neurons differentiate? |
| Structural plasticity | Implemented | Low — random rewiring meaningful? |
| Memory actually helping LM | Not yet proven | Unknown — need training run |
| Scale mismatch (CC vs graph signals) | Known issue | Needs investigation |
| Cross-chunk RL credit | Not captured | Known gap |

## Next Steps

1. **Long training run** to see if memory helps (lower loss than no-memory baseline)
2. **Monitor neuromod behavior**: do primitives diversify? do conn_weights develop structure?
3. **Ablation**: memory graph with random (untrained) neuromod — does RL matter?
4. **Scale investigation**: does the 16-port bottleneck limit memory usefulness?
5. **Speed optimization**: sub-segment message passing, custom kernels
