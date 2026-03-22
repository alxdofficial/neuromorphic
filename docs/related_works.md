# Related Works

A survey of prior work related to the Neural Memory Graph architecture, organized
by category. For each work, we note what it does, how it relates to our model, and
what distinguishes our approach.

---

## 1. Graph Neural Networks as Memory

**Graph Networks** (Battaglia et al., 2018) present a unified framework for
message-passing on graphs — nodes update based on neighbor messages. Our memory graph
is structurally a graph network, but with two key differences: our graph is outside
autograd (controlled by RL, not gradients), and it runs one hop per token continuously
rather than a fixed number of message-passing steps per input.

**Relational Memory Core** (Santoro et al., 2018) maintains a bank of memory vectors
that interact via multi-head attention at each time step. Both RMC and our model have
interacting memory vectors, but RMC uses full attention (quadratic in slots) while we
use sparse message passing (linear in connections). RMC is differentiable and resets
between sequences; our graph is RL-controlled and persistent.

**Temporal Graph Networks** (Rossi et al., 2020) maintain persistent node memory
updated by message passing over time on dynamic graphs. Shares the concept of persistent
node state updated by message passing, but targets graph-level tasks (link prediction)
rather than language modeling, and is fully differentiable.

**What's novel in our model**: Using a persistent, non-differentiable neuron graph with
per-token message passing as memory for a language model, with dynamics controlled by RL
rather than gradient descent, appears to be without direct precedent.

---

## 2. Differentiable Memory Systems

**Neural Turing Machine** (Graves et al., 2014) and **Differentiable Neural Computer**
(Graves et al., 2016) couple a neural controller with an external memory bank accessed
through differentiable attention-based read/write heads. The NTM/DNC memory is a passive
matrix read/written by a controller; our memory is an active network of neurons that
autonomously propagate signals. NTM addressing is differentiable attention; our addressing
is fixed sparse connectivity modulated by the neuromodulator.

**Memory Networks** (Weston et al., 2014; Sukhbaatar et al., 2015) store sentences as
vectors and perform multi-hop attention for QA. Multi-hop reasoning in MemNets
corresponds to multi-step message propagation in our graph (K tokens = K hops), but
MemNets store discrete items retrieved via attention while our memory stores continuous
activation patterns in a neuron graph.

**Modern Hopfield Networks** (Ramsauer et al., 2020) prove that transformer attention IS
the update rule of a continuous Hopfield network. Hopfield retrieval is a single global
energy minimization; our retrieval is multi-hop signal propagation through sparse
connectivity. The Hopfield perspective reveals that our model replaces attention-based
memory retrieval with an entirely different mechanism (graph dynamics).

**The Kanerva Machine** (Wu et al., 2018) uses Bayesian updates for distributed memory,
analogous to our decay-gated integration. Both blend new information with existing state,
but Kanerva is differentiable and was applied to image generation.

---

## 3. RL for Memory / Plasticity Control

**RL-NTM** (Zaremba & Sutskever, 2015) is the most direct ancestor — it uses REINFORCE
to learn discrete memory addressing in a Neural Turing Machine. Our model extends this
principle dramatically: instead of RL controlling only where to read/write, our
neuromodulator controls ALL aspects of memory behavior — neuron primitives (what neurons
broadcast), connection weights (routing), decay (temporal persistence), and structural
plasticity (which connections exist). Our RL agent acts on 1024 neurons simultaneously
with 225-dimensional per-neuron actions, far beyond RL-NTM's scope.

**Learning to Reinforcement Learn** (Wang et al., 2016) trains an LSTM via RL such that
the recurrent dynamics themselves implement a learned RL algorithm. Our neuromodulator is
analogous — it is learning a memory management policy via REINFORCE. If the policy
generalizes, it constitutes a learned rule for memory management.

**Hebbian Meta-Learning** (Najarro & Risi, 2020) searches for synapse-specific Hebbian
learning rules via evolution, allowing networks to self-organize from random weights.
Both approaches separate "what to learn" (plasticity rules) from "what is learned"
(network state). Their rules are local (pre/post-synaptic only); our neuromodulator is
global (observes all neurons, outputs coordinated actions).

**Mem-alpha** (Wang et al., 2025) uses RL to train LLM agents to manage complex memory
systems through interaction. Similar RL-for-memory philosophy, but operates at the
symbolic level (deciding what text to store) while our neuromodulator operates at the
sub-symbolic level (adjusting continuous-valued primitives and weights).

---

## 4. Neuromodulation in Neural Networks

**Differentiable Plasticity** (Miconi et al., 2018) adds Hebbian plastic connections
where each weight is `w_fixed + alpha * w_plastic`, with `w_plastic` updated by
pre/post-synaptic product and `alpha` trained by backprop. Both add experience-dependent
connection modification, but their plasticity is passive (triggered by activity) while
ours is active (the neuromodulator decides how much to change each weight).

**Backpropamine** (Miconi et al., 2019) is the closest prior work to our neuromodulation
approach. It adds a network-generated neuromodulatory signal M(t) that gates plasticity:
`w_plastic += M(t) * alpha * pre * post`. Critical differences:
1. Backpropamine's neuromodulator is a scalar signal; ours outputs 225-dim per-neuron
   actions for 1024 neurons
2. Backpropamine is fully differentiable; ours is trained by REINFORCE
3. Backpropamine modulates Hebbian update rates; our neuromodulator directly sets
   primitives, weights, and decay — controlling what neurons represent, not just how
   fast they learn
4. Our neuromodulator simultaneously controls graph topology through structural plasticity

---

## 5. State Space Models / Linear Recurrence

**S4** (Gu et al., 2021) demonstrated that structured state spaces with HiPPO
initialization can model long-range dependencies with parallel training. Our HGRN-style
diagonal scan is a simplification of this framework.

**Mamba** (Gu & Dao, 2023) introduces selective (input-dependent) state spaces, achieving
transformer-quality LM with linear complexity. Our scan layers are in the same family.
The fundamental difference: Mamba's hidden state is its only memory; our model supplements
scan layers with a 1024-neuron persistent graph for cross-sequence information.
Mamba-2's duality result (SSM = restricted attention) shows our memory graph provides
a complementary, non-attention-based retrieval mechanism.

**HGRN / HGRN2** (Qin et al., 2023-2024) is the direct basis of our scan layers — we
use the FLA HGRN kernel. HGRN2's state expansion makes the recurrent state larger; our
approach instead adds a qualitatively different memory system (neuron graph).

**Griffin / RecurrentGemma** (De et al., 2024) validates gated linear recurrence at scale,
matching Llama-2 with 6x fewer training tokens. Our scan layers are in the same family as
their RG-LRU. Griffin uses local attention for recall; we use a persistent memory graph.

**xLSTM** (Beck et al., 2024) revives LSTMs with matrix memory (mLSTM) and exponential
gating (sLSTM). Competitive with transformers at 7B scale. Like HGRN2, enriches the
recurrent state; our approach adds a separate memory system.

**DeltaNet** (Yang et al., 2024) uses the delta rule to enable forgetting/overwriting in
linear attention — the same motivation as our decay-gated integration. DeltaNet solves
this within differentiable recurrence; we solve it with an RL-controlled external memory.

**RWKV** (Peng et al., 2023-2024) uses linear attention approximation, trained up to 7.5B
params. Same scan-based philosophy as our LM backbone. Like Mamba, limited to fixed-size
state for long-range memory.

---

## 6. Persistent / Lifelong Memory for LMs

**Memorizing Transformers** (Wu et al., 2022) augment transformers with non-differentiable
kNN memory of past key-value pairs, growing linearly with time. Our memory is fixed-size
(1024 neurons) but continuously updated — compressed and integrated rather than verbatim.

**Infini-Attention** (Munkhdalai et al., 2024) compresses earlier segments into a
fixed-size memory matrix via linear attention. Both achieve infinite context with
fixed-size memory and segmented processing. Infini-Attention stays within the attention
framework; our memory is a qualitatively different system (graph dynamics).

**Titans** (Behrouz et al., 2025) is the most conceptually similar recent work. Both
have: (1) persistent memory that updates at test time, (2) a mechanism for deciding what
to memorize (Titans: surprise gradient; ours: neuromodulator policy), (3) separation of
short-term and long-term memory. Key differences: Titans' memory is differentiable
(meta-learning via in-context gradient descent); ours is non-differentiable (RL-controlled).
Titans' memory is parameter-based (weight updates); ours is activation-based (neuron
state dynamics). Our design principle is that memory should be an environment, not a
differentiable module.

**Block-Recurrent Transformers** (Hutchins et al., 2022) apply transformer layers
recurrently with LSTM-style gates. Both use recurrent state that persists across segments
with linear complexity. Their recurrence is attention-based and differentiable; ours is
graph-based and RL-controlled.

**Recurrent Memory Transformer** (Bulatov et al., 2022) adds memory tokens processed by
the same attention mechanism. Elegant but limited: a handful of memory tokens vs our 1024
neurons with graph connectivity.

---

## 7. Predictive Coding in Neural Networks

**Rao & Ballard** (1999) proposed hierarchical predictive coding in visual cortex:
feedback carries predictions, feedforward carries prediction errors. Our per-column PCMs
are directly inspired by this — each CC predicts neighboring activity, and the prediction
error ("surprise") serves as an auxiliary signal. We apply PC horizontally across columns
in a language model rather than vertically in a visual hierarchy.

**PC as Learning Algorithm** (Whittington & Bogacz, 2017; Millidge et al., 2021)
demonstrated that predictive coding can approximate backpropagation using local updates.
We do NOT use PC as the training algorithm — we use standard backprop for the LM and
REINFORCE for the neuromodulator. Our PCMs are an architectural component computing
surprise signals, closer to Rao & Ballard's neuroscience interpretation.

**What's novel**: Using per-column predictive coding modules that compute inter-column
prediction errors as auxiliary signals in a scan-based language model appears to lack
direct precedent. Most PC work in ML focuses on replacing backprop; we use PC as a
feature computation mechanism.

---

## 8. Structural Plasticity / Network Rewiring

**Sparse Evolutionary Training / SET** (Mocanu et al., 2018) is the closest prior work
to our structural plasticity. Both: start with random sparse connectivity, prune
low-magnitude connections, regrow randomly. Key differences: SET operates on model weights
during training; our plasticity operates on memory graph connections during inference. SET
prunes by magnitude alone; our pruning is driven by the neuromodulator. We track flow EMA
and co-activation correlation as utility metrics that SET lacks.

**RigL** (Evci et al., 2020) improves SET with gradient-guided regrowth (activate
connections with largest gradient on absent weights). More informed than random regrowth
but requires gradient computation — impossible in our non-differentiable memory graph,
making random regrowth a necessary choice.

**Lottery Ticket Hypothesis** (Frankle & Carlin, 2018) finds sparse subnetworks post-hoc
from dense networks. Our model starts sparse and evolves topology continuously, driven by
the neuromodulator. Shared insight: sparse networks can be as expressive as dense ones.

---

## 9. Biologically-Inspired Sequence Models

**SpikeGPT** (Zhu et al., 2023) replaces continuous activations with binary spikes in an
RWKV-inspired architecture for energy efficiency. Our model focuses on memory capabilities
rather than energy efficiency, using continuous activations in the neuron graph.
Complementary — spiking activations could work within our graph.

**Thousand Brains Theory / Numenta** (Hawkins et al., 2017-2024) proposes cortical columns
as semi-independent learning modules using reference frames. Our "cortical columns" share
the name and the principle of multiple semi-independent units, but our CCs are scan-based
LM slices trained by backprop while Numenta's use sensorimotor reference frames.

**Graph Neural ODEs** (Poli et al., 2019) model continuous-time dynamics on graphs. Our
per-token neuron dynamics are a discrete-time version of a graph ODE — exponential decay
ODE with message passing, discretized per token. Graph Neural ODEs use learned dynamics
within autograd; ours are fixed dynamics modulated by RL.

---

## Summary: What Is Genuinely Novel

| Aspect | Closest Prior Work | What We Do Differently |
|--------|-------------------|----------------------|
| Non-differentiable neuron graph as LM memory | RL-NTM (discrete addressing only) | Full per-neuron RL control: primitives, weights, decay, topology |
| Per-neuron RL neuromodulation (225 dims × 1024 neurons) | Backpropamine (scalar signal, differentiable) | Per-neuron actions, non-differentiable, controls structure not just plasticity rate |
| RL-driven structural plasticity | SET/RigL (magnitude/gradient pruning) | Neuromod drives weights → prune threshold, informed by flow/correlation metrics |
| Memory as RL environment (not differentiable module) | All of Category 2 is differentiable | Explicit design choice: memory outside autograd |
| Per-column predictive coding in scan-based LM | Rao & Ballard (visual cortex theory) | Applied as architectural component in a language model |
| Scan backbone + persistent neuron graph | Mamba/HGRN (scan only, no graph memory) | Complementary memory system with different dynamics |
| Per-token graph message passing for LM memory | Titans (differentiable meta-learning) | Activation-based (neuron state), not parameter-based (weight updates) |

The combination of these elements — a scan-based LM backbone, a persistent non-differentiable
neuron graph with per-token message passing, RL-trained neuromodulation controlling all
aspects of neuron behavior, structural plasticity driven by utility metrics, and per-column
predictive coding — does not appear in any single prior system.
