# Related Works

> **NOTE (2026-03-29):** Comprehensive literature survey for v9-backprop (branch
> `v9-backprop`). The memory graph is now differentiable and trained end-to-end by
> backprop. N=512 neurons, D_neuron=256, K=32, 4 scan layers split at 2, ~113M
> params, ~24.8K tok/s. No RL, no ES. Per-neuron MLPs (state, message, modulator),
> dendritic tree, Hebbian traces, structural plasticity, dynamic predictive coding.

A survey of prior work related to the Neuromorphic Language Model architecture,
organized by topic area. For each area, we review key papers, note their
relationship to our model, and highlight what distinguishes our approach.

---

## 1. Neuromorphic Computing and Language Models

Neuromorphic computing applies brain-inspired principles to hardware and
algorithms. While most neuromorphic work targets vision and robotics, recent
efforts bridge spiking neural networks (SNNs) with language models.

**Key Papers:**

- **BrainTransformers: SNN-LLM** (2024, 1 citation). Introduces a 3B-parameter
  LLM using spiking neural networks with SNN-compatible Transformer components
  (SNNMatmul, SNNSoftmax) and a Synapsis module simulating synaptic plasticity.
  Demonstrates competitive performance on MMLU (63.2) and GSM8K (76.3).
  [Paper](https://consensus.app/papers/details/8b088c31d02755929d0edbb5b0f9da5a/?utm_source=claude_code)

- **Spiking STDP Transformer (S2TDPT)** (2025). Implements self-attention through
  spike-timing-dependent plasticity (STDP), embedding query-key correlations in
  synaptic weights. Achieves 88.47% energy reduction vs. ANN Transformer on CIFAR-100.
  [Paper](https://consensus.app/papers/details/b94c76a7885d5c498296cee2bf75938b/?utm_source=claude_code)

- **Sorbet** (2024, 3 citations). A neuromorphic hardware-compatible spiking
  language model using PTsoftmax and Bit Shifting PowerNorm. Achieves 27x energy
  savings vs. BERT via knowledge distillation and binary weight quantization.
  [Paper](https://consensus.app/papers/details/f82c6136aaca5efd8c529539e0302fc9/?utm_source=claude_code)

- **Neuromorphic Computing at Scale** (2025, 119 citations). Nature review
  charting the future of large-scale neuromorphic systems, identifying key features
  for scalable architectures and potential applications.
  [Paper](https://consensus.app/papers/details/f452a8c5f47a54628cd3d2a5384b6b36/?utm_source=claude_code)

- **Biologically Grounded Neocortex Primitives on Neuromorphic Hardware** (2025,
  PNAS). Maps soft winner-take-all (sWTA) circuits from mouse V1 onto IBM's
  TrueNorth chip and uses them as a preprocessing filter in Vision Transformers,
  boosting out-of-distribution generalization by ~20%.
  [Paper](https://consensus.app/papers/details/664eca4f1771532b9fad8428028269b3/?utm_source=claude_code)

- **Advancing Neuromorphic Computing with Loihi** (2021, 502 citations). Survey
  of Intel's Loihi results showing that brain-inspired networks using recurrence,
  spike-timing, synaptic plasticity, and sparsity achieve orders of magnitude
  lower latency and energy vs. conventional approaches.
  [Paper](https://consensus.app/papers/details/ade3d30f38ee58ed842985b7d9ab6da8/?utm_source=claude_code)

**Relation to our work:** Existing neuromorphic LMs focus on energy-efficient
spiking implementations of existing architectures (transformers, attention). Our
approach is fundamentally different: we use continuous-valued neurons with
biologically-inspired dynamics (dendritic computation, structural plasticity,
Hebbian traces) as a memory system augmenting a scan-based LM, trained end-to-end
by backprop rather than requiring neuromorphic hardware. We share the goal of
bringing biological principles to language modeling but through architectural
innovation rather than hardware-level spiking.

---

## 2. Differentiable Memory Graphs and Graph Neural Networks as Memory

Graph neural networks (GNNs) use message passing to propagate information across
graph-structured data. Our memory graph can be viewed as a GNN operating as
persistent memory for a language model.

**Key Papers:**

- **Neural Turing Machine** (Graves et al., 2014, 2422 citations). External memory
  accessed via differentiable attention-based read/write heads. Pioneered the
  concept of differentiable external memory for neural networks.
  [Paper](https://consensus.app/papers/details/3bba79223e6d5b6b83813e9698098cf6/?utm_source=claude_code)

- **Differentiable Neural Computer (DNC)** (Graves et al., 2016, 1651 citations,
  Nature). Extends NTM with temporal link matrices and dynamic memory allocation.
  Learns graph traversal and reasoning tasks.
  [Paper](https://consensus.app/papers/details/8771254ffeaa5e6298bbec9c2609ad5f/?utm_source=claude_code)

- **Relational Memory Core (RMC)** (Santoro et al., 2018, 216 citations).
  Multi-head dot-product attention between memory slots enables relational
  reasoning across sequential information. State-of-the-art on WikiText-103.
  [Paper](https://consensus.app/papers/details/9a32b0d5b94d587b8265e391047dc92e/?utm_source=claude_code)

- **Sparse Access Memory (SAM)** (Rae et al., 2016, 161 citations). End-to-end
  differentiable memory with sparse reads/writes, 1000x faster and 3000x less
  memory than dense models. Shows that sparse memory access can retain
  representational power.
  [Paper](https://consensus.app/papers/details/5b39c5b1e2a05030b9a3819efe447c59/?utm_source=claude_code)

- **Meta-Learning with MANNs** (Santoro et al., 2016, 1922 citations). Demonstrates
  rapid one-shot learning using memory-augmented networks with content-based
  addressing.
  [Paper](https://consensus.app/papers/details/e509e987bb165caf8f9facbe4f5f2d76/?utm_source=claude_code)

- **NodeFormer** (Wu et al., 2023, 313 citations). All-pair message passing with
  kernelized Gumbel-Softmax for learning latent graph structures in a
  differentiable manner with linear complexity.
  [Paper](https://consensus.app/papers/details/9f2b4f953b8a59e4a0eaeac253e8e4e1/?utm_source=claude_code)

- **Hierarchical Message-Passing GNNs** (Zhong & Gao, 2020, 61 citations).
  Re-organizes flat graphs into multi-level super graphs with intra- and
  inter-level propagation for capturing long-range information efficiently.
  [Paper](https://consensus.app/papers/details/f44570a68dc75854b37f2609cd9c810c/?utm_source=claude_code)

- **Cooperative Graph Neural Networks** (Finkelshtein et al., 2023, 45 citations).
  Nodes choose to listen, broadcast, both, or isolate, providing flexible dynamic
  message passing.
  [Paper](https://consensus.app/papers/details/19896081ea015150a92b144e1fa19ab7/?utm_source=claude_code)

- **Graph Memory Neural Network** (2024). Adaptive message-passing with memory
  units capturing global patterns for both homophilic and heterophilic graphs.
  [Paper](https://consensus.app/papers/details/d6f960931ab15fe68c80fce67e455c61/?utm_source=claude_code)

- **Large Memory Models (LM2)** (2025, 6 citations). Decoder-only Transformer
  with auxiliary memory module using cross-attention and gating. Outperforms
  RMT by 37.1% and Llama-3.2 by 86.3% on BABILong multi-hop reasoning.
  [Paper](https://consensus.app/papers/details/b0ff1012b3a757b9998933e75cee9c6d/?utm_source=claude_code)

- **Relational Memory-Augmented Language Models** (2022, 34 citations). Conditions
  autoregressive LMs on knowledge graphs via relation triple retrieval, improving
  perplexity and coherence on WikiText-103.
  [Paper](https://consensus.app/papers/details/eb50b8ffea8355899a5e9f59e3fdfae6/?utm_source=claude_code)

**Relation to our work:** NTM/DNC use a passive memory matrix with controller-driven
read/write; our memory is an active network of neurons that autonomously propagate
signals. RMC uses full attention (quadratic in slots); we use sparse K=32
message passing (linear in connections). SAM demonstrates that sparse access
preserves representational power, supporting our sparse connectivity design. Unlike
all prior MANNs where memory is a flat bank of vectors, our memory has internal
graph structure with per-neuron MLPs that process, transform, and selectively
propagate information through dendritic trees.

---

## 3. Dendritic Computation in Deep Learning

Biological neurons perform sophisticated nonlinear computation in their dendritic
trees before integration at the soma. Several recent works incorporate dendritic
properties into artificial neural networks.

**Key Papers:**

- **Towards Deep Learning with Segregated Dendrites** (Guerguiev et al., 2016,
  389 citations, eLife). Shows that multi-compartment neurons receiving sensory
  input and feedback in segregated dendritic compartments can achieve deep
  learning. Demonstrates that dendritic segregation enables coordinated weight
  updates across layers.
  [Paper](https://consensus.app/papers/details/0656c26d6e9a5129a277309bdc5f5653/?utm_source=claude_code)

- **Dendritic Neuron Model** (Todo et al., 2019, 550 citations). Artificial neuron
  model incorporating nonlinear synapses and dendritic branches with pruning.
  Six evolutionary learning algorithms demonstrate effectiveness on classification,
  approximation, and prediction.
  [Paper](https://consensus.app/papers/details/2a77d9a22e2851dba288792dbe0a1582/?utm_source=claude_code)

- **Dendrites Endow ANNs with Accurate, Robust, and Parameter-Efficient Learning**
  (2025, 9 citations, Nature Communications). Structured dendritic connectivity
  and restricted sampling counteract overfitting, matching or outperforming
  traditional ANNs with significantly fewer parameters.
  [Paper](https://consensus.app/papers/details/b4aabad8e0685414b4503cec914cfa0f/?utm_source=claude_code)

- **Temporal Dendritic Heterogeneity in SNNs** (2024, 84 citations, Nature
  Communications). Multi-compartment spiking model with heterogeneous timing
  factors on dendritic branches. Achieves best-reported accuracy on speech
  recognition, visual recognition, and EEG benchmarks.
  [Paper](https://consensus.app/papers/details/7b6aaeee53425bde8c867c85714707b8/?utm_source=claude_code)

- **Dendrify Framework** (2022, 48 citations, Nature Communications). Open-source
  framework for incorporating dendrites into spiking neural networks, balancing
  biological accuracy with computational efficiency.
  [Paper](https://consensus.app/papers/details/9e6c71de8e5c55bd9cdd8e217b867b2f/?utm_source=claude_code)

- **Scalable Dendritic Modeling for Deep SNNs** (2024). Proposes the dendritic
  spiking neuron (DendSN) with efficient Triton kernels. Introduces dendritic
  branch gating for continual learning with reduced inter-task interference.
  [Paper](https://consensus.app/papers/details/ec5b3aa3726c5068aefe786ede1b5ced/?utm_source=claude_code)

- **Drawing Inspiration from Biological Dendrites** (Pagkalos et al., 2021, 57
  citations, Current Opinion in Neurobiology). Review highlighting dendritic
  anatomy, nonlinearities, and compartmentalized plasticity rules as features
  that can advance artificial neural networks.
  [Paper](https://consensus.app/papers/details/45e178841d8758f6acff2d51c9eb42ac/?utm_source=claude_code)

- **DeepDendrite** (Gao et al., 2022, 20 citations, Nature Communications).
  GPU-accelerated framework for biophysically detailed multi-compartment neuron
  simulations, achieving 2-3 orders of magnitude speedup. Demonstrates training
  of detailed dendritic models on MNIST.
  [Paper](https://consensus.app/papers/details/a02ed83adabe5a9a9496cbf7dc129014/?utm_source=claude_code)

- **Where is the Error? Hierarchical Predictive Coding through Dendritic Error
  Computation** (Sacramento et al., 2022, 86 citations, Trends in Neurosciences).
  Proposes that prediction errors are computed locally in dendritic compartments
  rather than separate error units, connecting predictive coding to efficient
  balanced networks.
  [Paper](https://consensus.app/papers/details/2642cdceabc75c918567677d13d476ac/?utm_source=claude_code)

**Relation to our work:** Our dendritic tree module directly implements hierarchical
signal integration inspired by this literature. Each neuron in our memory graph
gathers K=32 incoming messages, applies learned dendritic weights in a
hierarchical tree structure, and integrates the result before passing to the state
MLP. This is analogous to how biological dendrites nonlinearly combine synaptic
inputs before somatic integration. Unlike most dendritic ANNs that focus on
classification tasks, we use dendritic computation within a memory graph for
sequence modeling. Our Triton-fused dendritic gather kernel (forward + backward)
is related to the efficiency concerns of Scalable Dendritic Modeling and
DeepDendrite.

---

## 4. Structural Plasticity in Neural Networks

Structural plasticity--the creation and removal of synaptic connections--is
fundamental to brain development and learning. In our model, phi-correlation-based
rewiring reshapes the memory graph topology at chunk boundaries.

**Key Papers:**

- **NEAT: NeuroEvolution of Augmenting Topologies** (Stanley & Miikkulainen, 2002,
  3584 citations). Evolves both topology and weights of neural networks using
  speciation and incremental complexification.
  [Paper](https://consensus.app/papers/details/44a6b844f56753fd876576ae2251410e/?utm_source=claude_code)

- **Sparse Evolutionary Training (SET)** (Mocanu et al., 2018, 688 citations,
  Nature Communications). Evolves Erdos-Renyi random graphs into scale-free
  topologies during training, reducing parameters quadratically without accuracy
  loss. Demonstrated on RBMs, MLPs, and CNNs across 15 datasets.
  [Paper](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code)

- **ESL-SNNs** (Chen et al., 2023, 50 citations). Evolutionary structure learning
  for SNNs: pruning and regeneration of synaptic connections during learning
  while maintaining sparsity. Achieves 0.28% accuracy loss at 10% connection
  density on DVS-CIFAR10.
  [Paper](https://consensus.app/papers/details/113cc6ccf7d95d40842a08bf27713c67/?utm_source=claude_code)

- **GPU-Accelerated Structural Plasticity** (2025, 1 citation). Flexible
  framework for GPU-accelerated structural plasticity rules in spiking neural
  networks. Demonstrates DEEP R rewiring enabling sparse classifiers that match
  dense model performance at 10x faster training.
  [Paper](https://consensus.app/papers/details/56d1e3aea4d15323a42dbda803f5de88/?utm_source=claude_code)

- **Structural Plasticity for Neuromorphic Networks with PEDOT Dendrites** (2023,
  12 citations, Nature Communications). Hardware structural plasticity via
  dendritic growth of conducting polymer fibers, following Hebbian principles.
  Achieves 61% better sparsity on classification tasks.
  [Paper](https://consensus.app/papers/details/4b67417331e251b5b3b5fbdc43364900/?utm_source=claude_code)

- **Structural Plasticity on BrainScaleS-2** (2019, 26 citations). Implements
  structural plasticity on analog neuromorphic hardware with constant fan-in and
  sparse connectome, showing ability to optimize topology for training data.
  [Paper](https://consensus.app/papers/details/f79dc91d13a055cf81b38a41b9cd96ec/?utm_source=claude_code)

- **Adaptive Rewiring Evolves Brain-like Structure** (Jarman et al., 2020, 12
  citations, Scientific Reports). Diffusion-based adaptive rewiring steers
  random networks to modular or centralized small-world topologies, reflecting
  brain anatomy principles.
  [Paper](https://consensus.app/papers/details/d9c2f7393d01553fb46b76bd8d83e14e/?utm_source=claude_code)

- **Theoretical Framework for Learning Through Structural Plasticity** (2024,
  Physical Review E). Mean-field framework describing stabilization, pruning,
  and reorganization of synaptic connections and their effects on memory capacity.
  [Paper](https://consensus.app/papers/details/2241ab8eea72565fb019424f1ec62212/?utm_source=claude_code)

**Relation to our work:** Like SET, we start with sparse connectivity and
dynamically rewire. However, our rewiring criterion is fundamentally different:
we use phi-correlation (binary Pearson correlation) of neuron co-activation
patterns, directly inspired by Hebbian/anti-Hebbian principles, rather than
weight magnitude (SET) or gradient information (RigL). Our structural plasticity
operates on the memory graph topology at segment boundaries, not on model weights
during training. The adaptive rewiring literature (Jarman et al.) provides
theoretical support for our approach: activity-dependent rewiring naturally
produces brain-like small-world topologies.

---

## 5. Predictive Coding in Language and Sequence Models

Predictive coding posits that the brain continuously generates and updates
predictions, propagating only prediction errors. Our model uses dynamic predictive
coding--predicting state transitions rather than raw states.

**Key Papers:**

- **Rao & Ballard (1999)** -- *Seminal*. Proposed hierarchical predictive coding
  in visual cortex: top-down feedback carries predictions, feedforward pathways
  carry prediction errors. Foundation for all subsequent PC work.

- **Dynamic Predictive Coding** (Jiang & Rao, 2023, 25 citations, PLOS
  Computational Biology). Higher cortical levels modulate temporal dynamics of
  lower levels, correcting predictions of dynamics using prediction errors. Lower
  levels encode shorter timescales, higher levels encode longer timescales. When
  coupled with associative memory (hippocampus), supports episodic memory storage
  and retrieval.
  [Paper](https://consensus.app/papers/details/dc8725f0b2fa5adaa4f2fc2fcc27b8d8/?utm_source=claude_code)

- **Evidence of Predictive Coding Hierarchy in Human Brain Listening to Speech**
  (Caucheteux et al., 2023, 219 citations, Nature Human Behaviour). Shows that
  enhancing language models with predictions spanning multiple timescales improves
  brain-activity mapping. Frontoparietal cortices predict higher-level, longer-range
  representations than temporal cortices.
  [Paper](https://consensus.app/papers/details/b112f5479a7a54efae831f9f874785e8/?utm_source=claude_code)

- **Predictive Coding: A Theoretical and Experimental Review** (Millidge et al.,
  2021, 157 citations). Comprehensive review of PC theory covering microcircuit
  implementations, relationship to backpropagation, and connections to modern
  machine learning techniques.
  [Paper](https://consensus.app/papers/details/8da507b27003523ca44f67ef90328bab/?utm_source=claude_code)

- **Active Predictive Coding** (Rao, 2023, 18 citations, Neural Computation).
  Unified framework for perception, action, and cognition using hierarchical
  world models that combine task-invariant state transitions with task-dependent
  policy networks via hypernetworks.
  [Paper](https://consensus.app/papers/details/37858ba5711d5f8786e52d4b5b75ad47/?utm_source=claude_code)

- **A Review of Predictive Coding Algorithms** (Spratling, 2017, 391 citations,
  Brain and Cognition). Distinguishes five different algorithms called "predictive
  coding" from signal processing to cortical models, clarifying their similarities
  and differences.
  [Paper](https://consensus.app/papers/details/abe15fa19f7354ee8917affce1e2819e/?utm_source=claude_code)

- **Predictive Coding Across the Left Fronto-Temporal Hierarchy** (Kuperberg
  et al., 2022, 21 citations). MEG/EEG evidence for dynamic hierarchical
  predictive coding during language comprehension, with prediction errors and
  feedback sharpening at different cortical levels.
  [Paper](https://consensus.app/papers/details/16a05a144e905acfbb637351828b8f17/?utm_source=claude_code)

- **Predictive Coding Theories of Cortical Function** (Rao, 2021, 22 citations).
  Review framing predictive coding as Bayesian inference in hierarchical
  generative models, with top-down predictions and bottom-up errors.
  [Paper](https://consensus.app/papers/details/671e3d620ec157d0974c4ec1307f6ad9/?utm_source=claude_code)

**Relation to our work:** Our PCM (Predictive Coding Module) is directly inspired
by Jiang & Rao's dynamic predictive coding--we predict state *transitions*
(H_{t+1}) rather than raw states, matching their insight that higher levels should
modulate temporal dynamics. The PCM predicts the next hidden state directly (no
separate encoder), and the prediction error ("surprise") is fed to the split-point
MLP that combines H_mid with memory. This is architectural predictive coding (PC
as a computational feature) rather than algorithmic predictive coding (PC as a
replacement for backprop). The speech/language PC evidence (Caucheteux et al.)
supports the relevance of hierarchical prediction for language processing.

---

## 6. Hebbian Learning and Backpropagation Hybrids

Hebbian learning (local, activity-dependent synaptic modification) is the
brain's primary learning rule. Our model uses Hebbian traces as input to the
segment-boundary modulator, combining local correlation statistics with
backprop-trained processing.

**Key Papers:**

- **Predictive Coding Approximates Backprop Along Arbitrary Computation Graphs**
  (Millidge et al., 2020, 136 citations, Neural Computation). Demonstrates that
  predictive coding converges to exact backprop gradients on arbitrary computation
  graphs using only local, mostly Hebbian plasticity rules. Constructs PC
  equivalents of CNNs, RNNs, and LSTMs.
  [Paper](https://consensus.app/papers/details/133876e19f095a13bb9077df400bb14c/?utm_source=claude_code)

- **Approximation of Backprop in a Predictive Coding Network with Local Hebbian
  Plasticity** (Whittington & Bogacz, 2017, 311 citations, Neural Computation).
  Foundational result showing that predictive coding networks with simple local
  Hebbian plasticity can approximate the backpropagation algorithm.
  [Paper](https://consensus.app/papers/details/debe8458e42056219bd90392a9199982/?utm_source=claude_code)

- **Hebbian Deep Learning Without Feedback** (Journer et al., 2022, 65 citations).
  SoftHebb: trains deep networks without any feedback, target, or error signals
  using soft winner-take-all Hebbian learning. Achieves 80.3% on CIFAR-10 with
  up to five hidden layers.
  [Paper](https://consensus.app/papers/details/057d76bfe1dc52c6b4d1fc5306c31da3/?utm_source=claude_code)

- **Hebbian Semi-Supervised Learning** (Lagani et al., 2021, 23 citations).
  Hebbian pre-training of internal layers + SGD for classification layer.
  Outperforms end-to-end backprop in low-label regimes, demonstrating Hebbian
  learning's value for data-efficient feature extraction.
  [Paper](https://consensus.app/papers/details/6674a2a9304353b080d717c7d7d75df0/?utm_source=claude_code)

- **Local Plasticity Rules Can Learn Deep Representations** (Lowe et al., 2020,
  80 citations). Self-supervised contrastive predictive learning with local
  Hebbian rules and widely-broadcast modulation factors. Builds deep hierarchical
  representations of images, speech, and video.
  [Paper](https://consensus.app/papers/details/faae23937a89586b95a86c7c21c71868/?utm_source=claude_code)

- **A Theory of Local Learning** (Baldi, 2015, 72 citations). Systematic
  framework for studying local learning rules, proving that backprop maximizes
  the information rate of the "learning channel." Shows Hebbian learning is
  effective for feature extraction but insufficient alone for complex functions.
  [Paper](https://consensus.app/papers/details/055bb6cd2e9c526487ac6b9520c5059d/?utm_source=claude_code)

- **Hebbian Learning with Gradients** (Tallec & Ollivier, 2021, 18 citations).
  Shows that Hebbian rules can be implemented via specific loss functions whose
  gradients produce exactly the desired Hebbian updates, enabling use of modern
  deep learning frameworks.
  [Paper](https://consensus.app/papers/details/4c3b23ca4dc25e49b45c992629171e94/?utm_source=claude_code)

**Relation to our work:** Our model takes a pragmatic hybrid approach. The overall
system is trained by backpropagation, but Hebbian traces (running correlation of
pre/post-synaptic activity) are computed locally and fed as input features to the
segment-boundary modulator MLP. This modulator--itself trained by backprop--uses
Hebbian information alongside hidden state and connection weights to decide how to
update connections. This is neither pure Hebbian learning nor pure backprop but a
hybrid: local correlation signals inform a globally-trained controller. The Lowe
et al. result (local Hebbian + broadcast modulation = deep representations) is
particularly relevant, as our modulator similarly broadcasts decisions informed
by local traces.

---

## 7. Memory-Augmented Neural Networks

External memory systems for neural networks have a rich history. Our memory graph
is a fundamentally different architecture from prior flat memory banks.

**Key Papers:**

- **Neural Turing Machines** (Graves et al., 2014, 2422 citations). External
  memory with differentiable attention-based addressing. Learns copying, sorting,
  and associative recall.
  [Paper](https://consensus.app/papers/details/3bba79223e6d5b6b83813e9698098cf6/?utm_source=claude_code)

- **Differentiable Neural Computer** (Graves et al., 2016, 1651 citations,
  Nature). Adds temporal link matrices and dynamic allocation to NTM. Learns
  graph traversal and link inference.
  [Paper](https://consensus.app/papers/details/8771254ffeaa5e6298bbec9c2609ad5f/?utm_source=claude_code)

- **Survey on Memory-Augmented Neural Networks** (2023, 9 citations). Comprehensive
  survey covering Hopfield Networks, NTMs, Correlation Matrix Memories,
  Memformer, and Neural Attention Memory across NLP, CV, and multimodal tasks.
  [Paper](https://consensus.app/papers/details/b85fb0d541e851f5a82880577ad12a3c/?utm_source=claude_code)

- **Graph-Based Neural Memory** (2022, 4 citations). Proposes graph-based memory
  where data storage and access use graph structure rather than matrix model.
  Differentiable mechanisms for training via backprop. Outperforms NTM and
  LSTM on long-term dependency tasks.
  [Paper](https://consensus.app/papers/details/ba3184e973b95cf2aa151bf350488a13/?utm_source=claude_code)

- **Neural Stored-program Memory** (Le et al., 2019, 37 citations). Stores
  controller weights in memory (like stored-program computers), enabling program
  switching through time. Supports compositional and continual learning.
  [Paper](https://consensus.app/papers/details/b3f0264272eb57ea983900fc7e2ad2cb/?utm_source=claude_code)

- **Memory-Augmented Transformers: Systematic Review** (2025, 1 citation). Unified
  framework bridging neuroscience memory principles with engineering advances.
  Identifies shift from static caches to adaptive, test-time learning systems.
  [Paper](https://consensus.app/papers/details/906fb4a00f1d526bb61de7eb655474dc/?utm_source=claude_code)

**Relation to our work:** Unlike NTM/DNC where memory is a passive matrix
read/written by a controller, our memory is an active network of neurons that
autonomously propagate and transform information through per-neuron MLPs. The
graph-based memory (2022) is the closest conceptually, but uses a flat graph
without neuron-level computation. Our neurons have internal state MLPs, message
MLPs, dendritic trees, and modulators--each neuron is a small computational unit
rather than a simple storage slot.

---

## 8. Neuromodulation in Artificial Neural Networks

Neuromodulators (dopamine, acetylcholine, serotonin, noradrenaline) globally
modulate neural circuit dynamics in the brain. Our segment-boundary modulator
is inspired by this principle.

**Key Papers:**

- **Prefrontal Cortex as Meta-Reinforcement Learning System** (Wang et al., 2018,
  570 citations, Nature Neuroscience). Proposes that dopamine trains the PFC
  recurrent network to operate as its own free-standing learning system. This
  "meta-RL" framework accommodates a wide range of observations about
  reward-based learning.
  [Paper](https://consensus.app/papers/details/6338c79f9dbc5556bf28aeb8cbe15de9/?utm_source=claude_code)

- **Metalearning and Neuromodulation** (Doya, 2002, 687 citations). Proposes that
  dopamine signals reward prediction error, serotonin controls time scale of
  reward prediction, noradrenaline controls exploration randomness, and
  acetylcholine controls memory update speed.
  [Paper](https://consensus.app/papers/details/5d8268de7eb658648b7b96a255efb260/?utm_source=claude_code)

- **Informing DNNs by Multiscale Principles of Neuromodulatory Systems** (Avery
  et al., 2022, 38 citations, Trends in Neurosciences). Review of how diffuse
  neuromodulatory release fine-tunes neuronal and synaptic dynamics and
  plasticity. Outlines opportunities for integrating these principles into DNNs.
  [Paper](https://consensus.app/papers/details/bce1598240115a5ea47c626e22d9ea3c/?utm_source=claude_code)

- **ANML: A Neuromodulated Meta-Learning Algorithm** (Beaulieu et al., 2020, 155
  citations). A neuromodulatory network gates the forward pass of a prediction
  network, enabling context-dependent selective activation for continual learning
  of up to 600 classes.
  [Paper](https://consensus.app/papers/details/9723fc58bfbe55b2a1ae0c2645c92c1f/?utm_source=claude_code)

- **Neuromodulation-Assisted Credit Assignment (NACA)** (2023, 33 citations,
  Science Advances). Expectation signals induce neuromodulators to selectively
  modify long-term potentiation/depression. Achieves high accuracy with
  substantially reduced computational cost and mitigated catastrophic forgetting.
  [Paper](https://consensus.app/papers/details/2abeaba857725ca89358d2293fb7271f/?utm_source=claude_code)

- **Cell-Type-Specific Neuromodulation Guides Credit Assignment** (Bhatt et al.,
  2021, 34 citations, PNAS). Proposes that neurons communicate their contribution
  to learning via cell-type-specific local neuromodulation, suggesting neuron-type
  diversity may be critical for biological credit assignment.
  [Paper](https://consensus.app/papers/details/c3a1fadd5c8559228092d283821c7d85/?utm_source=claude_code)

- **Neuromodulators Generate Multiple Behaviors via Hyperchannels** (Bhowmik et al.,
  2021, 13 citations). Shows that diffuse synaptic weight modulation enables
  storage of multiple memories using common synapses. Neuromodulators "unlock"
  specific behaviors by creating task-specific hyperchannels.
  [Paper](https://consensus.app/papers/details/d006c5ad48f3567cae4abe6bb338507a/?utm_source=claude_code)

- **Distributional Reinforcement Learning in Dopamine** (Dabney et al., 2020, 375
  citations, Nature). Shows the brain represents possible future rewards as a
  probability distribution, not a single scalar, with different dopamine neurons
  encoding different quantiles.
  [Paper](https://consensus.app/papers/details/d153b5b0c3c75247bd3d9b9b957edf20/?utm_source=claude_code)

**Relation to our work:** Our segment-boundary modulator takes Hebbian traces,
hidden state, decay rate, and primitives as input and outputs new connection
weights, decay, and primitives--analogous to how neuromodulators in the brain
reshape synaptic dynamics based on context. Unlike ANML (which gates the forward
pass) or NACA (which modulates LTP/LTD), our modulator operates at segment
boundaries to restructure the memory graph's operational parameters. The Wang et al.
meta-RL insight is foundational: our LM backbone (trained by backprop) effectively
trains the memory system (with its modulator) to operate as its own memory
management system.

---

## 9. Scan/Recurrence Architectures (SSMs, Mamba, RWKV)

Our LM backbone uses a split-scan architecture (4 layers, split at 2). This
connects to the rapidly evolving landscape of efficient recurrent models.

**Key Papers:**

- **Mamba** (Gu & Dao, 2023, 4579 citations). Selective state spaces with
  input-dependent parameters. Achieves Transformer quality with linear complexity
  and 5x higher throughput. Mamba-3B outperforms Transformers of the same size.
  [Paper](https://consensus.app/papers/details/05dd434bcc4b5cbf9b326abcc58d7aec/?utm_source=claude_code)

- **Mamba-2 / State Space Duality** (Dao & Gu, 2024, 902 citations). Shows SSMs
  and attention variants are connected through structured semiseparable matrices.
  The core Mamba-2 layer is 2-8x faster while maintaining language modeling
  performance.
  [Paper](https://consensus.app/papers/details/646533b5bfc751cb9dffbd4e71137e5f/?utm_source=claude_code)

- **Samba** (2024, 104 citations). Hybrid architecture combining Mamba with
  Sliding Window Attention. Extrapolates from 4K to 256K context with perfect
  memory recall on Passkey Retrieval. 3.73x higher throughput than Transformers
  at 128K length.
  [Paper](https://consensus.app/papers/details/8537cc6b215d5c85aa69d6d9b30370a2/?utm_source=claude_code)

- **Linear State-Space Layer (LSSL)** (Gu et al., 2021, 860 citations). Unifies
  RNNs, temporal convolutions, and neural ODEs via continuous-time state-space
  representation. Introduces structured matrices A with long-range memory.
  Outperforms prior approaches by 24 accuracy points on length-16000 speech.
  [Paper](https://consensus.app/papers/details/7de08f3b91c3561b81ca1b6ee6db9873/?utm_source=claude_code)

- **Theoretical Foundations of Deep Selective SSMs** (2024, 51 citations). Uses
  Rough Path Theory to show selective SSMs compute projections of the input
  "signature," capturing non-linear interactions between tokens at distinct
  timescales.
  [Paper](https://consensus.app/papers/details/7eedd35c7e3e52c3849a3507d3205fa2/?utm_source=claude_code)

- **Demystify Mamba: A Linear Attention Perspective** (2024, 131 citations).
  Reformulates Mamba as a variant of linear attention with six key distinctions.
  Identifies forget gate and block design as core contributors to Mamba's success.
  Proposes MILA (Mamba-Inspired Linear Attention) that outperforms vision Mamba.
  [Paper](https://consensus.app/papers/details/fe9d03421502512394bec7b9605b3903/?utm_source=claude_code)

- **Graph-Mamba** (2024, 131 citations). Adapts Mamba to graph data with
  input-dependent node selection for long-range context modeling on graphs.
  [Paper](https://consensus.app/papers/details/fc2da6ca27d35528b3762ce20d22210f/?utm_source=claude_code)

- **MemMamba** (2025). Analyzes Mamba's exponential memory decay and proposes
  state summarization with cross-layer attention to alleviate long-range
  forgetting while preserving linear complexity.
  [Paper](https://consensus.app/papers/details/78cae4a4d2c2536999f214cfba72825f/?utm_source=claude_code)

**Relation to our work:** Our scan backbone is in the same family as Mamba/S4/HGRN.
The fundamental insight motivating our memory graph is MemMamba's finding: SSMs
suffer from exponential memory decay. Rather than patching this with attention
mechanisms (Samba) or state summarization (MemMamba), we add a qualitatively
different memory system--a persistent graph of neurons with their own dynamics.
The Mamba-2 duality result (SSM = restricted attention) shows our memory graph
provides a complementary, non-attention mechanism. Graph-Mamba validates that
Mamba-style computation can operate on graph structures, supporting our memory
graph architecture.

---

## 10. Tolman-Eichenbaum Machine and Hippocampal Memory Models

The Tolman-Eichenbaum Machine (TEM) provides a computational model of how the
hippocampal-entorhinal system supports spatial and relational memory. Its
key insight--that memory should be trained by prediction error--directly
influences our design.

**Key Papers:**

- **The Tolman-Eichenbaum Machine** (Whittington et al., 2019, 518 citations,
  Cell). Proposes medial entorhinal cells form a basis for structural knowledge,
  hippocampal cells link this basis with sensory representations. After learning,
  entorhinal cells display grid, band, border, and object-vector cell properties.
  Structural knowledge transfers across environments via remapping.
  [Paper](https://consensus.app/papers/details/12de2e1b677d533796b8d7cfcc3f03dc/?utm_source=claude_code)

- **Spiking TEM** (2025, bioRxiv). Extends TEM with spike-based computation and
  anatomically-inspired hippocampal-entorhinal architecture. Generates phase
  precession and predictive grid cells through biologically plausible learning.
  [Paper](https://consensus.app/papers/details/c7095e94ac295cff8446de5ad50678cf/?utm_source=claude_code)

- **Generative Model of Memory Construction and Consolidation** (Barron et al.,
  2023, 64 citations, Nature Human Behaviour). Hippocampal replay trains
  generative models (VAEs) to recreate sensory experiences from latent variables.
  Explains semantic memory, imagination, episodic future thinking, and
  schema-based distortions.
  [Paper](https://consensus.app/papers/details/0f8c7aa167aa54da88190dff4ec3f157/?utm_source=claude_code)

- **REMERGE: Generalization Through Recurrent Interaction of Episodic Memories**
  (Kumaran & McClelland, 2012, 332 citations, Psychological Review). Shows that
  recurrence within the hippocampal circuit enables generalization from
  pattern-separated episodic memories, resolving the tension between separation
  and generalization.
  [Paper](https://consensus.app/papers/details/1873ceb18c935d7d9b04012621749abf/?utm_source=claude_code)

- **Recurrent Predictive Coding for Associative Memory** (Salvatori et al., 2022,
  32 citations, PLOS Computational Biology). Combines predictive coding with
  recurrent CA3-like connectivity for associative memory. Shows that implicit
  covariance learning through recurrence is stable and biologically plausible.
  [Paper](https://consensus.app/papers/details/a1806ae40906541ebc9332df49505994/?utm_source=claude_code)

**Relation to our work:** TEM's core insight that memory systems should be trained
by next-observation prediction strongly influenced our design. Our PCM predicts
the next hidden state, and its prediction error drives memory integration--directly
analogous to TEM's training objective. The position module (path integration) in
TEM corresponds to our scan backbone, while the memory module (binding where+what)
corresponds to our memory graph. The REMERGE result--that recurrence enables
generalization from pattern-separated memories--provides theoretical justification
for our multi-pass message propagation within the memory graph. Like the hippocampal
system, our memory graph maintains sparse, pattern-separated representations that
interact recurrently.

---

## 11. Sparse Connectivity and Small-World Networks

Biological neural networks exhibit sparse, small-world connectivity with high
clustering and short path lengths. Our memory graph maintains K=32 sparse
connections per neuron.

**Key Papers:**

- **Collective Dynamics of Small-World Networks** (Watts & Strogatz, 1998, 41521
  citations, Nature). The foundational small-world paper. Shows networks with
  few long-range "rewired" connections achieve small characteristic path lengths
  while maintaining high clustering. Demonstrated on C. elegans neural network.
  [Paper](https://consensus.app/papers/details/ab0087e22a105f83a64a789052185be8/?utm_source=claude_code)

- **Small-World Brain Networks** (Bassett & Bullmore, 2006, 2539 citations).
  Shows the brain exhibits small-world topology supporting both specialized and
  integrated processing. Small-world networks are economical, minimizing wiring
  costs while supporting high dynamical complexity.
  [Paper](https://consensus.app/papers/details/2acdf23fa2f15521907c2566dca111c8/?utm_source=claude_code)

- **Sparse Connectivity Enables Efficient Processing in Cortex-like ANNs** (2025,
  1 citation, Frontiers in Neural Circuits). Shows that in large recurrent
  networks (matching cortical properties), sparse connectivity facilitates
  time- and data-efficient processing. Sparsity is most critical with fixed
  excitatory/inhibitory cell types.
  [Paper](https://consensus.app/papers/details/173d735111165b109bb1002150c63316/?utm_source=claude_code)

- **Scalable Training with Adaptive Sparse Connectivity** (Mocanu et al., 2018,
  688 citations, Nature Communications). Argues ANNs should not have
  fully-connected layers, inspired by sparsity and scale-freeness of biological
  networks. Evolves random sparse graphs into scale-free topologies.
  [Paper](https://consensus.app/papers/details/2c65cd5405d45036ae5824355e5dc489/?utm_source=claude_code)

- **Dendritic Normalization Improves Learning in Sparse Networks** (2021, 12
  citations, PLOS Computational Biology). Biophysics-based normalization (divide
  weight by number of afferents) significantly improves learning in sparse
  networks, including recurrent and self-organized architectures.
  [Paper](https://consensus.app/papers/details/30ea48bd3bdd565f9070fcc3a1fa9705/?utm_source=claude_code)

- **MOSAIC: In-Memory Computing for Small-World Spike-Based Networks** (2024, 45
  citations, Nature Communications). Neuromorphic architecture enforcing locality
  in connectivity for small-world SNN topologies. Achieves 10x higher routing
  efficiency than other SNN hardware.
  [Paper](https://consensus.app/papers/details/56ae5fd3b3a255318cb88c8bce25fd9c/?utm_source=claude_code)

- **Universal Structural Patterns in Sparse RNNs** (2023, 6 citations). Shows
  that optimized sparse RNNs share universal signed motif patterns and evolve
  toward structural balance, regardless of sparsification strategy.
  [Paper](https://consensus.app/papers/details/56ff8849c1f751b3b8760bffd767f94d/?utm_source=claude_code)

**Relation to our work:** Our K=32 sparse connectivity per neuron in a 512-neuron
graph reflects the biological principle that sparse, structured connectivity is
more efficient than full connectivity. Our structural plasticity (phi-correlation-
based rewiring) can be seen as an activity-dependent mechanism that evolves the
graph topology, analogous to how biological networks develop small-world
properties through activity-dependent plasticity (Bassett & Bullmore). The
dendritic normalization result is directly relevant to our readout mechanism:
we use 1/sqrt(N) normalization rather than 1/N (which kills gradients, as we
learned from v8).

---

## 12. Credit Assignment in Biological Networks

How the brain solves credit assignment (the "backprop problem") is one of
neuroscience's deepest questions. Our model uses backprop for efficiency but
incorporates biologically-motivated components.

**Key Papers:**

- **Brain-Inspired Machine Intelligence: Survey of Neurobiologically-Plausible
  Credit Assignment** (Ororbia & Mali, 2023, 17 citations). Organizes
  brain-inspired learning schemes into six families: Hebbian, perturbation-based,
  feedback alignment, target propagation, predictive coding, and energy-based.
  [Paper](https://consensus.app/papers/details/b2f66519f9b3594e9b32e7ac10e1071d/?utm_source=claude_code)

- **Can the Brain Do Backpropagation? Exact Implementation via Predictive Coding**
  (Song et al., 2020, 97 citations, NeurIPS). First framework showing exact
  equivalence between backprop and a biologically plausible model using local
  plasticity and simultaneous computation.
  [Paper](https://consensus.app/papers/details/daa27fe6c34b569abd50fecab5386f76/?utm_source=claude_code)

- **Inferring Neural Activity Before Plasticity: Prospective Configuration**
  (Song et al., 2024, 25 citations, Nature Neuroscience). Proposes that networks
  first infer the pattern of activity that should result from learning, then
  modify weights to consolidate it. More efficient and effective than backprop
  in biological contexts.
  [Paper](https://consensus.app/papers/details/c6b16d0a27015c219db4e27e7151a5a7/?utm_source=claude_code)

- **Predictive Forward-Forward Algorithm** (Ororbia & Mali, 2023, 42 citations).
  Combines predictive coding with the forward-forward algorithm. Updates synapses
  with forward passes only, eliminating backprop's structural constraints.
  [Paper](https://consensus.app/papers/details/43b490348f7453348c227490f11adf70/?utm_source=claude_code)

- **Generalized Latent Equilibrium (GLE)** (2024, 10 citations). Fully local
  spatio-temporal credit assignment in cortical microcircuits with continuously
  active plasticity. Exploits dendritic morphology for complex information
  storage and "prospective coding."
  [Paper](https://consensus.app/papers/details/f2575f32be9358d987b7ebe1aa8b83c3/?utm_source=claude_code)

- **Credit Assignment Through Deep Feedback Control** (Meulemans et al., 2021,
  42 citations). Uses feedback controller to drive networks to match targets.
  Relates to multi-compartment pyramidal neurons with voltage-dependent
  plasticity consistent with dendritic processing theories.
  [Paper](https://consensus.app/papers/details/c38b72cc0e655e47a4be16c2ec2929cc/?utm_source=claude_code)

- **Biologically-Plausible Backprop Through Arbitrary Timespans via Local
  Neuromodulators (ModProp)** (2022, 12 citations). Extra-synaptic diffusion of
  neuropeptides enables backprop-like credit assignment through time. Modulatory
  signals convolve eligibility traces via causal, synapse-type-specific filters.
  [Paper](https://consensus.app/papers/details/81274d6671a556819b1866897856d4be/?utm_source=claude_code)

- **Towards Biologically Plausible Computing: Comprehensive Comparison** (2024, 4
  citations). Evaluates Hebbian, STDP, feedback alignment, target propagation,
  predictive coding, forward-forward, perturbation learning, and energy-based
  methods. Compares learned representations to recorded brain activity.
  [Paper](https://consensus.app/papers/details/4f13c75587495deab17ce18e8135242e/?utm_source=claude_code)

**Relation to our work:** We pragmatically use backpropagation for training
efficiency but incorporate biologically-motivated architectural components that
address credit assignment concerns. Our TBPTT within segments (128 tokens) with
detach at boundaries limits temporal backprop depth, similar to truncated BPTT
used in biological models. The ModProp result (neuromodulators enable temporal
credit assignment) motivates our segment-boundary modulator, which operates at
a longer timescale than per-token updates. Our Hebbian traces provide locally
computed eligibility-like signals that the modulator uses for credit assignment
within the memory graph.

---

## 13. Graph Attention Networks and Message Passing

Our neuron message passing mechanism is related to, but distinct from, GNN
message passing and graph attention.

**Key Papers:**

- **Neuronal Message Passing Using Mean-field, Bethe, and Marginal Approximations**
  (Parr et al., 2019, 127 citations, Scientific Reports). Reviews Bayesian message
  passing in neuronal networks: variational message passing and belief propagation.
  Proposes marginal message passing as a compromise with simple architecture and
  good performance.
  [Paper](https://consensus.app/papers/details/d4e7cba33d88580aba50c661ee02a452/?utm_source=claude_code)

- **Building Powerful Equivariant GNNs with Structural Message-Passing** (Vignac
  et al., 2020, 134 citations). Propagates unique node identifiers to learn local
  context matrices, enabling learning of rich topological information while
  maintaining equivariance.
  [Paper](https://consensus.app/papers/details/1c5711b766aa579a9c293ad5b1967da8/?utm_source=claude_code)

- **Brain Network Communication: Concepts, Models and Applications** (2023, 173
  citations, Nature Reviews Neuroscience). Surveys network communication models
  beyond shortest paths, linking graph theory to biological neural signalling
  including transmission delays and metabolic cost.
  [Paper](https://consensus.app/papers/details/7d7005a8d4425fe1ab659386b61f9edf/?utm_source=claude_code)

- **Polarized Message-Passing** (2024, 53 citations). Captures both node similarity
  and dissimilarity for dual message sources, learning from sparse but strongly
  correlated neighbors.
  [Paper](https://consensus.app/papers/details/5fbb20b21adf5872904cef5e4cd75b2d/?utm_source=claude_code)

- **Dynamic Message Passing (N2)** (2024, 7 citations). Projects graph nodes and
  learnable pseudo nodes into a common space with evolving relations, enabling
  flexible pathway construction under linear complexity using a single recurrent
  layer.
  [Paper](https://consensus.app/papers/details/e226e74a66f85fe6a41f502689c8ff3a/?utm_source=claude_code)

**Relation to our work:** Our per-neuron message passing involves: (1) gathering
K=32 neighbor messages, (2) weighting via learned connection weights, (3)
hierarchical dendritic integration, (4) injection into state MLP, (5) state
update, (6) output via message MLP. This is substantially richer than standard
GNN message passing (which typically has a single aggregation + update step).
The 2-pass simulation (freeze inter-neuron messages per pass, run T MLP steps)
is related to the synchronous vs. asynchronous update debate in GNNs. Our
approach of per-neuron state MLPs producing messages that are then gathered and
dendritically integrated by neighbors creates a much more computationally
expressive per-step operation than standard GNN layers.

---

## 14. Leaky Integration, Continuous-Time RNNs, and Neural ODEs

Our neurons use leaky integration with structural decay for bounded states,
connecting to the rich literature on continuous-time neural dynamics.

**Key Papers:**

- **Neural Ordinary Differential Equations** (Chen et al., 2018, 5946 citations).
  Parameterizes the derivative of hidden state with a neural network, using ODE
  solvers for the forward pass. Enables continuous-depth models with constant
  memory cost.
  [Paper](https://consensus.app/papers/details/d56eaf837d575d4a88be03c74371e64e/?utm_source=claude_code)

- **Closed-form Continuous-Time Neural Networks** (Hasani et al., 2021, 121
  citations, Nature Machine Intelligence). Approximates liquid time-constant
  network dynamics in closed form, achieving 1-5 orders of magnitude faster
  training than ODE-based models. Time appears explicitly in closed form.
  [Paper](https://consensus.app/papers/details/82f0b686abb153f9aa5d925a2a07db5f/?utm_source=claude_code)

- **Linear State-Space Layer (LSSL)** (Gu et al., 2021, 860 citations). Unifies
  RNNs, convolutions, and neural ODEs via continuous-time state-space
  representation dx/dt = Ax + Bu, y = Cx + Du. Long-range memory through
  structured A matrices.
  [Paper](https://consensus.app/papers/details/7de08f3b91c3561b81ca1b6ee6db9873/?utm_source=claude_code)

- **LTC-SE: Liquid Time-Constant Networks** (2023, 15 citations). Unifies LIF
  spiking model, continuous-time RNNs, Neural ODEs, and gated RNNs. Time-varying
  time constants enable adaptive dynamics.
  [Paper](https://consensus.app/papers/details/a38940f2b09f5ff68b1ad8c8f90e59b8/?utm_source=claude_code)

- **Neural Differential Equations for Learning to Program Neural Nets** (Irie
  et al., 2022, 17 citations). Continuous-time counterparts of Fast Weight
  Programmers and linear Transformers. Shows ODEs can define learning rules for
  rapidly changing synaptic connections.
  [Paper](https://consensus.app/papers/details/69c26d6008745557b6453dd4f392fa13/?utm_source=claude_code)

- **Coincidence Detection and Integration in LIF Networks** (2023, 7 citations).
  Shows that membrane decay time determines whether LIF neurons operate as
  coincidence detectors (low decay) or integrators (high decay), with a power-law
  correlation factor between the two regimes.
  [Paper](https://consensus.app/papers/details/2ad044b2248950228ce3ed5e7d70417a/?utm_source=claude_code)

- **Advancing Spatio-Temporal Processing Through Adaptation in SNNs** (2024, 5
  citations, Nature Communications). Analyzes adaptive LIF neurons, showing the
  leakage plays a crucial role in balancing memory retention and robustness. The
  reset mechanism is essential for uninterrupted temporal processing.
  [Paper](https://consensus.app/papers/details/1aec02cda27f514689e27f75980be1d3/?utm_source=claude_code)

**Relation to our work:** Our neuron state dynamics are a discrete-time analogue of
a leaky integrator ODE: h_{t+1} = decay * h_t + (1 - decay) * f(input_t). The
decay parameter is per-neuron and modulated by the segment-boundary modulator,
making it analogous to the liquid time-constant in Hasani et al.'s work. The
coincidence detection vs. integration result (2023) is relevant: neurons with
fast decay (low memory) act as coincidence detectors, while slow-decay neurons
integrate over time. Our learned per-neuron decay rates allow the network to
develop heterogeneous temporal dynamics, with some neurons serving as short-term
coincidence detectors and others as long-term integrators.

---

## 15. Insect Brain Connectome Models

The fruit fly (Drosophila) brain demonstrates that simple neurons with the
right topology produce complex behavior, providing an existence proof for our
approach.

**Key Papers:**

- **FlyWire: Whole-Brain Annotation and Multi-Connectome Cell Typing** (2024, 149
  citations, Nature). Complete ~140,000-neuron connectome with 8,453 cell types.
  Finds broad stereotypy with occasional variability. Evidence for functional
  homeostasis through excitation/inhibition ratio maintenance.
  [Paper](https://consensus.app/papers/details/7bdd9216e5dc56a6931a722ed6b81188/?utm_source=claude_code)

- **Hemibrain Connectome** (Scheffer et al., 2020, 783 citations, eLife). Detailed
  circuits for most of the fruit fly central brain. Establishes cell types,
  computational compartments, and connection strength distributions.
  [Paper](https://consensus.app/papers/details/67d505a942bb586291e6684c15de3e81/?utm_source=claude_code)

- **Mushroom Body Connectome and Function** (Li et al., 2020, 253 citations, eLife).
  Dense connectome of the mushroom body revealing extensive visual input,
  unexpected structure in sensory modality transfer to MBONs, and modulation by
  dopaminergic neurons.
  [Paper](https://consensus.app/papers/details/987598ca27e65c429ea41b0186004121/?utm_source=claude_code)

- **Drosophila Computational Brain Model** (Shiu et al., 2024, 39 citations,
  Nature). Leaky integrate-and-fire model of entire Drosophila brain based on
  connectivity and neurotransmitter identity. Accurately predicts neurons
  responding to tastes, motor neuron firing, and sensorimotor transformations.
  [Paper](https://consensus.app/papers/details/91a3521fd6815a4b9f24273f6fa6a19e/?utm_source=claude_code)

- **Connectome-Constrained Networks Predict Neural Activity** (2024, 62 citations,
  Nature). Shows that connectivity alone (without detailed biophysical
  parameters) can predict neural activity when parameters are optimized via deep
  learning. Prediction accuracy improves with sparser connectivity.
  [Paper](https://consensus.app/papers/details/e5590ea6830753528b4eda6e1e89b8a5/?utm_source=claude_code)

- **Neuromorphic Simulation of Drosophila on Loihi 2** (2025, 1 citation).
  First biologically realistic connectome simulated on neuromorphic hardware.
  140K neurons, 50M synapses on 12 Loihi 2 chips. Orders of magnitude faster
  than conventional simulation with better performance on sparser activity.
  [Paper](https://consensus.app/papers/details/d968aeea2b4655619829d70bf7dc46a3/?utm_source=claude_code)

- **Can a Fruit Fly Learn Word Embeddings?** (Dasgupta et al., 2021, 23 citations).
  Shows the mushroom body network motif (sparse, high-dimensional representation
  + lateral inhibition) can learn semantic word embeddings as sparse binary hash
  codes, performing comparably to GloVe with a fraction of computational resources.
  [Paper](https://consensus.app/papers/details/80004b5b14615243a6ef338b67a0b62b/?utm_source=claude_code)

- **Compensatory Variability in Mushroom Body** (2021, 14 citations, PNAS).
  Shows memory performance is rescued when mushroom body parameters compensate
  each other to equalize average activity, with correlations predicted by the
  model appearing in the hemibrain connectome.
  [Paper](https://consensus.app/papers/details/e31c8ea54e3b5d4899a92ec44efcad23/?utm_source=claude_code)

- **Neuromorphic Model of Drosophila Larval Olfactory Processing** (2021, 12
  citations). Shows how feedback inhibition drives spatial sparseness and spike
  frequency adaptation with feedback inhibition drives temporal sparseness.
  Validated on neuromorphic hardware.
  [Paper](https://consensus.app/papers/details/5b7d68b909e8553dbc0a19ad65d51833/?utm_source=claude_code)

**Relation to our work:** The Drosophila connectome provides powerful evidence that
(a) simple neurons with the right connectivity produce complex behavior, (b) sparse
connectivity is biologically preferred and computationally sufficient, and (c)
optimization over connectivity structure yields accurate predictions even without
detailed biophysical modeling (2024 Nature paper). The mushroom body's architecture
(sparse coding + dopaminergic modulation for learning) is remarkably similar to our
design: sparse K=32 connections + segment-boundary modulator that adjusts connection
parameters. The "Can a Fruit Fly Learn Word Embeddings?" result is particularly
striking, showing that the biological network motif we draw inspiration from can
already perform NLP-relevant computation. Our 512-neuron memory graph is roughly 3x
the size of the mushroom body's Kenyon cell population in Drosophila larvae (~150
cells), placing us in a biologically relevant scale.

---

## Summary: What Is Genuinely Novel

| Aspect | Closest Prior Work | What We Do Differently |
|--------|-------------------|----------------------|
| Differentiable neuron graph as LM memory | DNC (passive matrix), Graph-based Memory (flat graph) | Active graph of neurons with per-neuron MLPs (state, message, modulator), dendritic trees, Hebbian traces |
| Dendritic tree in a memory graph | Dendritic Neuron Model (classification only) | Hierarchical dendritic integration of K=32 neighbor messages within a persistent memory graph for language modeling |
| Hebbian traces as modulator input | Backpropamine (scalar signal), NACA (expectation-based) | Hebbian traces computed locally, fed to a backprop-trained segment-boundary modulator that restructures connection parameters |
| Structural plasticity in a differentiable system | SET/RigL (magnitude/gradient on model weights) | Phi-correlation-based rewiring of memory graph connections at segment boundaries, trained within overall backprop framework |
| Dynamic predictive coding for LM | Jiang & Rao 2023 (visual cortex theory) | PCM predicts H_{t+1} directly; prediction error drives memory integration in a scan-based language model |
| Split-scan LM + persistent neuron graph | Mamba/HGRN (scan only), Titans (differentiable meta-learning) | Complementary memory system with biological neuron dynamics, separate from the LM's recurrent state |
| Per-neuron leaky integration with learned decay | Liquid Time-Constant Networks (single neuron model) | 512 neurons with heterogeneous decay rates modulated at segment boundaries, enabling multi-timescale memory |
| Biologically-motivated components trained by backprop | Bio-plausible methods (sacrifice performance), Standard DNNs (sacrifice biology) | Practical hybrid: backprop for training efficiency, bio-inspired architecture for memory capabilities |

The combination of these elements--a scan-based LM backbone, a differentiable
persistent neuron graph with per-neuron MLPs and dendritic trees, Hebbian traces
informing a learned modulator, phi-correlation structural plasticity, dynamic
predictive coding, and multi-timescale leaky integration--does not appear in any
single prior system. The closest individual precedents are the Tolman-Eichenbaum
Machine (prediction-driven memory with binding), the Drosophila mushroom body
(simple neurons + sparse connectivity + modulatory learning = complex behavior),
and Dynamic Predictive Coding (hierarchical prediction of state dynamics). Our
contribution is synthesizing these biological principles into a practical,
differentiable architecture that augments a modern language model.
