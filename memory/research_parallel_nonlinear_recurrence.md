# Parallelizing Nonlinear Recurrent Dynamics: Research Survey

**Date**: 2026-03-21
**Context**: Methods for parallelizing h[t] = f(W*h[t-1] + x[t]) on GPUs, where f is nonlinear.

## The Fundamental Problem

Linear/affine recurrences `h[t] = a*h[t-1] + b[t]` can be parallelized via associative scan
(prefix sum) in O(log T) parallel time. Nonlinear recurrences break the semigroup property
required for associative scan. This document surveys ALL known approaches to work around this.

---

## 1. Newton's Method / Fixed-Point Formulation (DEER, ELK, ParaRNN)

### The Big Idea
Reformulate the sequential recurrence as a system of simultaneous nonlinear equations,
then solve them all at once using Newton's method — whose inner linear solve IS parallelizable
via associative scan.

### DEER: Parallelizing Non-Linear Sequential Models (ICLR 2024)
- **Authors**: Yi Heng Lim, Qi Zhu, Joshua Selfridge, Muhammad Firmansyah Kasim
- **Paper**: arXiv:2309.12252
- **Key idea**: Define residual r(s_{1:T}) = [s_1 - f(s_0), s_2 - f(s_1), ..., s_T - f(s_{T-1})].
  The correct states satisfy r(s*) = 0. Apply Newton's method:
  ```
  Delta_s^{i+1} = -J(s^i)^{-1} r(s^i)
  ```
  The Jacobian has block-bidiagonal structure, so forward substitution reduces to a LINEAR
  recurrence: `Delta_s_t = [df/ds(s_{t-1})] * Delta_s_{t-1} - r_t(s)`.
  This linear recurrence CAN be solved via parallel associative scan in O(log T).
- **Result**: Up to 3 orders of magnitude speedup over sequential evaluation
- **Complexity**: O(TD^3) per Newton step (full Jacobian), O(TD^2) memory
- **Limitation**: Cubic in state size D; numerically unstable for chaotic systems

### ELK & Quasi-DEER: Towards Scalable and Stable Parallelization (NeurIPS 2024)
- **Authors**: Xavier Gonzalez, Andrew Warrington, Jimmy T.H. Smith, Scott W. Linderman
- **Paper**: arXiv:2407.19115 (code: github.com/lindermanlab/elk)
- **Improvements over DEER**:
  - **Quasi-DEER**: Replace full Jacobian with diagonal approximation -> O(TD) cost, O(TD) memory
  - **ELK**: Connection between Levenberg-Marquardt and Kalman smoothing provides trust-region
    damping for stability. Uses parallelized Kalman smoother.
  - **Quasi-ELK**: Both scalable (quasi-Newton) and stable (trust region). O(TD) cost.

| Method     | Work     | Memory   | Stability |
|------------|----------|----------|-----------|
| Sequential | O(TD^2)  | O(D)     | Very high |
| DEER       | O(TD^3)  | O(TD^2)  | Low       |
| Quasi-DEER | O(TD)    | O(TD)    | Low       |
| ELK        | O(TD^3)  | O(TD^2)  | High      |
| Quasi-ELK  | O(TD)    | O(TD)    | Moderate  |

### ParaRNN: Unlocking Parallel Training of Nonlinear RNNs for LLMs (2025)
- **Authors**: Federico Danieli, Pau Rodriguez, Miguel Sarabia, Xavier Suau, Luca Zappella (Apple)
- **Paper**: arXiv:2510.21450
- **Key result**: Newton iterations + custom parallel reductions -> 665x speedup over sequential
- **Scale**: Successfully trained 7B parameter models with adapted LSTM and GRU architectures
- **Performance**: Matches Transformer and Mamba2 perplexity at comparable scale
- **Open-source framework** for automatic parallelization of arbitrary nonlinear RNNs

### Predictability Determines Parallelizability (NeurIPS 2025)
- **Authors**: Xavier Gonzalez, Leo Kozachkov, David Zoltowski, Kenneth Clarkson, Scott Linderman
- **Paper**: arXiv:2508.16817
- **THE fundamental theoretical result**:
  - The **Largest Lyapunov Exponent (LLE)** of a dynamical system determines whether parallel
    Newton methods converge quickly
  - **Predictable systems** (LLE < 0): O((log T)^2) parallel time — dramatic speedup
  - **Chaotic systems** (LLE > 0): Conditioning degrades exponentially with T, parallelization fails
  - The Polyak-Lojasiewicz constant mu ~ sigma_min^2(J), which collapses for chaotic systems
  - **Design principle**: Make your recurrent model contractive/predictable to enable parallelization
  - Practical guideline: parameterize weights to guarantee contractivity (spectral norm < 1)

### Unified Theory: PhD Dissertation (Stanford, 2026)
- **Author**: Xavier Gonzalez
- **Paper**: arXiv:2603.16850
- Unifies DEER, ELK, Picard, Jacobi iterations into a single parallel Newton framework
- Establishes LLE as THE determining condition for all parallel Newton methods
- Covers RNNs, MCMC, and other sequential bottlenecks

**VERDICT**: This is the most promising family of methods for true nonlinear recurrences. Quasi-ELK
at O(TD) is practical. The key constraint is: YOUR SYSTEM MUST BE CONTRACTIVE (negative LLE).
Chaotic dynamics cannot be parallelized this way.

---

## 2. Deep Equilibrium Models (DEQ) / Implicit Layers

### Core Idea
Instead of running T layers of a recurrence to convergence, directly solve for the fixed point
z* = f(z*) using root-finding. Backprop through the fixed point using implicit differentiation
(no need to store intermediate states).

### DEQ: Deep Equilibrium Models (NeurIPS 2019)
- **Authors**: Shaojie Bai, J. Zico Kolter, Vladlen Koltun (CMU)
- **Paper**: arXiv:1909.01377
- **Forward pass**: Find z* such that z* = f(z*) using Broyden's method (quasi-Newton)
- **Backward pass**: Implicit function theorem: dL/dtheta = -(dL/dz*)(I - df/dz)^{-1} (df/dtheta)
  Solved via another fixed-point iteration, no need to store all intermediate states
- **Memory**: O(1) — constant regardless of effective "depth"
- **Training speed**: Slower per step than explicit models due to iterative solving
- **Key limitation**: Does NOT parallelize over sequence length T. It parallelizes over depth.
  For sequence modeling, you still process tokens sequentially.

### Multiscale DEQ (NeurIPS 2020)
- **Authors**: Shaojie Bai, Vladlen Koltun, J. Zico Kolter
- Extends DEQ to multiple resolution streams solved simultaneously
- Applied to vision tasks (ImageNet, Cityscapes)

### Jacobian-Free Backpropagation (JFB)
- **Authors**: Fung, Heaton, et al. (arXiv:2103.12803)
- Replaces implicit Jacobian solve with approximate "phantom gradients"
- Faster training, easier implementation, maintains accuracy
- Applied to image deblurring (2024 extension)

### Anderson Acceleration for DEQ (NeurIPS 2024 Workshop)
- **Authors**: Al Dajani, Keyes
- **Paper**: arXiv:2410.19460
- Anderson extrapolation on GPUs speeds up fixed-point convergence for DEQ
- Combines information from previous iterations to extrapolate

**VERDICT**: DEQ eliminates depth, not sequence length. Relevant for reducing memory of very deep
weight-tied recurrences, but does NOT solve the T-parallelism problem for sequence processing.
Could potentially be combined with Newton-based methods.

---

## 3. Predictor-Corrector / Parallel Cyclic Reduction

### DeepPCR: Parallelizing Sequential Operations in Neural Networks (2023)
- **Authors**: Federico Danieli, Xavier Suau, Pau Rodríguez, Miguel Sarabia, Luca Zappella (Apple)
- **Paper**: arXiv:2309.16318
- **Core idea**: Interpret L sequential steps as solution of a system of equations, recover using
  Parallel Cyclic Reduction (PCR) algorithm
- **PCR**: Classical algorithm for tridiagonal systems. Reduces O(L) to O(log_2 L) parallel steps.
  Works by eliminating half the variables simultaneously at each step.
- **Results**:
  - Forward pass (MLPs): 30x speedup
  - Backward pass (MLPs): 200x speedup
  - ResNet training (1024 layers): 7x faster
  - Diffusion model generation: 11x faster
- **Limitation**: Primarily for feedforward/layer-depth parallelism, not sequence-length RNN parallelism.
  The nonlinear extension requires iterative refinement.

### Accelerating Feedforward via Parallel Nonlinear Equation Solving (ICML 2021)
- **Authors**: Yang Song, Chenlin Meng, Renjie Liao, Stefano Ermon (Stanford)
- **Paper**: arXiv:2002.03629
- Reformulates feedforward as nonlinear equations, solves via Jacobi/Gauss-Seidel iterations
- Jacobi updates are fully parallel
- Speedups 2.1x-26x for RNN backprop, DenseNet eval, autoregressive sampling
- **Guaranteed identical** results to sequential computation

---

## 4. Architectural Linearization (Make the Recurrence Linear by Design)

### The Linear Scan Revolution
The dominant practical approach: design architectures where the recurrence IS linear, then stack
nonlinear layers between linear recurrence blocks.

### Parallelizing Linear Recurrent Neural Nets (ICLR 2018)
- **Authors**: Eric Martin, Chris Cundy
- **Paper**: arXiv:1709.04057
- First to apply parallel scan to RNNs
- Introduced GILR-LSTM: linear surrogate of LSTM that can use parallel scan
- Up to 40x throughput improvement, can handle 1M-length sequences

### Linear Recurrent Unit (LRU) — "Resurrecting RNNs" (ICML 2023)
- **Authors**: Antonio Orvieto, Samuel Smith, Albert Gu, et al. (DeepMind)
- **Paper**: arXiv:2303.06349
- Key insight: linearize the recurrence, diagonalize, use complex-valued diagonal matrices
- Linear recurrence + nonlinear MLP between layers = surprisingly expressive
- Matches S4/S5 performance on Long Range Arena
- Initialization and normalization are critical

### minGRU / minLSTM — "Were RNNs All We Needed?" (2024)
- **Authors**: Leo Feng, Frederick Tung, Mohamed Ahmed, Yoshua Bengio, Hossein Hajimirsadeghi
- **Paper**: arXiv:2410.01201
- **THE trick**: Remove hidden state h from gate computations. Gates depend ONLY on input x_t.
  ```
  Original GRU: z_t = sigma(W_z x_t + U_z h_{t-1})  <- depends on h
  minGRU:       z_t = sigma(W_z x_t)                  <- input-only
  ```
  This makes the recurrence LINEAR in h: h_t = (1-z_t)*h_{t-1} + z_t*tilde_h_t
  where both z_t and tilde_h_t are functions of x_t only -> precomputable in parallel
- **Speedup**: 175x (minGRU) to 1361x (minLSTM) for sequence length 4096 vs traditional
- **Parameters**: 33% fewer than standard GRU/LSTM
- **Trade-off**: Less expressive per-layer (no h-dependent gating), but competitive when stacked

### HGRN / HGRN2 (NeurIPS 2023, 2024)
- **Authors**: Zhen Qin et al.
- **Papers**: arXiv:2311.04823, arXiv:2404.07904
- Hierarchically gated linear recurrences with data-dependent decay
- Lower-bounded forget gates: lower layers = short-term, upper layers = long-term
- HGRN2 adds outer-product state expansion without extra parameters
- All parallelizable via scan

### Mamba (2024), Mamba-2 (2024), Mamba-3 (ICLR 2026)
- **Authors**: Albert Gu, Tri Dao
- Mamba: Selective state spaces with input-dependent parameters, hardware-aware scan
- Mamba-2: State Space Duality — SSM = masked self-attention with 1-semiseparable mask.
  2-8x faster core layer than Mamba-1. Uses matrix multiplication as primitive.
- Mamba-3: Trapezoidal discretization, complex-valued states (equivalent to data-dependent RoPE),
  MIMO formulation. 1.5B scale improvements.
- All use parallel scan / chunk-wise processing for training

### RWKV-7 "Goose" (2025-2026)
- Linear-time, constant-space, attention-free RNN with LLM-scale performance
- Parallel WKV CUDA kernel: O(T) training, O(1) per-token inference
- Scaled to 7B+ parameters, 206K tok/s training on 4x8xH100
- Custom CUDA kernels for optimized parallel processing

**VERDICT**: This is by far the most practically successful approach. The entire Mamba/RWKV/HGRN/GLA
ecosystem proves that linear recurrences + interleaved nonlinearities are competitive with
transformers. The question is whether this is expressive ENOUGH for bio-plausible dynamics.

---

## 5. Diagonal Jacobian Constraint (Nonlinear but Parallelizable)

### LrcSSM: Parallelization of Non-linear State-Space Models (NeurIPS 2025)
- **Authors**: Monica Farsang, Ramin Hasani, Daniela Rus, Radu Grosu
- **Paper**: arXiv:2505.21717
- **THE most relevant paper for our architecture**
- **Core trick**: Force the Jacobian of the nonlinear recurrence to be diagonal by design.
  ```
  dx_i/dt = -sigma(f_i(x_i, u)) * sigma(eps_i(x_i, u)) * x_i
            + tau(z_i(x_i, u)) * sigma(eps_i(x_i, u)) * e_i^leak
  ```
  Vectorized: dx/dt = A(x,u)*x + b(x,u), where A is DIAGONAL and state/input-dependent
- **Why it works**: Diagonal A(x,u) means each state dimension i evolves independently given
  the input. The recurrence for dimension i is: x_i[t] = a_i(x_i[t-1], u[t]) * x_i[t-1] + b_i(...)
  This IS a (nonlinear but per-dimension) recurrence that can be solved via parallel scan
  after Newton linearization, or directly via element-wise scan if you treat a_i, b_i as
  precomputed from a predictor.
- **Key insight**: Zeroing cross-state synaptic connections in the state transition. Only
  self-loops remain in A. Inter-neuron communication happens through b(x,u) (the input path).
- **Results**: O(TD) computation, O(log T) sequential depth
- **Outperforms**: Transformers, LRU, S5, Mamba on long-range forecasting
- **Gradient stability**: Formal proof that gradients decay geometrically, never explode
  (unlike Liquid-S4 and Mamba which lack such guarantees)

### Liquid-S4: Liquid Structural State-Space Models (2022)
- **Authors**: Ramin Hasani, Mathias Lechner, T.H. Wang, M. Chahine, A. Amini, D. Rus
- **Paper**: arXiv:2209.12951
- Input-dependent state transition (liquid time-constant networks + S4)
- Diagonal + low-rank decomposition of state matrix
- More expressive than S4 but harder to parallelize (dense Jacobian)
- LrcSSM is the "properly parallelizable" successor

**VERDICT**: The diagonal Jacobian constraint is an elegant middle ground — preserves nonlinear
dynamics (state-dependent gating, sigmoid saturation) while enabling parallel scan. This is
the closest to what we want for bio-plausible dynamics. Key trade-off: neurons can't directly
influence each other's state transitions, only through the input pathway.

---

## 6. Distillation from Nonlinear/Quadratic to Linear

### The Mamba in the Llama (NeurIPS 2024)
- **Authors**: Junxiong Wang, Daniele Paliotta, Avner May, Alexander Rush, Tri Dao
- **Paper**: arXiv:2408.15237
- Distills Transformer (quadratic attention) into hybrid linear RNN
- Reuses linear projection weights from attention layers
- Hybrid with 25% attention layers matches full Transformer on chat benchmarks
- Feasible with academic GPU resources

### LoLCATs: Low-Rank Linearizing of LLMs (ICLR 2025)
- **Authors**: Zhang, Arora, et al. (Stanford Hazy Research)
- **Paper**: arXiv:2410.10254
- Two-step: (1) train linear attention to match softmax via MSE, (2) LoRA fine-tune
- First linearized 70B LLM (18h on 8xH100), first linearized 405B LLM
- Closes 78% of MMLU gap between Transformer and linearized variant
- Only 0.2% of parameters, 40M training tokens

### Liger: Linearizing LLMs to Gated Recurrent Structures (ICML 2025)
- **Authors**: (OpenSparseLLMs)
- **Paper**: arXiv:2503.01496
- Repurposes pretrained key matrix weights for gating mechanisms
- No extra parameters needed
- LoRA fine-tuning recovers 93% of original Transformer performance
- Validated on 1B to 8B parameter models

### LAWCAT: Quadratic to Linear Attention with Convolution (EMNLP 2025)
- **Authors**: (arXiv:2509.18467)
- Integrates causal Conv1D for local dependency modeling
- Distills using <0.1% of pre-training tokens, <1/3 sequence length
- 90% passkey retrieval at 22K tokens (from 8K model)

### Effective Distillation to Hybrid xLSTM (2026)
- **Paper**: arXiv:2603.15590
- Pipeline for distilling Transformers into xLSTM-based students
- Replaces self-attention with hybrid SWA + mLSTM via data-dependent gating
- Applied to Llama, Qwen, Olmo families

### HALO: Hybrid Attention via Layer Optimization (2025)
- Pipeline for distilling Transformer -> RNN-attention hybrid
- Only 2.3B tokens needed (<0.01% of pre-training data)

**VERDICT**: Distillation from quadratic/nonlinear to linear is a mature, practical approach.
For our use case (bio-plausible nonlinear memory -> fast deployment), this is very viable:
train the nonlinear memory graph slowly, then distill its behavior into a fast linear scan model.
The entire ecosystem (LoLCATs, Liger, HALO) proves this works at scale.

---

## 7. Parareal / Parallel-in-Time Methods

### Layer-Parallel Training of Deep Residual Neural Networks (2018)
- **Authors**: S. Gunther, L. Ruthotto, J.B. Schroder, E.C. Cyr, N.R. Gauger
- **Paper**: arXiv:1812.04352
- Interprets ResNets as forward Euler discretizations of nonlinear IVP
- Replaces sequential forward/backward with parallel nonlinear multigrid
- Achieves layer-parallelism for training

### Multiple Shooting for Neural ODEs (2021)
- **Authors**: Evren Mert Turan, Johannes Jaschke
- **Paper**: arXiv:2109.06786
- Divides time interval into segments, integrates independently in parallel
- Continuity constraints via penalty/augmented Lagrangian
- Handles oscillatory data where single-shooting fails

### RandNet-Parareal (NeurIPS 2024)
- Uses random neural networks to learn coarse-fine solution discrepancy
- 125x speedup vs fine solver, 22x vs standard parareal
- Applicable to PDEs with 10^5 spatial mesh points

### Fully Discretized Simultaneous Optimization (2025)
- **Authors**: Mariia Shapovalova, Calvin Tsay
- **Paper**: arXiv:2502.15642
- Collocation-based formulation: optimize trajectory + network params simultaneously
- ADMM for distributed computation across data batches

**VERDICT**: Parareal methods are powerful for ODE/PDE integration but have limited applicability
to standard neural network training on GPUs. The multiple shooting idea is promising for
Neural ODE training specifically. Not directly applicable to discrete-time RNN training
(where Newton-based methods like DEER are more natural).

---

## 8. Spiking / Event-Driven Parallelization

### Parallel Spiking Unit (PSU) (IJCNN 2024)
- **Authors**: Yang Li, Yinqian Sun, Xiang He, Yiting Dong, Dongcheng Zhao, Yi Zeng
- **Paper**: arXiv:2402.00449
- **Key trick**: Decouple LIF sequential dependency via probabilistic reset
  - V = AI - BS where A = leak-integration matrix, B = reset matrix
  - Estimate spike state S' = sigma(AI) instead of actual spikes
  - Enables concurrent membrane potential computation at all timesteps
- **Speedup**: 3-4x on SHD dataset vs vanilla LIF
- Sparser spiking = better energy efficiency

### Fixed-Point Parallel Training (FPT) (2025)
- **Authors**: Wanjin Feng, Xingyu Gao, et al.
- **Paper**: arXiv:2506.12087
- Reformulates LIF neuron dynamics as fixed-point iteration
- Processes ALL T timesteps simultaneously
- Reduces O(T) to O(K) where K~3 (constant!)
- Unifying framework: existing parallel spiking neurons are special cases

### GPU Limitations for SNNs
- Sparsity of spikes means large tensor sub-blocks are inactive
- GPUs are NOT optimized for sparsity — cannot exploit event-driven nature
- Neuromorphic hardware (100-1000x energy efficiency) but not programmable like GPUs
- Current best: batch spikes into dense operations, lose some efficiency advantage

**VERDICT**: For spiking neurons specifically, the probabilistic reset / fixed-point tricks
are quite relevant to our architecture. The FPT method (O(K) with K~3) is particularly
interesting. However, GPU sparsity support remains the bottleneck — you can't exploit
biological sparsity well on current GPU hardware.

---

## 9. Global Linearization: Koopman Operators & Carleman Linearization

### Koopman Operator Theory
- Provides a LINEAR but infinite-dimensional operator to globally linearize nonlinear dynamics
- Learn Koopman eigenfunctions via deep learning (Deep-EDMD, etc.)
- In the Koopman space, dynamics ARE linear -> can use parallel scan!
- **Trade-off**: Finite-dimensional truncation introduces approximation error
- Active research on Invertible Koopman Networks (IKN) for lossless reconstruction (2025)
- **Practical issue**: Learning good Koopman embeddings is itself expensive

### Carleman Linearization
- Lifts finite-dimensional nonlinear system to infinite-dimensional linear system
- Truncation at order N gives approximate linear dynamics in N*d dimensions
- Exponential convergence for stable systems
- Primarily used for quantum computing applications currently
- **Not yet practical** for neural network training at scale

**VERDICT**: Theoretically elegant but practically challenging. The Koopman approach is the
most promising — if you can learn a good embedding, you get exact linear dynamics that
can be scanned. But learning the embedding is a chicken-and-egg problem. Worth monitoring
but not ready for production use.

---

## 10. Continuous-Time Formulations (Neural ODEs/CDEs)

### Neural ODEs
- **No inherent parallelization advantage** over discrete recurrences
- Training via adjoint method: constant memory but often SLOWER than ResNets
- Gauß-Legendre quadrature can speed up adjoint integration
- Spectral/collocation methods offer better accuracy per evaluation point

### Log Neural CDEs (ICML 2024)
- **Authors**: Walker et al.
- Log-signature features for efficient CDE solving
- Outperforms Mamba on some multivariate time series tasks

### Structured Linear CDEs (SLiCEs) (2025)
- Block-diagonal vector fields: maximally expressive yet efficient
- 20x faster training per step while matching accuracy
- Bridges continuous-time and linear recurrence approaches

**VERDICT**: Continuous-time formulations do NOT inherently help with parallelization.
They face the same sequential bottleneck. However, structured linear CDEs show that
the "linearize and diagonalize" strategy works in continuous time too.

---

## Summary: Practical Approaches Ranked by Relevance to Our Architecture

### Tier 1: Directly Applicable NOW
1. **Diagonal Jacobian constraint (LrcSSM-style)**: Force memory graph Jacobian diagonal,
   preserve nonlinear gating per-neuron, get O(log T) parallel scan. Most promising for
   bio-plausible nonlinear dynamics.
2. **Linear recurrence + stacked nonlinearity (Mamba/HGRN/minGRU style)**: Already what
   we do with HGRN scan blocks. Proven at scale.
3. **Distillation pipeline**: Train nonlinear bio-plausible model -> distill to linear scan
   for deployment.

### Tier 2: Promising, Needs Investigation
4. **Newton/quasi-ELK**: For truly nonlinear recurrences, quasi-ELK at O(TD) with ~3 Newton
   iterations is practical. Requires contractive dynamics (negative LLE). ParaRNN proves
   this works at 7B scale.
5. **Probabilistic reset / FPT for spiking**: If we add spike-like dynamics, the PSU/FPT
   tricks could parallelize LIF-style neurons with O(K) iterations.

### Tier 3: Interesting Theory, Not Yet Practical
6. **Koopman operator embedding**: Learn linear Koopman space -> scan. Chicken-and-egg problem.
7. **DEQ / implicit layers**: Eliminates depth, not sequence length. Could combine with Newton.
8. **Parareal / multiple shooting**: More relevant for Neural ODE training than discrete RNNs.
9. **Carleman linearization**: Beautiful theory, not GPU-practical yet.
