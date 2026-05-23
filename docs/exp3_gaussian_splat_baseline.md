# Gaussian Splat Baseline (Exp 3)

A new memory architecture for the v1h QA benchmark, complementary to the
existing A / B / MT / Mamba / plastic / vanilla variants. The substrate is a
fixed-K **signed Gaussian mixture** over a shared latent space, written by a
learned transformer, and read by **closed-form line integrals along
probe-emitted rays** (Gaussian-Splatting / NeRF-style volumetric reading
applied to memory).

---

## Why this exists

Across the v1h bench so far (flat, continuous, memorizing, recurrent, plastic)
nothing decisively beats a simple slot-attention memory. The shared design
flaw of the existing variants is that they all reduce to some variation of
"store K vector cells, do attention over them at read time." The blobs idea
breaks that mold:

1. The memory has explicit **geometric structure** (Gaussian blobs in R^d).
2. Memory can encode **negation** via signed components (positive blobs say
   "this region is supported," negative blobs carve out "this region is
   explicitly contradicted").
3. The read is **not attention**: a probe emits rays whose line integrals
   through the signed density field are the response. Two blobs along the
   same ray combine *spatially*, which attention cannot natively express.
4. The write transformer is trained to balance three explicit objectives
   (pin satisfaction, mass proportionality, proximal stickiness) — directly
   formalizing the "least controversial adjustment" intuition.

---

## Substrate

K fixed signed Gaussian blobs in a shared latent space `R^d`:

```
{ (μ_k, Σ_k, w_k, s_k) }_{k = 1..K}     fixed K  ⇒  bounded memory footprint
  μ_k ∈ R^d                              blob center
  Σ_k diagonal in R^{d×d}                blob covariance (diagonal in practice)
  w_k ≥ 0                                blob magnitude
  s_k ∈ {-1, +1}                         sign (soft via tanh of an unbounded scalar)
```

The **density field** the blobs define is:

```
ρ(x) = Σ_k  s_k · w_k · N(x; μ_k, Σ_k)
```

Positive regions = "what the world model supports." Negative regions = "what
the world model has carved out / contradicted." `ρ` can take any sign at any
point in R^d, including zero (boundary between support and contradiction).

The blobs are **forward-only state**: their numeric values evolve as the
write transformer outputs new parameters per chunk. No gradients flow into
`μ`, `Σ`, `w`, `s` directly. (Gradients flow into the *transformer that
produced them* via the read pipeline and auxiliary losses.)

---

## Pins (ephemeral)

For each incoming chunk of context tokens (4096 max), Llama produces token
embeddings. A learned encoder projects them into the shared latent space:

```
pins = f(token_embeds)     # [N, d]   N ≤ 4096
```

Pins are **ephemeral** — they exist only during the write step for this
chunk, then are discarded. Only the K blobs persist across chunks. This is
what keeps the memory bounded regardless of how much context has been seen.

---

## Write — transformer-updater predicting target blob parameters

```
blobs_new = TransformerUpdater(pins, blobs_old; θ_W)
```

`TransformerUpdater` is a small cross-attention stack (~2 layers):

```
1. Encode each old blob as a token by concatenating (μ_k, log_σ_k, w_k, s_k_raw).
   K learned positional embeddings give each blob a stable role across calls.
2. Blob tokens cross-attend to pins, then optionally self-attend among blobs.
3. Per-blob output head emits target parameters (μ_new, log_σ²_new, w_new, s_logit_new).
4. Initialize the output head so first call ≈ identity on old params
   (residual-style init); the model can move blobs arbitrarily once trained.
```

**Target prediction, not delta**. The transformer can completely re-place a
blob if it needs to. Proximal behavior is supplied by an explicit loss term,
not by a hard delta constraint. Reason: with `L_adjust` properly weighted,
the model learns *when* small changes are right and *when* large changes are
right — a hard delta would prevent the latter.

### Training objectives

Three auxiliary losses, applied at each chunk's write step:

#### L_pin — observations should be supported

Each pin wants positive density at its location:

```
L_pin  =  Σ_i  softplus( m − ρ(p_i) )
```

A hinge with margin `m`: pin density must exceed `m`. Pins in negative regions
(i.e., the model actively contradicts an observation) are penalized most
heavily.

#### L_proportional — don't waste blob mass on unobserved space

Total blob mass should track the pin cloud's mass. Simplest version is
mass conservation via `∫ ρ(x) dx = Σ_k s_k · w_k`:

```
L_mass  =  ( Σ_k s_k · w_k − N )²
```

Stricter version compares the blob field to a Parzen estimate of the pins on
sampled probe points:

```
L_proportional  =  (1/M) Σ_m  ( ρ_pins(q_m) − max(0, ρ_blobs(q_m)) )²
```

where `q_m` are M sampled probes (cheap subsample of pins + random points).

#### L_adjust — least controversial adjustment (proximal stickiness)

Sum of per-blob Wasserstein-2 distances between Gaussians (closed form):

```
L_adjust = Σ_k γ_k · W₂²( N(μ_k_old, Σ_k_old), N(μ_k_new, Σ_k_new) )
         + Σ_k γ_k · (w_k_old − w_k_new)²
         + Σ_k 1[sign flipped] · large_constant

γ_k = w_k_old   (weight by accumulated mass — established blobs stick more)

For diagonal Σ:
W₂²( N(μ₁, diag σ₁²), N(μ₂, diag σ₂²) )  =  ‖μ₁ − μ₂‖²  +  ‖σ₁ − σ₂‖²
```

This realizes the "established knowledge is sticky; new evidence can spawn
alongside or refine, but can't easily overwrite" principle.

### Total loss

```
L_total = L_QA  +  α · L_pin  +  β · L_proportional  +  λ · L_adjust
```

`L_QA` is the downstream QA cross-entropy on Llama. The transformer-updater
gets gradient from all four terms. Aux loss coefficients (α, β, λ) start
at non-trivial values (0.1 each) to prevent the QA loss from dominating
and collapsing the substrate structure.

---

## Read — rays through the signed density field

Llama's hidden state at every position becomes a probe. The probe emits **K
ray directions and one origin**, all in the shared latent space:

```
o   = OriginHead(h_llama)      # [d]   one origin per Llama position
D   = DirectionHead(h_llama)   # [K_rays, d]   K directions, init orthogonal
D_k = D_k / ‖D_k‖              # unit vectors
```

For each ray k, the response is the **line integral of the signed density**
along the ray:

```
I_k = Σ_j  s_j · w_j · ∫ N(o + t · D_k; μ_j, Σ_j) dt
```

This has a **closed-form** per (ray, blob) pair (verified — see Math
Appendix). The total cost per Llama position is `O(K_rays · K_blobs · d)` for
diagonal Σ — about 13 M ops at K_rays=8, K_blobs=64, d=256, T=100 positions.
Negligible.

The K scalar ray-integrals form a K-vector response. Augment with per-ray
auxiliary features (positive-mass-only integral, negative-mass-only integral,
peak-density location `t*`) for richer per-ray bandwidth:

```
ray_feats_k = [ I_k_total, I_k_pos_only, I_k_neg_only, t*_k ]
response    = concat over k                          # [4 · K_rays]
inject_vec  = W_out(response)                        # [d_llama]
```

This injects via a forward pre-hook at one Llama layer L (same plumbing as
`plastic_baseline`):

```
h_llama += scale · inject_vec
```

### Why this read is not attention

Standard attention is a softmax-weighted sum of stored values. The
ray-integral read is the **line integral of a continuous signed density
field**.

- Two blobs along the same ray combine *spatially* — their contributions
  add coherently if both positive, cancel if one is negative. Attention
  cannot natively express this.
- A negative blob along a ray subtracts from the integral — Llama gets a
  direct "this is contradicted" signal as a negative scalar in the response.
- The read cannot reduce to a flat KV bank because the line integral
  requires the blobs to have shape (covariance) — degeneracy to point
  masses makes all integrals identically zero almost surely.

---

## Bottleneck accounting

Per blob (diagonal Σ): `d + d + 1 + 1 = 2d + 2` floats = 514 floats at d=256.

K = 64 blobs ⇒ substrate state ≈ **33 K floats per batch element** —
in the same order of magnitude as the prepend variants' bottleneck
(`M × d = 36 × 725 = 26.1 K`). For verify_v1h's bottleneck-parity check
we list splat at `K · (2d + 2)` and the existing parity exemption for
plastic extends here.

Trainable params (target ~13 M to match the bench):

| Component | Estimate |
|---|---:|
| Pin encoder `f` (d_llama → d) | ~0.5 M |
| TransformerUpdater (2 layers, d=256, 4 heads) | ~3-4 M |
| OriginHead + DirectionHead (d_llama → d, d_llama → K·d) | ~3 M |
| W_out (4 K_rays → d_llama) | ~0.1 M |
| Llama-side scale parameter + biases | negligible |

Add a couple of MLP layers around the projections to land at ~13 M total,
matching the param band of B / MT / Mamba / plastic.

---

## Math Appendix — Line-Gaussian integral (verified)

For a single Gaussian `N(x; μ, Σ)` in `R^d` and a parametric line
`x = o + t · d`, the integral along the line is closed-form:

```
∫_{-∞}^{∞} N(o + t · d; μ, Σ) dt
   =  (2π)^{-(d-1)/2} · |Σ|^{-1/2} · (dᵀ Σ⁻¹ d)^{-1/2} · exp(-m²/2)
```

where `m²` is the perpendicular Mahalanobis distance from `μ` to the line:

```
m²  =  (o − μ)ᵀ Σ⁻¹ (o − μ)  −  [ (o − μ)ᵀ Σ⁻¹ d ]² / (dᵀ Σ⁻¹ d)
```

**Derivation sketch.** Substitute `δ = o − μ` and expand the quadratic in t:
`(δ + t·d)ᵀ Σ⁻¹ (δ + t·d) = t²·(dᵀΣ⁻¹d) + 2t·(δᵀΣ⁻¹d) + δᵀΣ⁻¹δ`.
Complete the square in t with `A = dᵀΣ⁻¹d`, `t* = −(δᵀΣ⁻¹d)/A`. The
residual is exactly `m²`. The integral over `t` of `exp(−½A(t − t*)²)` is
`√(2π/A)`. Combine with the Gaussian's normalization to get the formula.

This is the **same closed form** used in 3D Gaussian Splatting's
volumetrically-consistent rasterization derivations
([Talegaonkar et al. 2024](https://arxiv.org/abs/2412.03378),
[Huang et al. ICML 2024](https://arxiv.org/abs/2402.00752)) — applied here
to memory rather than rendering.

**Numerical implementation at d=256.**

- The `(2π)^{-(d−1)/2}` prefactor is ~10⁻¹⁰⁰ at d=256: **work in log space
  throughout**. Compute `log_I_k = -½·(d−1)·log(2π) − ½·log|Σ_k| − ½·log(dᵀΣ⁻¹d) − ½·m²`.
- `log|Σ|` for diagonal: `Σᵢ log σ²ᵢ`. For low-rank-plus-diagonal: matrix
  determinant lemma. **Never form the determinant directly**.
- The signed mixture sum `Σ_k s_k · w_k · I_k` cannot use vanilla
  logsumexp (positive and negative terms don't combine). Split into
  positive and negative log-sums then subtract:
  ```
  log_sum_pos = logsumexp over k with s_k > 0  of  (log w_k + log_I_k)
  log_sum_neg = logsumexp over k with s_k < 0  of  (log w_k + log_I_k)
  signed_sum  = exp(log_sum_pos) − exp(log_sum_neg)
  ```
  Catastrophic cancellation is possible when the two are nearly equal —
  acceptable for first-cut (cancellation IS the meaningful signal "this
  region is equally supported and contradicted").
- Diagonal Σ is strongly recommended (full Σ at d=256 is `O(d³)` per
  blob per inversion, and the Bures-Wasserstein metric on full Σ is also
  `O(d³)`).

---

## Related work and what we're betting on

A thorough literature survey informed this design. Key context:

### Direct precedents

- **∞-former** ([Martins et al. ACL 2022](https://arxiv.org/abs/2109.00301)).
  Long-term memory expressed as a linear combination of N RBFs over a 1-D
  parameter, attended to via continuous-space attention. The closest published
  ancestor — essentially the 1-D, positive-only, sparsemax-attended version of
  splat. **The strongest published baseline to beat.**
- **Continuous Attention**
  ([Martins et al. NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/file/f0b76267fbe12b936bd65e203dc675c1-Paper.pdf)).
  Foundational math for ∞-former. Our line-integral read is the multi-ray
  generalization of continuous attention's expectation-of-a-basis-function read.
- **Variational Bayes Gaussian Splatting (VBGS)**
  ([Catal et al. 2024](https://arxiv.org/abs/2410.03592)). Frames Gaussian
  Splatting as variational inference with closed-form conjugate updates,
  explicitly for continual learning. Our `L_adjust` Wasserstein-2 term is a
  proximal cousin of their KL-based update.
- **Gaussian Mixture Convolution Networks**
  ([Celarek et al. 2022](https://arxiv.org/abs/2202.09153)). Both data and
  kernels are signed Gaussian mixtures with closed-form convolution.
  Demonstrates signed mixtures are usable as neural primitives.

### Theoretical backing for signed components

- **Subtractive Mixture Models via Squaring**
  ([Loconte et al. ICLR 2024 spotlight](https://arxiv.org/abs/2310.00724)).
  Proves squared subtractive mixtures can be **exponentially more
  parameter-efficient** than additive ones for density modeling. Strongest
  theoretical argument for "why negative blobs."
- **Ratio of Signed Mixtures Models**
  ([Kannan, Cranmer et al. 2024](https://arxiv.org/pdf/2410.10216)).
  Negative weights are not a fundamental obstacle to optimization, but
  *variance* of weights matters more than the sign per se.

### Latent-set architectures (the macro-architecture)

- **Slot Attention** ([Locatello et al. NeurIPS 2020](https://arxiv.org/abs/2006.15055)),
  **Set Transformer** ([Lee et al. ICML 2019](https://arxiv.org/abs/1810.00825)),
  **Perceiver IO** ([Jaegle et al. 2021](https://arxiv.org/abs/2107.14795)).
  All use K-element learned latent sets with cross-attention writes. Splat's
  transformer-updater is structurally a slot-attention variant; the
  *distinguishing* bits are the signed-Gaussian-structured outputs, the
  three explicit auxiliary losses, and the line-integral read.

### Memory architectures with persistent latent state

- **Titans** ([Behrouz et al. NeurIPS 2025](https://arxiv.org/abs/2501.00663)).
  Persistent neural long-term memory with surprise-based momentum writes.
  Same persist-across-chunks topology; substrate is MLP, not density field.
- **TTT-Linear / TTT-MLP** ([Sun et al. 2024](https://arxiv.org/abs/2407.04620)).
  Hidden state IS a parametric model trained at test time. Shares "memory is
  parametric, not vectoric" philosophy.
- **Gated DeltaNet** ([Yang et al. ICLR 2025](https://arxiv.org/abs/2412.06464)).
  Gating + delta rule for clearing + targeted updates — same family as
  proximal + proportional losses.

### Energy / associative memory lineage

- **Modern Hopfield Networks**
  ([Ramsauer et al. ICLR 2021](https://arxiv.org/abs/2008.02217)).
  Softmax attention IS continuous-state Hopfield retrieval. The line-integral
  read is structurally a "soft retrieval over Gaussian-density stored
  patterns" — Hopfield-adjacent but with ray geometry that vanilla Hopfield
  attention lacks.
- **Bartunov EBMM**
  ([Bartunov et al. ICLR 2020](https://arxiv.org/abs/1910.02720)).
  Memory as a parameterized scalar field; reads via gradient descent on
  energy. We share the "memory is a field over R^d" framing; reads are
  closed-form rather than iterative.

### Unpublished but adjacent (non-peer-reviewed, mentioned for completeness)

- **Hierarchical Gaussian K-Splat Attention (HSA)**
  ([repo](https://github.com/bigattichouse/Guassian-Splat-Attention)) —
  splats with mitosis/death adaptation; no signed weights, no line-integral
  read, no benchmarks.
- **CogniRay / Holographic Projection Memory**
  ([repo](https://github.com/MnemoWare/CogniRay)) — directional projection
  rays through a voxelized latent field, soft kernels; no Gaussian-blob
  substrate, no math, no benchmarks.

These are gestural — neither has the **signed blobs + closed-form
Gaussian line-integral read + Wasserstein-2 proximal write loss**
combination that is the distinctive bet of this design.

---

## Known risks (what reviewers will push on)

1. **Information capacity vs. K.** K=64 blobs at d=256 with diagonal Σ
   stores roughly `K · (2d + 2) ≈ 33 K` scalars per batch element. Modern
   Hopfield stores ~exp(d) patterns. Reviewers will demand a capacity
   argument: how many independent facts can the substrate distinguish under
   the read mechanism, vs. a flat `K · d` memory bank at the same param
   count? Loconte et al. (ICLR 2024) provides ammunition for signed
   mixtures' density-modeling efficiency; we'd need to show it transfers to
   retrieval.

2. **High-D ray sparsity.** In R^256, a random direction is unlikely to be
   aligned with any specific structure. DirectionHead has to *learn* to
   emit directions that hit something. If it stays close to random, the
   response is mostly noise. Mitigation: orthogonal init for the K ray
   directions, ramp K up over training.

3. **No clear story for compositional / relational facts.** Density at a
   location is a unary predicate ("this region is supported"). Binding two
   entities through a relation requires either (i) blobs whose `μ` encodes
   a binding via TPR / HRR, or (ii) edge state between blobs — neither in
   this spec. This is the same limitation noted in the existing
   `project_capacity_concern.md` memory.

4. **What does "negative" actually mean during read?** A negative blob
   contributes a negative scalar to the K-vector response, and Llama
   projects it back via a learned linear map. There's no built-in inductive
   bias that negative output should function as "evidence against." It's
   just another signed dimension. Reviewers will ask whether the signed
   mixture is *functional* or just a wider memory with twice the parameters
   in disguise.

5. **L_proportional can fight L_pin.** Pin loss wants mass where
   observations are; proportional loss wants total mass to match observed
   density. With ephemeral pins per chunk, the proportional target is a
   moving window — total blob mass should grow as more chunks are seen, but
   the proportional loss resets per chunk. Need to either make it per-chunk
   or specify a global running target. Unbalanced OT formulations may be
   relevant ([Séjourné et al. 2019](https://arxiv.org/abs/1910.12958)).

6. **K-saturation behavior.** Fixed K with continuing writes means the
   updater must either overwrite (penalized by L_adjust) or merge / repurpose
   blobs. VBGS uses Bayesian model selection; HSA uses mitosis/death; this
   spec assumes the transformer learns whatever is right. May need an
   explicit "blob reset on saturation" heuristic for stability.

7. **Optimization landscape for GMMs is hostile.**
   [Jin et al. NeurIPS 2016](https://proceedings.neurips.cc/paper_files/paper/2016/file/3875115bacc48cca24ac51ee4b0e7975-Paper.pdf)
   showed randomly-initialized EM converges to spurious local maxima with
   high probability for K ≥ 3. We're using a transformer, not EM, so this
   risk transfers indirectly — but mode collapse via "mean alignment" and
   "vanishing weights" is generic to GMM-style training. `L_proportional`
   acts as an anti-collapse term; annealing the loss coefficients may help.

8. **Numerical care required.** Log-space arithmetic throughout (the
   `(2π)^{-(d-1)/2}` prefactor is ~10⁻¹⁰⁰ at d=256). Signed mixture sums
   can't use vanilla logsumexp — split into pos / neg log-sums. The
   gradients w.r.t. μ and Σ involve products of all precision-sensitive
   quantities and will need careful unit testing on synthetic blobs
   before scaling.

---

## What we'd be testing

**Hypothesis.** A signed Gaussian density field, written by a learned
transformer under explicit proportionality + proximal objectives, and read
by closed-form line integrals, is a viable memory substrate — competitive
with or better than slot-attention / Mamba / fast-weight memories at matched
params and compute, with qualitatively different per-family behavior driven
by:
- the ability to encode contradiction (signed blobs)
- the spatial coherence of reads (rays passing through multiple blobs
  produce correlated signal)
- the proximal stickiness (established knowledge resists overwrite)

**Win conditions** (in order of importance):

1. **Trains stably end-to-end** without aux-loss tuning hell.
2. **Matches B / Mamba** on val_recon at matched params and speed.
3. **Wins on at least one task family** in a way that's interpretable from
   the design (e.g., `revisions` where negation matters, or sequential
   tasks where ray-spatial coherence could help).
4. **Beats ∞-former** when implemented at matched scale (the strongest
   published direct precedent).

**Failure modes worth diagnosing:**

- Signs collapse to all-positive (negative blobs unused) — the inductive
  bias didn't matter
- Blobs collapse into a single mega-blob — L_proportional too weak
- Random ray directions throughout training — DirectionHead didn't learn
- Catastrophic cancellation in the signed-mixture log-sum — numerical fix
  needed, not a hypothesis problem

---

## Implementation plan

1. **`src/repr_learning/splat_substrate.py`** — module with:
   - `SplatState` dataclass: K signed Gaussian blobs as parallel tensors
   - `init_splat_state(B, K, d, device, dtype)` — zero / random init
   - `line_integral(o, D, blobs)` — closed-form line integral, batched over
     batches × Llama positions × K rays × K blobs, all in log space
   - `ray_features(o, D, blobs)` — augment with pos-only / neg-only / t*
   - `L_pin`, `L_proportional`, `L_adjust` as standalone functions

2. **`TransformerUpdater`** — small cross-attention stack, 2 layers.
   Input: encoded blob tokens + pin tokens. Output: per-blob target
   parameter vectors. Output head near-identity-init.

3. **`SplatBaselineEncoder` in encoder.py** — wraps the substrate with the
   standard `init_streaming_state` / `streaming_write` / `finalize_memory`
   interface used by other variants. `streaming_write` accumulates pins
   across windows then calls `TransformerUpdater` once per chunk.
   `finalize_memory` returns `([B, 0, d_llama], aux_with_blobs)`.

4. **`compute_qa_loss` plastic-style branch** — detects `splat_baseline`,
   installs a forward pre-hook on Llama layer L that calls the splat
   `inject(h_llama, blobs)`. Hook removed in `finally`.

5. **`verify_v1h.py`** — add `splat_baseline` to the variants list, with
   critical-param checks on the updater's output head, OriginHead,
   DirectionHead, W_out. Bottleneck reported as `K · (2d + 2)`.

6. **Smoke test** — 50 steps, confirm no NaN, gnorm in healthy range,
   loss decreasing. If the signed-mixture log-sum has cancellation
   issues, fix here.

7. **Full 10 K-step run** — alongside the existing 5 variants in
   `train_repr_qa.py`. Compare val_recon, per-family losses, step time.

---

## Hyperparameters (initial values)

| Param | Value | Notes |
|---|---:|---|
| d (latent dim) | 256 | Smaller than Llama's 2048; cheaper line integrals |
| K (blobs) | 64 | Bounded memory; can tune |
| K_rays | 8 | Per-position read bandwidth |
| Σ structure | diagonal | Critical for d=256 viability |
| TransformerUpdater layers | 2 | Cross-attn, 4 heads |
| TransformerUpdater dim | 256 | Matches d |
| α (L_pin coef) | 0.1 | Tune |
| β (L_proportional coef) | 0.1 | Tune; anti-collapse signal |
| λ (L_adjust coef) | 0.05 | Tune; stickiness |
| scale_init (residual gate) | 0.05 | tanh-bounded ReZero-style |
| Llama injection layer L | 8 | Mid-depth, same as plastic |

---

## File layout

```
docs/exp3_gaussian_splat_baseline.md      this file
src/repr_learning/
  splat_substrate.py                      blob state + line integral + losses + updater
  encoder.py                               + SplatBaselineEncoder
  model.py                                 + splat branch in compute_qa_loss
scripts/repr_learning/verify_v1h.py       + plastic_baseline to variants + CRITICAL_PARAMS
scripts/repr_learning/train_repr_qa.py    + splat_baseline to default variants
```
