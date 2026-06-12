# Graph v9 — autonomous overnight run log (2026-06-12 → 13)

**Checkpoint:** last human-vetted commit = `a3bc7bd` on `main` (pushed). All work
below lives on branch `v9-overnight-auto`. Nothing here is vetted; treat every
change as a hypothesis probe.

## Mandate (user, 2026-06-12 evening)
1. Match memory capacity to the baselines and param count to prior graph versions.
2. Run the 600-step emat_bio gate; analyze SHUF−REAL.
3. If flat: explore all plausible reasons, autonomously, documenting as I go.

## Research posture (user, added at launch)
Stay true to the thesis: vocabulary-as-memory, input enters by selection,
lifelong-compatible mechanisms. PREFER novel ideas and clean negative results
over regressing to known-good architectures (no "just make it DeltaNet/Mamba"
fallbacks). Failures are findings if diagnosed; out-there ideas are in scope.

## The bar
- v8c5 (last graph attempt): SHUF−REAL = **−0.0007** (flat), OFF−REAL +3.30,
  recon 1.03, 52.2M trainable.
- Beacon: SHUF−REAL = **+2.07** at 102M params.
- No-memory floor: recon 1.79 / 59% top1 (objective ~⅔ guessable).
- Gate protocol: 600 steps, BS=8, emat_bio, REAL/SHUF/OFF in the val loop
  (`val_shuf_minus_real` in the jsonl). Launch flags (from the probe-script
  launch constants): `--task emat_bio --chunk-size 640 --window-size 640
  --mem-tokens 144 --emat-n-pairs 12 --steps 600 --batch-size 8`.

## Capacity matching (decision)
- Baseline read-memory anchor: v8c = 3·768·2·64 = **294,912 floats**.
- v9 gate config: d_code=256, d_key=256, nodes=(576, 288), slots=(1, 4)
  → writable fast state = 288·4·257 = **296,064 floats** (+0.4% vs anchor).
  Arm C layer 0 (atoms) is slow weights = shared vocabulary, not per-example
  memory — excluded from the budget, same logic as not counting model weights.
- **Param asymmetry, flagged loudly:** v9 lands ~3M trainable vs v8c's 52.2M and
  Beacon's 102M. This is BY DESIGN (doc §12 success criterion: binding at far
  below baseline params), and it favors the baselines — a v9 win is stronger
  for it; a v9 loss has "too few params" as a standing alternative explanation.
  A param-scaled variant is queued as a follow-up arm if time permits.

## Findings so far (pre-gate, from the debug sweep — full details in git log)
- State separability at init = 1.0000 (different docs → nearly identical
  relocated states; template-dominated coactivation). THE number to watch:
  `graph_v9_state_sep_cos_L1` must drop during training or SHUF≡REAL is
  structural. Now logged every val.
- Effective β of the strongest applied factor ≈ 0.2 at init (softmax scores
  ~1/8 × strength 1.5): the chain is GENTLE at init (apply_rotation_cos 0.985).
  Training must sharpen routing and/or grow strengths for the memory to act.
- Absorption magnitudes healthy at init (~7% strength relocation/doc).
- Routing calibration transfers to real data (eff-k 7.9-9.3 vs target 8).

## Run index
(appended as runs complete)

---
## Run 1 — arm C gate (emat_bio_v9c1) — FLAT, with a precise diagnosis
600 steps, BS=8, state-matched budget. **SHUF−REAL = −0.0004** (v8c5: −0.0007),
OFF−REAL +3.05, recon 1.04, top1 0.725 — numerically the v8c profile.

**Telemetry tells the story (the panel earned its keep):**
1. `state_sep_cos_L1` = **1.0000 at every step, never moved.** Different docs
   produce identical states → SHUF≡REAL is structural, not a read failure.
2. `dirs_rot_L1` = 0.0000 throughout: **direction content never moved off base.**
   Absorption relocated strengths (flux ≈ 11/boundary) but spread over
   288 nodes × 4 slots ⇒ per-slot incoming ≈ 0.01 vs existing 1.5 ⇒ ~0.6%
   blend/boundary ⇒ directions frozen. In effect we ran STRENGTHS-ONLY
   relocation — which the design analysis already predicted cannot bind
   (content must sit at the nodes the query activates).
3. The shared/template component dominates coactivation: emat_bio docs share
   one template, so row-normalized coact (a per-MARGINAL measure) is nearly
   doc-agnostic; the doc-specific part (state_doc_var ≈ 0.9) is negligible
   against the shared relocation.
4. Secondary: **L0 atom usage collapsing** (eff frac 0.20 → 0.05, ~26 of 576
   atoms doing everything — v8c5's disease relocated to the alphabet);
   L1 routing actually specializes per token (overlap 0.126 → 0.049, healthy).
5. Read alive (inj_ratio → 0.15; OFF−REAL +3.05 = generic steering).
   Grammar MLP barely moved (dev 0.004) — 600 steps too few for it.

**Diagnosis: absorption is too DIFFUSE and too MARGINAL-DRIVEN.** The write
spreads tiny transfers across all pairs (gate_mean ≈ 2e-4), and the gate is
driven by raw co-occurrence, which is dominated by what's common to every doc.

**Next (in-thesis, queued):**
- Arms B/A controls (deposits) — same seeding/routing, isolates write philosophy.
- H-PMI: normalize coactivation by MARGINALS (PMI/NPMI-style) so the absorb gate
  responds to doc-specific co-firing BEYOND what hot nodes predict — kills the
  template-shared component. (Precedent: our own NPMI plasticity, wave1 SDPA.)
- H-SHARP: concentrate transfers (softmax over donors w/ learnable temp instead
  of row-fraction) so doc-specific pairs move REAL mass and directions rotate.

---
## Run 2 — arm B control (deposits, absorb OFF, emat_bio_v9b1) — FLAT, and the
## discriminating result of the night so far
SHUF−REAL = **+0.0001**, OFF−REAL +3.03, recon 1.03 (numerically identical
profile to arm C and v8c5). BUT: `state_sep_cos_L1` = **0.89–0.94**, not 1.0 —
deposits DO produce measurably doc-specific states, and the gate is STILL flat.

**What this splits apart:**
- Arm C fails at the WRITE (states identical, sep=1.0000).
- Arm B writes doc-specific content (sep≈0.91) and STILL fails ⇒ doc-specific
  state ≠ ADDRESSABLE binding.

**The bridge hypothesis (now the leading explanation for both arms):**
A fact's VALUE tokens deposit into the nodes the VALUE routes to; the question
routes via the KEY phrasing into the KEY's nodes. Without a mechanism that moves
value content INTO key-routed nodes, the query never visits the content. That
bridge is exactly what absorption-with-lagged-coactivation is FOR (key-fired
nodes absorb the value-fired nodes that follow them) — but in run 1 the bridge
carried template traffic (marginal-driven gate) and no real content (diffuse).
Arm B has content but NO bridge at all (absorb off). Neither configuration has
both halves.

**Secondary arm-B observations:** beta_eff_top only 0.09–0.16 (zero-init
strengths grow slowly via deposits — the chain is weak all run);
L0 atom-usage collapse reproduces (0.17 → 0.07); read loudness ≈ 0.16.

**Queue (synthesis experiments):**
- Run 3: arm C + npmi_sharp (doc-specific bridge + concentrated mass).
- Run 4: arm B + absorb ON + npmi_sharp — deposits supply content, absorption
  supplies the key→value bridge. The two halves together. (Still in-thesis:
  this is the original full design, with the diagnosed gate fix.)
- Arm A control deferred (B already isolates deposits; A adds leaf-content only).

---
## Probe — bridge hypothesis on the TRAINED arm-B encoder (step 599)
Method: key-ish ctx positions = token ids appearing in the question (verbatim
key); value-ish = other real tokens; compare question routing vs each group.
- L0: cos(Q, key) = 0.698, cos(Q, value) = **0.741** — the question routes more
  like VALUE tokens than key tokens. Q-mass on top-32 key-nodes 0.76 vs value
  0.71 (i.e., ~75% of question routing mass fits in 32 of 576 nodes EITHER way).
- L1: cos(Q,key)=0.353 vs value 0.295; mass 0.42 vs 0.32 — mild key preference
  emerges at L1, but absolute selectivity is LOW.

**Revised diagnosis — the bridge hypothesis is downstream of a deeper problem:
HUB CONVERGENCE in routing space.** Questions, key tokens, and value tokens all
concentrate on a small shared hot-node set (matches L0 usage_eff_frac 0.05).
Even a perfect key→value bridge would deliver content into hubs shared across
all facts/entities → retrieval mixes bindings. The marginal-driven absorption
gate (run 1) was a SYMPTOM of the same root: marginals dominated by hubs.
Economics: generic hub steering improves loss (OFF−REAL +3) while specific
binding doesn't pay until addressing separates — the collapse-to-generic
attractor, now located precisely in the routing projections.

**Implications for the queue:**
- Run 3 (arm C npmi_sharp, in flight): prediction — sep_cos may lift off 1.0
  (gate de-hubbed) but SHUF−REAL likely stays small because READ addressing
  still flows through hub space. Watch exactly this.
- Run 4 candidate (addressing fix, in-thesis): per-node routing-logit CENTERING
  with running statistics (BN-style, centering only — no scale). Removes the
  always-hot logit component; running stats used identically at write and read
  (symmetry preserved — a per-window mean would break it). The v8c audit
  proposed mean-centered routing; this is its symmetric, principled form.

---
## Run 3 — arm C + npmi_sharp (emat_bio_v9c2_npmi) — FLAT; npmi alone insufficient
SHUF−REAL = **−0.0001**. Telemetry: absorb_gate_mean unchanged (~1e-4 — row-
normalized gates have fixed mean by construction; only the distribution moved),
flux ~11.7 (≈ rowfrac), refusal 0.753 (identical), `dirs_rot_L1` still 0.0000,
sep_cos 0.996–1.0 (doc_var did double 0.9 → 2.1 — slightly more doc-specific
variance, not enough).

**Why (two compounding reasons):**
1. Concentration was per-ABSORBER (each absorber's top-4 donors) but all 288
   absorbers still absorb ⇒ incoming-per-slot still ≈ flux/(N·S) ≈ 0.01 vs
   existing 1.5 ⇒ directions still frozen. Needs ABSORBER-side selectivity too
   (only strongly-coactivated/surprised absorbers take mass) if pursued further.
2. The deeper one: npmi de-hubs the GATE but coactivation itself is computed
   from hub-collapsed routing (bridge-probe finding), so even above-chance
   pairs are similar across docs ⇒ sep_cos ≈ 1 persists. ROOT CAUSE remains
   routing hub convergence.

**Decision: run 4 = arm C + npmi_sharp + ROUTE CENTERING** (the root-cause
lever): per-node logit centering removes the hubs' constant advantage at write
AND read simultaneously — attacks L0 usage collapse, coact marginal domination,
and read addressing in one mechanism.

---
## Run 4 — arm C + npmi_sharp + ROUTE CENTERING (emat_bio_v9c3_center) — FLAT,
## but the centering fix WORKED — which isolates the last culprit
SHUF−REAL = −0.0004. BUT: L0 usage_eff_frac **0.05 → 0.45–0.62** (hub collapse
cured, ~10× more effective atoms), token overlap 0.03 (distinct routing),
coact asymmetry up to 0.43. Addressing is now healthy by every metric we have —
and `state_sep_cos_L1` is STILL 1.0000.

**Decisive separation:** routing differentiates tokens; relocation still does
not differentiate DOCUMENTS. Remaining explanation: the coact table aggregates
ALL co-firing — and emat_bio docs share one template, so ROLE-pair traffic
(identical across docs) dominates the lexical/entity pairs (doc-specific).
Aggregated over 640 tokens, relocation is a low-pass of routing; doc identity
lives in the fine structure and is averaged away. (Arm B's sep=0.91 fits: its
deposits carry lexical content directly, skipping the aggregation.)

**The in-thesis fix we already own: surprise-gate the TABLE, not the boundary.**
Currently surprise multiplies the absorb gate as a per-chunk SCALAR — the coact
table itself accumulates template traffic at full weight. Fix: weight each
token's contribution to trace/coact by ITS OWN surprise (template tokens,
predictable after line 1, contribute ~nothing; entity values dominate the
table). This is predictive-coding-faithful Hebbian (local plasticity × predict-
ion error — the project's founding research lesson) applied at the right site.
→ Run 5: centering + npmi_sharp + SURPRISE-WEIGHTED COACT.

---
## Probe — does surprise discriminate content from template? (frozen base, init squash)
Grouped per-token surprise gates on a val batch: repeated/template-ish tokens
0.390, unique/content-ish 0.546, key-ish 0.445. **Ratio 1.40 per token (~2.0
for content-content coact pairs).** The signal run 5 leans on EXISTS but is
modest at the init squash; the learnable sigmoid (a·z+b) can sharpen it if
training asks. Calibrated expectation: surprise-coact shifts the table's
composition partially — watch state_sep_cos for degree, not a binary flip.
Note: 0.39 vs 0.55 also means the template still gets ~40% weight per token —
a candidate refinement (if needed) is squaring/sharpening the gate for the
TABLE only (pair products already help).

---
## Run 5 — + SURPRISE-COACT (emat_bio_v9c4_surpco) — FLAT; but it exposed a
## STRUCTURAL NULL in arm C's write
SHUF−REAL = +0.0001. The mechanism itself worked: coact mass 4150 → **890**
(template traffic filtered, 4.7×), asymmetry 0.47. sep_cos still ~0.996.

**The smoking gun was in plain sight across ALL runs: absorb_flux ≈ 11.8 in
every configuration** (rowfrac, npmi, centering, surprise-coact — identical).
Row-normalized gates (rowfrac AND donor-softmax both) fix each absorber's
TOTAL intake to ~σ(as)·s̄rp·strength̄ by construction — an identity, independent
of the gate's distribution. Combined with conservation (debit ≈ landed in
aggregate, every node both donor and absorber at similar rates) and frozen
directions, the NET strength relocation is nearly doc-invariant: state → base +
shared drift. Doc identity survives only in WHICH donors fed each absorber —
the dirs-blend channel — which is ~100× too weak (0.01 incoming vs 1.5
existing). **Arm C as configured has a structural null: normalized intake ×
conservation × frozen dirs ⇒ doc-agnostic state. The flat gates were
guaranteed.** This is the kind of negative result worth the night: precise,
mechanistic, reproducible.

**Fix (one line, in-thesis): UNNORMALIZED gate.** gate = σ(as)·NPMI_affinity·
srp·plast — absorbers with strong above-chance partners take a lot, most take
~nothing ⇒ intake totals become doc-dependent ⇒ strengths carry doc identity.
Conservation + donor outflow cap stay (still a transfer; still bounded).
→ Run 6: full stack with absorb_gate="npmi_raw".

---
## Run 6 — npmi_raw, unnormalized intake (emat_bio_v9c5_raw) — FLAT; closes the
## strength-relocation question
SHUF−REAL = +0.0002, sep_cos = 1.0000 (sixth consecutive). Unnormalized gates
RAISED flux (11.8 → 25) but LOWERED net displacement (0.045 → 0.014): real NPMI
affinities are diffuse-and-weak at this routing sharpness, so freed intake
flowed more symmetrically and netted out harder. Conservation budget exact
throughout (1728.0 ± 0.1).

**Writer-side verdict after 6 controlled runs: conserving STRENGTH relocation
over a shared base — under every gate form tried (rowfrac / npmi-sharpened /
npmi-raw, with and without centering and surprise-filtering) — does not encode
document identity at this scale. The doc-identity channel that remains is
DIRECTION motion, frozen at ~1e-5/boundary by mass-ratio blending in every run.**

→ Run 7 (last writer-side variant): GATED DIRECTION BLENDING — keep strength
conservation, but let absorber slots rotate toward their incoming donor mix at
a learnable, scale-free rate (top-landing slot blends at σ(rate), others
proportionally; init 0.5). Trades "directions move ~0" for "the top absorbing
slot genuinely takes its donors' content each boundary." Strengths stay
conserved; the invariant survives.
→ Run 8 (closer): best-shot hybrid — arm B deposits (content, sep 0.91) +
absorption bridge (key→value) + centering + surprise-coact + n_query=4 +
1200 steps (binding pressure × horizon).
