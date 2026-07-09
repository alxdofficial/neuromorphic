# Training objectives — the binding ladder

Why this doc exists: the project's core finding is that **binding failed because of the OBJECTIVE, not
the model** — plain next-token CE is *loss-neutral* (the frozen LM minimizes it from its own priors, so
memory gets *used* but not *bound*; `SHUF−REAL ≈ 0`). This catalogs the objectives that break
loss-neutrality, with definitions, math, citations, and implementation notes, so we can reach for them
deliberately. Implemented objectives live in `src/memory/training/objectives.py`.

## The decomposition: USE / ADDRESSING / MEMBERSHIP

Binding is three separable sub-targets, each with its own objective family:

| sub-target | question | diagnostic | objective family |
|---|---|---|---|
| **USE** | does memory change the frozen LM's answer? | `OFF−REAL > 0` | behavioral-KL distillation |
| **ADDRESSING** | does the right key route to the right value? | routing_diversity; per-key recall | provenance / query→key InfoNCE |
| **MEMBERSHIP** | is this memory *this* example's (not another's)? | `SHUF−REAL > 0` | in-batch contrastive; input-grounding |

Plain token-wise CE targets **none** of these — the identity `E_c[KL(teacher‖student)] = I(context; answer)`
(Kujanpää 2412.14964) shows forward-KL puts gradient exactly where the context carries information, which
per-token CE does not.

**Exception — why MAE-CE is legitimate.** In verbatim reconstruction the *target is the passage itself*:
every token position is a distinct, non-guessable fact, so per-position CE is **high-rank and un-gameable**
— it's the endogenous anti-collapse pressure (ICAE, Ge 2024, arXiv:2307.06945; CALM ~99.9% recovery from
one vector). So it's a property of the *target's rank*, not the loss form. Keep token-wise CE for MAE;
never rely on it for next-token/QA binding.

---

## The ladder

### Rung 0 — MAE / verbatim reconstruction (high-rank anti-collapse). **Implemented.**
Per-position CE on the reconstructed passage. Run on every arm as the un-gameable rank floor countering
over-smoothing. `masked_reconstruction` in `model.py`.

### Rung 1 — behavioral-KL context distillation (USE). **Implemented** (`_behavioral_kl_step`).
```
L = w_ce·CE(student, y) + w_kl·KL( p_frozenLM(y | full context) ‖ p_frozenLM(y | memory) )
```
Forward-KL, temperature T≈2, on value-span answer positions; teacher = frozen LM on the full passage
(stop-grad + `disable_lora`, so it's a fixed reference); student = frozen LM on memory. Rewards the memory
to reproduce the full-context behavior = a *sufficient statistic* → forces USE.
Refs: Wingate 2210.03162; xRAG 2405.13792; DCD/Caccia 2503.08727; Cartridges 2506.06266; Kujanpää 2412.14964
(the `E[KL]=I` theorem); Padmanabhan 2306.09306 (score KL only downstream of injected content = our value-span mask).

### Rung 2 — provenance-supervised InfoNCE (ADDRESSING). *Not yet implemented — the priority "few other".*
The one that directly fights **routing collapse**. Train the write so that a query routes to the memory
token(s) **written from** the answer's source span. For a query `q` (its projected key), the positive is
the memory/edge token(s) whose provenance is the target value-span; negatives are memory tokens from other
spans (and, in-batch, other examples):
```
L_addr = − log  exp(sim(q, k+)/τ)  /  Σ_k exp(sim(q, k)/τ)          k+ = memory written from the target span
```
`sim` = cosine or scaled dot; `τ` ≈ 0.07–0.2. Provenance labels are **free in our synthetic data** (we
know which token/window each fact was written from — bio/babi/mqar). Most useful for **slotgraph3** (the
arm with an explicit routing head); largely redundant for FurlGraph (membership is free by input-grounding).
Refs: **M+ 2502.00592** (co-trained retriever, positives = tokens written from target context);
**EMAT 2210.16773** (query→key InfoNCE + KAE/VAE); TRIME (retrieval key inside the joint CE).

### Rung 3 — bypass-gap / comparative advantage (differentiable). *Not implemented — the "comparative advantage of the memory" made a loss.*
The clean formalization of "is the memory pulling its weight." Run the frozen LM twice — with memory and
*without* — and charge only when memory helps *less* than it should:
```
L_gap = λ · relu( CE_memory − stopgrad(CE_no_memory) )
```
`CE_no_memory` is a policy-independent per-example floor (the LM answering from priors alone), so the
gradient pushes the memory to beat that floor. Cheap (one extra frozen forward, no memory).
Refs: **Larimar 2403.11901** (reconstruction-through-memory + a parallel no-memory bypass term);
Compressive-Transformer stop-grad attention-reconstruction aux (Rae 1911.05507); EMAT KAE/VAE.

### Rung 4 — SHUF ≻ REAL contrastive (MEMBERSHIP). *Partially present* (the SHUF-roll InfoNCE in `objectives.py`).
Promote the eval gate to a training signal: each example's memory must out-score every *other* example's
rolled memory on its own answer (in-batch negatives = the SHUF control). "Comparative advantage of the
*right* memory." Redundant for FurlGraph (membership free by construction); keep as a slotgraph3 aux /
diagnostic. Use a SupCon same-answer mask to avoid false negatives on shared answers (bAbI locations).

### Rung 5 — trajectory / GRPO (write+read sequence). *Deferred — only once the memory is competent.*
Outcome-driven RL over a memory *trajectory* (a sequence of writes+reads across streaming windows).
- **Reward = counterfactual ablation gap (CFPO 2606.23206):** `advantage = CE_no_memory − CE_real`,
  group-baselined over G sampled memory encodings. Ablation is policy-independent (no reward-hacking a
  shuffle). Sample the *group over memory encodings* (encoder stochasticity), not decoder rollouts.
- **Decoupled decision/content (Mem-π 2605.21463):** split advantage into a **decision** term (whether/how
  to route — credit to the routing tokens) and a **content** term — *purpose-built* for the slotgraph3
  ("choose") vs FurlGraph ("inherit") A/B.
- **Full write/read episode (Memory-R1 2508.19828, Mem-α 2509.25911):** an ADD/UPDATE/DELETE/NOOP manager +
  answer agent, GRPO on downstream QA reward (GRPO > PPO here: +28% F1, faster convergence).
Keep behavioral-KL as the DENSE primary loss; GRPO only for the exact-match/execution residual.

### Offline bridge — DPO (COMEDY 2402.11975). *Optional, if on-policy GRPO is unstable at 135M.*
`chosen = memory-encoding → correct answer`, `rejected = ablated/shuffled-memory → answer`. One preference
bit, no reward model, no rollouts — a cheap, stable stand-in for GRPO on our continuous tokens.

---

## Two families (pick by variance)

- **Pathwise** (STE / gate-multiply / Gumbel / **SIMPLE exact-marginals** Ahmed 2023 / perturbed-optimizers
  Berthet 2020): low variance, needs a smooth surrogate *multiplied into something that reaches the loss*.
  Use for the graph **operator** (survival gate, soft topology) and Rungs 0–4.
- **Score-function** (REINFORCE / **GRPO** / contrastive-as-classification): no smooth surrogate needed,
  higher variance. Use for Rung 5 (trajectory-level, once memory is competent).

## Sequencing
KL + MAE-CE on both arms first (decisive, differentiable, fair) → add the bypass-gap term → provenance-InfoNCE
on slotgraph3's router (with a rank guard) → decoupled-counterfactual GRPO last. See `docs/DATA_PHASES_PLAN.md`
(Phase-1c GRPO) and `memory`/[[research_memory_landscape_objective]] for the full landscape.
