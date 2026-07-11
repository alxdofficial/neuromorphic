# Graph-generative memory — the "spider web" (design memo, future / GRPO-era)

*A memory that is not compressed into slots or folded from the chain, but GENERATED: a learned policy
builds a graph summary of the observation via discrete edits — add a strand, re-route, prune — under a
fixed budget. "Next-token prediction, where the next token is a graph-edit." The score-function/RL upgrade
that sits on top of a competent differentiable substrate (`docs/slotgraph_design.md`). Read with
`docs/graph_thesis.md` (why free structure collapses) and `docs/OBJECTIVES.md` Rung 5 (trajectory/GRPO).*

## 0. The idea

Memory is a **summary of observations** — the same operation as compacting an LLM agent's context window:
re-describe the observation more succinctly without discarding what matters. Human language linearizes an
underlying graph into a chain; a maximally-efficient memory need not be a chain — it can **sprawl as a
graph**. So: a "language model" that, instead of predicting a sequence of tokens, may **start from any
position in the current expression and branch** — like a spider on its web: crawl to any node, spin new web
to reach a new position, re-route an existing strand, or prune positions it doesn't use. A **fixed token
budget** bounds the web.

At each write window the memory model sees `[input tokens ‖ persistent memory graph]` and emits a sequence
of **graph-build operations** (not literal next tokens): `ADD_NODE`, `ADD_EDGE(i,j)`, `REROUTE(edge→j')`,
`UPDATE(node/edge content)`, `PRUNE(node/edge)`, `STOP`. The result is the new persistent web.

## 1. Why this is attractive
- **Maximally expressive** — the memory is a freely-constructed graph, not a fixed-slot bottleneck or a
  chain fold; it can invent abstractions, relation-nodes, and reentrant coreference (one node, many mentions).
- **Native forced-forgetting** — the fixed budget + `PRUNE` is a stability–plasticity mechanism by
  construction (the T2 goal): to add, you must decide what to drop.
- **Matches "memory = summary"** — the write is literally "re-describe the observation as a compact web,"
  the recursive-summarization view (AutoCompressor, MemGPT compaction) made structural.
- **Non-sequential** — realizes the "language could be multi-dimensional" lens: generation order is a choice,
  not a necessity (insertion / non-monotonic generation).

## 2. Why it is the HARD / LATER version (be clear-eyed)
This front-loads every problem the project has been fighting:
- **The edits are DISCRETE decisions** (which node, which edge, prune-or-not) → the score-function/REINFORCE
  regime. There is no differentiable path through the build actions (unlike slotgraph4's dense, pathwise
  edges). This is the gradient trap, chosen deliberately.
- **There is NO target graph.** Text next-token has a ground truth; the spider web does not. Williams-Drozdov-
  Bowman (TACL 2018) is the standing warning: a freely-built structure with no target and a loss-neutral
  objective **collapses to a trivial web** (`docs/graph_thesis.md`). So the built graph must be made
  *load-bearing*: it only earns gradient by being the exclusive channel that reconstructs/answers, or by a
  downstream reward.
- **Therefore it is the GRPO-era design.** The build sequence *is* a memory trajectory; training it is the
  trajectory/GRPO objective (`docs/OBJECTIVES.md` Rung 5) — which we hold until the memory is competent.

## 3. How it's trained (when we get there)
- **Bootstrap differentiably first:** the graph state + read is the `slotgraph4` substrate (node slots +
  fixed k-regular edge states, propose→commit gated write, prepend+bidir read); train it to competence
  with behavioral-KL + MAE-CE + provenance-InfoNCE so
  the *representation* is competent before adding a discrete build policy.
- **Then add the build policy (GRPO):** sample G build trajectories per window; reward = the
  **counterfactual ablation gap** (CFPO: `CE_no_memory − CE_real`, group-baselined) — policy-independent, so
  it can't be hacked by a shuffle. Credit the reward to the build decisions.
- **Decoupled decision/content (Mem-π 2605.21463):** split advantage into a *decision* term (which edit)
  and a *content* term (what to write) — directly matches "policy over graph edits."
- **Offline bridge (COMEDY DPO):** `chosen = web→correct answer`, `rejected = web→wrong` — a stable
  preference stand-in if on-policy GRPO is unstable at 135M.
- **Supervision option:** parse text → AMR / dependency graph as a *teacher web* to warm-start the build
  policy (structure-supervised, per the URNNG/DIORA lesson that induced structure needs a guide).

## 4. Prior art
- **Graph generation:** GraphRNN (You 2018), GRAN (Liao 2019), discrete graph diffusion (Vignac 2023) —
  autoregressive/iterative graph construction under a budget.
- **Non-sequential / start-anywhere generation:** Insertion Transformer (Stern 2019), non-monotonic
  sequential text generation (Welleck 2019) — the formal "crawl to any position and branch."
- **Memory-management policies (the edits):** Memory-R1 (2508.19828, ADD/UPDATE/DELETE/NOOP + GRPO), Mem-α
  (2509.25911), Mem-π (2605.21463, decoupled decision/content) — "spin / re-route / prune" trained by RL.
- **Memory-as-summary:** AutoCompressor (2305.14788), MemGPT context compaction, recursive summarization.
- **The two lenses + collapse:** `docs/graph_thesis.md` (AMR reentrancy; NRI; Williams-2018; EntNet).

## 5. Open questions / risks
- **The no-target problem** is the crux — without a load-bearing objective the web collapses; needs the
  exclusive-channel + reward discipline, not a cleverer generator.
- **Discrete-decision variance** at 135M — GRPO may be unstable; the DPO bridge + differentiable warm-start
  are the hedges.
- **Read** — same as slotgraph4 (node-centric + top-k explicit edges); the web must still fit M tokens.
- **Budget dynamics** — when does it ADD vs PRUNE; the representative-election criterion (latest / highest-
  degree mention survives) as a learned outcome, not a rule.

## 6. Status
**FUTURE — GRPO-era.** Documented now so the design is captured; build only *after* the differentiable
substrate (`slotgraph4`) is trained to competence and the GRPO machinery (Rung 5) is in place. It is the
score-function twin of slotgraph4 (pathwise), reusing its state + read verbatim.
