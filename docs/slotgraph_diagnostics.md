# slotgraph diagnostics — did the emergent graph form, or collapse?

**Date:** 2026-06-22 · **Model:** `slotgraph` (the current graph arm; supersedes the retired
relational-parser `graph` model) · **Backbone:** frozen SmolLM2-135M · **Objective:** mixed 4-task
(mae / babi / continuation / condrecon_bio), 1024→32 compression, 4000 steps.

## TL;DR
slotgraph leads the cohort on the binding task (bAbI EM **0.352** vs icae 0.273) and ties on gist —
but a four-way diagnosis shows the **emergent graph is inert**: an ablation with the structure turned
OFF performs identically (0.355), the structure machinery is sitting at its **random initialization**,
its gradient is **~100× weaker** than the content path, and 75% of the predicted edges are even
**invalid** (point to non-node slots). **The entire gain over icae comes from the fixed orthonormal
*id-tags* on the memory slots, not the graph.** This is the membership-only wall: the flat channel
(encoder-LoRA + id-tagged slots) already solves the task, so the structure gets no pressure to form.

## 1. What slotgraph is (one paragraph)
ICAE write (own frozen base + encoder-LoRA, M=32 slots appended to the passage, run through the LM's
own layers) **+** a per-LM-layer head predicting, hard via straight-through, node-vs-edge role and
(for edges) which two slot-positions it links — concretized as a TokenGT embedding `e = role +
identity` (id = fixed orthonormal per-position code; edge endpoint = transparent sum id[src]+id[dst])
and re-injected (gated by `inject_scale`) before each layer. Read = prepend the 32 slots.
`use_structure=False` = the same model with the structure path disabled.

## 2. Cohort result (4000 steps, sorted by bAbI EM)

| model | mae ↓ | **babi EM ↑** | babi loss ↓ | cont ↓ | cont-early ↓ | condrec ↓ |
|---|---|---|---|---|---|---|
| **slotgraph** | **6.574** | **0.352** | 0.970 | 3.364 | 4.022 | 2.378 |
| slotgraph (structure OFF) | 6.579 | **0.355** | 0.928 | 3.369 | — | 2.360 |
| icae | 6.581 | 0.273 | 0.984 | 3.363 | 4.017 | 2.353 |
| ccm | 6.610 | 0.242 | 1.032 | 3.373 | 4.040 | 2.358 |
| vqicae | 6.690 | 0.234 | 1.066 | 3.494 | 4.344 | 2.365 |
| autocompressor | 6.581 | 0.223 | 1.057 | 3.365 | 4.028 | 2.391 |
| biomem | 6.696 | 0.211 | 1.035 | 3.524 | 4.407 | 2.178 |
| graph (retired) | 6.634 | 0.203 | 1.026 | 3.418 | 4.157 | 2.370 |
| beacon | 6.587 | 0.176 | 1.116 | 3.405 | 4.109 | 2.478 |

## 3. The structure is inert — ablation (the decisive test)
`use_structure=False` keeps slotgraph's id-tagged slots but disables the per-layer role/edge
injection. It scores **babi EM 0.355 ≈ 0.352 with structure ON** (and ties on mae/cont/condrec). So
the emergent graph contributes nothing. The three-way attribution:

| variant | what it adds over icae | babi EM |
|---|---|---|
| icae | — | 0.273 |
| slotgraph, structure **OFF** | fixed orthonormal **id-tags** on slots | **0.355** |
| slotgraph, structure **ON** | id-tags **+ per-layer role/edges** | 0.352 |

→ **The +0.08 gain is the id-tags (distinct, addressable memory slots); the graph adds nothing.**

## 4. Structure diagnostics (`slotgraph_diag.py`, 64 bAbI examples)

| signal | value | reading |
|---|---|---|
| edge fraction | 0.229 | a real node/edge mix forms |
| **endpoint entropy** | **3.408 / 3.466 (ln32) = 98%** | edges are near-random (uniform src/dst) |
| **invalid edges** | **75% point to a non-node slot** | no edges→nodes constraint; mostly invalid graphs |
| node-target usage | 21/32 slots; top-2 = 40% | spread — but by *randomness* |
| memory eff-rank | 5.64 / 32 | healthy — slots diverse, no rank collapse |

**Node slots in UMAP, edges drawn as src→dst arrows** (edges are relations, not points; arrows that
point to non-node slots are invalid and not drawn):

![slotgraph UMAP + edges](figures/slotgraph_umap_example.png)

**Histogram panel** — note the per-edge endpoint entropy (top-right) sits on the dashed `ln32` max:

![slotgraph histograms](figures/slotgraph_histograms.png)

## 5. WHY the topology doesn't form (measured)
1. **Not a wiring bug.** All learnable topology components (`role/src/dst` heads, `inject_raw`,
   `log_temp`, `role_embed`) are `requires_grad=True` and in the optimizer; `id_embed` is a fixed
   buffer by design. So they *can* train.
2. **They didn't move.** Trained `inject_scale` = 0.1000 (its init), `temp` = 0.999 (init), and the
   role/src/dst head weight norms are within ~2% of fresh-init. The machinery sits at random init →
   hence the near-uniform (random) edges.
3. **The gradient is starved.** On the trained model, the edge-defining components get ~100–1000×
   weaker gradient than the content path:

   ```
   src_head 9.3e-3   dst_head 1.2e-2   inject_raw 2.5e-3   log_temp 1.1e-3
   vs.  slot_init 9.4e-1   encoder_LoRA 1.9   decoder_LoRA 6.2
   ```

**Root cause (confidence ~85%, measured):** the content path (encoder-LoRA + id-tagged slots) already
minimizes the loss, so the structure path offers no marginal benefit → tiny gradient → the heads and
the inject-gate stay at init → random edges → which keeps the gate small. A self-consistent dead
equilibrium = the **membership-only wall**: a free flat channel is the bypass the optimizer always
takes. (Not fully excluded: a "symmetry-stuck at the near-uniform init" optimization component; a
forced-gate run would separate it — see §6.)

## 6. How to improve the design (ranked by leverage)
1. **Close the bypass** — bottleneck the slot content / weaken the encoder-LoRA / make the read
   structure-dependent, so the graph *must* carry information. (Highest leverage; vqicae shows a blunt
   discreteness bottleneck just hurts, so it must be paired with a usable structure channel.)
2. **Make structure non-optional** — remove/force the `inject_scale` gate (it stayed at 0.10), so the
   LM must process the structure → real backward pressure on the heads. (Cheapest, most diagnostic.)
3. **Constrain edges→nodes** — mask `src/dst` to node-role slots (75% are currently invalid). Needed
   for a valid graph; not sufficient alone.
4. **Help endpoints escape the symmetric init** — Gumbel-softmax + temperature annealing. Only helps
   once pressure (1/2) exists.
5. **A structure-necessary objective** — gist/babi are solvable by flat memory, so *no architecture
   forces structure if the task doesn't need it*. A multi-hop relational task where a flat 32-slot
   bank provably fails is the real prerequisite.

**The honest takeaway:** the diagnostics say *the id-tags are the win and the graph is inert*. Before
more graph engineering, weigh (a) **following the evidence** — pursue richer/learnable memory-slot
*identities* and drop the graph framing — or (b) the **fast-weights pivot** (memory in fast synaptic
state, not prepended slots), which sidesteps the flat-prepend bypass entirely.

---
*Repro:* `scripts/diagnostics/slotgraph_diag.py` (structure + visuals), `slotgraph_gradflow.py`
(per-component gradient), `mixed_band_gate_eval.py` (REAL/SHUF/OFF gate). Ablation:
`train.py … --slotgraph-no-structure`.
