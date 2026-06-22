# Neighborhood-TokenGT read (node-centric)

**Status:** design, building. Replaces the bag-of-edges Perceiver read as the primary
read for the `graph` model. Control = the existing Perceiver (`graph_read_mode="perceiver"`).

## Why
The Perceiver read pools the 3E edge tokens into M latents as a **bag** — edges never
interact, so relational structure is never *computed*, only averaged. Diagnosis
(`project_graph_internals_diag`): the node-bank collapses to ~2 used nodes because the
soft pool lets the decoder solve from a blend → no gradient pressure on node addressing;
the select-softmax then locks. The fix is a read that (a) is node-centric so each memory
token is anchored to a specific node + its relational context, and (b) routes gradient
into the write-pointers so the topology becomes real.

## Mechanism
Inputs from the parser (per example): `node_bank [N,d]`, and per edge `e`:
`p_src[e,·], p_dst[e,·]` (pointer distributions over N, ~one-hot in the sharp regime) and
`edge_state[e,d]`.

For each of **M pre-planned slots** m:
1. **Select one node.** read-query `q_m` scores the bank (`q_m·key(node_bank)`) →
   `S[m,·]` over N via **Gumbel-softmax** (sharp, explorable). Center node embedding
   `c_m = S[m]·node_bank`.
2. **Gather its neighbourhood.** Slot↔edge incidence
   `w[m,e] = S[m]·p_src[e] + S[m]·p_dst[e]` (differentiable). Take the **top-k edges**
   by `w[m,e]` (k = degree cap) → the node's incident edges. *Gradient through `w`
   reaches both `S` (read-select) and `p_src`/`p_dst` (write-pointers) → topology repair.*
3. **TokenGT-render the window.** Tokens for slot m:
   - center: `c_m + type_center`
   - per incident edge e: `edge_state[e] + ID(src_e) + ID(dst_e) + type_edge`, where
     `ID(x) = p_x[e]·node_bank` (the bank IS the identifier table). The endpoint IDs let
     plain self-attention recover incidence (the TokenGT trick); they also carry the
     write-pointer gradient.
   - (neighbour-node tokens are implicit in the edges' endpoint IDs for v1.)
4. **Aggregate.** L `_SelfBlock` self-attention layers over the [1+k] window, batched as
   `[B·M, 1+k, d]`. Read out the **center token**.
5. **Differentiate slots.** A light self-attention over the M readouts (mirrors the
   Perceiver query self-attn) so the M memory tokens don't duplicate.
6. **Project + prepend.** `LN → Linear(d→d_llama)` → `[B, M, d_llama]`, prepended to the
   frozen decoder.

## Defaults
- `node_bank` as the ID table (decision 1).
- `k = max(1, E//8)` (`graph_read_degree_cap`), padded+masked.
- Gumbel-softmax selection (`graph_read_gumbel`, temp `graph_read_gumbel_temp`).
- M = `graph_n_read_queries` (32 in the mixed campaign), decoupled from E.

## Open / follow-ups (not in v1 core)
- **Dead-node revival** (re-seed unused bank rows toward live ones, VQ-style) — the
  select-unlock complement to Gumbel; needs a trainer hook. Tracks node-usage EMA.
- **Canary logging fix**: surface `graph_*` canaries through `run_val` (currently only
  `hlvocab_`/`spg_` are globbed) so node-use% / ptr-entropy / mem-effrank are logged.
- Straight-through on the top-k membership if soft-bias proves too weak.

## Test plan
Same mixed4k_bio harness, two variants (`perceiver` control vs `neighborhood_tokengt`),
canaries logged. Success = node-use% ≫ 2%, within-sample distinct ≫ 1%, and **SHUF−REAL > 0**
on babi/condrecon_bio (the binding signal), without losing the compression bands.
