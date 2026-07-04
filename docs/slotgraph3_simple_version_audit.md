# slotgraph3 "simple version" — build + 5-agent design audit (2026-07-04)

## TL;DR
The **simple version** (flat vector edge + `φ(src,edge,dst)` MLP expansion + additive write + a new
**per-layer id/role anchor**) is **built and mechanically verified (18/18 validity battery)**, but a
5-agent research audit finds its **design bets are mostly unsound**. The convergent verdict: it
**removes the one binding mechanism the project had already built** (the 24×24 matrix edge + keyed
delta-rule associative write) and reverts to a design with strong prior evidence of hitting the
membership wall — then adds an anchor that **re-asserts *identity* (address), not *content* (value)**.
Honest probability of *genuine* binding progress: **≈0.20**, with a real risk of a Goodhart'd
metric *false positive* or even underperforming the jun24 baseline (EM 0.31).

## What was built
The `--slotgraph3-layer-anchor` bundle (config `slotgraph3_layer_anchor`, encoder, trainer CLI):
- **WRITE**: after each frozen-LM write-suffix layer, re-inject the node/edge SLOT identity
  (`id_scale·nid + role`, unit-normalized × learned `anchor_gate` × current slot stream-norm).
- **READ**: emit `aux["prepend_struct"]` = the edge-token identity (`id_scale·(id_src+id_dst)+role`),
  re-injected before every one of the 30 frozen decoder layers via the existing reinforce hook.
- Gated to `read=edges` + `write=lm` + `write_layers>0`; fails loudly otherwise.
The rest is config selection of existing paths: `read=edges` (φ), `edge_state=flat`, `write_update=additive`.

## Debug sweep (18/18)
Config gating (3 bad combos raise) · shape invariants (boundary on/off, write_layers, K/topk) ·
`prepend_struct` is **100% identity** (captured=1.0000 in the id/role subspace, zero content leak),
boundaries zero · write-anchor rows exact · **functional lever check**: sweeping `anchor_gate` 0→1
moves distinctness monotonically the right way (emitted `mem_effrank` 14.53→15.17↑, `node_wcos`
+0.276→+0.100↓) · additive write drifts latents from init · train-mode finite, all grad groups finite.
**Caveat that the audit sharpens:** the effrank/cosine metrics are computed on `content ⊕ id`, and the
anchor adds K *orthonormal id* directions — so "the lever moves the metric" is partly the metric
*seeing the id dims*, not content re-diversifying. See finding #1.

## The three findings that matter most
1. **The anchor re-injects ADDRESS, not VALUE — and that Goodharts the diagnostic metrics.** The
   collapse is value-path (input-dependent content → shared mean). The anchor adds a *fixed,
   input-independent* id tag: `slot ≈ (collapsed content) + (fixed id_i)`. A constant carries zero
   per-input bits, so lost content is **not recovered** — nodes are distinct only by their constant
   labels. Because `mem_effrank`/`node_cos`/UMAP are computed on `content⊕id` and the ids are K
   orthonormal directions, the anchor mechanically inflates rank toward K and drops cosine
   *regardless of whether content re-diversified*. It spoofs the very instruments used to diagnose it.
   The only collapse-honest metric is the effrank of the **id-subtracted (content-only)** slots.
2. **Two independent walls; the simple version regresses on the harder one.** Wall A = over-smoothing
   (value). Wall B = pool-then-address / no multiplicative bind-in-write (a *theorem*: Set-Transformer
   /Perceiver PMA), independent of content rank. The simple version mis-targets A (address not value)
   and **demolishes the one Wall-B binder already in the repo** — the keyed delta-rule outer-product
   write `M_i ← (1−α)M_i + g(v−M·k̂)⊗k̂` — rebuilding the pre-T3 membership-wall design.
3. **Read anchor is not GCNII and is quantitatively harmful.** Raw un-normalized injection (unlike
   the norm-matched write side), **verified to accumulate to ≈4.4× the token's own norm over 30
   layers** (id_scale 3.18, token norm 3.18, 0.465/layer × 30 ≈ 13.9), 22% of it shared-DC `role[2]`
   → drowns the φ content the read exists to surface. Lose-lose gate: >0 drowns content, →0 inert.
   Plus an internal inconsistency: the anchor *forces* `read=edges` = the additive `id_src+id_dst`
   superposition the project's own geometry audit ranked **literature-worst**; and the symmetric sum
   gives `i→j` and `j→i` identical anchors (kills direction).

## Component verdicts
| Component | Verdict | Reason |
|---|---|---|
| φ over a **flat** edge | **UNSOUND** (binding) | flat vector can't key-retrieve; φ is a dst-invariant bias → 0 retrievable relation bits (Smolensky TPR; Jayakumar ICLR'20) |
| Additive write | **PARTIAL** distinct / **UNSOUND** content | preserves init better than gated-interp over short streams, but no decay → saturation, and forces `edge_write` off `assoc` |
| Anchor — write side | **PARTIAL** | bounded/norm-matched/adaptable, but wrong axis (within-window vs cross-window streaming collapse) |
| Anchor — read side | **UNSOUND** | raw 30× accumulation (4.4× token norm), drowns φ, symmetric id-sum contradicts geometry audit, lose-lose gate |
| Routing | **PARTIAL** | mechanism sound, but eval `rdiv`→0 (train fix was a noise artifact); route-by-node is hostage to collapsing `node_lat` |
| Edges-only read | **UNSOUND** | "where is Mary" needs a 2-hop (entity token → id-match to edge); dropping node tokens 3-way-overloads one vector; `mem_effrank_perex≈2` < entity floor |
| Bidir block + boundary | **SOUND** | Set-LLM geometry correct (verify `uniform_mem_pos` on) |

## Recommendation (priority order)
1. **Don't remove the binder.** Highest-EV change: run the anchor *on top of* the existing
   `matrix`+`assoc` delta-rule edge write, not as a replacement.
2. **Fix the read anchor**: norm-matched convex injection (mirror the write side), strip the shared-DC
   `role[2]`, inject on a few early layers not all 30.
3. **Add discriminating canaries before any run**: id-subspace vs content-subspace rank split (settles
   #1); `force_identity_A` control (is the graph load-bearing?); `edge_lat` ablation (does the flat
   edge store anything retrievable?) — so a Goodhart'd metric can't masquerade as success.

## Standing lesson
The instinct that the *read-side matrix engineering* felt over-built was half-right (the 24×24
factorization details were fussy) — but the **keyed associative write itself was the load-bearing
binder**, and the simple version threw out *that* rather than the fuss. The per-layer anchor is worth
keeping, but it must sit on top of a real binder and carry *content*, not a fixed label.

*(Audit: 5 parallel sub-agents — φ/flat-edge, additive write, per-layer anchor, routing/read-geometry,
and the meta "does it fix the diagnosed root" — grounded in the code + project memory + literature.)*
