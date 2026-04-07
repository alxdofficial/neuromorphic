# Technical Implementation Plan — Cell-Grid Memory Graph

This document replaces the flat global sparse-memory plan. The LM and PCM stay
as they are in `src/model/lm.py`; the implementation work is concentrated in
`src/model/config.py` and `src/model/memory.py`.

## Guiding Principles

1. Keep the memory graph token-synchronous: one memory step per token.
2. Optimize for locality first, not maximum graph generality.
3. Store runtime state in cell-major layout.
4. Use fixed grid-local border exchange instead of arbitrary global sparse edges.
5. Keep specialization in slow or medium-rate pathways, not in the hottest
   inner-loop tensors unless it clearly pays for itself.

## Target Default Config

```python
D = 2048
D_embed = 768
L_total = 4
scan_split_at = 2
d_inner = 580

D_n = 32
N_cells = 64
grid_h = 8
grid_w = 8
neurons_per_cell = 128
K_local = 32
alpha = 4
border_per_cell = 4

mlp_groups = 8
group_hidden = 128          # reused for state/message hidden
cell_mod_hidden = 64

modulation_interval = 4
plasticity_interval = 1024
tbptt_block = 8
checkpoint_every = 8
```

Derived:

```python
C_mem = D // D_n = 64
assert N_cells == C_mem
assert grid_h * grid_w == N_cells
N_total = N_cells * neurons_per_cell = 8192
N_port_total = N_cells * alpha = 256
internal_per_cell = neurons_per_cell - 2*alpha - border_per_cell = 116
```

## Files

```
src/model/config.py
src/model/memory.py
src/model/model.py
tests/test_memory.py
tests/test_integration.py
```

`lm.py`, `pcm.py`, and `scan.py` remain structurally unchanged.

## Config Changes

`Config` should move from a flat-manifold parameterization to a cell-grid one.

### Keep

- `D`, `D_embed`, `L_total`, `scan_split_at`, `d_inner`
- `pcm_*`
- `alpha`
- `T`
- `mem_lr_scale`

### Replace / add

```python
grid_h: int = 8
grid_w: int = 8
neurons_per_cell: int = 128
K_local: int = 32
border_per_cell: int = 4
mlp_groups: int = 8
cell_mod_hidden: int = 64
modulation_interval: int = 4
tbptt_block: int = 8
checkpoint_every: int = 8
```

### Derived values

```python
N_cells = D // D_n
N = N_cells * neurons_per_cell              # compatibility alias
K = K_local                                 # compatibility alias
N_port = N_cells * alpha                    # per-side total
N_internal = N_cells * internal_per_cell
```

## MemoryGraph Layout

### Persistent runtime tensors

All runtime tensors become cell-major:

```python
h                 : [BS, NC, Cn, Dn]
msg               : [BS, NC, Cn, Dn]
w_conn            : [BS, NC, Cn, K]
decay_logit       : [BS, NC, Cn]
cell_context      : [BS, NC, Dn]
border_gate_logit : [BS, NC, B]
hebbian_traces    : [BS, NC, Cn, K]
```

where:

- `NC = N_cells`
- `Cn = neurons_per_cell`
- `Dn = D_n`
- `K = K_local`
- `B = border_per_cell`

### Fixed parameter tensors

```python
neuron_id         : [NC, Cn, Dn]
conn_idx          : [NC, Cn, K]             # local indices 0..Cn-1
cell_to_group     : [NC]

state_w1          : [G, Hs, state_in]
state_b1          : [G, Hs]
state_w2          : [G, Dn, Hs]
state_b2          : [G, Dn]

msg_w1            : [G, Hm, msg_in]
msg_b1            : [G, Hm]
msg_w2            : [G, Dn, Hm]
msg_b2            : [G, Dn]

inject_w          : [G, alpha*Dn, Dn]
inject_b          : [G, alpha*Dn]

mod_w1            : [NC, mod_in, Hmod]
mod_b1            : [NC, Hmod]
mod_w2            : [NC, Hmod, mod_out]
mod_b2            : [NC, mod_out]
```

## Cell Indexing

Within each cell:

```python
input neurons   = [0:alpha]
output neurons  = [alpha:2*alpha]
border neurons  = [2*alpha:2*alpha + border_per_cell]
internal        = [2*alpha + border_per_cell : ]
```

Border neuron order:

```python
0 = north
1 = south
2 = west
3 = east
```

## Forward Step

### 1. Modulate cells

Run only when needed:

```python
if step_idx % modulation_interval == 0:
    w_conn, decay_logit, cell_context, border_gate_logit = _modulate_cells(...)
```

The per-cell modulator consumes:

```python
h_mean         = h.mean(dim=2)              # [BS, NC, Dn]
msg_mean       = msg.mean(dim=2)            # [BS, NC, Dn]
hebb_mean      = hebbian.mean(dim=2)        # [BS, NC, K]
decay_mean     = decay_logit.mean(dim=2, keepdim=True)
cell_context   = cell_context               # [BS, NC, Dn]
```

Recommended input concat:

```python
mod_input = cat([h_mean, msg_mean, cell_context, hebb_mean, decay_mean], dim=-1)
```

Recommended output split:

```python
dw_flat        : [BS, NC, Cn*K]
ddecay_flat    : [BS, NC, Cn]
dctx           : [BS, NC, Dn]
dborder        : [BS, NC, B]
```

reshape and add:

```python
w_conn            += dw_flat.view(BS, NC, Cn, K)
decay_logit       += ddecay_flat.view(BS, NC, Cn)
cell_context      += dctx
border_gate_logit += dborder
```

### 2. Local receive

Use local cell indices only:

```python
neighbor_msgs = msg[batch_idx, cell_idx, conn_idx_expanded]  # [BS, NC, Cn, K, Dn]
w = sigmoid(w_conn).unsqueeze(-1)
received = (neighbor_msgs * w).sum(dim=3)                    # [BS, NC, Cn, Dn]
```

This is still a gather, but it is now local to contiguous per-cell storage.

### 3. Inject

```python
cell_slice = H_aug_t.view(BS, NC, Dn)                       # [BS, NC, Dn]
inject_w = inject_w[cell_to_group]                          # [NC, alpha*Dn, Dn]
inject_b = inject_b[cell_to_group]                          # [NC, alpha*Dn]
inject = einsum('bni,noi->bno', cell_slice, inject_w) + inject_b[None]
inject = inject.view(BS, NC, alpha, Dn)                     # [BS, NC, alpha, Dn]
received = received.clone()
received[:, :, :alpha] += inject
```

This is deliberately a small grouped linear projection, not a full-neuron MLP.
It gives distinct signals to different input ports while keeping the cost tiny.
Do not allocate a full zero tensor every step.

### 4. Border exchange

Extract the 4 border neuron messages, reshape to the grid, and perform fixed
neighbor exchange with slice assignments.

Sketch:

```python
border = msg[:, :, border_lo:border_hi]                      # [BS, NC, 4, Dn]
border = border.view(BS, grid_h, grid_w, 4, Dn)
incoming = torch.zeros_like(border)

incoming[:, 1:, :, 0] = border[:, :-1, :, 1]                # north
incoming[:, :-1, :, 1] = border[:, 1:, :, 0]                # south
incoming[:, :, 1:, 2] = border[:, :, :-1, 3]                # west
incoming[:, :, :-1, 3] = border[:, :, 1:, 2]                # east

incoming = incoming.view(BS, NC, 4, Dn)
gate = sigmoid(border_gate_logit).unsqueeze(-1)
received[:, :, border_lo:border_hi] += gate * incoming
```

### 5. Grouped state MLP

Broadcast `neuron_id` and `cell_context`:

```python
identity = neuron_id.unsqueeze(0).expand(BS, -1, -1, -1)
ctx = cell_context.unsqueeze(2).expand(-1, -1, Cn, -1)
state_input = cat([received, h, identity, ctx, decay_logit.unsqueeze(-1)], dim=-1)
```

Gather group weights per cell:

```python
st_w1 = state_w1[cell_to_group]   # [NC, Hs, state_in]
st_b1 = state_b1[cell_to_group]   # [NC, Hs]
st_w2 = state_w2[cell_to_group]   # [NC, Dn, Hs]
st_b2 = state_b2[cell_to_group]   # [NC, Dn]
```

Then:

```python
hidden = tanh(einsum('bnci,nhi->bnch', state_input, st_w1) + st_b1[None, :, None, :])
candidate = tanh(einsum('bnch,nih->bnci', hidden, st_w2) + st_b2[None, :, None, :])
decay = sigmoid(decay_logit).unsqueeze(-1)
h = decay * h + (1.0 - decay) * candidate
```

### 6. Grouped message MLP

```python
msg_input = cat([h, identity, ctx], dim=-1)
hidden = tanh(einsum('bnci,nhi->bnch', msg_input, mg_w1) + mg_b1[None, :, None, :])
msg = tanh(einsum('bnch,nih->bnci', hidden, mg_w2) + mg_b2[None, :, None, :]) + identity
```

### 7. Readout

```python
out_ports = msg[:, :, alpha:2*alpha]                         # [BS, NC, alpha, Dn]
readout = out_ports.sum(dim=2) * (alpha ** -0.5)            # [BS, NC, Dn]
mem_out_t = readout.reshape(BS, D)
```

### 8. Hebbian update

```python
post = msg.detach().unsqueeze(3)                             # [BS, NC, Cn, 1, Dn]
pre = neighbor_msgs.detach()                                 # [BS, NC, Cn, K, Dn]
correlation = (pre * post).sum(dim=-1)                       # [BS, NC, Cn, K]
hebbian.mul_(ema).add_(correlation, alpha=1.0 - ema)
```

## Structural Plasticity

Structural plasticity is still infrequent and stays within-cell.

Algorithm:

1. batch-average `hebbian_traces` to `[NC, Cn, K]`
2. globally choose lowest-scoring local slots to prune
3. for each pruned slot, choose a replacement inside the same cell only
4. maintain:
   - no self-connections
   - no duplicate presynaptic neighbors
   - sorted local index rows
5. zero `w_conn` and `hebbian_traces` for rewired slots

The candidate score can initially stay simple:

```python
msg_mag = msg.detach().float().norm(dim=-1).mean(dim=0)      # [NC, Cn]
```

Use that for exploit regrowth inside each cell.

## TBPTT and Checkpointing

The current branch checkpoints every token and detaches every token. That is
correctness-friendly but too aggressive.

New policy:

- keep state values across the whole segment
- detach only every `tbptt_block` tokens
- checkpoint blocks of `checkpoint_every` tokens

Recommended first implementation order:

1. get the cell-major forward path correct
2. reintroduce block-level detach
3. move checkpointing from per-step to per-block

Do not bring back per-token eager detachment in the new design.

## Implementation Sequence

### Phase 1: Docs + config + reference memory graph

- rewrite `docs/design.md`
- rewrite `docs/implementation.md`
- update `Config` to cell-grid fields and derived values
- replace flat-manifold runtime state with cell-major runtime state
- implement local gather, inject, border exchange, grouped state/msg MLPs
- keep model integration the same

### Phase 2: Tests

Update tests to validate:

- forward shape under cell-major memory
- inject/readout on the new layout
- grouped MLP gradients
- per-cell modulator gradients
- within-cell rewiring invariants
- no duplicate local neighbors
- border exchange shape and non-crash behavior

### Phase 3: Throughput cleanup

- preallocate `mem_out`
- block checkpointing
- block TBPTT
- optional `torch.compile`

### Phase 4: Fused kernel work

Once the reference path is stable, target:

- fused local gather + weight + reduction
- optional fused state/message update at the cell granularity

That is the point where the new design should start delivering its intended
throughput advantage over the current flat sparse manifold.
