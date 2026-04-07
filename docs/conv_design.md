# Conv-Based Memory Graph — Implementation Plan

This document describes the conv-based redesign of the memory graph. It replaces
sparse per-neuron connectivity with structured local neighborhoods accessed via
unfold, enabling cuDNN-friendly operations throughout.

## Design Summary

**Core change**: neurons are laid out on a 2D spatial grid. Each neuron can read
from a fixed-size local kernel window (e.g., 7×7 = 49 neighbors). Per-neuron
connectivity is controlled by a learned `w_conn` tensor that the neuromodulator
adjusts over time. The reading uses `F.unfold` (structured strided access), the
weighting uses per-neuron scalars (cheap elementwise multiply).

**What we keep**: split-scan LM, PCM, H_aug.detach(), mem_scale readout, one
memory step per token, lifelong memory (no EOS resets), block TBPTT + checkpoint.

**What changes**: sparse gather → unfold, per-cell modulator → conv modulator,
cell-based layout → flat 2D grid, border exchange → automatic (kernel crosses
boundaries), structural plasticity → continuous w_conn adaptation.

## Grid Layout

### Dimensions

```
D       = 2048       # LM hidden dim
D_n     = 32         # neuron feature channels
H_grid  = 128        # grid height (neurons)
W_grid  = 64         # grid width (neurons)
N       = 8192       # total neurons = H_grid × W_grid
K_h     = 7          # kernel height
K_w     = 7          # kernel width
K_size  = 49         # = K_h × K_w, max connections per neuron
```

The grid dimensions are chosen so that `H_grid × W_grid = N` and
`H_grid × W_grid × D_n = D × alpha` still maps cleanly to the LM dim for
inject/readout. Specifically:

```
C_mem   = D / D_n = 64          # number of LM slices
alpha   = N / C_mem = 128       # neurons per LM slice
```

Each LM slice maps to a column of 128 neurons (one column of the 128×64 grid).

### Tensor Layout

All runtime state uses channels-first layout for conv2d compatibility:

```
h          : [BS, D_n, H, W]        # neuron hidden states
msg        : [BS, D_n, H, W]        # outgoing messages
w_conn     : [BS, K_size, H, W]     # per-neuron connection weights (49 per neuron)
decay_logit: [BS, 1, H, W]          # per-neuron decay
```

Learned parameters:

```
neuron_id  : [1, D_n, H, W]         # per-neuron identity embedding (broadcast over BS)
state_w1   : [Hs, state_in, 1, 1]   # shared state MLP layer 1 (as 1×1 conv weight)
state_b1   : [Hs]
state_gs1  : [G, Hs, 1, 1]          # per-group scale for state layer 1
state_gb1  : [G, Hs, 1, 1]          # per-group bias
state_w2   : [D_n, Hs, 1, 1]        # shared state MLP layer 2
state_b2   : [D_n]
state_gs2  : [G, D_n, 1, 1]
state_gb2  : [G, D_n, 1, 1]
msg_w1     : [Hm, msg_in, 1, 1]     # shared message MLP layer 1
msg_b1     : [Hm]
msg_gs1    : [G, Hm, 1, 1]
msg_gb1    : [G, Hm, 1, 1]
msg_w2     : [D_n, Hm, 1, 1]
msg_b2     : [D_n]
msg_gs2    : [G, D_n, 1, 1]
msg_gb2    : [G, D_n, 1, 1]
mod_w1     : [mod_hidden, mod_in, 1, 1]  # modulator conv layer 1 (1×1 or 3×3)
mod_b1     : [mod_hidden]
mod_w2     : [mod_out, mod_hidden, 1, 1]
mod_b2     : [mod_out]
```

### Group Mapping

Cells from the old design become spatial regions. For grouped MLPs, we assign
each column of the grid to a group:

```python
# G = 8 groups, W_grid = 64 columns → 8 columns per group
group_map : [1, 1, H, W]    # int, group index per spatial position
# group_map[:, :, :, col] = col % G   (or col // (W_grid // G))
```

The per-group scale/bias is gathered via `gs1[group_map]` and broadcast.

## Per-Token Step

### Step 1: Receive (unfold + per-neuron weighted sum)

This is the core operation. It replaces sparse gather entirely.

```python
def _receive(msg, w_conn):
    """
    msg:    [BS, D_n, H, W]
    w_conn: [BS, K_size, H, W]   — per-neuron weights, K_size = K_h * K_w

    Returns: [BS, D_n, H, W]
    """
    # Unfold: extract local patches with structured strided access
    # padding = K_h//2 ensures output has same H, W as input
    patches = F.unfold(msg, kernel_size=(K_h, K_w), padding=(K_h//2, K_w//2))
    # patches: [BS, D_n * K_size, H * W]

    patches = patches.reshape(BS, D_n, K_size, H * W)
    # [BS, D_n, K_size, H*W]

    w = torch.sigmoid(w_conn).reshape(BS, 1, K_size, H * W)
    # [BS, 1, K_size, H*W] — broadcast over D_n

    received = (patches * w).sum(dim=2)
    # [BS, D_n, H*W]

    return received.reshape(BS, D_n, H, W)
```

**Memory cost of patches**: BS × D_n × K_size × H × W × 2B.
At BS=16, D_n=32, K_size=49, H=128, W=64: `16 × 32 × 49 × 8192 × 2 = 411 MB`.
With 5×5 kernel (K_size=25): 210 MB. With 3×3 (K_size=9): 75 MB.

**Optimization**: `F.unfold` produces a contiguous tensor with perfectly regular
memory access. The multiply + sum fuses well under torch.compile. No random
indexing anywhere.

**Variant for pre-activated w_conn**: if w_conn_act = sigmoid(w_conn) is
precomputed (outside the hot loop, recomputed at modulation), skip the sigmoid:

```python
def _receive_activated(msg, w_conn_act):
    patches = F.unfold(msg, kernel_size=(K_h, K_w), padding=(K_h//2, K_w//2))
    patches = patches.reshape(BS, D_n, K_size, H * W)
    w = w_conn_act.reshape(BS, 1, K_size, H * W)
    return (patches * w).sum(dim=2).reshape(BS, D_n, H, W)
```

### Step 2: Inject (port region assignment)

Each LM slice (D_n=32 dims) maps to one column of the grid (128 neurons tall).
With alpha=4 input ports per slice, the top 4 rows of each column are input ports.

```python
def _inject(received, H_aug_t, inject_w, inject_b):
    """
    received:  [BS, D_n, H, W]
    H_aug_t:   [BS, D]

    The LM vector maps to columns: H_aug_t reshaped to [BS, D_n, 1, W_grid]
    (one D_n-dim slice per column). Injected into the top alpha=4 rows.
    """
    # Reshape LM signal to match grid columns
    lm_signal = H_aug_t.reshape(BS, D_n, W_grid)          # [BS, D_n, W]

    # Project through learned inject weights to get alpha port signals
    # inject_w: [alpha * D_n, D_n, 1, 1] or just a linear per port
    # Simpler: replicate to alpha rows, apply learned per-port transform
    lm_expanded = lm_signal.unsqueeze(2).expand(-1, -1, alpha, -1)
    # [BS, D_n, alpha, W] — same signal for each port row

    # Add to port region (top alpha rows)
    received[:, :, :alpha, :] = received[:, :, :alpha, :] + lm_expanded

    return received
```

**Note**: This is simpler than the old design. Each column of the grid naturally
maps to one LM slice. Input ports are the top `alpha` rows, output ports the
next `alpha` rows. No reshaping between cell-major and flat layouts needed.

### Step 3: State Update (1×1 conv with group conditioning)

The state MLP is identical to the current flat F.linear approach but expressed
as 1×1 conv2d for layout consistency.

```python
def _state_update(received, h, decay, identity, group_gs1, group_gb1,
                  group_gs2, group_gb2, w1, b1, w2, b2):
    """
    All inputs: [BS, C, H, W] in channels-first layout.
    w1: [Hs, state_in, 1, 1]  (1×1 conv weight)
    group_gs1: [BS, Hs, H, W]  (pre-indexed per spatial position)
    """
    state_input = torch.cat([received, h, identity, decay], dim=1)
    # [BS, state_in, H, W]  where state_in = 3*D_n + 1 = 97

    hidden = F.conv2d(state_input, w1, b1)              # [BS, Hs, H, W]
    hidden = hidden * group_gs1 + group_gb1             # per-group conditioning
    hidden = torch.tanh(hidden)

    candidate = F.conv2d(hidden, w2, b2)                 # [BS, D_n, H, W]
    candidate = candidate * group_gs2 + group_gb2
    candidate = torch.tanh(candidate)

    return decay * h + (1.0 - decay) * candidate
```

**Why 1×1 conv instead of F.linear**: identical math (pointwise transform at
each spatial position), but cuDNN can fuse the conv + bias + elementwise ops
into fewer kernel launches. Also keeps everything in [BS, C, H, W] layout
without reshape.

**Note**: `identity` is NOT included in `cat` here — it's broadcast from the
learned `neuron_id: [1, D_n, H, W]` parameter. `cell_context` from the old
design is REMOVED — the modulator's spatial conv captures cell-level context
implicitly.

Updated state_in = D_n (received) + D_n (h) + D_n (identity) + 1 (decay) = 97.
Same as before.

### Step 4: Emit Message (1×1 conv with group conditioning)

```python
def _emit_message(h, identity, group_gs1, group_gb1, group_gs2, group_gb2,
                  w1, b1, w2, b2):
    msg_input = torch.cat([h, identity], dim=1)         # [BS, 2*D_n, H, W]
    hidden = torch.tanh(F.conv2d(msg_input, w1, b1) * group_gs1 + group_gb1)
    msg = torch.tanh(F.conv2d(hidden, w2, b2) * group_gs2 + group_gb2)
    return msg + identity
```

msg_in = 2 * D_n = 64. Same as before.

### Step 5: Readout (output port region → LM dim)

```python
def _readout(msg):
    """
    msg: [BS, D_n, H, W]
    Output ports are rows alpha..2*alpha (rows 4..7).
    Each column maps to one LM slice.
    """
    out_ports = msg[:, :, alpha:2*alpha, :]            # [BS, D_n, alpha, W]
    readout = out_ports.sum(dim=2) * (alpha ** -0.5)   # [BS, D_n, W]
    return readout.reshape(BS, -1)                     # [BS, D]
```

### Step 6: Hebbian Correlation Map

Computed every `modulation_interval` steps (e.g., every 4 tokens). Measures
per-neuron pre/post correlation for each kernel position.

```python
def _compute_correlation_map(msg_prev, msg_new):
    """
    msg_prev: [BS, D_n, H, W]  — messages from before this step
    msg_new:  [BS, D_n, H, W]  — messages after this step

    Returns: corr_map [BS, K_size, H, W]
    Each value = dot product of center neuron's new msg with neighbor's prev msg.
    """
    # Unfold prev messages into local patches
    patches = F.unfold(msg_prev, kernel_size=(K_h, K_w), padding=(K_h//2, K_w//2))
    patches = patches.reshape(BS, D_n, K_size, H * W)
    # [BS, D_n, K_size, H*W]

    # Center neuron's new message
    center = msg_new.reshape(BS, D_n, 1, H * W)
    # [BS, D_n, 1, H*W]

    # Dot product per kernel position
    corr = (patches * center).sum(dim=1)
    # [BS, K_size, H*W]

    return corr.reshape(BS, K_size, H, W)
```

**Cost**: one unfold (same as receive) + one broadcast multiply + one sum over D_n.
At BS=16 with 7×7 kernel: ~411 MB intermediate, ~0.1-0.2 ms. Runs every 4 steps,
so amortized cost is ~0.03-0.05 ms/step.

**EMA accumulation**: maintain a running correlation map:

```python
corr_ema = ema_decay * corr_ema + (1 - ema_decay) * corr_map
```

This `corr_ema: [BS, K_size, H, W]` is the hebbian trace — per-neuron, per-kernel-
position running correlation.

### Step 7: Neuromodulator (conv-based)

Runs every `modulation_interval` steps. Consumes the correlation map and
current state, outputs deltas to w_conn and decay.

```python
def _modulate(corr_ema, h, decay_logit, mod_w1, mod_b1, mod_w2, mod_b2):
    """
    corr_ema:    [BS, K_size, H, W]     — hebbian correlation map
    h:           [BS, D_n, H, W]        — current hidden state
    decay_logit: [BS, 1, H, W]          — current decay

    mod_w1: [mod_hidden, mod_in, 1, 1]  — 1×1 conv (or 3×3 for neighbor context)
    mod_w2: [mod_out, mod_hidden, 1, 1]

    mod_in  = K_size + D_n + 1 = 49 + 32 + 1 = 82
    mod_out = K_size + 1 = 50     (delta_w_conn + delta_decay)
    """
    mod_input = torch.cat([corr_ema, h, decay_logit], dim=1)
    # [BS, 82, H, W]

    hidden = torch.tanh(F.conv2d(mod_input, mod_w1, mod_b1))
    # [BS, mod_hidden, H, W]

    output = F.conv2d(hidden, mod_w2, mod_b2)
    # [BS, 50, H, W]

    delta_w = output[:, :K_size]        # [BS, 49, H, W]
    delta_decay = output[:, K_size:]    # [BS, 1, H, W]

    return delta_w, delta_decay
```

**Modulator is a 1×1 conv** (per-neuron independent). This is GPU-efficient:
it's a pointwise MLP at each spatial position, but expressed as a conv so cuDNN
can fuse it. The weights are SHARED across all positions (unlike the old per-cell
modulator). Per-neuron differentiation comes from the per-neuron inputs (corr_ema,
h, decay), which are already different at each position.

**Option: 3×3 modulator conv** for neighbor-aware modulation. The modulator at
position (i,j) would also see the correlation maps and states of the 8 immediate
neighbors, giving it local context. This adds spatial awareness to the modulation
policy. Cost increase: weight size goes from [mod_hidden, 82] to
[mod_hidden, 82, 3, 3] — 9× more params in the modulator, but still tiny.

**Parameter count**:
- 1×1 modulator: 82 × 64 + 64 + 64 × 50 + 50 = 8,498 params (shared)
- 3×3 modulator: 82 × 64 × 9 + 64 + 64 × 50 × 9 + 50 = 76,466 params (shared)

Both are negligible compared to the LM's 52M params.

## Modulator Update Application

```python
# Every modulation_interval steps:
delta_w, delta_decay = _modulate(corr_ema, h, decay_logit, ...)

w_conn = w_conn + delta_w
decay_logit = decay_logit + delta_decay

# Recompute activated versions
w_conn_act = torch.sigmoid(w_conn)
decay = torch.sigmoid(decay_logit)
```

## Config Changes

```python
@dataclass
class Config:
    # Memory Graph (conv-based)
    D_n: int = 32
    H_grid: int = 128
    W_grid: int = 64
    kernel_h: int = 7
    kernel_w: int = 7
    alpha: int = 4           # port rows per column
    mlp_groups: int = 8
    state_mlp_hidden: int = 128
    msg_mlp_hidden: int = 128
    mod_hidden: int = 64
    modulation_interval: int = 4
    hebbian_ema_decay: float = 0.995

    # Removed
    # neurons_per_cell, K, border_per_cell, cell_mod_hidden
    # structural_plasticity, plasticity_pct, plasticity_exploration_frac
    # plasticity_interval

    # Derived
    N: int = -1              # H_grid * W_grid
    K_size: int = -1         # kernel_h * kernel_w
    C_mem: int = -1          # D // D_n
```

## Runtime State

```
h            : [BS, D_n, H, W]        — neuron hidden states
msg          : [BS, D_n, H, W]        — neuron messages
w_conn       : [BS, K_size, H, W]     — per-neuron connection weights
decay_logit  : [BS, 1, H, W]          — per-neuron decay
corr_ema     : [BS, K_size, H, W]     — hebbian correlation map (EMA)
```

Total state at BS=16:
- h:          16 × 32 × 128 × 64 × 2B = 16.8 MB
- msg:        16.8 MB
- w_conn:     16 × 49 × 128 × 64 × 2B = 12.6 MB
- decay:      16 × 1 × 128 × 64 × 2B  = 0.3 MB
- corr_ema:   12.6 MB
Total: ~59 MB. Down from ~200 MB in the cell-grid design.

## Learned Parameters

| Component | Shape | Params | Notes |
|-----------|-------|--------|-------|
| neuron_id | [1, 32, 128, 64] | 262K | Per-neuron identity |
| state MLP (shared) | [128,97,1,1]+[32,128,1,1] | 16.5K | 1×1 conv |
| state group scale/bias | [8,128,1,1]×2+[8,32,1,1]×2 | 2.6K | Per-group |
| msg MLP (shared) | [128,64,1,1]+[32,128,1,1] | 12.3K | 1×1 conv |
| msg group scale/bias | [8,128,1,1]×2+[8,32,1,1]×2 | 2.6K | Per-group |
| modulator (1×1) | [64,82,1,1]+[50,64,1,1] | 8.5K | Shared 1×1 conv |
| modulator (3×3 alt) | [64,82,3,3]+[50,64,3,3] | 76.5K | Shared 3×3 conv |
| mem_scale | [D] | 2K | LM-side readout scale |
| **Total** | | **~305K** (1×1 mod) or **~373K** (3×3 mod) | |

Down from 18.8M in the cell-grid design. The massive reduction is because:
- No per-cell modulator (was 64 × 8.5K = 544K)
- No per-cell connection index buffer
- Shared conv weights replace grouped einsum weights

The model's memory capacity now lives in the **runtime state** (w_conn at 12.6 MB
carries per-neuron connectivity information) rather than in learned parameters.
The learned parameters define the dynamics; the runtime state IS the memory.

## Complete Token Step

```python
for t in range(T):
    # Modulate (every M steps)
    if t % modulation_interval == 0:
        corr_map = _compute_correlation_map(msg_prev_for_corr, msg)
        corr_ema = ema_decay * corr_ema + (1 - ema_decay) * corr_map
        delta_w, delta_decay = _modulate(corr_ema, h, decay_logit, ...)
        w_conn = w_conn + delta_w
        decay_logit = decay_logit + delta_decay
        w_conn_act = sigmoid(w_conn)
        decay = sigmoid(decay_logit)
        msg_prev_for_corr = msg.detach()

    # Receive (unfold + per-neuron weighted sum)
    received = _receive_activated(msg, w_conn_act)

    # Inject (LM signal into port rows)
    received[:, :, :alpha, :] += H_aug_t.reshape(BS, D_n, 1, W).expand(-1, -1, alpha, -1)

    # State update (1×1 conv MLP + decay blend)
    h = _state_update(received, h, decay, identity, ...)

    # Emit message (1×1 conv MLP + identity residual)
    msg = _emit_message(h, identity, ...)

    # Readout (output port rows → LM dim)
    mem_out[:, t] = _readout(msg)
```

## Performance Estimate

Per-step cost breakdown (BS=16, compiled):

| Operation | Current (cell-grid) | Conv design (est.) |
|-----------|--------------------|--------------------|
| Receive | 0.042 ms (Triton) | ~0.08 ms (unfold + mul + sum) |
| Inject | 0.070 ms | ~0.01 ms (simple slice add) |
| Border exchange | 0.041 ms | 0 ms (automatic via kernel) |
| State MLP | 0.186 ms | ~0.08 ms (1×1 conv, cuDNN fused) |
| Msg MLP | 0.160 ms | ~0.06 ms (1×1 conv, cuDNN fused) |
| Readout | 0.017 ms | ~0.01 ms |
| Hebbian (amort) | 0.012 ms | ~0.03 ms (unfold + corr) |
| Modulate (amort) | 0.034 ms | ~0.02 ms (1×1 conv) |
| **Total/step** | **~0.32 ms** (compiled) | **~0.19 ms** (est. compiled) |

128 steps forward: ~24 ms (vs current 70 ms).
Forward throughput: ~85K tok/s (vs current 29K).
With 3.2× backward ratio: ~75 ms total → ~27K tok/s training.
With cuDNN's better backward (est 2.5× ratio): ~60 ms → **~34K tok/s training**.

These are estimates. The real test is implementation + benchmark.

## Why This Is Fast

1. **F.unfold**: structured strided memory access, no random indexing. The GPU
   prefetcher works perfectly.
2. **1×1 conv2d**: cuDNN fuses conv + bias + activation into one kernel.
   Tensor core utilization is high for [BS*H*W, C_in] × [C_in, C_out] shapes.
3. **No sparse gather**: eliminates the custom Triton kernels and their
   backward pass (atomic_add scatter was a bottleneck).
4. **No cell-major reshape**: everything stays in [BS, C, H, W] throughout.
   No reshape/permute between operations.
5. **cuDNN backward for conv2d**: highly optimized, much better than our custom
   Triton backward kernels.
6. **Smaller intermediate tensors**: the unfold patches (411 MB at 7×7) are
   structured and contiguous vs the old sparse gather intermediate.

## Implementation Order

1. Update `Config` — remove cell-grid fields, add H_grid/W_grid/kernel_h/kernel_w
2. Rewrite `MemoryGraph.__init__` — new parameter shapes, remove connectivity buffers
3. Implement `_receive_activated` — unfold + per-neuron weighted sum
4. Implement `_compute_correlation_map` — unfold + dot product
5. Implement `_modulate` — 1×1 conv consuming corr_ema + state
6. Implement `_state_update` and `_emit_message` — 1×1 conv2d with group conditioning
7. Implement `_inject` and `_readout` — simple slice operations on port rows
8. Wire up `_step` and `forward_segment` — same block TBPTT + checkpoint structure
9. Update tests
10. Benchmark vs current design

## Open Questions

- Kernel size: 5×5 (25 neighbors, 210 MB unfold) vs 7×7 (49, 411 MB) vs 9×9 (81, 680 MB)?
  Start with 7×7, benchmark smaller if memory is tight.
- Modulator conv: 1×1 (per-neuron independent) vs 3×3 (neighbor-aware)?
  Start with 1×1, upgrade if modulation quality is poor.
- Group conditioning: keep 8 groups or drop to 1 (fully shared)?
  Profile both — if 1 group is nearly as fast, the extra params aren't worth it.
- Inject/readout: should inject go through a learned projection (like current inject_w)
  or just direct add? Start with direct add (simpler, faster).
