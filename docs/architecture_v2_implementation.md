# Architecture v2: Detailed Implementation Plan

**Date:** 2026-02-26
**Prereq:** Read `architecture_v2_plan.md` for motivation and intuition.
**Scope:** Three changes — (1) batch B blocks, (2) Gated DeltaNet backbone,
(3) chunkwise parallel training. WM stays as-is for now.

---

## Phase 1: Batch the B Blocks

### What changes

Currently `model.forward_span` loops over B=4 blocks sequentially:

```python
# model.py:278-289 — CURRENT
for b, block in enumerate(self.blocks):
    h_b = block.forward_span(x_blocks_all[:,:,b], y_wm_all, ...)
    h_blocks.append(h_b)
h_final = torch.cat(h_blocks, dim=-1)  # [BS, P, D]
```

We reshape so the B dimension is absorbed into the batch dimension.
Instead of calling `block.forward_span` B times on [BS, P, D_h],
we call it once on [B*BS, P, D_h].

### Step-by-step

**1a. Merge Block into a single "BatchedBlock" module.**

All B blocks have identical architecture (same L, same layer sizes). The
only thing that differs is their weights. We stack their weights along a
new dimension and use `torch.vmap` or manual batch-dim reshaping.

**Approach: reshape B into batch dim.**

```python
# NEW: reshape [BS, P, B, D_h] → [B*BS, P, D_h]
x_all_batched = x_blocks_all.permute(2, 0, 1, 3).reshape(B*BS, P, D_h)
```

Problem: each block has its own `nn.Linear` weights. We can't just batch
the input — we need batched weights too.

**Solution: stack all block parameters into single tensors with a leading B dim.**

Instead of B separate `Layer` objects each with their own `gate_ab.weight`,
create one set of parameters with shape `[B, ...]`:

```python
# Instead of: self.blocks[b].layers[l].gate_ab.weight  [2*D_h, input_dim]
# Create:     self.gate_ab_weight[b, l]                 [B, L, 2*D_h, input_dim]
```

Then `F.linear(x_batched, weight_batched)` processes all blocks at once.

This is a significant refactor. The cleaner approach:

**Alternative: use `torch.vmap` over the block dimension.**

```python
# Define a function that runs one block
def run_block(block_params, block_buffers, x, y_wm, ...):
    return functional_call(block_module, block_params, block_buffers, (x, y_wm, ...))

# vmap it over the B dimension
batched_run = torch.vmap(run_block, in_dims=(0, 0, 0, None, ...))
h_all_blocks = batched_run(stacked_params, stacked_buffers, x_blocks_all, y_wm_all, ...)
# h_all_blocks: [B, BS, P, D_h]
```

`torch.vmap` + `torch.func.functional_call` handles the weight batching
automatically. This avoids manually restructuring all parameters.

**Complexity assessment:** `vmap` with `functional_call` requires:
- Stacking block parameters into a leading B dimension (one-time setup)
- Making the block forward function pure (no in-place state mutations)
- State tensors (h, pm_*, em_*) also need B dimension

The state mutation issue is the hard part — our blocks mutate `self.h`,
`self.pm_K`, etc. in-place. `vmap` needs pure functions. We'd need to
pass state in and get state out explicitly.

**Recommended approach: manual batching (not vmap).**

Restructure state and parameters to have an explicit B dimension:

```python
# All layer hidden states: [B, BS, D_h] instead of B separate [BS, D_h]
self.h = torch.zeros(B, BS, D_h)

# All gate weights: [B, L, 2*D_h, input_dim]
# Use einsum or bmm for batched linear

# Forward: reshape [B, BS, P, D_h] → [B*BS, P, D_h], run shared-structure
# operations, reshape back
```

### Files to modify

| File | Change |
|------|--------|
| `model.py` | Remove block loop, add batched forward |
| `block.py` | Add B dimension to all operations |
| `layer.py` | `h` state becomes `[B, BS, D_h]`, gate_ab becomes batched |
| `procedural_memory.py` | All PM state gets B dimension |
| `episodic_memory.py` | All EM state gets B dimension |
| `predictive_coding.py` | PCM state gets B dimension |
| `state.py` | `_walk_state_mixins` needs to handle batched state |
| `span_ops.py` | Boundary operations batched over B |
| All test files | Update expected shapes |

### Risk

Medium. This is a large refactor but no algorithmic changes. Every tensor
gains a B dimension. The logic is identical, just batched. Tests will catch
shape mismatches.

### Order of work

1. Add B dimension to all state tensors (h, pm_*, em_*, pcm z_hat)
2. Restructure Block to accept batched inputs [B*BS, P, D_h]
3. Restructure model.forward_span to call block once instead of B times
4. Update boundary operations (span_ops.py) to batch over B
5. Update state management (initialize, detach, reset, save/load)
6. Update tests
7. Benchmark

---

## Phase 2: Gated DeltaNet Layer

### What changes

Replace the vector recurrence in `layer.py`:

```python
# CURRENT: vector state h [BS, D_h], element-wise gated recurrence
a = sigmoid(a_raw)                    # [BS, P, D_h]
b = tanh(b_raw)                       # [BS, P, D_h]
h_t = a_t * (carry * h_{t-1}) + b_t   # element-wise, sequential scan
```

With matrix state via Gated DeltaNet:

```python
# NEW: matrix state S [BS, H, K, V], using fla kernel
q, k, v = project(input)              # [BS, P, H, K], [BS, P, H, K], [BS, P, H, V]
g = compute_forget_gate(input)         # [BS, P, H] — log-space scalar per head
beta = compute_write_gate(input)       # [BS, P, H] — sigmoid, scalar per head
o, S_new = chunk_gated_delta_rule(q, k, v, g, beta, initial_state=S)
```

### New layer architecture

```python
class DeltaNetLayer(nn.Module, StateMixin):
    _state_tensor_names = ["state"]  # was ["h"]

    def __init__(self, config, layer_idx):
        D_h = config.D_h
        self.num_heads = config.num_heads_layer        # NEW config param, e.g. 4
        self.head_dim = D_h // self.num_heads           # e.g. 128 for Tier B
        # head_v_dim can differ from head_dim, but start with same
        self.head_v_dim = self.head_dim

        # Input: x_block [D_h] + y_pm [D_h] + y_wm_proj [D_h] + y_em_proj [D_h] + surprise [D_pc]
        input_dim = 4 * D_h + surprise_dim

        # Projections (from combined input, not just x)
        self.q_proj = nn.Linear(input_dim, D_h, bias=False)
        self.k_proj = nn.Linear(input_dim, D_h, bias=False)
        self.v_proj = nn.Linear(input_dim, D_h, bias=False)

        # Gate projections (tiny — one scalar per head)
        self.a_proj = nn.Linear(input_dim, self.num_heads, bias=False)   # forget gate input
        self.b_proj = nn.Linear(input_dim, self.num_heads, bias=False)   # write gate input

        # Forget gate: Mamba2-style dt parameterization
        # A_log: learnable log of decay base (one per head)
        self.A_log = nn.Parameter(torch.log(torch.full((self.num_heads,), 0.9)))
        # dt_bias: learnable bias for softplus (one per head)
        self.dt_bias = nn.Parameter(torch.zeros(self.num_heads))

        # Short convolutions on Q, K, V (following fla's GatedDeltaNet)
        # These provide local context before the recurrence
        self.q_conv = nn.Conv1d(D_h, D_h, kernel_size=4, padding=3, groups=D_h)
        self.k_conv = nn.Conv1d(D_h, D_h, kernel_size=4, padding=3, groups=D_h)
        self.v_conv = nn.Conv1d(D_h, D_h, kernel_size=4, padding=3, groups=D_h)

        # Output projection + norm (same as current)
        self.W_o = nn.Linear(D_h, D_h)
        self.norm = nn.LayerNorm(D_h)

        # FFN (same as current)
        self.ffn_norm = nn.LayerNorm(D_h)
        self.ffn = nn.Sequential(
            nn.Linear(D_h, D_h * ffn_expansion),
            nn.GELU(approximate='tanh'),
            nn.Dropout(config.dropout),
            nn.Linear(D_h * ffn_expansion, D_h),
        )

        # Output gating (RMSNorm-gated, following fla)
        self.g_proj = nn.Linear(input_dim, D_h, bias=False)
        self.o_norm = RMSNorm(self.head_v_dim)  # per-head RMSNorm

        # State: matrix [BS, H, K, V] instead of vector [BS, D_h]
        self.state = None  # lazy init

    def _lazy_init(self, BS, device):
        self.state = torch.zeros(
            BS, self.num_heads, self.head_dim, self.head_v_dim,
            device=device, dtype=torch.float32,  # state always fp32
        )
```

### Forward pass

```python
def forward_span(self, x_all, y_pm_all, y_wm_proj_all, y_em_proj_all,
                 surprise_all, carry_all, ffn_gain_all=None):
    BS, P, D_h = x_all.shape
    H, K, V = self.num_heads, self.head_dim, self.head_v_dim

    # 1. Build combined input (same as current gate_ab input)
    u = torch.cat([x_all, y_pm_all, y_wm_proj_all, y_em_proj_all, surprise_all], dim=-1)

    # 2. Project Q, K, V from combined input
    q = self.q_proj(u)  # [BS, P, D_h]
    k = self.k_proj(u)
    v = self.v_proj(u)

    # 3. Short convolutions (local context, causal)
    q = self.q_conv(q.transpose(1,2))[:, :, :P].transpose(1,2)  # causal conv
    k = self.k_conv(k.transpose(1,2))[:, :, :P].transpose(1,2)
    v = self.v_conv(v.transpose(1,2))[:, :, :P].transpose(1,2)

    # 4. SiLU activation on Q, K (following fla convention)
    q = F.silu(q)
    k = F.silu(k)

    # 5. Reshape to multi-head: [BS, P, H, K]
    q = q.view(BS, P, H, K)
    k = k.view(BS, P, H, K)
    v = v.view(BS, P, H, V)

    # 6. Compute gates
    # Forget gate (log-space, always negative → decay)
    g = -self.A_log.float().exp() * F.softplus(self.a_proj(u).float() + self.dt_bias)
    # g: [BS, P, H]

    # Write gate (sigmoid → [0, 1])
    beta = self.b_proj(u).sigmoid()
    # beta: [BS, P, H]

    # 7. Handle doc boundary resets
    # carry_all is [BS, P, 1], 0.0 at doc boundaries
    # At boundary positions, force g to large negative (kill state)
    carry_bool = carry_all.squeeze(-1).bool()  # [BS, P]
    g = torch.where(
        carry_bool.unsqueeze(-1).expand_as(g),
        g,
        torch.full_like(g, -30.0),  # exp(-30) ≈ 0
    )

    # 8. Apply first-token reset to state
    first_carry = carry_all[:, 0, 0]  # [BS]
    keep = first_carry.view(BS, 1, 1, 1)
    self.state = self.state * keep

    # 9. DeltaNet kernel
    o, final_state = chunk_gated_delta_rule(
        q, k, v, g, beta,
        scale=1.0 / (K ** 0.5),
        initial_state=self.state.float(),
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,  # L2-normalize Q, K inside kernel
    )
    # o: [BS, P, H, V]
    # final_state: [BS, H, K, V]

    self.state = final_state  # carry to next span

    # 10. Output gating (RMSNorm-gated output, following fla)
    o = self.o_norm(o)  # per-head RMSNorm: [BS, P, H, V]
    gate = self.g_proj(u).view(BS, P, H, V).sigmoid()
    o = o * gate
    o = o.reshape(BS, P, D_h)  # merge heads

    # 11. Residual + LayerNorm + FFN (same as current)
    output = self.norm(self.W_o(o) + x_all)  # residual around recurrence
    if self.ffn is not None:
        ffn_in = self.ffn_norm(output)
        if ffn_gain_all is not None:
            ffn_in = ffn_in * ffn_gain_all
        output = output + self.ffn(ffn_in)

    return output  # [BS, P, D_h]
```

### What stays the same

- **PM interaction:** `pm.apply_batch(x)` before the layer, `pm.update_eligibility_batch()`
  after. PM reads from `x` (layer input), not from the recurrent state. PM does not
  need to know whether the backbone is a vector RNN or a matrix DeltaNet.

- **EM interaction:** EM retrieval happens at the block level before any layer.
  `y_em_proj` is passed into the layer as part of the combined input `u`. Unchanged.

- **PCM interaction:** PCM computes surprise at the block level, passed as `surprise_all`
  into the layer as part of `u`. FFN gain modulation works identically. Unchanged.

- **Residual structure:** `output = norm(W_o(recurrence_out) + x_all) + FFN(...)`.

- **Boundary operations:** PM commit, EM write, PCM boundary update are all
  span-boundary operations that don't touch the layer recurrence. Unchanged.

### What changes

| Aspect | Current | DeltaNet |
|--------|---------|----------|
| State shape | `[BS, D_h]` vector | `[BS, H, K, V]` matrix |
| State size (Tier B) | 512 floats | 4 × 128 × 128 = 65,536 floats |
| Gate computation | `gate_ab(u) → a, b` (2 × D_h) | `a_proj(u) → g`, `b_proj(u) → beta` (2 × H) |
| Scan | `h = a*h + b` sequential | `chunk_gated_delta_rule` chunkwise parallel |
| Projections | None (input feeds directly into gate) | Q, K, V projections + short convs |
| Output | `W_o(h)` | `W_o(o)` with RMSNorm gating |

### New config parameters

```python
# In ModelConfig:
num_heads_layer: int = 4          # heads for DeltaNet recurrence
head_dim_layer: int = None        # default: D_h // num_heads_layer
use_deltanet: bool = True         # False = legacy vector RNN
short_conv_kernel: int = 4        # short conv kernel size (0 = disable)
```

### State management changes

```python
# initialize_states: state is [BS, H, K, V] fp32
# detach_states: self.state = self.state.detach()
# reset_states(mask): self.state[mask] = 0
# state_dict_runtime: {"state": self.state}
```

All the same patterns, just different shape.

### PM eligibility update

Currently:
```python
layer.pm.update_eligibility_batch(x_in, x_out, surprise, reset_mask)
```

`x_in` is the layer input, `x_out` is the layer output. Both are [BS, P, D_h].
PM computes eligibility from these — it doesn't use `h` directly.

With DeltaNet, `x_in` and `x_out` are the same things (layer input/output),
so PM eligibility is completely unchanged.

---

## Phase 3: Chunkwise Parallel Training

### What changes

This is mostly "free" once Phase 2 is done. The `chunk_gated_delta_rule` kernel
from `fla` handles the parallelization. But we need to verify integration:

### Integration with our span structure

Our training processes T=256 tokens as 8 spans of P=32. Within each span, we
call `layer.forward_span` with P=32 tokens. The FLA kernel processes all P
tokens in parallel using the chunkwise algorithm.

The FLA kernel's internal chunk size is 64 (hardcoded in the Triton kernel).
Since P=32 < 64, the entire span fits in one internal chunk. This means:
- No inter-chunk state propagation within a span (just one chunk)
- The "matrix multiply trick" still applies within the chunk
- The initial_state handles cross-span propagation (our TBPTT boundary)

If we later increase P to 64 or 128, the kernel automatically uses multiple
internal chunks and propagates state between them.

### FLA kernel vs torch.compile

Currently our sequential scan runs inside torch.compile and gets fused. The
FLA kernel has `@torch.compiler.disable` which breaks compile graphs.

**For the DeltaNet layer, we should NOT use torch.compile on the recurrence.**
The FLA Triton kernel is already optimized — compile can't improve it. We
compile everything else (projections, FFN, norms) and let the FLA kernel
run natively.

Strategy:
```python
@torch.compile(mode="default")
def _pre_recurrence(self, u):
    """Projections, convs, gates — all compile-friendly."""
    q = silu(q_conv(q_proj(u)))
    k = silu(k_conv(k_proj(u)))
    v = v_conv(v_proj(u))
    g = -A_log.exp() * softplus(a_proj(u) + dt_bias)
    beta = b_proj(u).sigmoid()
    return q, k, v, g, beta

@torch.compiler.disable
def _recurrence(self, q, k, v, g, beta):
    """FLA kernel — already optimized Triton."""
    return chunk_gated_delta_rule(q, k, v, g, beta, ...)

@torch.compile(mode="default")
def _post_recurrence(self, o, u, x_all):
    """Output gating, norm, FFN — all compile-friendly."""
    o = o_norm(o) * g_proj(u).sigmoid()
    output = norm(W_o(o) + x_all) + FFN(...)
    return output
```

### Doc boundary handling

FLA's `chunk_gated_delta_rule` doesn't have a `reset_mask` parameter like our
custom scan. We handle doc boundaries by:

1. **First token of span:** Reset state before kernel call:
   `self.state = self.state * keep_mask`
2. **Mid-span boundaries:** Force `g` to -30 at boundary positions (already
   shown in the forward pass above). This makes `exp(g) ≈ 0`, effectively
   zeroing the state contribution from before the boundary.

This matches our existing FLA GLA integration pattern in `working_memory.py`.

### Fallback for CPU / non-CUDA

Keep the sequential scan as a fallback:

```python
if self.state.is_cuda and _HAS_FLA_DELTA:
    o, final_state = chunk_gated_delta_rule(...)
else:
    o, final_state = _sequential_delta_rule(q, k, v, g, beta, self.state)
```

The sequential fallback is needed for CPU tests. Implement as a simple Python
loop (same as the current `_sequential_scan` but with the delta rule math).

---

## Integration: How All Three Phases Fit Together

### Before (current architecture)

```
model.forward_span(input_ids [BS, P]):
    x_emb = embedding(input_ids)                    # [BS, P, D]
    y_wm = wm.forward_span(x_emb)                   # [BS, P, D]
    x_proj = W_in(x_emb)                             # [BS, P, D]
    x_blocks = x_proj.view(BS, P, B, D_h)            # [BS, P, B, D_h]

    for b in range(B):                                # ← SEQUENTIAL (Phase 1 fixes)
        x_b = x_blocks[:, :, b]                       # [BS, P, D_h]
        h_b = block[b].forward_span(x_b, ...)
            for l in range(L):                         # ← SEQUENTIAL (true dependency)
                y_pm = layer.pm.apply_batch(x)
                x = layer.forward_span(x, y_pm, ...)
                    u = cat(x, y_pm, y_wm, y_em, surprise)
                    a, b = gate_ab(u).chunk(2)
                    h_all = sequential_scan(a, b, h)   # ← SEQUENTIAL (Phase 2+3 fix)
                    output = norm(W_o(h_all) + x) + FFN(...)
                layer.pm.update_eligibility_batch(...)

    h_final = cat(h_blocks, dim=-1)                   # [BS, P, D]
    logits = lm_head(h_final)                          # [BS, P, vocab]
```

### After (v2 architecture)

```
model.forward_span(input_ids [BS, P]):
    x_emb = embedding(input_ids)                    # [BS, P, D]
    y_wm = wm.forward_span(x_emb)                   # [BS, P, D]
    x_proj = W_in(x_emb)                             # [BS, P, D]
    x_blocks = x_proj.view(BS, P, B, D_h)            # [BS, P, B, D_h]

    # Phase 1: batch B blocks into single call
    x_batched = x_blocks.permute(2,0,1,3)            # [B, BS, P, D_h]
                        .reshape(B*BS, P, D_h)

    h_batched = batched_block.forward_span(x_batched, ...)  # [B*BS, P, D_h]
        # Inside batched_block:
        for l in range(L):                             # still sequential (true dep)
            y_pm = pm[l].apply_batch(x)                # batched over B*BS
            x = deltanet_layer[l].forward_span(x, ...)
                u = cat(x, y_pm, y_wm, y_em, surprise)
                q, k, v = project(u)                   # [B*BS, P, H, K]
                g, beta = compute_gates(u)             # [B*BS, P, H]
                # Phase 2+3: chunkwise parallel DeltaNet
                o, state = chunk_gated_delta_rule(q, k, v, g, beta,
                                                  initial_state=state)
                output = norm(W_o(o) + x) + FFN(...)
            pm[l].update_eligibility_batch(...)

    h_final = h_batched.view(B, BS, P, D_h)           # [B, BS, P, D_h]
                       .permute(1,2,0,3)               # [BS, P, B, D_h]
                       .reshape(BS, P, D)              # [BS, P, D]
    logits = lm_head(h_final)                          # [BS, P, vocab]
```

### Key difference

The sequential scan over P=32 tokens is gone. The loop over B=4 blocks is gone.
What remains sequential is:
- L=10 layers (true data dependency — layer l+1 needs layer l output)
- 8 spans per chunk (TBPTT — state carries across spans)

These are fundamental and cannot be removed.

---

## Config Changes Summary

```python
# New fields in ModelConfig:
num_heads_layer: int = 4            # DeltaNet heads per layer
head_dim_layer: int = None          # None = D_h // num_heads_layer
use_deltanet: bool = True           # False = legacy vector RNN
short_conv_kernel: int = 4          # short conv before recurrence (0=disable)

# Removed / deprecated:
# use_fla_kernels — replaced by use_deltanet
# The old vector scan is only used as CPU fallback
```

## Test Strategy

1. **Gradient flow tests** (`test_gradients.py`): Must still pass — all params get grad.
2. **Shape tests** (`test_shapes.py`): Update expected state shapes.
3. **State tests** (`test_state.py`): detach, reset, save/load with new state shape.
4. **Numerical test**: Run both old (vector RNN) and new (DeltaNet) on same input,
   verify outputs are different but both produce valid gradients and finite loss.
5. **Legacy mode**: Keep `use_deltanet=False` path working for comparison.

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| Phase 1 (batch blocks) | Refactor state, block, model, boundary ops, tests | 1 week |
| Phase 2 (DeltaNet layer) | New layer module, config, projections, fallback scan | 1.5 weeks |
| Phase 3 (chunkwise parallel) | Wire FLA kernel, handle boundaries, compile strategy | 0.5 weeks |
| Integration testing | Full test suite, gradient verification, benchmarks | 1 week |
| **Total** | | **4 weeks** |

The phases can partially overlap: Phase 2 can start on a branch while Phase 1
is being tested. Phase 3 is trivial once Phase 2 is done (just swap kernel call).
