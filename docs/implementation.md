# Technical Implementation Plan

This document describes exactly how to implement the model defined in `design.md`,
structured for maximum GPU throughput. An implementing agent should be able to
write all code from this spec alone.

## Guiding Principles

1. **No Python loops over neurons.** Every operation across N=8096 neurons is
   expressed as a single bulk tensor op (matmul, gather, scatter, elementwise).
2. **Minimize kernel launches per token.** The only Python loop is over T tokens
   per segment. Each iteration should be ~5-10 kernel launches, not 50.
3. **Start with pure PyTorch.** Get correctness first with standard ops.
4. **Separate aggregation (sparse) from transformation (dense).**

## Constants (Tier-A Defaults)

```
D         = 2048    # LM hidden dim
D_embed   = 768     # embedding dim (projected up to D)
D_n       = 32      # neuron hidden dim
N         = 8096    # total neurons
K         = 64      # presynaptic connections per neuron
alpha     = 4       # port neuron multiplier
C_mem     = D / D_n = 64     # number of slice groups
N_port    = C_mem * alpha = 256  # input port count = output port count
T         = 128     # tokens per segment
L_total   = 4       # scan layers
split_at  = 2       # lower scan: layers 0..1, upper: layers 2..3
d_inner   = 580     # scan recurrence dim
C         = 16      # cortical columns for PCM
D_cc      = D / C = 128  # per-column dim
```

---

## File Structure

```
src/
  __init__.py
  data/               # (existing, unchanged)
    __init__.py
    config.py
    streaming.py
    tokenizer.py
    download.py
    debug.py
  model/
    __init__.py        # exports: Config, LM, MemoryGraph, Model
    scan.py            # (existing, unchanged) ScanLayer, fused_scan, RMSNorm
    config.py          # Config dataclass
    lm.py              # LM class
    pcm.py             # BatchedPCM class
    memory.py          # MemoryGraph class
    model.py           # Top-level Model class
  train.py             # CLI entry point
  trainer.py           # Training loop
  diagnostics.py       # Logging helpers
```

---

## 1. Config (`src/model/config.py`)

Single dataclass for all hyperparameters. Mirrors the v8 pattern but unified.

```python
@dataclass
class Config:
    # === LM ===
    D: int = 2048
    D_embed: int = 768
    L_total: int = 4
    scan_split_at: int = 2
    d_inner: int = 580
    glu_output: bool = True
    vocab_size: int = 32000
    eot_id: int = 2
    tie_embeddings: bool = True
    dropout: float = 0.1

    # === PCM ===
    pcm_enabled: bool = True
    pcm_pred_weight: float = 0.1
    pcm_hidden: int = 256
    C: int = 16                    # cortical columns

    # === Memory Graph ===
    N: int = 8096                  # total neurons
    D_n: int = 32                  # neuron hidden dim
    K: int = 64                    # connections per neuron
    alpha: int = 4                 # port multiplier
    neuromod_hidden: int = 32      # modulator MLP hidden (= D_n)
    state_mlp_hidden: int = 128    # shared state MLP hidden
    msg_mlp_hidden: int = 128      # shared message MLP hidden

    # === Structural Plasticity ===
    structural_plasticity: bool = True
    plasticity_pct: float = 0.02
    plasticity_exploration_frac: float = 0.2
    plasticity_interval: int = 1024  # tokens between rewiring (= 8 segments)
    hebbian_ema_decay: float = 0.995 # EMA decay for hebbian traces (per-token)

    # === Training ===
    T: int = 128              # tokens per segment
    mem_lr_scale: float = 0.3

    # === Derived (set by validate()) ===
    D_cc: int = -1                 # D // C
    C_mem: int = -1                # D // D_n
    N_port: int = -1               # C_mem * alpha
    N_internal: int = -1           # N - 2 * N_port

    def validate(self):
        assert self.D % self.C == 0
        self.D_cc = self.D // self.C
        assert self.D % self.D_n == 0
        self.C_mem = self.D // self.D_n
        self.N_port = self.C_mem * self.alpha
        assert 2 * self.N_port <= self.N, \
            f"Need 2*N_port={2*self.N_port} <= N={self.N}"
        self.N_internal = self.N - 2 * self.N_port
        assert self.K < self.N
        assert self.scan_split_at >= 1
        assert self.scan_split_at < self.L_total

    @classmethod
    def tier_a(cls, **kw) -> "Config":
        c = cls(**kw)
        c.validate()
        return c

    @classmethod
    def tier_tiny(cls, **kw) -> "Config":
        """For unit tests."""
        defaults = dict(
            D=64, D_embed=64, C=4, L_total=4, scan_split_at=2,
            d_inner=64, glu_output=False, vocab_size=256, T=8,
            N=128, D_n=8, K=16, alpha=2,
            neuromod_hidden=8, state_mlp_hidden=32, msg_mlp_hidden=32,
            pcm_hidden=32, structural_plasticity=False,
        )
        defaults.update(kw)
        c = cls(**defaults)
        c.validate()
        return c
```

---

## 2. PCM (`src/model/pcm.py`)

Port directly from `shared-scan:src/v8/pcm.py` (class `BatchedPCM`). The
interface is unchanged:

```python
class BatchedPCM(nn.Module):
    """Dynamic predictive coding — predicts state transitions."""

    def __init__(self, C: int, D_cc: int, hidden: int = 256): ...

    def forward(self, H: Tensor) -> tuple[Tensor, Tensor, Tensor, list[float]]:
        """
        Args:
            H: [BS, T, C, D_cc] — scan hidden states per column
        Returns:
            surprise:       [BS, T, C, D_cc]
            delta_hat:      [BS, T, C, D_cc]
            pred_loss:      scalar
            per_cc_loss:    list[float] of length C
        """
```

Copy verbatim from the shared-scan branch. The only change: import path
moves from `src.v8.pcm` to `src.model.pcm`.

---

## 3. LM (`src/model/lm.py`)

### Class: `LM(nn.Module)`

#### `__init__(self, config: Config)`

**Parameters created:**

| Name | Shape | Init | Notes |
|------|-------|------|-------|
| `embedding` | Embedding(vocab, D_embed) | N(0,1) | Standard |
| `proj_up` | Linear(D_embed, D) | Kaiming | Only if D_embed != D |
| `proj_down` | Linear(D, D_embed) | Kaiming | Only if D_embed != D |
| `pos_embed` | Parameter [T, D] | N(0, 0.02) | Positional encoding |
| `layers` | ModuleList of L_total ScanLayer | See scan.py | Each: ScanLayer(D, d_inner, dropout, n_layers=L_total, glu_output) |
| `pcm` | BatchedPCM(C, D_cc, pcm_hidden) | See PCM | Only if pcm_enabled |
| `split_mlp` | Sequential(Linear(2D, d_inner), SiLU, Linear(d_inner, D)) | Depth-scaled | Combines H_mid + surprise |
| `mem_scale` | Parameter [D] | sqrt(alpha) | Learnable readout scale |
| `ln_final` | LayerNorm(D_embed) | Standard | Before LM head |
| `lm_head` | Linear(D_embed, vocab, bias=False) | Tied to embedding.weight if tie_embeddings | |

**`split_mlp` init detail:**
```python
self.split_mlp = nn.Sequential(
    nn.Linear(2 * D, d_inner),
    nn.SiLU(),
    nn.Linear(d_inner, D),
)
# Depth-scaled init on output layer:
with torch.no_grad():
    self.split_mlp[2].weight.mul_(1.0 / math.sqrt(2 * L_total))
```

**`mem_scale` init detail:**
```python
# alpha replicas are summed and scaled by 1/sqrt(alpha) in readout.
# mem_scale starts at sqrt(alpha) to cancel that, giving ~unit magnitude.
self.mem_scale = nn.Parameter(torch.full((D,), math.sqrt(config.alpha)))
```

**Scan carries:** `self._carries = [None] * L_total` (mutable list, not parameter).

#### `forward_scan_lower(self, input_ids, reset_mask=None)`

```
Args:
    input_ids: [BS, T] int64
    reset_mask: [BS, T] bool or None — True at positions to reset recurrence
Returns:
    H_mid:    [BS, T, D]  — hidden states after lower scan layers
    surprise: [BS, T, D]  — PCM surprise stacked to full D (or zeros if PCM off)
    aux_loss: scalar       — PCM prediction loss
```

**Computation:**
```python
# 1. Embed
x = self.embedding(input_ids)                    # [BS, T, D_embed]
if self.proj_up is not None:
    x = self.proj_up(x)                          # [BS, T, D]
x = x + self.pos_embed[:T]                       # [BS, T, D]

# 2. Lower scan layers (0..split_at-1)
H = x
for i in range(self.config.scan_split_at):
    H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
    self._carries[i] = h_last                    # [BS, d_inner]

H_mid = H  # [BS, T, D]

# 3. PCM
aux_loss = torch.tensor(0.0, device=H.device)
if self.pcm is not None:
    H_cols = H_mid.view(BS, T, C, D_cc)          # [BS, T, C, D_cc]
    surprise_cols, _, pred_loss, _ = self.pcm(H_cols)
    surprise = surprise_cols.reshape(BS, T, D)    # [BS, T, D]
    aux_loss = pred_loss * self.config.pcm_pred_weight
else:
    surprise = torch.zeros_like(H_mid)

return H_mid, surprise, aux_loss
```

#### `augment(self, H_mid, surprise)`

Combines H_mid and surprise into H_aug. Called after lower scan, before memory.

```
Args:
    H_mid:    [BS, T, D]
    surprise: [BS, T, D]
Returns:
    H_aug:    [BS, T, D]
```

**Computation:**
```python
# RMSNorm on surprise to prevent unbounded growth
surp_rms = surprise.pow(2).mean(dim=-1, keepdim=True).add(1e-6).rsqrt()
surprise_normed = surprise * surp_rms              # [BS, T, D]

# split_mlp: residual connection
H_aug = H_mid + self.split_mlp(
    torch.cat([H_mid, surprise_normed], dim=-1)    # [BS, T, 2D]
)  # [BS, T, D]
return H_aug
```

#### `inject_memory(self, H_aug, mem_out)`

```
Args:
    H_aug:   [BS, T, D]
    mem_out: [BS, T, D]  — from memory graph
Returns:
    H_enriched: [BS, T, D]
```

**Computation:**
```python
return H_aug + self.mem_scale * mem_out.to(H_aug.dtype)
```

#### `forward_scan_upper(self, H_enriched, reset_mask=None)`

```
Args:
    H_enriched: [BS, T, D]
    reset_mask: [BS, T] bool or None
Returns:
    H: [BS, T, D]
```

**Computation:**
```python
H = H_enriched
for i in range(self.config.scan_split_at, self.config.L_total):
    H, h_last = self.layers[i](H, self._carries[i], reset_mask=reset_mask)
    self._carries[i] = h_last
return H
```

#### `forward_output(self, H)`

```
Args:
    H: [BS, T, D]
Returns:
    logits: [BS, T, vocab_size]
```

**Computation:**
```python
out = H
if self.proj_down is not None:
    out = self.proj_down(out)                    # [BS, T, D_embed]
out = self.ln_final(out)                         # [BS, T, D_embed]
return self.lm_head(out)                         # [BS, T, vocab_size]
```

#### Carry management

```python
def detach_carries(self):
    for i, h in enumerate(self._carries):
        if h is not None:
            self._carries[i] = h.detach()

def reset_carries(self, mask: Tensor):
    """mask: [BS] bool — True for batch elements to reset."""
    for i, h in enumerate(self._carries):
        if h is not None:
            keep = (~mask).to(h.dtype).unsqueeze(-1)  # [BS, 1]
            self._carries[i] = h * keep
```

---

## 4. Memory Graph (`src/model/memory.py`)

### Class: `MemoryGraph(nn.Module)`

This is the core of the system. All neuron dynamics are here.

#### `__init__(self, config: Config)`

**Buffers (not trained, not in optimizer):**

| Name | Shape | Dtype | Init |
|------|-------|-------|------|
| `conn_idx` | [N, K] | int64 | Random unique non-self indices per neuron, sorted |

**Init `conn_idx`:**
```python
N, K = config.N, config.K
scores = torch.rand(N, N)
scores[torch.arange(N), torch.arange(N)] = -float('inf')  # no self-connections
_, top_k = scores.topk(K, dim=1)                          # [N, K]
conn_idx, _ = top_k.sort(dim=-1)                          # sort for cache locality
self.register_buffer('conn_idx', conn_idx)
```

**Learned parameters:**

| Name | Shape | Dtype | Init | Notes |
|------|-------|-------|------|-------|
| `neuron_id` | [N, D_n] | f32 | N(0, 0.02) | Per-neuron identity embedding |
| `state_w1` | [state_hidden, state_in] = [128, 97] | f32 | Kaiming | F.linear convention: [out, in] |
| `state_b1` | [state_hidden] = [128] | f32 | zeros | |
| `state_w2` | [D_n, state_hidden] = [32, 128] | f32 | Kaiming, depth-scaled ×0.1 | |
| `state_b2` | [D_n] = [32] | f32 | zeros | |
| `msg_w1` | [msg_hidden, msg_in] = [128, 64] | f32 | Kaiming | msg_in = 2*D_n |
| `msg_b1` | [msg_hidden] = [128] | f32 | zeros | |
| `msg_w2` | [D_n, msg_hidden] = [32, 128] | f32 | Kaiming, depth-scaled ×0.1 | |
| `msg_b2` | [D_n] = [32] | f32 | zeros | |
| `mod_w1` | [N, mod_in, neuromod_hidden] | f32 | Kaiming | mod_in = D_n + K + K + 1 = 161 |
| `mod_b1` | [N, 1, neuromod_hidden] | f32 | zeros | unsqueezed for bmm broadcast |
| `mod_w2` | [N, neuromod_hidden, mod_out] | f32 | Kaiming × 0.01 | mod_out = K + 1 + D_n = 97 |
| `mod_b2` | [N, 1, mod_out] | f32 | zeros | unsqueezed for bmm broadcast |

**`state_w2` / `msg_w2` depth-scaled init:**
```python
nn.init.kaiming_uniform_(self.state_w2, a=math.sqrt(5))
with torch.no_grad():
    self.state_w2.mul_(0.1)  # small initial updates
# Same for msg_w2
```

**`mod_w2` small init:**
```python
nn.init.kaiming_uniform_(self.mod_w2, a=math.sqrt(5))
with torch.no_grad():
    self.mod_w2.mul_(0.01)  # near-zero initial deltas
```

**Store constants:**
```python
self.C_mem = config.C_mem          # 64
self.N_port = config.N_port        # 256
self.alpha = config.alpha          # 4
self.D_n = config.D_n              # 32
self.N = config.N                  # 8096
self.K = config.K                  # 64
```

#### `initialize_states(self, BS: int, device: torch.device)`

Creates all mutable per-batch state tensors. Called once before training starts.
All in **bf16** except co_activation_ema (f32).

```python
def initialize_states(self, BS: int, device: torch.device):
    N, D_n, K = self.N, self.D_n, self.K
    dt = torch.bfloat16

    self.h = torch.randn(BS, N, D_n, device=device, dtype=dt) * 0.01
    self.msg = torch.zeros(BS, N, D_n, device=device, dtype=dt)
    self.w_conn = torch.zeros(BS, N, K, device=device, dtype=dt)
    self.identity = self.neuron_id.unsqueeze(0).expand(BS, -1, -1).clone().to(dt)
        # [BS, N, D_n] — starts as copy of learned neuron_id, modulator adjusts per-batch
    self.decay_logit = torch.zeros(BS, N, device=device, dtype=dt)
        # sigmoid(0) = 0.5 → equal blend of old/new state at init
    self.hebbian_traces = torch.zeros(BS, N, K, device=device, dtype=dt)

    self._initialized = True
```

#### `detach_states(self)`

Called at segment boundaries for TBPTT.

```python
def detach_states(self):
    self.h = self.h.detach()
    self.msg = self.msg.detach()
    self.w_conn = self.w_conn.detach()
    self.identity = self.identity.detach()
    self.decay_logit = self.decay_logit.detach()
    # hebbian_traces are already outside autograd (EMA updated with no_grad)
```

#### `reset_states(self, mask: Tensor)`

Reset state for batch elements hitting document boundaries.

```python
def reset_states(self, mask: Tensor):
    """mask: [BS] bool — True for elements to reset."""
    keep = (~mask).to(self.h.dtype)
    k3 = keep[:, None, None]   # [BS, 1, 1]
    k2 = keep[:, None]         # [BS, 1]
    with torch.no_grad():
        self.h = self.h * k3
        self.msg = self.msg * k3
        self.w_conn = self.w_conn * k3
        self.decay_logit = self.decay_logit * k2
        self.hebbian_traces = self.hebbian_traces * k3
        # Re-init identity from learned neuron_id for reset elements
        reset_id = self.neuron_id.unsqueeze(0).to(self.identity.dtype)
        r3 = mask.to(self.identity.dtype)[:, None, None]
        self.identity = self.identity * k3 + reset_id * r3
```

#### `_modulate(self, identity, hebbian, w_conn, decay_logit)`

Per-neuron neuromodulator. Predicts DELTAS to w_conn, decay_logit, identity.

```
Args:
    identity:      [BS, N, D_n]
    hebbian:       [BS, N, K]
    w_conn:        [BS, N, K]
    decay_logit:   [BS, N]
Returns:
    new_w_conn:      [BS, N, K]
    new_decay_logit: [BS, N]
    new_identity:    [BS, N, D_n]
```

**Computation (exact):**
```python
def _modulate(self, identity, hebbian, w_conn, decay_logit,
              mod_w1, mod_b1, mod_w2, mod_b2):
    K, D_n = self.K, self.D_n

    # 1. Assemble input: [BS, N, mod_in]  where mod_in = D_n + K + K + 1 = 161
    mod_input = torch.cat([
        identity,                          # [BS, N, D_n=32]
        hebbian,                           # [BS, N, K=64]
        w_conn,                            # [BS, N, K=64]
        decay_logit.unsqueeze(-1),         # [BS, N, 1]
    ], dim=-1)                             # [BS, N, 161]

    # 2. Per-neuron batched GEMM
    # Permute to [N, BS, 161] for torch.bmm
    x = mod_input.permute(1, 0, 2)        # [N, BS, 161]

    # Weights are pre-cast to bf16 before the token loop (avoid 68M param
    # cast per step). Passed in as arguments.
    hidden = torch.tanh(torch.bmm(x, mod_w1) + mod_b1)  # [N, BS, 32]
    output = torch.bmm(hidden, mod_w2) + mod_b2          # [N, BS, 97]
    output = output.permute(1, 0, 2)                      # [BS, N, 97]

    # 3. Unpack deltas
    dw       = output[..., :K]            # [BS, N, K=64]
    ddecay   = output[..., K]             # [BS, N]
    didentity = output[..., K+1:]         # [BS, N, D_n=32]

    # 4. Apply deltas
    new_w_conn      = w_conn + dw
    new_decay_logit = decay_logit + ddecay
    new_identity    = identity + didentity

    return new_w_conn, new_decay_logit, new_identity
```

#### `_receive(self, msg, w_conn)`

Sparse gather + weighted sum. All N neurons, vectorized.

```
Args:
    msg:    [BS, N, D_n]  — messages from previous step
    w_conn: [BS, N, K]    — connection weights
Returns:
    received: [BS, N, D_n]
    gathered: [BS, N, K, D_n]  — kept for hebbian trace update
```

**Computation:**
```python
def _receive(self, msg, w_conn):
    # Gather presynaptic messages
    # conn_idx: [N, K] int64
    # msg[:, conn_idx] does: for each neuron n, gather msg[:, conn_idx[n, :], :]
    gathered = msg[:, self.conn_idx]              # [BS, N, K, D_n]

    # Weight by sigmoid(w_conn)
    w = torch.sigmoid(w_conn).unsqueeze(-1)       # [BS, N, K, 1]
    received = (gathered * w).sum(dim=2)           # [BS, N, D_n]

    return received, gathered
```

#### `_inject(self, received, H_aug_t)`

Add LM signal to input port neurons' received messages.

```
Args:
    received: [BS, N, D_n]  — will be modified in-place for port neurons
    H_aug_t:  [BS, D]       — augmented LM hidden state at this token
Returns:
    received: [BS, N, D_n]  — modified (input ports have inject added)
```

**Computation:**
```python
def _inject(self, received, H_aug_t):
    BS = H_aug_t.shape[0]
    C_mem, alpha, D_n = self.C_mem, self.alpha, self.D_n

    # Reshape LM vector into slice groups and replicate
    inject = H_aug_t.reshape(BS, C_mem, D_n)                 # [BS, 64, 32]
    inject = inject.unsqueeze(2).expand(-1, -1, alpha, -1)   # [BS, 64, 4, 32]
    inject = inject.reshape(BS, self.N_port, D_n)            # [BS, 256, 32]

    # Pad to full N dimension: [BS, N, D_n] with zeros for non-port neurons.
    # This avoids clone + in-place slice assignment, which is fragile with autograd.
    inject_full = torch.zeros(
        BS, self.N, D_n, device=received.device, dtype=received.dtype)
    inject_full[:, :self.N_port] = inject.to(received.dtype)

    return received + inject_full  # single fused add, autograd-safe
```

#### `_state_update(self, received, h, identity, decay_logit)`

Shared-weight state MLP applied to all neurons via standard GEMM.

```
Args:
    received:    [BS, N, D_n]
    h:           [BS, N, D_n]   — current hidden state
    identity:    [BS, N, D_n]
    decay_logit: [BS, N]
Returns:
    h_new: [BS, N, D_n]
```

**Computation:**
```python
def _state_update(self, received, h, identity, decay_logit,
                  w1, b1, w2, b2):
    BS = received.shape[0]

    # 1. Concatenate MLP input: [BS, N, 3*D_n + 1 = 97]
    state_input = torch.cat([
        received,                              # [BS, N, 32]
        h,                                     # [BS, N, 32]
        identity,                              # [BS, N, 32]
        decay_logit.unsqueeze(-1),             # [BS, N, 1]
    ], dim=-1)                                 # [BS, N, 97]

    # 2. Flatten for GEMM: [BS*N, 97]
    flat = state_input.reshape(-1, state_input.shape[-1])

    # 3. Two-layer MLP (weights pre-cast before token loop)
    hidden = torch.tanh(F.linear(flat, w1, b1))        # [BS*N, 128]
    candidate = torch.tanh(F.linear(hidden, w2, b2))   # [BS*N, 32]
    candidate = candidate.reshape(BS, self.N, self.D_n) # [BS, N, 32]

    # 4. Structural decay blend
    decay = torch.sigmoid(decay_logit).unsqueeze(-1)   # [BS, N, 1]
    h_new = decay * h + (1.0 - decay) * candidate      # [BS, N, 32]

    return h_new
```

**Note on F.linear shapes:** `F.linear(input, weight, bias)` computes
`input @ weight.T + bias`. So `weight` shape is `[out_features, in_features]`.
The parameter shapes listed above follow this convention:
- `state_w1: [state_hidden, state_in]` i.e. `[128, 97]`
- `state_w2: [D_n, state_hidden]` i.e. `[32, 128]`

(Same for msg MLP weights.)

#### `_emit_message(self, h, identity)`

Shared-weight message MLP applied to all neurons.

```
Args:
    h:        [BS, N, D_n]  — updated hidden state
    identity: [BS, N, D_n]
Returns:
    msg_new: [BS, N, D_n]
```

**Computation:**
```python
def _emit_message(self, h, identity, w1, b1, w2, b2):
    BS = h.shape[0]

    # 1. Concatenate: [BS, N, 2*D_n = 64]
    msg_input = torch.cat([h, identity], dim=-1)  # [BS, N, 64]

    # 2. Flatten: [BS*N, 64]
    flat = msg_input.reshape(-1, msg_input.shape[-1])

    # 3. Two-layer MLP (weights pre-cast before token loop)
    hidden = torch.tanh(F.linear(flat, w1, b1))    # [BS*N, 128]
    msg_raw = torch.tanh(F.linear(hidden, w2, b2)) # [BS*N, 32]
    msg_new = msg_raw.reshape(BS, self.N, self.D_n) # [BS, N, 32]

    # 4. Add identity residual (helps neurons identify message sender)
    msg_new = msg_new + identity

    return msg_new
```

#### `_readout(self, msg)`

Collect output port neuron messages, reassemble to LM dim.

```
Args:
    msg: [BS, N, D_n]
Returns:
    readout: [BS, D]
```

**Computation:**
```python
def _readout(self, msg):
    BS = msg.shape[0]

    # Output ports are neurons at indices N_port..2*N_port (i.e. 256..511)
    port_msg = msg[:, self.N_port : 2 * self.N_port]   # [BS, 256, 32]

    # Reshape to [BS, C_mem, alpha, D_n] = [BS, 64, 4, 32]
    port_msg = port_msg.reshape(BS, self.C_mem, self.alpha, self.D_n)

    # Sum over alpha replicas, scale by 1/sqrt(alpha)
    readout = port_msg.sum(dim=2) * (self.alpha ** -0.5)  # [BS, 64, 32]

    # Flatten to LM dim
    return readout.reshape(BS, -1)             # [BS, 2048]
```

#### `_update_hebbian(self, gathered, msg_new)`

EMA update of per-connection correlation traces. Runs outside autograd.

```
Args:
    gathered: [BS, N, K, D_n]  — presynaptic messages from _receive
    msg_new:  [BS, N, D_n]     — this neuron's new outgoing message
```

**Computation:**
```python
def _update_hebbian(self, gathered, msg_new, hebbian):
    """Update hebbian traces in-place. Called every token step.

    Args:
        gathered: [BS, N, K, D_n] — presynaptic messages (detach before use)
        msg_new:  [BS, N, D_n]    — this neuron's new outgoing message
        hebbian:  [BS, N, K]      — traces to update (modified in-place)
    """
    with torch.no_grad():
        # Correlation: dot product of presynaptic msg and postsynaptic msg
        post = msg_new.detach().unsqueeze(2)               # [BS, N, 1, D_n]
        pre = gathered.detach()                            # [BS, N, K, D_n]
        correlation = (pre * post).sum(dim=-1)             # [BS, N, K]

        # EMA update in-place
        hebbian.mul_(0.995).add_(correlation, alpha=0.005)
```

#### `forward_segment(self, H_aug: Tensor) -> Tensor`

Main entry point. Processes T tokens sequentially, all N neurons vectorized per step.

```
Args:
    H_aug: [BS, T, D]  — augmented LM hidden states (detached from LM graph)
Returns:
    mem_out: [BS, T, D] — readout signal for each token
```

**Computation:**
```python
def forward_segment(self, H_aug: Tensor) -> Tensor:
    BS, T, D = H_aug.shape
    device = H_aug.device
    mem_out = torch.empty(BS, T, D, device=device, dtype=H_aug.dtype)

    # Detach persistent state at segment boundary (TBPTT)
    h = self.h.detach()
    msg = self.msg.detach()
    w_conn = self.w_conn.detach()
    identity = self.identity.detach()
    decay_logit = self.decay_logit.detach()
    hebbian = self.hebbian_traces  # outside autograd, but updated per-token

    # Cast H_aug to memory compute dtype
    H_aug = H_aug.to(h.dtype)

    # Pre-cast ALL weights once before the loop.
    # Avoids repeated f32→bf16 casts: 68M modulator + 29K shared = ~68M params.
    # Without this: 128 steps × 12 weight casts = 1536 unnecessary kernel launches.
    dt = h.dtype
    mod_w1 = self.mod_w1.to(dt)         # [N, 161, 32]
    mod_b1 = self.mod_b1.to(dt)         # [N, 1, 32]
    mod_w2 = self.mod_w2.to(dt)         # [N, 32, 97]
    mod_b2 = self.mod_b2.to(dt)         # [N, 1, 97]
    st_w1 = self.state_w1.to(dt)        # [128, 97]
    st_b1 = self.state_b1.to(dt)        # [128]
    st_w2 = self.state_w2.to(dt)        # [32, 128]
    st_b2 = self.state_b2.to(dt)        # [32]
    mg_w1 = self.msg_w1.to(dt)          # [128, 64]
    mg_b1 = self.msg_b1.to(dt)          # [128]
    mg_w2 = self.msg_w2.to(dt)          # [32, 128]
    mg_b2 = self.msg_b2.to(dt)          # [32]

    for t in range(T):
        H_aug_t = H_aug[:, t]                             # [BS, D]

        # Step 0: Neuromodulate — updates w_conn, decay_logit, identity
        # Uses CURRENT hebbian traces (updated at end of each step)
        w_conn, decay_logit, identity = self._modulate(
            identity, hebbian, w_conn, decay_logit,
            mod_w1, mod_b1, mod_w2, mod_b2)

        # Step 1: Receive — sparse gather + weighted sum
        received, gathered = self._receive(msg, w_conn)    # [BS,N,D_n], [BS,N,K,D_n]

        # Step 2: Inject — add LM signal to input port neurons
        received = self._inject(received, H_aug_t)

        # Step 3: State update — shared MLP + decay blend
        h = self._state_update(received, h, identity, decay_logit,
                               st_w1, st_b1, st_w2, st_b2)

        # Step 4: Emit message — shared MLP + identity residual
        msg = self._emit_message(h, identity,
                                 mg_w1, mg_b1, mg_w2, mg_b2)

        # Step 5: Readout — output port neurons → LM dim
        mem_out[:, t] = self._readout(msg)

        # Step 6: Hebbian trace update (no grad, updates in-place)
        self._update_hebbian(gathered, msg, hebbian)

    # Save final state for next segment
    self.h = h
    self.msg = msg
    self.w_conn = w_conn
    self.identity = identity
    self.decay_logit = decay_logit
    self.hebbian_traces = hebbian

    return mem_out  # [BS, T, D]
```

#### `rewire_connections(self)`

Structural plasticity. Called every `plasticity_interval` tokens (1024 = 8
segments). Uses hebbian traces to prune weak connections and grow new ones.

The model tracks `self._tokens_since_rewire` (incremented by T each segment
in `detach_states`). When it hits `plasticity_interval`, rewiring runs and
the counter resets.

```python
def rewire_connections(self):
    if not self.config.structural_plasticity:
        return

    N, K = self.N, self.K
    n_swap = max(1, int(N * K * self.config.plasticity_pct))
    explore_frac = self.config.plasticity_exploration_frac
    device = self.conn_idx.device

    with torch.no_grad():
        conn = self.conn_idx  # [N, K]

        # === PRUNING: use hebbian traces (batch-averaged) ===
        # hebbian_traces: [BS, N, K] — average across batch
        hebb = self.hebbian_traces.mean(dim=0)     # [N, K]

        # Find globally weakest connections
        flat_hebb = hebb.reshape(-1)               # [N*K]
        _, prune_flat = flat_hebb.topk(n_swap, largest=False)
        prune_n = prune_flat // K
        prune_k = prune_flat % K

        # === REGROWTH ===
        # Build mask of existing connections: [N, N] bool
        # IMPORTANT: this must be rebuilt after EACH assignment to prevent
        # duplicate neighbors.
        exists = torch.zeros(N, N, dtype=torch.bool, device=device)
        row_idx = torch.arange(N, device=device).unsqueeze(1).expand_as(conn)
        exists[row_idx, conn] = True

        # For exploit candidates: use message magnitude as lightweight proxy
        # for affinity (avoids materializing [N, N] correlation matrix).
        # Higher magnitude neurons are more "active" and likely useful targets.
        msg_mag = self.msg.detach().float().norm(dim=-1).mean(dim=0)  # [N]

        n_exploit = n_swap - int(n_swap * explore_frac)
        new_targets = torch.empty(n_swap, dtype=torch.long, device=device)

        for i in range(n_swap):
            n = prune_n[i].item()

            if i < n_exploit:
                # Exploit: pick highest-magnitude non-connected, non-self neuron
                scores = msg_mag.clone()
                scores[conn[n]] = -float('inf')    # exclude current neighbors
                scores[n] = -float('inf')          # exclude self
                # Also exclude any targets already assigned in this pass
                # (exists is updated below after each assignment)
                scores[exists[n]] = -float('inf')
                best = scores.argmax().item()
                new_targets[i] = best
            else:
                # Explore: random non-connected, non-self
                candidates = (~exists[n]).nonzero(as_tuple=True)[0]
                candidates = candidates[candidates != n]
                if len(candidates) == 0:
                    new_targets[i] = conn[n, prune_k[i]]  # keep existing
                    continue
                idx = torch.randint(len(candidates), (1,), device=device)
                new_targets[i] = candidates[idx]

            # Update exists mask to prevent duplicates
            exists[n, new_targets[i]] = True
            # Remove old connection from exists
            exists[n, conn[n, prune_k[i]]] = False

        # Apply rewiring
        conn[prune_n, prune_k] = new_targets

        # Re-sort each modified row to maintain sorted invariant for cache locality
        modified_neurons = prune_n.unique()
        for n in modified_neurons:
            conn[n], _ = conn[n].sort()

        # Reset state for rewired connections
        self.w_conn[:, prune_n, prune_k] = 0
        self.hebbian_traces[:, prune_n, prune_k] = 0
```

**Note on the Python loop:** This runs once every 1024 tokens (not per token),
so the loop over n_swap (~10K) is acceptable. ~50ms at most.

---

## 5. Top-Level Model (`src/model/model.py`)

### Class: `Model(nn.Module)`

```python
class Model(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        config.validate()
        self.config = config
        self.lm = LM(config)
        self.memory = MemoryGraph(config)
        self._initialized = False

    def forward_chunk(
        self,
        input_ids: Tensor,          # [BS, T] int64
        target_ids: Tensor | None = None,  # [BS, T] int64, shifted
        use_memory: bool = True,
    ) -> dict:
        BS, T = input_ids.shape
        device = input_ids.device

        # Initialize memory states on first call (MUST come before resets)
        if not self._initialized:
            self.memory.initialize_states(BS, device)
            self._initialized = True

        # Build per-position reset mask for internal document boundaries.
        # If input_ids[b, t] == eot_id, the NEXT position (t+1) should
        # start with a fresh recurrent state.
        eos_positions = (input_ids == self.config.eot_id)     # [BS, T]
        internal_reset = torch.zeros_like(eos_positions)
        internal_reset[:, 1:] = eos_positions[:, :-1]         # reset at t+1

        # Reset memory + scan state for batch elements that contain any EOS.
        # This is conservative: a single EOS anywhere resets the whole memory
        # for that batch element. Internal-EOS within the chunk is handled by
        # the internal_reset mask passed to scan layers (which zero their carry
        # at those positions).
        batch_has_eos = eos_positions.any(dim=1)               # [BS]
        if batch_has_eos.any():
            self.memory.reset_states(batch_has_eos)
            self.lm.reset_carries(batch_has_eos)

        # 1. Lower scan + PCM
        reset_mask = internal_reset if internal_reset.any() else None
        H_mid, surprise, aux_loss = self.lm.forward_scan_lower(
            input_ids, reset_mask=reset_mask)

        # 2. Augment H_mid with surprise
        H_aug = self.lm.augment(H_mid, surprise)

        # 3. Memory graph
        if use_memory:
            # Detach: memory gets its own gradient path through mem_scale
            mem_out = self.memory.forward_segment(H_aug.detach())
            H_enriched = self.lm.inject_memory(H_aug, mem_out)
        else:
            H_enriched = H_aug

        # 4. Upper scan
        H = self.lm.forward_scan_upper(H_enriched, reset_mask=reset_mask)

        # 5. Output
        logits = self.lm.forward_output(H)

        result = {"logits": logits, "aux_loss": aux_loss}

        # 6. Loss (if targets provided)
        if target_ids is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
            )
            result["loss"] = loss + aux_loss

        return result

    def detach_states(self):
        """Call between chunks for TBPTT."""
        self.lm.detach_carries()
        self.memory.detach_states()

        # Structural plasticity: run every plasticity_interval tokens
        self._tokens_since_rewire = getattr(self, '_tokens_since_rewire', 0)
        self._tokens_since_rewire += self.config.T
        if (self.config.structural_plasticity and
                self._tokens_since_rewire >= self.config.plasticity_interval):
            self.memory.rewire_connections()
            self._tokens_since_rewire = 0

    def param_count(self):
        return sum(p.numel() for p in self.parameters())

    def lm_param_count(self):
        return sum(p.numel() for p in self.lm.parameters())

    def memory_param_count(self):
        return sum(p.numel() for p in self.memory.parameters())
```

---

## 6. Trainer (`src/trainer.py`)

Port from `shared-scan:src/v8/trainer.py` with these changes:
- Import paths: `src.model.model.Model`, `src.model.config.Config`
- Remove 2-pass simulation logic (now single forward_segment call)
- Keep: optimizer param groups (LM decay/no-decay, memory decay/no-decay),
  cosine LR with warmup, gradient clipping, autocast (bf16), checkpointing,
  async plotting

### Optimizer Groups

```python
# LM params
lm_decay = [p for n, p in model.lm.named_parameters()
            if p.requires_grad and p.ndim > 1 and not n.endswith('.bias')]
lm_no_decay = [p for n, p in model.lm.named_parameters()
               if p.requires_grad and (p.ndim <= 1 or n.endswith('.bias'))]

# Memory params (lower LR)
mem_decay = [p for n, p in model.memory.named_parameters()
             if p.requires_grad and p.ndim > 1 and not n.endswith('.bias')]
mem_no_decay = [p for n, p in model.memory.named_parameters()
                if p.requires_grad and (p.ndim <= 1 or n.endswith('.bias'))]

optimizer = torch.optim.AdamW([
    {"params": lm_decay,     "lr": lr, "weight_decay": 0.01},
    {"params": lm_no_decay,  "lr": lr, "weight_decay": 0.0},
    {"params": mem_decay,    "lr": lr * config.mem_lr_scale, "weight_decay": 0.001},
    {"params": mem_no_decay, "lr": lr * config.mem_lr_scale, "weight_decay": 0.0},
], betas=(0.9, 0.95), fused=True)
```

### Training Step

```python
def train_step(self, batch):
    input_ids = batch["input_ids"]          # [BS, T+1]
    tokens = input_ids[:, :-1]              # [BS, T]
    targets = input_ids[:, 1:]              # [BS, T]

    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        result = self.model.forward_chunk(
            tokens, target_ids=targets, use_memory=not self.no_memory)

    loss = result["loss"]

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
    self.optimizer.step()
    self.scheduler.step()

    # TBPTT: detach carries + structural plasticity
    self.model.detach_states()

    return {
        "loss": loss.item(),
        "aux_loss": result["aux_loss"].item(),
    }
```

---

## 7. Entry Point (`src/train.py`)

Port from `shared-scan:src/v8/train.py`. Same CLI args:
```
--bs, --lr, --steps, --warmup, --seed, --save-dir, --save-interval,
--log-interval, --tokenizer, --no-memory, --resume, --d-inner
```

The main changes:
- Import `Config`, `Model` from `src.model`
- Use `Config.tier_a()` instead of `V8Config.tier_a()`
- Create `Trainer` with the new model

---

## 8. Tests

### `tests/test_memory.py` — Memory graph unit tests

Using `Config.tier_tiny()` (N=128, D_n=8, K=16, D=64):

1. **test_shapes**: forward_segment produces correct output shape
2. **test_modulator_delta**: verify modulator produces near-zero deltas at init
3. **test_inject_readout_roundtrip**: inject a known signal, verify it appears
   in readout after one step
4. **test_state_decay**: with decay_logit=+inf (decay≈1), h should persist unchanged
5. **test_hebbian_per_token**: run 2 token steps, verify hebbian_traces differ
   between step 1 and step 2 (confirms per-token update, not stale)
6. **test_hebbian_modulator_timing**: verify that the modulator at step t+1
   sees the hebbian traces updated at step t (not segment-start traces)
7. **test_structural_plasticity**: verify connections change after rewire
8. **test_rewire_no_duplicates**: run rewire_connections 10 times, verify
   `conn_idx[n]` has K unique values for all n after each pass
9. **test_rewire_sorted**: verify conn_idx rows remain sorted after rewire
10. **test_gradient_flow**: verify gradients reach mod_w1 after backward

### `tests/test_integration.py` — End-to-end

1. **test_forward_backward**: full Model.forward_chunk + loss.backward runs without error
2. **test_no_memory_baseline**: with use_memory=False, produces valid logits
3. **test_detach_states**: verify TBPTT boundary doesn't crash
4. **test_eos_reset**: verify EOS triggers state reset and memory is zeroed
5. **test_init_before_reset**: verify first call with EOS in input doesn't crash
   (initialization happens before reset logic)
6. **test_post_eos_invariance**: after an EOS reset, the next chunk should
   produce identical output regardless of what preceded the EOS
7. **test_multi_segment_carries**: run 3 segments, verify scan carries and
   memory state persist (non-zero) across segment boundaries

---

## Parameter Count Verification

At tier_a defaults:

```
Neuromodulator:
  mod_w1:  8096 × 161 × 32 =  41,709,568
  mod_b1:  8096 × 1 × 32   =     259,072
  mod_w2:  8096 × 32 × 97  =  25,129,984
  mod_b2:  8096 × 1 × 97   =     785,312
  Subtotal:                    67,883,936  (67.9M)

State MLP:
  state_w1:  128 × 97  =  12,416
  state_b1:  128        =     128
  state_w2:  32 × 128   =   4,096
  state_b2:  32          =      32
  Subtotal:                 16,672  (16.7K)

Message MLP:
  msg_w1:  128 × 64    =   8,192
  msg_b1:  128          =     128
  msg_w2:  32 × 128     =   4,096
  msg_b2:  32            =      32
  Subtotal:                 12,448  (12.4K)

Neuron IDs:  8096 × 32  =  259,072

mem_scale:  2048         =    2,048

Memory total:              68,174,176  (~68.2M)

LM (estimated from v9):   ~52M
Grand total:               ~120M
```

---

## Kernel Launch Analysis (Per Token Step)

Every operation is a bulk tensor op over all N=8096 neurons. No Python loops
over neurons. The per-step kernel count:

| Operation | Kernels | Notes |
|-----------|---------|-------|
| _modulate: cat | 1 | Concatenate 4 tensors |
| _modulate: permute | 0 | View only |
| _modulate: bmm + tanh | 2 | Layer 1 |
| _modulate: bmm | 1 | Layer 2 |
| _modulate: add bias×2 | 2 | May fuse with bmm |
| _modulate: slice + add×3 | 3 | Unpack deltas, apply |
| _receive: gather | 1 | msg[:, conn_idx] |
| _receive: sigmoid + mul + sum | 3 | Weighted reduction |
| _inject: reshape+expand+pad+add | 2 | Zero-pad + single add |
| _state_update: cat | 1 | |
| _state_update: linear×2 + tanh×2 | 4 | May partially fuse under compile |
| _state_update: sigmoid + blend | 3 | decay * h + (1-decay) * cand |
| _emit_message: cat | 1 | |
| _emit_message: linear×2 + tanh×2 | 4 | |
| _emit_message: add identity | 1 | |
| _readout: slice + reshape + sum + mul | 2 | Views + reduction |
| _update_hebbian: mul + sum + mul_ + add_ | 4 | All in-place or simple |
| **Total** | **~35** | |

At 128 steps: ~4480 kernel launches. At ~5μs overhead each: **~22ms** of pure
launch overhead. This is ~6% of the estimated 384ms per segment — acceptable.

**Key optimizations already applied:**
- All weights pre-cast before the loop (0 casts per step vs. 12 without)
- All shared MLPs use `F.linear` on [BS*N, D] (single GEMM, not N small ones)
- Modulator uses `torch.bmm` (single batched GEMM, not N separate ones)
- No Python loops over neurons anywhere
- Inject uses zero-pad + add instead of clone + in-place slice

**Future Triton fusions (Phase 2):**
- Fuse gather + sigmoid + weighted_sum → eliminates 1.58 GB intermediate
- Fuse linear + tanh + linear + tanh → eliminates MLP hidden activation
- Fuse sigmoid + decay blend → single kernel for state blending
- Expected reduction: ~35 → ~10 kernels per step

## Autocast / Dtype Strategy

- **All nn.Parameters stored in f32.** This is critical for the neuromodulator —
  tiny gradients round to zero in bf16.
- **Forward computation in bf16** via `torch.amp.autocast('cuda', dtype=torch.bfloat16)`.
  Inside autocast, `F.linear` and `torch.bmm` automatically cast inputs/weights
  to bf16 for compute and return bf16 outputs.
- **Explicit casts** in memory.py: we manually cast weights to `h.dtype` (bf16)
  before `torch.bmm` / `F.linear` because some ops (like advanced indexing) don't
  trigger autocast. This ensures consistent dtype throughout.
- **Modulator weights are cast once** before the token loop (`mod_w1.to(dt)` etc.),
  NOT per step. This avoids 68M × 128 = 8.7B parameter copies per segment.
- **Hebbian traces**: updated with `torch.no_grad()` and in-place ops
  (`mul_`, `add_`), so they bypass autocast. Stored in bf16 (same as runtime state).

---

## Gradient Flow Diagram

```
CE Loss
  │
  ├─► logits ◄─ lm_head ◄─ ln_final ◄─ proj_down ◄─ upper scan layers
  │                                                        ▲
  │                                              H_aug + mem_scale * mem_out
  │                                                 │              │
  │                                          ┌──────┘              │
  │                                          │                     │
  │                                   split_mlp(H_mid, surp)    mem_scale (grad ✓)
  │                                     │          │               │
  │                                  H_mid      surprise         mem_out
  │                                   (grad ✓)  (grad ✓)    ┌─────┘
  │                                     │          │         │
  │                                lower scan    PCM     memory graph
  │                                   │                      │
  │                                 embedding          (grad through mod_w1,
  │                                                     mod_w2, state_w*, msg_w*,
  │                                                     neuron_id, mem_scale)
  │
  ├─► pcm_loss (aux)
  │
  └─► Total loss = CE + pcm_weight * pcm_loss
```

The memory graph receives `H_aug.detach()` as input, so the **only** gradient
path from the CE loss to memory parameters is:
```
loss → logits → upper_scan → H_enriched = H_aug + mem_scale * mem_out
                                                       │
                                              grad flows through mem_scale
                                              and through mem_out to all
                                              memory graph parameters
```

This means the memory graph must produce useful `mem_out` to reduce CE loss.
The `mem_scale` learnable parameter controls the magnitude of this signal.

---

## Training Signal Analysis & Potential Issues

### 1. Gradient chain depth through modulator

The modulator runs every token, and its outputs (w_conn, decay_logit, identity)
are carried forward and fed back as input at the next step. Over T=128 tokens,
this creates a 128-step recurrence through the modulator:

```
step 0: mod(identity_0, hebb, w_0, dec_0) → w_1, dec_1, id_1
step 1: mod(id_1, hebb, w_1, dec_1)       → w_2, dec_2, id_2
...
step 127: mod(id_127, ...) → w_128, dec_128, id_128
```

Each delta is small (mod_w2 init × 0.01), but 128 chained additions could
still cause drift. Gradient through this chain may vanish or explode.

**Mitigations built in:**
- `mod_w2` initialized × 0.01 → initial deltas near zero
- `tanh` activation bounds hidden representations
- Gradient clipping (max_norm=1.0) at the optimizer level

**Watch for during training:**
- Monitor `identity.norm()` over steps — should stay bounded
- Monitor `w_conn` range — sigmoid keeps effective weights in [0,1] but logits could grow
- If gradient issues appear: consider running modulator every M steps (M=4 or M=8)
  instead of every token, reducing chain depth to 16-32

### 2. Memory graph gradient path is narrow

The ONLY gradient signal to memory parameters comes through:
```
CE loss → mem_scale * mem_out → readout → msg of output ports → msg_mlp → h → state_mlp → received → w_conn (from modulator)
```

This is a long chain. If any link attenuates too much, memory params get no
useful gradient. Historical failure modes from v8/v9:
- **1/N readout scaling killed gradients** → solved by 1/sqrt(N) + mem_scale
- **MLP backward attenuated 20×** → solved by using learnable scale instead of MLP for combination
- **bf16 params with tiny gradients** → solved by keeping params in f32

The current design addresses all three, but the chain is still ~8 ops deep
from loss to modulator weights. The key safeguard is `mem_scale` init at
sqrt(alpha) ≈ 2.0 — this ensures gradients arrive at readout with reasonable
magnitude from the start.

### 3. Port neuron gradient concentration

Only 512 out of 8096 neurons (6.3%) directly interface with the LM. Gradient
signal enters through output port neurons and must propagate through the graph
to reach internal neurons. With one step per token, a message from an internal
neuron takes at least 2 hops to reach an output port (internal → neighbor → port).
Many internal neurons are further away.

This means:
- **Output port neurons** and their direct presynaptic neighbors get strong gradients
- **Distant internal neurons** get weak/no gradient signal initially
- Structural plasticity partially compensates: it rewires based on co-activation,
  which can promote connections that route gradient-carrying signals

**This is by design** — the memory graph should develop spatial organization
where information flows from input ports through internal neurons to output ports.
But early training may show the internal neurons are "dead" until the graph
topology develops useful pathways.

### 4. What each component learns

| Component | Learns to... | Signal source |
|-----------|-------------|---------------|
| State MLP (shared) | Update neuron state given messages + context | CE loss via output ports |
| Message MLP (shared) | Encode useful information in messages | CE loss via output ports |
| Neuromodulator (per-neuron) | Adjust connectivity and decay for each neuron | CE loss, very indirect |
| Neuron IDs (learned) | Provide useful initial identity vectors | CE loss + modulator use |
| mem_scale | Balance memory readout magnitude | CE loss, direct |
| PCM | Predict state transitions | PCM aux loss (direct) |
| split_mlp | Combine surprise with H_mid | CE loss, direct |

The neuromodulator gets the weakest signal because it's furthest from the loss.
This is why per-neuron weights matter — shared weights would need even stronger
signal to differentiate neuron behavior.

### 5. Activation memory for backward

The token loop builds a computation graph across 128 steps. Each step
materializes several intermediate tensors that must be kept for backward:

Per step:
- `gathered`: [BS, N, K, D_n] = 1.58 GB at BS=48 (needed for w_conn grad)
- State MLP hidden: [BS*N, 128] = 190 MB
- Msg MLP hidden: [BS*N, 128] = 190 MB
- Modulator hidden: [N, BS, 32] = 50 MB
- Various intermediates: ~200 MB

Total per step: ~2.2 GB. Over 128 steps: **~280 GB** — far exceeds 24 GB VRAM.

**Mitigation: gradient checkpointing on the token loop.** This is REQUIRED,
not optional. Checkpoint every M steps (e.g., M=8), recomputing 8 steps
during backward. This trades 128/8 = 16× recomputation for 16× less stored
activations: ~2.2 GB × 8 = 17.6 GB.

Implementation: split the T=128 loop into chunks of M steps. Wrap each chunk
in `torch.utils.checkpoint.checkpoint`:

```python
def forward_segment(self, H_aug):
    BS, T, D = H_aug.shape
    CKPT_EVERY = 8  # checkpoint every 8 steps

    # ... setup, pre-cast weights ...

    mem_out = torch.empty(BS, T, D, device=device, dtype=H_aug.dtype)

    def run_steps(t_start, t_end, h, msg, w_conn, identity, decay_logit, hebbian):
        for t in range(t_start, t_end):
            # ... full neuron step (steps 0-6) ...
        return h, msg, w_conn, identity, decay_logit, hebbian, mem_out_chunk

    for chunk_start in range(0, T, CKPT_EVERY):
        chunk_end = min(chunk_start + CKPT_EVERY, T)
        if self.training:
            h, msg, w_conn, identity, decay_logit, hebbian, chunk_out = \
                torch.utils.checkpoint.checkpoint(
                    run_steps, chunk_start, chunk_end,
                    h, msg, w_conn, identity, decay_logit, hebbian,
                    use_reentrant=False)
        else:
            h, msg, w_conn, identity, decay_logit, hebbian, chunk_out = \
                run_steps(chunk_start, chunk_end,
                          h, msg, w_conn, identity, decay_logit, hebbian)
        mem_out[:, chunk_start:chunk_end] = chunk_out
```

For initial testing at BS=4-8, checkpointing may not be needed and can be
toggled off for easier debugging.
