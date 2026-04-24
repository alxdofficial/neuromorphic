"""Graph-transformer neuromodulator.

Fires at the start of each plasticity window. Observes a snapshot of
per-touched-column state from the just-closed window and produces a
delta to `E_bias_flat` for edges whose endpoints are both touched.
The delta is grad-carrying during the new window, so loss gradients
from that window flow back through the E_bias lookup in routing into
the neuromod's parameters.

At segment close the delta is detached and folded into the persistent
`E_bias_flat` buffer (same store the legacy Hebbian rule updates).

Design choices (see docs/graph_walker.md §Neuromod):
- Sparse observation: only columns with visit_count > 0 this window.
- Sparse write: only edges whose source AND destination are in the
  touched set. Unvisited parts of the graph stay unchanged.
- Low-rank edge decoder: `ΔE_bias[i→j] = (src_proj(x_i) · dst_proj(x_j)) / √R`.
  Avoids materialising the [U, U] edge matrix.
- Topology-biased attention: the graph's adjacency among touched
  columns is added to attention scores, so the transformer reasons
  about nearby columns the graph already considers neighbours.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.graph_walker.readout import _FallbackRMSNorm


def _rmsnorm(dim: int) -> nn.Module:
    return _FallbackRMSNorm(dim)


class _GraphAttnLayer(nn.Module):
    """Pre-norm multi-head self-attention with additive adjacency bias + FFN."""

    def __init__(self, D: int, n_heads: int, ffn_mult: int = 4) -> None:
        super().__init__()
        assert D % n_heads == 0, f"D={D} must be divisible by n_heads={n_heads}"
        self.n_heads = n_heads
        self.d_head = D // n_heads
        self.scale = 1.0 / (self.d_head ** 0.5)
        self.norm1 = _rmsnorm(D)
        self.q_proj = nn.Linear(D, D, bias=False)
        self.k_proj = nn.Linear(D, D, bias=False)
        self.v_proj = nn.Linear(D, D, bias=False)
        self.o_proj = nn.Linear(D, D, bias=False)
        self.norm2 = _rmsnorm(D)
        self.ff_up = nn.Linear(D, ffn_mult * D)
        self.ff_down = nn.Linear(ffn_mult * D, D)
        # Depth-scaled init on residual outputs
        nn.init.normal_(self.o_proj.weight, mean=0.0, std=(2.0 / D) ** 0.5)
        nn.init.normal_(self.ff_down.weight, mean=0.0, std=(2.0 / (ffn_mult * D)) ** 0.5)

    def forward(
        self,
        x: torch.Tensor,                # [U, D]
        adj_bias: torch.Tensor | None,  # [U, U] fp32 — added to attn scores
    ) -> torch.Tensor:
        U, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        # scores: [n_heads, U, U]
        scores = torch.einsum("hud,hvd->huv", q, k) * self.scale
        if adj_bias is not None:
            scores = scores + adj_bias.unsqueeze(0).to(scores.dtype)
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("huv,hvd->hud", attn, v)
        out = out.transpose(0, 1).reshape(U, D)
        x = x + self.o_proj(out)
        # FFN
        h2 = self.norm2(x)
        x = x + self.ff_down(F.gelu(self.ff_up(h2)))
        return x


class NeuromodGraphTransformer(nn.Module):
    """Start-of-window neuromod producing grad-carrying per-edge targets.

    Emits a direct target value per touched edge (tanh-clamped to
    `E_bias_max`), not a delta. The caller converts target → delta via an
    EMA blend gated by `self.gamma` (a learnable scalar blend rate):

        _active_delta_nm[edge] = γ · (target[edge] - E_bias_base[edge])

    which, when added to `E_bias_base`, gives the classic EMA blend
    `(1 - γ) · E_bias_base + γ · target`. Same numerical behavior as the
    pattern used on `main` (attention_modulator), which zero-inits the
    head and uses a learnable per-row EMA gate to keep updates bounded.

    Stability mechanisms:
    - **tanh clamp** on the per-edge prediction: target magnitude bounded
      by `E_bias_max` regardless of the edge MLP's raw output.
    - **EMA blend** via σ(blend_logit): at init blend_logit = -5, so
      γ ≈ 0.007 → day-0 active delta ≈ 0 even if target is non-zero.
      Training opens γ upward as the neuromod learns to commit to its
      predictions.
    - **Zero-init edge_mlp output layer**: raw=0 at init → target=0.

    Args:
      D_feat: dimensionality of per-column input features
              (typically D_s + D_id + 1 for state + id + visit_count)
      D_mod: internal width of the transformer
      n_layers: number of attention+FFN layers
      n_heads: attention heads per layer
      edge_hidden: hidden dim of the per-edge target MLP
      E_bias_max: tanh-clamp scale (match the E_bias clamp in the plasticity
                  step so target and persistent buffer share a range)
    """

    def __init__(
        self,
        D_feat: int,
        D_mod: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        edge_hidden: int = 64,
        E_bias_max: float = 4.0,
    ) -> None:
        super().__init__()
        self.D_mod = D_mod
        self.E_bias_max = E_bias_max
        self.feature_proj = nn.Linear(D_feat, D_mod)
        nn.init.normal_(self.feature_proj.weight, mean=0.0, std=0.014)

        self.layers = nn.ModuleList([
            _GraphAttnLayer(D_mod, n_heads=n_heads) for _ in range(n_layers)
        ])

        # Per-edge target MLP: cat(x_src, x_dst) → scalar pre-tanh logit.
        # Full-rank (not low-rank outer product) so the predictor can express
        # non-bilinear interactions between the source and destination
        # features. Zero-init final layer so raw=0 → target=0 at init.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * D_mod, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, 1),
        )
        nn.init.normal_(self.edge_mlp[0].weight, mean=0.0, std=0.014)
        nn.init.zeros_(self.edge_mlp[0].bias)
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

        # Learnable blend rate γ = σ(blend_logit). Start near-zero so the
        # neuromod's live influence during the first few windows is tiny,
        # and training can open it up when useful.
        self.blend_logit = nn.Parameter(torch.tensor(-5.0))

    @property
    def gamma(self) -> torch.Tensor:
        """Scalar blend rate in (0, 1)."""
        return torch.sigmoid(self.blend_logit)

    def forward(
        self,
        features: torch.Tensor,     # [U, D_feat] — detached per-touched-col feats
        adj_bias: torch.Tensor,     # [U, U] fp32 — topology bias for attention
        edge_src_local: torch.Tensor,  # [E] int64 — into [0, U)
        edge_dst_local: torch.Tensor,  # [E] int64 — into [0, U)
    ) -> torch.Tensor:
        """Return per-touched-edge TARGET values, shape [E].

        Values are in [-E_bias_max, +E_bias_max] (tanh-clamped). The caller
        converts these to deltas via `γ · (target - E_bias_base)` and
        scatters into the global [N*K] layout.
        """
        x = self.feature_proj(features)
        for layer in self.layers:
            x = layer(x, adj_bias)
        src_feats = x[edge_src_local]                       # [E, D_mod]
        dst_feats = x[edge_dst_local]                       # [E, D_mod]
        edge_in = torch.cat([src_feats, dst_feats], dim=-1)  # [E, 2·D_mod]
        raw = self.edge_mlp(edge_in).squeeze(-1)            # [E]
        return self.E_bias_max * torch.tanh(raw)


@torch.no_grad()
def enumerate_touched_edges(
    touched_ids: torch.Tensor,    # [U] int64, sorted — global column indices
    out_nbrs: torch.Tensor,       # [N, K] int64
    K: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Find all edges (i→j) where both i ∈ touched_ids AND j ∈ touched_ids.

    Returns:
      edge_src_local: [E] int64 — positions in touched_ids for source cols
      edge_dst_local: [E] int64 — positions in touched_ids for dest cols
      edge_flat: [E] int64 — global flat edge indices (src_global * K + k)
    """
    U = touched_ids.shape[0]
    if U == 0:
        empty = torch.empty(0, dtype=torch.int64, device=touched_ids.device)
        return empty, empty, empty

    # For each touched source, get its K outgoing neighbors' global ids.
    nbrs_global = out_nbrs[touched_ids]                  # [U, K]

    # Which (u, k) pairs land on a touched destination?
    # torch.isin: element-of test. Slow-ish (O(U * K * log U)) but cheap at our sizes.
    is_dst_touched = torch.isin(nbrs_global, touched_ids)  # [U, K]

    # Indices where is_dst_touched is True
    valid_u, valid_k = is_dst_touched.nonzero(as_tuple=True)  # [E], [E]

    edge_src_local = valid_u                              # [E], 0..U
    # Map each valid neighbor's global id back to its position in touched_ids.
    # touched_ids is sorted, so searchsorted gives the position.
    valid_dst_global = nbrs_global[valid_u, valid_k]      # [E]
    edge_dst_local = torch.searchsorted(touched_ids, valid_dst_global)

    # Global flat edge index for writing into E_bias_flat
    edge_flat = touched_ids[valid_u] * K + valid_k        # [E]

    return edge_src_local, edge_dst_local, edge_flat


@torch.no_grad()
def build_adjacency_bias(
    touched_ids: torch.Tensor,    # [U] int64 sorted
    out_nbrs: torch.Tensor,       # [N, K] int64
) -> torch.Tensor:
    """Build a [U, U] fp32 adjacency bias for attention.

    Entry (i, j) = 1.0 iff there's an edge from touched_ids[i] to touched_ids[j]
    in the existing topology, else 0.0.

    Used as an additive bias in attention scores: columns already connected
    in the graph get attention preference.
    """
    U = touched_ids.shape[0]
    device = touched_ids.device
    if U == 0:
        return torch.zeros(0, 0, device=device, dtype=torch.float32)
    nbrs_global = out_nbrs[touched_ids]                   # [U, K]
    is_dst_touched = torch.isin(nbrs_global, touched_ids) # [U, K]
    valid_u, valid_k = is_dst_touched.nonzero(as_tuple=True)
    valid_dst_global = nbrs_global[valid_u, valid_k]
    valid_dst_local = torch.searchsorted(touched_ids, valid_dst_global)
    bias = torch.zeros(U, U, device=device, dtype=torch.float32)
    bias[valid_u, valid_dst_local] = 1.0
    return bias
