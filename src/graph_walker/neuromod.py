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
        adj_bias: torch.Tensor | None,  # [U, U] OR [n_heads, U, U] fp32
    ) -> torch.Tensor:
        """`adj_bias` may be either:
          - `[U, U]`:           same bias across all attention heads (legacy /
                                topology-only path).
          - `[n_heads, U, U]`:  per-head bias (option C — per-edge stats project
                                to per-head scalars added to attn scores).
        """
        U, D = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        k = self.k_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        v = self.v_proj(h).view(U, self.n_heads, self.d_head).transpose(0, 1)
        # scores: [n_heads, U, U]
        scores = torch.einsum("hud,hvd->huv", q, k) * self.scale
        if adj_bias is not None:
            if adj_bias.ndim == 2:
                scores = scores + adj_bias.unsqueeze(0).to(scores.dtype)
            elif adj_bias.ndim == 3:
                if adj_bias.shape[0] != self.n_heads:
                    raise ValueError(
                        f"adj_bias[0]={adj_bias.shape[0]} must equal n_heads="
                        f"{self.n_heads} for per-head bias."
                    )
                scores = scores + adj_bias.to(scores.dtype)
            else:
                raise ValueError(
                    f"adj_bias must be 2D or 3D; got shape "
                    f"{tuple(adj_bias.shape)}"
                )
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
        D_per_edge_extra: int = 0,    # Hebbian-flavored per-edge inputs
                                       # (co_visit, E_bias_old) when neuromod_only;
                                       # 0 in legacy hebbian_plus_neuromod mode.
    ) -> None:
        super().__init__()
        self.D_mod = D_mod
        self.n_heads = n_heads
        self.E_bias_max = E_bias_max
        self.D_per_edge_extra = D_per_edge_extra
        self.feature_proj = nn.Linear(D_feat, D_mod)
        nn.init.normal_(self.feature_proj.weight, mean=0.0, std=0.014)

        self.layers = nn.ModuleList([
            _GraphAttnLayer(D_mod, n_heads=n_heads) for _ in range(n_layers)
        ])

        # Per-edge target MLP. In legacy mode, input is just cat(x_src, x_dst).
        # In neuromod_only mode, append per-edge extras (co_visit, E_bias_old)
        # so the MLP can express e.g. "high co_visit AND surprise → push target
        # up" rules without going through the column-level features.
        edge_in_dim = 2 * D_mod + D_per_edge_extra
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, edge_hidden),
            nn.GELU(),
            nn.Linear(edge_hidden, 1),
        )
        nn.init.normal_(self.edge_mlp[0].weight, mean=0.0, std=0.014)
        nn.init.zeros_(self.edge_mlp[0].bias)
        nn.init.zeros_(self.edge_mlp[-1].weight)
        nn.init.zeros_(self.edge_mlp[-1].bias)

        # Per-edge → per-head attention bias projection (option C). Lets the
        # graph transformer's attention pattern be shaped by activity stats,
        # not just topology. One scalar per (edge, head) — a head can
        # specialize on co_visit, another on E_bias, etc.
        # Zero-init so day-0 attention is identical to the topology-only
        # adjacency bias; training opens up per-edge influence as the
        # neuromod learns which stats matter for which heads.
        if D_per_edge_extra > 0:
            self.edge_bias_proj = nn.Linear(D_per_edge_extra, n_heads)
            nn.init.zeros_(self.edge_bias_proj.weight)
            nn.init.zeros_(self.edge_bias_proj.bias)
        else:
            self.edge_bias_proj = None

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
        features: torch.Tensor,           # [U, D_feat] — detached per-touched-col feats
        adj_bias: torch.Tensor,           # [U, U] fp32 — topology bias for attention
        edge_src_local: torch.Tensor,     # [E] int64 — into [0, U)
        edge_dst_local: torch.Tensor,     # [E] int64 — into [0, U)
        per_edge_extras: torch.Tensor | None = None,
                                          # [E, D_per_edge_extra] — co_visit, E_bias_old etc.
                                          # Required in neuromod_only mode; None in legacy.
    ) -> torch.Tensor:
        """Return per-touched-edge TARGET values, shape [E].

        Values are in [-E_bias_max, +E_bias_max] (tanh-clamped). The caller
        converts these to deltas via `γ · (target - E_bias_base)` and
        scatters into the global [N*K] layout.
        """
        x = self.feature_proj(features)
        U = x.shape[0]

        # Build the attention bias the layers see. In legacy / topology-only
        # mode it's just `adj_bias [U, U]`. In neuromod_only mode we project
        # per-edge extras to per-head scalars and scatter them into a
        # [n_heads, U, U] enriched bias so attention is shaped by activity
        # stats, not just topology (option C).
        if self.D_per_edge_extra > 0 and per_edge_extras is not None:
            assert per_edge_extras.shape[-1] == self.D_per_edge_extra, (
                f"per_edge_extras last-dim {per_edge_extras.shape[-1]} != "
                f"D_per_edge_extra {self.D_per_edge_extra}."
            )
            # [E, n_heads]: per-edge per-head attention bias contribution.
            edge_bias_per_head = self.edge_bias_proj(per_edge_extras)
            # Scatter into [n_heads, U, U] dense layout. Linear flat index per
            # (head, src, dst): h*(U*U) + src*U + dst.
            n_heads = self.n_heads
            edge_bias_dense = torch.zeros(
                n_heads * U * U,
                device=adj_bias.device, dtype=adj_bias.dtype,
            )
            head_offsets = (
                torch.arange(n_heads, device=adj_bias.device).view(-1, 1)
                * (U * U)
            )                                                # [n_heads, 1]
            edge_offsets = (
                edge_src_local.view(1, -1) * U + edge_dst_local.view(1, -1)
            )                                                # [1, E]
            flat_idx = (head_offsets + edge_offsets).flatten()  # [n_heads * E]
            flat_values = (
                edge_bias_per_head.t().contiguous().to(adj_bias.dtype).flatten()
            )                                                # [n_heads * E]
            edge_bias_dense.scatter_add_(0, flat_idx, flat_values)
            edge_bias_dense = edge_bias_dense.view(n_heads, U, U)
            adj_bias_for_layers = (
                adj_bias.unsqueeze(0).expand(n_heads, U, U) + edge_bias_dense
            )                                                # [n_heads, U, U]
        else:
            adj_bias_for_layers = adj_bias                   # [U, U]

        for layer in self.layers:
            x = layer(x, adj_bias_for_layers)
        src_feats = x[edge_src_local]                       # [E, D_mod]
        dst_feats = x[edge_dst_local]                       # [E, D_mod]
        if self.D_per_edge_extra > 0:
            assert per_edge_extras is not None, (
                f"D_per_edge_extra={self.D_per_edge_extra} but no extras passed; "
                "caller must supply per_edge_extras in neuromod_only mode."
            )
            edge_in = torch.cat(
                [src_feats, dst_feats, per_edge_extras], dim=-1,
            )
        else:
            edge_in = torch.cat([src_feats, dst_feats], dim=-1)
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
