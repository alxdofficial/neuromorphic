"""Post-synaptic-local Hebbian + Oja decay for E_bias.

Each column B's neuromod decides how to update B's *incoming* edges
E_bias[*, B]. The symmetric edge E_bias[B, *] is updated by the other
columns' neuromods. The two directed edges between A and B evolve
through completely independent decisions.

The update is always computed in fp32 under autocast(enabled=False).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_activities(
    m_out: torch.Tensor,           # [B, N, D_s]
    incoming: torch.Tensor,        # [B, N, D_s]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-column activity proxies pre[c] and post[c], shape [B, N] each.

    Uses normed magnitude; fp32 under autocast-disabled by the caller.
    """
    D_s = m_out.shape[-1]
    # / sqrt(D_s) so expected magnitude is ~1 regardless of D_s
    scale = 1.0 / (D_s ** 0.5)
    pre = m_out.float().norm(dim=-1) * scale             # [B, N]
    post = incoming.float().norm(dim=-1) * scale         # [B, N]
    return pre, post


def hebbian_update_incoming(
    E_bias_flat: torch.Tensor,     # [N*K]  fp32
    w_out_flat: torch.Tensor,       # [B, N*K] sigmoid-gated edge weights
    edge_src: torch.Tensor,         # [N*K] int64 — source column of each edge
    edge_dst: torch.Tensor,         # [N*K] int64 — destination column of each edge
    pre: torch.Tensor,              # [B, N]  sender activity
    post: torch.Tensor,             # [B, N]  receiver activity
    eta_global: torch.Tensor,       # scalar fp32
    eta: torch.Tensor,              # [N]  per-destination rate (post-synaptic gate)
    beta: torch.Tensor,             # [N]  per-destination LTP/LTD bias (Oja decay strength)
    E_max: float = 4.0,
) -> torch.Tensor:
    """Return the new E_bias_flat after one Hebbian+Oja step (averaged over batch).

    All math is forced fp32 by the caller's autocast(enabled=False) context.

    Post-synaptic-local math:
        coact[edge] = pre[src(edge)] · post[dst(edge)] · w_out[edge]
        decay[edge] = beta[dst(edge)] · E_bias[edge]
        ΔE_bias[edge] = eta_global · eta[dst(edge)] · (coact - decay)

    Batch mean over B samples before writing (plastic state is not
    batch-indexed; plastic state reflects the cohort's shared substrate).
    """
    # Gather pre at source; post at destination.
    # w_out_flat is [B, N*K]; gather indices broadcast over B automatically.
    pre_per_edge = pre[:, edge_src]                       # [B, N*K]
    post_per_edge = post[:, edge_dst]                     # [B, N*K]

    coact = pre_per_edge * post_per_edge * w_out_flat      # [B, N*K]
    # coact averaged over batch
    coact_mean = coact.mean(dim=0)                         # [N*K]

    # Post-synaptic factors, gathered at destination
    eta_dst = eta[edge_dst]                                # [N*K]
    beta_dst = beta[edge_dst]                              # [N*K]

    decay = beta_dst * E_bias_flat                         # [N*K]

    delta = eta_global * eta_dst * (coact_mean - decay)    # [N*K]
    E_new = (E_bias_flat + delta).clamp(-E_max, E_max)
    return E_new


def update_input_ctx_ema(
    input_ctx_ema: torch.Tensor,     # [B, D_ctx]
    h_input: torch.Tensor,            # [B, D_ctx]
    alpha: float,
) -> torch.Tensor:
    """Running EMA of the injected input vector."""
    return (1.0 - alpha) * input_ctx_ema + alpha * h_input


def update_tile_stats(
    mag_ema: torch.Tensor,           # [B, num_tiles]
    var_ema: torch.Tensor,           # [B, num_tiles]
    s: torch.Tensor,                  # [B, N, D_s]
    tile_ids: torch.Tensor,           # [N]   int64
    num_tiles: int,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tile EMAs of column-state magnitude and variance."""
    # Compute per-column mag = ||s||_2 / sqrt(D_s)
    D_s = s.shape[-1]
    mag_col = s.float().norm(dim=-1) / (D_s ** 0.5)       # [B, N]
    var_col = s.float().var(dim=-1)                        # [B, N]

    # Aggregate per tile by scatter-mean
    B = s.shape[0]
    # Use index_add to sum, then divide by counts
    counts = torch.bincount(tile_ids, minlength=num_tiles).float()  # [num_tiles]
    sum_mag = torch.zeros(B, num_tiles, device=s.device, dtype=torch.float32)
    sum_var = torch.zeros(B, num_tiles, device=s.device, dtype=torch.float32)
    sum_mag.index_add_(1, tile_ids, mag_col)
    sum_var.index_add_(1, tile_ids, var_col)
    mean_mag = sum_mag / counts.clamp(min=1)
    mean_var = sum_var / counts.clamp(min=1)

    new_mag = (1.0 - alpha) * mag_ema + alpha * mean_mag
    new_var = (1.0 - alpha) * var_ema + alpha * mean_var
    return new_mag, new_var
