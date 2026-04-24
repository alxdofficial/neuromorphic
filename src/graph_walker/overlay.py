"""Active-row overlay for sparse-update column state.

Goal: keep `[B, N, D_s]` column state WITHOUT materializing the dense tensor
inside the autograd graph. Only the ~O(H · T_block) rows actually touched in
a TBPTT block live as differentiable tensors.

Layout:
    s_base:         [B, N, D_s] detached  (long-term persistent state)
    active_flat_idx: [U] int64 sorted      (flat indices b·N + c that were
                                            touched this block)
    active_val:     [U, D_s] differentiable(current value at those rows)

Reads (`overlay_gather`) dispatch per-row: if the queried flat index is in
`active_flat_idx`, return `active_val[pos]`; otherwise `s_base[idx]`.

Writes (`overlay_lif_update`) perform the LIF blend:
    s_new[c] = α(c) · s_old[c] + (1-α(c)) · tanh( Σ msgs_to_c )
then merge into the overlay. Existing rows are replaced via out-of-place
scatter; new rows are appended and re-sorted. Each op creates a small new
tensor so the autograd chain is intact without ever materializing a
[B·N, D_s] intermediate.

At block boundary (`commit_overlay_to_base`), `active_val.detach()` is
scattered into `s_base` and the overlay resets to empty.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def empty_overlay(
    D_s: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an empty (active_flat_idx, active_val) pair."""
    idx = torch.empty(0, dtype=torch.int64, device=device)
    val = torch.empty(0, D_s, dtype=dtype, device=device)
    return idx, val


def overlay_gather(
    active_flat_idx: torch.Tensor,   # [U] int64 sorted
    active_val: torch.Tensor,        # [U, D_s] differentiable
    s_base: torch.Tensor,            # [B, N, D_s] detached
    flat_indices: torch.Tensor,      # [Q] int64 — b·N + c per query
) -> torch.Tensor:
    """Per-row read: overlay if present, else s_base. Returns [Q, D_s]."""
    D_s = s_base.shape[-1]
    s_base_flat = s_base.view(-1, D_s)

    # Fast path: empty overlay → straight base read.
    U = active_flat_idx.shape[0]
    if U == 0:
        return s_base_flat.index_select(0, flat_indices)

    pos = torch.searchsorted(active_flat_idx, flat_indices)        # [Q]
    pos_clamp = pos.clamp(max=U - 1)
    # is_in_active: the queried idx equals the one at its searchsorted pos.
    # Also require pos < U (searchsorted can return U for idx > max).
    is_in_active = (pos < U) & (
        active_flat_idx.index_select(0, pos_clamp) == flat_indices
    )
    overlay_vals = active_val.index_select(0, pos_clamp)           # [Q, D_s]
    base_vals = s_base_flat.index_select(0, flat_indices)          # [Q, D_s]
    return torch.where(is_in_active.unsqueeze(-1), overlay_vals, base_vals)


def overlay_lif_update(
    active_flat_idx: torch.Tensor,   # [U] int64 sorted
    active_val: torch.Tensor,        # [U, D_s] differentiable
    s_base: torch.Tensor,            # [B, N, D_s] detached
    messages: torch.Tensor,          # [M, D_s] — each message, already gated
    dests: torch.Tensor,             # [M] int64 — flat destination indices
    alpha: torch.Tensor,             # [N] float32 — per-column decay
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """LIF update with overlay semantics.

    Returns the updated (active_flat_idx, active_val).

    Semantics equivalent to `SparseLIFUpdate` but operating on the overlay:
      1. Sum messages per unique destination.
      2. Read s_old for each unique dest from overlay or base.
      3. s_new = α(c) · s_old + (1-α(c)) · tanh(incoming).
      4. For each unique dest:
         - if already in overlay → replace its value (out-of-place scatter).
         - if not → append, then re-sort the overlay.
    """
    D_s = messages.shape[-1]
    device = messages.device
    fast_dtype = active_val.dtype

    if messages.numel() == 0:
        return active_flat_idx, active_val

    # 1. Per-unique-destination aggregation.
    unique_dests, inverse = torch.unique(dests, return_inverse=True)
    U_d = unique_dests.shape[0]
    incoming_fp32 = torch.zeros(U_d, D_s, device=device, dtype=torch.float32)
    incoming_fp32 = incoming_fp32.index_add(0, inverse, messages.float())

    # 2. Gather s_old for each unique dest (overlay-aware).
    s_old = overlay_gather(
        active_flat_idx, active_val, s_base, unique_dests,
    )                                                              # [U_d, D_s]

    # 3. LIF blend.
    alpha_u = alpha.index_select(0, unique_dests % N).float()      # [U_d]
    alpha_col = alpha_u.unsqueeze(-1)
    s_new = (
        alpha_col * s_old.float()
        + (1.0 - alpha_col) * torch.tanh(incoming_fp32)
    ).to(fast_dtype)                                               # [U_d, D_s]

    # 4. Merge into overlay.
    U = active_flat_idx.shape[0]
    if U == 0:
        # Fresh overlay — just sort unique_dests and adopt.
        sort_perm = unique_dests.argsort()
        return unique_dests.index_select(0, sort_perm), s_new.index_select(0, sort_perm)

    pos = torch.searchsorted(active_flat_idx, unique_dests)        # [U_d]
    pos_clamp = pos.clamp(max=U - 1)
    is_in_active = (pos < U) & (
        active_flat_idx.index_select(0, pos_clamp) == unique_dests
    )

    new_active_idx = active_flat_idx
    new_active_val = active_val

    # 4a. Overwrite existing overlay rows for dests already present.
    if bool(is_in_active.any()):
        replace_pos = pos_clamp.masked_select(is_in_active)
        replace_val = s_new.index_select(0, is_in_active.nonzero(as_tuple=False).squeeze(-1))
        # unique_dests is unique, so replace_pos contains no duplicates —
        # scatter is deterministic.
        new_active_val = torch.scatter(
            new_active_val,
            0,
            replace_pos.unsqueeze(-1).expand(-1, D_s),
            replace_val,
        )

    # 4b. Append new dests not already in overlay; re-sort.
    not_in_active = ~is_in_active
    if bool(not_in_active.any()):
        new_idx = unique_dests.masked_select(not_in_active)
        new_val = s_new.index_select(0, not_in_active.nonzero(as_tuple=False).squeeze(-1))
        cat_idx = torch.cat([new_active_idx, new_idx], dim=0)
        cat_val = torch.cat([new_active_val, new_val], dim=0)
        sort_perm = cat_idx.argsort()
        new_active_idx = cat_idx.index_select(0, sort_perm)
        new_active_val = cat_val.index_select(0, sort_perm)

    return new_active_idx, new_active_val


@torch.no_grad()
def commit_overlay_to_base(
    s_base: torch.Tensor,            # [B, N, D_s] detached; will be modified
    active_flat_idx: torch.Tensor,   # [U] int64
    active_val: torch.Tensor,        # [U, D_s]
) -> None:
    """Fold overlay values back into the detached base (in place).

    Called at TBPTT block boundary. The overlay's values are detached
    before scatter — we are freezing the just-closed block's trajectory
    into the long-term state.
    """
    if active_flat_idx.numel() == 0:
        return
    D_s = s_base.shape[-1]
    s_base_flat = s_base.view(-1, D_s)
    s_base_flat.index_copy_(0, active_flat_idx, active_val.detach())
