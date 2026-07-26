"""Stage 7 — mixed node vectors -> per-layer KV entries the frozen decoder reads.

The injected entry is a **residual correction to the pooled KV**, not a fresh synthesis:

    K_inject[l] = RoPE( normmatch( K_pooled[l] + W_k(mixed) * scale_k[l] + shift_k[l] ), pos )
    V_inject[l] =       normmatch( V_pooled[l] + W_v(mixed) * scale_v[l] + shift_v[l] )

With the projections zero-initialised, injection at step 0 is *exactly* the raw pooled KV — already a sane
compressed cache, roughly a structured KVzip. Training then only has to improve on a working starting point
instead of discovering one, which removes most of the distribution-shift risk that is the highest-ranked
instability in this design.

**The zero-init costs one step of mixer training, and that is fine.** Because `to_k.weight == 0` at init,
`dL/d(mixed) = to_k.weight^T · ...` is zero, so NO gradient reaches the mixer on step 0 — only `to_k`/`to_v`
themselves move (their own gradient, `dL/d(dk) (x) mixed`, is non-zero). From step 1 the projections are off
zero and the mixer trains normally. This is exactly LoRA's bootstrap and is verified by
`tests/test_kvgraph.py::test_zero_init_bootstraps_mixer_gradient`. Do NOT "fix" it by zero-initialising the
per-layer FiLM scale instead: that kills the gradient to `to_k` as well and the branch never wakes up.

The per-layer adaptation is FiLM (one shared projection + per-layer scale/shift) rather than L independent
projections. A correction to something already good does not need L x d x d_kv parameters, and at 30 layers
the independent version would cost ~50x more than the entire relational operator bank — parameters spent in
the wrong place.

**RoPE.** Three modes, because this is an open empirical question and the harness's existing per-layer-KV
arms (h2o / gisting / memoryllm) all use `none`:

  compact  pool pre-RoPE, re-rotate at the node's RANK in first-mention order. Our default. Compact ranks
           look like an ordinary short prompt, which is in-distribution; many entries sharing one position
           is not, since the model only ever saw strictly increasing positions.
  none     inject unrotated. NOT "position-free": an unrotated key behaves exactly as a key at position 0,
           because the query is rotated regardless and we cannot change that. Matches the rest of the
           harness, so it is the apples-to-apples setting.
  original inject rotated at each node's first-mention token index. Puts long-document nodes in the deep
           RoPE-decay regime; included so the claim can be measured rather than asserted.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Injector(nn.Module):
    def __init__(self, d_mix: int, n_layers: int, n_kv: int, head_dim: int, *,
                 rope_mode: str = "compact", norm_match: bool = True):
        super().__init__()
        if rope_mode not in ("compact", "none", "original"):
            raise ValueError(f"rope_mode must be compact|none|original, got {rope_mode!r}")
        self.L, self.n_kv, self.head_dim = n_layers, n_kv, head_dim
        self.rope_mode, self.norm_match = rope_mode, norm_match
        kv_dim = n_kv * head_dim
        self.to_k = nn.Linear(d_mix, kv_dim)
        self.to_v = nn.Linear(d_mix, kv_dim)
        for lin in (self.to_k, self.to_v):                  # zero-init => injection == pooled KV at step 0
            nn.init.zeros_(lin.weight)
            nn.init.zeros_(lin.bias)
        self.k_scale = nn.Parameter(torch.ones(n_layers, kv_dim))
        self.v_scale = nn.Parameter(torch.ones(n_layers, kv_dim))
        self.k_shift = nn.Parameter(torch.zeros(n_layers, kv_dim))
        self.v_shift = nn.Parameter(torch.zeros(n_layers, kv_dim))

    @staticmethod
    def _apply_rope(k: torch.Tensor, base_model, positions: torch.Tensor) -> torch.Tensor:
        """Rotate `[n, n_kv, head_dim]` keys at `positions` using the backbone's own rotary embedding.

        Using the model's `rotary_emb` rather than reimplementing it matters: theta base, scaling type and
        partial-rotary fractions all vary by checkpoint, and a hand-rolled RoPE that disagrees with the
        backbone is a silent, position-dependent corruption.
        """
        rot = getattr(base_model.model, "rotary_emb", None)
        if rot is None:                                     # pragma: no cover - non-Llama backbone
            return k
        x = k.permute(1, 0, 2)[None]                        # [1, n_kv, n, hd] == HF's [B, H, T, D]
        pos = positions[None].to(k.device)
        cos, sin = rot(x, pos)
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
        _, k_rot = apply_rotary_pos_emb(x, x, cos, sin)
        return k_rot[0].permute(1, 0, 2)

    def forward(self, K_pooled: torch.Tensor, V_pooled: torch.Tensor, mixed: torch.Tensor,
                *, base_model=None, first_tokens: torch.Tensor | None = None,
                k_rms: torch.Tensor | None = None, v_rms: torch.Tensor | None = None):
        """K/V pooled `[n, L, n_kv, hd]`, mixed `[n, d_mix]` -> (Ks, Vs) lists of `[1, n_kv, n, hd]`.

        `first_tokens` is each node's first-mention token index; the compact rank is its ARGSORT, never the
        index itself — rotating at the raw index would place a node from 50k tokens ago at position 50k and
        put it in the deep-decay regime, which is the whole thing compact ranking avoids.
        """
        n, L = K_pooled.shape[0], self.L
        # The mixer and this module run in fp32 while the backbone's KV is bf16. Do the arithmetic in fp32
        # and cast once at the end: mixing dtypes mid-expression silently downcasts the correction, and a
        # bf16 residual added to a bf16 base loses most of the precision the correction was carrying.
        out_dtype = K_pooled.dtype
        K_pooled, V_pooled = K_pooled.float(), V_pooled.float()
        mixed = mixed.float()
        dk = self.to_k(mixed).view(n, self.n_kv * self.head_dim)
        dv = self.to_v(mixed).view(n, self.n_kv * self.head_dim)

        if self.rope_mode == "compact":
            order = torch.argsort(first_tokens if first_tokens is not None
                                  else torch.arange(n, device=K_pooled.device))
            pos = torch.empty(n, dtype=torch.long, device=K_pooled.device)
            pos[order] = torch.arange(n, device=K_pooled.device)
        elif self.rope_mode == "original":
            pos = (first_tokens if first_tokens is not None
                   else torch.arange(n, device=K_pooled.device)).long()
        else:
            pos = None

        Ks, Vs = [], []
        for li in range(L):
            k = K_pooled[:, li].reshape(n, -1) + dk * self.k_scale[li] + self.k_shift[li]
            v = V_pooled[:, li].reshape(n, -1) + dv * self.v_scale[li] + self.v_shift[li]
            if self.norm_match:
                # Mean-pooling shrinks vectors by ~1/sqrt(m), so the most-mentioned nodes would otherwise be
                # the quietest in attention. Rescale to the layer's real-token RMS.
                if k_rms is not None:
                    k = k * (k_rms[li] / k.pow(2).mean().sqrt().clamp(min=1e-6))
                if v_rms is not None:
                    v = v * (v_rms[li] / v.pow(2).mean().sqrt().clamp(min=1e-6))
            k = k.view(n, self.n_kv, self.head_dim).to(out_dtype)
            v = v.view(n, self.n_kv, self.head_dim).to(out_dtype)
            if pos is not None and base_model is not None:
                k = self._apply_rope(k, base_model, pos)
            Ks.append(k.permute(1, 0, 2)[None].contiguous())    # [1, n_kv, n, hd]
            Vs.append(v.permute(1, 0, 2)[None].contiguous())
        return Ks, Vs
