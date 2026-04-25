"""GraphWalker Triton kernels.

See docs/triton_rewrite_plan.md for the design. The hot per-step path
fuses the dispatch-bound chains of small ops into Triton kernels, while
matmuls (content_mlp, q_proj, k_proj) stay in PyTorch / cuBLAS where
they are already near-peak.

Public surface:
- `lif.lif_deposit` — sparse LIF deposit (replaces SparseLIFUpdate)
- `step_postlif.step_postlif` — fused gather + scoring + Gumbel + STE +
  endpoint readout (replaces ~7 small PyTorch ops per step)
- `anchor_pick.anchor_pick` — fused input-plane scoring + Gumbel STE
  (window-start only)
- `walker_step.walker_step` — autograd.Function bundling one step's matmul +
  Triton chain
- `walker_block.walker_block` — autograd.Function bundling T_block steps
  for clean cudagraph capture
"""

try:
    import triton  # noqa: F401
    HAS_TRITON = True
except ImportError:                                     # pragma: no cover
    HAS_TRITON = False
