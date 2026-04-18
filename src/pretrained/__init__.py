"""Pretrained-LM + memory integration.

Host: Llama-3.2-3B (or 1B for dev). Frozen backbone. Memory graph reads and
writes at one mid-stack layer via a per-dim scale gate. Phase 1 backprop
bootstrap, Phase 2 autoregressive GRPO.

Entry points:
- `PretrainedLMWithMemory` wires the pieces together.
- `MemInjectLayer` wraps the chosen LlamaDecoderLayer.

Keep this module self-contained; do not import from `src.model.lm` or
`src.model.scan` (those are the from-scratch LM, unused on this branch).
"""
