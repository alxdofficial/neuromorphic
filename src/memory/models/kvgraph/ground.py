"""Stage 4 — pool the frozen LM's KV over each node's token positions.

This is where the graph stops being an annotation over text and becomes made of the model's own state.
Out comes, per node, exactly the shape of ONE TOKEN'S cache entry `[L, n_kv, head_dim]` — which is what
makes a node a drop-in cache entry a frozen decoder can read, and what makes the compression ratio directly
comparable to KVzip/H2O at matched budget.

Two things here are easy to get wrong and silent when wrong:

**RoPE.** Rotation is applied per-token inside attention, so a cached key is already spun by its own
position. Pooling positions 4 and 87 averages vectors whose high-frequency dims differ by ~83 radians —
effectively a random relative angle, so they CANCEL. The damage scales with mention distance, i.e. it is
worst exactly for the long-range coreference merges the design exists to exploit. So we hook `k_proj`,
which fires BEFORE rotation, and re-rotate later at a position of our choosing (inject.py). V is never
rotated and pools straight from `v_proj`.

**Norm.** Mean-pooling n roughly-orthogonal vectors shrinks the result by ~1/sqrt(n), so the most-mentioned
nodes — the important ones — come out the quietest in attention. Rescaled in inject.py against the layer's
real-token statistics, which is why those statistics are captured here.
"""
from __future__ import annotations

import torch


class KVCapture:
    """Runs the frozen LM once over a window and captures pre-RoPE K/V plus per-layer norm statistics.

    Deliberately NOT a forward hook left permanently installed: hooks that outlive their scope are how you
    end up capturing a different forward than you think you are.
    """

    def __init__(self, base):
        cfg = base.config
        self.base = base
        self.L = cfg.num_hidden_layers
        self.n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)

    @torch.no_grad()
    def run(self, embeds: torch.Tensor, mask: torch.Tensor) -> dict:
        """-> {"k","v": [L][B,T,n_kv,hd] pre-RoPE, "hidden": [L+1][B,T,d], "k_rms","v_rms": [L]}

        `hidden` is kept because the mixer reads from a mid-stack layer (entity/coreference information
        peaks in the middle of the stack, not at the end) and because the edge vector is attention-pooled
        from the tokens that licensed the arc.
        """
        B, T, _ = embeds.shape
        kbuf: list[torch.Tensor | None] = [None] * self.L
        vbuf: list[torch.Tensor | None] = [None] * self.L

        def _hook(buf, li):
            def fn(module, inp, out):
                buf[li] = out.detach().view(B, T, self.n_kv, self.head_dim)
            return fn

        handles = []
        for li, layer in enumerate(self.base.model.layers):
            handles.append(layer.self_attn.k_proj.register_forward_hook(_hook(kbuf, li)))
            handles.append(layer.self_attn.v_proj.register_forward_hook(_hook(vbuf, li)))
        try:
            out = self.base.model(inputs_embeds=embeds.to(next(self.base.parameters()).dtype),
                                  attention_mask=mask.long(), output_hidden_states=True, use_cache=False)
        finally:
            for h in handles:
                h.remove()
        if kbuf[0] is None:
            raise RuntimeError("kvgraph.ground: no k_proj output captured — the backbone does not expose "
                               "`model.layers[i].self_attn.k_proj` (non-Llama architecture?)")

        # Per-layer RMS over VALID tokens only; pad rows would drag the target norm toward zero.
        m = mask.bool()[:, :, None, None]
        k_rms = torch.stack([((k.float() ** 2) * m).sum() / (m.sum() * self.n_kv * self.head_dim)
                             for k in kbuf]).sqrt()
        v_rms = torch.stack([((v.float() ** 2) * m).sum() / (m.sum() * self.n_kv * self.head_dim)
                             for v in vbuf]).sqrt()
        return {"k": kbuf, "v": vbuf, "hidden": out.hidden_states, "k_rms": k_rms, "v_rms": v_rms}


def pool_nodes(cap: dict, node_tokens: list[list[int]], *, head_tokens: list[int] | None = None,
               head_weight: float = 2.0, batch_index: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """-> (K, V), each `[n_nodes, L, n_kv, head_dim]`, pooled over each node's token positions.

    Head-weighted mean: the syntactic head of a span carries most of its meaning ("farm" in "the family
    farm"), so it gets `head_weight`x the mass of the other tokens. `head_weight=1.0` recovers a plain mean,
    which is the control.

    An entity node's V coming out as a generic "Maria" is CORRECT, not a bug: the specific propositions live
    in the event nodes attached to it. That is the property that makes naive mean-pooling safe here, and it
    only holds because events are reified.
    """
    K_all, V_all = cap["k"], cap["v"]
    L = len(K_all)
    dev, dt = K_all[0].device, K_all[0].dtype
    n = len(node_tokens)
    n_kv, hd = K_all[0].shape[2], K_all[0].shape[3]
    K = torch.zeros(n, L, n_kv, hd, device=dev, dtype=dt)
    V = torch.zeros(n, L, n_kv, hd, device=dev, dtype=dt)
    for i, toks in enumerate(node_tokens):
        if not toks:
            continue
        idx = torch.as_tensor(toks, device=dev, dtype=torch.long)
        w = torch.ones(len(toks), device=dev, dtype=torch.float32)
        if head_tokens is not None and head_tokens[i] in toks:
            w[toks.index(head_tokens[i])] = head_weight
        w = (w / w.sum())[:, None, None]
        for li in range(L):
            K[i, li] = (K_all[li][batch_index, idx].float() * w).sum(0).to(dt)
            V[i, li] = (V_all[li][batch_index, idx].float() * w).sum(0).to(dt)
    return K, V


def pool_hidden(cap: dict, spans: list[list[int]], layer: int, batch_index: int = 0) -> torch.Tensor:
    """Mean-pool a mid-stack hidden layer over token spans -> `[n, d]`. Used for node summaries (the
    mixer's input) and for edge vectors (pooled over an arc's licensing tokens)."""
    h = cap["hidden"][layer][batch_index]                       # [T, d]
    out = torch.zeros(len(spans), h.shape[-1], device=h.device, dtype=h.dtype)
    for i, toks in enumerate(spans):
        if toks:
            out[i] = h[torch.as_tensor(toks, device=h.device, dtype=torch.long)].mean(0)
    return out
