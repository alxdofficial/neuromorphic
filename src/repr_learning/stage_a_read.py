"""Stage-A model: write a passage to an arm's memory, then read a value by key
(see docs/memory_warmup_curriculum.md §2–3).

- WRITE: the arm's existing `streaming_write` → memory [B, M, d_llama] (unchanged).
- READ: a tiny shared cross-attention head. Per value slot k the query is
  (key-conditioning + position) → cross-attend the memory → linear → token.
  Non-autoregressive: each slot is independent of previous tokens, so there is no
  language model and nothing to "guess" — the value must come from the memory.
- Llama is used ONLY as a frozen embed/un-embed lookup (input embeddings + tied
  output classifier); it is NEVER the decoder.

The same shared head (separate weights per arm) gives an apples-to-apples
write-quality test; graph_v6 can additionally use its native read.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import (
    FlatBaselineEncoder, ContinuousBaselineEncoder,
    RecurrentBaselineEncoder, GraphV6BaselineEncoder,
)

ARM_CLASSES = {
    "flat_baseline": FlatBaselineEncoder,
    "continuous_baseline": ContinuousBaselineEncoder,
    "recurrent_baseline": RecurrentBaselineEncoder,
    "graph_v6_baseline": GraphV6BaselineEncoder,
}


class SharedKVReadHead(nn.Module):
    """(key-conditioning + position) → cross-attend memory → d_llama vector.
    Logits come from a tied frozen un-embedding outside this module."""

    def __init__(self, d_key: int, d_out: int, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4, max_slots: int = 32):
        super().__init__()
        # LazyLinear infers the memory dim per arm: d_llama for the prepend
        # baselines, graph_v6's fact dim (d_read) for the graph. bias=False so the
        # zero-memory control (memory=0) -> zero keys/values (valid ablation).
        self.mem_proj = nn.LazyLinear(d_model, bias=False)
        self.key_proj = nn.Linear(d_key, d_model)
        self.pos = nn.Parameter(torch.randn(max_slots, d_model) * 0.02)
        self.blocks = nn.ModuleList(_ReadBlock(d_model, n_heads) for _ in range(n_layers))
        self.out = nn.Linear(d_model, d_out)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory: torch.Tensor, mem_mask: Optional[torch.Tensor],
                key_vec: torch.Tensor, n_slots: int) -> torch.Tensor:
        # memory [N, M, d_mem]; key_vec [N, d_key]; -> [N, n_slots, d_out]
        assert n_slots <= self.pos.size(0), f"value len {n_slots} > max_slots {self.pos.size(0)}"
        mem = self.mem_proj(memory)
        q = self.key_proj(key_vec)[:, None, :] + self.pos[:n_slots][None]      # [N, L, d_model]
        kpm = (~mem_mask) if mem_mask is not None else None                    # True = ignore
        for blk in self.blocks:
            q = blk(q, mem, kpm)
        return self.out(self.norm(q))                                          # [N, L, d_out]


class _ReadBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, q, mem, kpm):
        h = self.n1(q); q = q + self.sa(h, h, h, need_weights=False)[0]        # slots coherence
        h = self.n2(q); q = q + self.ca(h, mem, mem, key_padding_mask=kpm, need_weights=False)[0]
        h = self.n3(q); q = q + self.ff(h)
        return q


class StageAModel(nn.Module):
    def __init__(self, cfg: ReprConfig, arm: str, d_read: int = 256, max_slots: int = 32):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.encoder = ARM_CLASSES[arm](cfg)                                   # the WRITE (trainable)
        # frozen Llama embed/un-embed (lookup only — never the decoder)
        from transformers import AutoModelForCausalLM
        llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
        self.embed = llama.get_input_embeddings()
        for p in self.embed.parameters():
            p.requires_grad_(False)
        # frozen un-embedding as a registered buffer (plain attrs are NOT moved by .to(device))
        self.register_buffer("unembed_w", llama.lm_head.weight.detach(), persistent=False)  # [V, d_llama]
        self.read = SharedKVReadHead(cfg.d_llama, cfg.d_llama, d_model=d_read, max_slots=max_slots)

    def write(self, passage, passage_mask):
        # streaming_write updates STATE; finalize_memory(state) yields the memory.
        # Passage is short (<= one window) so a single write call suffices.
        te = self.embed(passage).to(torch.float32)
        st = self.encoder.init_streaming_state(passage.size(0), passage.device, te.dtype)
        st, _ = self.encoder.streaming_write(st, te, attention_mask=passage_mask, chunk_offset=0)
        memory, aux = self.encoder.finalize_memory(st)
        if memory.size(1) == 0 and isinstance(aux, dict) and aux.get("graph_v6_facts") is not None:
            memory = aux["graph_v6_facts"]["value"]   # graph injects fact tokens (dim d_read), not prepend
        return memory.to(torch.float32)               # [B, M, d_mem]

    def forward(self, batch, n_slots: int, zero_memory=False, shuffle_memory=False):
        passage, p_mask = batch["passage"], batch["passage_mask"]
        keys, k_mask = batch["keys"], batch["keys_mask"]                       # [B,P,Tk]
        B, P, Tk = keys.shape
        memory = self.write(passage, p_mask)                                  # [B,M,d]
        if zero_memory:
            memory = torch.zeros_like(memory)
        if shuffle_memory:
            assert B >= 2, "shuffle_memory control needs batch size >= 2"
            memory = memory.roll(1, 0)                                        # bind to the WRONG item
        M = memory.size(1)
        mem_mask = torch.ones(B, M, dtype=torch.bool, device=memory.device)  # arms emit dense memory
        # key conditioning: mean-pool key token embeddings
        ke = self.embed(keys).to(torch.float32)                              # [B,P,Tk,d]
        kw = k_mask.unsqueeze(-1).float()
        key_vec = (ke * kw).sum(2) / kw.sum(2).clamp_min(1.0)                 # [B,P,d]
        # flatten P into batch; each query reads its own item's memory
        mem_exp = memory.repeat_interleave(P, 0)                              # [B*P,M,d]
        mm_exp = mem_mask.repeat_interleave(P, 0)
        r = self.read(mem_exp, mm_exp, key_vec.reshape(B * P, -1), n_slots)   # [B*P,L,d_llama]
        logits = r @ self.unembed_w.t()                                      # [B*P,L,V]
        return logits.view(B, P, n_slots, -1)

    def loss_and_recall(self, batch, zero_memory=False, shuffle_memory=False):
        vals, v_mask = batch["values"], batch["values_mask"]                 # [B,P,Tv]
        L = vals.size(-1)
        logits = self(batch, L, zero_memory=zero_memory, shuffle_memory=shuffle_memory)  # [B,P,L,V]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), vals.reshape(-1),
                             reduction="none").view_as(vals)
        loss = (ce * v_mask).sum() / v_mask.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = logits.argmax(-1)
            # exclude the always-last EOS slot from recall (predictable from position alone);
            # EOS stays in the CE so the read still learns where the value ends.
            last = v_mask.sum(-1, keepdim=True) - 1                           # [B,P,1] EOS index per row
            posL = torch.arange(vals.size(-1), device=vals.device).view(1, 1, -1)
            content = v_mask & (posL != last)                                # value tokens, no EOS
            tok_ok = (pred == vals) | ~content
            recall = tok_ok.all(-1).float().mean()                           # exact match on the value span
        return loss, recall


if __name__ == "__main__":  # forward/backward smoke (graph_v6, tiny)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a

    cfg = ReprConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    for arm in ["graph_v6_baseline", "flat_baseline", "continuous_baseline", "recurrent_baseline"]:
        m = StageAModel(cfg, arm).to(dev)
        b = next(iter(DataLoader(StageAKVDataset(tok, n_pairs=4, seed=3), batch_size=4,
                                 collate_fn=collate_stage_a)))
        b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
        loss, rec = m.loss_and_recall(b)
        loss.backward()
        gn = sum(p.grad.norm().item() for p in m.read.parameters() if p.grad is not None)
        wn = sum(p.grad.norm().item() for p in m.encoder.parameters() if p.grad is not None)
        z, _ = m.loss_and_recall(b, zero_memory=True)
        print(f"{arm:22s} loss={loss.item():.3f} recall={rec.item():.2f} "
              f"read_gnorm={gn:.2e} write_gnorm={wn:.2e} | zero-mem loss={z.item():.3f}")
        del m; torch.cuda.empty_cache() if dev == "cuda" else None
    print("OK — write+read both get gradient; zero-mem loss should be >= real loss.")
