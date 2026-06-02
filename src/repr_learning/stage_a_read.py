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


class _ARBlock(nn.Module):
    """Causal self-attn (over value tokens) + cross-attn (to memory) + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.sa = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ca = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n1, self.n2, self.n3 = nn.LayerNorm(d_model), nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, mem, kpm, causal):
        h = self.n1(x); x = x + self.sa(h, h, h, attn_mask=causal, need_weights=False)[0]
        h = self.n2(x); x = x + self.ca(h, mem, mem, key_padding_mask=kpm, need_weights=False)[0]
        h = self.n3(x); x = x + self.ff(h)
        return x


class ARReadHead(nn.Module):
    """Small causal decoder: autoregress value tokens while cross-attending memory.
    Token k's input = embed(token k-1) + key-conditioning + pos[k]; predicts token k.
    Handles BPE subword dependence (which the non-AR (key,pos) head cannot). Llama is
    used ONLY as frozen embed/un-embed outside this module — this reader is its own
    ~1-2M-param module, NOT the Llama LM, so we do not 'end up where we started'."""

    def __init__(self, d_key: int, d_out: int, d_model: int = 256,
                 n_layers: int = 2, n_heads: int = 4, max_slots: int = 32):
        super().__init__()
        self.mem_proj = nn.LazyLinear(d_model, bias=False)   # bias=False keeps zero-memory control valid
        self.in_proj = nn.Linear(d_out, d_model)             # prev-token embeddings (d_llama) -> d_model
        self.key_proj = nn.Linear(d_key, d_model)
        self.pos = nn.Parameter(torch.randn(max_slots, d_model) * 0.02)
        self.blocks = nn.ModuleList(_ARBlock(d_model, n_heads) for _ in range(n_layers))
        self.out = nn.Linear(d_model, d_out)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, memory, mem_mask, key_vec, tok_embeds):
        # memory [N,M,d_mem]; key_vec [N,d_key]; tok_embeds [N,L,d_out] (prev tokens) -> [N,L,d_out]
        N, L, _ = tok_embeds.shape
        assert L <= self.pos.size(0), f"value len {L} > max_slots {self.pos.size(0)}"
        mem = self.mem_proj(memory)
        x = self.in_proj(tok_embeds) + self.key_proj(key_vec)[:, None, :] + self.pos[:L][None]
        kpm = (~mem_mask) if mem_mask is not None else None
        causal = torch.triu(torch.full((L, L), float("-inf"), device=x.device), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mem, kpm, causal)
        return self.out(self.norm(x))                                        # [N,L,d_out]


class StageAModel(nn.Module):
    def __init__(self, cfg: ReprConfig, arm: str, d_read: int = 256, max_slots: int = 32,
                 read_mode: str = "ar", bos_id: int = 128_000):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.read_mode = read_mode                                             # "ar" (default) | "nonar"
        self.bos_id = bos_id                                                   # Llama-3 BOS (decode start token)
        self.encoder = ARM_CLASSES[arm](cfg)                                   # the WRITE (trainable)
        # frozen Llama embed/un-embed (lookup only — never the decoder)
        from transformers import AutoModelForCausalLM
        llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
        self.embed = llama.get_input_embeddings()
        for p in self.embed.parameters():
            p.requires_grad_(False)
        # frozen un-embedding as a registered buffer (plain attrs are NOT moved by .to(device))
        self.register_buffer("unembed_w", llama.lm_head.weight.detach(), persistent=False)  # [V, d_llama]
        ReadCls = ARReadHead if read_mode == "ar" else SharedKVReadHead
        self.read = ReadCls(cfg.d_llama, cfg.d_llama, d_model=d_read, max_slots=max_slots)

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

    def _memory(self, batch, zero_memory, shuffle_memory, oracle_memory):
        """Return (mem_exp [B*P,M,d], mem_mask [B*P,M], key_vec [B*P,d_key], B, P)."""
        keys, k_mask = batch["keys"], batch["keys_mask"]                       # [B,P,Tk]
        B, P, Tk = keys.shape
        ke = self.embed(keys).to(torch.float32)                              # key conditioning
        kw = k_mask.unsqueeze(-1).float()
        key_vec = (ke * kw).sum(2) / kw.sum(2).clamp_min(1.0)                 # [B,P,d] mean-pool
        if oracle_memory:
            # POSITIVE CONTROL: ground-truth value embeddings ARE the memory (per query).
            vals, v_mask = batch["values"], batch["values_mask"]             # [B,P,Tv]
            mem_exp = self.embed(vals.reshape(B * P, -1)).to(torch.float32)  # [B*P,Tv,d]
            mm_exp = v_mask.reshape(B * P, -1)
        else:
            memory = self.write(batch["passage"], batch["passage_mask"])     # [B,M,d]
            if zero_memory:
                memory = torch.zeros_like(memory)
            if shuffle_memory:
                assert B >= 2, "shuffle_memory control needs batch size >= 2"
                memory = memory.roll(1, 0)                                  # bind to the WRONG item
            M = memory.size(1)
            mem_mask = torch.ones(B, M, dtype=torch.bool, device=memory.device)  # arms emit dense memory
            mem_exp = memory.repeat_interleave(P, 0)                        # [B*P,M,d] each query -> own item
            mm_exp = mem_mask.repeat_interleave(P, 0)
        return mem_exp, mm_exp, key_vec.reshape(B * P, -1), B, P

    def forward(self, batch, n_slots, zero_memory=False, shuffle_memory=False, oracle_memory=False):
        # NON-AR path: per-slot (key,pos) query -> logits [B,P,n_slots,V]
        mem_exp, mm_exp, key_flat, B, P = self._memory(batch, zero_memory, shuffle_memory, oracle_memory)
        r = self.read(mem_exp, mm_exp, key_flat, n_slots)                    # [B*P,L,d_llama]
        return (r @ self.unembed_w.t()).view(B, P, n_slots, -1)

    def _ar_logits_tf(self, mem_exp, mm_exp, key_flat, vflat):
        # teacher-forced AR logits. decoder input = [BOS, v0..v_{L-2}] embeddings.
        N, L = vflat.shape
        bos = torch.full((N, 1), self.bos_id, dtype=torch.long, device=vflat.device)
        tok_in = torch.cat([bos, vflat[:, :-1]], dim=1)                      # shift-right
        tok_emb = self.embed(tok_in).to(torch.float32)                      # [N,L,d_llama]
        h = self.read(mem_exp, mm_exp, key_flat, tok_emb)                    # [N,L,d_llama]
        return h @ self.unembed_w.t()                                       # [N,L,V]

    @torch.no_grad()
    def _ar_generate(self, mem_exp, mm_exp, key_flat, L):
        # greedy autoregressive decode (eval-only) -> [N,L] generated ids.
        N = mem_exp.size(0)
        ids = torch.full((N, 1), self.bos_id, dtype=torch.long, device=mem_exp.device)
        for _ in range(L):
            tok_emb = self.embed(ids).to(torch.float32)                     # [N,t,d]
            h = self.read(mem_exp, mm_exp, key_flat, tok_emb)               # [N,t,d_llama]
            nxt = (h[:, -1] @ self.unembed_w.t()).argmax(-1)               # [N]
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        return ids[:, 1:]                                                   # [N,L]

    def loss_and_recall(self, batch, zero_memory=False, shuffle_memory=False, oracle_memory=False):
        vals, v_mask = batch["values"], batch["values_mask"]                 # [B,P,Tv]
        L = vals.size(-1)
        if self.read_mode == "ar":
            mem_exp, mm_exp, key_flat, B, P = self._memory(batch, zero_memory, shuffle_memory, oracle_memory)
            vflat, vmflat = vals.reshape(B * P, -1), v_mask.reshape(B * P, -1)
            logits = self._ar_logits_tf(mem_exp, mm_exp, key_flat, vflat)    # TF [B*P,L,V]
            ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), vflat.reshape(-1),
                                 reduction="none").view_as(vflat)
            loss = (ce * vmflat).sum() / vmflat.sum().clamp_min(1.0)
            with torch.no_grad():
                gen = self._ar_generate(mem_exp, mm_exp, key_flat, L)        # honest greedy AR decode
                last = vmflat.sum(-1, keepdim=True) - 1                      # EOS index per row
                posL = torch.arange(L, device=vflat.device).view(1, -1)
                content = vmflat & (posL != last)                           # value tokens, no EOS
                recall = ((gen == vflat) | ~content).all(-1).float().mean()
            return loss, recall
        # NON-AR path
        logits = self(batch, L, zero_memory=zero_memory, shuffle_memory=shuffle_memory,
                      oracle_memory=oracle_memory)                           # [B,P,L,V]
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), vals.reshape(-1),
                             reduction="none").view_as(vals)
        loss = (ce * v_mask).sum() / v_mask.sum().clamp_min(1.0)
        with torch.no_grad():
            pred = logits.argmax(-1)
            last = v_mask.sum(-1, keepdim=True) - 1                           # exclude always-last EOS slot
            posL = torch.arange(vals.size(-1), device=vals.device).view(1, 1, -1)
            content = v_mask & (posL != last)
            recall = ((pred == vals) | ~content).all(-1).float().mean()
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
