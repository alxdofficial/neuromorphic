"""Stage-A model: write a passage to an arm's memory, then read a value by a
question-key (see docs/memory_warmup_curriculum.md §2–3).

- WRITE: the arm's existing `streaming_write`/`finalize_memory` → memory tokens.
- READ: ONE unified prefix-LM transformer, identical across all arms. The memory
  module hands over a set of tokens; we form the sequence
      [ memory tokens | key (question) tokens | answer-so-far ]
  and self-attend with prefix-LM masking (memory+key visible, answer causal), then
  decode the answer autoregressively. De-duplication (don't retrieve the same fact
  for every answer position) falls out of the answer self-attention. Llama is used
  ONLY as a frozen embed/un-embed; it is never the decoder.
- WRITE training signal: token-recon CE on the value (QA) PLUS an optional
  reconstruction objective — decode the whole passage from memory — that forces the
  write to actually encode its input (500xCompressor/ICAE recipe).
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.repr_learning.config import ReprConfig
from src.repr_learning.encoder import (
    VQVAEBaselineEncoder, SlotAttentionBaselineEncoder,
    MambaBaselineEncoder, GraphV6BaselineEncoder,
)

ARM_CLASSES = {
    "vqvae_baseline": VQVAEBaselineEncoder,
    "slot_attention_baseline": SlotAttentionBaselineEncoder,
    "mamba_baseline": MambaBaselineEncoder,
    "graph_v6_baseline": GraphV6BaselineEncoder,
    # back-compat aliases
    "flat_baseline": VQVAEBaselineEncoder,
    "continuous_baseline": SlotAttentionBaselineEncoder,
    "recurrent_baseline": MambaBaselineEncoder,
}

T_MEM, T_COND, T_ANS = 0, 1, 2          # stream type ids for the read transformer


class _SelfAttnBlock(nn.Module):
    """Pre-norm self-attention (with attn_mask + key_padding_mask) + FFN."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.n1, self.n2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.n1(x)
        x = x + self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        x = x + self.ff(self.n2(x))
        return x


class PrefixLMReadHead(nn.Module):
    """[memory | cond | answer] prefix-LM transformer. memory+cond are a bidirectional
    prefix; answer positions are causal and read out. Same head for every arm — the
    only per-arm difference is `mem_proj`, a LazyLinear that infers the arm's memory
    width on first forward (d_llama for prepend arms, d_read for graph fact-tokens)."""

    def __init__(self, d_in: int, d_out: int, d_model: int = 256,
                 n_layers: int = 3, n_heads: int = 4, max_answer: int = 320):
        super().__init__()
        self.mem_proj = nn.LazyLinear(d_model, bias=False)   # bias=False keeps zero-memory control valid
        self.tok_proj = nn.Linear(d_in, d_model)             # key + answer token embeds (d_llama) -> d_model
        self.type_emb = nn.Parameter(torch.randn(3, d_model) * 0.02)
        self.ans_pos = nn.Parameter(torch.randn(max_answer, d_model) * 0.02)
        self.blocks = nn.ModuleList(_SelfAttnBlock(d_model, n_heads) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, d_out)
        self.max_answer = max_answer

    @staticmethod
    def _prefix_lm_mask(P: int, A: int, device) -> torch.Tensor:
        # True = BLOCK. prefix<->prefix bidirectional; answer sees all prefix + causal answer.
        seq = P + A
        block = torch.ones(seq, seq, dtype=torch.bool, device=device)
        block[:P, :P] = False
        block[P:, :P] = False
        block[P:, P:] = torch.triu(torch.ones(A, A, dtype=torch.bool, device=device), diagonal=1)
        return block

    def forward(self, memory, mem_mask, cond_emb, cond_mask, ans_emb):
        # memory [N,M,d_mem]; cond_emb [N,C,d_in]; ans_emb [N,A,d_in]
        N, M = memory.shape[0], memory.shape[1]
        C, A = cond_emb.shape[1], ans_emb.shape[1]
        assert A <= self.max_answer, f"answer len {A} > max {self.max_answer}"
        mem_t = self.mem_proj(memory) + self.type_emb[T_MEM]
        cond_t = self.tok_proj(cond_emb) + self.type_emb[T_COND]
        ans_t = self.tok_proj(ans_emb) + self.type_emb[T_ANS] + self.ans_pos[:A][None]
        x = torch.cat([mem_t, cond_t, ans_t], dim=1)                       # [N, M+C+A, d_model]
        P = M + C
        attn_mask = self._prefix_lm_mask(P, A, x.device)
        # key padding: ignore padded memory + padded cond tokens; answer keys all kept
        kpm = torch.cat([~mem_mask, ~cond_mask,
                         torch.zeros(N, A, dtype=torch.bool, device=x.device)], dim=1)
        for blk in self.blocks:
            x = blk(x, attn_mask, kpm)
        return self.out(self.norm(x[:, P:]))                              # [N, A, d_out]


class StageAModel(nn.Module):
    def __init__(self, cfg: ReprConfig, arm: str, d_read: int = 256, max_answer: int = 512,
                 bos_id: int = 128_000, deterministic_write: bool = False,
                 recon_weight: float = 0.0):
        super().__init__()
        self.cfg, self.arm = cfg, arm
        self.bos_id = bos_id
        self.recon_weight = recon_weight                                   # passage-reconstruction objective
        self.encoder = ARM_CLASSES[arm](cfg)                               # the WRITE (trainable)
        if deterministic_write:
            self.encoder.deterministic_write = True
        from transformers import AutoModelForCausalLM
        llama = AutoModelForCausalLM.from_pretrained(cfg.llama_model, torch_dtype=torch.float32)
        self.embed = llama.get_input_embeddings()
        for p in self.embed.parameters():
            p.requires_grad_(False)
        self.register_buffer("unembed_w", llama.lm_head.weight.detach(), persistent=False)  # [V, d_llama]
        self.read = PrefixLMReadHead(cfg.d_llama, cfg.d_llama, d_model=d_read, max_answer=max_answer)
        # learned marker that stands in for the "cond" stream during reconstruction
        # (no question — "decode the whole passage from memory").
        self.recon_marker = nn.Parameter(torch.randn(1, 1, cfg.d_llama) * 0.02)

    # ── write → per-item memory ──────────────────────────────────────────────
    def _write_memory(self, passage, p_mask):
        te = self.embed(passage).to(torch.float32)
        st = self.encoder.init_streaming_state(passage.size(0), passage.device, te.dtype)
        st, _ = self.encoder.streaming_write(st, te, attention_mask=p_mask, chunk_offset=0)
        # continuous baseline: the carried slot state [B, n_flat_codes, d_enc] IS the
        # per-item bottleneck. Hand it to the unified read directly (its mem_proj would
        # only undo the vestigial d_llama projection) so interface == carried == matched B,
        # exactly as graph hands its fact tokens. (flat keeps its codebook path; recurrent's
        # state is non-tokenwise — both still emit d_llama until they're matched too.)
        if self.arm == "continuous_baseline" and torch.is_tensor(st) and st.dim() == 3:
            return st.to(torch.float32)                                    # [B, n_flat_codes, d_enc]
        memory, aux = self.encoder.finalize_memory(st)
        if memory.size(1) == 0 and isinstance(aux, dict) and aux.get("graph_v6_facts") is not None:
            memory = aux["graph_v6_facts"]["value"]                        # graph fact-tokens (dim d_read)
        return memory.to(torch.float32)                                    # [B, M, d]

    def _memory_query(self, batch, zero_memory, shuffle_memory, oracle_memory, passage_memory):
        """Return (mem_exp [B*P,M,d], mem_mask [B*P,M], mem_item or None, B, P).
        mem_item [B,M,d] is the per-item REAL memory (None for oracle/passage), reused
        for the reconstruction objective so the write runs only once."""
        keys = batch["keys"]
        B, P = keys.shape[0], keys.shape[1]
        if oracle_memory:
            vals, v_mask = batch["values"], batch["values_mask"]
            mem_exp = self.embed(vals.reshape(B * P, -1)).to(torch.float32)
            return mem_exp, v_mask.reshape(B * P, -1), None, B, P
        if passage_memory:
            pas, pm = batch["passage"], batch["passage_mask"]
            return (self.embed(pas).to(torch.float32).repeat_interleave(P, 0),
                    pm.repeat_interleave(P, 0), None, B, P)
        memory = self._write_memory(batch["passage"], batch["passage_mask"])  # [B,M,d] — write ONCE
        if zero_memory:
            memory = torch.zeros_like(memory)
        if shuffle_memory:
            assert B >= 2, "shuffle_memory control needs batch size >= 2"
            memory = memory.roll(1, 0)
        M = memory.size(1)
        mm = torch.ones(B, M, dtype=torch.bool, device=memory.device)
        return memory.repeat_interleave(P, 0), mm.repeat_interleave(P, 0), memory, B, P

    def _key_embeds(self, batch):
        keys, k_mask = batch["keys"], batch["keys_mask"]                   # [B,P,Tk]
        B, P, Tk = keys.shape
        return self.embed(keys.reshape(B * P, Tk)).to(torch.float32), k_mask.reshape(B * P, Tk)

    # ── decode (one prefix-LM pass; reused for QA and reconstruction) ─────────
    def _decode_tf(self, memory, mem_mask, cond_emb, cond_mask, target_ids, target_mask):
        N, L = target_ids.shape
        bos = torch.full((N, 1), self.bos_id, dtype=torch.long, device=target_ids.device)
        ans_in = torch.cat([bos, target_ids[:, :-1]], dim=1)
        ans_emb = self.embed(ans_in).to(torch.float32)
        h = self.read(memory, mem_mask, cond_emb, cond_mask, ans_emb)      # [N,L,d_llama]
        logits = h @ self.unembed_w.t()
        ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
                             reduction="none").view_as(target_ids)
        return (ce * target_mask).sum() / target_mask.sum().clamp_min(1.0), logits

    @torch.no_grad()
    def _decode_generate(self, memory, mem_mask, cond_emb, cond_mask, L):
        N = memory.size(0)
        ids = torch.full((N, 1), self.bos_id, dtype=torch.long, device=memory.device)
        for _ in range(L):
            ans_emb = self.embed(ids).to(torch.float32)
            h = self.read(memory, mem_mask, cond_emb, cond_mask, ans_emb)
            nxt = (h[:, -1] @ self.unembed_w.t()).argmax(-1)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
        return ids[:, 1:]

    def loss_and_recall(self, batch, zero_memory=False, shuffle_memory=False,
                        oracle_memory=False, passage_memory=False):
        mem_exp, mm_exp, mem_item, B, P = self._memory_query(
            batch, zero_memory, shuffle_memory, oracle_memory, passage_memory)
        key_emb, key_m = self._key_embeds(batch)
        vals, v_mask = batch["values"], batch["values_mask"]               # [B,P,Tv]
        vflat, vmflat = vals.reshape(B * P, -1), v_mask.reshape(B * P, -1)
        L = vflat.size(-1)
        qa_loss, _ = self._decode_tf(mem_exp, mm_exp, key_emb, key_m, vflat, vmflat)
        loss = qa_loss

        if self.recon_weight > 0 and self.training and mem_item is not None:
            # reconstruct the whole passage from the per-item memory (write must encode input)
            pas, pm = batch["passage"], batch["passage_mask"]
            marker = self.recon_marker.expand(B, -1, -1).to(torch.float32)
            mk = torch.ones(B, 1, dtype=torch.bool, device=pas.device)
            mm_item = torch.ones(B, mem_item.size(1), dtype=torch.bool, device=pas.device)
            recon_loss, _ = self._decode_tf(mem_item, mm_item, marker, mk, pas, pm)
            loss = loss + self.recon_weight * recon_loss

        with torch.no_grad():
            gen = self._decode_generate(mem_exp, mm_exp, key_emb, key_m, L)  # greedy AR decode
            last = vmflat.sum(-1, keepdim=True) - 1                          # EOS index per row
            posL = torch.arange(L, device=vflat.device).view(1, -1)
            content = vmflat & (posL != last)                               # value tokens, no EOS
            recall = ((gen == vflat) | ~content).all(-1).float().mean()
        return loss, recall


if __name__ == "__main__":  # forward/backward smoke (all arms, tiny)
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from transformers import AutoTokenizer
    from torch.utils.data import DataLoader
    from src.repr_learning.data_stage_a import StageAKVDataset, collate_stage_a

    cfg = ReprConfig()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg.llama_model)
    for arm in ["graph_v6_baseline", "continuous_baseline", "flat_baseline", "recurrent_baseline"]:
        m = StageAModel(cfg, arm, recon_weight=1.0).to(dev)
        b = next(iter(DataLoader(StageAKVDataset(tok, n_pairs=2, seed=3), batch_size=4,
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
    print("OK — prefix-LM read + recon; write+read both get gradient.")
