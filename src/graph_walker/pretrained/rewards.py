"""Reward functions for Phase-2 GRPO.

Used by ``grpo_step`` via the ``reward_fn`` callable. A reward function
maps ``(generated [K, T_pre + L], reference [L]) -> rewards [K]``.

Production reward: BERT-cosine via ``sentence-transformers/all-mpnet-base-v2``.
Decode the generated tail and reference back to text via the LM tokenizer,
embed both with sentence-transformers, take cosine similarity. Pure
semantic, paraphrase-tolerant, no LLM-as-judge.

Why all-mpnet-base-v2:
- Strong semantic similarity (top of MTEB at its size class).
- ~110M params, ~5-10ms per forward at K=8 batch on a 4090.
- Already cached in HF.

Cost per GRPO step: 1 sentence-transformer forward of K+1 sentences.
At K=8, ~9 sequences through a 110M-param model = ~10 ms additional
per step on top of the ~1.3s baseline GRPO step. <1% overhead.

Caching the reference embedding across steps that share a reference
(e.g., Wave 3's 500-fact corpus revisited) saves the +1 part — see
``BertCosineReward.reference_cache``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class BertCosineReward:
    """BERT-cosine reward function, callable per the ``grpo_step`` contract.

    Construct once at training start; pass the bound ``__call__`` as
    ``reward_fn`` to ``grpo_step``.

    Args:
        bert_model: a ``sentence_transformers.SentenceTransformer`` (or
            an HF model with mean-pooled embedding extraction).
        tokenizer: the LM's tokenizer used to decode token ids back to text.
        device: where to run the BERT forward.
        clamp_min/max: optional clamp on the cosine before returning. Negative
            cosines are theoretically possible but rare for natural text;
            clamping to [0, 1] makes advantage normalization cleaner.
    """

    bert_model: object
    tokenizer: object
    device: torch.device | str = "cpu"
    clamp_min: float = 0.0
    clamp_max: float = 1.0
    # Optional: when the same reference is used across many GRPO steps
    # (Wave 3's 500-fact corpus), the trainer can pre-compute the
    # reference embedding once and pass it via __call__'s
    # `cached_ref_embedding` arg to skip K+1 -> K BERT forwards.
    reference_cache: dict[bytes, torch.Tensor] | None = None

    def __post_init__(self) -> None:
        self.device = torch.device(self.device)

    @torch.no_grad()
    def __call__(
        self,
        generated: torch.Tensor,                                 # [B*K, gen_length] long
        reference: torch.Tensor | list[torch.Tensor],            # [L] (B=1) or list of B tensors
    ) -> torch.Tensor:                                           # [B*K] float
        """Score B*K generations against B references under BERT-cosine.

        Layout convention (matches ``grpo_step`` post-BS_outer):
          rollout 0..K-1   → reference[0]
          rollout K..2K-1  → reference[1]
          ...
        i.e. ``generated`` is the result of ``prefix_ids[B, T_pre].
        repeat_interleave(K, dim=0)`` going through sample/replay.

        Args:
            generated: [B*K, gen_length] gen-token-only tail (NOT prefix+gen).
            reference: a single [L_ref] tensor (B=1 back-compat) or a
                list of B variable-length tensors.

        Returns:
            cosines [B*K] in [clamp_min, clamp_max], on ``generated.device``.

        Contract notes (2026-05-05):
        - ``generated`` MUST be the continuation tail only. Previous
          contract sliced ``[:, -L:]`` and silently grabbed the end of
          the gen for short refs (CRIT bug — caught by Codex audit).
        - Per-group cosine: gen[b*K:(b+1)*K] is scored against
          reference[b]. Caller is responsible for laying gens out in
          contiguous K-blocks per prefix.
        """
        if isinstance(reference, torch.Tensor):
            refs: list[torch.Tensor] = [reference]
        else:
            refs = list(reference)
        if len(refs) == 0:
            raise ValueError("reference list is empty")
        B = len(refs)
        BK = generated.shape[0]
        if BK % B != 0:
            raise ValueError(
                f"generated.shape[0]={BK} not divisible by B={B} "
                f"(K-block layout assumption)"
            )
        K = BK // B

        # Decode all gen tails in one batch_decode call.
        gen_texts = self.tokenizer.batch_decode(
            generated.cpu().tolist(), skip_special_tokens=True,
        )

        # Per-reference encoding, with cache lookup.
        ref_embs: list[torch.Tensor] = []
        for r_tok in refs:
            r_text = self.tokenizer.decode(
                r_tok.cpu().tolist(), skip_special_tokens=True,
            )
            ref_emb: torch.Tensor | None = None
            if self.reference_cache is not None:
                key = r_tok.cpu().numpy().tobytes()
                ref_emb = self.reference_cache.get(key)
                if ref_emb is None:
                    ref_emb = self._encode_one(r_text)
                    self.reference_cache[key] = ref_emb
            if ref_emb is None:
                ref_emb = self._encode_one(r_text)
                if self.reference_cache is not None:
                    key = r_tok.cpu().numpy().tobytes()
                    self.reference_cache[key] = ref_emb
            ref_embs.append(ref_emb)
        ref_emb_stack = torch.stack(ref_embs, dim=0)             # [B, D]

        # Tile each ref's embedding K times to align with gen's K-block layout.
        ref_emb_per_gen = ref_emb_stack.repeat_interleave(K, dim=0)  # [B*K, D]

        # B*K generated embeddings in one batch.
        gen_emb = self._encode_batch(gen_texts)                  # [B*K, D]

        cos = F.cosine_similarity(gen_emb, ref_emb_per_gen, dim=-1)  # [B*K]
        cos = cos.clamp(min=self.clamp_min, max=self.clamp_max)
        return cos.to(generated.device).float()

    # ---- internal ----

    def _encode_batch(self, texts: list[str]) -> torch.Tensor:
        """Encode a list of texts; return [N, D] L2-normalized embeddings."""
        emb = self.bert_model.encode(
            texts,
            convert_to_tensor=True,
            normalize_embeddings=True,    # L2-normalize so cosine = dot
            show_progress_bar=False,
            device=str(self.device),
        )
        return emb

    def _encode_one(self, text: str) -> torch.Tensor:
        """Encode a single text; return [D] L2-normalized embedding."""
        emb = self._encode_batch([text])
        return emb[0]


def load_default_bert(device: torch.device | str = "cpu"):
    """Load the production BERT model (`all-mpnet-base-v2`).

    Returns a ``sentence_transformers.SentenceTransformer``. The model is
    already cached locally per the project setup.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(
        "sentence-transformers/all-mpnet-base-v2",
        device=str(device),
    )
