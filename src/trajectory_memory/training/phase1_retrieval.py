"""Phase 1 retrieval-pretraining trainer (Wave 1 v4).

Per `docs/wave1_retrieval_pretraining.md`. Each training step:
  1. Reset manifold state.
  2. Sample 8 facts with distinct (entity_class, attribute) keys + pick a target.
  3. Pass each fact's passage through Llama+memory (8 write windows, no loss).
  4. Pass question+answer concatenated as one window (1 read window, loss on
     answer tokens only).
  5. Backward through 9 Llama forwards → memory module gradients.

Loss: teacher-forced per-token cross-entropy on the answer positions only.
Llama is frozen; gradient passes through it without parameter updates.

Companion entry point: `scripts/training/train_wave1_retrieval.py`.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer

from src.trajectory_memory.integrated_lm import IntegratedLM


# ── Telemetry / metrics dataclass ──

@dataclass
class RetrievalMetrics:
    loss: float = 0.0
    answer_token_count: int = 0
    aux_load_balance: float = 0.0
    aux_z_loss: float = 0.0
    grad_norm: float = 0.0
    # Routing diagnostics (over the read+answer window only).
    r_uf: float = 0.0          # read routing uniformity (1 - entropy/max_entropy)
    w_uf: float = 0.0          # write routing uniformity
    r_ent: float = 0.0         # read entropy (over visited concepts)
    w_gn: float = 0.0          # write_module gradient norm
    mem_inject_scale: float = 0.0
    write_logit_scale: float = 0.0
    read_logit_scale: float = 0.0


# ── Sampler ──

class RetrievalSampler:
    """Samples 8-fact chunks from a JSONL fact pool.

    Each fact row must have: entity_class, entity_key, attribute,
    passage_token_ids (list[int]), question_token_ids, answer_token_ids.

    Constraint: in any sampled chunk of 8, all 8 must have distinct
    (entity_class, attribute) keys. Each step also picks a uniform-random
    target index 0..7.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        seed: int = 0,
        chunk_size: int = 8,
    ):
        self.facts = [
            json.loads(line) for line in Path(jsonl_path).read_text().splitlines()
            if line.strip()
        ]
        if not self.facts:
            raise ValueError(f"no facts in {jsonl_path}")
        self.chunk_size = chunk_size
        self.rng = random.Random(seed)
        # Group facts by (entity_class, attribute) for distinctness checking.
        self.by_key: dict[tuple[str, str], list[int]] = defaultdict(list)
        for i, f in enumerate(self.facts):
            self.by_key[(f["entity_class"], f["attribute"])].append(i)
        self.keys = list(self.by_key.keys())
        if len(self.keys) < chunk_size:
            raise ValueError(
                f"only {len(self.keys)} distinct (class,attr) keys; need >= "
                f"{chunk_size}"
            )

    def sample_chunk(self) -> dict:
        """Sample one (8 facts, target_idx) chunk. Returns dict with
        `fact_passages_token_ids` (list of 8 lists), `question_token_ids`,
        `answer_token_ids`, `target_idx`, `target_fact_id`, `metadata`."""
        # Pick 8 distinct keys; for each, sample one fact at that key.
        keys = self.rng.sample(self.keys, self.chunk_size)
        fact_indices = [self.rng.choice(self.by_key[k]) for k in keys]
        facts = [self.facts[i] for i in fact_indices]
        target_idx = self.rng.randrange(self.chunk_size)
        target = facts[target_idx]
        return {
            "fact_passages_token_ids": [f["passage_token_ids"] for f in facts],
            "question_token_ids": target["question_token_ids"],
            "answer_token_ids": target["answer_token_ids"],
            "target_idx": target_idx,
            "target_fact_id": target["fact_id"],
            "metadata": {
                "target_attribute": target["attribute"],
                "target_entity_class": target["entity_class"],
            },
        }

    def sample_batch(self, batch_size: int) -> list[dict]:
        return [self.sample_chunk() for _ in range(batch_size)]


# ── Window-building helpers ──

def _pad_or_trunc(ids: list[int], target_len: int, pad_id: int) -> list[int]:
    if len(ids) >= target_len:
        return ids[:target_len]
    return ids + [pad_id] * (target_len - len(ids))


def _build_qa_window(
    question_ids: list[int],
    answer_ids: list[int],
    T_window: int,
    pad_id: int,
) -> tuple[list[int], list[bool], int, int]:
    """Pack question + answer into a single T_window window.
    Returns (qa_token_ids, answer_mask, answer_start, answer_end)."""
    qa = list(question_ids) + list(answer_ids)
    if len(qa) > T_window:
        qa = qa[:T_window]
    answer_start = min(len(question_ids), T_window)
    answer_end = min(answer_start + len(answer_ids), T_window)
    qa_padded = _pad_or_trunc(qa, T_window, pad_id)
    mask = [False] * T_window
    for i in range(answer_start, answer_end):
        mask[i] = True
    return qa_padded, mask, answer_start, answer_end


# ── Trainer ──

class Phase1RetrievalTrainer:
    """Write-then-retrieve trainer. See module docstring.

    Usage:
        trainer = Phase1RetrievalTrainer(
            model, optimizer,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        for step in range(num_steps):
            batch = sampler.sample_batch(M)
            metrics = trainer.step(batch)
    """

    def __init__(
        self,
        model: IntegratedLM,
        optimizer: Optimizer,
        *,
        pad_token_id: int,
        scheduler: Any | None = None,
        grad_clip: float | None = 1.0,
        load_balance_coef: float = 1e-2,
        z_loss_coef: float = 1e-3,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.pad_token_id = pad_token_id
        self.load_balance_coef = load_balance_coef
        self.z_loss_coef = z_loss_coef
        self._step_count = 0

    @property
    def step_count(self) -> int:
        return self._step_count

    def state_dict(self) -> dict:
        return {"step_count": self._step_count}

    def load_state_dict(self, state: dict) -> None:
        self._step_count = state["step_count"]

    # ── core step ──

    def _build_tensors(
        self, batch: list[dict], device: torch.device,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Convert batch (list of M dicts) into stacked tensors.
        Returns (passages [M, 8, T], qa [M, T], answer_mask [M, T])."""
        cfg = self.model.cfg
        T = cfg.T_window
        M = len(batch)
        passages = torch.empty(M, 8, T, dtype=torch.int64, device=device)
        qa = torch.empty(M, T, dtype=torch.int64, device=device)
        answer_mask = torch.zeros(M, T, dtype=torch.bool, device=device)
        for m, chunk in enumerate(batch):
            for i in range(8):
                ids = _pad_or_trunc(
                    chunk["fact_passages_token_ids"][i], T, self.pad_token_id,
                )
                passages[m, i] = torch.tensor(ids, device=device)
            qa_ids, mask, _, _ = _build_qa_window(
                chunk["question_token_ids"], chunk["answer_token_ids"],
                T, self.pad_token_id,
            )
            qa[m] = torch.tensor(qa_ids, device=device)
            answer_mask[m] = torch.tensor(mask, device=device)
        return passages, qa, answer_mask

    def step(self, batch: list[dict]) -> RetrievalMetrics:
        """One gradient update over a batch of M (fact, Q, A) chunks."""
        cfg = self.model.cfg
        T = cfg.T_window
        M = len(batch)
        device = next(self.model.parameters()).device

        passages, qa, answer_mask = self._build_tensors(batch, device)

        self.optimizer.zero_grad(set_to_none=True)

        # ── Reset manifold state for this batch of chunks.
        prev_state = self.model.manifold.reset_states(batch_size=M)
        prev_hiddens: Tensor | None = None

        # ── 8 write windows (no loss, gradient alive through write_module).
        aux_lb_acc = None
        aux_z_acc = None
        for i in range(8):
            ids = passages[:, i, :]                                # [M, T]
            out = self.model.forward_window(
                lm_input_ids=ids,
                prev_window_hiddens=prev_hiddens,
                prev_states=prev_state,
                target_mask=None,        # no loss; surprise still feeds write_module
                hard_routing=True,       # STE: gradient flows through routing logits
                use_kv_cache=False,
                write_only_grad=True,    # write_module's autograd graph stays alive
            )
            prev_state = out["new_states"]
            prev_hiddens = out["current_hiddens"]
            # Accumulate aux routing losses across windows.
            lb = out.get("aux_load_balance")
            z = out.get("aux_z_loss")
            if lb is not None:
                aux_lb_acc = lb if aux_lb_acc is None else aux_lb_acc + lb
                aux_z_acc = z if aux_z_acc is None else aux_z_acc + z

        # ── Read + answer window (loss on answer tokens only).
        out_qa = self.model.forward_window(
            lm_input_ids=qa,
            prev_window_hiddens=prev_hiddens,
            prev_states=prev_state,
            target_mask=answer_mask,
            hard_routing=True,           # STE: gradient flows through routing logits
            use_kv_cache=False,
        )
        # forward_window returns logits over the whole T window. We compute
        # NTP CE manually on the answer slice (shifted by 1 for prediction).
        logits = out_qa["logits"]                                   # [M, T, V]
        V = logits.shape[-1]
        shift_logits = logits[:, :-1, :]                            # [M, T-1, V]
        shift_targets = qa[:, 1:]                                   # [M, T-1]
        shift_mask = answer_mask[:, 1:]                             # [M, T-1]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V),
            shift_targets.reshape(-1),
            reduction="none",
            ignore_index=-100,  # not used; mask handles selection
        ).reshape(M, T - 1)
        n_answer = shift_mask.float().sum().clamp_min(1.0)
        answer_loss = (per_tok_ce * shift_mask.float()).sum() / n_answer

        # Accumulate aux losses from the answer window too.
        lb_qa = out_qa.get("aux_load_balance")
        z_qa = out_qa.get("aux_z_loss")
        if lb_qa is not None:
            aux_lb_acc = lb_qa if aux_lb_acc is None else aux_lb_acc + lb_qa
            aux_z_acc = z_qa if aux_z_acc is None else aux_z_acc + z_qa
        if aux_lb_acc is not None:
            aux_lb_acc = aux_lb_acc / 9.0                           # 8 writes + 1 read
            aux_z_acc = aux_z_acc / 9.0
            total_loss = (
                answer_loss
                + self.load_balance_coef * aux_lb_acc
                + self.z_loss_coef * aux_z_acc
            )
            aux_lb_val = float(aux_lb_acc.detach())
            aux_z_val = float(aux_z_acc.detach())
        else:
            total_loss = answer_loss
            aux_lb_val = 0.0
            aux_z_val = 0.0

        # Backward.
        total_loss.backward()

        # Grad-norm clipping + measurement.
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if self.grad_clip is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, self.grad_clip)
        else:
            with torch.no_grad():
                grad_norm = torch.norm(
                    torch.stack([p.grad.norm() for p in trainable_params if p.grad is not None])
                )

        # write_module grad norm specifically (for diagnostic).
        wm_grad_sqsum = 0.0
        for p in self.model.write_module.parameters():
            if p.grad is not None:
                wm_grad_sqsum += float(p.grad.detach().pow(2).sum())
        wm_grad_norm = math.sqrt(wm_grad_sqsum)

        # Step + zero.
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self._step_count += 1

        # Telemetry — read final scales + routing uniformity from the read+answer window.
        read_visited = out_qa.get("read_visited")     # [M, J, K]
        write_visited = out_qa.get("write_visited")
        r_uf = _routing_uniformity(read_visited, self.model.cfg.N) if read_visited is not None else 0.0
        w_uf = _routing_uniformity(write_visited, self.model.cfg.N) if write_visited is not None else 0.0
        r_ent = _routing_entropy(read_visited, self.model.cfg.N) if read_visited is not None else 0.0

        return RetrievalMetrics(
            loss=float(answer_loss.detach()),
            answer_token_count=int(n_answer.item()),
            aux_load_balance=aux_lb_val,
            aux_z_loss=aux_z_val,
            grad_norm=float(grad_norm.detach()) if isinstance(grad_norm, Tensor) else float(grad_norm),
            r_uf=r_uf,
            w_uf=w_uf,
            r_ent=r_ent,
            w_gn=wm_grad_norm,
            mem_inject_scale=_mem_inject_scale(self.model),
            read_logit_scale=_logit_scale(self.model.read_module),
            write_logit_scale=_logit_scale(self.model.write_module),
        )

    @torch.no_grad()
    def eval_step(self, batch: list[dict]) -> RetrievalMetrics:
        """Validation step — no backward, no optimizer.step."""
        cfg = self.model.cfg
        T = cfg.T_window
        M = len(batch)
        device = next(self.model.parameters()).device
        passages, qa, answer_mask = self._build_tensors(batch, device)

        prev_state = self.model.manifold.reset_states(batch_size=M)
        prev_hiddens: Tensor | None = None
        for i in range(8):
            out = self.model.forward_window(
                lm_input_ids=passages[:, i, :],
                prev_window_hiddens=prev_hiddens,
                prev_states=prev_state,
                target_mask=None,
                hard_routing=False,
                use_kv_cache=False,
                write_only_grad=False,
            )
            prev_state = out["new_states"]
            prev_hiddens = out["current_hiddens"]

        out_qa = self.model.forward_window(
            lm_input_ids=qa,
            prev_window_hiddens=prev_hiddens,
            prev_states=prev_state,
            target_mask=answer_mask,
            hard_routing=False,
            use_kv_cache=False,
        )
        logits = out_qa["logits"]
        V = logits.shape[-1]
        shift_logits = logits[:, :-1, :]
        shift_targets = qa[:, 1:]
        shift_mask = answer_mask[:, 1:]
        per_tok_ce = F.cross_entropy(
            shift_logits.reshape(-1, V), shift_targets.reshape(-1),
            reduction="none",
        ).reshape(M, T - 1)
        n_answer = shift_mask.float().sum().clamp_min(1.0)
        answer_loss = (per_tok_ce * shift_mask.float()).sum() / n_answer

        return RetrievalMetrics(
            loss=float(answer_loss.detach()),
            answer_token_count=int(n_answer.item()),
        )


# ── Helpers ──

def _routing_uniformity(visited: Tensor, N: int) -> float:
    """1 - (entropy / max_entropy). 0 = perfectly uniform routing, 1 = peaked
    on one concept. Computed across the batch dim."""
    if visited is None:
        return 0.0
    flat = visited.reshape(-1)
    counts = torch.bincount(flat.long(), minlength=N).float()
    p = counts / counts.sum().clamp_min(1.0)
    p_nz = p[p > 0]
    if p_nz.numel() <= 1:
        return 1.0
    H = -(p_nz * p_nz.log()).sum()
    H_max = math.log(N)
    return float((1.0 - H / H_max).clamp(0.0, 1.0))


def _routing_entropy(visited: Tensor, N: int) -> float:
    """Bare entropy of routing distribution (nats)."""
    if visited is None:
        return 0.0
    flat = visited.reshape(-1)
    counts = torch.bincount(flat.long(), minlength=N).float()
    p = counts / counts.sum().clamp_min(1.0)
    p_nz = p[p > 0]
    H = -(p_nz * p_nz.log()).sum()
    return float(H)


def _mem_inject_scale(model: IntegratedLM) -> float:
    """Return scalar mean of mem_inject layer's effective scale."""
    layer = model._mem_inject_layer()
    if layer is None:
        return 0.0
    if hasattr(layer, "scale_raw") and hasattr(layer, "scale_max"):
        eff = float(layer.scale_max) * float(layer.scale_raw.detach().tanh().abs().mean())
        return eff
    return 0.0


def _logit_scale(module: Any) -> float:
    """Read .logit_scale_raw if present (CLIP-style learnable scale)."""
    if module is None or not hasattr(module, "logit_scale_raw"):
        return 0.0
    return float(module.logit_scale_raw.detach().exp().clamp_max(20.0))
