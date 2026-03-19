"""
MetricsCollector — collects training metrics and saves to JSONL.

Two-tier collection:
  - Basic (every step): loss, ppl, lr, throughput, grad_norm, reg — ~8 floats
  - Full (every N steps): PM/EM state, memory write stats, per-module grad norms,
    activation norms, health checks
"""

import json
import math
import os
import torch
from torch import Tensor
from tqdm import tqdm

from ..model.model import NeuromorphicLM
from ..model.config import ModelConfig


class MetricsCollector:
    def __init__(
        self,
        model: NeuromorphicLM,
        config: ModelConfig,
        output_path: str = "checkpoints/metrics.jsonl",
        collect_every: int = 50,
        basic_every: int = 1,
        phase: str | None = None,
    ):
        self.model = model
        self.config = config
        self.output_path = output_path
        self.collect_every = collect_every
        self.basic_every = basic_every
        self.phase = phase

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        self._file = open(output_path, "a")
        self._writes_since_flush = 0
        self._flush_every = 50  # flush to disk every N writes

        # Health check state
        self._health_warned = set()  # track which warnings have fired

    def should_collect_full(self, step: int) -> bool:
        """Whether this step requires full collection."""
        return self.collect_every > 0 and step % self.collect_every == 0

    def log_basic(self, step: int, loss: float, ppl: float, lr: float,
                  tok_s: float, grad_norm: float, reg: float, elapsed: float,
                  extras: dict = None, mode: str = "train"):
        """Write a basic metrics line (respects basic_every interval)."""
        if self.basic_every > 1 and step % self.basic_every != 0:
            return
        record = {
            "step": step,
            "mode": mode,
            "loss": loss,
            "ppl": ppl,
            "lr": lr,
            "tok_s": tok_s,
            "grad_norm": grad_norm,
            "reg": reg,
            "elapsed": elapsed,
        }
        if extras:
            record.update(extras)
        self._write(record)

    def log_full(self, step: int, basic: dict,
                 extras: dict = None, mode: str = "train"):
        """Write a full metrics line merging basic + memory + grad stats."""
        record = dict(basic)
        record["step"] = step
        record["mode"] = mode
        record["full"] = True
        if extras:
            record.update(extras)

        # Memory subsystem state (read directly from model)
        self._collect_pm_stats(record)
        self._collect_em_stats(record)

        # Memory write diagnostics (from model's _dbg_memory_stats)
        self._collect_memory_write_stats(record)

        # Per-module gradient norms
        self._collect_grad_norms(record)

        # Activation magnitudes at integration point
        self._collect_activation_norms(record)

        # Health checks — print warnings for dead subsystems
        self._check_health(step, record)

        # Lifelong persistence stats (Phase B)
        if self.config.lifelong_mode:
            self._collect_lifelong_stats(record)

        self._write(record)

    def _collect_pm_stats(self, record: dict):
        """Read PM state tensors and compute summary stats.

        v6: Hebbian fast-weight W_pm [BS, B, D_pm, D_pm].
        """
        if not self.config.pm_enabled:
            return
        pm = self.model.pm
        if pm.W_pm is None:
            return
        W = pm.W_pm.detach()  # [BS, B, D_pm, D_pm]
        frob = W.flatten(-2).norm(dim=-1)  # [BS, B]
        record["pm_W_frob_mean"] = frob.mean().item()
        record["pm_W_frob_max"] = frob.max().item()
        record["pm_W_max"] = W.abs().max().item()

    def _collect_em_stats(self, record: dict):
        """Read EM state tensors and compute summary stats."""
        if not self.config.em_enabled:
            return
        em = self.model.em
        if em.em_S is None:
            return
        em_S = em.em_S.detach()  # [BS, B, M]
        record["em_S_mean"] = em_S.mean().item()
        record["em_S_max"] = em_S.max().item()
        record["em_S_sum"] = em_S.sum(dim=-1).mean().item()
        record["em_nonzero"] = (em_S > 0.01).float().mean().item()

    def _collect_memory_write_stats(self, record: dict):
        """Read memory write diagnostics saved by forward_segment.

        These are snapshot values from the last segment in the TBPTT chunk:
        - em_novelty_mean: mean novelty across all tokens/banks
        - em_g_em_mean: mean neuromodulator write gate
        - pm_surprise_gate_mean: mean sigmoid(||surprise||) — PM write strength
        """
        stats = getattr(self.model, "_dbg_memory_stats", None)
        if stats is None:
            return
        for key, val in stats.items():
            record[key] = val

    def _check_health(self, step: int, record: dict):
        """Print warnings to stdout when memory subsystems are dead.

        Only warns once per issue to avoid spam. Checks fire after step 100
        to allow warmup.
        """
        if step < 100:
            return

        # EM dead: no active primitives
        em_nonzero = record.get("em_nonzero")
        if em_nonzero is not None and em_nonzero < 0.01:
            if "em_dead" not in self._health_warned:
                self._health_warned.add("em_dead")
                tqdm.write(
                    f"\n*** HEALTH WARNING (step {step}): EM IS DEAD ***\n"
                    f"    em_nonzero={em_nonzero:.4f} (fraction of slots with S > 0.01)\n"
                    f"    em_S_mean={record.get('em_S_mean', '?')}\n"
                    f"    Likely cause: reset_states zeroing em_S, or writes not keeping up with decay.\n"
                )

        # EM write gate collapsed
        em_g = record.get("em_g_em_mean")
        if em_g is not None and em_g < 1e-3:
            if "em_write_collapsed" not in self._health_warned:
                self._health_warned.add("em_write_collapsed")
                tqdm.write(
                    f"\n*** HEALTH WARNING (step {step}): EM write gate collapsed ***\n"
                    f"    em_g_em_mean={em_g:.6f}\n"
                    f"    Neuromodulator is suppressing all writes.\n"
                )

        # PM signal dead (after sufficient training)
        if step > 1000:
            act_pm = record.get("act_norm_pm")
            act_h = record.get("act_norm_H")
            if act_pm is not None and act_h is not None and act_h > 0:
                ratio = act_pm / act_h
                if ratio < 1e-6:
                    if "pm_signal_dead" not in self._health_warned:
                        self._health_warned.add("pm_signal_dead")
                        tqdm.write(
                            f"\n*** HEALTH WARNING (step {step}): PM signal dead ***\n"
                            f"    pm/H ratio={ratio:.2e}\n"
                            f"    PM is not contributing to the output.\n"
                        )

        # EM recovered — clear warning
        if em_nonzero is not None and em_nonzero > 0.1:
            self._health_warned.discard("em_dead")

    def _collect_lifelong_stats(self, record: dict):
        """Collect cross-document memory persistence stats (Phase B)."""
        pm = self.model.pm
        if pm.W_pm is not None:
            W = pm.W_pm.detach()
            frob = W.flatten(-2).norm(dim=-1)  # [BS, B]
            record["pm_W_frob_lifelong"] = frob.mean().item()
            # Deviation from identity: how much has PM learned?
            D_pm = W.shape[-1]
            eye = torch.eye(D_pm, device=W.device, dtype=W.dtype) * (1.0 / pm.B)
            deviation = (W - eye).flatten(-2).norm(dim=-1)
            record["pm_W_deviation"] = deviation.mean().item()

        em = self.model.em
        if em.em_S is not None:
            em_S = em.em_S.detach()
            em_nonzero = (em_S > 0.01).float().sum().item()
            em_slots = em_S.numel()
            em_budget_total = em_S.sum().item()
            em_budget_cap = em.budget * em_S.shape[0]

            if em_slots > 0:
                record["em_persistence"] = em_nonzero / em_slots
            if em_budget_cap > 0:
                record["em_budget_util"] = em_budget_total / em_budget_cap

    def _collect_grad_norms(self, record: dict):
        """Per-module gradient norms after backward.

        v6 structure: embedding, lm_head, per-layer scan layers, pm, em,
        em_neuromod, W_seed_w, pcm.
        """
        module_groups = {
            "embedding": [self.model.embedding],
            "lm_head": [self.model.lm_head],
        }
        # Per-layer scan layers for depth-resolved diagnostics
        for i, layer in enumerate(self.model.layers):
            module_groups[f"layer_L{i}"] = [layer]
        if self.model.proj_up is not None:
            module_groups["proj_up"] = [self.model.proj_up]
            module_groups["proj_down"] = [self.model.proj_down]
        module_groups["W_seed_w"] = [self.model.W_seed_w]
        if self.model.pcm is not None:
            module_groups["pcm"] = [self.model.pcm]
        if self.config.pm_enabled:
            module_groups["pm"] = [self.model.pm]
        if self.config.em_enabled:
            module_groups["em"] = [self.model.em]
            module_groups["em_neuromod"] = [self.model.em_neuromod]

        for name, modules in module_groups.items():
            total_norm_sq = 0.0
            for mod in modules:
                for p in mod.parameters():
                    if p.grad is not None:
                        total_norm_sq += p.grad.detach().norm().item() ** 2
            record[f"gnorm_{name}"] = math.sqrt(total_norm_sq)

    def _collect_activation_norms(self, record: dict):
        """Read debug activation norms stored by forward_segment."""
        norms = getattr(self.model, "_dbg_act_norms", None)
        if norms is None:
            return
        for key, val in norms.items():
            record[f"act_norm_{key}"] = val

    def _write(self, record: dict):
        """Write a single JSON line."""
        if self.phase is not None and "phase" not in record:
            record = {**record, "phase": self.phase}
        # Convert any remaining non-serializable values
        clean = {}
        for k, v in record.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    clean[k] = None
                else:
                    clean[k] = round(v, 6)
            else:
                clean[k] = v
        self._file.write(json.dumps(clean) + "\n")
        self._writes_since_flush += 1
        if self._writes_since_flush >= self._flush_every:
            self._file.flush()
            self._writes_since_flush = 0

    def log_record(self, record: dict):
        """Write an arbitrary metrics record."""
        self._write(record)

    def close(self):
        """Flush remaining buffered writes and close the output file."""
        self._file.flush()
        self._file.close()
