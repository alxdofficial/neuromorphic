"""
NeuromorphicLM — top-level model module.

Combines embedding, working memory, B parallel blocks, and LM head.
Processes one token at a time (online, no [BS, T, vocab] materialization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ModelConfig
from .working_memory import WorkingMemory
from .block import Block
from .decoder import SpatialDecoder
from .utils import StateMixin


class NeuromorphicLM(nn.Module, StateMixin):
    _state_tensor_names = ["surprise"]

    def __init__(self, config: ModelConfig):
        super().__init__()
        config.validate()
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.D)
        self.wm = WorkingMemory(config)
        self.blocks = nn.ModuleList([
            Block(config, b) for b in range(config.B)
        ])
        self.lm_head = nn.Linear(config.D, config.vocab_size, bias=False)

        # Input projection: D -> D (split across blocks after)
        self.W_in = nn.Linear(config.D, config.D, bias=False)

        # Spatial decoder (hierarchical aggregation + deep cross-attention)
        self.spatial_decoder = SpatialDecoder(config) if config.snapshot_enabled else None

        # Surprise signal (lazily initialized)
        self.surprise: Tensor = None

        # Initialize weights
        self._init_weights()

        # Small-init the decoder output_proj after global init so it starts
        # near-identity (h_final + small_noise) while still allowing gradient flow.
        # Zero-init would kill ALL gradients through the decoder (chain rule: grad × 0 = 0).
        if self.spatial_decoder is not None:
            nn.init.normal_(self.spatial_decoder.output_proj.weight, std=0.01)

    def _init_weights(self):
        """Initialize weights with small values for stability."""
        for name, p in self.named_parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)

    def forward_one_token(self, input_id: Tensor, reset_mask: Tensor,
                          collect: bool = False):
        """Process one token through the full model.

        Args:
            input_id: [BS] — single token IDs
            reset_mask: [BS] bool — True for streams at doc boundary
            collect: bool — if True, return (logits, x_emb, y_wm, stats_dict)

        Returns:
            logits: [BS, vocab]
            x_emb: [BS, D] — token embedding
            y_wm: [BS, D] — working memory output
            stats: dict (only when collect=True) — per-block, per-layer gate stats
        """
        BS = input_id.shape[0]
        device = input_id.device

        # Initialize surprise on first call
        if self.surprise is None:
            self.surprise = torch.zeros(BS, 1, device=device)

        # Respect reset_on_doc_boundary across all reset-related paths.
        if self.config.reset_on_doc_boundary:
            effective_reset_mask = reset_mask
        else:
            effective_reset_mask = torch.zeros_like(reset_mask)

        # Reset states for masked streams when doc-boundary reset is enabled.
        if effective_reset_mask.any():
            self.reset_at_doc_boundary(effective_reset_mask)

        # Embed token
        x = self.embedding(input_id)  # [BS, D]

        # Working memory
        if self.config.wm_enabled:
            y_wm = self.wm.step(x, effective_reset_mask)  # [BS, D]
        else:
            y_wm = torch.zeros_like(x)

        # Split input across blocks
        x_proj = self.W_in(x)  # [BS, D]
        x_blocks = x_proj.view(BS, self.config.B, self.config.D_h)  # [BS, B, D_h]

        # Carry mask: 0 at doc boundaries, 1 otherwise
        carry = (~effective_reset_mask).float().unsqueeze(-1)  # [BS, 1]

        # Process each block
        snapshot = self.config.snapshot_enabled
        h_blocks = []
        block_layer_outputs = [] if snapshot else None
        block_stats = {}
        for b, block in enumerate(self.blocks):
            result = block.step(x_blocks[:, b], y_wm, x, self.surprise, carry,
                                collect=collect, return_layers=snapshot)
            if collect and snapshot:
                h_b, bstats, layers_b = result
                block_stats[b] = bstats
                block_layer_outputs.append(layers_b)
            elif collect:
                h_b, bstats = result
                block_stats[b] = bstats
            elif snapshot:
                h_b, layers_b = result
                block_layer_outputs.append(layers_b)
            else:
                h_b = result
            h_blocks.append(h_b)

        # Merge block outputs
        h_final = torch.cat(h_blocks, dim=-1)  # [BS, D]

        # Spatial decoder or direct LM head
        if snapshot:
            pm_summary = self._compute_pm_summary(BS, device)
            em_summary = self._compute_em_summary(BS, device)
            h_decoded = self.spatial_decoder(
                block_layer_outputs, pm_summary, em_summary, y_wm, h_final,
            )
            logits = self.lm_head(h_decoded)  # [BS, vocab]
        else:
            logits = self.lm_head(h_final)  # [BS, vocab]

        if collect:
            return logits, x, y_wm, block_stats
        return logits, x, y_wm

    def _compute_pm_summary(self, BS: int, device: torch.device) -> Tensor:
        """Strength-weighted readout of PM slots, averaged across all instances.

        Returns: [BS, D_h] — zero vector if PM is disabled or uninitialized.
        """
        if not self.config.pm_enabled:
            return torch.zeros(BS, self.config.D_h, device=device)
        readouts = []
        for block in self.blocks:
            for layer in block.layers:
                pm = layer.pm
                if pm.pm_a is not None and pm.pm_V is not None:
                    weights = pm.pm_a.unsqueeze(-1)  # [BS, r, 1]
                    denom = pm.pm_a.sum(dim=1, keepdim=True) + 1e-8  # [BS, 1]
                    readout = (weights * pm.pm_V).sum(dim=1) / denom  # [BS, D_h]
                    readouts.append(readout)
        if readouts:
            return torch.stack(readouts, dim=0).mean(dim=0)  # [BS, D_h]
        return torch.zeros(BS, self.config.D_h, device=device)

    def _compute_em_summary(self, BS: int, device: torch.device) -> Tensor:
        """Strength-weighted readout of EM slots, averaged across all instances.

        Returns: [BS, D_em] — zero vector if EM is disabled or uninitialized.
        """
        if not self.config.em_enabled:
            return torch.zeros(BS, self.config.D_em, device=device)
        readouts = []
        for block in self.blocks:
            em = block.em
            if em.em_S is not None and em.em_V is not None:
                weights = em.em_S.unsqueeze(-1)  # [BS, M, 1]
                denom = em.em_S.sum(dim=1, keepdim=True) + 1e-8  # [BS, 1]
                readout = (weights * em.em_V).sum(dim=1) / denom  # [BS, D_em]
                readouts.append(readout)
        if readouts:
            return torch.stack(readouts, dim=0).mean(dim=0)  # [BS, D_em]
        return torch.zeros(BS, self.config.D_em, device=device)

    def update_surprise(self, logits: Tensor, target: Tensor, mask: Tensor = None):
        """Update surprise signal: -log p(target).

        During training, target is the ground-truth next token (teacher forcing).
        During inference, target is the model's own sampled/chosen token,
        giving an equivalent self-supervised surprise signal with identical scale.

        Args:
            logits: [BS, vocab] — model output
            target: [BS] — target token ids (ground truth or sampled)
            mask: [BS] bool — optional update mask; masked-out streams get 0
        """
        with torch.no_grad():
            logp = F.log_softmax(logits, dim=-1)
            next_surprise = -logp.gather(-1, target.unsqueeze(-1))  # [BS, 1]
            if mask is not None:
                if mask.dtype is not torch.bool:
                    mask = mask.bool()
                next_surprise = next_surprise * mask.unsqueeze(-1).float()
            self.surprise = next_surprise

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 0,
                 top_p: float = 1.0) -> Tensor:
        """Autoregressive generation with self-supervised surprise.

        Processes the prompt token-by-token (updating all memory systems),
        then generates new tokens. Surprise is computed from the model's
        own predictions: -log p(sampled_token), identical in form to
        training surprise but without requiring ground truth.

        Args:
            prompt_ids: [BS, T_prompt] — prompt token ids (must have T_prompt >= 1)
            max_new_tokens: number of tokens to generate (0 = just process prompt)
            temperature: sampling temperature (1.0 = unchanged)
            top_k: if > 0, keep only top-k logits before sampling
            top_p: nucleus sampling threshold (1.0 = disabled)

        Returns:
            generated: [BS, T_prompt + max_new_tokens] — full sequence
        """
        BS, T_prompt = prompt_ids.shape
        device = prompt_ids.device
        eot_id = self.config.eot_id

        if T_prompt == 0:
            raise ValueError("prompt_ids must have at least 1 token")

        generated = [prompt_ids]

        # --- Process prompt ---
        # First token always resets (doc start); subsequent tokens check for EOT
        reset_mask = torch.ones(BS, dtype=torch.bool, device=device)
        for t in range(T_prompt):
            logits, _, _ = self.forward_one_token(prompt_ids[:, t], reset_mask)

            # Next token's reset: True if current token is EOT
            if t < T_prompt - 1:
                is_eot = (prompt_ids[:, t] == eot_id)
                loss_mask = ~is_eot if self.config.reset_on_doc_boundary else None
                self.update_surprise(logits, prompt_ids[:, t + 1], mask=loss_mask)
                reset_mask = is_eot if self.config.reset_on_doc_boundary else \
                    torch.zeros(BS, dtype=torch.bool, device=device)

            elif max_new_tokens > 0:
                # Last prompt token: sample first generated token
                is_eot = (prompt_ids[:, t] == eot_id)
                loss_mask = ~is_eot if self.config.reset_on_doc_boundary else None
                next_token = self._sample(logits, temperature, top_k, top_p)
                self.update_surprise(logits, next_token, mask=loss_mask)
                generated.append(next_token.unsqueeze(1))
                reset_mask = is_eot if self.config.reset_on_doc_boundary else \
                    torch.zeros(BS, dtype=torch.bool, device=device)

        # --- Generate remaining tokens ---
        for i in range(max_new_tokens - 1):
            prev_token = generated[-1].squeeze(1)
            logits, _, _ = self.forward_one_token(prev_token, reset_mask)
            next_token = self._sample(logits, temperature, top_k, top_p)

            # EOT handling: mask surprise, set reset for next step
            is_eot = (prev_token == eot_id)
            loss_mask = ~is_eot if self.config.reset_on_doc_boundary else None
            self.update_surprise(logits, next_token, mask=loss_mask)
            reset_mask = is_eot if self.config.reset_on_doc_boundary else \
                torch.zeros(BS, dtype=torch.bool, device=device)

            generated.append(next_token.unsqueeze(1))

        return torch.cat(generated, dim=1)

    @staticmethod
    def _sample(logits: Tensor, temperature: float = 1.0,
                top_k: int = 0, top_p: float = 1.0) -> Tensor:
        """Sample from logits with temperature, top-k, and nucleus filtering."""
        logits = logits / max(temperature, 1e-8)

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            threshold = logits.topk(top_k, dim=-1).values[:, -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            probs_sorted = sorted_logits.softmax(dim=-1)
            cumulative_probs = probs_sorted.cumsum(dim=-1)
            # Keep the first token that pushes cumulative above top_p,
            # remove everything after
            remove_mask = (cumulative_probs - probs_sorted) >= top_p
            sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))
            logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def commit_at_boundary(self, force_mode: str = "normal",
                           span_surprise: Tensor = None):
        """Called every P tokens. Triggers PM commits + EM writes.

        Args:
            force_mode: "normal" — use controller decisions
                        "force_on" — commit all streams
                        "force_off" — skip all commits
            span_surprise: [BS] — mean surprise over span (for PM controller)
        """
        commit_info = {}
        for b_idx, block in enumerate(self.blocks):
            if self.config.pm_enabled:
                commit_info[b_idx] = block.commit_pm(
                    force_mode=force_mode, span_surprise=span_surprise
                )
            # EM writes are handled by the trainer (needs candidate buffers)
        return commit_info

    def reset_at_doc_boundary(self, mask: Tensor):
        """Per-stream reset of all memory states.

        Args:
            mask: [BS] bool — True for streams to reset
        """
        # Reset surprise for masked streams
        if self.surprise is not None:
            self.surprise = self.surprise * (~mask).float().unsqueeze(-1)

        # Reset all blocks
        for block in self.blocks:
            block.reset_states(mask)

        # WM resets are handled internally by wm.step()

    def detach_states(self):
        """TBPTT boundary: detach all recurrent states."""
        if self.surprise is not None:
            self.surprise = self.surprise.detach()

        self.wm.detach_states()
        for block in self.blocks:
            block.detach_states()

    def rl_parameters(self):
        """Yield neuromodulator MLP parameters (for separate RL optimizer)."""
        for name, param in self.named_parameters():
            if "neuromodulator" in name:
                yield param

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
