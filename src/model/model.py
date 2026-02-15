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

        # Tie embedding and LM head weights (shared [vocab_size, D] matrix)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight

        # Input projection: D -> D (split across blocks after)
        self.W_in = nn.Linear(config.D, config.D, bias=False)

        # Spatial decoder (hierarchical aggregation + deep cross-attention)
        self.spatial_decoder = SpatialDecoder(config) if config.snapshot_enabled else None
        self.use_spatial_decoder = config.snapshot_enabled  # can be toggled off for fast inference

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
        snapshot = self.use_spatial_decoder
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

    def forward_span(self, input_ids: Tensor,
                     reset_mask_first: Tensor,
                     collect: bool = False) -> tuple:
        """Process P tokens in parallel (one plasticity span).

        Surprise is frozen at the span-initial value for all P tokens.
        PM/EM are read-only within the span (same as sequential path).

        Args:
            input_ids: [BS, P] — token IDs for one span
            reset_mask_first: [BS] bool — reset mask for the first token
                (subsequent resets derived from eot_id in input_ids)
            collect: if True, 4th return element is gate_stats dict

        Returns:
            logits_all: [BS, P, vocab]
            x_emb_all: [BS, P, D]
            y_wm_all: [BS, P, D]
            (if collect: gate_stats: {block_idx: {layer_idx: stats}})
        """
        BS, P = input_ids.shape
        device = input_ids.device

        # Initialize surprise on first call
        if self.surprise is None:
            self.surprise = torch.zeros(BS, 1, device=device)

        # 1. Compute per-token reset masks
        reset_mask_all = self._compute_reset_masks(
            input_ids, reset_mask_first
        )  # [BS, P]

        # Respect reset_on_doc_boundary
        if not self.config.reset_on_doc_boundary:
            reset_mask_all = torch.zeros_like(reset_mask_all)

        # 2. Handle reset for first token (state resets happen here)
        if reset_mask_all[:, 0].any():
            self.reset_at_doc_boundary(reset_mask_all[:, 0])

        # Handle mid-span resets (tokens 1..P-1): reset states
        # We need to process these before the forward pass. The sequential
        # path resets at the start of each token. For the parallel path,
        # mid-span resets only affect the carry mask (which zeros h at
        # boundaries). Block/EM/PM state resets for mid-span boundaries
        # are handled by the carry mask zeroing out h, which is equivalent
        # to what reset_at_doc_boundary does to Layer.h.
        # WM handles resets internally in forward_span.

        # 3. Freeze surprise for this span
        surprise_span = self.surprise  # [BS, 1]

        # 4. Carry mask: [BS, P, 1]
        carry_all = (~reset_mask_all).float().unsqueeze(-1)

        # 5. Embed all tokens
        x_emb_all = self.embedding(input_ids)  # [BS, P, D]

        # 6. Working memory
        if self.config.wm_enabled:
            y_wm_all = self.wm.forward_span(
                x_emb_all, reset_mask_all
            )  # [BS, P, D]
        else:
            y_wm_all = torch.zeros_like(x_emb_all)

        # 7. Project and split across blocks
        x_proj_all = self.W_in(x_emb_all)  # [BS, P, D]
        x_blocks_all = x_proj_all.view(
            BS, P, self.config.B, self.config.D_h
        )  # [BS, P, B, D_h]

        # 8. Process each block
        h_blocks = []
        gate_stats = {} if collect else None
        for b, block in enumerate(self.blocks):
            result = block.forward_span(
                x_blocks_all[:, :, b],  # [BS, P, D_h]
                y_wm_all, x_emb_all, surprise_span, carry_all,
                collect=collect,
            )
            if collect:
                h_b, bstats = result
                gate_stats[b] = bstats
            else:
                h_b = result
            h_blocks.append(h_b)

        # 9. Merge block outputs
        h_final = torch.cat(h_blocks, dim=-1)  # [BS, P, D]

        # 10. Spatial decoder or direct LM head
        if self.config.snapshot_enabled and self.spatial_decoder is not None:
            # Collect per-layer outputs from blocks (already cached)
            # Each: [BS, P, L, D_h]
            block_layer_outputs = [block._last_layer_stack for block in self.blocks]

            # PM/EM summaries are frozen within span — compute once
            pm_summary = self._compute_pm_summary(BS, device)   # [BS, D_h]
            em_summary = self._compute_em_summary(BS, device)    # [BS, D_em]

            # Reshape everything to [BS*P, ...] for decoder
            L = self.config.L
            D_h = self.config.D_h
            D = self.config.D
            BP = BS * P
            # [BS, P, L, D_h] -> [BS*P, L, D_h]
            block_layer_flat = [blo.reshape(BP, L, D_h) for blo in block_layer_outputs]
            pm_flat = pm_summary.unsqueeze(1).expand(BS, P, -1).reshape(BP, D_h)
            em_flat = em_summary.unsqueeze(1).expand(BS, P, -1).reshape(BP, -1)
            wm_flat = y_wm_all.reshape(BP, D)
            h_flat = h_final.reshape(BP, D)

            # Run decoder on all BS*P positions at once
            h_decoded = self.spatial_decoder(
                block_layer_flat, pm_flat, em_flat, wm_flat, h_flat
            )  # [BS*P, D]

            logits_all = self.lm_head(h_decoded.reshape(BS, P, D))  # [BS, P, vocab]
        else:
            logits_all = self.lm_head(h_final)  # [BS, P, vocab]

        if collect:
            return logits_all, x_emb_all, y_wm_all, gate_stats
        return logits_all, x_emb_all, y_wm_all

    def _compute_reset_masks(self, input_ids: Tensor,
                             reset_mask_first: Tensor) -> Tensor:
        """Derive per-token reset masks from input_ids.

        Token t gets reset if:
        - t == 0 and reset_mask_first is True, OR
        - t > 0 and input_ids[:, t-1] == eot_id

        Args:
            input_ids: [BS, P]
            reset_mask_first: [BS] bool

        Returns:
            reset_mask_all: [BS, P] bool
        """
        BS, P = input_ids.shape
        eot_id = self.config.eot_id
        device = input_ids.device

        reset_mask_all = torch.zeros(BS, P, dtype=torch.bool, device=device)
        reset_mask_all[:, 0] = reset_mask_first

        if P > 1:
            # Token t resets if previous token (t-1) was eot
            reset_mask_all[:, 1:] = (input_ids[:, :-1] == eot_id)

        return reset_mask_all

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

    @torch.inference_mode()
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

    def compile_for_training(self):
        """Apply torch.compile to performance-critical submodules.

        Compiles the inner _core methods (no lazy init, no .item(), no
        data-dependent branches) with fullgraph=True for maximum fusion.
        The outer wrappers handle lazy init without compilation.

        WM is NOT compiled because forward_span has a data-dependent dispatch
        (reset_mask_all[:, 1:].any()) that would cause graph breaks.
        """
        for block in self.blocks:
            for layer in block.layers:
                layer._forward_span_core = torch.compile(
                    layer._forward_span_core, fullgraph=True,
                )
                layer.pm._update_eligibility_core = torch.compile(
                    layer.pm._update_eligibility_core, fullgraph=True,
                )

    def compile_for_inference(self, batch_size: int = 1,
                              use_spatial_decoder: bool = False):
        """Apply torch.compile to the single-token inference path.

        Compiles layer.step and pm.apply/update_eligibility for each layer.
        A warmup forward pass triggers all lazy inits before compilation.

        WM is NOT compiled (data-dependent reset_mask.any() branch).

        Args:
            batch_size: BS for warmup — must match generation BS.
            use_spatial_decoder: if False, skip the spatial decoder during
                inference for ~15% speedup (default: off for inference).
        """
        self.use_spatial_decoder = use_spatial_decoder
        device = next(self.parameters()).device

        # Warmup: trigger all lazy inits
        with torch.inference_mode():
            dummy_ids = torch.zeros(batch_size, dtype=torch.long, device=device)
            dummy_reset = torch.ones(batch_size, dtype=torch.bool, device=device)
            self.forward_one_token(dummy_ids, dummy_reset)

        # Compile per-layer methods
        for block in self.blocks:
            for layer in block.layers:
                layer.step = torch.compile(layer.step)
                layer.pm.apply = torch.compile(layer.pm.apply)
                layer.pm.update_eligibility = torch.compile(
                    layer.pm.update_eligibility
                )

    def param_count(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
