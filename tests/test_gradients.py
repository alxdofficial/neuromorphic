"""Gradient flow tests (v4) — NEVER change.

If these fail, there's a dead gradient bug.
"""

import torch
import torch.nn.functional as F
import pytest
from tests.conftest import make_tiny_config

from src.model.model import NeuromorphicLM


BS = 2
VOCAB = 64


def _compute_loss(model, N=None):
    """Forward one segment and compute CE loss. Returns loss scalar."""
    if N is None:
        N = model.config.N
    input_ids = torch.randint(0, VOCAB, (BS, N))
    target_ids = torch.randint(0, VOCAB, (BS, N))
    reset_mask = torch.zeros(BS, dtype=torch.bool)

    model.initialize_states(BS, torch.device("cpu"))
    logits, aux_loss = model.forward_segment(input_ids, reset_mask)
    ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
    return ce_loss + aux_loss


class TestGradientFlow:
    def test_all_params_get_gradient(self):
        """Every parameter should receive a gradient from a single forward+loss."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        # decay_head only affects em_S state evolution (detached between segments),
        # so gradient doesn't flow through the loss in a single segment
        expected_no_grad = {"decay_head"}

        no_grad = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is None:
                if not any(skip in name for skip in expected_no_grad):
                    no_grad.append(name)

        assert len(no_grad) == 0, f"Parameters with no gradient: {no_grad}"

    def test_nonzero_gradients(self):
        """Gradients should be nonzero for key parameters."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        # Check key structural parameters
        for name in ["embedding.weight", "fan_in.weight",
                      "lm_head.weight", "lambda_logit"]:
            parts = name.split(".")
            obj = model
            for p in parts:
                obj = getattr(obj, p)
            assert obj.grad is not None, f"No gradient for {name}"
            assert obj.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_gradient_with_pcm(self):
        """PCM parameters should get gradients when enabled."""
        cfg = make_tiny_config(pcm_enabled=True)
        model = NeuromorphicLM(cfg)

        # Two passes needed: first pass generates z_hat, second generates loss
        model.initialize_states(BS, torch.device("cpu"))

        N = cfg.N
        total_loss = torch.tensor(0.0)

        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        # Check PCM encoder gets gradient
        pcm = model.columns.pcm
        assert pcm is not None
        assert pcm.encoder.weight.grad is not None
        assert pcm.encoder.weight.grad.abs().sum() > 0

    def test_gradient_through_pm_read(self):
        """PM read should be differentiable."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        # Run 2 segments so PM has content from first
        model.initialize_states(BS, torch.device("cpu"))
        N = cfg.N
        total_loss = torch.tensor(0.0)

        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        # PM post-processing weights should get gradient
        col = model.columns
        assert col.W_post_fused.weight.grad is not None

    def test_gradient_through_em_read(self):
        """EM read should be differentiable."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        model.initialize_states(BS, torch.device("cpu"))
        N = cfg.N
        total_loss = torch.tensor(0.0)

        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        col = model.columns
        assert col.W_post_fused.weight.grad is not None

    def test_lambda_logit_gradient(self):
        """Damped mixing parameter should get gradient."""
        cfg = make_tiny_config(R=3)  # Need R>1 for damped mixing
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.lambda_logit.grad is not None
        assert model.lambda_logit.grad.abs() > 0

    def test_lateral_mixer_gradient(self):
        """LateralMixer mix param should receive gradients (zero-init residual)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        lateral = model.columns.lateral
        assert lateral.mix.grad is not None
        assert lateral.mix.grad.abs().sum() > 0

    def test_cross_block_mixer_gradient(self):
        """CrossBlockMixer cross_mix param should receive gradients."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        cross_block = model.columns.cross_block
        assert cross_block.cross_mix.grad is not None
        assert cross_block.cross_mix.grad.abs().sum() > 0

    def test_stacked_ffn_gradient(self):
        """All FFN layers in ffn_pre and ffn_post should receive gradients."""
        cfg = make_tiny_config(ffn_depth=3)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        for stack_name in ("ffn_pre", "ffn_post"):
            stack = getattr(model.columns, stack_name)
            for i in range(3):
                up_w = stack.ups[i].weight
                down_w = stack.downs[i].weight
                assert up_w.grad is not None, f"No gradient for {stack_name}.ups[{i}]"
                assert up_w.grad.abs().sum() > 0, f"Zero gradient for {stack_name}.ups[{i}]"
                assert down_w.grad is not None, f"No gradient for {stack_name}.downs[{i}]"
                assert down_w.grad.abs().sum() > 0, f"Zero gradient for {stack_name}.downs[{i}]"

    def test_proj_up_down_gradient(self):
        """proj_up/proj_down should receive gradients when D_embed != D."""
        cfg = make_tiny_config(D=64, D_embed=32)
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.proj_up.weight.grad is not None
        assert model.proj_up.weight.grad.abs().sum() > 0
        assert model.proj_down.weight.grad is not None
        assert model.proj_down.weight.grad.abs().sum() > 0

    def test_position_attention_gradient(self):
        """PositionAttention Q/K/V/out_proj should all receive nonzero gradients.

        out_proj is zero-init, so Q/K/V get zero gradient at exact init (gradient
        blocked by zero weight). We perturb out_proj slightly to simulate the state
        after a few training steps, then verify gradient flows through all projections.
        """
        cfg = make_tiny_config()  # pos_attn enabled by default (D_col//4 = 8)
        assert cfg.position_attn_dim > 0
        model = NeuromorphicLM(cfg)

        # Perturb out_proj so gradient flows through Q/K/V
        pa = model.columns.pos_attn
        assert pa is not None
        with torch.no_grad():
            pa.out_proj.weight.add_(torch.randn_like(pa.out_proj.weight) * 0.01)

        loss = _compute_loss(model)
        loss.backward()

        for name in ["q_proj", "k_proj", "v_proj", "out_proj"]:
            proj = getattr(pa, name)
            assert proj.weight.grad is not None, f"No gradient for pos_attn.{name}"
            assert proj.weight.grad.abs().sum() > 0, f"Zero gradient for pos_attn.{name}"

    def test_fan_in_gradient(self):
        """fan_in (zero-init D_col->D skip projection) should receive gradients."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        assert model.fan_in.weight.grad is not None
        assert model.fan_in.weight.grad.abs().sum() > 0
        assert model.fan_in.bias.grad is not None

    def test_skip_connection_gradient(self):
        """Gradient should flow through the skip connection path."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        loss = _compute_loss(model)
        loss.backward()

        # embedding gets gradient through both the skip path (x_input)
        # and the column processing path (via fan_in)
        assert model.embedding.weight.grad is not None
        assert model.embedding.weight.grad.abs().sum() > 0

    def test_gradient_through_read_sliced(self):
        """read_sliced should be differentiable (PM + EM)."""
        cfg = make_tiny_config()
        model = NeuromorphicLM(cfg)

        # Run 2 segments so PM/EM have content from first
        model.initialize_states(BS, torch.device("cpu"))
        N = cfg.N
        total_loss = torch.tensor(0.0)

        for seg in range(2):
            input_ids = torch.randint(0, VOCAB, (BS, N))
            target_ids = torch.randint(0, VOCAB, (BS, N))
            reset_mask = torch.zeros(BS, dtype=torch.bool)
            logits, aux_loss = model.forward_segment(input_ids, reset_mask)
            ce_loss = F.cross_entropy(logits.reshape(-1, VOCAB), target_ids.reshape(-1))
            total_loss = total_loss + ce_loss + aux_loss

        total_loss.backward()

        # Column W_post_fused should get gradient through read_sliced path
        col = model.columns
        assert col.W_post_fused.weight.grad is not None
        assert col.W_post_fused.weight.grad.abs().sum() > 0
