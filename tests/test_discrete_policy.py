"""Unit tests for DiscreteActionPolicy."""

import pytest
import torch

from src.model.discrete_policy import DiscreteActionPolicy


@pytest.fixture
def policy():
    return DiscreteActionPolicy(
        n_cells=8,
        mod_in_dim=64,
        action_dim=1056,
        num_codes=256,
        code_dim=32,
    )


@pytest.fixture
def obs():
    return torch.randn(4, 8, 64)  # [BS=4, NC=8, mod_in=64]


def test_compute_logits_shape(policy, obs):
    logits = policy.compute_logits(obs)
    assert logits.shape == (4, 8, 256)


def test_sample_discrete(policy, obs):
    logits = policy.compute_logits(obs)
    codes, log_pi = policy.sample_discrete(logits)
    assert codes.shape == (4, 8)
    assert codes.dtype == torch.long
    assert codes.min() >= 0 and codes.max() < 256
    assert log_pi.shape == (4, 8)
    # log_pi should be negative (log of probability ≤ 1)
    assert (log_pi <= 0).all()


def test_sample_gumbel_soft(policy, obs):
    logits = policy.compute_logits(obs)
    soft, codes = policy.sample_gumbel_soft(logits, tau=1.0, hard=False)
    assert soft.shape == (4, 8, 256)
    # Simplex — rows sum to 1
    assert torch.allclose(soft.sum(-1), torch.ones(4, 8), atol=1e-4)
    assert codes.shape == (4, 8)

    # Hard Gumbel: argmax matches codes exactly, output still has soft gradient
    soft_hard, codes_hard = policy.sample_gumbel_soft(logits, tau=1.0, hard=True)
    assert (soft_hard.argmax(-1) == codes_hard).all()


def test_decode_shapes(policy, obs):
    logits = policy.compute_logits(obs)
    codes, _ = policy.sample_discrete(logits)
    action = policy.decode(codes)
    assert action.shape == (4, 8, 1056)


def test_decode_soft_differentiable(policy, obs):
    """Gradient from action loss should reach logit_head via soft decode."""
    obs.requires_grad_(True)
    logits = policy.compute_logits(obs)
    soft, _ = policy.sample_gumbel_soft(logits, tau=1.0, hard=False)
    action = policy.decode_soft(soft)
    loss = action.pow(2).mean()
    loss.backward()
    # Gradient should reach logit_head and codebook
    assert policy.logit_w1.grad is not None
    assert policy.logit_w1.grad.abs().max() > 0
    assert policy.codebook.grad is not None


def test_log_prob_matches_sample_discrete(policy, obs):
    """log_prob() called on the same codes returned by sample_discrete() should match."""
    logits = policy.compute_logits(obs)
    codes, lp_sampled = policy.sample_discrete(logits)
    lp_scored = policy.log_prob(logits, codes)
    assert torch.allclose(lp_sampled, lp_scored, atol=1e-5)


def test_entropy(policy, obs):
    """Entropy at uniform logits ≈ log(K). Entropy at extreme logits ≈ 0."""
    uniform = torch.zeros(2, 8, 256)
    H_uniform = policy.entropy(uniform)
    # log(256) = 5.545...
    assert torch.allclose(H_uniform, torch.full_like(H_uniform, torch.log(torch.tensor(256.0))),
                          atol=1e-4)

    extreme = torch.zeros(2, 8, 256)
    extreme[..., 0] = 100.0
    H_extreme = policy.entropy(extreme)
    assert (H_extreme < 1e-4).all()


def test_grpo_loss_gradient(policy, obs):
    """log_prob should carry gradient back to logit_head in phase 2 mode."""
    logits = policy.compute_logits(obs)
    codes, _ = policy.sample_discrete(logits)

    # Re-compute log_prob with gradient enabled (standard GRPO pattern)
    codes = codes.detach()
    logits_new = policy.compute_logits(obs)
    log_pi = policy.log_prob(logits_new, codes)

    # Fake advantage — push codes 0-3 positive, 4-7 negative
    advantage = torch.tensor([1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0])
    advantage = advantage.unsqueeze(0).expand(4, -1)  # [BS, NC]
    loss = -(advantage * log_pi).mean()
    loss.backward()

    # logit_head should have non-zero gradient; codebook+decoder should NOT
    # (those aren't touched in phase 2 gradient path)
    assert policy.logit_w1.grad is not None
    assert policy.logit_w1.grad.abs().max() > 0


def test_usage_tracking(policy, obs):
    logits = policy.compute_logits(obs)
    codes, _ = policy.sample_discrete(logits)
    policy.update_usage(codes)
    assert policy.usage_count.sum() > 0
    assert policy.usage_total > 0


def test_dead_code_reset(policy):
    """Manually make some codes dead, verify reset_dead_codes reinitializes them."""
    policy.usage_count.zero_()
    policy.usage_total.fill_(1000.0)
    # Mark codes 0-9 as "very used", all others as dead
    policy.usage_count[:10] = 100.0
    # Save original dead-code embeddings
    cb_before = policy.codebook.clone()
    n_reset = policy.reset_dead_codes(threshold=0.01)
    assert n_reset == 246  # 256 - 10
    # Embedding for active codes shouldn't change
    assert torch.allclose(cb_before[:10], policy.codebook[:10])
    # Dead codes should have changed
    assert not torch.allclose(cb_before[10:], policy.codebook[10:])


def test_forward_phase1(policy, obs):
    out = policy.forward(obs, phase="phase1", tau=1.0)
    assert "logits" in out
    assert "codes" in out
    assert "action" in out
    assert out["action"].shape == (4, 8, 1056)


def test_forward_phase2(policy, obs):
    out = policy.forward(obs, phase="phase2", tau=1.0)
    assert out["action"].shape == (4, 8, 1056)
    assert out["codes"].dtype == torch.long
