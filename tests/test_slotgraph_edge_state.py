"""CPU-only tests for SlotGraph's relation/confidence edge recurrence."""

import torch
import torch.nn as nn

from src.memory.models.slotgraph.encoder import SlotGraphEncoder, _confidence_update


def _zero_linear(in_features, out_features):
    layer = nn.Linear(in_features, out_features)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def _small_commit_cell(d=3, de=3):
    """Construct only the modules used by _commit, without loading a language model."""
    enc = SlotGraphEncoder.__new__(SlotGraphEncoder)
    nn.Module.__init__(enc)
    enc.N = 1
    enc.de = de
    enc.phi_ln = nn.Identity()

    enc.gate_a_i = _zero_linear(d, de)
    enc.gate_a_j = _zero_linear(d, de)
    enc.gate_b_i = _zero_linear(d, de)
    enc.gate_b_j = _zero_linear(d, de)
    enc.gate_a_bias = nn.Parameter(torch.full((de,), 2.0))
    enc.gate_b_bias = nn.Parameter(torch.full((de,), 1.5))
    enc.rd_k = nn.Linear(de, de, bias=False)
    nn.init.eye_(enc.rd_k.weight)
    enc.edge_norm = nn.LayerNorm(de, elementwise_affine=False)

    enc.conf_gate_a_i = _zero_linear(d, 1)
    enc.conf_gate_a_j = _zero_linear(d, 1)
    enc.conf_gate_b_i = _zero_linear(d, 1)
    enc.conf_gate_b_j = _zero_linear(d, 1)
    enc.conf_gate_a_bias = nn.Parameter(torch.tensor(3.0))
    enc.conf_gate_b_bias = nn.Parameter(torch.tensor(1.5))
    enc.conf_gate_a_obs = nn.Parameter(torch.tensor(-1.0))
    enc.conf_gate_b_obs = nn.Parameter(torch.tensor(1.0))
    return enc


def test_confidence_update_is_bounded_and_decays_without_evidence():
    conf = torch.tensor([0.8], requires_grad=True)
    obs = torch.tensor([0.0], requires_grad=True)
    retain = torch.tensor([0.5], requires_grad=True)
    write = torch.tensor([0.9], requires_grad=True)

    updated = _confidence_update(conf, obs, retain, write)
    assert torch.allclose(updated, torch.tensor([0.4]))

    full_obs = _confidence_update(torch.zeros(1), torch.ones(1), torch.ones(1), torch.tensor([0.25]))
    assert torch.allclose(full_obs, torch.tensor([0.25]))
    assert 0.0 <= float(updated.detach()) <= 1.0

    updated.backward()
    for value in (conf, obs, retain, write):
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()


def test_commit_separates_relation_direction_from_input_dependent_strength():
    enc = _small_commit_cell()
    batch = 2
    relation = torch.zeros(batch, 1, 1, 3)
    confidence = torch.zeros(batch, 1, 1, 1)
    node_h = torch.tensor([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]])
    semantic_obs = torch.tensor([[[[1.0, 0.0, -1.0]]], [[[0.0, 1.0, -1.0]]]])
    strength_obs = torch.tensor([[[[0.2]]], [[[0.8]]]])
    active = torch.ones(batch, dtype=torch.bool)

    new_relation, new_confidence = enc._commit(
        node_h, relation, confidence, node_h, semantic_obs, strength_obs,
        active, relation, confidence,
    )

    assert torch.allclose(new_relation.norm(dim=-1), torch.ones(batch, 1, 1), atol=1e-6)
    assert not torch.allclose(new_relation[0], new_relation[1])
    assert new_confidence[1].item() > new_confidence[0].item() > 0.0
    assert torch.allclose((new_confidence * new_relation).norm(dim=-1),
                          new_confidence.squeeze(-1), atol=1e-6)


def test_commit_freezes_inactive_examples_exactly():
    enc = _small_commit_cell()
    relation = torch.nn.functional.normalize(torch.randn(2, 1, 1, 3), dim=-1)
    confidence = torch.tensor([[[[0.3]]], [[[0.7]]]])
    node_h = torch.randn(2, 1, 3)
    semantic_obs = torch.randn(2, 1, 1, 3)
    strength_obs = torch.full((2, 1, 1, 1), 0.9)
    active = torch.tensor([True, False])

    new_relation, new_confidence = enc._commit(
        node_h, relation, confidence, node_h, semantic_obs, strength_obs,
        active, relation, confidence,
    )

    assert torch.equal(new_relation[1], relation[1])
    assert torch.equal(new_confidence[1], confidence[1])
    assert not torch.equal(new_confidence[0], confidence[0])
