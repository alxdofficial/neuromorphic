"""CPU-only tests for SlotGraph identity re-injection (_restamp_id).

Guards the fix for the accumulation bug: the stamp must measure the scale from the ID-FREE content, not
from the full (id-carrying) norm, so identity does not compound over depth and strength=1.0 = true 50/50.
"""

import math

import torch

from src.memory.models.slotgraph.encoder import SlotGraphEncoder


def _id_frac(nh, idn):
    """|projection onto own id| / ‖row‖  — 1/√2≈0.707 at a true 50/50 id:content energy split."""
    amp = (nh * idn).sum(-1).abs()
    return (amp / nh.norm(dim=-1).clamp_min(1e-9)).mean().item()


def _setup(N=8, d=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    idn = torch.nn.functional.normalize(torch.randn(N, d, generator=g), dim=-1).unsqueeze(0)  # [1,N,d]
    content = torch.randn(1, N, d, generator=g)
    # start from pure content orthogonalized against id so the initial state is unambiguous
    content = content - (content * idn).sum(-1, keepdim=True) * idn
    return idn, content


def test_strength_one_is_50_50():
    idn, content = _setup()
    out = SlotGraphEncoder._restamp_id(content, idn, 1.0)
    assert abs(_id_frac(out, idn) - 1 / math.sqrt(2)) < 1e-4  # equal id/content energy


def test_no_accumulation_over_depth():
    """Repeated stamping with an IDENTITY inter-layer transform must hold id_frac fixed, not grow it.
    The pre-fix formula (scale = full norm) would compound toward id-domination (id_frac → 1)."""
    idn, content = _setup()
    nh = content
    fracs = []
    for _ in range(30):                       # emulate 30 layers, worst case = identity transform between them
        nh = SlotGraphEncoder._restamp_id(nh, idn, 1.0)
        fracs.append(_id_frac(nh, idn))
    assert max(fracs) - min(fracs) < 1e-4                     # stable, no drift
    assert abs(fracs[-1] - 1 / math.sqrt(2)) < 1e-4           # still 50/50 at depth 30 (not → 1.0)


def test_strength_controls_ratio_monotonically():
    idn, content = _setup()
    prev = 0.0
    for s in (0.0, 0.4, 0.7, 1.0, 2.0):
        f = _id_frac(SlotGraphEncoder._restamp_id(content, idn, s), idn)
        assert f >= prev - 1e-6                               # larger strength → more id energy
        # closed form: id_frac = s/√(1+s²)
        assert abs(f - s / math.sqrt(1 + s * s)) < 1e-4
        prev = f


def test_content_direction_preserved():
    """The id-free content subspace must pass through unchanged (only the id coordinate is rewritten)."""
    idn, content = _setup()
    out = SlotGraphEncoder._restamp_id(content, idn, 1.0)
    out_content = out - (out * idn).sum(-1, keepdim=True) * idn
    assert torch.allclose(out_content, content, atol=1e-5)
