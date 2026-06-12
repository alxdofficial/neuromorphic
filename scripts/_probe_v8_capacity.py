"""Capacity-doom probe for graph_v8 at d_mem=64.

Uses the ACTUAL hrr_bind (unitary phase-only key) from graph_substrate_v8.
We model the SUBSTRATE's value-write semantics: into one node-slot we accumulate
a SUPERPOSITION of self-binds bound_i = unit(K_i (*) V_i). The reader fetches a
value vector; the question's job is whether a specific V can be recovered.

Two recovery models are probed because the v8 read does NOT do an explicit HRR
unbind of the slot value -- it just attention-averages V_l and the decoder must
read content out. So we test:
  (model-bind)  recover V_j from slot by HRR-unbind with K_j: corr(K_j^-1 (*) slot, V_j)
  (model-read)  recover the j-th value's content by routing: how distinguishable
                are the K_j-addressed reads when many self-binds share a slot.

We report cos(recovered, true) and the crosstalk floor (cos to a random unrelated
value) for K = 1,2,3,5,8 superposed pairs, d_mem in {64, 3072}.
"""
import math
import torch
import torch.nn.functional as F
import sys
sys.path.insert(0, ".")
from src.repr_learning.graph_substrate_v8 import hrr_bind, _unit

torch.manual_seed(0)
device = "cpu"


def hrr_unbind(key, slot):
    """Unbind by the unitary key: irfft( conj(phase(K)) * rfft(slot) )."""
    dim = key.shape[-1]
    ks = torch.fft.rfft(key.float(), n=dim, dim=-1)
    ks = ks / ks.abs().clamp_min(1e-6)          # phase-only (same as bind)
    ss = torch.fft.rfft(slot.float(), n=dim, dim=-1)
    return torch.fft.irfft(ks.conj() * ss, n=dim, dim=-1)


def run(d, Ks, trials=200):
    print(f"\n=== d_mem = {d} ===")
    print(f"{'K':>3} {'recov_cos_mean':>14} {'recov_cos_std':>13} "
          f"{'floor_mean':>11} {'frac>0.3':>9} {'SNR':>6}")
    for K in Ks:
        recov = []
        floors = []
        for _ in range(trials):
            # K random unit (key,value) pairs, keys projected unitary inside bind
            keys = _unit(torch.randn(K, d))
            vals = _unit(torch.randn(K, d))
            binds = _unit(hrr_bind(keys, vals))          # [K,d] self-binds, unit-norm
            # substrate slot = convex-ish sum of self-binds (here: plain sum, the
            # worst case for crosstalk; substrate uses coact-weighted partner sum).
            slot = binds.sum(0)                            # superpose all K
            # recover value 0 by unbinding with key 0
            rec = hrr_unbind(keys[0], slot)
            c = F.cosine_similarity(rec.unsqueeze(0), vals[0].unsqueeze(0)).item()
            recov.append(c)
            # crosstalk floor: cos(recovered, an UNRELATED random value)
            other = _unit(torch.randn(d))
            floors.append(F.cosine_similarity(rec.unsqueeze(0), other.unsqueeze(0)).item())
        recov = torch.tensor(recov)
        floors = torch.tensor(floors).abs()
        frac = (recov > 0.3).float().mean().item()
        snr = recov.mean().item() / (floors.mean().item() + 1e-9)
        print(f"{K:>3} {recov.mean().item():>14.3f} {recov.std().item():>13.3f} "
              f"{floors.mean().item():>11.3f} {frac:>9.2f} {snr:>6.1f}")


run(64, [1, 2, 3, 5, 8, 12])
run(3072, [1, 2, 3, 5, 8, 12])

# Read-distinguishability model: with K self-binds in one slot, can K-addressed
# routing tell value_j apart? Cos between unbind(K_j, slot) recovered value and
# EACH true value (target should win).
print("\n=== read separation @ d=64: recovered-by-K_j vs all true values (K=3) ===")
torch.manual_seed(1)
d, K = 64, 3
keys = _unit(torch.randn(K, d)); vals = _unit(torch.randn(K, d))
slot = _unit(hrr_bind(keys, vals)).sum(0)
for j in range(K):
    rec = hrr_unbind(keys[j], slot)
    cos = F.cosine_similarity(rec.unsqueeze(0), vals, dim=-1)
    print(f"  unbind K_{j}: cos to V_0,V_1,V_2 = "
          f"[{cos[0]:.3f} {cos[1]:.3f} {cos[2]:.3f}]  argmax={cos.argmax().item()} (want {j})")
