#!/usr/bin/env python3
"""Pre-launch smoke for LoRA-all (every arm + both vanillas get the same ~1.70M
rank-16 q/v LoRA on the frozen Llama-1B).

Checks: (1) LoRA param count ≈ 1.70M; (2) vanilla arm becomes TRAINABLE (n_trainable
> 0 → not eval-only); (3) per-variant Llama isolation — building two LoRA models
sequentially (each self-loads) works, while SHARING one Llama across two LoRA
decoders fails (the contamination we avoid); (4) LoRA grads flow; (5) the checkpoint
save filter persists lora_ params. Run: python scripts/repr_learning/smoke_lora_all.py
"""
import sys, tempfile
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.decoder import FrozenLlamaDecoder, load_frozen_llama, apply_lora_to_llama

results = []
def check(name, cond, extra=""):
    ok = bool(cond); results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + extra) if extra else ''}")
    return ok

# Operative config (matched sizing) + LoRA-all ON.
cfg = ReprConfig(
    d_node_state=128, n_edges=68, n_flat_codes=192,
    d_continuous=1432, d_concept_baseline=1432, d_mt_value=1432, d_recurrent=1432,
    d_enc=816, enc_n_layers=4, enc_n_heads=12, enc_ffn_dim=3264,
    d_mamba=1256, edge_token_packing="fused",
    use_llama_lora=True, llama_lora_rank=16, llama_lora_alpha=16,
)
dev = "cuda" if torch.cuda.is_available() else "cpu"

def trainable_breakdown(model):
    lora = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    other = sum(p.numel() for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n)
    return lora, other

print("== build vanilla_llama with LoRA (self-loads fresh Llama) ==")
vanilla = ReprLearningModel(cfg, variant="vanilla_llama", llama_model=None).to(dev)
v_lora, v_other = trainable_breakdown(vanilla)
v_total = vanilla.n_trainable_params()
print(f"  vanilla trainable: lora={v_lora:,}  non-lora={v_other:,}  total={v_total:,}")
check("LoRA ≈ 1.70M (rank-16 q/v × 16 layers)", 1_600_000 <= v_lora <= 1_800_000, f"{v_lora:,}")
check("vanilla is now TRAINABLE (n_trainable > 0 → not eval-only)", v_total > 0)
check("vanilla trainable is LoRA-dominated (no big encoder)", v_lora >= v_other,
      f"lora {v_lora:,} ≥ non-lora {v_other:,}")

print("== isolation: build flat AFTER vanilla in the SAME process (must not double-wrap) ==")
ok_seq = True
try:
    flat = ReprLearningModel(cfg, variant="flat_baseline", llama_model=None).to(dev)
    f_lora, f_other = trainable_breakdown(flat)
    print(f"  flat trainable: lora={f_lora:,}  non-lora(memory)={f_other:,}")
    check("flat LoRA present ≈ 1.70M", 1_600_000 <= f_lora <= 1_800_000, f"{f_lora:,}")
    check("flat memory params ≈ 48.6M", 47e6 <= f_other <= 50e6, f"{f_other:,}")
except Exception as e:
    ok_seq = False
    print(f"  ERROR building flat after vanilla: {type(e).__name__}: {e}")
check("sequential LoRA builds isolated (no double-wrap)", ok_seq)

print("== contamination demo: SHARING one Llama silently reuses ONE adapter (why we self-load) ==")
# apply_lora_to_llama only wraps nn.Linear; on the 2nd decoder the q/v are already
# LoRALinear, so it SILENTLY skips re-wrapping → both decoders share d1's adapter.
# That silent cross-variant contamination is exactly what self-loading avoids.
shared = load_frozen_llama(cfg.llama_model)[0]
d1 = FrozenLlamaDecoder(cfg, llama_model=shared)
d2 = FrozenLlamaDecoder(cfg, llama_model=shared)
def first_lora_A(dec):
    for n, p in dec.named_parameters():
        if n.endswith("lora_A"):
            return p
    return None
shared_adapter = first_lora_A(d1) is first_lora_A(d2)
check("shared-Llama path silently shares ONE adapter (self-load per variant prevents this)",
      shared_adapter, "d1 and d2 lora_A are the same tensor" if shared_adapter else "")
# and confirm the self-load path gives DISTINCT adapters
sa = first_lora_A(vanilla); sb = first_lora_A(flat)
check("self-loaded variants get DISTINCT adapters", sa is not None and sb is not None and sa is not sb)

print("== LoRA grad flows through a real Llama forward ==")
ids = torch.randint(0, 32000, (1, 16), device=dev)
out = vanilla.decoder.llama(input_ids=ids)
loss = out.logits.float().pow(2).mean()
loss.backward()
lora_grads = [(n, p.grad) for n, p in vanilla.named_parameters() if "lora_" in n and p.requires_grad]
n_with_grad = sum(1 for _, g in lora_grads if g is not None and g.abs().sum() > 0)
check("LoRA params receive gradient", n_with_grad > 0, f"{n_with_grad}/{len(lora_grads)} lora tensors")

print("== checkpoint save filter keeps lora_ (under decoder.llama.) ==")
sd = vanilla.state_dict()
def keep(k):  # mirror save_checkpoint.keep
    if not k.startswith("decoder.llama."):
        return True
    return "lora_" in k
saved = {k: v for k, v in sd.items() if keep(k)}
saved_lora = [k for k in saved if "lora_" in k]
dropped_llama = [k for k in sd if k.startswith("decoder.llama.") and "lora_" not in k]
check("lora_ keys survive the save filter", len(saved_lora) > 0, f"{len(saved_lora)} lora tensors saved")
check("frozen base-Llama weights are dropped (not saved)", len(dropped_llama) > 0,
      f"{len(dropped_llama)} base tensors filtered out")
# round-trip
with tempfile.NamedTemporaryFile(suffix=".pt", delete=True) as fp:
    torch.save(saved, fp.name)
    reloaded = torch.load(fp.name, map_location="cpu", weights_only=False)
    res = vanilla.load_state_dict(reloaded, strict=False)
    bad = [k for k in (res.missing_keys + res.unexpected_keys) if "llama" not in k.lower()]
    check("checkpoint round-trips (only frozen-Llama keys 'missing')", len(bad) == 0,
          f"non-llama mismatches: {bad[:3]}")

nfail = sum(1 for _, ok in results if not ok)
print(f"\n{'ALL PASS' if nfail == 0 else f'{nfail}/{len(results)} FAILED'}")
sys.exit(1 if nfail else 0)
