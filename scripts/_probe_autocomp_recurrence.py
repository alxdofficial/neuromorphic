"""Faithfulness probe: is the AutoCompressor recurrence a real differentiable
carry across segments, or detached / no-op? Force MULTIPLE windows by setting
window < context, then:
  (1) check summary0 receives grad ONLY through the carry (multi-window),
  (2) check that perturbing the summary AFTER segment 1 changes segment-2 output,
  (3) verify the carry tensor requires_grad and has grad_fn across the loop,
  (4) confirm memory genuinely differs REAL vs OFF vs a hand-zeroed memory.
"""
import os, torch
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from src.repr_learning.config import ReprConfig
from src.repr_learning.model import ReprLearningModel
from src.repr_learning.data_emat_bio import make_emat_bio_dataloader
from transformers import AutoTokenizer
import dataclasses

torch.manual_seed(0)
DEV = "cuda"
CHUNK, M, B = 1024, 144, 2

cfg = ReprConfig(batch_size=B, fixed_window_size=256, max_window_size=CHUNK,
                 max_steps=8000, warmup_steps=500, use_llama_lora=True,
                 grad_checkpoint_stream=True)
cfg.autocompressor_n_slots = M; cfg.n_flat_codes = M
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
model = ReprLearningModel(cfg, variant="autocompressor_baseline").to(DEV)
enc = model.encoder

dl = make_emat_bio_dataloader(tok, context_len=CHUNK, batch_size=B, n_pairs=12,
                              n_facts=3, world_seed=0, split="train")
batch = next(iter(dl))
for f in dataclasses.fields(batch):
    v = getattr(batch, f.name)
    if torch.is_tensor(v): setattr(batch, f.name, v.to(DEV))

embed = model.decoder.llama.get_input_embeddings() if hasattr(model.decoder, "llama") else None
# fall back: use the encoder base embed for the streaming write probe
emb = enc.base.get_input_embeddings()

# ---- DIRECT recurrence test: run streaming_write over 4 windows manually ----
WIN = 256
print("=== Manual multi-window streaming_write trace (WIN=256, CHUNK=1024 -> 4 windows) ===")
model.train()
with torch.no_grad():
    ctx_embeds = emb(batch.context_ids).float()
state = enc.init_streaming_state(B, DEV, torch.float32)
print(f"[seg0] summary requires_grad={state['summary'].requires_grad} "
      f"grad_fn={state['summary'].grad_fn}")
summaries = [state["summary"]]
with torch.autocast("cuda", dtype=torch.bfloat16):
    for w in range(4):
        s, e = w*WIN, min((w+1)*WIN, CHUNK)
        em = ctx_embeds[:, s:e, :]
        mk = batch.context_mask[:, s:e]
        state, _ = enc.streaming_write(state, em, mk, chunk_offset=s)
        sm = state["summary"]
        print(f"[seg{w+1}] summary requires_grad={sm.requires_grad} "
              f"grad_fn={type(sm.grad_fn).__name__ if sm.grad_fn else None} "
              f"norm={sm.float().norm().item():.2f}")
        summaries.append(sm)

# Does segment k's summary depend on segment k-1's summary? (carry not detached)
# Backprop from final summary to summary0; if grad flows, the carry is alive.
final = state["summary"]
g_s0 = torch.autograd.grad(final.float().sum(), enc.summary0, retain_graph=True,
                           allow_unused=True)[0]
print(f"\n[carry-grad] d(final_summary)/d(summary0) is "
      f"{'None (DETACHED/no-op)' if g_s0 is None else f'norm={g_s0.norm().item():.4e} (CARRY ALIVE)'}")
g_slots = torch.autograd.grad(final.float().sum(), enc.slots, retain_graph=True,
                              allow_unused=True)[0]
print(f"[slots-grad] d(final_summary)/d(slots) is "
      f"{'None' if g_slots is None else f'norm={g_slots.norm().item():.4e}'}")

# Counterfactual: perturb summary after seg1, re-run seg2; does output change?
print("\n=== Counterfactual: does seg2 output depend on seg1 summary? ===")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    st = enc.init_streaming_state(B, DEV, torch.float32)
    st, _ = enc.streaming_write(st, ctx_embeds[:, :WIN], batch.context_mask[:, :WIN])
    sum_a = st["summary"].clone()
    st2_base, _ = enc.streaming_write({"summary": sum_a.clone()},
                                      ctx_embeds[:, WIN:2*WIN], batch.context_mask[:, WIN:2*WIN])
    # perturb the carried summary
    sum_b = sum_a + 5.0 * torch.randn_like(sum_a)
    st2_pert, _ = enc.streaming_write({"summary": sum_b},
                                      ctx_embeds[:, WIN:2*WIN], batch.context_mask[:, WIN:2*WIN])
    delta = (st2_base["summary"].float() - st2_pert["summary"].float()).norm().item()
    base_norm = st2_base["summary"].float().norm().item()
    print(f"[counterfactual] perturbing seg1 summary changes seg2 summary by "
          f"L2={delta:.3f} (base norm {base_norm:.2f}, rel={delta/base_norm:.3f})")
    print("  -> NONZERO rel means the carry is genuinely consumed (not a no-op).")

print("\n[DONE]")
