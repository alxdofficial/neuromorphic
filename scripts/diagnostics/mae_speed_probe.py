"""Training-speed + max-batch-size probe for the ACTIVE masked_reconstruction
(MAE) line, at the capacity-matched ranks. Mirrors mae_smoke.py's data/config so
the numbers reflect a real run. For each variant it sweeps batch size up to OOM
and reports peak VRAM, ms/step, and throughput (samples/s) — per the "bench at
each path's own optimal BS" convention.

Run: python scripts/diagnostics/mae_speed_probe.py
"""
import sys, os, time, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader

dev = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"
BS_LIST = [8, 16, 32, 64, 128, 256]
VARIANTS = ["hlvocab_baseline", "icae_baseline", "ccm_baseline",
            "autocompressor_baseline", "beacon_baseline",
            "vanilla_llama", "vanilla_full_context"]
_TF = ("context_ids", "context_mask", "question_ids", "question_mask",
       "answer_ids", "answer_mask", "answer_content_mask")


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.n_flat_codes = 16
    cfg.icae_n_slots = 16; cfg.icae_lora_rank = 60; cfg.icae_lora_alpha = 120
    cfg.ccm_n_comp = 16; cfg.ccm_lora_rank = 30; cfg.ccm_lora_alpha = 60
    cfg.autocompressor_n_slots = 16
    cfg.autocompressor_lora_rank = 30; cfg.autocompressor_lora_alpha = 60
    cfg.beacon_ratio = 8; cfg.beacon_wrap_layers = (0, 6, 12, 17, 23, 29)
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    return cfg


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
# one big batch we slice down (fixed T across BS → consistent throughput compare)
dl = make_sentence_dataloader(tok, batch_size=max(BS_LIST), src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
full = next(iter(dl))
for f in _TF:                                   # move to device in place (keeps k_slots dynamic attr)
    v = getattr(full, f, None)
    if torch.is_tensor(v):
        setattr(full, f, v.to(dev))
T = full.context_ids.shape[1]
total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
print(f"MAE speed probe — backbone={BACKBONE} d=576, sentence T={T}, "
      f"k_slots={getattr(full, 'k_slots', None)}, GPU={total_gb:.0f}GB\n")


def slice_batch(bs):
    b = copy.copy(full)                          # shallow copy preserves k_slots
    for f in _TF:
        v = getattr(full, f, None)
        if torch.is_tensor(v):
            setattr(b, f, v[:bs])
    return b


rows = {}
for variant in VARIANTS:
    cfg = matched(ReprConfig())
    model = ReprLearningModel(cfg, variant=variant).to(dev)
    model.task_mode = "masked_reconstruction"
    if model.n_trainable_params() == 0:
        print(f"[{variant:24}] eval-only (no trainable) — skipped"); del model
        torch.cuda.empty_cache(); continue
    model.train()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    best = None
    for bs in BS_LIST:
        try:
            batch = slice_batch(bs)
            torch.cuda.reset_peak_memory_stats()
            for _ in range(2):                      # warmup
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.compute_loss(batch, window_size=T)
                out["loss"].backward(); opt.step()
            torch.cuda.synchronize(); t0 = time.time()
            for _ in range(5):
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model.compute_loss(batch, window_size=T)
                out["loss"].backward(); opt.step()
            torch.cuda.synchronize()
            dt = (time.time() - t0) / 5
            peak = torch.cuda.max_memory_allocated() / 2**30
            best = (bs, peak, dt, bs / dt)
            print(f"[{variant:24}] BS={bs:4d}  peak={peak:5.1f}GB  "
                  f"{dt*1e3:6.0f}ms/step  {bs/dt:7.1f} samp/s")
        except torch.cuda.OutOfMemoryError:
            print(f"[{variant:24}] BS={bs:4d}  OOM"); torch.cuda.empty_cache(); break
    rows[variant] = best
    del model, opt; torch.cuda.empty_cache()

print(f"\n==== MAE max-fitting BS per arm ({total_gb:.0f}GB GPU) ====")
print(f"  {'arm':<24}{'maxBS':>7}{'peakGB':>8}{'ms/step':>9}{'samp/s':>9}")
for v, b in rows.items():
    if b: print(f"  {v:<24}{b[0]:>7}{b[1]:>8.1f}{b[2]*1e3:>9.0f}{b[3]:>9.1f}")
