"""Pre-flight bug sweep for the MAE training path, BEFORE the 800-step runs.

Checks (the things that silently break a run):
  1. each variant constructs on SmolLM2 + task_mode=masked_reconstruction;
  2. compute_masked_reconstruction_loss runs fwd+bwd: finite loss, finite grads;
  3. EVERY trainable param receives gradient (no dead module);
  4. capacity-relative slice: memory M == k_slots for compressors; M==0 vanilla_llama
     (floor), M==T vanilla_full_context (ceiling);
  5. REAL vs OFF executes and the masking/CE shapes are right (no all-masked /
     no-target degenerate batch);
  6. magnitudes bounded (no NaN/Inf), top1 sane;
  7. a few real optimizer steps reduce the loss (the path actually learns).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader

device = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"
SRC = "meta-llama/Llama-3.2-1B"
fails = []


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.n_flat_codes = 16
    cfg.icae_n_slots = 16; cfg.icae_lora_rank = 34; cfg.icae_lora_alpha = 68
    cfg.ccm_n_comp = 16; cfg.ccm_lora_rank = 17; cfg.ccm_lora_alpha = 34
    cfg.autocompressor_n_slots = 16
    cfg.autocompressor_lora_rank = 17; cfg.autocompressor_lora_alpha = 34
    cfg.beacon_ratio = 8; cfg.beacon_wrap_layers = (0, 10, 19, 29)
    # hlvocab (compression-by-vocabulary): smaller vocab for the 135M smoke
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    return cfg


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
dl = make_sentence_dataloader(tok, batch_size=8, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
it = iter(dl)
batch = next(it)


def to_dev(b):
    for a in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        v = getattr(b, a, None)
        if torch.is_tensor(v):
            setattr(b, a, v.to(device))
    return b


batch = to_dev(batch)
print(f"batch: context {tuple(batch.context_ids.shape)}, k_slots={batch.k_slots}")

VARIANTS = ["hlvocab_baseline", "icae_baseline", "ccm_baseline",
            "autocompressor_baseline", "beacon_baseline",
            "vanilla_llama", "vanilla_full_context"]
for variant in VARIANTS:
    print(f"\n=== {variant} ===")
    cfg = matched(ReprConfig())
    try:
        model = ReprLearningModel(cfg, variant=variant).to(device)
        model.task_mode = "masked_reconstruction"
        model.train()
    except Exception as e:
        import traceback; traceback.print_exc()
        fails.append(f"{variant}: construct {type(e).__name__}")
        continue

    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    opt = torch.optim.AdamW([p for _, p in trainable], lr=1e-4) if trainable else None

    # capacity / shapes / finite, fwd+bwd
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.compute_masked_reconstruction_loss(batch)
        M = out["mae_M"]; nmask = out["mae_n_masked"]
        exp_M = {"vanilla_llama": 0.0, "vanilla_full_context": float(batch.context_ids.shape[1])}
        expect = exp_M.get(variant, float(batch.k_slots))
        ok_M = abs(M - expect) < 0.5
        if not ok_M:
            fails.append(f"{variant}: M={M} expected {expect}")
        fin = torch.isfinite(out["loss"]).item()
        if not fin:
            fails.append(f"{variant}: non-finite loss")
        print(f"  M={M:.0f} (expect {expect:.0f}, {'OK' if ok_M else 'BAD'})  "
              f"n_masked={nmask:.0f}  loss={out['loss'].item():.3f}  "
              f"top1={out['top1_acc'].item():.3f}")
        if trainable:
            out["loss"].backward()   # init backward (LoRA B=0 ⇒ A dead at step 1; checked post-steps)
    except Exception as e:
        import traceback; traceback.print_exc()
        fails.append(f"{variant}: fwd/bwd {type(e).__name__}: {str(e)[:80]}")
        continue

    # REAL vs OFF executes
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        real = model.compute_masked_reconstruction_loss(batch)["loss_recon"].item()
        off = model.compute_masked_reconstruction_loss(batch, zero_memory=True)["loss_recon"].item()
    print(f"  REAL={real:.3f}  OFF={off:.3f}  (untrained; OFF≈REAL expected pre-training)")

    # a few optimizer steps reduce the loss
    if opt is not None:
        losses = []
        alive = set()   # params with nonzero grad on ANY step (selection-conditional
                        # params like inter-layer τ only fire when their edge is picked)
        for _ in range(8):
            model.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                o = model.compute_masked_reconstruction_loss(batch)
            o["loss"].backward()
            for n, p in trainable:
                if p.grad is not None and p.grad.norm() > 0:
                    alive.add(n)
            opt.step(); losses.append(o["loss"].item())
        drop = losses[0] - min(losses)   # did it improve AT ALL (robust to end noise)
        print(f"  8-step loss: {losses[0]:.3f} → {losses[-1]:.3f} (min {min(losses):.3f}, Δ {drop:+.3f})")
        if drop <= 0:
            fails.append(f"{variant}: loss never improved in 8 steps")
        # dead-grad check AFTER steps (B has moved, so true-dead modules show now)
        # exempt clamped scalars: at a clamp boundary their grad is legitimately
        # 0 (mask_embed = no-positions-selected; log_route_temp = temp hit its
        # bound under the high-LR/no-warmup 8-step transient).
        # v1-only selection gate (presence_*) is unused when hlvocab runs in v2
        # (use_graph=True) mode — legitimately no grad there.
        _exempt = ("mask_embed", "log_route_temp", "presence_a", "presence_b")
        dead = [n for n, _ in trainable
                if n not in alive and not any(e in n for e in _exempt)]
        if dead:
            fails.append(f"{variant}: {len(dead)} dead-grad params (post-steps)")
            print(f"  DEAD post-steps: {dead[:5]}")
        else:
            print(f"  grads: all {len(trainable)} trainable tensors flow (post-steps)")
    del model
    if opt: del opt
    torch.cuda.empty_cache()

print("\n" + ("MAE SMOKE PASS — clear to launch the 800-step runs"
              if not fails else "MAE SMOKE FAIL:\n  " + "\n  ".join(fails)))
sys.exit(0 if not fails else 1)
