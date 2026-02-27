"""Find max training batch size and measure peak throughput for each model.

Uses conservative binary search that verifies full train step (fwd+bwd+step).
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import torch
import torch.nn as nn
import gc

DEVICE = torch.device("cuda")
SEQ_LEN = 128
VOCAB = 32000
WARMUP = 5
MEASURE = 20


def try_train_step(create_model_fn, bs):
    """Try a FULL training step (fwd + bwd + optimizer) at given batch size."""
    gc.collect()
    torch.cuda.empty_cache()
    try:
        model, forward_fn = create_model_fn(bs)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        input_ids = torch.randint(0, VOCAB, (bs, SEQ_LEN), device=DEVICE)

        # Full training step
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()

        # Do a second step to check stability
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()

        torch.cuda.synchronize()

        del model, optimizer, input_ids, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
        return True
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower():
            gc.collect()
            torch.cuda.empty_cache()
            return False
        raise


def find_max_bs(name, create_model_fn):
    """Binary search for max training batch size (must survive full train step)."""
    # Find upper bound
    lo, hi = 8, 8
    while hi <= 1024:
        if try_train_step(create_model_fn, hi):
            lo = hi
            hi *= 2
        else:
            break
    hi = min(hi, 1024)

    # Binary search
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if try_train_step(create_model_fn, mid):
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    # Round down to multiple of 8
    best = (best // 8) * 8
    if best == 0:
        best = 8
    print(f"  {name}: max training BS = {best}")
    return best


def bench(name, create_model_fn, bs):
    """Benchmark at given batch size."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model, forward_fn = create_model_fn(bs)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    input_ids = torch.randint(0, VOCAB, (bs, SEQ_LEN), device=DEVICE)

    # Warmup
    for _ in range(WARMUP):
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(MEASURE):
        optimizer.zero_grad()
        logits = forward_fn(model, input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, VOCAB).float(),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        optimizer.step()
        if hasattr(model, 'detach_states'):
            model.detach_states()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    tokens = bs * SEQ_LEN * MEASURE
    tok_s = tokens / elapsed
    ms = elapsed / MEASURE * 1000
    params = sum(p.numel() for p in model.parameters())
    vram = torch.cuda.max_memory_allocated() / 1e9

    print(f"  {name:30s} | {params/1e6:6.1f}M | BS={bs:>3d} | "
          f"{tok_s:>10,.0f} tok/s | {ms:6.1f} ms/step | {vram:.1f} GB")

    del model, optimizer, input_ids
    gc.collect()
    torch.cuda.empty_cache()
    return tok_s


# ---- Model factories ----

def make_pythia_160m(bs):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    cfg = GPTNeoXConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=3072, vocab_size=VOCAB,
        max_position_embeddings=2048, use_cache=False,
    )
    model = GPTNeoXForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd

def make_mamba(bs):
    from mamba_ssm.models.config_mamba import MambaConfig
    from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
    cfg = MambaConfig(d_model=768, n_layer=24, vocab_size=VOCAB)
    model = MambaLMHeadModel(cfg, device=DEVICE, dtype=torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd

def make_gpt2(bs):
    from transformers import GPT2Config, GPT2LMHeadModel
    cfg = GPT2Config(
        n_embd=768, n_layer=12, n_head=12, vocab_size=VOCAB,
        n_positions=1024, use_cache=False,
    )
    model = GPT2LMHeadModel(cfg).to(DEVICE).to(torch.bfloat16)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return m(ids).logits
    return model, fwd

def make_neuromorphic(bs):
    import sys
    sys.path.insert(0, "/home/alex/code/neuromorphic/src")
    from model.config import ModelConfig
    from model.model import NeuromorphicLM
    config = ModelConfig.tier_a(vocab_size=VOCAB)
    model = NeuromorphicLM(config).to(DEVICE).to(torch.bfloat16)
    model.initialize_states(bs, DEVICE)
    def fwd(m, ids):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, aux = m.forward_segment(ids)
            return logits
    return model, fwd


print(f"GPU: {torch.cuda.get_device_name()}")
print(f"seq_len={SEQ_LEN}, warmup={WARMUP}, measure={MEASURE}")
print(f"{'='*95}")

models = [
    ("Pythia-160M", make_pythia_160m),
    ("Mamba-130M", make_mamba),
    ("GPT2-124M", make_gpt2),
    ("Neuromorphic-TierA", make_neuromorphic),
]

print("\n--- Finding max training batch sizes ---")
max_bs = {}
for name, factory in models:
    try:
        bs = find_max_bs(name, factory)
        max_bs[name] = (factory, bs)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {name}: FAILED - {e}")

print(f"\n{'='*95}")
print("--- Benchmarking at max batch size ---\n")

for name, (factory, bs) in max_bs.items():
    try:
        bench(name, factory, bs)
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  {name}: FAILED - {e}")
