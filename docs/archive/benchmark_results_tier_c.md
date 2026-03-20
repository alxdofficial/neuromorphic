# Tier C Throughput Benchmark Results

**Date**: 2026-03-07
**GPU**: NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB VRAM, 102.0 GB reported)
**Platform**: RunPod, Ubuntu, Python 3.12.3
**PyTorch**: 2.8.0+cu128, CUDA 12.8
**Precision**: bf16 mixed precision (`torch.autocast(dtype=torch.bfloat16)`, fp32 weights)
**Benchmark**: 5 warmup + 20 timed steps per config, random data (`torch.randint`), no I/O overhead
**Script**: `scripts/benchmark_throughput.py --tiers C`

---

## Summary Table

| Model | Params | T=1024 tok/s | T=2048 tok/s | BS@1K | BS@2K | VRAM@1K | VRAM@2K | ms/step@2K | TFLOPS@2K | Compile | GradCkpt | 1.5B tok @2K |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **neuromorphic-c** | 898M | 24,200 | 23,900 | 8 | 4 | 63.5 GB | 63.3 GB | 342.5 ms | 129.5 | yes | no | 17.4h |
| **pythia-1b** | 937M | 35,500 | 27,700 | 64 | 64 | 32.8 GB | 54.3 GB | 4733.0 ms | 155.7 | yes | yes | 15.0h |
| **tinyllama-1b** | 1,100M | 30,400 | 25,000 | 64 | 48 | 39.5 GB | 59.1 GB | 3928.8 ms | 165.1 | yes | yes | 16.7h |

---

## Key Observations

### Throughput (tok/s)
- At T=2048, all three models are in a similar range: 23.9K - 27.7K tok/s.
- Pythia-1b is fastest at 27.7K tok/s (T=2048), ~16% faster than neuromorphic-c.
- **Neuromorphic throughput is nearly constant across T=1024 and T=2048** (24.2K vs 23.9K, only 1.2% drop). This is the O(T) linear scan advantage — no quadratic attention cost.
- **Transformers degrade significantly from T=1024 to T=2048**: Pythia drops 22% (35.5K → 27.7K), TinyLlama drops 18% (30.4K → 25.0K). This gap would widen further at T=4096+.

### Batch Size
- Neuromorphic fits much smaller batches: BS=8 at T=1024, BS=4 at T=2048. This is because it stores runtime memory state (PM fast-weight matrices, EM key/value banks) on GPU — these scale with BS, not with T.
- Transformers with gradient checkpointing fit BS=64 at T=1024 and BS=48-64 at T=2048 on 96GB VRAM. Grad checkpointing trades compute for memory, enabling large batches.
- Despite smaller BS, neuromorphic achieves competitive tok/s because each step processes fewer tokens but completes faster (342ms vs 4733ms per step for Pythia).

### VRAM
- Neuromorphic uses the most VRAM: 63.3 GB at T=2048 with BS=4. Per-sample memory cost is high due to PM/EM state.
- Pythia-1b: 54.3 GB at T=2048 with BS=64. Gradient checkpointing dramatically reduces per-sample memory.
- TinyLlama-1b: 59.1 GB at T=2048 with BS=48.
- On a 24GB card (RTX 4090), neuromorphic Tier C would likely only fit BS=1 or not at all. Tier C is designed for 48GB+ GPUs.

### TFLOPS
- Estimated using `6 * N_params * tok/s / 1e12` (standard approximation for fwd+bwd+optimizer).
- Neuromorphic: 129.5 TFLOPS — lower than baselines because smaller batch size means less parallelism for GPU tensor cores.
- Pythia: 155.7 TFLOPS, TinyLlama: 165.1 TFLOPS — better GPU utilization due to larger batch sizes.
- MFU (Model FLOPs Utilization) shows `---` because the Blackwell GPU is not yet in our lookup table (its peak bf16 TFLOPS spec is not confirmed).

### Sequence Length Scaling
- **This is the key structural advantage of neuromorphic / linear recurrence**: throughput barely changes with T.
- Transformers: attention is O(T^2), so doubling T roughly halves throughput even with grad checkpointing.
- At T=4096 or T=8192 (not tested), the gap would be much larger. Transformers would need even more aggressive checkpointing or simply OOM, while neuromorphic would stay at ~24K tok/s.

---

## Cost Extrapolation (1.5B tokens, T=2048)

| Model | Hours | Relative |
|---|---|---|
| pythia-1b | 15.0h | 1.00x (fastest) |
| tinyllama-1b | 16.7h | 1.11x |
| neuromorphic-c | 17.4h | 1.16x |

All three models can train 1.5B tokens in under 18 hours on a single Blackwell GPU. The neuromorphic model is only 16% slower than Pythia despite storing and updating runtime memory state every step.

For a 10B token run: multiply by ~6.7x → Pythia ~100h, TinyLlama ~112h, Neuromorphic ~116h.

---

## Detailed Per-Run Data

### neuromorphic-c @ T=1024
- Params: 898,042,233 (898.0M)
- Batch size: 8 (auto-detected, max that fits)
- tok/s: 24,200
- Step time: 338.4 ± 2.1 ms
- Peak VRAM: 63.5 GB
- TFLOPS: 130.4
- torch.compile: yes
- Gradient checkpointing: no

### neuromorphic-c @ T=2048
- Params: 902,236,537 (902.2M) — slightly more params due to N=2048 vs N=1024 changing some buffer sizes
- Batch size: 4 (auto-detected)
- tok/s: 23,900
- Step time: 342.5 ± 1.6 ms
- Peak VRAM: 63.3 GB
- TFLOPS: 129.5
- torch.compile: yes
- Gradient checkpointing: no

### pythia-1b @ T=1024
- Params: 936,808,448 (936.8M)
- Batch size: 64 (auto-detected)
- tok/s: 35,500
- Step time: 1847.7 ± 4.8 ms
- Peak VRAM: 32.8 GB
- TFLOPS: 199.4
- torch.compile: yes
- Gradient checkpointing: yes (required for GPT-NeoX at large BS)

### pythia-1b @ T=2048
- Params: 936,808,448 (936.8M)
- Batch size: 64 (auto-detected)
- tok/s: 27,700
- Step time: 4733.0 ± 6.5 ms
- Peak VRAM: 54.3 GB
- TFLOPS: 155.7
- torch.compile: yes
- Gradient checkpointing: yes

### tinyllama-1b @ T=1024
- Params: 1,100,048,384 (1,100.0M)
- Batch size: 64 (auto-detected)
- tok/s: 30,400
- Step time: 2157.5 ± 8.2 ms
- Peak VRAM: 39.5 GB
- TFLOPS: 200.5
- torch.compile: yes
- Gradient checkpointing: yes (required for LLaMA at large BS)

### tinyllama-1b @ T=2048
- Params: 1,100,048,384 (1,100.0M)
- Batch size: 48 (auto-detected)
- tok/s: 25,000
- Step time: 3928.8 ± 16.8 ms
- Peak VRAM: 59.1 GB
- TFLOPS: 165.1
- torch.compile: yes
- Gradient checkpointing: yes

---

## Model Configurations

### neuromorphic-c (Tier C)
- Architecture: Dense linear scan + PM (Hebbian fast-weight) + EM (episodic memory) + PCM
- D=4096, D_embed=2048, L_scan=16, d_inner=2048, B=8, C=16, D_pm=64
- M=768 (EM primitives), n_trail_steps=3
- No gradient checkpointing (not needed — O(T) memory)
- No dropout
- Vocab: 32,000 (TinyLlama tokenizer)

### pythia-1b (GPT-NeoX)
- Architecture: Transformer (GPT-NeoX with rotary embeddings, parallel residual)
- hidden_size=2048, num_layers=16, num_heads=8, intermediate_size=8192
- max_position_embeddings=2048, rotary_pct=0.25
- Gradient checkpointing: enabled (O(T^2) attention OOMs without it)
- Vocab: 32,000

### tinyllama-1b (LLaMA)
- Architecture: Transformer (LLaMA with GQA, SiLU activation, RMSNorm)
- hidden_size=2048, num_layers=22, num_heads=32, num_kv_heads=4, intermediate_size=5632
- max_position_embeddings=2048
- Gradient checkpointing: enabled
- Vocab: 32,000

---

## Previous Run (with compile safety margin, same session)

An earlier run on the same GPU used a conservative batch size (stepping down one level from max). Results for reference:

| Model | T | BS | tok/s | VRAM | ms/step |
|---|---|---|---|---|---|
| neuromorphic-c | 1024 | 6 | 22,600 | 51.7 GB | 271.4 ms |
| neuromorphic-c | 2048 | 2 | 21,900 | 39.8 GB | 187.3 ms |
| pythia-1b | 1024 | 48 | 33,800 | 27.5 GB | 1454.2 ms |
| pythia-1b | 2048 | 32 | 26,300 | 32.8 GB | 2490.8 ms |
| tinyllama-1b | 1024 | 48 | 30,100 | 33.0 GB | 1630.4 ms |
| tinyllama-1b | 2048 | 32 | 25,100 | 43.9 GB | 2609.6 ms |

The final run (without safety margin) improved neuromorphic throughput by ~9% at T=2048 (21.9K → 23.9K) by using BS=4 instead of BS=2.

---

## Gradient Checkpointing Experiment (neuromorphic-c only)

Tested whether `gradient_checkpointing=True` on scan layers improves throughput by allowing larger batch sizes.

| Setting | T | BS | tok/s | VRAM | ms/step | TFLOPS | 1.5B tok |
|---|---|---|---|---|---|---|---|
| No grad_ckpt | 1024 | 8 | 24,200 | 63.5 GB | 338.4 ms | 130.4 | 17.2h |
| No grad_ckpt | 2048 | 4 | 23,900 | 63.3 GB | 342.5 ms | 129.5 | 17.4h |
| **With grad_ckpt** | 1024 | 16 | 21,200 | 58.0 GB | 773.6 ms | 114.1 | 19.7h |
| **With grad_ckpt** | 2048 | 12 | 20,800 | 72.5 GB | 1180.6 ms | 112.7 | 20.0h |

**Verdict: Not worth it.** Grad checkpointing doubles/triples the batch size but throughput drops ~13%. The recomputation overhead costs more than the parallelism gains. The memory bottleneck is PM/EM state per sample (`[BS, B, D_pm, D_pm]` for PM, `[BS, B, M, D]` for EM), not scan activations. Checkpointing frees scan activations but doesn't touch the real memory hog.

Training time increases from 17.4h to 20.0h for 1.5B tokens at T=2048 (15% slower). Neuromorphic should train **without** gradient checkpointing.

---

## Notes

- **Mamba-1.4b was excluded** from this benchmark. The `mamba-ssm` CUDA kernel package requires building from source on Blackwell (no prebuilt wheel), and the HuggingFace sequential fallback is unusably slow (77GB VRAM for BS=1). Mamba benchmarks require a separate run with `pip install mamba-ssm causal-conv1d` pre-built.
- **RWKV-7 was excluded** because the `fla` (flash-linear-attention) package was not available on the RunPod image.
- **Real training overhead**: Expect ~95% of benchmark tok/s during actual training. The benchmark uses random data on GPU (no I/O), while real training adds data loading, validation, and checkpointing overhead.
- **The benchmark uses fp32 weights with bf16 autocast** (standard mixed precision). All models use the same precision setup.
