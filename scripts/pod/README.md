# Remote training on RunPod

> Full runbook (setup, GPU selection, the launch-bound perf finding, and every gotcha solved):
> **`docs/ops/runpod_workflow.md`**. This file is the terse command list.

One pod per training arm. Data + code come from R2/GitHub, results go back to R2, and
`runpod.py reap` terminates pods so a stuck job can't bleed credit. The billing-capable
RunPod key **never leaves this machine**.

```
 local ──(1 create)──▶ RunPod pod ──(2 drive: ssh+creds+tmux)──▶ bootstrap.sh
   ▲                        │                                        │
   │                        │ train one arm (behavioral_kl)          │
   └──(4 pull results)── R2 ◀──(3 push results + _STATUS)────────────┘
                          ▲
        runpod.py reap polls _STATUS, terminates on done/fail/cap
```

## Files
| file | runs where | does |
|------|-----------|------|
| `runpod.py` | local | create (GPU fallback) / drive (ssh+creds+tmux) / list / reap / terminate via the RunPod SDK |
| `pack_data.sh` | local (once) | tar the training data subset → upload `data.tar.gz` to R2 |
| `bootstrap.sh` | pod | **provider-agnostic** — pull code+data+models, train one arm, push results + `_STATUS` to R2 |
| `pull_results.sh` | local | download a run's results from R2 into `outputs/memory/` |
| `r2.sh` | local+pod | thin `aws s3` wrapper for the R2 bucket |

## Credentials (already set up)
- `~/.runpod/config.toml` — `apikey` for the funded RunPod account (local only). Its account SSH key
  matches `~/.ssh/id_ed25519`, which RunPod injects into every pod.
- `~/.config/r2/credentials` — R2 access (bucket + prefix `neuromorphic`).
- `~/.cache/huggingface/token` — gated `meta-llama/Llama-3.2-1B` FineWeb src-tokenizer.

## Launch preconditions
1. **Push the training code.** Pods clone a git ref (default `main`) — commit + push first, or pods train stale code.
2. **Re-pack data if it changed:** `bash scripts/pod/pack_data.sh` (uploads `data.tar.gz`); `models.tar.gz` pre-seed already on R2.
3. **Balance** covers the run (~$0.34–0.69/hr per 4090, ~2.8 h/arm).

## The flow
```bash
# 1. create one pod per arm (GPU fallback chain: 4090 → A40 → A6000 → A5000 → 3090)
python scripts/pod/runpod.py create icae_baseline
python scripts/pod/runpod.py list                              # confirm RUNNING + ssh endpoint

# 2. drive: ship creds + code, launch bootstrap in a detached tmux
python scripts/pod/runpod.py drive <pod_id> icae_baseline podrun-NNNN

# 3. monitor (poll FREQUENTLY — every 1–2 min; RunPod bills while idle)
ssh -i ~/.ssh/id_ed25519 -p <port> root@<host> 'tail -20 /workspace/bootstrap.log'

# 4. reap finished pods + pull results
python scripts/pod/runpod.py reap podrun-NNNN
scripts/pod/pull_results.sh podrun-NNNN
python scripts/diagnostics/mixed/mixed_band_gate_eval.py --out-tag podrun-NNNN
```

## Billing safety
1. **wall-clock cap** — `bootstrap.sh` wraps training in `timeout ${MAX_HOURS}h` (default 8 h).
2. **reap** — `runpod.py reap <run>` terminates pods on terminal `_STATUS`. Poll it (or the logs) frequently.
3. **key locality** — the RunPod key stays local; pods can't extend themselves.
4. **credit ceiling** — the account only holds its balance.

Panic button: `python scripts/pod/runpod.py terminate all`.

## Result layout in R2
```
neuromorphic/data.tar.gz                                   # training data
neuromorphic/models.tar.gz                                 # HF cache pre-seed (SmolLM2 + Llama tokenizer)
neuromorphic/results/<run_id>/<arm>/run.jsonl              # per-step metrics
neuromorphic/results/<run_id>/<arm>/ckpts/                 # checkpoints
neuromorphic/results/<run_id>/<arm>/summary.json
neuromorphic/results/<run_id>/<arm>/bootstrap.log
neuromorphic/results/<run_id>/<arm>/_STATUS                # RUNNING|DONE|FAILED|TIMEOUT
```
