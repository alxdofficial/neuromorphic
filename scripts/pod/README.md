# Remote training on vast.ai pods

> Full runbook (setup, tmux→ssh flow, GPU selection, and every gotcha solved):
> **`docs/vast_ai_workflow.md`**. This file is the terse command list.

One pod per training arm. Data + code come from R2/GitHub, results go back to R2, and
a local watchdog reaps pods so a stuck job can't bleed credit. The billing-capable
vast.ai key **never leaves this machine**.

```
 local ──(1 launch)──▶ vast.ai pods ──(2 drive: ssh + creds)──▶ bootstrap.sh
   ▲                        │                                        │
   │                        │ train one arm (behavioral_kl)          │
   └──(4 pull results)── R2 ◀──(3 push results + _STATUS)────────────┘
                          ▲
       watchdog polls _STATUS + vast list, destroys on done/fail/cap
```

## Files
| file | runs where | does |
|------|-----------|------|
| `pack_data.sh` | local (once) | tar the 5-task training subset → upload `data.tar.gz` to R2 ✅ done |
| `launch.py` | local | pick vast offers, create one pod per arm (dry-run by default) |
| `drive.sh` | local | ssh into each pod, copy R2 creds, start `bootstrap.sh` in a remote tmux |
| `bootstrap.sh` | pod | pull code+data, train one arm, push results + `_STATUS` to R2 |
| `watchdog.py` | local | poll `_STATUS`/vast, destroy pods on done/fail/wall-cap (billing safety) |
| `pull_results.sh` | local | download a run's results from R2 into `outputs/memory/` |
| `r2.sh` | local+pod | thin `aws s3` wrapper for the R2 bucket |

## Credentials (already set up)
- `~/.config/vastai/api_key` — the funded vast.ai account. Local only.
- `~/.config/r2/credentials` — R2 access (`gen-purp-bucket`, prefix `neuromorphic`).
- Your SSH pubkey must be registered with vast once: `scripts/pod/drive.sh --register-key`.

## Launch preconditions (do these before `--go`)
1. **Push the training fixes.** Pods clone a git ref (default `main`). The KL-teacher,
   continuation-routing, checkpoint `cfg_all`, and titans `copy.copy` fixes must be on
   that ref or the pods train stale/buggy code. Commit + push, then launch with the
   matching `--ref` (e.g. `--ref main` or `--ref pod-run`).
2. **Register your SSH key** (ssh mode): `scripts/pod/drive.sh --register-key`.
3. **Add credit** if needed — the account gates all spend at its balance.

## The flow (held until you say go)
```bash
# 0. see the plan + worst-case cost, spend nothing:
python scripts/pod/launch.py --arms slotgraph_baseline icae_baseline

# 1. launch bare ssh pods (spends money):
python scripts/pod/launch.py --arms icae_baseline autocompressor_baseline titans_baseline \
       gisting_baseline memoryllm_baseline slotgraph_baseline --ref main --steps 2000 --go
#    → writes scripts/pod/state/podrun-NNNN.json

# 2. start training on every pod (ssh + creds over the encrypted channel):
scripts/pod/drive.sh podrun-NNNN

# 3. billing backstop — leave running:
python scripts/pod/watchdog.py podrun-NNNN

# watch one arm live (optional):
scripts/pod/drive.sh podrun-NNNN --attach slotgraph_baseline   # Ctrl-b d to detach
scripts/pod/pull_results.sh podrun-NNNN --status               # quick status poll

# 4. when statuses say DONE, pull + evaluate:
scripts/pod/pull_results.sh podrun-NNNN
python scripts/diagnostics/mixed/mixed_band_gate_eval.py --out-tag podrun-NNNN
```

## Billing safety (four independent layers)
1. **wall-clock cap** — `bootstrap.sh` wraps training in `timeout ${MAX_HOURS}h` (default 8h).
2. **watchdog** — destroys any pod past `max_hours + grace`, or on terminal `_STATUS`.
3. **key locality** — the vast key stays local; pods can't spawn or extend themselves.
4. **credit ceiling** — the account only holds its balance; vast stops pods at $0.

Panic button: `python scripts/pod/watchdog.py podrun-NNNN --destroy-all`.

## Modes
- `--mode ssh` (default) — bare pods; R2 creds delivered over SSH by `drive.sh`. Secrets
  never touch vast.ai metadata. Matches the "local tmux → ssh into pod" workflow.
- `--mode auto` — onstart runs `bootstrap.sh` with R2 creds passed via `--env`. Fully
  hands-off, but R2 creds are stored in vast's instance metadata. Skip `drive.sh`.

## Result layout in R2
```
neuromorphic/data.tar.gz                                    # training data (820 MB)
neuromorphic/results/<run_id>/<arm>/run.jsonl              # per-step metrics
neuromorphic/results/<run_id>/<arm>/<arm>.last.pt          # final weights
neuromorphic/results/<run_id>/<arm>/summary.json
neuromorphic/results/<run_id>/<arm>/bootstrap.log
neuromorphic/results/<run_id>/<arm>/_STATUS               # RUNNING|DONE|FAILED|TIMEOUT
```
