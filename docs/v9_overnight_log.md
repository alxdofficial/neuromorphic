# Graph v9 — autonomous overnight run log (2026-06-12 → 13)

**Checkpoint:** last human-vetted commit = `a3bc7bd` on `main` (pushed). All work
below lives on branch `v9-overnight-auto`. Nothing here is vetted; treat every
change as a hypothesis probe.

## Mandate (user, 2026-06-12 evening)
1. Match memory capacity to the baselines and param count to prior graph versions.
2. Run the 600-step emat_bio gate; analyze SHUF−REAL.
3. If flat: explore all plausible reasons, autonomously, documenting as I go.

## The bar
- v8c5 (last graph attempt): SHUF−REAL = **−0.0007** (flat), OFF−REAL +3.30,
  recon 1.03, 52.2M trainable.
- Beacon: SHUF−REAL = **+2.07** at 102M params.
- No-memory floor: recon 1.79 / 59% top1 (objective ~⅔ guessable).
- Gate protocol: 600 steps, BS=8, emat_bio, REAL/SHUF/OFF in the val loop
  (`val_shuf_minus_real` in the jsonl). Launch flags (from the probe-script
  launch constants): `--task emat_bio --chunk-size 640 --window-size 640
  --mem-tokens 144 --emat-n-pairs 12 --steps 600 --batch-size 8`.

## Capacity matching (decision)
- Baseline read-memory anchor: v8c = 3·768·2·64 = **294,912 floats**.
- v9 gate config: d_code=256, d_key=256, nodes=(576, 288), slots=(1, 4)
  → writable fast state = 288·4·257 = **296,064 floats** (+0.4% vs anchor).
  Arm C layer 0 (atoms) is slow weights = shared vocabulary, not per-example
  memory — excluded from the budget, same logic as not counting model weights.
- **Param asymmetry, flagged loudly:** v9 lands ~3M trainable vs v8c's 52.2M and
  Beacon's 102M. This is BY DESIGN (doc §12 success criterion: binding at far
  below baseline params), and it favors the baselines — a v9 win is stronger
  for it; a v9 loss has "too few params" as a standing alternative explanation.
  A param-scaled variant is queued as a follow-up arm if time permits.

## Findings so far (pre-gate, from the debug sweep — full details in git log)
- State separability at init = 1.0000 (different docs → nearly identical
  relocated states; template-dominated coactivation). THE number to watch:
  `graph_v9_state_sep_cos_L1` must drop during training or SHUF≡REAL is
  structural. Now logged every val.
- Effective β of the strongest applied factor ≈ 0.2 at init (softmax scores
  ~1/8 × strength 1.5): the chain is GENTLE at init (apply_rotation_cos 0.985).
  Training must sharpen routing and/or grow strengths for the memory to act.
- Absorption magnitudes healthy at init (~7% strength relocation/doc).
- Routing calibration transfers to real data (eff-k 7.9-9.3 vs target 8).

## Run index
(appended as runs complete)
