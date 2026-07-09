# Cohort results — mixed 4-task, same-code

Frozen SmolLM2-135M (d=576), 1024→M=32 (32:1), 4000 steps, batch 8. All models trained on the **same code commit** (cross-era comparison is invalid). Tasks: mae / babi / continuation / condrecon_bio. Cells are **mean ± std** over the seeds present (n shown per model); single-seed cells show the mean only.

Seeds per model: biomem (membrane on) (n=3), biomem (membrane off) (n=3), slotgraph (n=3), icae (n=3), ccm (n=3), autocompressor (n=3), beacon (n=3)

## Table 1 — REAL performance (val loss; lower is better)

babi also shows exact-match EM (final). Loss is unitless.

| model | mae | babi | continuation | condrecon_bio | babi EM |
|---|---|---|---|---|---|
| biomem (membrane on) | 6.651±0.010 | 1.006±0.031 | 3.504±0.002 | 2.152±0.016 | 23±4% |
| biomem (membrane off) | 6.650±0.010 | 1.026±0.055 | 3.503±0.002 | 2.122±0.020 | 22±5% |
| slotgraph | 6.541±0.012 | 0.954±0.100 | 3.327±0.007 | 2.323±0.026 | 30±9% |
| icae | 6.587±0.013 | 0.989±0.068 | 3.370±0.006 | 2.349±0.005 | 31±6% |
| ccm | 6.593±0.004 | 1.068±0.062 | 3.370±0.007 | 2.355±0.022 | 25±4% |
| autocompressor | 6.576±0.005 | 0.947±0.122 | 3.359±0.005 | 2.373±0.017 | 31±8% |
| beacon | 6.601±0.004 | 1.110±0.042 | 3.423±0.004 | 2.522±0.024 | 22±2% |

## Table 2 — binding gate (loss units)

OFF−REAL = cost of zeroing memory (usage). SHUF−REAL = cost of the *wrong* example's memory (example-specificity). **Caveat:** the loss-based SHUF gate is unreliable on babi (consecutive examples share answers → wrong memory often coincidentally fits); Table 3 is the reliable binding signal.

| model | mae OFF / SHUF | babi OFF / SHUF | continuation OFF / SHUF | condrecon_bio OFF / SHUF |
|---|---|---|---|---|
| biomem (membrane on) | 0.30±0.18 / 0.001±0.000 | 8.59±0.13 / 0.006±0.006 | 0.15±0.01 / 0.001±0.001 | 1.58±0.05 / 0.000±0.002 |
| biomem (membrane off) | 0.35±0.21 / 0.001±0.001 | 8.67±0.13 / 0.012±0.005 | 0.14±0.00 / 0.002±0.000 | 1.63±0.06 / -0.000±0.002 |
| slotgraph | 0.25±0.02 / 0.459±0.022 | 8.08±0.26 / 0.203±0.093 | 0.29±0.02 / 0.492±0.027 | 1.02±0.05 / 0.020±0.017 |
| icae | 0.23±0.02 / 0.335±0.019 | 8.14±0.11 / 0.167±0.132 | 0.25±0.01 / 0.367±0.018 | 0.92±0.10 / 0.028±0.009 |
| ccm | 0.21±0.00 / 0.319±0.032 | 8.11±0.20 / 0.066±0.048 | 0.25±0.02 / 0.360±0.022 | 0.91±0.11 / 0.018±0.005 |
| autocompressor | 0.23±0.01 / 0.359±0.048 | 7.70±0.38 / 0.160±0.157 | 0.27±0.02 / 0.377±0.036 | 0.95±0.10 / 0.022±0.013 |
| beacon | 0.16±0.01 / 0.256±0.018 | 7.22±0.21 / -0.015±0.040 | 0.18±0.01 / 0.306±0.009 | 0.76±0.05 / 0.004±0.010 |

## Table 3 — babi_em binding (exact-match REAL / SHUF / OFF)

The reliable binding test. **OFF=0 → memory is essential** (answer impossible without it). SHUF≈REAL across *all* models (incl. published icae) confirms the SHUF metric is diluted on babi, not that models fail to bind.

| model | REAL | SHUF | OFF |
|---|---|---|---|
| biomem (membrane on) | 29.2±11.8% | 28.1±8.3% | 0.0±0.0% |
| biomem (membrane off) | 22.9±9.0% | 21.9±5.4% | 0.0±0.0% |
| slotgraph | 31.2±8.3% | 26.0±7.2% | 0.0±0.0% |
| icae | 31.2±8.3% | 24.0±4.8% | 0.0±0.0% |
| ccm | 24.0±9.5% | 24.0±6.5% | 0.0±0.0% |
| autocompressor | 32.3±3.6% | 30.2±7.9% | 0.0±0.0% |
| beacon | 29.2±7.9% | 27.1±1.8% | 0.0±0.0% |
