# slotgraph attribution — clean 2×2 (from-scratch controls)

Each cell trained from scratch, same config, LoRA rank 82, 3 seeds {42,1,2}. Isolates the contribution of the **id-tags** and the **graph structure (MP read)** without the eval-time distribution-shift confound. `neither` = slotgraph code with both off (same-code icae anchor at rank 82); `icae (native)` = the real icae baseline.

Seeds present: full (struct+id) (n=3), struct-only (id OFF) (n=3), id-only (struct OFF) (n=3), neither (same-code icae) (n=3), icae (native) (n=3)

## Table 1 — REAL loss (↓) + babi EM (↑)

| model | mae | babi | continuation | condrecon_bio | babi EM |
|---|---|---|---|---|---|
| full (struct+id) | 6.541±0.012 | 0.954±0.100 | 3.327±0.007 | 2.323±0.026 | 30±9% |
| struct-only (id OFF) | 6.557±0.003 | 0.976±0.049 | 3.341±0.003 | 2.335±0.044 | 31±5% |
| id-only (struct OFF) | 6.614±0.051 | 1.027±0.006 | 3.390±0.033 | 2.373±0.034 | 26±6% |
| neither (same-code icae) | 6.596±0.015 | 1.090±0.029 | 3.372±0.008 | 2.390±0.065 | 23±4% |
| icae (native) | 6.587±0.013 | 0.989±0.068 | 3.370±0.006 | 2.349±0.005 | 31±6% |

## Table 2 — example-specificity (mae / continuation SHUF−REAL, ↑=binds)

Reliable on mae/continuation (no babi correlated-batch dilution).

| model | mae SHUF−REAL | continuation SHUF−REAL |
|---|---|---|
| full (struct+id) | 0.459±0.022 | 0.492±0.027 |
| struct-only (id OFF) | 0.413±0.010 | 0.437±0.019 |
| id-only (struct OFF) | 0.276±0.097 | 0.327±0.071 |
| neither (same-code icae) | 0.299±0.031 | 0.338±0.034 |
| icae (native) | 0.335±0.019 | 0.367±0.018 |

## Table 3 — babi_em binding (exact-match REAL / SHUF / OFF)

| model | REAL | SHUF | OFF |
|---|---|---|---|
| full (struct+id) | 31.2±8.3% | 26.0±7.2% | 0.0±0.0% |
| struct-only (id OFF) | 34.4±3.1% | 22.9±7.9% | 0.0±0.0% |
| id-only (struct OFF) | 26.0±7.9% | 29.2±11.0% | 0.0±0.0% |
| neither (same-code icae) | 18.8±9.4% | 21.9±5.4% | 0.0±0.0% |
| icae (native) | 31.2±8.3% | 24.0±4.8% | 0.0±0.0% |
