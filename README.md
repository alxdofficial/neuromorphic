# Trajectory-Memory LM

A Llama-3.2-1B backbone with a side-car concept-trajectory memory.

The model wraps a frozen Llama with a manifold of N concepts. Every
256-token window: read J parallel trajectories from the manifold using
the previous window's tokens as a query, predict the window's tokens
with Llama (cross-attending to the read trajectories), then write J
trajectories back to persistently mutate concept states.

See **[`docs/plan_trajectory_memory.md`](docs/plan_trajectory_memory.md)**
for the full design, implementation plan, training waves, and efficiency
analysis.

## Layout

```
src/trajectory_memory/    — core architecture (manifold, read/write modules,
                            IntegratedLM, TBPTT) plus data/ + training/ subpackages
src/pretrained/           — reused Llama host adapters + MemInjectLayer
tests/                    — unit + smoke tests for the trajectory modules
scripts/                  — entry points (training/, data/, bench/, diagnostics/)
                            see scripts/README.md
docs/                     — design, plan, eval, bench results, research backlog
```

## Tests

```bash
pytest tests/test_trajectory_memory_*.py
```

## History

The earlier per-token `graph_walker` lineage is archived under the
`abandoned/graph-walker` branch.
