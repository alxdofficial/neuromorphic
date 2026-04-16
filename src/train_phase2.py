"""Phase 2 (GRPO) training — PLACEHOLDER.

Non-functional on the conv-grid-modulator branch. Phase 2 GRPO is deferred
until the pretrained-LM pivot makes autoregressive sampling viable; the
previous teacher-forced GRPO implementation had an unfixable SNR problem
(see `memory/grpo_teacher_forcing_failure.md`).

When phase 2 is reintroduced, it will:
  - Freeze everything except the conv-grid-modulator encoder (conv + logit head)
  - Sample codes via hard Categorical (not Gumbel)
  - Use autoregressive rollouts on the frozen pretrained LM so K trajectories
    diverge in token space; reward variance is real, GRPO signal is recovered
"""


def main():
    raise NotImplementedError(
        "Phase 2 GRPO is not available on the conv-grid-modulator branch. "
        "See docs/design_conv_modulator.md and the pretrained_lm_pivot memory "
        "entry for the planned revival path.")


if __name__ == "__main__":
    main()
