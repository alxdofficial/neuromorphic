"""Phase 2 (GRPO) training — PLACEHOLDER.

Non-functional on this branch. Phase 2 GRPO is deferred until the
pretrained-LM pivot makes autoregressive sampling viable; the previous
teacher-forced GRPO implementation had an unfixable SNR problem (see
`memory/grpo_teacher_forcing_failure.md`).

When phase 2 is reintroduced, it will:
  - Freeze everything except the attention modulator (qkv/FFN weights +
    logit head) and the codebook.
  - Sample codes via hard Categorical (not Gumbel) using
    `discrete_policy.sample_discrete`, which returns `log_pi` for GRPO.
  - Use autoregressive rollouts on the (frozen) pretrained LM so K
    trajectories diverge in token space, giving real reward variance.

See `docs/design.md` for current architecture.
"""


def main():
    raise NotImplementedError(
        "Phase 2 GRPO is not available yet. See docs/design.md for current "
        "architecture and the pretrained-LM-pivot memory entry for the "
        "planned revival path.")


if __name__ == "__main__":
    main()
