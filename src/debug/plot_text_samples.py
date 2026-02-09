"""
Text prediction visualization — model generation vs ground truth.

Produces a PNG with side-by-side comparisons for multiple batch items,
showing prompt (gray), model prediction (blue), and ground truth (green).

Usage (standalone):
    python -m src.debug.plot_text_samples  # requires model + data

Programmatic (called from train.py):
    from src.debug.plot_text_samples import generate_text_sample_plot
    generate_text_sample_plot(model, tokenizer, batch, step, loss, save_path, ...)
"""

import textwrap

import matplotlib.pyplot as plt
import torch


def _decode_safe(tokenizer, ids: list[int]) -> str:
    """Decode token ids, replacing errors."""
    try:
        return tokenizer.decode(ids, skip_special_tokens=False)
    except Exception:
        return "<decode error>"


def _wrap(text: str, width: int = 80) -> str:
    """Wrap text for display, preserving newlines."""
    lines = text.split("\n")
    wrapped = []
    for line in lines:
        if len(line) <= width:
            wrapped.append(line)
        else:
            wrapped.extend(textwrap.wrap(line, width=width) or [""])
    return "\n".join(wrapped)


@torch.no_grad()
def generate_text_sample_plot(
    model,
    tokenizer,
    batch: torch.Tensor,
    step: int,
    loss: float | None,
    save_path: str,
    prompt_len: int = 20,
    gen_len: int = 40,
    n_samples: int = 3,
    temperature: float = 0.8,
    top_k: int = 50,
):
    """Generate text samples and save comparison plot.

    Args:
        model: NeuromorphicLM instance (on device, in eval mode)
        tokenizer: tokenizer with decode()
        batch: [BS, T] token ids from validation data
        step: current training step (for title)
        loss: current loss (for title), or None
        save_path: output PNG path
        prompt_len: number of tokens to use as prompt
        gen_len: number of tokens to generate
        n_samples: number of batch items to show
        temperature: sampling temperature
        top_k: top-k for sampling
    """
    was_training = model.training
    model.eval()

    BS, T = batch.shape
    n_samples = min(n_samples, BS)
    # Need enough tokens for prompt + ground truth continuation
    effective_prompt = min(prompt_len, T // 2)
    effective_gt_len = min(gen_len, T - effective_prompt)

    if effective_prompt < 2 or effective_gt_len < 2:
        model.train(was_training)
        return

    device = next(model.parameters()).device
    prompt = batch[:n_samples, :effective_prompt].to(device)
    gt_continuation = batch[:n_samples, effective_prompt:effective_prompt + effective_gt_len]

    # Reset model memory state for clean generation
    reset_mask = torch.ones(n_samples, dtype=torch.bool, device=device)
    model.reset_at_doc_boundary(reset_mask)

    try:
        generated = model.generate(
            prompt,
            max_new_tokens=effective_gt_len,
            temperature=temperature,
            top_k=top_k,
        )
        pred_continuation = generated[:, effective_prompt:effective_prompt + effective_gt_len]
    except Exception as e:
        print(f"  Text sample generation failed: {e}")
        model.train(was_training)
        return

    # Decode all samples
    samples = []
    for i in range(n_samples):
        prompt_text = _decode_safe(tokenizer, prompt[i].cpu().tolist())
        pred_text = _decode_safe(tokenizer, pred_continuation[i].cpu().tolist())
        gt_text = _decode_safe(tokenizer, gt_continuation[i].cpu().tolist())
        samples.append((prompt_text, pred_text, gt_text))

    # Build figure
    fig, axes = plt.subplots(n_samples, 1, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = [axes]

    title = f"Text Samples — Step {step}"
    if loss is not None:
        title += f" | Loss {loss:.4f}"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for idx, (ax, (prompt_text, pred_text, gt_text)) in enumerate(zip(axes, samples)):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")

        prompt_display = _wrap(prompt_text, width=90)
        pred_display = _wrap(pred_text, width=90)
        gt_display = _wrap(gt_text, width=90)

        ax.text(
            0.02, 0.95, "PROMPT:", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#555555",
            verticalalignment="top", fontfamily="monospace",
        )
        ax.text(
            0.02, 0.88, prompt_display, transform=ax.transAxes,
            fontsize=8, color="#555555",
            verticalalignment="top", fontfamily="monospace",
            linespacing=1.3,
        )
        ax.text(
            0.02, 0.58, "MODEL:", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#1f77b4",
            verticalalignment="top", fontfamily="monospace",
        )
        ax.text(
            0.02, 0.51, pred_display, transform=ax.transAxes,
            fontsize=8, color="#1f77b4",
            verticalalignment="top", fontfamily="monospace",
            linespacing=1.3,
        )
        ax.text(
            0.02, 0.28, "TRUTH:", transform=ax.transAxes,
            fontsize=9, fontweight="bold", color="#2ca02c",
            verticalalignment="top", fontfamily="monospace",
        )
        ax.text(
            0.02, 0.21, gt_display, transform=ax.transAxes,
            fontsize=8, color="#2ca02c",
            verticalalignment="top", fontfamily="monospace",
            linespacing=1.3,
        )

        if idx < n_samples - 1:
            ax.axhline(y=0, color="gray", linewidth=0.5, alpha=0.3)

        ax.set_title(f"Sample {idx + 1}", fontsize=10, loc="left", pad=2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Text samples: {save_path}")

    model.train(was_training)
