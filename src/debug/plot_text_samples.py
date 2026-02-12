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


def _find_clean_offset(row: torch.Tensor, eot_id: int, need: int) -> int:
    """Find the first offset in row where row[offset:offset+need] has no EOT.

    Scans forward looking for a contiguous region within a single document.
    Returns 0 as fallback if no clean region exists (short docs).
    """
    T = row.shape[0]
    if need > T:
        return 0
    for start in range(T - need + 1):
        chunk = row[start:start + need]
        if not (chunk == eot_id).any():
            return start
    return 0  # fallback: use start of batch


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

    For each batch row, finds a contiguous region within a single document
    (no EOT tokens) to use as prompt + ground truth. This avoids confusing
    displays where prompt and truth span different documents.

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
    n_display = min(n_samples, BS)
    effective_prompt = min(prompt_len, T // 2)
    effective_gt_len = min(gen_len, T - effective_prompt)
    need = effective_prompt + effective_gt_len

    if effective_prompt < 2 or effective_gt_len < 2:
        model.train(was_training)
        return

    device = next(model.parameters()).device
    eot_id = model.config.eot_id

    # Per-sample: find clean region within a single document
    offsets = []
    for i in range(n_display):
        off = _find_clean_offset(batch[i], eot_id, need)
        offsets.append(off)

    # Build per-sample prompt and ground truth tensors
    prompt_list = []
    gt_list = []
    for i in range(n_display):
        off = offsets[i]
        prompt_list.append(batch[i, off:off + effective_prompt])
        gt_list.append(batch[i, off + effective_prompt:off + effective_prompt + effective_gt_len])
    # For generation we need uniform prompt shape — pad remaining batch rows
    # with the first sample's prompt (they won't be displayed)
    all_prompts = []
    for i in range(BS):
        if i < n_display:
            all_prompts.append(batch[i, offsets[i]:offsets[i] + effective_prompt])
        else:
            all_prompts.append(batch[i, offsets[0]:offsets[0] + effective_prompt])
    prompt = torch.stack(all_prompts).to(device)
    gt_continuation = torch.stack(gt_list)

    # Reset model memory state for clean generation
    reset_mask = torch.ones(BS, dtype=torch.bool, device=device)
    model.reset_at_doc_boundary(reset_mask)

    try:
        generated = model.generate(
            prompt,
            max_new_tokens=effective_gt_len,
            temperature=temperature,
            top_k=top_k,
        )
        pred_continuation = generated[:n_display, effective_prompt:effective_prompt + effective_gt_len]
    except Exception as e:
        print(f"  Text sample generation failed: {e}")
        model.train(was_training)
        return

    # Decode all samples
    samples = []
    for i in range(n_display):
        prompt_text = _decode_safe(tokenizer, prompt_list[i].cpu().tolist())
        pred_text = _decode_safe(tokenizer, pred_continuation[i].cpu().tolist())
        gt_text = _decode_safe(tokenizer, gt_continuation[i].cpu().tolist())
        samples.append((prompt_text, pred_text, gt_text))

    # Build figure
    fig, axes = plt.subplots(n_display, 1, figsize=(14, 4 * n_display))
    if n_display == 1:
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
