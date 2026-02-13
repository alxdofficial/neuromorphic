"""Smoke test: load each baseline model and generate a short sample."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "mamba-130m": "state-spaces/mamba-130m-hf",
}

PROMPT = "The quick brown fox"


def smoke_test(name: str, repo_id: str):
    print(f"\n{'='*60}")
    print(f"Loading {name} ({repo_id})...")

    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        dtype=torch.float16,
        device_map="auto",
    )

    # Model info
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,} ({n_params/1e6:.1f}M)")
    print(f"  Device: {next(model.parameters()).device}")
    print(f"  Dtype: {next(model.parameters()).dtype}")

    # Generate
    inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"  Prompt: {PROMPT!r}")
    print(f"  Output: {text!r}")

    # Quick perplexity on a single sentence
    test_text = "The cat sat on the mat and looked out the window at the birds."
    test_inputs = tokenizer(test_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**test_inputs, labels=test_inputs["input_ids"])
    ppl = torch.exp(out.loss).item()
    print(f"  Test PPL: {ppl:.2f} (on: {test_text!r})")
    print(f"  OK")

    # Free memory
    del model
    torch.cuda.empty_cache()


def main():
    print("Baseline Model Smoke Tests")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM free: {torch.cuda.mem_get_info(0)[0]/1e9:.1f}GB / {torch.cuda.mem_get_info(0)[1]/1e9:.1f}GB")

    for name, repo_id in MODELS.items():
        try:
            smoke_test(name, repo_id)
        except Exception as e:
            print(f"\n  FAILED: {name}: {e}")

    print(f"\n{'='*60}")
    print("Smoke tests complete.")


if __name__ == "__main__":
    main()
