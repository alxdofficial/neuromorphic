"""
Tokenizer setup for neuromorphic LM training.

Supports multiple tokenizer families for comparison with modern SLMs:
- LLaMA-style (TinyLlama, SmolLM2): 32K vocab, BOS/EOS tokens
- GPT-2 style: 50K vocab, EOT token

Default: LLaMA-style (TinyLlama) for comparison with modern small LMs.
"""

from typing import Optional, Literal
from transformers import AutoTokenizer, PreTrainedTokenizerFast


# Tokenizer presets for common small LMs
TOKENIZER_PRESETS = {
    # LLaMA-style tokenizers (modern SLMs)
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "llama2": "meta-llama/Llama-2-7b-hf",
    "smollm": "HuggingFaceTB/SmolLM2-135M",

    # GPT-style tokenizers
    "gpt2": "gpt2",
    "gpt-neox": "EleutherAI/gpt-neox-20b",
}

# Default tokenizer (LLaMA-style for modern SLM comparison)
DEFAULT_TOKENIZER = "tinyllama"


def get_tokenizer(
    preset: str = DEFAULT_TOKENIZER,
    cache_dir: Optional[str] = None,
) -> PreTrainedTokenizerFast:
    """
    Load tokenizer for training.

    Args:
        preset: Tokenizer preset name or HuggingFace model path
            Options: "tinyllama", "llama2", "smollm", "gpt2", "gpt-neox"
            Or any HuggingFace model path with a tokenizer
        cache_dir: Optional cache directory for tokenizer files

    Returns:
        Configured tokenizer
    """
    # Resolve preset to model path
    model_path = TOKENIZER_PRESETS.get(preset, preset)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        cache_dir=cache_dir,
    )

    # Ensure pad token is set (use EOS if not defined)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def get_special_token_ids(tokenizer: PreTrainedTokenizerFast) -> dict:
    """
    Get special token IDs for a tokenizer.

    Returns dict with:
        - eos_token_id: End of sequence / document separator
        - bos_token_id: Beginning of sequence (may be None for GPT-2)
        - pad_token_id: Padding token
        - vocab_size: Total vocabulary size
    """
    return {
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "vocab_size": len(tokenizer),
    }


def tokenize_document(
    text: str,
    tokenizer: PreTrainedTokenizerFast,
    add_eos: bool = True,
) -> list[int]:
    """
    Tokenize a single document.

    Args:
        text: Document text
        tokenizer: Tokenizer instance
        add_eos: Whether to append EOS token (document separator)

    Returns:
        List of token IDs
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if add_eos and tokenizer.eos_token_id is not None:
        tokens.append(tokenizer.eos_token_id)
    return tokens


def tokenize_documents(
    texts: list[str],
    tokenizer: PreTrainedTokenizerFast,
    add_eos: bool = True,
) -> list[int]:
    """
    Tokenize multiple documents into a single flat token array.

    Each document is followed by EOS token to mark boundaries.

    Args:
        texts: List of document texts
        tokenizer: Tokenizer instance
        add_eos: Whether to append EOS token after each document

    Returns:
        Flat list of token IDs with EOS separators
    """
    all_tokens = []
    eos_id = tokenizer.eos_token_id

    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        if add_eos and eos_id is not None:
            all_tokens.append(eos_id)
    return all_tokens


def count_tokens(text: str, tokenizer: PreTrainedTokenizerFast) -> int:
    """Count tokens in text without allocating full token list."""
    return len(tokenizer.encode(text, add_special_tokens=False))


# Legacy compatibility - these will be set dynamically based on tokenizer
EOT_TOKEN_ID = None  # Use tokenizer.eos_token_id instead
VOCAB_SIZE = None    # Use len(tokenizer) instead


if __name__ == "__main__":
    print("Available tokenizer presets:")
    for name, path in TOKENIZER_PRESETS.items():
        print(f"  {name}: {path}")

    print(f"\n--- Testing default tokenizer ({DEFAULT_TOKENIZER}) ---")
    tokenizer = get_tokenizer()
    special_ids = get_special_token_ids(tokenizer)

    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {special_ids['vocab_size']}")
    print(f"EOS token: {tokenizer.eos_token} (id={special_ids['eos_token_id']})")
    print(f"BOS token: {tokenizer.bos_token} (id={special_ids['bos_token_id']})")

    # Test tokenization
    test_text = "Hello, world! This is a test."
    tokens = tokenize_document(test_text, tokenizer)
    print(f"\nTest text: {test_text}")
    print(f"Tokens: {tokens}")
    print(f"Decoded: {tokenizer.decode(tokens)}")

    # Compare with GPT-2
    print(f"\n--- Testing GPT-2 tokenizer ---")
    gpt2_tok = get_tokenizer("gpt2")
    gpt2_ids = get_special_token_ids(gpt2_tok)
    print(f"Vocab size: {gpt2_ids['vocab_size']}")
    print(f"EOS token: {gpt2_tok.eos_token} (id={gpt2_ids['eos_token_id']})")
    gpt2_tokens = tokenize_document(test_text, gpt2_tok)
    print(f"Tokens: {gpt2_tokens}")
