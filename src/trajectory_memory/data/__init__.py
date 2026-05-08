"""Data utilities for trajectory-memory training.

- `tokenizer.py`     — Llama-3.2 tokenizer wrapper + role/special tokens.
- `chat.py`          — chat-template formatting + assistant-token mask.
- `turn_pair.py`     — session_to_turn_pairs for multi-turn data (W2/W4).
- `streaming.py`     — Wave 1 long-doc streaming chunker for TBPTT.
- `needle_haystack.py` — synthetic needle-in-haystack generator (W1).
"""

from src.trajectory_memory.data.chat import (
    apply_chat_template,
    build_assistant_mask,
)
from src.trajectory_memory.data.streaming import (
    StreamingTokenChunker,
    pack_documents,
)
from src.trajectory_memory.data.tokenizer import get_tokenizer
from src.trajectory_memory.data.turn_pair import (
    TurnPair,
    session_to_turn_pairs,
)

__all__ = [
    "get_tokenizer",
    "apply_chat_template",
    "build_assistant_mask",
    "TurnPair",
    "session_to_turn_pairs",
    "StreamingTokenChunker",
    "pack_documents",
]
