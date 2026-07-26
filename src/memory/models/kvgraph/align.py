"""Stage 0 — char spans <-> token indices.

The parser works on characters; the KV cache is indexed by token position. Every node's content is pooled
from `KV[token_positions]`, so this mapping is the seam the whole pipeline rests on.

It is also the pipeline's most dangerous silent failure. If the parser and the tokenizer ever see different
text — one whitespace-normalised and the other not — every offset shifts, no exception is raised, and the
nodes are pooled from the wrong tokens. `assert_alignment` exists to turn that into a loud failure.

We reconstruct offsets by INCREMENTAL DECODE rather than `return_offsets_mapping`, because the encoder is
handed token ids (the harness embeds before it calls us) and never sees the source string. Decoding prefixes
is O(T^2) in characters, which at a 256-token window is ~65k char-copies — irrelevant next to a forward pass.
"""
from __future__ import annotations


def token_char_offsets(tokenizer, ids: list[int]) -> tuple[str, list[tuple[int, int]]]:
    """-> (text, [(char_start, char_end)] per token).

    `text` is the detokenisation of `ids` and is THE string the parser must see — never re-derive it, never
    normalise it. Offsets are half-open and contiguous: token i owns `text[start_i:end_i]`, which for
    byte-level BPE includes any leading space (`" the"` starts at the space, not at `t`).

    A token that decodes to nothing (special tokens under `skip_special_tokens=False` usually do decode, but
    some tokenizers emit empty strings) gets a zero-width span and will simply never overlap a mention.
    """
    text = ""
    offsets: list[tuple[int, int]] = []
    for i in range(len(ids)):
        prev = text
        text = tokenizer.decode(ids[: i + 1], skip_special_tokens=False)
        if not text.startswith(prev):
            # Byte-level BPE can split a multi-byte character across tokens, so a prefix decode is not
            # always a string prefix. Fall back to a zero-width span at the current end: the token is
            # unaddressable for pooling, which is correct — it does not correspond to any whole character.
            offsets.append((len(prev), len(prev)))
            text = prev + text[len(prev):] if len(text) > len(prev) else prev
            continue
        offsets.append((len(prev), len(text)))
    return text, offsets


def tokens_for_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> list[int]:
    """Token indices whose character span OVERLAPS [char_start, char_end).

    Overlap, not containment: BPE tokens routinely straddle a mention boundary (`" the"` carries the
    preceding space, `"rum"+"our"` splits a word), and containment would silently drop them.
    """
    if char_end <= char_start:
        return []
    return [i for i, (s, e) in enumerate(offsets)
            if s < char_end and e > char_start and e > s]


def assert_alignment(text: str, offsets: list[tuple[int, int]], n_tokens: int) -> None:
    """Raise if the offset table is not a valid tiling of `text`. Cheap; run it on every window in tests
    and on the first window of every run."""
    if len(offsets) != n_tokens:
        raise ValueError(f"alignment: {len(offsets)} offsets for {n_tokens} tokens")
    prev_end = 0
    for i, (s, e) in enumerate(offsets):
        if s < prev_end or e < s or e > len(text):
            raise ValueError(f"alignment: token {i} span ({s},{e}) is not monotone within {len(text)} chars")
        prev_end = e
    if offsets and offsets[-1][1] != len(text):
        raise ValueError(f"alignment: offsets end at {offsets[-1][1]}, text is {len(text)} chars — "
                         "the parser and the tokenizer are not seeing the same string")
