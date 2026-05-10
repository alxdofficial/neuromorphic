"""Synthetic needle-in-haystack documents for Wave 1 memory pretraining
(plan §4.5: "Plant fact at position X, query at Y > X+2K. Forces measurable
memory contribution to NTP loss.").

Generates documents of the form:
    [filler_1] ... [needle: "The secret code is XYZ123."] ... [filler_K]
    ... [query: "What is the secret code?"] [answer: "XYZ123"]

Where:
  - The needle is planted at position X.
  - The query/answer are at position Y > X + 2K (beyond Llama's 2K cap).
  - The filler is real-world text drawn from a base corpus.

Training NTP loss on the answer-completion segment is reducible only if
memory has carried the needle past the LM's effective context cap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Iterator

from src.trajectory_memory.data.tokenizer import get_tokenizer


# Diverse needle templates — 30+ patterns spanning multiple domains so the
# model can't learn shallow surface-pattern shortcuts ("if I see a 6-char
# alphanumeric near position-X plus 'What is...?' near position-Y, just
# emit it"). Each entry is (statement, query, answer_marker).
#
# Domains: codes, IDs, dates, locations, names, numbers, prefs, json-ish,
# conversational, mid-sentence injections. Query phrasings vary too:
# explicit "What is X?", implicit "X was…", "Recall the…", etc.
_NEEDLE_TEMPLATES = [
    # ── codes / identifiers ──────────────────────────────────────────
    (
        "The secret access code is {answer}. Remember this; it will be tested.",
        "What is the secret access code?",
        "{answer}",
    ),
    (
        "The customer's user ID is {answer}.",
        "What was the customer's user ID?",
        "{answer}",
    ),
    (
        "Reference number for this order: {answer}.",
        "What is the order's reference number?",
        "{answer}",
    ),
    (
        "Tracking ID assigned: {answer}.",
        "Recall the tracking ID assigned earlier.",
        "{answer}",
    ),
    (
        "Session token issued: {answer}.",
        "What session token was issued?",
        "{answer}",
    ),
    # ── magic words / phrases ────────────────────────────────────────
    (
        "Note: the magic phrase is {answer}.",
        "What is the magic phrase?",
        "{answer}",
    ),
    (
        "The passphrase you'll need later is {answer}.",
        "What is the passphrase?",
        "{answer}",
    ),
    (
        "Codeword for the operation: {answer}.",
        "What was the operation's codeword?",
        "{answer}",
    ),
    # ── numbers / quantities / targets ───────────────────────────────
    (
        "Important: the quarterly target value is {answer}.",
        "What is the quarterly target value?",
        "{answer}",
    ),
    (
        "Total budget allocated: {answer}.",
        "How much budget was allocated?",
        "{answer}",
    ),
    (
        "The threshold setting was configured to {answer}.",
        "What was the threshold setting?",
        "{answer}",
    ),
    (
        "Result of the calculation: {answer}.",
        "What was the calculation result?",
        "{answer}",
    ),
    # ── names / people ───────────────────────────────────────────────
    (
        "The contact person assigned is {answer}.",
        "Who is the contact person?",
        "{answer}",
    ),
    (
        "Project lead: {answer}.",
        "Who is the project lead?",
        "{answer}",
    ),
    (
        "The patient's name on file is {answer}.",
        "What was the patient's name?",
        "{answer}",
    ),
    # ── dates / times ────────────────────────────────────────────────
    (
        "Scheduled date for the meeting: {answer}.",
        "When is the meeting scheduled?",
        "{answer}",
    ),
    (
        "The deadline is set to {answer}.",
        "What is the deadline?",
        "{answer}",
    ),
    (
        "Event timestamp: {answer}.",
        "What was the event timestamp?",
        "{answer}",
    ),
    # ── locations / addresses ────────────────────────────────────────
    (
        "Meeting location: {answer}.",
        "Where is the meeting?",
        "{answer}",
    ),
    (
        "Server endpoint: {answer}.",
        "What is the server endpoint?",
        "{answer}",
    ),
    (
        "The shipping address recorded was {answer}.",
        "What is the shipping address?",
        "{answer}",
    ),
    # ── preferences / settings ───────────────────────────────────────
    (
        "User's preferred language is {answer}.",
        "What language does the user prefer?",
        "{answer}",
    ),
    (
        "Default time zone: {answer}.",
        "What is the default time zone?",
        "{answer}",
    ),
    (
        "Currency setting: {answer}.",
        "What currency is configured?",
        "{answer}",
    ),
    # ── JSON-ish / structured ────────────────────────────────────────
    (
        "Config: {{\"key\": \"{answer}\"}}.",
        "What value is stored under \"key\" in the config?",
        "{answer}",
    ),
    (
        "The API response field `id` returned: {answer}.",
        "What was the value of the `id` field?",
        "{answer}",
    ),
    # ── conversational / casual ──────────────────────────────────────
    (
        "Hey, just so you remember — the answer they need is {answer}.",
        "What was the answer I needed to remember?",
        "{answer}",
    ),
    (
        "Btw, the password he gave us was {answer}, don't lose it.",
        "What password did he give us?",
        "{answer}",
    ),
    # ── implicit / mid-sentence ──────────────────────────────────────
    (
        "Among the records reviewed, only one stood out — namely, {answer}.",
        "Which record stood out from the rest?",
        "{answer}",
    ),
    (
        "The witness identified the suspect as {answer}, no other matches.",
        "Who did the witness identify?",
        "{answer}",
    ),
    # ── factual / encyclopedic ───────────────────────────────────────
    (
        "Recently published research credits the discovery to {answer}.",
        "Who is credited with the discovery?",
        "{answer}",
    ),
    (
        "The original specification was authored by {answer}.",
        "Who authored the original specification?",
        "{answer}",
    ),
]


_FIRST_NAMES = [
    "Aisha", "Bjorn", "Camila", "Dmitri", "Elena", "Farouk", "Gabriela",
    "Hiroshi", "Ingrid", "Jamal", "Keiko", "Liam", "Maya", "Nikolai",
    "Olufemi", "Priya", "Quentin", "Rosalind", "Saoirse", "Tariq",
    "Uma", "Viktor", "Wren", "Xiulan", "Yuki", "Zara",
]
_LAST_NAMES = [
    "Adeyemi", "Bergstrom", "Castellanos", "Delacroix", "Ezeji",
    "Fernandez", "Gunnarsson", "Hartwell", "Iqbal", "Johansson",
    "Kowalski", "Lindqvist", "Mukherjee", "Nakashima", "Okafor",
    "Petrov", "Quintero", "Rasmussen", "Saito", "Tanaka",
    "Underwood", "Vasquez", "Whitfield", "Xu", "Yamamoto", "Zou",
]
_PHRASE_WORDS = [
    "raven", "crimson", "lattice", "tundra", "echo", "vector", "harvest",
    "pivot", "marble", "kindle", "azure", "obsidian", "phoenix", "cipher",
    "meridian", "verdigris", "halcyon", "saffron", "thistle", "willow",
]
_LOCATIONS = [
    "Reykjavik, Iceland", "Kyoto, Japan", "Marrakech, Morocco",
    "Buenos Aires, Argentina", "Stockholm, Sweden", "Lagos, Nigeria",
    "Vancouver, Canada", "Lisbon, Portugal", "Cape Town, South Africa",
    "Helsinki, Finland", "Edinburgh, Scotland", "Tashkent, Uzbekistan",
    "Auckland, New Zealand", "Dakar, Senegal", "Hanoi, Vietnam",
]
_LANGUAGES = [
    "Mandarin", "Swahili", "Tamil", "Quechua", "Welsh", "Icelandic",
    "Yoruba", "Esperanto", "Vietnamese", "Tagalog", "Norwegian", "Hebrew",
    "Portuguese", "Hungarian", "Korean",
]
_TIMEZONES = [
    "America/Los_Angeles", "Europe/Berlin", "Asia/Tokyo", "Pacific/Auckland",
    "Africa/Nairobi", "America/Sao_Paulo", "Europe/Istanbul", "Asia/Dubai",
]
_CURRENCIES = [
    "USD", "EUR", "JPY", "GBP", "CNY", "AUD", "CHF", "INR", "BRL", "ZAR",
]


def _random_answer(rng: random.Random, kind: str = "mixed") -> str:
    """Generate a unique tokenize-safe answer string. Returns a random
    type if kind=="mixed" (default) so the model can't shortcut on
    answer-shape priors."""
    # Mixed mode: rotate through a curriculum of answer types so memory
    # has to encode actual content not just a fixed shape pattern.
    if kind == "mixed":
        kinds = ["alphanum_short", "alphanum_long", "phrase",
                 "name", "number", "date", "location", "language",
                 "timezone", "currency_amount", "url_id"]
        kind = rng.choice(kinds)

    if kind == "alphanum_short":
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        return "".join(rng.choice(chars) for _ in range(6))
    if kind == "alphanum_long":
        chars = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghjkmnpqrstuvwxyz23456789"
        return "".join(rng.choice(chars) for _ in range(12))
    if kind == "phrase":
        return f"{rng.choice(_PHRASE_WORDS)}-{rng.choice(_PHRASE_WORDS)}-{rng.choice(_PHRASE_WORDS)}"
    if kind == "name":
        return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"
    if kind == "number":
        return str(rng.randint(1000, 999999))
    if kind == "date":
        # ISO-ish, mostly synthetic — tokenizer-stable with leading zeros.
        y = rng.randint(2018, 2030)
        m = rng.randint(1, 12)
        d = rng.randint(1, 28)
        return f"{y}-{m:02d}-{d:02d}"
    if kind == "location":
        return rng.choice(_LOCATIONS)
    if kind == "language":
        return rng.choice(_LANGUAGES)
    if kind == "timezone":
        return rng.choice(_TIMEZONES)
    if kind == "currency_amount":
        return f"{rng.randint(100, 999999):,} {rng.choice(_CURRENCIES)}"
    if kind == "url_id":
        # short URL/ID-like slug
        chars = "abcdefghijklmnopqrstuvwxyz0123456789"
        return f"id-{''.join(rng.choice(chars) for _ in range(8))}"
    raise ValueError(f"unknown kind: {kind}")


@dataclass
class NeedleDoc:
    """A synthetic document with a planted needle + answer at the end."""

    text: str            # the full document text
    needle_pos_chars: int   # character offset of the needle in text
    query_pos_chars: int    # character offset of the query (= answer site)
    answer: str             # the gold answer string
    target_distance: int    # tokens between needle and query (approximate)


def _split_filler(text: str, n_chunks: int) -> list[str]:
    """Split filler text into roughly equal chunks (by character)."""
    L = len(text)
    chunk_size = max(L // n_chunks, 1)
    return [text[i * chunk_size : (i + 1) * chunk_size] for i in range(n_chunks)]


def make_needle_doc(
    filler_text: str,
    *,
    target_distance_tokens: int,
    rng: random.Random | None = None,
    answer_kind: str = "mixed",
) -> NeedleDoc:
    """Build one needle-in-haystack document.

    Document layout: `[pre-filler] [needle statement] [post-filler] [query+answer]`.

    The needle position is RANDOMIZED within the doc (not always at 0):
    pre-filler length is drawn from [0, max_pre] where max_pre is chosen
    so that needle→query distance still ≈ target_distance_tokens. This
    forces the model to actually detect "this is a needle worth
    remembering" from context, instead of learning "always encode
    whatever's at position 0."

    Args:
        filler_text: a body of natural text to use as filler (FineWeb,
                     SlimPajama, etc.).
        target_distance_tokens: approximate token distance from NEEDLE
                                to QUERY. >2K forces memory.
        rng: random source (default: fresh).
        answer_kind: "mixed" (default — random per-doc), or one of the
                     specific kinds in `_random_answer`.
    """
    rng = rng or random.Random()
    tok = get_tokenizer()

    # Pick a template + answer
    statement_tpl, query_tpl, answer_tpl = rng.choice(_NEEDLE_TEMPLATES)
    answer = _random_answer(rng, kind=answer_kind)
    statement = statement_tpl.format(answer=answer)
    query = query_tpl
    answer_text = answer_tpl.format(answer=answer)

    # Token-level chars-per-token estimate (English text via Llama-3 BPE).
    chars_per_token = 4

    # Total filler needed = enough for needle→query distance to be ~target.
    # Randomize the SPLIT between pre- and post-filler so the needle can
    # be anywhere in the doc (not always position 0).
    total_filler_chars = target_distance_tokens * chars_per_token
    # max pre-filler — keep some room for needle statement itself + at
    # least 200 chars of post-filler so the doc has structural variety.
    statement_chars = len(statement)
    pre_max = max(0, total_filler_chars - statement_chars - 200)
    pre_chars = rng.randint(0, pre_max) if pre_max > 0 else 0
    post_chars = max(0, total_filler_chars - pre_chars - statement_chars)

    # Build filler text long enough for both halves.
    filler_chars_needed = pre_chars + post_chars + 100
    if len(filler_text) < filler_chars_needed:
        repeats = (filler_chars_needed // max(len(filler_text), 1)) + 1
        filler_text = filler_text * repeats
    pre_filler = filler_text[:pre_chars]
    post_filler = filler_text[pre_chars : pre_chars + post_chars]

    # Compose: [pre_filler] [statement] [post_filler] [query+answer]
    document = (
        pre_filler + ("\n\n" if pre_filler else "")
        + statement + "\n\n"
        + post_filler + "\n\n"
        + query + " " + answer_text + "\n"
    )
    needle_pos_chars = len(pre_filler) + (2 if pre_filler else 0)
    query_pos_chars = needle_pos_chars + statement_chars + 2 + len(post_filler) + 2

    # Token-distance metadata (for analytics).
    pre_token_len = len(tok.encode(pre_filler, add_special_tokens=False)) if pre_filler else 0
    needle_token_len = len(tok.encode(statement, add_special_tokens=False))
    post_token_len = len(tok.encode(post_filler, add_special_tokens=False)) if post_filler else 0
    actual_distance = needle_token_len + post_token_len

    return NeedleDoc(
        text=document,
        needle_pos_chars=needle_pos_chars,
        query_pos_chars=query_pos_chars,
        answer=answer,
        target_distance=actual_distance,
    )


def generate_needle_docs(
    fillers: Iterable[str],
    *,
    target_distances: list[int] = (3000, 8000, 16000, 32000),
    seed: int = 0,
    answer_kind: str = "mixed",
) -> Iterator[NeedleDoc]:
    """Generate needle docs from an iterable of filler bodies.

    Cycles through `target_distances` so the dataset has a curriculum of
    increasing memory pressure.
    """
    rng = random.Random(seed)
    for i, filler in enumerate(fillers):
        target = target_distances[i % len(target_distances)]
        yield make_needle_doc(
            filler, target_distance_tokens=target, rng=rng,
            answer_kind=answer_kind,
        )
