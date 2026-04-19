"""Passphrase-retrieval data — prompt assembly for multi-needle recall.

Task format:
    <instruction>
    <filler>
    OK 1st the pass phrase is: <phrase_1>.
    <filler>
    The 2nd pass phrase is: <phrase_2>.
    ...
    The Nth pass phrase is: <phrase_N>.
    <filler>
    Now please give the pass phrases in order separated by a period with no preamble.

Model is expected to generate:
    <phrase_1>. <phrase_2>. ... <phrase_N>.

This module provides the pure-text primitives (no tokenizer dependency) so
the prompt structure can be unit-tested without HF loads. The CLI at
`scripts/build_passphrase_data.py` composes these with a tokenizer to size
filler token budgets and emit jsonl.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


INSTRUCTION = (
    "You will be given several pass phrases hidden in filler text. "
    "Memorize each one in the order you see it. "
    "After the filler I will ask you to recite them back in order."
)

FINAL_PROMPT = (
    "Now please give the pass phrases in order separated by a period "
    "with no preamble."
)

FILLER_PARAGRAPH = (
    "The grass is green and the sky is blue. "
    "The sun rises in the east and sets in the west every single day. "
    "Coffee is a warm beverage popular in the morning with many people. "
    "Rivers flow from the mountains down toward the distant ocean shore. "
    "Birds build their nests in the tallest trees during the spring months. "
)

# ~60 curated sentence-long passphrases. Each:
#   - Has no period inside (so `split('.')` is unambiguous).
#   - Is concrete and unambiguous (easy to verify surface form).
#   - Is 8-14 words, typical Llama token length 10-18.
# Content ranges broad so the pool doesn't collide with filler semantics.
PHRASE_POOL: tuple[str, ...] = (
    "The ancient lighthouse stood watch over the rocky shore",
    "She traded her old guitar for a well used fishing pole",
    "The chef prepared a soup with seven different root vegetables",
    "A small red balloon floated gently past the kitchen window",
    "The hiker found a silver coin buried beneath the green moss",
    "The old clock in the hallway struck midnight with a heavy chime",
    "Her grandmother grew tomatoes in a sunny corner of the garden",
    "A curious fox darted across the narrow path into the dark woods",
    "The baker pulled a fresh loaf of bread from the stone oven",
    "He kept a small notebook of sketches in his left coat pocket",
    "The children chased fireflies across the meadow until late evening",
    "A thin layer of frost covered the windshield every morning in October",
    "The librarian wore reading glasses on a silver chain around her neck",
    "The sailor tied three complicated knots into the heavy wet rope",
    "Her father taught her how to whistle using a thick blade of grass",
    "The painter mixed a shade of blue from cobalt and ultramarine",
    "A wooden bridge crossed the creek behind the abandoned mill",
    "The cat curled up on the warm laundry folded inside the basket",
    "He found his lost keys inside a drawer full of tangled batteries",
    "The mountain trail ended at a small lake fed by melted snow",
    "She wrote her best poems on the back of used grocery receipts",
    "The train pulled into the station exactly six minutes behind schedule",
    "A single bright star appeared above the church steeple at dusk",
    "The gardener planted rows of sunflowers along the wooden fence",
    "He wore his grandfather's pocket watch on a long brass chain",
    "The bakery on the corner smelled like cinnamon and warm butter",
    "A flock of geese flew south in a perfect angled line",
    "The violinist tuned her instrument before every morning rehearsal",
    "He carved small wooden birds from leftover scraps of cedar wood",
    "The cabin in the woods had a red door and a green metal roof",
    "She collected old postcards from small towns in every state",
    "The river froze solid for only three days in the entire winter",
    "A bowl of apples sat on the kitchen table near the sunny window",
    "The dog buried his favorite bone under the oak tree out back",
    "He repaired broken watches in the back room of a jewelry store",
    "The teacher kept a jar of marbles on her desk for quiet rewards",
    "The canoe drifted slowly past the reeds along the quiet riverbank",
    "She knitted a blue scarf every year for her youngest grandson",
    "The astronomer watched the meteor shower from the flat rooftop",
    "A heavy stone marker showed the border between the two farms",
    "The blacksmith hammered a horseshoe into shape on the anvil",
    "He brewed strong coffee in a battered pot over the open campfire",
    "She painted watercolors of the harbor from the upstairs balcony",
    "The old map showed a forgotten path through the thick pine forest",
    "The storm shook the shutters and rattled the windows all night",
    "A worn leather journal sat on the shelf above the wooden desk",
    "The mailman delivered three letters and a package wrapped in brown paper",
    "She grew herbs in small clay pots arranged on the sunny windowsill",
    "The shepherd counted his sheep twice before closing the wooden gate",
    "He fixed the broken fence along the edge of the empty pasture",
    "The bookstore kept a sleeping cat on the shelf near the door",
    "A long dirt road wound through fields of yellow wheat and barley",
    "The young pianist practiced the same song for three straight hours",
    "She found a rare seashell on the beach during low tide",
    "The carpenter built a tall bookcase from reclaimed oak planks",
    "He kept a collection of smooth river stones on the windowsill",
    "The fisherman cast his line into the pond just before sunrise",
    "A narrow footbridge spanned the small brook behind the farmhouse",
    "The candle maker dipped long wicks into pots of hot yellow wax",
    "She tied her hair back with a piece of bright red ribbon",
)


def ordinal(k: int) -> str:
    """1 -> '1st', 2 -> '2nd', 3 -> '3rd', 4 -> '4th', ..., 11-13 -> 'th'."""
    assert k >= 1
    if 10 <= (k % 100) <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def phrase_prefix(k: int) -> str:
    """Prefix introducing the k-th phrase inside the prompt."""
    if k == 1:
        return "OK 1st the pass phrase is:"
    return f"The {ordinal(k)} pass phrase is:"


@dataclass
class PromptParts:
    """Prompt assembled as ordered text segments.

    Segments are ordered, and `filler_slots` indicates which segments are
    filler (True) vs structural (False). This lets the CLI size filler
    token budgets per slot when building a fixed-length training example.
    """
    segments: list[str]
    filler_slots: list[bool]
    phrases: list[str]

    def assemble(self) -> str:
        return "\n".join(self.segments)


def build_prompt_parts(phrases: list[str], filler_text_per_gap: list[str] | None = None) -> PromptParts:
    """Assemble the N+2 prompt segments given phrases and per-gap filler.

    Slot layout (for N=3):
        [INSTRUCTION, FILLER0, PHRASE1, FILLER1, PHRASE2, FILLER2, PHRASE3, FILLER3, FINAL_PROMPT]
    So there are N+1 filler slots between instruction and final prompt.
    """
    N = len(phrases)
    assert N >= 1
    for p in phrases:
        assert "." not in p, f"phrase must not contain periods: {p!r}"
        assert p.strip() == p, f"phrase must be trimmed: {p!r}"
    if filler_text_per_gap is None:
        filler_text_per_gap = [""] * (N + 1)
    assert len(filler_text_per_gap) == N + 1, (
        f"need N+1={N+1} filler slots, got {len(filler_text_per_gap)}")

    segments: list[str] = [INSTRUCTION]
    is_filler: list[bool] = [False]

    for k, phrase in enumerate(phrases, start=1):
        segments.append(filler_text_per_gap[k - 1])
        is_filler.append(True)
        segments.append(f"{phrase_prefix(k)} {phrase}.")
        is_filler.append(False)

    segments.append(filler_text_per_gap[N])
    is_filler.append(True)
    segments.append(FINAL_PROMPT)
    is_filler.append(False)

    return PromptParts(segments=segments, filler_slots=is_filler, phrases=list(phrases))


def build_answer(phrases: list[str]) -> str:
    """Ground-truth continuation: phrases joined by '. ' with trailing period."""
    return ". ".join(phrases) + "."


def sample_phrases(rng: random.Random, n: int, pool: tuple[str, ...] = PHRASE_POOL) -> list[str]:
    """Sample `n` distinct phrases from the pool."""
    assert n <= len(pool), f"pool has {len(pool)} phrases; cannot sample {n} distinct"
    return rng.sample(pool, n)
