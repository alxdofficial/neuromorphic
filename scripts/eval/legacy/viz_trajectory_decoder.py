#!/usr/bin/env python3
"""viz_trajectory_decoder.py — Tier 4: inversion probe — trajectory → text.

**STUB — not implemented yet.** The most ambitious interpretability tool
we have. Trains a decoder that translates memory trajectories back to
natural language, then uses it to *speak* the manifold's contents.

## Big idea

Once a trajectory-memory model is sufficiently trained, every read/write
trajectory is a compressed representation of language content. A small
decoder `trajectory → text` lets us invert that compression — pull the
language back out and inspect what each concept (and each path) is
actually encoding.

This is structurally the same as Morris et al. 2023 "Text Embeddings
Reveal Almost as Much as Text" (inverts sentence embeddings to text),
applied to our concept-trajectory representation instead.

## What it would produce

  A. **Concept dictionary** (`outputs/wave{N}/concept_dictionary.md`)
     For each of the 4096 concepts, synthesize a length-1 trajectory
     visiting only that concept, decode 100 text samples, summarize as
     "concept #042 → 'France', 'Paris', 'European capital', ...".
     The interpretability gold standard.

  B. **Interpolation panel** (`outputs/wave{N}/viz_interpolations.md`)
     Pick concept pairs (A, B). Decode along the shortest graph path
     A→B. Should give semantic interpolations if the manifold has
     learned a meaningful metric.

  C. **Counterfactual probe** (`outputs/wave{N}/viz_counterfactuals.md`)
     Take a real document's trajectory, swap one hop (concept X → Y),
     decode the perturbed trajectory. Shows what each hop contributes.

  D. **Needle-recall direct readout** (`outputs/wave{N}/viz_needle_decode.md`)
     For each needle doc, decode the write_trajectory at the needle
     position. If the decoder emits the needle's actual content, that's
     direct evidence of memory encoding (much stronger than the
     overlap-vs-distance probe in Tier 3).

## Implementation outline

### Decoder design

Cheapest option: re-use the frozen Llama backbone with a *second*
`MemInjectLayer` (separate bridge weights). Cross-attention input shape
is identical to the trained model's — `[J·K_read, D_concept]`. Train
only the bridge MLP + cross-attn weights.

Alternative (cleaner causal story): small (~50M-param) from-scratch
transformer that ONLY sees the trajectory. Less fluent output but no
Llama-pretraining shortcuts.

### Training data

Every Wave 1 window emits `(read_visited_ids, read_visited_states,
window_text)`. Run inference on existing val sets to dump ~50K such
triples. Format:

```
{
    "concept_ids":    [J, K_read],      # which concepts were visited
    "concept_states": [J, K_read, D],   # the state vectors at visit time
    "window_text":    str,              # the 256 target tokens (decoded)
    "source":         "fineweb_edu" | ...,
}
```

### Training

```
decoder_loss = NTP(decoder(trajectory) , window_text)
# Optionally: mask N% of trajectory hops to encourage redundancy
# learning, like denoising autoencoder pretraining.
```

### Probing API

```python
def speak_concept(concept_id: int, n_samples: int = 100) -> list[str]:
    # Synthesize trajectory [J=1, K=1] visiting only concept_id
    # Decode with temperature sampling
    ...

def speak_path(concept_ids: list[int]) -> str:
    # Synthesize trajectory of given length
    ...

def speak_interpolation(a: int, b: int) -> list[str]:
    # Find shortest path a→b on edge_indices
    # Decode at each step
    ...
```

## Risks and mitigations

1. **Distribution mismatch on synthesized paths.** Constrain synthesized
   trajectories to lie within graph-neighbor adjacency (small-world).
2. **Mode collapse to plausible English.** Hold out 50% of concepts
   during decoder training; if decode quality on held-out concepts is
   still high, the decoder is shortcutting via Llama's pretraining —
   switch to from-scratch decoder.
3. **Llama-pretraining leakage.** From-scratch decoder is the gold
   standard. Costlier to train but clean causal story.
4. **Ambiguity.** Use temperature sampling + ensemble of N decoded
   samples per probe.

## Cost

- **GPU required** for decoder training (~3-5 hours on the available H100).
- **Disk:** ~1 GB for the (trajectory, text) corpus.
- **Output:** concept_dictionary.md is the headline. ~50-200 KB markdown.

## Why this is paper-worthy

Most memory-augmented LM work reports task performance only. A concept
dictionary extracted from a learned manifold would be a unique
contribution to the interpretability story for structured memory in LMs.
Closest published analog is Anthropic's SAE work (Bricken et al. 2023,
Templeton et al. 2024 "Scaling Monosemanticity"), which extracts static
post-hoc dictionaries from frozen-model residual streams. Ours would be
*online, trainable, and traversable* — qualitatively different.

See `scripts/diagnostics/README.md` for the full Tier 1-4 roadmap.
"""

raise SystemExit(
    "viz_trajectory_decoder.py is a Tier 4 stub — not yet implemented. "
    "See the docstring for the design and scripts/diagnostics/README.md "
    "for the roadmap."
)
