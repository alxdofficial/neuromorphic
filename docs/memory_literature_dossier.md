# Memory-for-LLM Literature Dossier (2026-06-04)

Consolidated findings from two multi-agent literature workflows, run while diagnosing why our
Stage-A side-car memory shows **"memory-on ≈ memory-shuffled"** (the trained memory carries no
key-addressable binding: a linear probe and the LM both read REAL == SHUF == chance, while a
bound-KV oracle reads 0.625). Raw outputs:
- Dossier A (original architecture papers): `wf_4f77041e` → `/tmp/.../tasks/wajd3drru.output`
- Dossier B (these architectures used AS external memory for an LLM): `wf_036598e1` → `/tmp/.../tasks/wqzj5b57l.output`

---

## TL;DR — the cure, and why our config fails

**Every clean "memory-on ≫ memory-shuffled" result in the literature shares three structural
ingredients. Our setup has none of them, and our exact configuration appears on NO published
success path.**

1. **Explicit, content-DERIVED key→value index** — the address is *derived from* the content, so a
   shuffle mechanically returns the wrong value. NOT a learned scatter-write read by free attention.
2. **Write-side address/content DECOUPLING** — keys are frozen-after-init (DKVB) or are the literal
   cached activation (GRACE, Memorizing-T, LongMem), or written by a closed-form / delta rule
   (Larimar, Infini-attention). The answer-shaping gradient can never rotate/blur the address.
3. **A co-adapted reader OR the LLM's own attention** — a *fully-frozen decoder + a separately-trained
   reader over a learned blob* is the **one regime with no verified success precedent.** That is
   exactly our config.
   Plus: **reconstruction-from-memory installs the binding** (ICAE/500x/EMAT/Larimar) — if the answer
   is regenerable *from* memory, the binding provably lives in memory. End-task QA loss alone never
   forces this.

**Compression is NOT the wall** — every verified success binds at ratio≈1 (kNN-LM, Memorizing-T, EMAT).
The wall is the missing explicit index + decoupled write + co-adapted reader. (High compression ratios
in the original papers are a *vision* artifact — pixels are redundant; language is lean.)

---

## Dossier A — what the ORIGINAL papers trained on (used AS the model)

| Work | Data | Objective | Reader | Compression | Key→value index? |
|---|---|---|---|---|---|
| VQ-VAE (van den Oord 2017) | vision/audio (no text/QA) | recon + codebook + commitment | trained from scratch | ~40× | No (NN *quantizer*) |
| Slot Attention (Locatello 2020) | vision (CLEVR) | pixel-MSE recon | trained from scratch | ~400–1000× | No (unconditional) |
| PKM / Memory Layers (Lample 2019; Berges 2024) | text LM, **1T tokens** | next-token CE | host trained jointly | adds params | **Yes** (product-key) |
| Memorizing Transformers / kNN-LM | text LM | next-token CE / none | trained / frozen | **~1** | **Yes** (external kNN) |
| Mamba (Gu & Dao 2023) | Pile 300B | next-token CE | from scratch | fixed state | in-context only |
| Gisting / AutoCompressor | instructions / Pile | instr-FT / segment-LM | **decoder fully fine-tuned** | 26× / 1–2 OoM | No |
| ICAE / 500×Compressor | Pile / arXiv-QA | AE-recon→QA | encoder **LoRA** vs frozen dec | 4× / 6–480× | partial (QA-trained) |

**Four levers; working methods have ≥2, our main probe had 0:** (I) trained/adapted reader;
(II) objective that forces content in (recon/LM); (III) compression pressure; (IV) explicit
key/value-split addressing. **Two escape routes:** (A) no index → force structure with heavy
compression + co-adapted reader (vision); (B) ratio≈1 → supply an explicit external index + trained
reader (kNN-LM/Memorizing-T). We are Route A minus compression, Route B minus the index.

kNN-LM's own ablation at matched ppl: implicit-in-weights **+0.1 ppl** vs explicit index **+1.9** →
parametric/dense storage is **not** re-addressable.

---

## Dossier B — who used these AS external memory for an LLM (the precedents that matter)

| Work | Arch-as-memory | Backbone | Write | Read addressing | Memory helps content-specifically? |
|---|---|---|---|---|---|
| **Larimar** (IBM, ICML 2024, 2403.11901) | Kanerva associative matrix between enc & dec | full (write training-free at inference) | closed-form M=W0†Z, one-shot/fact | **explicit** W̄=ZM† | **Yes (strongest)** — edit-success + selective fact **forgetting** probe |
| **LongMem** (NeurIPS 2023, 2306.07174) | cached frozen-LLM KV bank + trainable SideNet | **frozen** + side-net | cache the LLM's own K,V | **explicit** token-query→chunk-key kNN | Yes — *the honest frozen-backbone template* |
| **Memorizing Transformers** (ICLR 2022) | external (k,v) store, 262K | full | append model's own K,V | **explicit** approx-kNN, gated | Yes |
| **kNN-LM** (ICLR 2020) | datastore (ctx-rep→next-token) | **frozen** (no training) | store hidden→token, ratio≈1 | **explicit** kNN | Yes |
| **GRACE** (NeurIPS 2023, 2211.11031) | discrete key-value codebook at one layer | **frozen** | **key = raw cached activation; learn only VALUE** | **explicit** ε-ball NN + defer | Yes — ε-ball IS a built-in on/off control |
| **Discrete Key-Value Bottleneck** (ICML 2023) | C VQ codebooks, frozen enc→small dec | **frozen** | **VQ-init keys then FREEZE; learn only values** | **explicit** VQ argmin/head | Yes — class-incremental forgetting probe |
| **EMAT** (EMNLP 2022, 2210.16773) | dense KV memory of QA pairs | full | key/value **auto-encoded** | **explicit** MIPS kNN | Yes — ablating key/value-AE costs >10 EM |
| **ARMT** (2024, 2407.04841) | RMT tokens + per-layer assoc. delta-rule matrix | full | delta-rule kᵀv | **explicit** linear-attn q·M | Yes — BABILong 79.9% single-fact / 50M tok |
| **Infini-attention** (Google 2024) | per-head compressive assoc. matrix | full / continual-pretrain | delta-rule kᵀv | **explicit** σ(Q)M, gated | Yes — 1M passkey |
| **MemoryLLM / M+** (ICML 2024/25) | in-layer latent pool (+ LT store) | full (gradient-free overwrite write) | drop-and-add overwrite | soft (M+: co-trained retriever) | Yes — retention curves |
| **CAMELoT** (2024) | non-param consolidated KV/layer | **frozen, training-free** | novelty/recency consolidation | **explicit** kNN, LLM's own attn reads | Yes |
| ICAE / 500×Compressor | k learnable soft memory slots | **LoRA** | LoRA-enc emits slots | **soft** prompt (positional) | Yes via **[AE] reconstruction** objective |

**Anti-precedent (do NOT model the fix on this):** MEMORY-VQ — VQ only *compresses* an
already-retrieved memory; freezing the model and updating only codes *hurts*. Binding lives in the
outer retriever, not the codebook.

### Gaps (themselves findings)
- **Slot-attention as LLM episodic key→value memory: essentially a GAP.** Nobody uses competitive
  slot-attention as a question→answer store; the competition binds slots to input *positions* under
  reconstruction — no key→value template, will **not** fix the shuffle pathology.
- **Mamba/SSM as a frozen-LLM side-car episodic memory with key→value: a GAP.** "Lost in State Space"
  shows pretrained-Mamba states are anisotropic (cos≈0.9999) — un-addressable without the output
  projection, a direct mirror of our shuffle-invariance.
- **Our exact config (learned scatter-write + free-attention read + end-task-QA loss + frozen decoder
  + separately-trained reader): no verified success precedent.**

### Closest precedents
- **In ROLE: Larimar** — write a passage once, later read the bound value by querying with the
  question; its *selective-forgetting probe* is literally our memory-on>shuffle test. (But it
  co-trains enc/dec.)
- **Under the FROZEN-backbone constraint: LongMem** — frozen backbone + separate trainable side-car
  reader + explicit content-derived kNN key→value. The honest template for us.
- **Write discipline: GRACE / DKVB** — frozen-key / learned-value split.
- **Objective: EMAT** — key auto-encodes the question, value auto-encodes the answer.

---

## The two value propositions (the WHY — what we're actually building toward)

1. **Fixed-footprint memory over unbounded context** — constant memory size regardless of history
   length (O(1) state vs O(n) KV cache). Family: Mamba/SSM, RMT, Infini-attention, Titans.
2. **Continuous test-time adaptation** — the memory updates *during inference*, no gradient, implicitly
   learning preferences (agentic). Family: fast-weights, Test-Time Training (Sun 2024), Titans; also
   the project's own predictive-coding/Hebbian roots.

Both want a **recurrent, fixed-size, incrementally-updated** state — not a one-shot slot encoder.
**Our Stage-A one-shot 108-token probe tested neither** (it fits in context → memory not even
required → a marginal-guessing shortcut competes with real memory use). See `project_value_proposition`.

---

## The recipe the literature points to (for our fair re-run)

- **WRITE:** explicit key→value index, **address derived from content + decoupled** — freeze the key
  (GRACE/DKVB) or closed-form solve (Larimar) or delta/partial-overwrite. Value = a representation the
  reader can consume.
- **READ:** a deterministic query→address→value operator (kNN / dot-product / VQ-argmin / ε-ball) with
  an **OFF/defer path** — a shuffled key falls outside the matching ball and *physically cannot fire*,
  which manufactures memory-on > shuffle **by construction**.
- **OBJECTIVE:** **reconstruction-of-the-answer-from-memory through the actual read path** (ICAE/500x/
  EMAT/Larimar) — "reconstruction IS the task," no aux loss. EMAT: force the key to reconstruct the
  question and the value to reconstruct the answer.
- **READER:** co-adapt it (write-side/reader LoRA) **or** use the LLM's own attention over an explicit
  index (CAMELoT). Never a fully-frozen decoder + separately-trained reader (no success precedent).
- **COMPRESSION:** start ratio≈1 — it is NOT the wall.

### Proposed next experiment (the fast, shortcut-proof gate)
**MQAR (Multi-Query Associative Recall)** — the literature-standard memory diagnostic (Zoology/Based) —
with an explicit decoupled key/value memory + a small co-adapted read head. Stream N random (key,value)
pairs, query keys, emit values; sweep N for the capacity curve per mechanism. This isolates "does the
write+read store and retrieve a key→value binding at all" (fast, synthetic, shortcut-proof) before
integrating the winner with frozen Llama (LongMem template). See chat proposal 2026-06-04.
