# Data Plan — Full-Corpus Training & Test-Eval Phases

Plan for the two phases *after* the architecture-scrutiny phase, synthesizing a research sweep (2025-era
memory/compression literature) with our prior data audit. Companion to `SCRUTINY_PHASE_DATA.md` (phase-0
data) and `DATA_TASK_GUIDE.md`. **Status: PLAN — not yet TRAINED on / not yet evaluated.** Dataset
numbers/licenses were research-gathered; verify before ingesting (see "Verify-before-use" at the end).

> **UPDATE (2026-07-08): the Phase-1 sources + Phase-2 eval readers below have SHIPPED as code** (they
> exist in `SOURCE_REGISTRY`/`REGISTRY` and can be loaded), but are **not yet in `DEFAULT_TRAIN_MIX`** —
> the architecture-scrutiny sweep (`SCRUTINY_PHASE_DATA.md`) still trains only on the 5-task mix. Shipped
> Phase-1 sources: `wildchat`, `lmsys_chat` (gated), `msc`, `qasper`, `longcite`, `govreport`, `pg19`,
> `ruler_niah`, `babilong_train`, `wikibigedit`, `swe_trajectories`, `perltqa`. Shipped Phase-2 eval
> readers: `longmemeval`, `longbench`, `infinitebench`, `niah`. Still NOT built: `streaming_update`/
> `trajectory`/`tool_call` task styles, the `grpo` objective, weighted-mix/`Curriculum` wiring, the
> `books`/`arxiv`/`long_web`/`long_wiki`/`ultrachat`/`ms_tod`/`repo`/`commits`/`swebench_train`/
> `webarena_traces`/`aitw`/`toolbench`/`agent_instruct`/`gsm8k`/`math`/`apps` sources (see also
> the former `docs/PHASE_PLAN.md`, now MERGED into this doc — its `books` ≈ this doc's shipped
> `pg19`, its `multisession_chat` ≈ this doc's shipped `msc`).

## The three phases

| phase | what | data | objective |
|---|---|---|---|
| **0. Scrutiny** (done) | pick the memory architecture | synthetic bio/babi + real QA/continuation/MAE (small 135M) | CE + SHUF/REAL binding gate |
| **1. Full-corpus training** | train the *chosen* arch to be genuinely useful | real chat/agent/GUI/coding/QA at scale | SFT compression + streaming-write finetune + **GRPO** |
| **2. Test-eval** | measure quality + headline head-to-heads | eval-only benchmarks | task success vs ceiling/floor/competitors |

**The thesis anchor** (what makes our story defensible, from `project_eval_protocol`): *good memory =
compression × write/update (stability-plasticity).* We are an **always-on implicit memory cache** with
streaming **write/update** — the axis **no context-compression paper tests** (they're all single-shot
compress-then-read). Our competitors are **RAG + long-context**, not just the compression baselines.

---

## PHASE 1 — Full-corpus training

Three stages, warm-start → streaming → RL.

### 1a. Compression pretrain (encoder warm-start)
Reuse phase-0 tasks at scale. Corpora: **FineWeb-EDU, SlimPajama/RedPajama, PG19, The-Stack-v2 (code)**.
Objectives: `reconstruct` (ICAE-style AE) + `continuation` (multi-horizon) + **behavioral_kl** context
distillation (the loss-neutrality fix — already built). Scale `total_len` 2k→8k, M up, streaming windows.

### 1b. Streaming write/update finetune — **the differentiator (axis 2)**
The regime nothing else tests: ordered multi-segment streams, **write → overwrite → query the CURRENT
value**, with SHUF−REAL *in the loss* (makes binding non-loss-neutral) + verifiable targets. Needs the
architectural unlock our prior audit flagged: **honor `chunk_offset`** in the encoder (currently `del`'d)
+ a batch contract for ordered sequential segments.

| axis | datasets | task shape | why memory is load-bearing |
|---|---|---|---|
| **multi-session recall** | Multi-Session Chat (MSC), Conversation-Chronicles, DMR/MSC | recall a fact from session *k* at session *n* | answer only in an early session |
| **fact update / stability-plasticity** | TemporalWiki, NewsEdits, **WikiBigEdit**, TAQA | write fact → update → query current | STALE memory must HURT (the forced-forgetting gate) |
| **code cross-file recall** | RepoBench `cross_file_first` over Stack-v2 | predict a symbol defined in an earlier file | natural who-defined-what + EM + in-file control |
| **agentic tool-use (RLVR)** | **Nemotron-RL-Agentic-Tool-Use-Pivot** (97k, CC-BY-4.0), Nemotron-Agentic-v1 (335k SFT) | encode turns-so-far → memory → emit next tool call | recall auth tokens/IDs/lookup results many turns later; `expected_action`/`pass_rate` gate retention |
| **long agent trajectories** | SWE-Gym / SWE-agent rollouts, THUDM/AgentInstruct (format seed) | compress the trajectory → next action | **the defensibly-OPEN novelty** (see positioning) |
| **GUI** | Mind2Web, AndroidControl/AITW, OS-World, WebArena traces | compress screen-state + action history → next action | observations accumulate to 1000s of tokens |

### 1c. GRPO / RL polish — **genuinely novel; get the mechanics right**
**Reward = task success given ONLY the compressed memory** — forces the memory to carry the load-bearing
bits (a memory that drops the auth token fails the checker). Research verdict: *GRPO to train a soft-token
compression encoder read by a FROZEN decoder, with reward from memory-only decoding, does not yet exist in
the literature* — a genuinely novel combination (the whole compression line trains by supervised
recon/CE/behavioral-KL; nearest RL works each miss one axis: Cmprsr = abstractive text not soft tokens;
MEM1 = same model compresses+reasons; "2602.08382" = decoder not frozen; Titans = not RL).

The load-bearing design decisions (from the sweep):
1. **Form the GRPO group by sampling G different MEMORY ENCODINGS, not decoder rollouts.** GRPO's advantage
   is a REINFORCE estimator on the policy being optimized = the *encoder*. Inject encoder stochasticity
   (reparameterized Gaussian head / dropout / Gumbel over discrete memory), decode each memory greedily,
   group-normalize across memories. Fixing the memory and sampling decoder rollouts is evidence-starved
   (variance = decoding noise only) and maximally exposed to zero-variance-group collapse (DAPO).
2. **Two distinct KLs, both useful.** KL-1 = GRPO's β·KL(encoder‖frozen-initial-encoder) — prevents
   memory-policy collapse. KL-2 = **behavioral KL(decoder|full-context ‖ decoder|memory)** = the
   context-distillation objective (already built). **Make behavioral-KL the dense always-on PRIMARY loss
   for fidelity; reserve GRPO for the residual non-differentiable / exact-match reward** — running RL
   without it throws away the better-conditioned gradient for the fidelity sub-problem.
3. **Pitfalls to instrument:** reward hacking/Goodhart (gold peaks then degrades — memory memorizes
   probe-surface cues); GRPO std-normalization length/verbosity bias → an "inflate memory-embedding
   norm/entropy" shortcut; **posterior collapse** (strong AR decoder ignores the latent, KL→0) + VQ
   codebook collapse (EMA/dead-code reset); reward sparsity → **probe MULTIPLE facts per compressed
   context** (densifies reward) + DAPO dynamic sampling to drop zero-variance groups.

**Verifiable-reward sources — the honest picture:** math/code/tool RLVR sets are overwhelmingly
SHORT-context (the problem statement IS the whole input → compression isn't load-bearing). The genuinely
long-context + verifiable material is **synthetic generators** + **repo-scale code envs**:
- **RULER generator** (`NVIDIA/RULER`, `scripts/data/prepare.py`) — unlimited synthetic long-context at any
  length, exact-substring reward, 13 configs incl. variable-tracking (multi-hop). **Fork `prepare.py` to
  inject value-OVERWRITE events → this is our novel streaming-write data** (the "RULER-fork overwrite"
  recipe — not published).
- **BABILong** (eval `RMT-team/babilong` + train `-train-5k-samples`) — facts-in-haystack, single-word
  exact-match, public generator; extends our bAbI line to long context.
- **SWE-Gym / R2E-Gym / Nebius-SWE-agent-trajectories (80k reward-labeled)** — execution-verified repo RL
  where compression must localize the bug.
- Short-context math/code (GSM8K/MATH/NuminaMath/Big-Math-RL-Verified; APPS/CodeContests/xLAM) → make them
  "needles" by embedding at random depth in FineWeb/PG-19 distractors, reward = `math_verify`/tests.

**Cleanest recipe:** behavioral-KL context distillation as the dense warm-start (fixes most fidelity + the
loss-neutrality problem) → layer GRPO for the exact-match/execution residual, group over memory encodings,
encoder-KL anchor. **Start on RULER + BABILong** (unlimited, cheap exact-match, and they cover the
streaming-write/overwrite gap), validate the loop, then add QASPER/LongCite (real long-doc) + SWE-Gym.

### Phase-1 top-8 dataset shortlist (genuine long-range dependency AND cheap verifiable reward)
Ranked by fit; paths + licenses research-gathered (verify before ingest). **Shipped-as-code status
(2026-07-08) annotated per item** — shipping the Source doesn't mean it's wired into `DEFAULT_TRAIN_MIX`
or that the GRPO/streaming-write task built on top of it exists yet (see the update note at the top):
1. **RULER generator** (`NVIDIA/RULER`, research-use) — unlimited synthetic long-context, exact-match, and
   the base for the fork-overwrite streaming-write data. **Start here.** — **shipped as `sources/ruler_niah.py`**
   (the overwrite-fork task itself is the separate `ruler_overwrite` source, already in the scrutiny mix).
2. **SWE-Gym** (`SWE-Gym/SWE-Gym`, MIT, prebuilt Docker envs) — cheapest real repo-scale execution reward.
   NOT shipped (distinct from the shipped `swe_trajectories` static reward-labeled source below).
3. **QASPER** (`allenai/qasper`, CC-BY-4.0) — real train split, provably non-gist-gameable, span/yes-no/
   unanswerable → EM + a built-in refusal axis. — **shipped as `sources/qasper.py`**.
4. **LongCite-45k** (`zai-org/LongCite-45k`, Apache-2.0) — 128k-word contexts, citation-span verification =
   purely programmatic long-range reward. — **shipped as `sources/longcite.py`**.
5. **BABILong-train** (`RMT-team/babilong-train-5k-samples`, Apache/BSD) — facts-in-haystack, exact-match,
   public generator; extends our bAbI line. — **shipped as `sources/babilong_train.py`**.
6. **WikiBigEdit** (`lukasthede/WikiBigEdit`, Apache-2.0, 502k, 8 sequential timesteps) — the only
   train-scale genuinely-streaming EM-rewardable set; direct fit for the forced-forgetting gate. —
   **shipped as `sources/wikibigedit.py`**.
7. **R2E-Gym / Nebius-SWE-agent-trajectories** (`R2E-Gym/R2E-Gym-V1` Apache; `nebius/SWE-agent-trajectories`
   CC-BY-4.0, 80k reward-labeled) — scale execution-verified repo RL + SFT warm-start. — the Nebius half
   **shipped as `sources/swe_trajectories.py`**; R2E-Gym itself NOT shipped.
8. **PerLTQA + LoCoMo-generation** (`Elvin-Yiming-Du/PerLTQA` train 5,155 CC-BY-NC; `snap-research/locomo`
   generator) — the only checkable conversational-memory train set + the scriptable recipe to scale it. —
   PerLTQA **shipped as `sources/perltqa.py`**; the LoCoMo-generation recipe NOT built.

*Honorable mentions:* RepoQA / LoCoDiff (clean long-code memory probes), LongCoder/LCC (~100k train),
Nemotron-RL-Agentic-Tool-Use-Pivot (RLVR tool-use), Big-Math-RL-Verified (math-needle reward sanity).

> **⚠ Personalization has almost no public TRAINING data** — nearly every conversational-memory resource
> is eval-only (LongMemEval, MSC, LoCoMo, Conversation-Chronicles). The only real checkable train splits
> are PerLTQA (5k, CC-BY-NC), WikiBigEdit (502k, factual not conversational), PrefEval (3k, 4-way-MC). So
> the practical path is **synthetic generation**: sample personas + atomic facts on a temporal/causal graph
> → generate N filler sessions injecting each fact at a scheduled session → probe whose gold = the injected
> fact's value span → reward = EM/containment (mirrors our existing condrecon value-span mask). Reuse
> LoCoMo's `generate_conversations.py` (event-graph) + PrefEval's generator + WikiBigEdit's Wikidata-diff pipeline.
>
> **⚠ Code repo-context caveat** (arXiv:2510.13697): much of the apparent repo-context gain ≈ RoPE
> long-context adaptation, not genuine cross-file reasoning → hold out **cross-file-dependent** targets
> (CrossCodeEval-style) to force real memory use. NarrativeQA is also gist-gameable (Q&A written from the
> plot summary alone) — prefer QASPER/QMSum/LongCite for genuine long-range long-doc.

### Harness integration (Source × Task × EpisodeSpec × Objective)
- **New Sources** (`sources/`): `msc` — **shipped**. Still to add: `conversation_chronicles`,
  `temporalwiki`/`newsedits`, `repobench_xfile`, `nemotron_agentic`, `swe_gym`, `gui_*`. Each gets a
  `pack_n_queries` + preprocessing to ordered segments.
- **New Tasks** (`tasks/`): **`streaming_update`** (ordered segments written sequentially, query the
  current value — the axis-2 task; needs the segment-stream batch contract + `chunk_offset`), **`trajectory`**
  (encode-history → next-action), **`tool_call`**.
- **New Objective**: **`grpo`** in `objectives.py` (reward via per-source verifiable checkers; decode from
  memory only) alongside the existing `behavioral_kl`. Keep **SHUF−REAL in the loss** here.
- **EpisodeSpec** additions: `n_sessions`/`n_segments`, `update_lag` (how many segments between write and
  query), reuse `total_len`/`window_size`/`query_lag`. Curriculum ratchets ctx 2k→8k→32k→BEAM-scale.
- The `streaming_update` task is the one real contract change (sequential segments) — everything else fits
  the existing packer/loader.

---

## PHASE 2 — Test-eval

### 2a. The headline table (fair comparison — the crucial design)
The compression literature (ICAE, AutoCompressors, Gist, xRAG, LLMLingua, **Cartridges**, **CCM**, 500x)
converges on a protocol — emulate the strongest (CCM/Cartridges):

- **Rows:** our LLM+memory · frozen-LLM + **full context** (the *ceiling* everyone reports) · frozen-LLM +
  **no context** (the *floor*) · **bigger-context** LLM · **bigger** LLM · **RAG** · **top competitor**
  (Mem0/MemGPT/Titans/Cartridges).
- **Efficiency axis (report BOTH):** the token ratio (raw tokens ÷ M) **and** the **KV-memory (MB/GB)
  ratio** — the memory-based denominator is the honest one for soft tokens (M tokens can carry a large KV
  footprint) and is what CCM/Cartridges headline. Plus **throughput** + peak GPU memory.
- **Fairness controls (CCM = gold standard):** identical frozen decoder, identical LoRA/training-compute,
  matched compression factor across every compressed arm.
- **Bigger-LLM comparison:** don't claim a single point — plot **quality vs KV-memory/throughput** and show
  the **Pareto frontier** (AutoCompressors Fig 4 / CCM Fig 6 / Cartridges). Our headline framing:
  *"match full-context quality at X× less KV-memory, Y× throughput"* (Cartridges: 38.6× / 26.4×).

### 2b. Memory / long-context benchmarks (the core panel)
Have (eval readers): `babilong`, `locomo`, `ruler`, `narrativeqa`, `hotpot`, `musique` — **plus, shipped
2026-07-08:** `longmemeval`, `longbench` (v1+v2), `infinitebench`, `niah` (all in `REGISTRY`, not yet run
as the headline table below). Verdicts from the
sweep (differentiator = score collapses without context AND can't be recovered from the frozen decoder's
parametric knowledge):

**Headline trio (decisive, low-confound, and where our family competes):**
| benchmark | verdict | why |
|---|---|---|
| **LongMemEval** (`xiaowu0162/longmemeval`, _S 115k / _M 1.5M / _Oracle) | **YES — strongest** | maps onto "always-on memory cache"; 30–60% oracle-vs-full gap = recall/update-bound; **no trainable compressor has published on it — genuine whitespace**. Baselines GPT-4o 60.6/Oracle 87.0. Competitors: Mem0, Zep |
| **RULER** (`NVIDIA/RULER` generator) | **YES — cleanest** | fully synthetic, zero knowledge-leakage, decisive length collapse; core soft-token compressors haven't used it (open slot); Titans reports S-NIAH |
| **BABILong** (have reader) | **YES — decisive** | scattered facts + multi-hop; **RMT and Titans compete here directly** (Titans ~94%@1M) → natural head-to-head; = our forced-forgetting/retention-vs-lag gate |

**Comparability panel (report because the compression baselines do):**
| benchmark | note |
|---|---|
| **LongBench v1** (+ v2) | de-facto standard the compression line reports (LongLLMLingua, Activation Beacon, InfiniRetri) → include for cross-method comparability |
| **multi-doc NQ + MuSiQue** (concatenated-distractor form) | the RAG-vs-compression battleground; xRAG/LongLLMLingua give reference numbers. **QA differentiator ranking: MuSiQue > HotpotQA > 2Wiki > NQ > TriviaQA** (TriviaQA gist-gameable — skip) |
| **∞Bench Retrieve.\*** or **NIAH multi-needle/multi-hop** | cheap retrieval sanity + full-context comparison (single-needle is saturated — skip) |

**Secondary / caveated:** WikiBigEdit (great for TRAINING the forced-forgetting gate, but as an *eval* it's
RAG-saturated → may not separate a learned compressor from plain RAG); StreamingQA/MSC (weak metrics; MSC
→ DMR is the standard bridge, MemGPT 93.4/Zep 94.8); NoLiMa (associative NIAH, lexical overlap removed);
LaMP/PerLTQA (personalization); BEAM/MemoryAgentBench (scale — audit integrity first); LoCoMo only as a
caveated cross-reference (audited dirty, ~6% wrong keys). **LOFT: off-path** (heavy/multimodal, no
compression paper used it).

### 2c. Capability benchmarks — retention controls vs differentiators
- **Capability-retention control (NOT memory evidence):** MT-Bench, AlpacaEval-2-LC, HumanEval+/MBPP, and
  the reasoning/math sets (MMLU/GPQA/GSM8K/MATH) all have **nothing to compress** — single-turn / closed-
  book. For a FROZEN decoder the "didn't break it" check is near-trivial. **Report at most one, explicitly
  labeled "capability retention," never as a memory claim** (precedent: Cartridges uses MMLU only as a
  generality control; LLMLingua uses GSM8K only as "CoT-survives-compression").
- **Differentiators (memory is *supposed* to help):** **WebArena** (accumulating observations),
  **AgentBench** (5–50 turns), **CrossCodeEval** / **RepoBench** (EM/Edit-Sim; hold out *cross-file* targets
  per the RoPE caveat), **SWE-bench-Verified** (%Resolved — and the soft-token *trajectory* angle here is
  our defensible novelty).
- **Skip:** Arena-Hard (redundant vs AlpacaEval), GAIA (gist-gameable), plain HumanEval/MBPP (EvalPlus
  dominates), TriviaQA (parametric confound), single-needle NIAH (saturated), 2Wiki (prefer MuSiQue).

### 2d. Competitor head-to-heads (shared benchmark to beat them on)
| competitor | approach | shared benchmark for head-to-head |
|---|---|---|
| Mem0, MemGPT/Letta | agent-memory (RAG-ish + summaries) | **LoCoMo, LongMemEval, DMR/MSC** |
| Titans | test-time memorization (fast weights) | **BABILong, NIAH** |
| Cartridges | trained KV-cache per corpus | **LongHealth, MTOB** + KV-memory ratio |
| CCM / AutoCompressors / ICAE | context compression | perplexity, ICL, the compression band |
| xRAG | 1-token RAG | single-passage QA (NQ / TriviaQA / HotpotQA) |

### 2e. Positioning (novelty — from the research course-correction)
- **NOT novel:** repo-level *code completion* with a soft-token compressor + frozen decoder — already
  scooped (CoRoVA arXiv:2510.19644; "empirical investigation" arXiv:2604.13725). Use it as a training
  signal / baseline, don't claim it.
- **LEAD WITH:** (a) **always-on implicit memory cache with streaming WRITE/UPDATE** (the axis no
  compression paper tests); (b) **soft-token compression of long AGENT TRAJECTORIES** (SWE-bench —
  defensibly open vs the NL-summary methods); (c) **first cross-method run of the modern long-context
  suites** (RULER/∞Bench/BABILong/LongBench-v2/LOFT); (d) the **stability-plasticity / forced-forgetting**
  instrument (WikiBigEdit + our SHUF−REAL-in-loss gate).

---

## Preprocessing / assembly / scheduling (cross-cutting)
- **Per-source packing config** (as in phase 0): item size, `pack_n_queries`, predict/target length per
  source — item size varies wildly, so never one constant.
- **Streaming assembly**: ordered segments (sessions / turns / edits / files), causality preserved (query
  the CURRENT value; never ask about a not-yet-written fact), `update_lag` controls write→query distance.
- **Firewalls**: train/eval disjoint (hotpot/musique already use TRAIN slices vs eval sets; MSC by session;
  TemporalWiki by date; SWE-Gym vs SWE-bench-Verified).
- **Scaling curriculum**: ctx 2k→4k→8k→32k→BEAM; M scaled with it; window 256; session/update count
  ratcheted (the `Curriculum` in `schedule.py`, not yet wired).
- **GRPO integration**: per-source verifiable checkers (math evaluator / code execution / tool-use match);
  reward computed on the memory-only decode.

## Verify-before-use (flagged by the research sweep)
- Licenses: THUDM/AgentInstruct (dataset README omits it; code repo Apache-2.0); Nemotron sets are
  CC-BY-4.0 (attribution).
- Runnability: whether NeMo-Gym envs for the Nemotron Pivot RLVR set run standalone (needed for live
  rollouts vs static replay).
- Data integrity: LoCoMo dirty (~6% wrong keys); BEAM / MemoryAgentBench are new 2025 benchmarks — audit
  first. Frozen 135M is near-chance on LongBench-v2/GAIA — those are aspirational at the winning-arch
  scale, not at the scrutiny scale.
- Competitor numbers: spot-check the papers' PDFs before quoting (several were HTML-summarizer-extracted).
- Positioning claim (repo-code scooped) is high-confidence; the "soft-token trajectory" white space is
  moderate-high — re-check for new arXiv before submission.

---

# Test-eval: comparison axes, run matrix & invariants

> **Consolidated from the former `docs/PHASE_PLAN.md` (2026-07-09 docs cleanup).** This is the
> single canonical phase plan; the full-corpus training data/objectives are in **PHASE 1** above,
> the headline table + benchmark panels in **PHASE 2** above, and the test-eval comparison
> framework + cross-phase invariants below. If this doc disagrees with the code, the code wins.

## What "phase" means

| phase | trains? | question | data character | scale (ctx / steps) |
|---|---|---|---|---|
| **architecture-scrutiny** (Phase 0, current) | yes | *which encoder architecture binds best?* | controlled / synthetic / short (bio, babi, fineweb) | 2k ctx, 4–8k steps, M=96 |
| **full-corpus** (Phase 1) | yes | *is the chosen memory genuinely useful on real tasks?* | real long docs / chat / agents / code | 8k–32k ctx, 20–50k steps, M=96–192 |
| **test-eval** (Phase 2) | **no** | *how does LM+memory compare to the alternatives?* | canonical benchmarks only | benchmark-defined |

**The four decisive comparison axes (test-eval must cover all four):** (1) vs same-backbone
**long-context** (does compression+memory beat raw extended context at equal-or-lower decode
FLOPs?); (2) vs same-backbone **RAG** (does implicit memory beat explicit retrieval at equal
decoder-read budget?); (3) vs **bigger frozen LM + same memory** (does binding transfer across
backbones?); (4) vs **other memory projects** on their own benchmarks.

## Axis 1 — long-context (vs extended-context LM). Run at 4k/8k/16k/32k → length-vs-accuracy curve.
| benchmark | what | reader | metric |
|---|---|---|---|
| ✅ `ruler` | NIAH / multi-key / var-tracking / KV-retrieval | `data/ruler.py` | accuracy |
| ✅ `locomo` | very-long-term dialogue memory | `data/locomo.py` | judge + EM |
| ✅ `babilong` | bAbI-in-haystack | `data/babilong.py` | EM |
| ✅ `longbench` (v1+v2) | 21 tasks / 503 hard MCQ (incl. code-repo, structured) | `data/longbench.py` | task-specific |
| ✅ `infinitebench` | 100k+ tok; `Retrieve.KV`, `Code.Debug`, `Math.Find` | `data/infinitebench.py` | accuracy/ROUGE |
| ✅ `niah` | needle-in-haystack (extend to multi-needle 64:1) | `data/niah.py` | accuracy |

A memory paper without LongBench + ∞Bench is not comparable to the literature.

## Axis 2 — RAG (vs same-backbone retriever). New eval-harness **mode** `--eval-arm rag`: swap the
encoder for BM25 / e5-base-v2 prepending top-k chunks at the SAME decoder-read budget (M tokens).
Headline: at equal decoder-read budget, does implicit memory beat explicit retrieval? (+ `locomo_rag`.)

## Axis 3 — backbone scaling. Re-run axis-1 with `--backbone Qwen/Qwen2.5-{1.5B,3B}` (frozen) + the
same trained encoder. A config matrix, no new benchmarks. Chart: memory benefit vs backbone size.

## Axis 4 — vs external memory projects (MemGPT/Landmark/A-MEM/Mem0) on their benchmarks (= axis-1
set + `multisession_chat_eval`, `ms_tod_eval`, optional `comedy`).

## Base-capability regression (does memory DEGRADE the LM?). Run lm-eval-harness core on the
memory-equipped model vs the bare frozen backbone via a thin `scripts/diagnostics/eval/lm_eval_runner.py`
adapter (don't reimplement): `mmlu`, `hellaswag`, `arc_challenge`, `winogrande`, `truthfulqa`,
`gsm8k`, `humaneval`/`mbpp`, `lambada`. **Pass = memory ≥ bare backbone within noise on every task**;
any degradation is a headline-negative result.

## Instruction/chat quality (fixed frozen judge): `mtbench`, `alpaca_eval`, `ifeval`, `longbench_chat`.
## Agentic / SWE (final checkpoint only — expensive): `swebench_verified` (the strongest "is this
useful" headline), `webarena`, `gaia`, `tau_bench`.
## Binding-gate diagnostics carry over from scrutiny (REAL/SHUF/OFF + `babi_em` + structure canaries)
as **diagnostics, not headline** — they explain *why* a model uses memory, on any bench with a SHUF control.

## Config matrix & run plan
One full-corpus checkpoint → scored across the 4-axis matrix by `scripts/diagnostics/eval/run_testeval_suite.py`:
```
for arm in [memory_equipped, long_context_32k, long_context_64k, rag, bigger_backbone+memory, bare_frozen]:
  for bench in [ruler, locomo, babilong, longbench, infinitebench, multisession_chat, mmlu, …, swebench_verified]:
    for length in [4k, 8k, 16k, 32k]:  run eval → outputs/testeval/<arm>_<bench>_<length>.json
```
Headline table (arm × benchmark-family) + length-vs-accuracy curves → `docs/testeval_results.md`.
Report memory arms at **M=96 (matched) AND M=192 (scaling)**; hold decoder-read tokens EQUAL across
memory vs RAG; fix + name the judge model in every table caption.

## Open decisions (call before implementation)
1. **Frozen 135M throughout full-corpus** (the comparability anchor); axis-3 is a test-eval-only
   scaling probe; restrict stage-3 GRPO to *binding*-verifiable rewards (EM-QA / slot / code-binding),
   not *reasoning*-verifiable (skip gsm8k/math until a bigger backbone is on the table).
2. **Context ceiling**: SmolLM2-135M native RoPE = 8192; extend to 32k via NTK/YaRN on frozen weights
   (needs the `transformers>=5.0` HF cache API). If 32k unstable, cap at 16k (still 170:1 at M=96).
3. **GUI data**: text a11y-tree only (covers webarena/tau_bench without a vision detour).
4. **Mix representation**: extend `mixes.py` in place behind a `--phase` flag (scrutiny preset stays
   byte-identical as the default), not a parallel `mixes_full.py`.
5. **External eval harness** (`lm_eval`/`swebench`/`webarena`): thin adapters in
   `scripts/diagnostics/eval/`, not `src/memory/data/` (in-tree `REGISTRY` readers stay for in-tree benches).

## Sequencing (dependency order)
1. Long-corpus sources + doc-disjoint firewall → unblocks stage-1 + the length curriculum (cheapest, highest leverage).
2. `multisession` task + chat/multisession sources → the core differentiator; unblocks axis-4.
3. Test-eval readers (longbench/infinitebench) BEFORE the full-corpus run (scoring suite ready + catch contamination early).
4. Weighted mixes + `Curriculum` wiring + `--phase` flag → the harness keystone.
5. `repo`/`repo_qa` (code binding) → pairs with SWE-bench.
6. Verifiable-reward GRPO → only after storage+addressing are solid (riskiest; lands last).
7. Agentic task + sources + agent evals → most novel/expensive; after the text+code foundation is proven.
(1–3 low-risk parallel; 4 is the keystone; 5–7 stack on top.)

## Invariants (across both new phases; inherited from `SCRUTINY_PHASE_DATA.md` §10)
1. **Causality** — a queried fact is always written before the query (session/turn N's query after its write).
2. **Un-guessability** — the answer requires the compressed context; distractors never contain it.
3. **Fixed compression denominator** — every context is exactly `total_len` (phase-scheduled); `M` uniform across arms.
4. **Train/eval firewall** — every full-corpus train source is disjoint from every test-eval benchmark.
5. **Frozen-backbone comparability** — backbone frozen in every arm except the decoder-LoRA (reported per-result).
6. **Decode-budget parity** — memory arm and RAG arm read the SAME #decoder tokens (M ≈ K chunks); long-context reads more.
7. **No silent Goodhart** — GRPO only with an external checker; binding-advantage reward stays a diagnostic, never a headline training signal.
