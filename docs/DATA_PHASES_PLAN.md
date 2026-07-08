# Data Plan — Full-Corpus Training & Test-Eval Phases

Plan for the two phases *after* the architecture-scrutiny phase, synthesizing a research sweep (2025-era
memory/compression literature) with our prior data audit. Companion to `SCRUTINY_PHASE_DATA.md` (phase-0
data) and `DATA_TASK_GUIDE.md`. **Status: PLAN — not yet built.** Dataset numbers/licenses were research-
gathered; verify before ingesting (see "Verify-before-use" at the end).

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

### 1c. GRPO / RL polish
**Reward = task success given ONLY the compressed memory** — forces the memory to carry the load-bearing
bits (a memory that drops the auth token fails the checker). Verifiable-reward sources: math
(GSM8K/MATH/NuminaMath), code-with-tests (execution), tool-use success (Nemotron Pivot `pass_rate`),
long-context checkable answers. Recipe: SFT warm-start (Nemotron-Agentic-v1) → GRPO on RLVR. Cleanest
plug: **Nemotron Pivot** (`expected_action`/`pass_rate`, NeMo-Gym-compatible) — reuse its reward instead
of building one.

### Harness integration (Source × Task × EpisodeSpec × Objective)
- **New Sources** (`sources/`): `msc`, `conversation_chronicles`, `temporalwiki`/`newsedits`,
  `repobench_xfile`, `nemotron_agentic`, `swe_gym`, `gui_*`. Each gets a `pack_n_queries` + preprocessing
  to ordered segments.
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
Have (eval readers): `babilong`, `locomo`, `ruler`, `narrativeqa`, `hotpot`, `musique`. **Add:**

| benchmark | measures | note |
|---|---|---|
| **LongMemEval** | multi-session chat memory (updates + multi-session splits) | **PRIMARY** — updates make a generic prior provably wrong → REAL≫SHUF only if memory binds current value |
| **∞Bench / InfiniteBench**, **HELMET**, **LongBench-v2**, **LOFT** | modern long-context suites | **WHITE SPACE**: no prior compression paper ran these cross-method — running us + ceiling/floor/competitors is a genuine contribution |
| **NoLiMa** | associative NIAH (lexical overlap removed) | the un-gameable needle |
| **WikiBigEdit** | Locality (stability) / Update (plasticity) / Multi-hop (binding) | the **stability-plasticity instrument** — our axis-2 headline |
| **LaMP**, PerLTQA | personalization / lifelong | the "implicit preference applied later" claim |
| **BEAM-1M/10M**, **MemoryAgentBench** | scale-up headline | verify data integrity first (2025 benchmarks) |

LoCoMo only as a caveated cross-reference (audited dirty, ~6% wrong keys).

### 2c. Capability benchmarks — two tiers
- **Sanity ("didn't break the frozen LLM"):** MT-Bench, AlpacaEval-2-LC, HumanEval+/MBPP, **GSM8K**
  (reasoning-preservation — LLMLingua-2 precedent). Single-turn/standalone → nothing to compress; just
  certify base skill survived.
- **Differentiators (memory is *supposed* to help):** **WebArena** (accumulating observations),
  **AgentBench** (5–50 turns), **CrossCodeEval** / **RepoBench** (EM/Edit-Sim), **SWE-bench-Verified**
  (%Resolved).
- **Skip:** Arena-Hard (redundant vs AlpacaEval), GAIA (gist-gameable), plain HumanEval/MBPP (EvalPlus
  dominates).

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
