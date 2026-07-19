"""Published baseline numbers for LongMemEval + MemoryAgentBench — CITED REFERENCE ONLY.

These come from the PRIMARY papers and are scored by the papers' **LLM-as-judge (GPT-4o)**, NOT by our
deterministic scorer. So they are NOT directly comparable to our columns: the judge credits paraphrase /
superset answers, so the published numbers sit ABOVE what a deterministic (EM+containment+BEM) scorer gives
on the same model outputs. `report.py` renders them as a SEPARATE, clearly-labeled block — never merged into
our deterministic table. Vendor / memory-startup self-reported numbers are deliberately excluded (contradictory
and inflated). Calibration point: our own Llama-3.1-8B full-context row (deterministic) vs the published 45.4%
(judge) quantifies the deterministic↔judge offset once our run lands.
"""
from __future__ import annotations

# --- LongMemEval-S (Wu et al., arXiv:2410.10813, ICLR'25). Judge = gpt-4o-2024-08-06 (>97% human agreement).
LONGMEMEVAL_PUBLISHED = {
    "source": "Wu et al., LongMemEval-S, arXiv:2410.10813 (ICLR'25) — scoring: gpt-4o-2024-08-06 judge",
    "rows": [                          # (model, condition, accuracy)
        ("GPT-4o", "oracle (evidence session only)", 0.870),
        ("GPT-4o", "full-context (~115k)", 0.606),
        ("GPT-4o", "full-context +Chain-of-Note", 0.640),
        ("Llama-3.1-8B-Instruct", "oracle", 0.710),
        ("Llama-3.1-8B-Instruct", "full-context", 0.454),   # ← our deterministic Llama-8B full_context calibrates here
        ("Llama-3.1-70B-Instruct", "oracle", 0.744),
        ("Llama-3.1-70B-Instruct", "full-context", 0.334),  # worse than the 8B — the paper's long-context-degradation finding
        ("Phi-3-Medium-128k", "full-context", 0.380),
    ],
    "note": "Llama-3.1-8B full-context (0.454) is the base number; verify base-vs-CoN split against Fig 3(b) "
            "before publishing. GPT-4o-mini / Mistral / Qwen / Claude are NOT in the primary -S table.",
}

# --- MemoryAgentBench (Hu et al., arXiv:2507.05257). AR/TTL/CR deterministic; LRU = GPT-4o-judge F1.
MEMORYAGENTBENCH_PUBLISHED = {
    "source": "Hu et al., MemoryAgentBench, arXiv:2507.05257 — AR/TTL/CR deterministic; LRU = GPT-4o-judge F1",
    "competencies": ["Accurate-Retrieval", "Test-Time-Learning", "Long-Range*(judge)", "Conflict-Res(single-hop)"],
    "rows": [                          # (method, AR, TTL, LRU, CR) — AR shown as a per-subtask range
        ("Long-context GPT-4o", "53.5–61.5", "87.6", "32.2", "60.0"),
        ("Long-context GPT-4o-mini", "44.9–53.5", "82.4", "28.9", "45.0"),
        ("Long-context Claude-3.7-Sonnet", "50.6–74.6", "89.4", "52.5", "43.0"),
        ("RAG BM25", "45.6–74.6", "75.4", "20.9", "56.0"),
        ("RAG Embedding (NV-Embed-v2)", "51.4–83.0", "69.4", "20.7", "55.0"),
        ("Mem0", "22.4–37.5", "3.4", "0.8", "18.0"),
    ],
    "note": "* LRU is LLM-judged (GPT-4o F1) upstream — our detective_qa (LRU) uses deterministic exact_match, "
            "so that column is the LEAST comparable. Conflict-Resolution MULTI-hop collapses to ≤6% for all "
            "methods upstream (only single-hop shown). AR is a per-subtask range.",
}


def _esc(s) -> str:
    return str(s).replace("|", "\\|")


def render_published_markdown() -> str:
    """Render both published tables as a clearly-labeled CITED block (judge-scored; NOT our deterministic scale)."""
    out = ["", "---", "",
           "## Published reference (CITED — paper LLM-judge, NOT comparable to the deterministic columns above)",
           "",
           "> These are the papers' own numbers, graded by a **GPT-4o judge** (which credits paraphrase/superset "
           "answers). Our deterministic scores sit **below** these on the same outputs — read them as recognized "
           "anchors, not head-to-head. Vendor self-reported numbers are excluded.", ""]

    lme = LONGMEMEVAL_PUBLISHED
    out += [f"### LongMemEval-S — published", f"_{_esc(lme['source'])}_", "",
            "| model | condition | accuracy |", "|---|---|---|"]
    for model, cond, acc in lme["rows"]:
        out.append(f"| {_esc(model)} | {_esc(cond)} | {acc:.3f} |")
    out += ["", f"_{_esc(lme['note'])}_", ""]

    mab = MEMORYAGENTBENCH_PUBLISHED
    out += [f"### MemoryAgentBench — published (long-context / RAG backbones)", f"_{_esc(mab['source'])}_", "",
            "| method | " + " | ".join(_esc(c) for c in mab["competencies"]) + " |",
            "|" + "---|" * (len(mab["competencies"]) + 1)]
    for row in mab["rows"]:
        out.append("| " + " | ".join(_esc(x) for x in row) + " |")
    out += ["", f"_{_esc(mab['note'])}_", ""]
    return "\n".join(out)
