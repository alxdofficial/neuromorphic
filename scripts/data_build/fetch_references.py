#!/usr/bin/env python3
"""Fetch every paper catalogued in docs/REFERENCES.md as a local PDF + a grep-able text extract.

WHY: we cite ~55 baselines/datasets by arXiv URL only. An offline agent (or a pod with no web) can't read a
URL. This vendors the actual publications into docs/references/{pdf,txt}/ so any agent can open or grep them.
The .txt extracts (PyMuPDF) are the agent-facing surface; the PDFs preserve figures/layout for humans.

Sources: arXiv PDFs from https://arxiv.org/pdf/<id> (polite delay between requests). Works with no arXiv
(RedPajama, NIAH, WikiBigEdit, MultiWOZ) are recorded in the manifest with source=None and skipped for PDF —
handled separately (landing-page snapshot / note in the index).

Idempotent: skips a paper whose PDF already exists. Re-run to fill gaps. `--only <id>` fetches one.
"""
from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Papers live at the MASTER level (sibling of the code repo): ~/code/neuromorphic/papers (HALO-style layout).
REF_DIR = REPO.parent / "papers"
PDF_DIR = REF_DIR / "pdf"
TXT_DIR = REF_DIR / "txt"

# (arxiv_id | None, slug, category, role) — category groups the index; role = what it is to us.
# arxiv_id None → no arXiv version (handled out-of-band). Deduped by arxiv_id at fetch time.
MANIFEST = [
    # --- backbone ---
    ("2502.02737", "smollm2", "backbone", "Frozen decoder (SmolLM2-135M)"),
    ("2407.21783", "llama3", "backbone", "FineWeb source-tokenizer only (Llama-3.2-1B)"),
    # --- baseline compressors / memory techniques (active) ---
    ("2307.06945", "icae", "baseline", "ICAE — in-context autoencoder compressor"),
    ("2305.14788", "autocompressor", "baseline", "AutoCompressor — summary-accumulation compressor"),
    ("2304.08467", "gisting", "baseline", "Gisting — gist-token prompt compression"),
    ("2402.04624", "memoryllm", "baseline", "MemoryLLM — self-updatable KV pool (our M+ ancestor)"),
    ("2501.00663", "titans", "baseline", "Titans — test-time memory (reimpl reference)"),
    ("2306.14048", "h2o", "baseline", "H2O — heavy-hitter KV eviction (Tier-2 GPU baseline)"),
    ("2505.23416", "kvzip", "baseline", "KVzip — query-agnostic KV compression (Tier-2 GPU baseline)"),
    ("2606.09659", "lclm", "baseline", "LCLM — end-to-end soft-token compressor (Tier-2 GPU baseline, closest competitor)"),
    ("2210.05062", "relational_attention", "baseline", "Relational Attention — slotgraph edge-on-value lineage"),
    # --- retired baselines (citations kept for provenance) ---
    ("2312.03414", "ccm", "retired_baseline", "CCM — compressed context memory (posterior-collapse ref)"),
    ("2401.03462", "activation_beacon", "retired_baseline", "Activation Beacon — 4K→400K compression"),
    ("1711.00937", "vqvae", "retired_baseline", "VQ-VAE — the VQ in VQ-ICAE"),
    ("2406.06484", "deltanet", "retired_baseline", "DeltaNet — delta-rule write lineage"),
    ("2412.06464", "gated_deltanet", "retired_baseline", "Gated DeltaNet — gated delta-rule write"),
    ("2006.15055", "slot_attention", "retired_baseline", "Slot Attention — anti-collapse read"),
    ("1612.03969", "entnet", "retired_baseline", "Recurrent Entity Networks — keyed slot memory"),
    ("2207.06881", "rmt", "retired_baseline", "Recurrent Memory Transformer — streaming recurrence"),
    # --- training objective (behavioral-KL context distillation) lineage ---
    ("2503.08727", "dcd", "objective", "Deep Context Distillation — output-KL + hidden-L1"),
    ("2210.03162", "wingate", "objective", "Wingate — KL(hard-prompt ‖ soft-prompt)"),
    ("2405.13792", "xrag", "objective", "xRAG — CE + α·KL over a frozen LLM"),
    ("2506.06266", "cartridges", "objective", "Cartridges / Self-Study — per-layer KV + context distillation"),
    ("2412.14964", "kujanpaa", "objective", "Kujanpää — E[KL]=I(context;answer) theorem"),
    ("2306.09306", "padmanabhan", "objective", "Padmanabhan — value-span-masked KL"),
    # --- Phase-0 data sources ---
    ("1502.05698", "babi", "data_phase0", "bAbI tasks"),
    ("1806.03822", "squad2", "data_phase0", "SQuAD 2.0"),
    ("1705.03551", "triviaqa", "data_phase0", "TriviaQA"),
    ("1809.09600", "hotpotqa", "data_phase0", "HotpotQA"),
    ("2108.00573", "musique", "data_phase0", "MuSiQue"),
    ("1810.00278", "multiwoz", "data_phase0", "MultiWOZ 2.2"),
    ("2112.08608", "quality", "data_phase0", "QuALITY"),
    ("2406.17557", "fineweb", "data_phase0", "FineWeb / FineWeb-Edu"),
    ("2101.00027", "pile", "data_phase0", "The Pile"),
    ("2312.04927", "zoology_mqar", "data_phase0", "Zoology — associative recall / MQAR"),
    ("2404.06654", "ruler", "data_phase0", "RULER (also ruler_overwrite fork + Phase-2 reader)"),
    # --- Phase-1 data sources ---
    ("2405.01470", "wildchat", "data_phase1", "WildChat-1M"),
    ("2309.11998", "lmsys_chat", "data_phase1", "LMSYS-Chat-1M"),
    ("2107.07567", "msc", "data_phase1", "Multi-Session Chat"),
    ("2105.03011", "qasper", "data_phase1", "Qasper"),
    ("2409.02897", "longcite", "data_phase1", "LongCite-45k"),
    ("2104.02112", "govreport", "data_phase1", "GovReport"),
    ("1911.05507", "pg19", "data_phase1", "PG-19 / Compressive Transformer"),
    ("2406.10149", "babilong", "data_phase1", "BABILong (train + Phase-2 reader)"),
    ("2405.15793", "swe_agent", "data_phase1", "SWE-agent (swe_trajectories)"),
    ("2402.16288", "perltqa", "data_phase1", "PerLTQA"),
    ("2503.05683", "wikibigedit", "data_phase1", "WikiBigEdit — lifelong knowledge editing (500K+ edits)"),
    # --- Phase-2 test-eval benchmarks ---
    ("2410.10813", "longmemeval", "data_phase2", "LongMemEval — Phase-2 HEADLINE"),
    ("2507.05257", "memoryagentbench", "data_phase2", "MemoryAgentBench — Phase-2 second dataset (was missing from REFERENCES.md)"),
    ("2308.14508", "longbench", "data_phase2", "LongBench v1+v2"),
    ("2402.13718", "infinitebench", "data_phase2", "∞Bench / InfiniteBench"),
    ("2402.17753", "locomo", "data_phase2", "LoCoMo"),
    ("1712.07040", "narrativeqa", "data_phase2", "NarrativeQA"),
    # --- competitor memory systems (Phase-2 head-to-heads) ---
    ("2403.11901", "larimar", "competitor", "Larimar — Kanerva episodic memory"),
    ("2502.00592", "mplus", "competitor", "M+ — MemoryLLM + retriever (Tier-2 GPU baseline)"),
    ("2405.14768", "wise", "competitor", "WISE — lifelong editing"),
    ("2505.23735", "atlas", "competitor", "ATLAS — test-time memory"),
    # LCLM + KVzip: adopted Tier-2 methods not yet in REFERENCES.md — arXiv verified separately, add when confirmed.
    # --- no-arXiv works (recorded for the index; no PDF) ---
    (None, "redpajama", "data_phase0", "RedPajama-Data — github.com/togethercomputer/RedPajama-Data"),
    (None, "niah", "data_phase2", "Needle-in-a-Haystack — github.com/gkamradt/LLMTest_NeedleInAHaystack"),
]

UA = "neuromorphic-refs/1.0 (research; local vendoring of cited papers)"


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        dest.write_bytes(r.read())


def _extract_text(pdf: Path, txt: Path) -> int:
    import fitz  # PyMuPDF
    doc = fitz.open(pdf)
    parts = [page.get_text() for page in doc]
    doc.close()
    body = "\n".join(parts)
    txt.write_text(body)
    return len(body)


def fetch_one(arxiv_id: str, slug: str, delay: float) -> tuple[str, str]:
    stem = f"{arxiv_id}_{slug}"
    pdf = PDF_DIR / f"{stem}.pdf"
    txt = TXT_DIR / f"{stem}.txt"
    if pdf.exists() and txt.exists() and pdf.stat().st_size > 1000:
        return "skip", stem
    try:
        _download(f"https://arxiv.org/pdf/{arxiv_id}", pdf)
        if pdf.stat().st_size < 1000:
            return "fail(too-small)", stem
        n = _extract_text(pdf, txt)
        time.sleep(delay)   # be polite to arXiv
        return f"ok({pdf.stat().st_size // 1024}KB/{n // 1000}k chars)", stem
    except Exception as e:  # noqa: BLE001
        return f"FAIL: {type(e).__name__}: {e}", stem


_CATEGORY_TITLES = {
    "backbone": "Backbone (frozen decoder)",
    "baseline": "Baseline compressors / memory techniques (active, incl. Tier-2 GPU methods)",
    "retired_baseline": "Retired baselines (citations kept for provenance)",
    "objective": "Training objective — behavioral-KL context distillation lineage",
    "data_phase0": "Data — Phase-0 (architecture scrutiny)",
    "data_phase1": "Data — Phase-1 (full-corpus training)",
    "data_phase2": "Data — Phase-2 (test-eval benchmarks)",
    "competitor": "Competitor memory systems (Phase-2 head-to-heads)",
}


def write_index() -> None:
    """(Re)generate docs/references/README.md from the manifest + what's actually on disk."""
    lines = [
        "# Local reference library — vendored publications for every cited baseline & dataset",
        "",
        "Every paper we cite (baselines, datasets, objective, competitors — all tiers) stored **locally** as a",
        "`pdf/` (fidelity) + a `txt/` extract (PyMuPDF; grep-able, for agents). So an offline agent or a pod with",
        "no web can read the actual source, not just a URL. This `papers/` dir is a sibling of the `code/` repo",
        "under the neuromorphic master. Catalog + roles: [`../code/docs/REFERENCES.md`](../code/docs/REFERENCES.md).",
        "",
        "Regenerate / fill gaps: `python scripts/data_build/fetch_references.py` (idempotent — skips what's present).",
        "",
    ]
    for cat, title in _CATEGORY_TITLES.items():
        rows = [(a, s, r) for (a, s, c, r) in MANIFEST if c == cat]
        if not rows:
            continue
        lines += [f"## {title}", "", "| arXiv | file | role |", "|---|---|---|"]
        for a, s, r in rows:
            if a:
                stem = f"{a}_{s}"
                present = (PDF_DIR / f"{stem}.pdf").exists()
                fcell = f"`{stem}.pdf` {'' if present else '⚠ missing'}"
                acell = f"[{a}](https://arxiv.org/abs/{a})"
            else:
                fcell, acell = "— (no arXiv; see role)", "—"
            lines.append(f"| {acell} | {fcell} | {r} |")
        lines.append("")
    n_pdf = len(list(PDF_DIR.glob("*.pdf"))) if PDF_DIR.exists() else 0
    lines += [f"_Auto-generated by `scripts/data_build/fetch_references.py`. {n_pdf} PDFs on disk._"]
    (REF_DIR / "README.md").write_text("\n".join(lines))
    print(f"[refs] wrote index → {REF_DIR / 'README.md'} ({n_pdf} PDFs)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", help="fetch just this arXiv id")
    ap.add_argument("--delay", type=float, default=3.0, help="seconds between arXiv requests")
    ap.add_argument("--index-only", action="store_true", help="just (re)write README.md, no downloads")
    args = ap.parse_args()

    if args.index_only:
        write_index()
        return

    PDF_DIR.mkdir(parents=True, exist_ok=True)
    TXT_DIR.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    todo = [(a, s) for (a, s, _c, _r) in MANIFEST if a and a not in seen and not seen.add(a)]
    if args.only:
        todo = [(a, s) for (a, s) in todo if a == args.only]
    print(f"[refs] {len(todo)} arXiv papers to fetch → {REF_DIR}")
    ok = skip = fail = 0
    for i, (a, s) in enumerate(todo, 1):
        status, stem = fetch_one(a, s, args.delay)
        print(f"[refs] {i:>2}/{len(todo)}  {a:<12} {status}", flush=True)
        if status.startswith("ok"):
            ok += 1
        elif status == "skip":
            skip += 1
        else:
            fail += 1
    print(f"\n[refs] done: {ok} downloaded, {skip} already present, {fail} failed")
    write_index()
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
