"""Shared corpus-source loader for the HF-streamed bucket-1 text sources (pile, redpajama).

Both ``PileSource`` and ``RedpajamaSource`` do the same thing — load a BOUNDED sample of natural-text
documents, re-tokenize with the BACKBONE tokenizer, and keep docs ≥ ``min_len`` — differing only in
dataset name / HF id. This factors that loader (the FinewebSource interface, minus the Llama→text
decode firewall that only the local FineWeb parquet needs).

Resolution order (BEST-EFFORT, never hangs on the procedural sources):
  1. Local ``data/<name>/<split>.jsonl`` (``{"text": ...}`` per line) written by the ingest script.
  2. Else HF-stream a small sample (``n_docs`` docs, ``skip`` for a disjoint val slice).
  3. Else (HF unreachable / offline) raise a clear "run ingest first" error — no silent hang.

Ingest scripts: ``scripts/data_build/ingest/{pile,redpajama}/download.py``.
See DATASETS.md / docs/history/docs/history/data_arch_plan.md (Layer L1).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable, Iterator, List, Optional

import numpy as np

from .base import CorpusItem, REPO


def local_jsonl(name: str, split: str) -> Optional[Path]:
    """The local staged sample for ``name``/``split``, if the ingest script has run."""
    p = REPO / "data" / name / f"{split}.jsonl"
    return p if p.exists() else None


def _tokenize_cached(source_path: Path, tokenizer, texts_fn: Callable[[], Iterator[str]]) -> List[np.ndarray]:
    """Backbone-tokenize every text once and DISK-CACHE the token ids next to the source, so the many
    diagnostics / per-variant startups don't retokenize the full corpus each construction (the slow
    path). Stored PICKLE-FREE as a flat int64 id-stream + per-doc lengths in an ``.npz`` (loaded with
    the numpy default ``allow_pickle=False`` — no code-exec risk even though the cache is self-written).
    Keyed by the backbone tokenizer identity+vocab; invalidated when the source file is newer. Returns
    the PRE-min_len-filter doc list (filtering is cheap and varies per task)."""
    fp = f"{getattr(tokenizer, 'name_or_path', 'tok')}|vocab={getattr(tokenizer, 'vocab_size', 0)}"
    tag = hashlib.md5(fp.encode()).hexdigest()[:10]
    cache = source_path.parent / "cache" / f"{source_path.stem}.{tag}.tokids.npz"
    if cache.exists() and cache.stat().st_mtime >= source_path.stat().st_mtime:
        try:
            z = np.load(cache)                                       # allow_pickle=False by default → safe
            flat, lengths = z["flat"], z["lengths"]
            offs = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
            return [flat[offs[i]:offs[i + 1]] for i in range(len(lengths))]
        except Exception:
            pass                                                     # corrupt/incompatible → re-tokenize
    docs = [np.asarray(tokenizer(t, add_special_tokens=False).input_ids, dtype=np.int64) for t in texts_fn()]
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        flat = np.concatenate(docs) if docs else np.zeros(0, dtype=np.int64)
        np.savez(cache, flat=flat, lengths=np.array([len(d) for d in docs], dtype=np.int64))
    except Exception:
        pass                                                         # cache is best-effort; never fatal
    return docs


def _text_of(obj: dict) -> str:
    return obj.get("text") or obj.get("content") or ""


def _iter_local_texts(path: Path) -> Iterator[str]:
    with open(path) as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            t = _text_of(json.loads(line))
            if t:
                yield t


def _iter_hf_texts(name: str, hf_name: str, hf_config: Optional[str], hf_split: str,
                   n_docs: int, skip: int, hf_data_dir: Optional[str] = None) -> Iterator[str]:
    """Stream up to ``n_docs`` texts from HF (after skipping ``skip``). ``hf_data_dir`` selects a
    subdir (e.g. the-stack-smol ``data/python``). Raises a clear ingest-first error if the hub is
    unreachable — so an offline env fails fast, never hangs."""
    try:
        from datasets import load_dataset
        _kw = {"split": hf_split, "streaming": True}
        if hf_config:
            _kw["name"] = hf_config
        if hf_data_dir:
            _kw["data_dir"] = hf_data_dir
        ds = load_dataset(hf_name, **_kw)
    except Exception as e:  # network / offline / missing dataset
        raise RuntimeError(
            f"[{name}] HF dataset {hf_name!r} unreachable ({type(e).__name__}: {str(e)[:120]}). "
            f"Run  python scripts/data_build/ingest/{name}/download.py  to stage a local "
            f"data/{name}/{{train,val}}.jsonl sample, then retry (works fully offline once staged)."
        ) from e
    count = 0
    seen = 0
    for ex in ds:
        t = _text_of(ex)
        if not t:
            continue
        if seen < skip:
            seen += 1
            continue
        yield t
        count += 1
        if count >= n_docs:
            break


def load_corpus_docs(tokenizer, *, name: str, hf_name: str, hf_config: Optional[str],
                     split: str, min_len: int, n_docs: int,
                     hf_data_dir: Optional[str] = None) -> List[np.ndarray]:
    """Load → re-tokenize (backbone) → keep docs ≥ min_len. Local jsonl first, else HF stream."""
    # Whole docs are tokenized then sliced to a window at emit time (Task), so a doc longer than the
    # tokenizer's model_max_length is expected/harmless — silence that warning (restored after).
    from transformers.utils import logging as _hf_logging
    _prev = _hf_logging.get_verbosity()
    _hf_logging.set_verbosity_error()
    try:
        local = local_jsonl(name, split)
        if local is not None:
            docs_all = _tokenize_cached(local, tokenizer, lambda: _iter_local_texts(local))  # disk-cached
            origin = f"data/{name}/{split}.jsonl"
        else:
            # These sample sets typically expose only a 'train' split; carve a disjoint val slice by
            # skipping the first n_docs docs (train takes [0:n_docs], val takes [n_docs:2·n_docs]).
            skip = 0 if split == "train" else n_docs
            texts = _iter_hf_texts(name, hf_name, hf_config, "train", n_docs, skip, hf_data_dir)
            origin = f"HF:{hf_name}" + (f"/{hf_data_dir}" if hf_data_dir else "")
            docs_all = [np.asarray(tokenizer(t, add_special_tokens=False).input_ids, dtype=np.int64)
                        for t in texts]
    finally:
        _hf_logging.set_verbosity(_prev)

    n_total = len(docs_all)
    docs: List[np.ndarray] = [a for a in docs_all if a.shape[0] >= min_len]

    if not docs:
        raise ValueError(
            f"[{name}] no doc ≥ {min_len} tok from {origin} ({n_total} scanned). "
            f"Lower the task's total_len, or stage more docs via the ingest script.")
    print(f"[data.{name}] {split}: {len(docs)}/{n_total} docs ≥ {min_len} tok from {origin}",
          flush=True)
    return docs


class _CorpusSampleMixin:
    """``sample`` shared by the HF-streamed corpus sources (uniform doc draw, Task's stdlib RNG)."""

    docs: List[np.ndarray]

    def sample(self, rng, n: int) -> list:
        return [CorpusItem(tokens=self.docs[rng.randrange(len(self.docs))]) for _ in range(n)]
