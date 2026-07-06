"""LoCoMo — very-long-term conversational memory (EVAL-ONLY, OOD, long-context).

EVAL reader. Static JSON, downloaded once to `data/eval/locomo10.json` (from
snap-research/locomo) and cached locally. Marked "dirty" — cross-ref only.
See DATASETS.md.
"""
from __future__ import annotations

import random
from typing import Iterator

import torch
from torch.utils.data import DataLoader, IterableDataset

from .common import REPO_ROOT, collate_qa


class LoCoMoQADataset(IterableDataset):
    """LoCoMo (Maharana et al., 2024) — very-long-term conversational memory.

    EVAL-ONLY, OOD: never trained on. 10 multi-session human-machine dialogues
    (~300 turns, ~9k–26k tokens each) with ~200 QA pairs per conversation
    across 5 categories: 1=multi-hop, 2=temporal, 3=open-domain, 4=single-hop,
    5=adversarial (answer = "not mentioned" — tests knowing what you DON'T know).

    The FULL rendered conversation is the context. It exceeds 8k, so run this
    family at a larger chunk_size (e.g. 32768): the streaming encoder windows
    the whole dialogue into the fixed-footprint O(1) memory. This is a
    length-generalization probe — trained at 8k, evaluated to ~26k, exactly the
    regime where a fixed-footprint compressor should hold up while context
    grows. Answers are short (dates/names/phrases); headline metric is the
    LLM judge (abstractive + adversarial-negative answers).

    Source: snap-research/locomo `data/locomo10.json` (cached locally).
    """

    _URL = ("https://raw.githubusercontent.com/snap-research/locomo/"
            "main/data/locomo10.json")
    _CACHE = REPO_ROOT / "data/eval/locomo10.json"
    _CAT_NAME = {1: "multihop", 2: "temporal", 3: "open_domain",
                 4: "single_hop", 5: "adversarial"}

    def __init__(
        self,
        split: str,                          # ignored (eval-only, all 10 convs)
        tokenizer,
        chunk_size: int = 24576,             # max LoCoMo conv ≈ 24.4k tokens
        sep_token_id: int = 198,
        pad_token_id: int = 128_001,
        seed: int = 0,
    ):
        super().__init__()
        del split
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.sep_token_id = sep_token_id
        self.pad_token_id = pad_token_id
        self.seed = seed

        import json
        if not self._CACHE.exists():
            import urllib.request
            self._CACHE.parent.mkdir(parents=True, exist_ok=True)
            print(f"[data v1h] downloading LoCoMo → {self._CACHE} ...")
            urllib.request.urlretrieve(self._URL, self._CACHE)
        with open(self._CACHE) as f:
            raw = json.load(f)

        # Pre-render + pre-tokenize each conversation ONCE (10 of them). Build a
        # flat (conv_idx, qa) work-list so each QA pair shares its conversation's
        # cached token list.
        tok = tokenizer
        self._conv_ids: list[list[int]] = []
        self._qa: list[tuple[int, dict]] = []
        n_trunc = 0
        for ci, sample in enumerate(raw):
            text = self._render_conversation(sample["conversation"])
            ids = tok(text, add_special_tokens=False,
                      return_attention_mask=False)["input_ids"]
            if len(ids) > chunk_size:
                n_trunc += 1
            self._conv_ids.append(ids)
            for qa in sample.get("qa", []):
                # Category-5 (adversarial) rows carry their gold under
                # `adversarial_answer`, not `answer` — without this fallback the
                # entire "knowing what you DON'T know" subtask is dropped.
                ans = qa.get("answer")
                if ans is None:
                    ans = qa.get("adversarial_answer")
                if qa.get("question") is None or ans is None:
                    continue
                self._qa.append((ci, qa))
        lens = [len(x) for x in self._conv_ids]
        print(f"[data v1h]   LoCoMo: {len(raw)} conversations, {len(self._qa)} QA; "
              f"conv tokens min/mean/max = {min(lens)}/{sum(lens)//len(lens)}/{max(lens)}; "
              f"{n_trunc} conv(s) exceed chunk_size={chunk_size} (truncated)")
        if n_trunc:
            print(f"[data v1h]   WARN: {n_trunc} conversation(s) > {chunk_size} "
                  f"tokens — run LoCoMo with a larger --chunk-size to avoid "
                  f"dropping late-session evidence.")

    def _render_conversation(self, conv: dict) -> str:
        """Render all sessions in numeric order as dated speaker turns."""
        sess_ids = sorted(
            (int(k.split("_")[1]) for k in conv
             if k.startswith("session_") and not k.endswith("date_time")),
        )
        lines: list[str] = []
        for sid in sess_ids:
            turns = conv.get(f"session_{sid}")
            if not turns:
                continue
            date = conv.get(f"session_{sid}_date_time", "")
            lines.append(f"[Session {sid} — {date}]")
            for t in turns:
                spk = t.get("speaker", "")
                txt = (t.get("text") or "").strip()
                cap = t.get("blip_caption")
                if cap:
                    txt = f"{txt} [shares an image: {cap}]".strip()
                lines.append(f"{spk}: {txt}")
        return "\n".join(lines)

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info is not None else 0
        rng = random.Random(self.seed + 4127 + wid * 100_003)
        tok = self.tokenizer
        order = list(range(len(self._qa)))
        rng.shuffle(order)

        cs = self.chunk_size
        pos = 0
        while True:
            if pos >= len(order):
                rng.shuffle(order)
                pos = 0
            ci, qa = self._qa[order[pos]]
            pos += 1

            ctx_ids = list(self._conv_ids[ci])
            if len(ctx_ids) > cs:
                ctx_ids = ctx_ids[:cs]
            valid_len = len(ctx_ids)
            if valid_len < cs:
                ctx_ids = ctx_ids + [self.pad_token_id] * (cs - valid_len)

            q_text = str(qa["question"])
            # cat-5 adversarial gold lives in `adversarial_answer`.
            a_text = str(qa.get("answer") or qa.get("adversarial_answer") or "").strip()
            cat = qa.get("category")
            q_ids = tok(q_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]
            a_ids = tok(a_text, add_special_tokens=False,
                        return_attention_mask=False)["input_ids"]

            yield {
                "context_ids": torch.tensor(ctx_ids, dtype=torch.long),
                "context_mask": torch.tensor(
                    [True] * valid_len + [False] * (cs - valid_len),
                    dtype=torch.bool,
                ),
                "question_ids": torch.tensor(q_ids, dtype=torch.long),
                "answer_ids": torch.tensor(a_ids, dtype=torch.long),
                "answer_content_mask_list": [True] * len(a_ids),
                "answer_refs": [a_text],
                "task_family": "locomo",
                "question_type": self._CAT_NAME.get(cat, f"cat{cat}"),
            }


def make_locomo_dataloader(
    tokenizer,
    batch_size: int,
    *,
    split: str = "validation",
    chunk_size: int = 24576,
    sep_token_id: int = 198,
    pad_token_id: int = 128_001,
    seed: int = 0,
    num_workers: int = 0,
) -> DataLoader:
    """Standalone LoCoMo loader (EVAL, dirty — cross-ref only)."""
    ds = LoCoMoQADataset(split=split, tokenizer=tokenizer, chunk_size=chunk_size,
                         sep_token_id=sep_token_id, pad_token_id=pad_token_id, seed=seed)
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                      collate_fn=lambda s: collate_qa(s, pad_token_id=pad_token_id),
                      pin_memory=torch.cuda.is_available())
