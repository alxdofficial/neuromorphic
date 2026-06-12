"""DATA INTEGRITY probe for the emat_bio gate. No GPU. Builds train+val emat_bio
at the LAUNCH config (context_len=1024, n_pairs=64, n_facts=3, world_seed=0) and
checks: (1) firewall name disjointness, (2) eval value verbatim NOT in train,
(3) closed-book (answer span not in question/context-as-decoder-input),
(4) SHUF construction breaks key->value binding."""
import sys, random
sys.path.insert(0, "/home/alex/code/neuromorphic")
from transformers import AutoTokenizer
from src.repr_learning.config import ReprConfig
from src.repr_learning.data_emat_bio import (
    EMATBioDataset, make_emat_bio_dataloader, _train_names, _canon, _WORLD,
)
from scripts.data.wave1.tasks.biographical.state import build_scenario

cfg = ReprConfig()
tok = AutoTokenizer.from_pretrained(cfg.llama_model)
PAD = cfg.pad_token_id
CTX = 1024
NP = 64
NF = 3
WS = 0

print("=== (2) FIREWALL: train vs val world entity-name disjointness ===")
train_names = _train_names(WS)              # canonical names from train world
val_ws = WS + 10_000
scen_val = build_scenario(random.Random(val_ws), 0, **_WORLD)
val_all_names = {_canon(e) for e in scen_val.world.entities.values()}
overlap = train_names & val_all_names
print(f"train world_seed={WS}: {len(train_names)} canonical names")
print(f"val   world_seed={val_ws}: {len(val_all_names)} canonical names")
print(f"raw name overlap BEFORE firewall drop: {len(overlap)}")
print(f"  sample overlap: {sorted(list(overlap))[:8]}")

# Build the ACTUAL val dataset the trainer builds (exclude_names=train_names)
val_ds = EMATBioDataset(tok, context_len=CTX, n_pairs=NP, n_query=1, n_facts=NF,
                        world_seed=val_ws, stream_seed=7, pad_token_id=PAD,
                        exclude_names=train_names)
val_kept = {_canon(e) for e in val_ds.entities}
leak = val_kept & train_names
print(f"val entities AFTER firewall drop: {len(val_ds.entities)} (>= n_pairs={NP}? {len(val_ds.entities) >= NP})")
print(f"val-kept names that still collide with train names (MUST be 0): {len(leak)}")

# train dataset (no exclude)
train_ds = EMATBioDataset(tok, context_len=CTX, n_pairs=NP, n_query=1, n_facts=NF,
                          world_seed=WS, stream_seed=42, pad_token_id=PAD,
                          exclude_names=None)
train_kept = {_canon(e) for e in train_ds.entities}
print(f"train entities: {len(train_ds.entities)}; train∩val-kept names: {len(train_kept & val_kept)}")

print("\n=== (2b) VALUE-LEVEL leak: are rendered eval VALUES ever verbatim in train? ===")
# Sample many val examples, collect value strings; sample many train examples, collect value strings.
def collect_values(ds, n):
    rng = random.Random(1234)
    vals = set()
    refs = []
    it = iter(ds)
    for _ in range(n):
        s = next(it)
        # answer_refs holds the gold value string
        for r in s["answer_refs"]:
            vals.add(r)
            refs.append(r)
    return vals, refs

# Use fresh datasets with deterministic stream so we can iterate
train_vals, _ = collect_values(EMATBioDataset(tok, context_len=CTX, n_pairs=NP, n_query=1,
    n_facts=NF, world_seed=WS, stream_seed=42, pad_token_id=PAD), 300)
val_vals, val_refs = collect_values(val_ds, 300)
vleak = train_vals & val_vals
print(f"distinct train values sampled: {len(train_vals)}; val values: {len(val_vals)}")
print(f"verbatim value overlap (train∩val) MUST be 0: {len(vleak)}")
if vleak:
    print(f"  LEAK EXAMPLES: {list(vleak)[:3]}")

print("\n=== (3) CLOSED-BOOK: answer-value tokens must NOT appear in question_ids ===")
# For a batch of val examples, check the gold value (answer) is not contained in the question.
it = iter(val_ds)
n_q_contains_ans = 0
n_checked = 0
samples = []
for _ in range(50):
    s = next(it)
    q_str = tok.decode(s["question_ids"].tolist())
    ans_str = s["answer_refs"][0]
    n_checked += 1
    # the value sentence should NOT be a substring of the key/question
    if ans_str.strip() and ans_str.strip() in q_str:
        n_q_contains_ans += 1
    samples.append((q_str, ans_str))
print(f"checked {n_checked} val examples; question contains full value: {n_q_contains_ans} (want 0)")
print("sample [question  ->  value]:")
for q, a in samples[:3]:
    print(f"  Q={q!r}\n   V={a!r}")

print("\n=== (3b) content-mask scoring: name tokens excluded, fact tokens scored ===")
s0 = samples_full = next(iter(val_ds))
cm = s0["answer_content_mask_list"]
ans_ids = s0["answer_ids"].tolist()
scored = tok.decode([a for a, m in zip(ans_ids, cm) if m])
unscored = tok.decode([a for a, m in zip(ans_ids, cm) if not m])
print(f"answer_tokens={len(ans_ids)} scored={sum(cm)} unscored={len(cm)-sum(cm)}")
print(f"  SCORED  : {scored!r}")
print(f"  UNSCORED: {unscored!r}")

print("\n=== (4) SHUF: roll-by-1 on memory breaks key->value binding ===")
# We cannot run the encoder here (no GPU build), but we verify the DATA property
# that makes SHUF meaningful: within a single batch the per-row context (and thus
# per-row memory) is DIFFERENT, so rolling memory by 1 gives a row a DIFFERENT
# passage's facts while keeping its own question. Build a real batch via the loader.
dl = make_emat_bio_dataloader(tok, context_len=CTX, batch_size=8, n_pairs=NP,
                              n_query=1, n_facts=NF, split="validation",
                              world_seed=WS, stream_seed=7, pad_token_id=PAD,
                              num_workers=0)
batch = next(iter(dl))
# QABatch: inspect context_ids per row distinctness + question/answer
import torch
ctx = batch.context_ids
B = ctx.shape[0]
# distinct rows?
distinct = len({tuple(ctx[i].tolist()) for i in range(B)})
print(f"batch B={B}; distinct context rows: {distinct} (want {B} so roll gives a different passage)")
# Does row i's question/value appear in row i-1's (rolled) context? It should NOT,
# i.e. the rolled memory genuinely lacks the answer.
qids = batch.question_ids
ans = batch.answer_ids
# Decode each row's gold value and check membership in the rolled (i-1) context.
hits_real = 0
hits_shuf = 0
for i in range(B):
    # reconstruct gold value from ALL valid (non-pad) answer tokens of this row
    a_ids = batch.answer_ids[i].tolist()
    a_mask = batch.answer_mask[i].tolist()
    val = tok.decode([t for t, m in zip(a_ids, a_mask) if m]).strip()
    own_ctx = tok.decode([t for t in ctx[i].tolist() if t != PAD])
    rolled_ctx = tok.decode([t for t in ctx[(i - 1) % B].tolist() if t != PAD])
    if val and val in own_ctx:
        hits_real += 1
    if val and val in rolled_ctx:
        hits_shuf += 1
print(f"gold value present in OWN context (REAL, want {B}): {hits_real}")
print(f"gold value present in ROLLED i-1 context (SHUF, want ~0): {hits_shuf}")
print("\nDONE.")
