"""Why do hlvocab + soft_pointer_graph plateau ~6.6 (barely below the no-memory
floor)? Load the TRAINED checkpoints and compare against FRESH init to diagnose
the bottleneck. Reports, per model:

  READ UTILITY (the money question — is memory used, ignored, or pooled?):
    REAL / OFF / SHUF reconstruction loss (same seeded mask) →
      OFF-REAL  = how much memory HELPS (≈0 → ignored)
      SHUF-REAL = binding (wrong memory HURTS → the channel is addressed)
    memory effective-rank of the emitted tokens (low → redundant/blurry = pooled)

  hlvocab vocabulary health (fresh vs trained):
    node_keys/values collapse (mean|cos|, eff_rank), node USAGE/coverage,
    routing entropy/hub, edge-selection diversity (slot_uniq, inter-layer frac).

  soft_pointer_graph health (fresh vs trained):
    node-bank collapse, soft-pointer read entropy + active-node fraction,
    edge-state effect (is the relation state used?).

Run AFTER the GPU is free. Usage: python scripts/diagnostics/mae_trained_diag.py
"""
import sys, os, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch, torch.nn.functional as F
from transformers import AutoTokenizer
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.data_masked_reconstruction import make_sentence_dataloader
from src.memory.models.hierarchical_learned_vocab.substrate import _unit, _unit_rms

dev = "cuda"
BACKBONE = "HuggingFaceTB/SmolLM2-135M"; SRC = "meta-llama/Llama-3.2-1B"
TAG = "mae_4k_matched_bs64"
CK = lambda v: f"outputs/memory/{TAG}_{v}/ckpts/{v}.best.pt"


def matched(cfg):
    cfg.llama_model = BACKBONE; cfg.d_llama = 576; cfg.llama_vocab_size = 49152
    cfg.pad_token_id = 0; cfg.task_mode = "masked_reconstruction"
    cfg.use_llama_lora = True; cfg.llama_lora_rank = 16; cfg.llama_lora_alpha = 32
    cfg.n_flat_codes = 16
    cfg.hlvocab_d_code = 256; cfg.hlvocab_nodes = (512, 256, 128)
    cfg.hlvocab_top_k = 4; cfg.hlvocab_m_max = 16; cfg.hlvocab_tap_layer = 6
    cfg.spg_K_edge = 16; cfg.spg_K_node = 64
    cfg.spg_d_node = 176; cfg.spg_d_state = 176; cfg.spg_d_read = 176
    cfg.spg_d_updater = 240; cfg.spg_updater_layers = 2; cfg.spg_updater_heads = 8
    cfg.spg_read_ffn_mult = 2; cfg.spg_builder_mlp_hidden = 224; cfg.spg_film_hidden = 176
    return cfg


def eff_rank(X):  # participation-ratio effective rank of rows (centered, unit-norm)
    X = F.normalize(X.float(), dim=-1)
    s = torch.linalg.svdvals(X - X.mean(0, keepdim=True))
    s2 = s * s
    return (s2.sum() ** 2 / (s2 * s2).sum().clamp_min(1e-12)).item()


def build(variant, trained):
    m = ReprLearningModel(matched(ReprConfig()), variant=variant).to(dev)
    if trained:
        sd = torch.load(CK(variant), map_location="cpu", weights_only=False)
        m.load_state_dict(sd["model_state_dict"], strict=False)
    m.eval()
    return m


tok = AutoTokenizer.from_pretrained(BACKBONE)
if tok.pad_token is None: tok.pad_token = tok.eos_token
dl = make_sentence_dataloader(tok, batch_size=16, src_tokenizer_name=SRC,
                              split="val", num_workers=0, pad_token_id=0)
it = iter(dl)
batches = []
for _ in range(6):
    b = next(it)
    for a in ("context_ids", "context_mask", "question_ids", "question_mask",
              "answer_ids", "answer_mask", "answer_content_mask"):
        v = getattr(b, a, None)
        if torch.is_tensor(v): setattr(b, a, v.to(dev))
    batches.append(b)


def read_utility(m):
    """REAL/OFF/SHUF (same seeded mask) + memory eff_rank, averaged over batches."""
    real = off = shuf = 0.0; ranks = []
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for b in batches:
            torch.manual_seed(0); real += m.compute_masked_reconstruction_loss(b)["loss_recon"].item()
            torch.manual_seed(0); off += m.compute_masked_reconstruction_loss(b, zero_memory=True)["loss_recon"].item()
            torch.manual_seed(0); shuf += m.compute_masked_reconstruction_loss(b, shuffle_memory=True)["loss_recon"].item()
            st = m.encoder.init_streaming_state(b.context_ids.shape[0], dev, torch.float32)
            emb = m.decoder.llama.get_input_embeddings()(b.context_ids)
            st, _ = m.encoder.streaming_write(st, emb, b.context_mask)
            mem, _ = m.encoder.finalize_memory(st)
            for bi in range(mem.shape[0]):
                ranks.append(eff_rank(mem[bi]))
    n = len(batches)
    return real / n, off / n, shuf / n, sum(ranks) / len(ranks), mem.shape[1]


def report_read(tag, m):
    r, o, s, er, M = read_utility(m)
    print(f"  [{tag}] REAL={r:.3f} OFF={o:.3f} SHUF={s:.3f}  | "
          f"OFF-REAL={o-r:+.3f} (memory helps)  SHUF-REAL={s-r:+.3f} (binding)  | "
          f"mem eff_rank={er:.1f}/{M}")


# ════════════════════ hlvocab ════════════════════
print("=" * 80); print("HLVOCAB (v2)  — fresh vs trained"); print("=" * 80)
for tag, trained in (("fresh", False), ("trained", True)):
    m = build("hlvocab_baseline", trained); sub = m.encoder.sub
    print(f"\n--- {tag} ---")
    # node collapse + eff_rank per layer
    for l in range(sub.depth):
        for nm, P in (("keys", sub.node_keys[l]), ("vals", sub.node_values[l])):
            u = F.normalize(P.float(), dim=-1); c = (u @ u.t()).abs()
            N = c.shape[0]; off = (c.sum() - N) / (N * (N - 1))
            print(f"  L{l} node_{nm}: N={N:4d} eff_rank={eff_rank(P):6.1f} mean|cos|={off:.3f}")
    # usage + routing over batches
    usage = [torch.zeros(n, device=dev) for n in sub.config.nodes]
    ent_acc = [0.0] * sub.depth; hub_acc = [0.0] * sub.depth
    with torch.no_grad():
        for b in batches:
            st = m.encoder.init_streaming_state(b.context_ids.shape[0], dev, torch.float32)
            emb = m.decoder.llama.get_input_embeddings()(b.context_ids)
            st, _ = m.encoder.streaming_write(st, emb, b.context_mask)
            hiddens, mask = st["hiddens"], st["mask"].float()
            mm = mask.unsqueeze(-1)
            x = _unit_rms(sub.route_projs[0](hiddens.float())) * mm if sub.config.use_graph else None
            xx = hiddens.float()
            for l in range(sub.depth):
                sc = sub.route(l, (hiddens.float() if l == 0 else xx)) * mm
                am = sc.argmax(-1)[mask.bool()]
                usage[l].scatter_add_(0, am, torch.ones_like(am, dtype=torch.float))
                p = sc.clamp_min(1e-12)
                ent_acc[l] += (-(p * p.log()).sum(-1) * mask).sum().item() / mask.sum().item()
                actm = sc.sum(1); hub_acc[l] += (actm / actm.sum(-1, keepdim=True).clamp_min(1e-6)).max(-1).values.mean().item()
                if l < sub.depth - 1: xx = sub._perturb(l, xx, sc) * mm
    nb = len(batches)
    for l in range(sub.depth):
        used = (usage[l] > 0).sum().item(); N = sub.config.nodes[l]
        print(f"  L{l}: used {used:4d}/{N} ({100*used/N:4.1f}%)  route_ent={ent_acc[l]/nb:.2f}  hub={hub_acc[l]/nb:.3f}")
    # edge selection telemetry (v2)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _, aux = sub(st["hiddens"], st["mask"].float())
    for k in ("hlvocab_slot_uniq_edges", "hlvocab_edge_inter_frac", "hlvocab_sel_attn_entropy",
              "hlvocab_sel_attn_max", "hlvocab_sel_temp", "hlvocab_memory_norm"):
        if k in aux: print(f"  {k} = {float(aux[k]):.3f}")
    report_read(tag, m)
    del m, sub; torch.cuda.empty_cache()

# ════════════════════ soft_pointer_graph ════════════════════
print("\n" + "=" * 80); print("SOFT_POINTER_GRAPH — fresh vs trained"); print("=" * 80)
for tag, trained in (("fresh", False), ("trained", True)):
    m = build("soft_pointer_graph_baseline", trained)
    print(f"\n--- {tag} ---")
    with torch.no_grad():
        b = batches[0]
        st = m.encoder.init_streaming_state(b.context_ids.shape[0], dev, torch.float32)
        emb = m.decoder.llama.get_input_embeddings()(b.context_ids)
        st, _ = m.encoder.streaming_write(st, emb, b.context_mask)
        _, aux = m.encoder.finalize_memory(st)
    for k in ("spg_node_collapse_cos", "spg_read_src_entropy", "spg_read_dst_entropy",
              "spg_node_active_frac", "spg_state_effect", "spg_fact_norm",
              "spg_node_gate_mean_avg", "spg_edge_gate_mean_avg"):
        if k in aux: print(f"  {k} = {float(aux[k]):.3f}")
    report_read(tag, m)
    del m; torch.cuda.empty_cache()

print("\nDONE.")
