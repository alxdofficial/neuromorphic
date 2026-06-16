"""Memory-related trainable-param count per MAE variant at the SmolLM2-135M
(d=576) scale — the capacity-match probe.

"Memory params" = total trainable MINUS the shared floor (vanilla_llama: decoder
LoRA + mask_embed, identical across every arm). That isolates the cost of each
compression MECHANISM so the baselines can be matched to hlvocab (~graph_v9).

Builds on CPU (param counts are device-independent) so the multiple frozen base
copies the ports clone don't OOM the GPU. Set ranks via the MAE_RANKS dict to
preview a re-match before editing train.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import torch
from src.memory.config import ReprConfig
from src.memory.model import ReprLearningModel
from src.memory.common import beacon_wrap_layers

BACKBONE = "HuggingFaceTB/SmolLM2-135M"
N_LAYERS = 30  # SmolLM2-135M

# The masked_reconstruction override block from scripts/train/train.py, mirrored
# here so this probe measures exactly what the trainer builds. Edit to preview.
MAE_RANKS = dict(
    icae_lora_rank=76, icae_lora_alpha=152,
    ccm_lora_rank=38, ccm_lora_alpha=76,
    autocompressor_lora_rank=38, autocompressor_lora_alpha=76,
    beacon_ratio=8, beacon_wrap_layers=beacon_wrap_layers(N_LAYERS, 8),
    # soft_pointer_graph capacity-matched to hlvocab (~3.30M memory)
    spg_K_edge=16, spg_K_node=64, spg_d_node=176, spg_d_state=176, spg_d_read=176,
    spg_d_updater=240, spg_updater_layers=2, spg_updater_heads=8,
    spg_read_ffn_mult=2, spg_builder_mlp_hidden=224, spg_film_hidden=176,
)


def mae_cfg():
    c = ReprConfig()
    c.llama_model = BACKBONE; c.d_llama = 576; c.llama_vocab_size = 49152
    c.pad_token_id = 0; c.task_mode = "masked_reconstruction"; c.device = "cpu"
    c.use_llama_lora = True; c.llama_lora_rank = 16; c.llama_lora_alpha = 32
    c.n_flat_codes = 16
    c.icae_n_slots = 16; c.ccm_n_comp = 16; c.autocompressor_n_slots = 16
    c.hlvocab_d_code = 256; c.hlvocab_nodes = (512, 256, 128)
    c.hlvocab_top_k = 4; c.hlvocab_m_max = 16; c.hlvocab_tap_layer = 6
    c.hlvocab_use_graph = True
    for k, v in MAE_RANKS.items():
        setattr(c, k, v)
    return c


def n_trainable(variant):
    m = ReprLearningModel(mae_cfg(), variant=variant)
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


VARIANTS = ["vanilla_llama", "graph_baseline", "hlvocab_baseline", "soft_pointer_graph_baseline",
            "icae_baseline", "ccm_baseline", "autocompressor_baseline", "beacon_baseline"]

print(f"backbone={BACKBONE} d=576; ranks={MAE_RANKS}\n")
floor = n_trainable("vanilla_llama")
print(f"{'variant':<26}{'total_trainable':>16}{'memory_params':>16}")
print("-" * 58)
for v in VARIANTS:
    tot = floor if v == "vanilla_llama" else n_trainable(v)
    mem = tot - floor
    tag = "  (shared floor: decoder LoRA + mask_embed)" if v == "vanilla_llama" else ""
    print(f"{v:<26}{tot:>16,}{mem:>16,}{tag}")
