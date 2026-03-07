"""
Entry point for neuromorphic LM training.

Usage:
    python -m src.train                              # file defaults (backward compat)
    python -m src.train --phases A,B --tier a        # auto-transition full run
    python -m src.train --phase A --steps 5000       # single phase
    python -m src.train --phase B --resume ckpt.pt   # lifelong phase with resume
    python -m src.train --phases A,B --no-plots      # skip plot generation

All configuration constants below serve as defaults when no CLI args are given.
CLI args override the corresponding constant.
"""

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import time
import torch
from typing import Any

from .model import ModelConfig, NeuromorphicLM
from .model.state import save_runtime_state, load_runtime_state
from .data import PHASE_CONFIGS, create_dataloader, get_special_token_ids, get_tokenizer
from .training import TBPTTTrainer, evaluate_validation
from .debug import MetricsCollector


# ============================================================================
# Configuration — edit these directly (CLI args override when provided)
# ============================================================================

# -- Model tier --
TIER = "a"          # D=2048, D_embed=384, C=16, L_scan=6, exp=8, B=4, M=384
# TIER = "b"        # D=3072, D_embed=512, C=16, L_scan=12, B=6, d_inner=1024 (~251M)

# -- Training phase --
PHASE = "A"         # PM + EM + PCM (all systems)
# PHASE = "B"       # A + lifelong (PM/EM persist across docs)

# -- Data phase (which datasets to use) --
DATA_PHASE = None   # None => auto: uses The Pile (local)
# DATA_PHASE = "B"  # FineWeb-Edu + DCLM (legacy)

# -- Tokenizer --
TOKENIZER = "tinyllama"
# TOKENIZER = "gpt2"
# TOKENIZER = "smollm"

# -- Batch size (persistent streams) --
BS = 16             # Tier A: K_segments=4, grad_ckpt — same tok/step as K=8 BS=8, better GPU util
# BS = 6            # Tier B default
# BS = 4            # Tier C default

# -- Memory horizon --
K_SEGMENTS = 4      # TBPTT chunk = K_segments * N tokens (4 × 512 = 2048)
                    # 2048-token gradient horizon for PM/EM neuromodulators; fits larger BS than K=8
GRADIENT_CHECKPOINTING = True  # halves scan activation memory; enables K_segments=8

# -- Seeds --
TRAIN_SEED = 42

# -- Learning rate --
LR = 3e-4
LR_MIN = 3e-5           # 10% of peak (standard cosine floor)
WARMUP_STEPS = 1000

# -- Training length --
# Priority: MAX_STEPS (explicit) > MAX_TOKENS (derived) > PHASE_DEFAULT_STEPS.
MAX_STEPS = None            # absolute step target; e.g. 5000
MAX_TOKENS = None           # token budget; converted via BS*T
USE_PHASE_DEFAULT_STEPS = True
PHASE_DEFAULT_STEPS = {
    "A": 30_517,            # 1.5B tokens @ BS=24, K_segments=4, N=512 (49,152 tok/step)
    "B": 15_258,            # 750M tokens — same budget, lifelong mode
}
# Phase A: PM/EM reset at doc boundaries — memory systems learn basic function
# Phase B: PM/EM persist across all docs — lifelong accumulation

# -- Regularization --
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0

# -- Logging --
LOG_INTERVAL = 50

# -- Validation / monitoring --
VAL_INTERVAL = 2000     # run held-out eval every N train steps (0 = disabled)
VAL_STEPS = 20          # validation chunks per eval pass
VAL_DATA_PHASE = None   # None => same policy as DATA_PHASE auto-logic
VAL_SEED = 4242
ABLATE_INTERVAL = 2000  # run PM/EM ablation eval every N steps (0 = disabled)
PLOT_INTERVAL = 2000    # regenerate training curve plots every N steps (0 = disabled)
TEXT_SAMPLE_INTERVAL = 2000  # generate text comparison samples every N steps (0 = disabled)

# -- Safety checks --
SAFETY_FAIL_FAST = True
MAX_CONSEC_ZERO_VALID = 3

# -- Checkpointing --
SAVE_DIR = "checkpoints"
SAVE_INTERVAL = 10000
RESUME = None       # set to checkpoint path to resume, e.g. "checkpoints/neuromorphic_a_A_step5000.pt"

# -- Metrics collection --
METRICS_FILE = "checkpoints/metrics.jsonl"
COLLECT_EVERY = 50  # full collection every N steps (0 = disabled)

# -- Device --
DEVICE = None       # auto-detect (cuda if available, else cpu)
# DEVICE = "cuda"
# DEVICE = "cpu"


# ============================================================================
# CLI argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. All optional; file-level constants are defaults."""
    p = argparse.ArgumentParser(
        description="Neuromorphic LM training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("--phases", type=str, default=None,
                       help="Comma-separated auto-transition sequence (e.g. A,B)")
    group.add_argument("--phase", type=str, default=None,
                       help="Single phase to run (A or B)")
    p.add_argument("--tier", type=str, default=None,
                   choices=["a", "b", "c"],
                   help="Model size tier")
    p.add_argument("--resume", type=str, default=None,
                   help="Checkpoint path to resume from")
    p.add_argument("--steps", type=int, default=None,
                   help="Override max steps (per phase in auto mode)")
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate")
    p.add_argument("--lr-min", type=float, default=None,
                   help="Minimum learning rate for cosine decay floor")
    p.add_argument("--warmup-steps", type=int, default=None,
                   help="LR warmup steps")
    p.add_argument("--bs", type=int, default=None,
                   help="Batch size")
    p.add_argument("--save-dir", type=str, default=None,
                   help="Checkpoint output directory")
    p.add_argument("--metrics-file", type=str, default=None,
                   help="JSONL metrics path")
    p.add_argument("--no-plots", action="store_true", default=False,
                   help="Skip automatic plot generation")
    p.add_argument("--tokenizer", type=str, default=None,
                   help="Tokenizer name")
    p.add_argument("--data-phase", type=str, default=None,
                   help="Training dataset phase key (e.g. A, B, B-diverse, longctx)")
    p.add_argument("--val-data-phase", type=str, default=None,
                   help="Validation dataset phase key (default follows train data phase policy)")
    p.add_argument("--seed", type=int, default=None,
                   help="Training seed")
    p.add_argument("--config", type=str, default=None,
                   help="YAML config path with presets")
    p.add_argument("--preset", type=str, default=None,
                   help="Preset name inside YAML config")
    p.add_argument("--output-root", type=str, default=None,
                   help="Base directory for per-run outputs")
    p.add_argument("--run-name", type=str, default=None,
                   help="Run grouping name under output-root")
    p.add_argument("--save-interval", type=int, default=None,
                   help="Checkpoint save interval in steps")
    p.add_argument("--plot-interval", type=int, default=None,
                   help="Live plot regeneration interval in steps (0 = disabled)")
    p.add_argument("--val-interval", type=int, default=None,
                   help="Validation interval in steps")
    p.add_argument("--text-sample-interval", type=int, default=None,
                   help="Text sample generation interval in steps (0 = disabled)")
    p.add_argument("--log-interval", type=int, default=None,
                   help="Log print interval in steps")
    # torch.compile
    p.add_argument("--compile", action="store_true", default=None,
                   help="Enable torch.compile for CUDA training")
    p.add_argument("--no-compile", dest="compile", action="store_false",
                   help="Disable torch.compile")
    # Predictive Coding Module
    p.add_argument("--pcm", action="store_true", default=None,
                   help="Enable Predictive Coding Module")
    p.add_argument("--no-pcm", dest="pcm", action="store_false",
                   help="Disable Predictive Coding Module")
    return p.parse_args()


def _load_yaml_config(path: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError(
            "PyYAML is required for --config support. Install with: pip install pyyaml"
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError("Top-level YAML config must be a mapping/object.")
    return data


def _slugify(value: str) -> str:
    s = value.strip().lower()
    s = re.sub(r"[^a-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s.strip("_") or "run"


def _normalize_phase_entry(entry: Any) -> dict[str, Any]:
    if isinstance(entry, str):
        phase = entry.strip().upper()
        return {
            "phase": phase,
            "steps": None,
            "tokens": None,
            "resume": "auto_previous",
        }
    if isinstance(entry, dict):
        raw_phase = entry.get("phase")
        if raw_phase is None:
            raise ValueError("Each phase entry in config must include 'phase'.")
        phase = str(raw_phase).strip().upper()
        resume = entry.get("resume", "auto_previous")
        if isinstance(resume, str):
            resume = resume.strip()
        if resume in ("", "null", "none", "None"):
            resume = None
        return {
            "phase": phase,
            "steps": entry.get("steps"),
            "tokens": entry.get("tokens"),
            "resume": resume,
        }
    raise ValueError(f"Invalid phase entry type: {type(entry)!r}")


def _normalize_phase_plan(raw: Any) -> list[dict[str, Any]] | None:
    if raw is None:
        return None
    if not isinstance(raw, list):
        raise ValueError("Config field 'phases' must be a list.")
    plan = [_normalize_phase_entry(e) for e in raw]
    for item in plan:
        p = item["phase"]
        if p not in ("A", "B"):
            raise ValueError(f"Unknown phase '{p}' in config phases. Expected A/B.")
    return plan


def _merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for k, v in override.items():
        merged[k] = v
    return merged


def _prepare_run_paths(settings: dict, run_id: str) -> dict[str, str]:
    explicit_save = settings.get("save_dir")
    explicit_metrics = settings.get("metrics_file")
    output_root = settings.get("output_root", "outputs")
    run_name = _slugify(settings.get("run_name", "manual"))

    if explicit_save is None and explicit_metrics is None:
        run_dir = os.path.join(output_root, run_name, run_id)
        save_dir = os.path.join(run_dir, "checkpoints")
        metrics_file = os.path.join(run_dir, "metrics.jsonl")
    else:
        if explicit_save is not None and explicit_metrics is None:
            save_dir = explicit_save
            run_dir = save_dir
            metrics_file = os.path.join(run_dir, "metrics.jsonl")
        elif explicit_metrics is not None and explicit_save is None:
            metrics_file = explicit_metrics
            run_dir = os.path.dirname(metrics_file) or "."
            save_dir = os.path.join(run_dir, "checkpoints")
        else:
            save_dir = explicit_save or SAVE_DIR
            metrics_file = explicit_metrics or METRICS_FILE
            run_dir = os.path.dirname(metrics_file) or save_dir

    return {
        "run_dir": run_dir,
        "save_dir": save_dir,
        "metrics_file": metrics_file,
        "metadata_file": os.path.join(run_dir, "run_meta.json"),
        "state_file": os.path.join(run_dir, "run_state.json"),
    }


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)


def resolve_settings(args: argparse.Namespace) -> dict:
    """Merge constants, YAML preset, and CLI args (CLI wins)."""
    config_payload: dict[str, Any] = {}
    preset_name: str | None = args.preset
    preset_payload: dict[str, Any] = {}

    config_path = args.config
    if config_path is None and preset_name is not None:
        # Auto-load default preset config when --preset is used without --config
        default_config = os.path.join(os.path.dirname(__file__), "..", "configs", "train_presets.yaml")
        if os.path.exists(default_config):
            config_path = default_config
    if config_path is not None:
        config_payload = _load_yaml_config(config_path)
        all_presets = config_payload.get("presets", {})
        if not isinstance(all_presets, dict):
            raise ValueError("Config field 'presets' must be a mapping/object.")

        if preset_name is None:
            preset_name = config_payload.get("default_preset")
            if preset_name is None and len(all_presets) == 1:
                preset_name = next(iter(all_presets.keys()))
        if preset_name is not None:
            if preset_name not in all_presets:
                raise ValueError(
                    f"Preset '{preset_name}' not found in {args.config}. "
                    f"Available: {list(all_presets.keys())}"
                )
            defaults_payload = config_payload.get("defaults", {})
            if defaults_payload is None:
                defaults_payload = {}
            if not isinstance(defaults_payload, dict):
                raise ValueError("Config field 'defaults' must be a mapping/object.")
            preset_payload = _merge_dict(defaults_payload, all_presets[preset_name] or {})

    phase_plan = _normalize_phase_plan(preset_payload.get("phases"))
    phases_list = None
    if phase_plan is not None:
        phases_list = [p["phase"] for p in phase_plan]
    if args.phases is not None:
        phases_list = [p.strip().upper() for p in args.phases.split(",")]
        phase_plan = _normalize_phase_plan(phases_list)

    phase_single = str(preset_payload.get("phase", PHASE)).upper()
    if args.phase is not None:
        phase_single = args.phase.upper()
        phases_list = None
        phase_plan = None

    run_name_default = preset_name or ("_".join(phases_list).lower() if phases_list else f"phase_{phase_single.lower()}")

    settings = {
        "config_path": args.config,
        "preset": preset_name,
        "phase_plan": phase_plan,
        "phases": phases_list,
        "phase": phase_single,
        "tier": args.tier if args.tier is not None else str(preset_payload.get("tier", TIER)),
        "resume": args.resume if args.resume is not None else preset_payload.get("resume", RESUME),
        "steps_override": args.steps if args.steps is not None else preset_payload.get("steps_override"),
        "max_steps": preset_payload.get("max_steps", MAX_STEPS),
        "max_tokens": preset_payload.get("max_tokens", MAX_TOKENS),
        "use_phase_default_steps": preset_payload.get("use_phase_default_steps", USE_PHASE_DEFAULT_STEPS),
        "lr": args.lr if args.lr is not None else float(preset_payload.get("lr", LR)),
        "lr_min": args.lr_min if args.lr_min is not None else float(preset_payload.get("lr_min", LR_MIN)),
        "warmup_steps": (
            args.warmup_steps
            if args.warmup_steps is not None
            else int(preset_payload.get("warmup_steps", WARMUP_STEPS))
        ),
        "bs": args.bs if args.bs is not None else int(preset_payload.get("bs", BS)),
        "save_dir": args.save_dir or preset_payload.get("save_dir"),
        "metrics_file": args.metrics_file or preset_payload.get("metrics_file"),
        "save_interval": (
            args.save_interval
            if args.save_interval is not None
            else int(preset_payload.get("save_interval", SAVE_INTERVAL))
        ),
        "plot_interval": (
            args.plot_interval
            if args.plot_interval is not None
            else int(preset_payload.get("plot_interval", PLOT_INTERVAL))
        ),
        "val_interval": (
            args.val_interval
            if args.val_interval is not None
            else int(preset_payload.get("val_interval", VAL_INTERVAL))
        ),
        "text_sample_interval": (
            args.text_sample_interval
            if args.text_sample_interval is not None
            else int(preset_payload.get("text_sample_interval", TEXT_SAMPLE_INTERVAL))
        ),
        "log_interval": (
            args.log_interval
            if args.log_interval is not None
            else int(preset_payload.get("log_interval", LOG_INTERVAL))
        ),
        "no_plots": args.no_plots or bool(preset_payload.get("no_plots", False)),
        "tokenizer": args.tokenizer or preset_payload.get("tokenizer", TOKENIZER),
        "data_phase": (
            args.data_phase if args.data_phase is not None
            else preset_payload.get("data_phase", DATA_PHASE)
        ),
        "val_data_phase": (
            args.val_data_phase if args.val_data_phase is not None
            else preset_payload.get("val_data_phase", VAL_DATA_PHASE)
        ),
        "seed": args.seed if args.seed is not None else int(preset_payload.get("seed", TRAIN_SEED)),
        "output_root": args.output_root or preset_payload.get("output_root", "outputs"),
        "run_name": args.run_name or preset_payload.get("run_name", run_name_default),
        # torch.compile
        "use_compile": (
            args.compile
            if args.compile is not None
            else preset_payload.get("use_compile")
        ),
        # Predictive Coding Module
        "pcm_enabled": (
            args.pcm
            if args.pcm is not None
            else preset_payload.get("pcm_enabled")
        ),
    }

    if isinstance(settings["resume"], str):
        r = settings["resume"].strip()
        if r in ("", "null", "none", "None"):
            settings["resume"] = None
        else:
            settings["resume"] = r

    return settings


# ============================================================================
# Helpers
# ============================================================================

def _get_device() -> torch.device:
    if DEVICE:
        return torch.device(DEVICE)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _build_config(tier: str, phase: str, settings: dict | None = None) -> ModelConfig:
    tier_fns = {
        "a": ModelConfig.tier_a,
        "b": ModelConfig.tier_b,
        "c": ModelConfig.tier_c,
        "tiny": ModelConfig.tier_tiny,
    }
    if tier not in tier_fns:
        raise ValueError(f"Unknown tier {tier!r}. Available: {list(tier_fns)}")
    tier_fn = tier_fns[tier]
    config = tier_fn()
    config.set_phase(phase)
    # Memory horizon and activation memory settings
    config.K_segments = K_SEGMENTS
    config.gradient_checkpointing = GRADIENT_CHECKPOINTING
    if settings is not None:
        if settings.get("use_compile") is not None:
            config.use_compile = settings["use_compile"]
        if settings.get("pcm_enabled") is not None:
            config.pcm_enabled = settings["pcm_enabled"]
    return config


def get_git_commit() -> str:
    """Best-effort current git commit hash."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except Exception:
        return "unknown"


def config_digest(config: ModelConfig) -> str:
    """Stable short hash for the active config."""
    payload = json.dumps(config.__dict__, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def _resolve_data_phase(train_phase: str, requested: str | None) -> str:
    """Resolve dataset phase: explicit value or phase-aware default."""
    if requested is not None:
        req = str(requested).strip()
        if not req:
            raise ValueError("Data phase cannot be an empty string.")
        if req in PHASE_CONFIGS:
            return req
        # Accept case-insensitive names and return canonical key.
        canon = {k.lower(): k for k in PHASE_CONFIGS}
        key = req.lower()
        if key in canon:
            return canon[key]
        raise ValueError(
            f"Unknown data phase '{requested}'. Available: {list(PHASE_CONFIGS.keys())}"
        )
    # All phases use the same dataset (The Pile) so loss changes at phase
    # transitions reflect only model capability changes, not distribution shift.
    return "A"


def _resolve_max_steps(
    train_phase: str,
    tokens_per_step: int,
    settings: dict,
    phase_entry: dict[str, Any] | None = None,
) -> tuple[int, str]:
    """Resolve absolute max steps from CLI override / explicit steps / tokens / phase defaults."""
    if phase_entry is not None:
        if phase_entry.get("steps") is not None:
            return int(phase_entry["steps"]), "phase_plan.steps"
        if phase_entry.get("tokens") is not None:
            steps = math.ceil(float(phase_entry["tokens"]) / max(tokens_per_step, 1))
            return int(steps), "phase_plan.tokens"
    if settings["steps_override"] is not None:
        return int(settings["steps_override"]), "cli_override"
    max_steps_cfg = settings.get("max_steps", MAX_STEPS)
    if max_steps_cfg is not None:
        return int(max_steps_cfg), "max_steps"
    max_tokens_cfg = settings.get("max_tokens", MAX_TOKENS)
    if max_tokens_cfg is not None:
        steps = math.ceil(float(max_tokens_cfg) / max(tokens_per_step, 1))
        return int(steps), "max_tokens"
    if settings.get("use_phase_default_steps", USE_PHASE_DEFAULT_STEPS):
        phase = train_phase.upper()
        if phase in PHASE_DEFAULT_STEPS:
            return int(PHASE_DEFAULT_STEPS[phase]), "PHASE_DEFAULT_STEPS"
    return 5000, "fallback_5000"


# ============================================================================
# run_phase — one complete training phase
# ============================================================================

def run_phase(
    phase: str,
    tier: str,
    resume_path: str | None,
    run_id: str,
    settings: dict,
    metrics_file: str,
    device: torch.device,
    tokenizer,
    special_ids: dict,
    is_auto: bool = False,
    global_step_offset: int = 0,
    phase_entry: dict[str, Any] | None = None,
) -> tuple[str | None, int, int]:
    """Run a single training phase.

    Returns:
        (final_checkpoint_path, steps_completed_in_this_call, phase_end_global_step)
    """

    phase_name = phase.upper()
    bs = settings["bs"]
    lr = settings["lr"]
    lr_min = settings.get("lr_min", LR_MIN)
    warmup_steps = settings.get("warmup_steps", WARMUP_STEPS)
    seed = settings["seed"]
    save_dir = settings["save_dir"]
    ckpt = None

    # Config
    config = _build_config(tier, phase_name, settings)
    config.vocab_size = len(tokenizer)
    config.validate()
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    config.eot_id = eot_id

    train_data_phase = _resolve_data_phase(phase_name, settings.get("data_phase", DATA_PHASE))
    val_data_phase = _resolve_data_phase(phase_name, settings.get("val_data_phase", VAL_DATA_PHASE))
    tokens_per_step = bs * config.T
    max_steps_total, max_steps_source = _resolve_max_steps(
        phase_name, tokens_per_step, settings, phase_entry=phase_entry
    )
    target_tokens = max_steps_total * tokens_per_step

    print(f"Tier: {tier.upper()} | Phase: {phase_name} | "
          f"D={config.D}, B={config.B}, C={config.C}, D_col={config.D_col}, L_scan={config.L_scan}")
    print(f"BS={bs}, T={config.T}, N={config.N}, K_segments={config.K_segments}")
    print(f"PM={config.pm_enabled}, EM={config.em_enabled}")
    print(
        f"Run length: max_steps={max_steps_total} "
        f"(source={max_steps_source}), tokens/step={tokens_per_step}, "
        f"target_tokens~{target_tokens:,}"
    )
    print(f"Data phase: train={train_data_phase}, val={val_data_phase}")
    if phase_name in ("B", "C") and resume_path is None:
        print(
            "Warning: running a later phase without resume. "
            "This is valid for debugging, but not a smooth phase transition."
        )

    # Model
    model = NeuromorphicLM(config).to(device)
    if device.type == "cuda":
        model = model.to(torch.bfloat16)
    param_count = model.param_count()
    print(f"Parameters: {param_count:,} ({param_count / 1e6:.1f}M)")

    # Optimizer — exclude biases and LayerNorm from weight decay
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": WEIGHT_DECAY},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        fused=(device.type == "cuda"),
    )

    # LR scheduler: linear warmup + cosine decay (per-phase: fresh warmup each phase)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(max_steps_total - warmup_steps, 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return lr_min / lr + (1.0 - lr_min / lr) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Resume from checkpoint
    phase_changed = False
    if resume_path:
        print(f"Resuming from {resume_path}")
        if ckpt is None:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        # Handle vocab size mismatch (e.g. tokenizer changed after checkpoint).
        # Pop mismatched embedding/lm_head keys from state dict so
        # load_state_dict doesn't raise on shape mismatch, then manually
        # copy the overlapping rows into the (already correctly-sized) model.
        state_dict = ckpt["model_state_dict"]
        ckpt_emb = state_dict.get("embedding.weight")
        vocab_mismatch = (
            ckpt_emb is not None and ckpt_emb.shape[0] != config.vocab_size
        )
        if vocab_mismatch:
            ckpt_vocab_size = ckpt_emb.shape[0]
            print(f"  Vocab size mismatch: checkpoint={ckpt_vocab_size}, "
                  f"current={config.vocab_size}. "
                  f"Will copy overlapping rows after load.")
            # Remove vocab-dependent keys so load_state_dict won't error
            emb_weight = state_dict.pop("embedding.weight")
            lm_head_weight = state_dict.pop("lm_head.weight", None)

        incompat = model.load_state_dict(state_dict, strict=False)

        # Patch embedding / lm_head rows from checkpoint
        if vocab_mismatch:
            copy_n = min(ckpt_vocab_size, config.vocab_size)
            with torch.no_grad():
                model.embedding.weight.data[:copy_n] = emb_weight[:copy_n]
                if not config.tie_embeddings and lm_head_weight is not None:
                    model.lm_head.weight.data[:copy_n] = lm_head_weight[:copy_n]

        if incompat.missing_keys:
            print(f"  Missing keys on resume: {len(incompat.missing_keys)}")
        if incompat.unexpected_keys:
            print(f"  Unexpected keys on resume: {len(incompat.unexpected_keys)}")
        # Detect phase/architecture transition: optimizer param groups may have changed.
        # Check phase toggles, structural memory sizes, decoder config, and layer config.
        ckpt_config = ckpt.get("config")
        _structural_fields = (
            # Phase toggles
            "pm_enabled", "em_enabled",
            # Architecture
            "D", "B", "C", "D_col", "D_embed",
            "L_scan", "scan_expansion",
            # Memory dimensions
            "M", "n_trail_steps",
            # Lifelong mode
            "lifelong_mode",
        )
        _changed_fields = []
        if ckpt_config is not None:
            for f in _structural_fields:
                old_val = getattr(ckpt_config, f, None)
                new_val = getattr(config, f, None)
                if old_val != new_val:
                    _changed_fields.append(f)
        phase_changed = len(_changed_fields) > 0
        if phase_changed:
            print("  Architecture transition detected — reinitializing optimizer state.")
            print("  Changed fields:")
            for f in _changed_fields:
                old_val = getattr(ckpt_config, f, None)
                new_val = getattr(config, f, None)
                print(f"    {f}: {old_val} -> {new_val}")
        else:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError) as e:
                print(f"  Optimizer state incompatible ({e}); reinitializing.")
        # Scheduler: only restore for same-phase, single-phase resume.
        # Phase transitions (phase_changed) and auto mode always get a fresh
        # warmup + cosine schedule so the new phase's LR ramp isn't polluted
        # by the prior phase's last_epoch.
        if "scheduler_state_dict" in ckpt:
            if not phase_changed:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except (ValueError, KeyError):
                    print("  Scheduler state incompatible; reinitializing.")
            else:
                print("  Fresh LR scheduler for new phase.")
        runtime_state_loaded = False
        if "runtime_state" in ckpt:
            # Initialize state tensors so load_state_runtime can shape-check.
            model.initialize_states(bs, device)

            if phase_changed:
                print("  Phase transition: loading compatible runtime state (shape-safe).")
            load_runtime_state(model, ckpt["runtime_state"])
            runtime_state_loaded = True
        start_step = ckpt.get("step", 0)
    else:
        start_step = 0
        runtime_state_loaded = False

    # Phase transitions and auto mode always train the full step budget.
    # Only same-phase single-phase resume continues from the checkpoint step.
    if phase_changed or is_auto:
        start_step = 0

    # Dataloader — creates a fresh stream from seed. On resume, the model's
    # runtime state (h, PM, EM) is restored but the dataloader restarts from
    # the beginning. Only prev_token is patched (below) to avoid a false
    # doc-boundary reset on the first batch. This means resumed memory state
    # is inconsistent with the token stream for the first few chunks until
    # PM/EM naturally adapt. This is acceptable because PM/EM are designed
    # to be robust to distribution shifts (same mechanism as Phase C lifelong).
    dataloader = create_dataloader(
        phase=train_data_phase,
        tokenizer=tokenizer,
        batch_size=bs,
        seq_length=config.T,
        seed=seed,
        max_steps=max_steps_total,
    )

    # Metrics collector (append mode, with phase tag)
    collector = MetricsCollector(
        model=model,
        config=config,
        output_path=metrics_file,
        collect_every=COLLECT_EVERY,
        phase=phase_name,
    )
    print(f"Metrics: {metrics_file} (full every {COLLECT_EVERY} steps)")

    # Phase start marker
    phase_start_step = (global_step_offset + start_step) if is_auto else start_step
    collector.log_record({
        "mode": "phase_start",
        "phase": phase_name,
        "step": phase_start_step,
        "run_id": run_id,
        "resume_from": resume_path,
    })

    # Run metadata
    collector.log_record({
        "mode": "run_meta",
        "run_id": run_id,
        "start_unix": time.time(),
        "git_commit": get_git_commit(),
        "tier": tier.upper(),
        "phase": phase_name,
        "data_phase": train_data_phase,
        "val_data_phase": val_data_phase,
        "tokenizer": settings["tokenizer"],
        "seed_train": seed,
        "seed_val": VAL_SEED,
        "max_steps_total": max_steps_total,
        "max_steps_source": max_steps_source,
        "tokens_per_step": tokens_per_step,
        "target_tokens": target_tokens,
        "config_digest": config_digest(config),
        "config": dict(config.__dict__),
    })

    # Trainer
    trainer = TBPTTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        dataloader=dataloader,
        config=config,
        device=device,
        max_grad_norm=MAX_GRAD_NORM,
        log_interval=settings.get("log_interval", LOG_INTERVAL),
        collector=collector,
        fail_fast=SAFETY_FAIL_FAST,
        max_consecutive_zero_valid=MAX_CONSEC_ZERO_VALID,
    )
    # Continuous step numbering for JSONL in auto mode
    trainer.global_step = (global_step_offset + start_step) if is_auto else start_step

    # If runtime state was loaded, mark states as initialized so train_chunk()
    # doesn't reinitialize and wipe the loaded PM/EM state.
    if runtime_state_loaded:
        trainer._states_initialized = True

    # Restore last prev_token to prevent false doc-boundary reset on resume
    if resume_path and ckpt.get("last_prev_token") is not None:
        trainer.override_prev_token = ckpt["last_prev_token"]

    # Training loop
    remaining = max_steps_total - start_step
    if remaining <= 0:
        print(
            f"No training steps remaining: start_step={start_step}, "
            f"max_steps_total={max_steps_total}. Exiting phase {phase_name}."
        )
        collector.log_record({
            "mode": "phase_end",
            "phase": phase_name,
            "step": trainer.global_step,
            "final_checkpoint": None,
        })
        collector.close()
        return None, 0, trainer.global_step

    print(f"\nStarting phase {phase_name} training ({remaining} steps from step {trainer.global_step})...")
    print("=" * 60)

    os.makedirs(save_dir, exist_ok=True)

    def save_checkpoint(step: int) -> str:
        save_path = os.path.join(
            save_dir,
            f"neuromorphic_{tier}_{phase_name}_{run_id}_step{step}.pt"
        )
        ckpt_data = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "step": step,
            "config": config,
            "runtime_state": save_runtime_state(model),
            "last_prev_token": (
                trainer._last_prev_token.cpu()
                if torch.is_tensor(getattr(trainer, "_last_prev_token", None))
                else getattr(trainer, "_last_prev_token", None)
            ),
        }
        torch.save(ckpt_data, save_path)
        print(f"\nSaved checkpoint: {save_path}")
        return save_path

    _val_iter = None

    def run_validation(step: int, pm_enabled: bool, em_enabled: bool,
                       mode: str) -> dict:
        nonlocal _val_iter
        if _val_iter is None:
            _val_iter = create_dataloader(
                phase=val_data_phase,
                tokenizer=tokenizer,
                batch_size=bs,
                seq_length=config.T,
                seed=VAL_SEED,
                max_steps=VAL_STEPS,
            )
        else:
            _val_iter.dataset.reset_streams()
            _val_iter._it = iter(_val_iter.dataset)
        val_loader = _val_iter
        t0 = time.time()
        metrics = evaluate_validation(
            model=model,
            dataloader=val_loader,
            config=config,
            device=device,
            num_steps=VAL_STEPS,
            pm_enabled=pm_enabled,
            em_enabled=em_enabled,
        )
        elapsed = time.time() - t0
        record = {
            "mode": mode,
            "step": step,
            "val_loss": metrics["loss"],
            "val_ppl": metrics["ppl"],
            "val_valid_tokens": metrics["valid_tokens"],
            "val_steps_done": metrics["steps_done"],
            "val_valid_fraction": metrics["valid_fraction"],
            "val_eot_input_fraction": metrics["eot_input_fraction"],
            "val_reset_fraction": metrics["reset_fraction"],
            "val_elapsed": elapsed,
            "val_pm_enabled": pm_enabled,
            "val_em_enabled": em_enabled,
        }
        collector.log_record(record)
        return record

    last_save_path = None
    save_interval = settings.get("save_interval", SAVE_INTERVAL)
    plot_interval = settings.get("plot_interval", PLOT_INTERVAL)
    val_interval = settings.get("val_interval", VAL_INTERVAL)
    text_sample_interval = settings.get("text_sample_interval", TEXT_SAMPLE_INTERVAL)
    no_plots = settings.get("no_plots", False)

    # Draw a fresh validation batch for text samples each time
    _text_sample_call_count = [0]

    _text_sample_dataset = None

    def _get_text_sample_batch():
        nonlocal _text_sample_dataset
        try:
            if _text_sample_dataset is None:
                _text_sample_dataset = create_dataloader(
                    phase=val_data_phase,
                    tokenizer=tokenizer,
                    batch_size=bs,
                    seq_length=config.T,
                    seed=VAL_SEED + 99,
                    max_steps=1,
                )
            else:
                _text_sample_dataset.dataset.reset_streams()
                _text_sample_dataset._it = iter(_text_sample_dataset.dataset)
            return next(iter(_text_sample_dataset))
        except Exception:
            return None

    def on_step(_step_metrics: dict):
        nonlocal last_save_path
        if save_interval > 0 and trainer.global_step % save_interval == 0:
            last_save_path = save_checkpoint(trainer.global_step)

        val_record = None
        if val_interval > 0 and trainer.global_step % val_interval == 0:
            val_record = run_validation(
                trainer.global_step,
                pm_enabled=config.pm_enabled,
                em_enabled=config.em_enabled,
                mode="val",
            )
            print(
                f"[val] step {trainer.global_step:5d} | "
                f"loss {val_record['val_loss']:.4f} | "
                f"ppl {val_record['val_ppl']:.1f} | "
                f"valid {val_record['val_valid_fraction']:.3f}"
            )

        # Text sample generation
        if text_sample_interval > 0 and trainer.global_step % text_sample_interval == 0:
            sample_batch = _get_text_sample_batch()
            if sample_batch is not None:
                try:
                    from .debug.plot_text_samples import generate_text_sample_plot
                    from .model.state import save_runtime_state as _save_rt, load_runtime_state as _load_rt
                    text_sample_dir = os.path.join(save_dir, "text_samples")
                    os.makedirs(text_sample_dir, exist_ok=True)
                    sample_path = os.path.join(
                        text_sample_dir,
                        f"text_samples_step{trainer.global_step}.png",
                    )
                    current_loss = _step_metrics.get("loss")
                    # Save full runtime state so generation doesn't corrupt training
                    _rt_state = _save_rt(model)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=(device.type == "cuda"),
                    ):
                        generate_text_sample_plot(
                            model=model,
                            tokenizer=tokenizer,
                            batch=sample_batch.input_ids,
                            step=trainer.global_step,
                            loss=current_loss,
                            save_path=sample_path,
                        )
                    # Restore runtime state (not just reset — preserves PM/EM)
                    _load_rt(model, _rt_state)
                except Exception as e:
                    print(f"  Warning: text sample generation failed: {e}")

        # Live plot regeneration
        if not no_plots and plot_interval > 0 and trainer.global_step % plot_interval == 0:
            try:
                from .debug.plot_combined import generate_phase_plots
                generate_phase_plots(metrics_file, phase_name, save_dir)
            except Exception as e:
                print(f"  Warning: live plot generation failed: {e}")

        if ABLATE_INTERVAL > 0 and trainer.global_step % ABLATE_INTERVAL == 0:
            if val_record is None:
                val_record = run_validation(
                    trainer.global_step,
                    pm_enabled=config.pm_enabled,
                    em_enabled=config.em_enabled,
                    mode="val",
                )
            off_record = run_validation(
                trainer.global_step,
                pm_enabled=False,
                em_enabled=False,
                mode="ablate",
            )
            ablate = {
                "mode": "ablate_summary",
                "step": trainer.global_step,
                "ablate_ppl_normal": val_record["val_ppl"],
                "ablate_ppl_plasticity_off": off_record["val_ppl"],
                "ablate_gap": off_record["val_ppl"] - val_record["val_ppl"],
            }
            collector.log_record(ablate)
            print(
                f"[ablate] step {trainer.global_step:5d} | "
                f"normal {ablate['ablate_ppl_normal']:.1f} | "
                f"off {ablate['ablate_ppl_plasticity_off']:.1f} | "
                f"gap {ablate['ablate_gap']:+.2f}"
            )

    metrics = trainer.train_epoch(remaining, step_callback=on_step)

    # Final save
    final_ckpt = None
    if metrics:
        final_ckpt = save_checkpoint(trainer.global_step)
    elif last_save_path:
        final_ckpt = last_save_path

    steps_completed = len(metrics)

    # Phase end marker
    collector.log_record({
        "mode": "phase_end",
        "phase": phase_name,
        "step": trainer.global_step,
        "final_checkpoint": final_ckpt,
    })

    collector.close()

    final = metrics[-1] if metrics else {}
    print(f"\nPhase {phase_name} complete. Final loss: {final.get('loss', 'N/A')}")

    return final_ckpt, steps_completed, trainer.global_step


# ============================================================================
# main — orchestrates single-phase or auto-transition mode
# ============================================================================

def main():
    args = parse_args()
    settings = resolve_settings(args)
    device = _get_device()
    print(f"Device: {device}")

    tokenizer = get_tokenizer(settings["tokenizer"])
    special_ids = get_special_token_ids(tokenizer)
    vocab_size = len(tokenizer)
    eot_id = special_ids.get("eos_token_id", tokenizer.eos_token_id)
    print(f"Tokenizer: {settings['tokenizer']} (vocab={vocab_size}, eot={eot_id})")

    run_id = time.strftime("%Y%m%d_%H%M%S")
    tier = str(settings["tier"]).lower()
    run_paths = _prepare_run_paths(settings, run_id)
    run_dir = run_paths["run_dir"]
    save_dir = run_paths["save_dir"]
    metrics_file = run_paths["metrics_file"]
    metadata_file = run_paths["metadata_file"]
    state_file = run_paths["state_file"]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    settings["save_dir"] = save_dir
    settings["metrics_file"] = metrics_file

    print(f"Run ID: {run_id}")
    print(f"Run Dir: {run_dir}")
    print(f"Checkpoints: {save_dir}")
    print(f"Metrics: {metrics_file}")

    if settings.get("phase_plan") is not None:
        effective_phase_plan = settings["phase_plan"]
    elif settings["phases"]:
        effective_phase_plan = _normalize_phase_plan(settings["phases"]) or []
    else:
        effective_phase_plan = [{"phase": settings["phase"], "steps": None, "tokens": None, "resume": settings["resume"]}]

    run_meta = {
        "run_id": run_id,
        "started_unix": time.time(),
        "git_commit": get_git_commit(),
        "device": str(device),
        "tokenizer": settings["tokenizer"],
        "tier": tier,
        "config_path": settings.get("config_path"),
        "preset": settings.get("preset"),
        "run_name": settings.get("run_name"),
        "paths": run_paths,
        "settings": settings,
        "phase_plan": effective_phase_plan,
        "training_constants": {
            "warmup_steps": settings.get("warmup_steps", WARMUP_STEPS),
            "lr_min": settings.get("lr_min", LR_MIN),
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
            "save_interval": settings.get("save_interval", SAVE_INTERVAL),
            "plot_interval": settings.get("plot_interval", PLOT_INTERVAL),
            "text_sample_interval": settings.get("text_sample_interval", TEXT_SAMPLE_INTERVAL),
            "val_interval": settings.get("val_interval", VAL_INTERVAL),
            "log_interval": settings.get("log_interval", LOG_INTERVAL),
            "val_steps": VAL_STEPS,
            "ablate_interval": ABLATE_INTERVAL,
            "collect_every": COLLECT_EVERY,
        },
    }
    _write_json(metadata_file, run_meta)

    run_state = {
        "run_id": run_id,
        "status": "running",
        "started_unix": run_meta["started_unix"],
        "phase_history": [],
        "global_step_offset": 0,
        "last_checkpoint": None,
    }
    _write_json(state_file, run_state)

    if settings["phases"]:
        # ================================================================
        # Auto-transition mode
        # ================================================================
        phase_plan = effective_phase_plan
        phases = [p["phase"] for p in phase_plan]
        auto_resume = settings["resume"]
        global_step_offset = 0

        print(f"\nAuto-transition mode: {' -> '.join(phases)}")
        print("=" * 60)

        for i, phase_entry in enumerate(phase_plan):
            phase = phase_entry["phase"]
            resume_rule = phase_entry.get("resume", "auto_previous")
            if isinstance(resume_rule, str) and resume_rule.lower() == "auto_previous":
                resume_path = auto_resume
            elif resume_rule is None:
                resume_path = None
            else:
                resume_path = str(resume_rule)

            # Skip phases with 0 default steps (e.g. lifelong disabled),
            # unless explicitly overridden via phase_entry.steps or --steps
            phase_steps = PHASE_DEFAULT_STEPS.get(phase.upper(), None)
            has_explicit_steps = (
                phase_entry.get("steps") is not None
                or settings.get("steps_override") is not None
            )
            if phase_steps is not None and phase_steps <= 0 and not has_explicit_steps:
                print(f"\nSkipping Phase {phase} (0 steps configured)")
                continue

            print(f"\n{'='*60}")
            print(f"Phase {phase} ({i+1}/{len(phases)})")
            print(f"{'='*60}")
            print(f"Resume: {resume_path}")

            ckpt_path, steps_done, phase_end_step = run_phase(
                phase=phase,
                tier=tier,
                resume_path=resume_path,
                run_id=run_id,
                settings=settings,
                metrics_file=metrics_file,
                device=device,
                tokenizer=tokenizer,
                special_ids=special_ids,
                is_auto=True,
                global_step_offset=global_step_offset,
                phase_entry=phase_entry,
            )
            global_step_offset = phase_end_step
            run_state["global_step_offset"] = global_step_offset
            run_state["last_checkpoint"] = ckpt_path
            run_state["phase_history"].append({
                "phase": phase,
                "index": i,
                "resume_rule": resume_rule,
                "resume_used": resume_path,
                "steps_done": steps_done,
                "checkpoint": ckpt_path,
                "ended_unix": time.time(),
            })
            _write_json(state_file, run_state)

            if not settings["no_plots"]:
                try:
                    from .debug.plot_combined import generate_phase_plots
                    generate_phase_plots(metrics_file, phase, save_dir)
                except Exception as e:
                    print(f"Warning: plot generation failed for phase {phase}: {e}")

            auto_resume = ckpt_path  # chain to next phase when resume=auto_previous
            if auto_resume is None and i < len(phases) - 1:
                print(
                    f"\nERROR: Phase {phase} produced no checkpoint. "
                    "Cannot chain to next phase. Aborting auto-transition."
                )
                run_state["status"] = "aborted_no_checkpoint"
                _write_json(state_file, run_state)
                break

        if not settings["no_plots"] and len(phases) > 1:
            try:
                from .debug.plot_combined import generate_combined_plot
                generate_combined_plot(metrics_file, phases, save_dir)
            except Exception as e:
                print(f"Warning: combined plot generation failed: {e}")

        print(f"\n{'='*60}")
        print(f"All phases complete. Total steps: {global_step_offset}")
        print(f"{'='*60}")
        if run_state.get("status") == "running":
            run_state["status"] = "completed"
        run_state["ended_unix"] = time.time()
        _write_json(state_file, run_state)
        run_meta["ended_unix"] = run_state["ended_unix"]
        run_meta["status"] = run_state["status"]
        run_meta["last_checkpoint"] = run_state["last_checkpoint"]
        run_meta["global_step_offset"] = run_state["global_step_offset"]
        _write_json(metadata_file, run_meta)

    else:
        # ================================================================
        # Single-phase mode (backward compatible)
        # ================================================================
        phase = settings["phase"]
        print(f"\nSingle-phase mode: {phase}")
        phase_entry = effective_phase_plan[0] if effective_phase_plan else None

        ckpt_path, steps_done, _phase_end_step = run_phase(
            phase=phase,
            tier=tier,
            resume_path=settings["resume"],
            run_id=run_id,
            settings=settings,
            metrics_file=metrics_file,
            device=device,
            tokenizer=tokenizer,
            special_ids=special_ids,
            is_auto=False,
            global_step_offset=0,
            phase_entry=phase_entry,
        )
        run_state["global_step_offset"] = steps_done
        run_state["last_checkpoint"] = ckpt_path
        run_state["phase_history"].append({
            "phase": phase,
            "index": 0,
            "resume_rule": settings["resume"],
            "resume_used": settings["resume"],
            "steps_done": steps_done,
            "checkpoint": ckpt_path,
            "ended_unix": time.time(),
        })
        run_state["status"] = "completed"
        run_state["ended_unix"] = time.time()
        _write_json(state_file, run_state)
        run_meta["ended_unix"] = run_state["ended_unix"]
        run_meta["status"] = run_state["status"]
        run_meta["last_checkpoint"] = run_state["last_checkpoint"]
        run_meta["global_step_offset"] = run_state["global_step_offset"]
        _write_json(metadata_file, run_meta)


if __name__ == "__main__":
    main()
