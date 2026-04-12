"""
Dataset configurations for neuromorphic LM training.

Datasets are selected based on modern small LM training practices:
- SmolLM2, TinyLlama, Phi, StableLM use similar data strategies
- Quality filtering > raw quantity
- Mix of web, synthetic, code, and math data

Key datasets:
- FineWeb-Edu: Educational web text (SmolLM2, Phi)
- DCLM: Diverse conversational web text (SmolLM2)
- Cosmopedia: Synthetic textbooks/stories (SmolLM)
- StarCoder: Code in 86 languages (TinyLlama, SmolLM2)
- FineMath: Mathematical reasoning (SmolLM2)
"""

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""

    name: str                           # Human-readable name
    hf_path: str                        # HuggingFace dataset path
    hf_name: Optional[str] = None       # HuggingFace config name (subset)
    split: str = "train"                # Dataset split
    text_column: str = "text"           # Column containing text data
    streaming: bool = True              # Whether to stream (vs full download)
    download_first: bool = False        # Download before training starts
    estimated_tokens: Optional[int] = None  # Approximate token count
    estimated_disk_gb: Optional[float] = None  # Disk space if downloaded
    description: str = ""               # Brief description


# =============================================================================
# Phase A: Sanity Check
# =============================================================================

TINYSTORIES = DatasetConfig(
    name="TinyStories",
    hf_path="roneneldan/TinyStories",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,  # Small enough to download
    download_first=True,
    estimated_tokens=470_000_000,  # ~470M tokens
    estimated_disk_gb=1.5,
    description="Synthetic short stories for children. Simple vocabulary, clear narrative. "
                "Perfect for verifying basic LM training works.",
)

# =============================================================================
# The Pile (primary training data — matches Pythia/Mamba/RWKV baselines)
# =============================================================================

THE_PILE = DatasetConfig(
    name="The Pile (deduplicated)",
    hf_path="EleutherAI/the_pile_deduplicated",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=300_000_000_000,  # ~300B tokens (full dataset)
    estimated_disk_gb=800.0,
    description="Diverse English text corpus. Same data as Pythia, Mamba, RWKV-7 baselines. "
                "Use scripts/prepare_data.py to download a local subset.",
)

_PILE_DATA_DIR = "data/pile"

PILE_LOCAL = DatasetConfig(
    name="The Pile (local)",
    hf_path=f"{_PILE_DATA_DIR}/pile_train.parquet",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=False,
    description="Local pre-downloaded subset of The Pile. "
                "Run scripts/prepare_data.py first.",
)

PILE_VAL_LOCAL = DatasetConfig(
    name="The Pile validation (local)",
    hf_path=f"{_PILE_DATA_DIR}/pile_val.parquet",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=False,
    description="Local held-out Pile validation set.",
)

# =============================================================================
# Phase B-C: Main Training (Modern SLM Datasets)
# =============================================================================

FINEWEB_EDU = DatasetConfig(
    name="FineWeb-Edu",
    hf_path="HuggingFaceFW/fineweb-edu",
    hf_name="sample-10BT",  # 10B token sample
    split="train",
    text_column="text",
    streaming=True,
    download_first=True,
    estimated_tokens=10_000_000_000,  # ~10B tokens
    estimated_disk_gb=27.0,
    description="Educational web text filtered by Llama3-70B classifier. "
                "Primary dataset for SmolLM2 and Phi models.",
)

DCLM = DatasetConfig(
    name="DCLM-Baseline",
    hf_path="mlfoundations/dclm-baseline-1.0",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=3_800_000_000_000,  # 3.8T tokens total (stream subset)
    estimated_disk_gb=5000.0,
    description="DataComp for Language Models. Diverse conversational web text. "
                "Excels on commonsense reasoning. Used by SmolLM2.",
)

# =============================================================================
# Local (pre-downloaded) variants — see scripts/prepare_data.py
# =============================================================================

_LOCAL_DATA_DIR = "data/phase_B"

FINEWEB_EDU_LOCAL = DatasetConfig(
    name="FineWeb-Edu (local)",
    hf_path=f"{_LOCAL_DATA_DIR}/fineweb_edu.parquet",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=False,
    estimated_tokens=1_250_000_000,
    estimated_disk_gb=3.0,
    description="Local subset of FineWeb-Edu sample-10BT (~1.25B tokens).",
)

DCLM_LOCAL = DatasetConfig(
    name="DCLM (local)",
    hf_path=f"{_LOCAL_DATA_DIR}/dclm.parquet",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=False,
    estimated_tokens=830_000_000,
    estimated_disk_gb=2.0,
    description="Local subset of DCLM (~830M tokens).",
)

VAL_FINEWEB_EDU_LOCAL = DatasetConfig(
    name="FineWeb-Edu validation (local)",
    hf_path=f"{_LOCAL_DATA_DIR}/val_fineweb_edu.parquet",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=False,
    estimated_tokens=5_000_000,
    estimated_disk_gb=0.02,
    description="Local held-out validation from FineWeb-Edu (seed=1337, ~5M tokens).",
)

COSMOPEDIA = DatasetConfig(
    name="Cosmopedia-v2",
    hf_path="HuggingFaceTB/cosmopedia",
    hf_name="web_samples_v2",  # Largest subset
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=25_000_000_000,  # ~25B tokens
    estimated_disk_gb=80.0,
    description="Synthetic textbooks, blogs, stories from Mixtral-8x7B. "
                "High-quality synthetic data used by SmolLM.",
)

SLIMPAJAMA = DatasetConfig(
    name="SlimPajama-6B",
    hf_path="DKYoon/SlimPajama-6B",  # 6B token sample (full 627B requires access)
    hf_name=None,
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=6_000_000_000,  # ~6B tokens
    estimated_disk_gb=15.0,
    description="6B sample of SlimPajama. Web, books, code, Wikipedia, StackExchange. "
                "Used by TinyLlama and StableLM.",
)

PG19 = DatasetConfig(
    name="PG19 (Deprecated)",
    hf_path="deepmind/pg19",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=False,
    download_first=True,
    estimated_tokens=1_900_000_000,  # ~1.9B tokens across 28K books
    estimated_disk_gb=12.0,
    description="NOTE: Dataset uses deprecated loading script. "
                "Use Wikipedia for long-context evaluation instead.",
)

# =============================================================================
# Code Training (NOTE: Most code datasets are gated - request access on HF)
# =============================================================================

STARCODER = DatasetConfig(
    name="StarCoder (Gated)",
    hf_path="bigcode/starcoderdata",
    hf_name=None,
    split="train",
    text_column="content",
    streaming=True,
    download_first=False,
    estimated_tokens=250_000_000_000,  # ~250B tokens
    estimated_disk_gb=500.0,
    description="REQUIRES ACCESS: Request at huggingface.co/datasets/bigcode/starcoderdata. "
                "Code in 86 languages. Used by TinyLlama and SmolLM2.",
)

# =============================================================================
# Math Training
# =============================================================================

FINEMATH = DatasetConfig(
    name="FineMath",
    hf_path="HuggingFaceTB/finemath",
    hf_name="finemath-4plus",  # Highest quality (score >= 4)
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=34_000_000_000,  # ~34B tokens
    estimated_disk_gb=100.0,
    description="High-quality mathematical text. Step-by-step reasoning. "
                "Created for SmolLM2 to address math capability gaps.",
)

OPENWEBMATH = DatasetConfig(
    name="OpenWebMath",
    hf_path="open-web-math/open-web-math",
    hf_name=None,
    split="train",
    text_column="text",
    streaming=True,
    download_first=False,
    estimated_tokens=14_700_000_000,  # ~14.7B tokens
    estimated_disk_gb=50.0,
    description="High-quality mathematical web text. 14.7B tokens. "
                "Alternative to ProofPile-2 for math training.",
)

# =============================================================================
# Lifelong Learning & Evaluation
# =============================================================================

WIKIPEDIA = DatasetConfig(
    name="Wikipedia",
    hf_path="wikimedia/wikipedia",
    hf_name="20231101.en",
    split="train",
    text_column="text",
    streaming=False,
    download_first=True,
    estimated_tokens=4_000_000_000,  # ~4B tokens
    estimated_disk_gb=25.0,
    description="English Wikipedia for domain adaptation evaluation.",
)

# =============================================================================
# Instruction Tuning / Agentic (Future)
# =============================================================================

SMOLTALK = DatasetConfig(
    name="SmolTalk",
    hf_path="HuggingFaceTB/smoltalk",
    hf_name=None,
    split="train",
    text_column="messages",  # Conversation format
    streaming=False,
    download_first=True,
    estimated_tokens=100_000_000,
    estimated_disk_gb=2.0,
    description="Instruction-tuning dataset from SmolLM2. Magpie-Ultra + curated subsets.",
)

OPENHERMES = DatasetConfig(
    name="OpenHermes-2.5",
    hf_path="teknium/OpenHermes-2.5",
    hf_name=None,
    split="train",
    text_column="conversations",  # Needs preprocessing
    streaming=False,
    download_first=True,
    estimated_tokens=500_000_000,  # ~1M examples
    estimated_disk_gb=15.0,
    description="Diverse instruction-following. Used by many open models.",
)


# =============================================================================
# Dataset Registry
# =============================================================================

DATASET_CONFIGS = {
    # The Pile (primary training data)
    "pile": THE_PILE,
    "pile-local": PILE_LOCAL,
    "val-pile-local": PILE_VAL_LOCAL,

    # Sanity check
    "tinystories": TINYSTORIES,

    # Legacy: FineWeb-Edu + DCLM (modern SLM datasets)
    "fineweb-edu": FINEWEB_EDU,
    "dclm": DCLM,
    "cosmopedia": COSMOPEDIA,
    "slimpajama": SLIMPAJAMA,

    # Local (pre-downloaded) variants
    "fineweb-edu-local": FINEWEB_EDU_LOCAL,
    "dclm-local": DCLM_LOCAL,
    "val-fineweb-edu-local": VAL_FINEWEB_EDU_LOCAL,

    # Code training (gated - requires HF access)
    "starcoder": STARCODER,

    # Math training
    "finemath": FINEMATH,
    "openwebmath": OPENWEBMATH,

    # Long-context / Lifelong learning
    "wikipedia": WIKIPEDIA,
    "pg19": PG19,  # Deprecated - use wikipedia

    # Instruction tuning
    "smoltalk": SMOLTALK,
    "openhermes": OPENHERMES,
}


# =============================================================================
# Training Phase Configurations
# =============================================================================

@dataclass
class PhaseConfig:
    """Configuration for a training phase."""
    name: str
    datasets: List[str]                 # Dataset keys from DATASET_CONFIGS
    mix_weights: Optional[List[float]] = None  # Mixing weights (must sum to 1)
    description: str = ""


# Phase configs modeled after SmolLM2 training strategy:
# - Stage 1: FineWeb-Edu (60%) + DCLM (40%) web mix
# - Stage 2: Add code and math
# - Stage 3: Increase code/math, add Cosmopedia synthetic data

PHASE_CONFIGS = {
    "A": PhaseConfig(
        name="Phase A: The Pile (Local)",
        datasets=["pile-local"],
        description="Training on the local pre-tokenized Pile subset. "
                    "Run scripts/prepare_data.py first.",
    ),
    "A-val": PhaseConfig(
        name="Phase A validation (Local)",
        datasets=["val-pile-local"],
        description="Held-out validation subset of The Pile. Use for eval loaders "
                    "so metrics aren't in-sample.",
    ),
    "B": PhaseConfig(
        name="Phase B: Lifelong Learning (Local Pile)",
        datasets=["pile-local"],
        description="Phase A + lifelong mode (memory state persists across "
                    "document boundaries). Same data as Phase A.",
    ),
    "B-legacy": PhaseConfig(
        name="Phase B Legacy: FineWeb-Edu + DCLM (Local)",
        datasets=["fineweb-edu-local", "dclm-local"],
        mix_weights=[0.6, 0.4],
        description="Educational + conversational web text from local parquet files.",
    ),
    "B-streaming": PhaseConfig(
        name="Phase B: Base Language (Streaming)",
        datasets=["fineweb-edu", "dclm"],
        mix_weights=[0.6, 0.4],
        description="Educational + conversational web text. Streaming from HF hub.",
    ),
    "B-diverse": PhaseConfig(
        name="Phase B: Diverse Sources",
        datasets=["fineweb-edu", "slimpajama"],
        mix_weights=[0.7, 0.3],
        description="FineWeb-Edu + SlimPajama. TinyLlama-style training.",
    ),
    "C": PhaseConfig(
        name="Phase C: With Math",
        datasets=["fineweb-edu", "dclm", "finemath"],
        mix_weights=[0.5, 0.35, 0.15],
        description="Add mathematical reasoning to the training mix.",
    ),
    "C-synthetic": PhaseConfig(
        name="Phase C: Synthetic Data (Phi-style)",
        datasets=["fineweb-edu", "cosmopedia", "finemath"],
        mix_weights=[0.5, 0.35, 0.15],
        description="Synthetic textbooks from Cosmopedia mixed with web text.",
    ),
    "D": PhaseConfig(
        name="Phase D: Math Focus",
        datasets=["finemath", "openwebmath"],
        mix_weights=[0.5, 0.5],
        description="Mathematical reasoning focus (proofs, definitions).",
    ),
    "E": PhaseConfig(
        name="Phase E: Lifelong Learning",
        datasets=["fineweb-edu", "wikipedia"],
        mix_weights=[0.7, 0.3],
        description="Memory state persists across document boundaries.",
    ),
    "longctx": PhaseConfig(
        name="Long Context Evaluation",
        datasets=["wikipedia"],
        description="Wikipedia articles for long-context evaluation.",
    ),
}


def get_phase_datasets(phase: str) -> List[DatasetConfig]:
    """Get dataset configs for a training phase."""
    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(PHASE_CONFIGS.keys())}")

    phase_cfg = PHASE_CONFIGS[phase]
    return [DATASET_CONFIGS[name] for name in phase_cfg.datasets]
