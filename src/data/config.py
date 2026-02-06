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
    # Phase A: Sanity check
    "tinystories": TINYSTORIES,

    # Phase B-C: Main training (modern SLM datasets)
    "fineweb-edu": FINEWEB_EDU,
    "dclm": DCLM,
    "cosmopedia": COSMOPEDIA,
    "slimpajama": SLIMPAJAMA,

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
        name="Phase A: Sanity Check",
        datasets=["tinystories"],
        description="Verify backbone + WM learns language. PM/EM disabled.",
    ),
    "B": PhaseConfig(
        name="Phase B: Base Language (SmolLM2-style)",
        datasets=["fineweb-edu", "dclm"],
        mix_weights=[0.6, 0.4],
        description="Educational + conversational web text. PM with heuristic commits.",
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
        description="Add mathematical reasoning. PM + EM with heuristics.",
    ),
    "C-synthetic": PhaseConfig(
        name="Phase C: Synthetic Data (Phi-style)",
        datasets=["fineweb-edu", "cosmopedia", "finemath"],
        mix_weights=[0.5, 0.35, 0.15],
        description="Synthetic textbooks from Cosmopedia. Tests if synthetic helps PM/EM.",
    ),
    "D": PhaseConfig(
        name="Phase D: Math Focus",
        datasets=["finemath", "openwebmath"],
        mix_weights=[0.5, 0.5],
        description="Mathematical reasoning focus. Tests PM/EM on proofs/definitions.",
    ),
    "longctx": PhaseConfig(
        name="Long Context Evaluation",
        datasets=["wikipedia"],
        description="Wikipedia articles for long-context PM/EM evaluation.",
    ),
}


def get_phase_datasets(phase: str) -> List[DatasetConfig]:
    """Get dataset configs for a training phase."""
    if phase not in PHASE_CONFIGS:
        raise ValueError(f"Unknown phase: {phase}. Available: {list(PHASE_CONFIGS.keys())}")

    phase_cfg = PHASE_CONFIGS[phase]
    return [DATASET_CONFIGS[name] for name in phase_cfg.datasets]
