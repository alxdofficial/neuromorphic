"""Download baseline model weights from HuggingFace."""

import os
import sys

from huggingface_hub import snapshot_download

MODELS = {
    "pythia-160m": "EleutherAI/pythia-160m",
    "mamba-130m": "state-spaces/mamba-130m-hf",
}

CACHE_DIR = os.path.join(os.path.dirname(__file__), "..", "model_cache")


def download_all():
    os.makedirs(CACHE_DIR, exist_ok=True)
    for name, repo_id in MODELS.items():
        print(f"Downloading {name} ({repo_id})...")
        try:
            path = snapshot_download(
                repo_id,
                cache_dir=CACHE_DIR,
                local_dir=os.path.join(CACHE_DIR, name),
            )
            print(f"  -> {path}")
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
    print("Done.")


if __name__ == "__main__":
    download_all()
