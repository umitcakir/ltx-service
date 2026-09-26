"""Download only the Diffusers components needed by the service.

Usage: python -m scripts.download_models --config config.yaml
The Hugging Face model is gated; accept its license in a browser and run
``hf auth login`` before downloading.
"""

import argparse
from pathlib import Path


def download_base_model(repo_id: str, checkpoint: Path) -> Path:
    if (checkpoint / "model_index.json").is_file():
        return checkpoint
    from huggingface_hub import snapshot_download

    checkpoint.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(checkpoint),
        ignore_patterns=["transformer_full/*", "diffusion_decoder/*", "*.md", "*.png", "*.webp"],
    )
    if not (checkpoint / "model_index.json").is_file():
        raise RuntimeError(f"Download incomplete: {checkpoint / 'model_index.json'} missing")
    return checkpoint


def main() -> None:
    from app.config import load_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    print(download_base_model(config.model.repo_id, config.model.checkpoint_path))


if __name__ == "__main__":
    main()