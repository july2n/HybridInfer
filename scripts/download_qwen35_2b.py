"""Download the official checkpoint without importing the inference engine."""

import argparse
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="models/Qwen3.5-0.8B")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--endpoint", help="Optional Hugging Face compatible endpoint")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint
    try:
        from huggingface_hub import HfApi, snapshot_download
    except ImportError:
        parser.exit(1, "Install the downloader first: python3 -m pip install -U huggingface_hub\n")

    repo = "Qwen/Qwen3.5-0.8B"
    # Resolve once so every file comes from the same commit.
    revision = HfApi().model_info(repo, revision=args.revision).sha
    output = Path(args.output).expanduser().resolve()
    print(f"Downloading {repo}@{revision} to {output}", flush=True)
    snapshot_download(
        repo_id=repo,
        revision=revision,
        local_dir=str(output),
        allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja", "README.md"],
        max_workers=args.workers,
    )
    if not (output / "config.json").is_file():
        raise RuntimeError("Missing config.json")
    index = output / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
    else:
        shards = {"model.safetensors"}
    missing = [name for name in shards if not (output / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing weight files: {missing}")
    print(f"Download complete: {output}\nRevision: {revision}")
    print("Run: python3 scripts/try_qwen35_2b.py --model " + str(output))


if __name__ == "__main__":
    main()
