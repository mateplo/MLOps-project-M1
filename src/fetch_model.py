"""Download the published model from the Hugging Face Hub into artifacts/.

Used by the serving image entrypoint (e.g. on a HF Space, where nothing is mounted).
Deliberately free of heavy imports: only huggingface_hub.

Env:
    MODEL_REPO      e.g. mateplo/adult-income-classifier   (required to do anything)
    MODEL_REVISION  branch or tag of that repo, default "main"
    MODEL_PATH      where to put model.joblib, default artifacts/model.joblib
    HF_TOKEN        only needed for a private model repo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

MODEL_FILES = ("model.joblib", "model_meta.json")
log = logging.getLogger("fetch_model")


def fetch(repo: str, revision: str, dest_dir: Path, force: bool = False) -> dict:
    """Download model files into dest_dir. Returns the model metadata."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    meta_path = dest_dir / "model_meta.json"
    if not force and (dest_dir / "model.joblib").exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        log.info("model already present (v%s), skipping download", meta.get("version"))
        return meta
    for name in MODEL_FILES:
        cached = hf_hub_download(repo_id=repo, filename=name, revision=revision, repo_type="model")
        shutil.copyfile(cached, dest_dir / name)
    meta = json.loads(meta_path.read_text())
    log.info("downloaded %s@%s -> %s (registry v%s)", repo, revision, dest_dir, meta.get("version"))
    return meta


def main() -> None:
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Fetch the published model from the HF Hub")
    parser.add_argument("--repo", default=os.getenv("MODEL_REPO"))
    parser.add_argument("--revision", default=os.getenv("MODEL_REVISION", "main"))
    parser.add_argument(
        "--dest", default=str(Path(os.getenv("MODEL_PATH", "artifacts/model.joblib")).parent)
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.repo:
        log.info("MODEL_REPO not set: nothing to fetch (using mounted artifacts/)")
        return
    fetch(args.repo, args.revision, Path(args.dest), args.force)


if __name__ == "__main__":
    main()
