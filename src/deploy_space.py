"""(Re)deploy the Hugging Face *static* Space that runs the model in the browser.

The Space is fully generated from deploy/space/ (README with Space metadata, index.html, app.js)
plus a config.json telling the page which HF model repo / revision to load the ONNX model from.
Run by the `deploy-space` CI job on every `v*` tag; can also be run by hand.

Usage:
    HF_TOKEN=... python -m src.deploy_space [--config configs/config.yaml] [--space owner/name]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = PROJECT_ROOT / "deploy" / "space"
logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("deploy_space")


def load_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_env_file(path: Path) -> dict[str, str]:
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def render_config(model_repo: str, revision: str, image: str | None = None) -> dict:
    """config.json consumed by app.js."""
    cfg = {"model_repo": model_repo, "model_revision": revision}
    if image:
        cfg["api_image"] = image
    return cfg


def build_site(dest: Path, model_repo: str, revision: str, image: str | None = None) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("README.md", "index.html", "app.js"):
        shutil.copyfile(SPACE_DIR / name, dest / name)
        files.append(name)
    (dest / "config.json").write_text(
        json.dumps(render_config(model_repo, revision, image), indent=2)
    )
    files.append("config.json")
    return files


def deploy(
    cfg: dict, image: str | None = None, space: str | None = None, token: str | None = None
) -> str:
    from huggingface_hub import HfApi

    token = token or os.getenv("HF_TOKEN")
    if not token:
        raise SystemExit("HF_TOKEN is not set (needs a write token).")
    space = space or os.getenv("HF_SPACE_REPO") or cfg["hf"]["space_repo"]
    model_repo = os.getenv("HF_MODEL_REPO") or cfg["hf"]["model_repo"]
    revision = read_env_file(PROJECT_ROOT / "deploy" / "space.env").get("MODEL_REVISION", "main")

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        files = build_site(Path(tmp), model_repo, revision, image)
        api.upload_folder(
            repo_id=space,
            repo_type="space",
            folder_path=tmp,
            commit_message=f"Deploy static site (model {model_repo}@{revision}"
            + (f", api image {image}" if image else "")
            + ")",
        )
    url = f"https://huggingface.co/spaces/{space}"
    log.info("deployed %s -> %s", files, url)
    return url


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy the static demo to a HF Space")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--image", default=None, help="API image reference, recorded in config.json"
    )
    parser.add_argument(
        "--space", default=None, help="owner/name (default: hf.space_repo in config)"
    )
    args = parser.parse_args()
    deploy(load_config(args.config), args.image, args.space)


if __name__ == "__main__":
    main()
