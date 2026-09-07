"""Export a registry model (default: the champion) to artifacts/ for the lightweight API image.

Writes artifacts/model.joblib and artifacts/model_meta.json (version, run_id, threshold, ...).

Usage:
    python -m src.export --config configs/config.yaml [--alias champion | --version N]
"""

from __future__ import annotations

import argparse
import time

import joblib
import mlflow
import mlflow.sklearn

from src.utils import get_logger, load_config, meta_path, resolve, setup_mlflow, write_meta

log = get_logger(__name__)


def export(cfg: dict, alias: str | None = None, version: str | None = None) -> dict:
    setup_mlflow(cfg)
    ml_cfg = cfg["mlflow"]
    name = ml_cfg["registered_model_name"]
    alias = None if version else (alias or ml_cfg.get("champion_alias", "champion"))
    client = mlflow.MlflowClient()

    mv = (
        client.get_model_version(name, version)
        if version
        else client.get_model_version_by_alias(name, alias)
    )
    run = client.get_run(mv.run_id)
    model = mlflow.sklearn.load_model(f"models:/{name}/{mv.version}")

    model_path = resolve(cfg["artifacts"]["model_path"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    meta = {
        "model_name": name,
        "version": str(mv.version),
        "alias": alias,
        "run_id": mv.run_id,
        "experiment": run.info.experiment_id,
        "model_uri": f"models:/{name}/{mv.version}",
        "model_type": mv.tags.get("model_type"),
        "decision_threshold": float(mv.tags.get("decision_threshold", 0.5)),
        "metrics": {k: v for k, v in run.data.metrics.items() if k.startswith("test_")},
        "data_sha256": mv.tags.get("data_sha256"),
        "trained_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.localtime(run.info.start_time / 1000)
        ),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    write_meta(meta_path(cfg), meta)
    log.info("exported %s v%s (%s) -> %s", name, mv.version, alias or "by version", model_path)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a registry model to artifacts/")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--alias", default=None)
    parser.add_argument("--version", default=None)
    args = parser.parse_args()
    export(load_config(args.config), args.alias, args.version)


if __name__ == "__main__":
    main()
