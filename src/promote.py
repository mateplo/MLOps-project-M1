"""Champion / challenger promotion on the MLflow Model Registry.

The candidate version (by default the one written by the last `train` in model_meta.json)
is compared with the current `champion` on the promotion metric (default: test_roc_auc,
computed on the same held-out split). It becomes champion only if it is strictly better
by at least `min_improvement`; otherwise it stays `challenger`.

Usage:
    python -m src.promote --config configs/config.yaml [--version N] [--force]
"""

from __future__ import annotations

import argparse
import time

import mlflow

from src.utils import get_logger, load_config, meta_path, read_meta, setup_mlflow, write_meta

log = get_logger(__name__)


def _version_score(client: mlflow.MlflowClient, mv, metric: str) -> float | None:
    run = client.get_run(mv.run_id)
    return run.data.metrics.get(metric)


def promote(cfg: dict, version: str | None = None, force: bool = False) -> dict:
    setup_mlflow(cfg)
    ml_cfg = cfg["mlflow"]
    name = ml_cfg["registered_model_name"]
    champion_alias = ml_cfg.get("champion_alias", "champion")
    challenger_alias = ml_cfg.get("challenger_alias", "challenger")
    promo = ml_cfg.get("promotion", {})
    metric = promo.get("metric", "test_roc_auc")
    min_improvement = float(promo.get("min_improvement", 0.0))

    client = mlflow.MlflowClient()
    meta = read_meta(meta_path(cfg))
    version = str(version or meta.get("version") or "")
    if not version:
        raise SystemExit("No candidate version: pass --version or run `make train` first.")
    candidate = client.get_model_version(name, version)
    cand_score = _version_score(client, candidate, metric)
    if cand_score is None:
        raise SystemExit(f"Candidate v{version} has no metric {metric!r}.")

    try:
        champion = client.get_model_version_by_alias(name, champion_alias)
    except mlflow.exceptions.MlflowException:
        champion = None
    champ_score = _version_score(client, champion, metric) if champion else None

    if champion is not None and champion.version == candidate.version:
        decision, reason = False, "candidate is already the champion"
    elif champion is None:
        decision, reason = True, "no champion yet"
    elif force:
        decision, reason = True, "forced"
    elif cand_score > champ_score + min_improvement:
        decision, reason = (
            True,
            f"{metric} {cand_score:.4f} > {champ_score:.4f} + {min_improvement}",
        )
    else:
        decision, reason = (
            False,
            f"{metric} {cand_score:.4f} <= {champ_score:.4f} + {min_improvement}",
        )

    result = {
        "model": name,
        "candidate_version": candidate.version,
        "candidate_score": cand_score,
        "champion_version": champion.version if champion else None,
        "champion_score": champ_score,
        "metric": metric,
        "promoted": decision,
        "reason": reason,
    }

    if decision:
        client.set_registered_model_alias(name, champion_alias, candidate.version)
        client.set_model_version_tag(
            name, candidate.version, "promoted_at", time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        client.set_model_version_tag(name, candidate.version, "promotion_reason", reason)
        if champion is not None:
            client.set_model_version_tag(
                name, champion.version, "demoted_at", time.strftime("%Y-%m-%dT%H:%M:%S")
            )
        try:  # legacy stage, deprecated
            client.transition_model_version_stage(name, candidate.version, stage="Production")
        except Exception as exc:  # pragma: no cover
            log.warning("could not set legacy stage: %s", exc)
        if str(meta.get("version")) == str(candidate.version):
            meta["alias"] = champion_alias
            write_meta(meta_path(cfg), meta)
        log.info("PROMOTED v%s to %s (%s)", candidate.version, champion_alias, reason)
    else:
        client.set_model_version_tag(
            name, candidate.version, "promotion_rejected_at", time.strftime("%Y-%m-%dT%H:%M:%S")
        )
        client.set_model_version_tag(name, candidate.version, "promotion_reason", reason)
        log.info("NOT promoted: v%s stays %s (%s)", candidate.version, challenger_alias, reason)

    # Trace the decision as a small MLflow run so it shows up in the experiment timeline.
    with mlflow.start_run(run_name=f"promote-v{candidate.version}"):
        mlflow.set_tags({"stage": "promote", "promoted": str(decision), "reason": reason})
        mlflow.log_params({k: v for k, v in result.items() if k not in ("reason",)})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Promote a model version to champion if better")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--version", default=None, help="candidate version (default: last trained)")
    parser.add_argument("--force", action="store_true", help="promote regardless of the metric")
    args = parser.parse_args()
    result = promote(load_config(args.config), args.version, args.force)
    for k, v in result.items():
        log.info("%-18s %s", k, v)


if __name__ == "__main__":
    main()
