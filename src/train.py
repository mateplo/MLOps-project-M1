"""Train + tune the pipeline with GridSearchCV, track with MLflow, register a *challenger*.

Steps
  1. load + validate data, log its fingerprint (lineage)
  2. GridSearchCV (stratified k-fold) on the training split, MLflow autolog
  3. out-of-fold probabilities -> F1-optimal decision threshold
  4. held-out test metrics at that threshold, feature importance plot
  5. log model (signature + metadata), register as `challenger` (promotion is a separate step)
  6. write artifacts/model.joblib + artifacts/model_meta.json for serving

Usage:
    python -m src.train --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.data
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from src.metrics import best_threshold, compute_metrics
from src.pipeline import (
    build_pipeline_from_config,
    get_feature_importance,
    get_feature_names,
    param_prefix,
)
from src.utils import (
    file_fingerprint,
    get_logger,
    load_config,
    load_dataframe,
    meta_path,
    plot_feature_importance,
    resolve,
    setup_mlflow,
    split_data,
    split_features,
    write_meta,
)

log = get_logger(__name__)


def make_param_grid(params: dict, prefix: str = "model__") -> dict:
    """Prefix the config grid with the pipeline step name (`model__C`, ...)."""
    return {f"{prefix}{k}": (v if isinstance(v, list) else [v]) for k, v in params.items()}


def register_challenger(
    model_uri: str, cfg: dict, tags: dict
) -> mlflow.entities.model_registry.ModelVersion | None:
    """Register the logged model as a new version tagged as the current challenger."""
    ml_cfg = cfg.get("mlflow", {})
    if not ml_cfg.get("register", False):
        return None
    name = ml_cfg["registered_model_name"]
    mv = mlflow.register_model(model_uri=model_uri, name=name)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(
        name, ml_cfg.get("challenger_alias", "challenger"), mv.version
    )
    for k, v in tags.items():
        client.set_model_version_tag(name, mv.version, k, str(v))
    # Stages are deprecated since MLflow 2.9 but the course asks for "Staging".
    try:
        client.transition_model_version_stage(name, mv.version, stage="Staging")
    except Exception as exc:  # pragma: no cover
        log.warning("could not set legacy stage: %s", exc)
    log.info("registered %s version %s as challenger", name, mv.version)
    return mv


def train(cfg: dict, cfg_path: Path = Path("configs/config.yaml")) -> str:
    experiment = setup_mlflow(cfg)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False, silent=True)

    # ---- Data + lineage ---------------------------------------------------
    df = load_dataframe(cfg, validate=True)
    X, y = split_features(df, cfg)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)
    fingerprint = file_fingerprint(cfg["data"]["csv_path"])

    # ---- Model + search space -----------------------------------------------
    model_type = cfg["model"]["type"]
    pipe = build_pipeline_from_config(cfg)
    cv_cfg = cfg["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=True, random_state=cfg["data"]["random_state"]
    )
    grid = GridSearchCV(
        pipe,
        param_grid=make_param_grid(cfg["model"].get("params", {}), param_prefix(cfg)),
        cv=cv,
        scoring=cv_cfg["scoring"],
        n_jobs=cv_cfg.get("n_jobs", -1),
        refit=True,
        return_train_score=True,
    )

    run_name = f"{model_type}-{time.strftime('%Y%m%d-%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags(
            {
                "model_type": model_type,
                "dataset": "uci-adult-income",
                "stage": "train",
                "config": str(cfg_path),
                "calibrated": bool(cfg["model"].get("calibrate")),
            }
        )
        dataset = mlflow.data.from_pandas(
            df,
            source=fingerprint["data_path"],
            name="adult-income-raw",
            targets=cfg["data"]["target"],
        )
        mlflow.log_input(dataset, context="training")
        mlflow.log_params(
            {
                **fingerprint,
                "data_rows": len(df),
                "data_positive_rate": round(float(y.mean()), 4),
                "n_train": len(X_train),
                "n_test": len(X_test),
                "features_numeric": ",".join(cfg["features"]["numeric"]),
                "features_categorical": ",".join(cfg["features"]["categorical"]),
                "features_log1p": ",".join(cfg["features"].get("log1p", []) or []),
                "ohe_min_frequency": cfg["features"].get("min_frequency"),
                "cv_strategy": cv_cfg["strategy"],
                "cv_n_splits": cv_cfg["n_splits"],
                "cv_scoring": cv_cfg["scoring"],
            }
        )
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        # ---- Tuning ------------------------------------------------------------
        log.info("grid search: %s", grid.param_grid)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        log.info(
            "best params %s (cv %s=%.4f)", grid.best_params_, cv_cfg["scoring"], grid.best_score_
        )

        # ---- Decision threshold from out-of-fold predictions ---------------------
        oof = cross_val_predict(
            best, X_train, y_train, cv=cv, method="predict_proba", n_jobs=cv_cfg.get("n_jobs", -1)
        )[:, 1]
        threshold, cv_f1 = best_threshold(y_train, oof)
        mlflow.log_param("decision_threshold", round(threshold, 4))
        mlflow.log_metric("cv_f1_at_threshold", cv_f1)
        mlflow.log_metric("cv_best_roc_auc", float(grid.best_score_))

        # ---- Held-out test metrics -----------------------------------------------
        y_score = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_score, threshold)
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "decision_threshold"})

        # ---- Feature importance ----------------------------------------------------
        art_dir = resolve(cfg["artifacts"]["dir"])
        art_dir.mkdir(parents=True, exist_ok=True)
        importance = get_feature_importance(best)
        if importance is not None:
            path = plot_feature_importance(get_feature_names(best), importance, art_dir)
            mlflow.log_artifact(str(path))

        # ---- Log + register -----------------------------------------------------------
        model_meta = {
            "decision_threshold": threshold,
            "model_type": model_type,
            "data_sha256": fingerprint["data_sha256"],
        }
        signature = infer_signature(X_train, y_score)
        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
            metadata=model_meta,
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        version_tags = {**model_meta, "test_roc_auc": round(metrics["test_roc_auc"], 5)}
        mv = register_challenger(model_uri, cfg, version_tags)

        # ---- Local artifacts for serving -----------------------------------------------
        model_path = resolve(cfg["artifacts"]["model_path"])
        joblib.dump(best, model_path)
        write_meta(
            meta_path(cfg),
            {
                "model_name": cfg["mlflow"]["registered_model_name"],
                "version": str(mv.version) if mv else None,
                "alias": cfg["mlflow"].get("challenger_alias", "challenger") if mv else None,
                "run_id": run.info.run_id,
                "experiment": experiment,
                "model_uri": model_uri,
                "model_type": model_type,
                "decision_threshold": threshold,
                "best_params": grid.best_params_,
                "metrics": metrics,
                "data_sha256": fingerprint["data_sha256"],
                "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

        log.info("run_id=%s", run.info.run_id)
        log.info("threshold=%.3f (cv F1 %.4f)", threshold, cv_f1)
        log.info(
            "test roc_auc=%.4f ap=%.4f f1=%.4f (f1@0.5=%.4f) brier=%.4f",
            metrics["test_roc_auc"],
            metrics["test_average_precision"],
            metrics["test_f1"],
            metrics["test_f1_at_0.5"],
            metrics["test_brier"],
        )
        log.info("model saved to %s", model_path)
        return run.info.run_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Adult Income classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config), Path(args.config))


if __name__ == "__main__":
    main()
