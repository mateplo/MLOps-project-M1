"""Train + tune the pipeline with GridSearchCV, track with MLflow, register the best model.

Usage:
    python src/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from mlflow.models import infer_signature
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.pipeline import build_pipeline, get_feature_names
from src.utils import (
    PROJECT_ROOT,
    load_config,
    load_data,
    plot_feature_importance,
    setup_mlflow,
    split_data,
)


def make_param_grid(params: dict) -> dict:
    """Prefix the config grid with the pipeline step name (`model__C`, ...)."""
    return {f"model__{k}": (v if isinstance(v, list) else [v]) for k, v in params.items()}


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    return {
        "test_roc_auc": roc_auc_score(y_true, y_score),
        "test_average_precision": average_precision_score(y_true, y_score),
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
    }


def register_model(model_uri: str, cfg: dict) -> str | None:
    """Register the logged model and attach an alias (MLflow >= 2.9) + legacy stage."""
    ml_cfg = cfg.get("mlflow", {})
    if not ml_cfg.get("register", False):
        return None
    name = ml_cfg["registered_model_name"]
    mv = mlflow.register_model(model_uri=model_uri, name=name)
    client = mlflow.MlflowClient()
    alias = ml_cfg.get("alias")
    if alias:
        client.set_registered_model_alias(name, alias, mv.version)
    # Stages are deprecated in MLflow 2.9+, but still requested by the course proposal.
    try:
        client.transition_model_version_stage(name, mv.version, stage="Staging")
    except Exception as exc:  # pragma: no cover - depends on MLflow version/backend
        print(f"[warn] could not set legacy stage: {exc}")
    return f"models:/{name}/{mv.version}"


def train(cfg: dict, cfg_path: Path = Path("configs/config.yaml")) -> str:
    experiment = setup_mlflow(cfg)
    mlflow.sklearn.autolog(log_models=False, log_input_examples=False, silent=True)

    X, y = load_data(cfg)
    X_train, X_test, y_train, y_test = split_data(X, y, cfg)

    feats = cfg["features"]
    model_type = cfg["model"]["type"]
    pipe = build_pipeline(
        feats["numeric"], feats["categorical"], model_type, cfg["data"]["random_state"]
    )

    cv_cfg = cfg["cv"]
    cv = StratifiedKFold(
        n_splits=cv_cfg["n_splits"], shuffle=True, random_state=cfg["data"]["random_state"]
    )
    grid = GridSearchCV(
        pipe,
        param_grid=make_param_grid(cfg["model"].get("params", {})),
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
            }
        )
        mlflow.log_params(
            {
                "n_train": len(X_train),
                "n_test": len(X_test),
                "n_numeric": len(feats["numeric"]),
                "n_categorical": len(feats["categorical"]),
                "cv_strategy": cv_cfg["strategy"],
                "cv_n_splits": cv_cfg["n_splits"],
                "cv_scoring": cv_cfg["scoring"],
            }
        )
        mlflow.log_artifact(str(cfg_path), artifact_path="config")

        grid.fit(X_train, y_train)
        best = grid.best_estimator_

        # ---- Test-set metrics (held out, never seen during CV) -------------
        y_pred = best.predict(X_test)
        y_score = best.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_score)
        metrics["cv_best_roc_auc"] = float(grid.best_score_)
        mlflow.log_metrics(metrics)

        # ---- Feature importance / coefficients ------------------------------
        art_dir = PROJECT_ROOT / cfg["artifacts"]["dir"]
        art_dir.mkdir(parents=True, exist_ok=True)
        model = best.named_steps["model"]
        names = get_feature_names(best)
        values = None
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        elif hasattr(model, "coef_"):
            values = model.coef_.ravel()
        if values is not None:
            mlflow.log_artifact(str(plot_feature_importance(names, values, art_dir)))

        # ---- Log + register the best model ----------------------------------
        signature = infer_signature(X_train, y_score)
        mlflow.sklearn.log_model(
            best,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(5),
        )
        model_uri = f"runs:/{run.info.run_id}/model"
        registered_uri = register_model(model_uri, cfg)

        # ---- Local copy for evaluate.py / predict.py / API -------------------
        model_path = PROJECT_ROOT / cfg["artifacts"]["model_path"]
        joblib.dump(best, model_path)
        with open(art_dir / "train_run.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "run_id": run.info.run_id,
                    "experiment": experiment,
                    "model_uri": model_uri,
                    "registered_model_uri": registered_uri,
                    "best_params": {k: _jsonable(v) for k, v in grid.best_params_.items()},
                    "metrics": metrics,
                },
                f,
                indent=2,
            )

        print(f"Run id           : {run.info.run_id}")
        print(f"Best params      : {grid.best_params_}")
        print(f"CV best ROC-AUC  : {grid.best_score_:.4f}")
        print(f"Test ROC-AUC     : {metrics['test_roc_auc']:.4f}")
        print(f"Model saved to   : {model_path}")
        if registered_uri:
            print(f"Registered as    : {registered_uri}")
        return run.info.run_id


def _jsonable(v):
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Adult Income classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    train(load_config(args.config), Path(args.config))


if __name__ == "__main__":
    main()
