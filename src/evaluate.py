"""Final evaluation of the trained model: metrics, ROC/PR/confusion plots, predictions CSV,
per-subgroup metrics. Everything is logged to MLflow as an `evaluate` run.

Usage:
    python src/evaluate.py --config configs/config.yaml [--model artifacts/model.joblib]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.train import compute_metrics
from src.utils import (
    PROJECT_ROOT,
    load_config,
    load_data,
    plot_confusion,
    plot_pr,
    plot_roc,
    setup_mlflow,
    split_data,
)

SUBGROUP_COLS = ("sex", "race")


def load_model(model_ref: str):
    """Accept a local joblib path or an MLflow URI (models:/... or runs:/...)."""
    if model_ref.startswith(("models:/", "runs:/")):
        return mlflow.sklearn.load_model(model_ref)
    return joblib.load(model_ref)


def subgroup_metrics(X_test: pd.DataFrame, y_test, y_score) -> pd.DataFrame:
    """ROC-AUC + positive rate per subgroup, a lightweight fairness check."""
    rows = []
    for col in SUBGROUP_COLS:
        if col not in X_test.columns:
            continue
        for value, idx in X_test.groupby(col).groups.items():
            yt, ys = y_test.loc[idx], pd.Series(y_score, index=X_test.index).loc[idx]
            if yt.nunique() < 2:
                continue
            rows.append(
                {
                    "feature": col,
                    "group": value,
                    "n": len(idx),
                    "positive_rate": float(yt.mean()),
                    "roc_auc": float(roc_auc_score(yt, ys)),
                }
            )
    return pd.DataFrame(rows)


def evaluate(cfg: dict, model_ref: str) -> dict:
    setup_mlflow(cfg)
    X, y = load_data(cfg)
    _, X_test, _, y_test = split_data(X, y, cfg)
    model = load_model(model_ref)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_score)

    art_dir = PROJECT_ROOT / cfg["artifacts"]["dir"]
    art_dir.mkdir(parents=True, exist_ok=True)

    train_info_path = art_dir / "train_run.json"
    parent_run_id = None
    if train_info_path.exists():
        parent_run_id = json.loads(train_info_path.read_text()).get("run_id")

    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.set_tags({"stage": "evaluate", "model_ref": model_ref})
        if parent_run_id:
            mlflow.set_tag("train_run_id", parent_run_id)
        mlflow.log_metrics(metrics)

        # Plots
        for path in (
            plot_roc(y_test, y_score, art_dir),
            plot_pr(y_test, y_score, art_dir),
            plot_confusion(y_test, y_pred, art_dir),
        ):
            mlflow.log_artifact(str(path), artifact_path="plots")

        # Classification report
        report = classification_report(
            y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True
        )
        report_path = art_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        # Predictions CSV for error analysis
        preds = X_test.copy()
        preds["y_true"] = y_test.values
        preds["y_pred"] = y_pred
        preds["y_score"] = y_score
        preds["error"] = (preds["y_true"] != preds["y_pred"]).astype(int)
        preds_path = art_dir / "predictions.csv"
        preds.to_csv(preds_path, index=False)
        mlflow.log_artifact(str(preds_path))

        # Subgroup metrics
        sub = subgroup_metrics(X_test, y_test, y_score)
        sub_path = art_dir / "subgroup_metrics.csv"
        sub.to_csv(sub_path, index=False)
        mlflow.log_artifact(str(sub_path))
        for _, r in sub.iterrows():
            mlflow.log_metric(f"roc_auc_{r['feature']}_{_slug(r['group'])}", r["roc_auc"])

        print(f"Evaluate run id : {run.info.run_id}")
        for k, v in metrics.items():
            print(f"{k:<26} {v:.4f}")
        print(f"Artifacts in    : {art_dir}")
        return metrics


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Adult Income classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="joblib path or MLflow URI (models:/AdultIncomeClassifier@staging). "
        "Defaults to artifacts.model_path from the config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_ref = args.model or str(PROJECT_ROOT / cfg["artifacts"]["model_path"])
    if not model_ref.startswith(("models:/", "runs:/")) and not Path(model_ref).exists():
        raise SystemExit(f"Model not found at {model_ref}. Run `make train` first.")
    evaluate(cfg, model_ref)


if __name__ == "__main__":
    main()
