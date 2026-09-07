"""Final evaluation of a model: metrics at the tuned threshold, ROC/PR/confusion/calibration plots,
predictions CSV, per-subgroup metrics. Everything is logged to MLflow as an `evaluate` run.

Usage:
    python -m src.evaluate --config configs/config.yaml [--model <joblib | models:/Name@champion>]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score

from src.metrics import compute_metrics
from src.utils import (
    get_logger,
    load_config,
    load_data,
    meta_path,
    plot_calibration,
    plot_confusion,
    plot_pr,
    plot_roc,
    read_meta,
    resolve,
    setup_mlflow,
    split_data,
)

log = get_logger(__name__)
SUBGROUP_COLS = ("sex", "race")


def load_model(model_ref: str, cfg: dict) -> tuple[object, dict]:
    """Return (model, meta). Accepts a joblib path or an MLflow URI (models:/..., runs:/...)."""
    if model_ref.startswith(("models:/", "runs:/")):
        model = mlflow.sklearn.load_model(model_ref)
        info = mlflow.models.get_model_info(model_ref)
        meta = dict(info.metadata or {})
        meta.setdefault("run_id", info.run_id)
        return model, meta
    model = joblib.load(model_ref)
    return model, read_meta(meta_path(cfg))


def subgroup_metrics(X_test: pd.DataFrame, y_test, y_score, threshold: float) -> pd.DataFrame:
    """ROC-AUC, positive rate and predicted-positive rate per subgroup (fairness check)."""
    rows = []
    score = pd.Series(y_score, index=X_test.index)
    for col in SUBGROUP_COLS:
        if col not in X_test.columns:
            continue
        for value, idx in X_test.groupby(col).groups.items():
            yt, ys = y_test.loc[idx], score.loc[idx]
            if yt.nunique() < 2:
                continue
            rows.append(
                {
                    "feature": col,
                    "group": value,
                    "n": len(idx),
                    "positive_rate": float(yt.mean()),
                    "predicted_positive_rate": float((ys >= threshold).mean()),
                    "roc_auc": float(roc_auc_score(yt, ys)),
                }
            )
    return pd.DataFrame(rows)


def evaluate(cfg: dict, model_ref: str) -> dict:
    setup_mlflow(cfg)
    X, y = load_data(cfg)
    _, X_test, _, y_test = split_data(X, y, cfg)
    model, meta = load_model(model_ref, cfg)
    threshold = float(meta.get("decision_threshold", 0.5))

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)
    metrics = compute_metrics(y_test, y_score, threshold)

    art_dir = resolve(cfg["artifacts"]["dir"])
    art_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name="evaluate") as run:
        mlflow.set_tags({"stage": "evaluate", "model_ref": model_ref})
        if meta.get("run_id"):
            mlflow.set_tag("train_run_id", meta["run_id"])
        if meta.get("version"):
            mlflow.set_tag("model_version", str(meta["version"]))
        mlflow.log_param("decision_threshold", round(threshold, 4))
        mlflow.log_metrics({k: v for k, v in metrics.items() if k != "decision_threshold"})

        for path in (
            plot_roc(y_test, y_score, art_dir),
            plot_pr(y_test, y_score, art_dir, threshold),
            plot_confusion(y_test, y_pred, art_dir),
            plot_calibration(y_test, y_score, art_dir),
        ):
            mlflow.log_artifact(str(path), artifact_path="plots")

        report = classification_report(
            y_test, y_pred, target_names=["<=50K", ">50K"], output_dict=True
        )
        report_path = art_dir / "classification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        mlflow.log_artifact(str(report_path))

        preds = X_test.copy()
        preds["y_true"] = y_test.values
        preds["y_score"] = y_score
        preds["y_pred"] = y_pred
        preds["error"] = (preds["y_true"] != preds["y_pred"]).astype(int)
        preds_path = art_dir / "predictions.csv"
        preds.to_csv(preds_path, index=False)
        mlflow.log_artifact(str(preds_path))

        sub = subgroup_metrics(X_test, y_test, y_score, threshold)
        sub_path = art_dir / "subgroup_metrics.csv"
        sub.to_csv(sub_path, index=False)
        mlflow.log_artifact(str(sub_path))
        for _, r in sub.iterrows():
            mlflow.log_metric(f"roc_auc_{r['feature']}_{_slug(r['group'])}", r["roc_auc"])
        if not sub.empty:
            gap = sub.groupby("feature")["roc_auc"].agg(lambda s: s.max() - s.min())
            for feat, g in gap.items():
                mlflow.log_metric(f"roc_auc_gap_{feat}", float(g))

        log.info("evaluate run_id=%s (threshold=%.3f)", run.info.run_id, threshold)
        for k, v in metrics.items():
            log.info("%-26s %.4f", k, v)
        return metrics


def _slug(s: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(s)).strip("_").lower()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the Adult Income classifier")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--model",
        default=None,
        help="joblib path or MLflow URI (models:/AdultIncomeClassifier@champion). "
        "Defaults to artifacts.model_path from the config.",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    model_ref = args.model or str(resolve(cfg["artifacts"]["model_path"]))
    if not model_ref.startswith(("models:/", "runs:/")) and not Path(model_ref).exists():
        raise SystemExit(f"Model not found at {model_ref}. Run `make train` first.")
    evaluate(cfg, model_ref)


if __name__ == "__main__":
    main()
