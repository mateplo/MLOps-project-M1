"""Feature drift report: Population Stability Index (PSI) between the training data and
the features seen by the API (logs/predictions.jsonl). The report is written to
artifacts/drift_report.csv and logged to MLflow as a `drift` run (metrics psi_<feature>).

PSI rule of thumb: < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.

Usage:
    python -m src.drift --config configs/config.yaml [--log logs/predictions.jsonl] [--min-rows 200]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, load_dataframe, resolve, setup_mlflow

log = get_logger(__name__)
EPS = 1e-4


def psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref, cur = ref.dropna().astype(float), cur.dropna().astype(float)
    if ref.empty or cur.empty:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # near-constant feature
        edges = np.array([ref.min() - 1, ref.median(), ref.max() + 1])
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, bins=edges)[0] / len(ref)
    c = np.histogram(cur, bins=edges)[0] / len(cur)
    return _psi(r, c)


def psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref, cur = ref.fillna("<NA>").astype(str), cur.fillna("<NA>").astype(str)
    cats = sorted(set(ref.unique()) | set(cur.unique()))
    r = ref.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = cur.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    return _psi(r, c)


def _psi(r: np.ndarray, c: np.ndarray) -> float:
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def load_prediction_log(path: Path) -> pd.DataFrame:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                feats = rec.get("features", {})
                feats = {k: (None if v in ("?", "") else v) for k, v in feats.items()}
                rows.append(
                    {
                        **feats,
                        "score": rec.get("score"),
                        "label": rec.get("label"),
                        "ts": rec.get("ts"),
                    }
                )
    return pd.DataFrame(rows)


def drift_report(cfg: dict, log_path: Path, min_rows: int = 200) -> pd.DataFrame:
    ref = load_dataframe(cfg, validate=False)
    cur = load_prediction_log(log_path)
    if len(cur) < min_rows:
        log.warning("only %d logged predictions (< %d): PSI will be noisy", len(cur), min_rows)
    feats = cfg["features"]
    rows = []
    for col in feats["numeric"]:
        if col in cur.columns:
            rows.append(
                {
                    "feature": col,
                    "type": "numeric",
                    "psi": psi_numeric(ref[col], pd.to_numeric(cur[col], errors="coerce")),
                }
            )
    for col in feats["categorical"]:
        if col in cur.columns:
            rows.append(
                {"feature": col, "type": "categorical", "psi": psi_categorical(ref[col], cur[col])}
            )
    # prediction drift: predicted positive rate vs training positive rate
    if "label" in cur.columns:
        rows.append(
            {
                "feature": "__predicted_positive_rate__",
                "type": "prediction",
                "psi": psi_categorical(
                    ref[cfg["data"]["target"]].map({0: "<=50K", 1: ">50K"}), cur["label"]
                ),
            }
        )
    rep = pd.DataFrame(rows)
    rep["status"] = pd.cut(
        rep["psi"], [-np.inf, 0.1, 0.2, np.inf], labels=["ok", "moderate", "alert"]
    ).astype(str)
    rep["n_reference"], rep["n_current"] = len(ref), len(cur)
    return rep.sort_values("psi", ascending=False).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Feature drift (PSI) between training data and served requests"
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--log", default="logs/predictions.jsonl")
    parser.add_argument("--min-rows", type=int, default=200)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    log_path = resolve(args.log)
    if not log_path.exists():
        raise SystemExit(
            f"{log_path} not found: serve the API and send some requests first (make simulate)."
        )

    rep = drift_report(cfg, log_path, args.min_rows)
    out = resolve(cfg["artifacts"]["dir"]) / "drift_report.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    rep.to_csv(out, index=False)
    log.info("\n%s", rep[["feature", "type", "psi", "status"]].to_string(index=False))
    alerts = rep[rep["status"] == "alert"]["feature"].tolist()
    log.info("%d features in alert: %s", len(alerts), alerts or "-")

    if not args.no_mlflow:
        setup_mlflow(cfg)
        with mlflow.start_run(run_name="drift"):
            mlflow.set_tags({"stage": "drift", "n_alerts": len(alerts)})
            mlflow.log_params(
                {
                    "n_current": int(rep["n_current"].iloc[0]),
                    "n_reference": int(rep["n_reference"].iloc[0]),
                }
            )
            mlflow.log_metrics(
                {
                    f"psi_{r.feature.strip('_')}": float(r.psi)
                    for r in rep.itertuples()
                    if np.isfinite(r.psi)
                }
            )
            mlflow.log_metric("psi_max", float(rep["psi"].max()))
            mlflow.log_artifact(str(out))


if __name__ == "__main__":
    main()
