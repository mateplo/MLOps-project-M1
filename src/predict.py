"""Batch inference: read a CSV of raw features, write a CSV with scores + labels.

Uses the decision threshold stored in artifacts/model_meta.json.

Usage:
    python -m src.predict --input data/new_people.csv --output artifacts/scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils import get_logger, load_config, meta_path, read_meta, resolve

log = get_logger(__name__)


def prepare_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[feats].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].str.strip().replace({"?": np.nan})
    for c in cfg["features"]["numeric"]:
        X[c] = pd.to_numeric(X[c], errors="coerce").astype("float64")
    return X


def predict_df(model, df: pd.DataFrame, cfg: dict, threshold: float = 0.5) -> pd.DataFrame:
    X = prepare_features(df, cfg)
    out = df.copy()
    out["score"] = model.predict_proba(X)[:, 1]
    out["prediction"] = (out["score"] >= threshold).astype(int)
    out["label"] = out["prediction"].map({0: "<=50K", 1: ">50K"})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch predictions")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model", default=None, help="joblib path (default from config)")
    parser.add_argument("--input", required=True, help="CSV with raw feature columns")
    parser.add_argument("--output", required=True, help="CSV to write")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = joblib.load(args.model or resolve(cfg["artifacts"]["model_path"]))
    meta = read_meta(meta_path(cfg))
    threshold = float(meta.get("decision_threshold", 0.5))
    df = pd.read_csv(args.input, skipinitialspace=True)
    out = predict_df(model, df, cfg, threshold)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    log.info(
        "wrote %d predictions to %s (threshold=%.3f, model v%s)",
        len(out),
        args.output,
        threshold,
        meta.get("version"),
    )
    log.info("\n%s", out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
