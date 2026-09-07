"""Batch inference: read a CSV of raw features, write a CSV with predictions.

Usage:
    python src/predict.py --input data/new_people.csv --output artifacts/scored.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from src.utils import PROJECT_ROOT, load_config


def predict_df(model, df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    missing = [c for c in feats if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")
    X = df[feats].copy()
    for c in X.select_dtypes(include="object").columns:
        X[c] = X[c].str.strip().replace({"?": None})
    out = df.copy()
    out["score"] = model.predict_proba(X)[:, 1]
    out["prediction"] = model.predict(X)
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
    model_path = args.model or PROJECT_ROOT / cfg["artifacts"]["model_path"]
    model = joblib.load(model_path)
    df = pd.read_csv(args.input, skipinitialspace=True)
    out = predict_df(model, df, cfg)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out):,} predictions to {args.output}")
    print(out["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
