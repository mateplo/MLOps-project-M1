"""Data validation: fail fast when data/raw.csv no longer looks like the Adult dataset.

Rules live in config.yaml under `validation:` so they are visible and reviewable.

Usage:
    python -m src.validate --config configs/config.yaml
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.utils import PROJECT_ROOT, get_logger, load_config

log = get_logger(__name__)


class DataValidationError(ValueError):
    pass


def validate_dataframe(df: pd.DataFrame, cfg: dict) -> list[str]:
    """Return a list of violations (empty = valid). Operates on the *raw* dataframe."""
    rules = cfg.get("validation", {})
    d = cfg["data"]
    target = d["target"]
    errors: list[str] = []

    # 1. Columns
    required = d["columns"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors  # nothing else is meaningful

    # 2. Size
    min_rows = rules.get("min_rows", 1)
    if len(df) < min_rows:
        errors.append(f"too few rows: {len(df)} < {min_rows}")

    # 3. Target values + positive rate
    tgt = df[target].astype(str).str.replace(".", "", regex=False).str.strip()
    allowed = set(rules.get("target_values", [d["positive_label"]]))
    bad = sorted(set(tgt.unique()) - allowed) if allowed else []
    if bad:
        errors.append(f"unexpected target values: {bad}")
    pos_rate = float((tgt == d["positive_label"]).mean())
    lo, hi = rules.get("positive_rate", [0.0, 1.0])
    if not lo <= pos_rate <= hi:
        errors.append(f"positive rate {pos_rate:.3f} outside [{lo}, {hi}]")

    # 4. Numeric ranges + dtype
    for col, (lo, hi) in rules.get("numeric_ranges", {}).items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        if (
            s.isna().mean() > 0.0
            and df[col].notna().any()
            and s.isna().sum() > df[col].isna().sum()
        ):
            errors.append(f"{col}: non-numeric values found")
        if lo is not None and (s < lo).any():
            errors.append(f"{col}: values below {lo} (min={s.min()})")
        if hi is not None and (s > hi).any():
            errors.append(f"{col}: values above {hi} (max={s.max()})")

    # 5. Missing rate per feature
    max_missing = rules.get("max_missing_rate", 1.0)
    feats = cfg["features"]["numeric"] + cfg["features"]["categorical"]
    for col in feats:
        raw_missing = df[col].isna() | (df[col].astype(str).str.strip() == "?")
        rate = float(raw_missing.mean())
        if rate > max_missing:
            errors.append(f"{col}: missing rate {rate:.3f} > {max_missing}")

    # 6. Categorical cardinality
    for col, max_card in rules.get("max_cardinality", {}).items():
        if col in df.columns and df[col].nunique() > max_card:
            errors.append(f"{col}: {df[col].nunique()} categories > {max_card}")

    return errors


def validate_or_raise(df: pd.DataFrame, cfg: dict) -> None:
    errors = validate_dataframe(df, cfg)
    if errors:
        raise DataValidationError("Data validation failed:\n  - " + "\n  - ".join(errors))
    log.info("data validation passed (%d rows, %d columns)", len(df), df.shape[1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate data/raw.csv against the config rules")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    df = pd.read_csv(PROJECT_ROOT / cfg["data"]["csv_path"])
    validate_or_raise(df, cfg)


if __name__ == "__main__":
    main()
