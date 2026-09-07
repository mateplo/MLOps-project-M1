"""Helpers: logging, config, MLflow setup, data loading/cleaning, splitting, plots, metadata."""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

from src.meta import META_FILENAME, read_meta, write_meta  # noqa: F401  (re-exported)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def get_logger(name: str = "adult_income") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
    return logger


log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Config & environment
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    """Absolute path: as-is if already absolute, else relative to the project root."""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def setup_mlflow(cfg: dict[str, Any]) -> str:
    """Load .env, configure tracking URI + experiment. Returns the experiment name.

    Precedence: environment variables > config.yaml > defaults.
    """
    import mlflow

    load_dotenv(PROJECT_ROOT / ".env")
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    experiment = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", cfg.get("mlflow", {}).get("experiment_name", "default")
    )
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)
    log.info("mlflow tracking_uri=%s experiment=%s", tracking_uri, experiment)
    return experiment


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def file_fingerprint(path: str | Path) -> dict[str, Any]:
    """sha256 + size of a file, for data lineage."""
    p = resolve(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return {"data_sha256": h.hexdigest()[:16], "data_bytes": p.stat().st_size, "data_path": str(p)}


def clean_adult(df: pd.DataFrame, target: str, positive_label: str) -> pd.DataFrame:
    """Normalise the raw UCI Adult dataset.

    - strips whitespace on string columns
    - maps '?' to NaN (missing values in workclass / occupation / native_country)
    - normalises the target ('>50K.' in adult.test has a trailing dot) to 0/1
    - casts numeric columns to float64 so the model signature tolerates NaN at inference
    """
    df = df.copy()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})
    df[target] = (
        df[target].astype(str).str.replace(".", "", regex=False).str.strip() == positive_label
    ).astype(int)
    for col in df.select_dtypes(include="number").columns:
        if col != target:
            df[col] = df[col].astype("float64")
    return df


def load_dataframe(cfg: dict[str, Any], validate: bool = True) -> pd.DataFrame:
    """Read the raw CSV, validate it against config rules, clean it, optionally subsample."""
    d = cfg["data"]
    csv_path = resolve(d["csv_path"])
    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found. Run `make data` first.")
    df = pd.read_csv(csv_path)
    if validate:
        from src.validate import validate_or_raise

        validate_or_raise(df, cfg)
    df = clean_adult(df, d["target"], d["positive_label"])
    sample = d.get("sample_rows")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=d["random_state"]).reset_index(drop=True)
        log.info("subsampled to %d rows", len(df))
    return df


def split_features(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    feats = cfg["features"]
    cols = feats["numeric"] + feats["categorical"]
    return df[cols], df[cfg["data"]["target"]]


def load_data(cfg: dict[str, Any], validate: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """(X, y) ready for the pipeline."""
    return split_features(load_dataframe(cfg, validate=validate), cfg)


def split_data(X: pd.DataFrame, y: pd.Series, cfg: dict[str, Any]):
    """Stratified train/test split driven by the config."""
    return train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=cfg["data"]["random_state"],
        stratify=y,
    )


# --------------------------------------------------------------------------- #
# Model metadata (artifacts/model_meta.json): the bridge between registry and serving
# --------------------------------------------------------------------------- #
def meta_path(cfg: dict[str, Any]) -> Path:
    return resolve(cfg["artifacts"]["dir"]) / META_FILENAME


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_roc(y_true, y_score, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_title("ROC curve")
    return _save(fig, out_dir / "roc_curve.png")


def plot_pr(y_true, y_score, out_dir: Path, threshold: float | None = None) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    if threshold is not None:
        y_pred = (np.asarray(y_score) >= threshold).astype(int)
        tp = int(((y_pred == 1) & (np.asarray(y_true) == 1)).sum())
        p = tp / max(int(y_pred.sum()), 1)
        r = tp / max(int(np.asarray(y_true).sum()), 1)
        ax.scatter([r], [p], c="red", zorder=5, label=f"threshold={threshold:.2f}")
        ax.legend()
    ax.set_title("Precision-Recall curve")
    return _save(fig, out_dir / "pr_curve.png")


def plot_confusion(y_true, y_pred, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix")
    return _save(fig, out_dir / "confusion_matrix.png")


def plot_calibration(y_true, y_score, out_dir: Path, n_bins: int = 10) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    CalibrationDisplay.from_predictions(y_true, y_score, n_bins=n_bins, ax=ax, name="model")
    ax.set_title("Calibration curve (reliability diagram)")
    return _save(fig, out_dir / "calibration_curve.png")


def plot_feature_importance(names, values, out_dir: Path, top_n: int = 20) -> Path:
    order = np.argsort(np.abs(values))[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order][::-1], [values[i] for i in order][::-1])
    ax.set_title(f"Top {top_n} feature importances")
    return _save(fig, out_dir / "feature_importance.png")
