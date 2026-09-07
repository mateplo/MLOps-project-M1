"""Small helpers: config loading, MLflow setup, data cleaning, splitting, plotting."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Config & environment
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    return experiment


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def clean_adult(df: pd.DataFrame, target: str, positive_label: str) -> pd.DataFrame:
    """Normalise the raw UCI Adult dataset.

    - strips whitespace on string columns
    - maps '?' to NaN (missing values in workclass / occupation / native_country)
    - normalises the target ('>50K.' in adult.test has a trailing dot) to 0/1
    """
    df = df.copy()
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})

    df[target] = (
        df[target].astype(str).str.replace(".", "", regex=False).str.strip() == positive_label
    ).astype(int)
    return df


def load_data(cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.Series]:
    """Read data/raw.csv (created by src/data.py) and return (X, y)."""
    data_cfg = cfg["data"]
    csv_path = PROJECT_ROOT / data_cfg["csv_path"]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found. Run `make data` (python src/data.py --config ...) first."
        )
    df = pd.read_csv(csv_path)
    df = clean_adult(df, data_cfg["target"], data_cfg["positive_label"])

    feats = cfg["features"]
    cols = feats["numeric"] + feats["categorical"]
    X = df[cols]
    y = df[data_cfg["target"]]
    return X, y


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


def plot_pr(y_true, y_score, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title("Precision-Recall curve")
    return _save(fig, out_dir / "pr_curve.png")


def plot_confusion(y_true, y_pred, out_dir: Path) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["<=50K", ">50K"], ax=ax, colorbar=False
    )
    ax.set_title("Confusion matrix")
    return _save(fig, out_dir / "confusion_matrix.png")


def plot_feature_importance(names, values, out_dir: Path, top_n: int = 20) -> Path:
    order = np.argsort(np.abs(values))[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh([names[i] for i in order][::-1], [values[i] for i in order][::-1])
    ax.set_title(f"Top {top_n} feature importances")
    return _save(fig, out_dir / "feature_importance.png")
