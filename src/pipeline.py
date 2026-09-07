"""Build the scikit-learn preprocessing + model Pipeline."""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

SUPPORTED_MODELS = ("logreg", "random_forest", "gradient_boosting", "hgb")


def build_preprocessor(
    numeric: list[str],
    categorical: list[str],
    log1p: list[str] | None = None,
    min_frequency: float | int | None = None,
) -> ColumnTransformer:
    """ColumnTransformer with three branches.

    - num : median imputation + standard scaling
    - log : median imputation + log1p (for heavy-tailed columns) + scaling
    - cat : most-frequent imputation + one-hot (rare categories grouped when
            `min_frequency` is set, unknown categories ignored)
    """
    log_cols = [c for c in (log1p or []) if c in numeric]
    plain_cols = [c for c in numeric if c not in log_cols]

    num = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    log = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scaler", StandardScaler()),
        ]
    )
    ohe_kwargs: dict = {"sparse_output": False}
    if min_frequency:
        ohe_kwargs.update(min_frequency=min_frequency, handle_unknown="infrequent_if_exist")
    else:
        ohe_kwargs.update(handle_unknown="ignore")
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(**ohe_kwargs)),
        ]
    )

    transformers = [("num", num, plain_cols)]
    if log_cols:
        transformers.append(("log", log, log_cols))
    transformers.append(("cat", cat, categorical))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_model(model_type: str = "logreg", random_state: int = 42):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_type == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=random_state)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    if model_type == "hgb":
        return HistGradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model_type={model_type!r}. Choose from {SUPPORTED_MODELS}")


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    random_state: int = 42,
    log1p: list[str] | None = None,
    min_frequency: float | int | None = None,
    calibrate: dict | None = None,
) -> Pipeline:
    """Full pipeline: ColumnTransformer -> classifier (optionally calibrated).

    `calibrate={"method": "isotonic"|"sigmoid", "cv": 3}` wraps the classifier in a
    CalibratedClassifierCV; hyper-parameters are then addressed as `model__estimator__<p>`.
    """
    model = build_model(model_type, random_state)
    if calibrate:
        model = CalibratedClassifierCV(
            estimator=model, method=calibrate.get("method", "isotonic"), cv=calibrate.get("cv", 3)
        )
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(numeric, categorical, log1p, min_frequency)),
            ("model", model),
        ]
    )


def build_pipeline_from_config(cfg: dict) -> Pipeline:
    feats = cfg["features"]
    return build_pipeline(
        numeric=feats["numeric"],
        categorical=feats["categorical"],
        model_type=cfg["model"]["type"],
        random_state=cfg["data"]["random_state"],
        log1p=feats.get("log1p"),
        min_frequency=feats.get("min_frequency"),
        calibrate=cfg["model"].get("calibrate"),
    )


def param_prefix(cfg: dict) -> str:
    """GridSearchCV key prefix for the classifier's hyper-parameters."""
    return "model__estimator__" if cfg["model"].get("calibrate") else "model__"


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Human-readable output feature names of a fitted pipeline's preprocessor."""
    return [n.split("__", 1)[-1] for n in pipe.named_steps["pre"].get_feature_names_out()]


def get_feature_importance(pipe: Pipeline) -> np.ndarray | None:
    """feature_importances_ or |coef_| of the final estimator, unwrapping calibration."""
    model = pipe.named_steps["model"]
    if isinstance(model, CalibratedClassifierCV):
        model = model.calibrated_classifiers_[0].estimator
    if hasattr(model, "feature_importances_"):
        return np.asarray(model.feature_importances_)
    if hasattr(model, "coef_"):
        return np.asarray(model.coef_).ravel()
    return None
