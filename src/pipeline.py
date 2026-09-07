"""Build the scikit-learn preprocessing + model Pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SUPPORTED_MODELS = ("logreg", "random_forest", "gradient_boosting")


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[("num", num, numeric), ("cat", cat, categorical)],
        remainder="drop",
    )


def build_model(model_type: str = "logreg", random_state: int = 42):
    if model_type == "logreg":
        return LogisticRegression(max_iter=1000, random_state=random_state)
    if model_type == "random_forest":
        return RandomForestClassifier(n_jobs=-1, random_state=random_state)
    if model_type == "gradient_boosting":
        return GradientBoostingClassifier(random_state=random_state)
    raise ValueError(f"Unsupported model_type={model_type!r}. Choose from {SUPPORTED_MODELS}")


def build_pipeline(
    numeric: list[str],
    categorical: list[str],
    model_type: str = "logreg",
    random_state: int = 42,
) -> Pipeline:
    """Full pipeline: ColumnTransformer -> classifier."""
    return Pipeline(
        steps=[
            ("pre", build_preprocessor(numeric, categorical)),
            ("model", build_model(model_type, random_state)),
        ]
    )


def get_feature_names(pipe: Pipeline) -> list[str]:
    """Human-readable output feature names of a fitted pipeline's preprocessor."""
    return [n.split("__", 1)[-1] for n in pipe.named_steps["pre"].get_feature_names_out()]
