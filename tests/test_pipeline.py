import numpy as np
import pandas as pd
import pytest

from src.pipeline import (
    SUPPORTED_MODELS,
    build_pipeline,
    get_feature_importance,
    get_feature_names,
    param_prefix,
)
from tests.conftest import CATEGORICAL, NUMERIC


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert list(pipe.named_steps) == ["pre", "model"]


@pytest.mark.parametrize("model_type", SUPPORTED_MODELS)
def test_pipeline_fits_and_predicts(toy_data, model_type):
    X, y = toy_data
    pipe = build_pipeline(NUMERIC, CATEGORICAL, model_type).fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert set(pipe.predict(X)) <= {0, 1}


def test_pipeline_handles_missing_and_unknown_categories(toy_data):
    X, y = toy_data
    pipe = build_pipeline(NUMERIC, CATEGORICAL, "logreg", min_frequency=0.05).fit(X, y)
    new = pd.DataFrame(
        {"age": [np.nan], "hours_per_week": [40.0], "workclass": ["Martian"], "sex": [None]}
    )
    assert pipe.predict_proba(new).shape == (1, 2)


def test_log1p_branch_and_feature_names(toy_data):
    X, y = toy_data
    pipe = build_pipeline(NUMERIC, CATEGORICAL, "logreg", log1p=["hours_per_week"]).fit(X, y)
    names = get_feature_names(pipe)
    assert "age" in names and "hours_per_week" in names
    assert any(n.startswith("sex_") for n in names)
    assert "log" in dict(pipe.named_steps["pre"].named_transformers_)


def test_calibrated_pipeline_and_importance(toy_data):
    X, y = toy_data
    pipe = build_pipeline(
        NUMERIC, CATEGORICAL, "logreg", calibrate={"method": "sigmoid", "cv": 2}
    ).fit(X, y)
    assert pipe.predict_proba(X).shape == (len(X), 2)
    imp = get_feature_importance(pipe)
    assert imp is not None and len(imp) == len(get_feature_names(pipe))


def test_param_prefix():
    assert param_prefix({"model": {}}) == "model__"
    assert param_prefix({"model": {"calibrate": {"method": "isotonic"}}}) == "model__estimator__"


def test_unsupported_model_type_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_pipeline(["a"], ["b"], "svm")
