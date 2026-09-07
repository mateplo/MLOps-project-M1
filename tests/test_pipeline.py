import numpy as np
import pandas as pd
import pytest

from src.pipeline import SUPPORTED_MODELS, build_pipeline, get_feature_names
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
    pipe = build_pipeline(NUMERIC, CATEGORICAL, "logreg").fit(X, y)
    new = pd.DataFrame(
        {"age": [np.nan], "hours_per_week": [40], "workclass": ["Martian"], "sex": [None]}
    )
    assert pipe.predict_proba(new).shape == (1, 2)


def test_feature_names_after_fit(toy_data):
    X, y = toy_data
    pipe = build_pipeline(NUMERIC, CATEGORICAL, "logreg").fit(X, y)
    names = get_feature_names(pipe)
    assert names[:2] == NUMERIC
    assert any(n.startswith("sex_") for n in names)


def test_unsupported_model_type_raises():
    with pytest.raises(ValueError, match="Unsupported"):
        build_pipeline(["a"], ["b"], "svm")
