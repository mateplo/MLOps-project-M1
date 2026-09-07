import json

import numpy as np
import pandas as pd
import pytest

from src.export_onnx import (
    INFREQUENT,
    MISSING,
    browser_preprocess,
    prepare_for_export,
    to_onnx,
    validate,
)
from src.pipeline import build_pipeline

pytest.importorskip("skl2onnx")
pytest.importorskip("onnxruntime")

NUMERIC = ["age", "capital_gain"]
CATEGORICAL = ["workclass", "country"]


@pytest.fixture
def fitted():
    rng = np.random.default_rng(0)
    n = 600
    X = pd.DataFrame(
        {
            "age": rng.integers(18, 80, n).astype(float),
            "capital_gain": np.where(rng.random(n) < 0.2, rng.integers(0, 50000, n), 0).astype(
                float
            ),
            "workclass": rng.choice(["Private", "Gov", "Self", None], n, p=[0.6, 0.2, 0.15, 0.05]),
            # 'Rare1'/'Rare2' are below min_frequency -> grouped as infrequent by sklearn
            "country": rng.choice(["US", "MX", "Rare1", "Rare2"], n, p=[0.85, 0.11, 0.02, 0.02]),
        }
    )
    y = pd.Series(((X["age"] > 40) | (X["capital_gain"] > 0)).astype(int))
    pipe = build_pipeline(NUMERIC, CATEGORICAL, "hgb", log1p=["capital_gain"], min_frequency=0.05)
    return pipe.fit(X, y), X


def test_export_matches_sklearn_including_rare_and_missing(fitted):
    pipe, X = fitted
    export_pipe, spec = prepare_for_export(pipe)
    assert spec["numeric"] == ["age", "capital_gain"] and spec["categorical"] == CATEGORICAL
    assert set(spec["frequent_categories"]["country"]) == {"US", "MX"}
    onnx_bytes = to_onnx(export_pipe, spec)
    assert len(onnx_bytes) > 1000
    # unseen category + missing value + rare category all handled like sklearn
    probe = pd.concat(
        [
            X,
            pd.DataFrame(
                {"age": [30.0], "capital_gain": [0.0], "workclass": [None], "country": ["Atlantis"]}
            ),
        ]
    )
    diff = validate(onnx_bytes, pipe, probe, spec, tol=1e-4)
    assert diff < 1e-4


def test_browser_preprocess_tokens():
    spec = {
        "numeric": ["age"],
        "categorical": ["c"],
        "frequent_categories": {"c": ["A", "B"]},
        "missing_token": MISSING,
        "infrequent_token": INFREQUENT,
    }
    X = pd.DataFrame({"age": ["31", None], "c": [" A ", "Z"]})
    feed = browser_preprocess(X, spec)
    assert feed["age"].dtype == np.float32 and np.isnan(feed["age"][1, 0])
    assert feed["c"][:, 0].tolist() == ["A", INFREQUENT]
    X2 = pd.DataFrame({"age": [1], "c": [None]})
    assert browser_preprocess(X2, spec)["c"][0, 0] == MISSING
    json.dumps(spec)  # spec must be JSON-serialisable for the browser
