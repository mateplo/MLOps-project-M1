import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.pipeline import build_pipeline

NUMERIC = ["age", "fnlwgt", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
CATEGORICAL = [
    "workclass",
    "marital_status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native_country",
]
EXAMPLE = json.loads((Path(__file__).parents[1] / "examples" / "person.json").read_text())


@pytest.fixture
def client(tmp_path, monkeypatch):
    rows = [EXAMPLE, {**EXAMPLE, "age": 22, "education_num": 9, "capital_gain": 0}]
    X = pd.DataFrame(rows * 10)
    y = pd.Series([1, 0] * 10)
    model = build_pipeline(NUMERIC, CATEGORICAL, "logreg").fit(X, y)
    path = tmp_path / "model.joblib"
    joblib.dump(model, path)
    monkeypatch.setattr(app_module, "MODEL_PATH", str(path))
    monkeypatch.setattr(app_module, "_model", None)
    return TestClient(app_module.app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {">50K", "<=50K"}
    assert 0.0 <= body["score"] <= 1.0


def test_predict_validation_error(client):
    r = client.post("/predict", json={**EXAMPLE, "age": -5})
    assert r.status_code == 422


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setattr(app_module, "MODEL_PATH", "/nonexistent/model.joblib")
    monkeypatch.setattr(app_module, "_model", None)
    r = TestClient(app_module.app).post("/predict", json=EXAMPLE)
    assert r.status_code == 503
