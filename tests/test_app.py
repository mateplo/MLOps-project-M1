import json
from pathlib import Path

import joblib
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src import app as app_module
from src.pipeline import build_pipeline
from src.utils import META_FILENAME, write_meta

NUMERIC = ["age", "education_num", "capital_gain", "capital_loss", "hours_per_week"]
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
    X = pd.DataFrame(rows * 10)[NUMERIC + CATEGORICAL].astype({c: float for c in NUMERIC})
    y = pd.Series([1, 0] * 10)
    model = build_pipeline(NUMERIC, CATEGORICAL, "logreg").fit(X, y)
    joblib.dump(model, tmp_path / "model.joblib")
    write_meta(
        tmp_path / META_FILENAME,
        {"model_name": "Test", "version": "42", "run_id": "abc", "decision_threshold": 0.3},
    )
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    monkeypatch.setenv("PREDICTION_LOG", str(tmp_path / "predictions.jsonl"))
    app_module._state.update(model=None, meta={}, loaded_from=None, mtime=None)
    return TestClient(app_module.app)


def test_health_exposes_model_identity(client):
    client.post("/predict", json=EXAMPLE)  # triggers lazy load
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "42" and body["run_id"] == "abc"
    assert body["decision_threshold"] == 0.3


def test_predict_uses_threshold_and_logs(client, tmp_path):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {">50K", "<=50K"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["threshold"] == 0.3
    assert body["label"] == (">50K" if body["score"] >= 0.3 else "<=50K")
    assert body["model_version"] == "42" and body["run_id"] == "abc"
    lines = (tmp_path / "predictions.jsonl").read_text().strip().splitlines()
    rec = json.loads(lines[-1])
    assert rec["features"]["age"] == EXAMPLE["age"] and rec["label"] == body["label"]


def test_metrics_endpoint(client):
    client.post("/predict", json=EXAMPLE)
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "predictions_total" in r.text and "prediction_latency_seconds" in r.text


def test_predict_validation_error(client):
    r = client.post("/predict", json={**EXAMPLE, "age": -5})
    assert r.status_code == 422


def test_optional_fields_can_be_omitted(client):
    r = client.post("/predict", json={"age": 30, "education_num": 10})
    assert r.status_code == 200


def test_missing_model_returns_503(monkeypatch):
    monkeypatch.setenv("MODEL_PATH", "/nonexistent/model.joblib")
    app_module._state.update(model=None, meta={}, loaded_from=None, mtime=None)
    r = TestClient(app_module.app).post("/predict", json=EXAMPLE)
    assert r.status_code == 503
