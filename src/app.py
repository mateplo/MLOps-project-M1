"""FastAPI microservice serving the exported model.

- /health   : status + model identity (registry version, run_id, threshold, data fingerprint)
- /predict  : score one person; every call is appended to a JSONL prediction log for drift analysis
- /metrics  : Prometheus exposition (request count, latency, score distribution)

    uvicorn src.app:app --reload
    curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/person.json
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src.meta import META_FILENAME, read_meta

app = FastAPI(title="Adult Income Classifier", version="2.0.0")

PREDICTIONS = Counter("predictions_total", "Predictions served", ["label", "model_version"])
LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1),
)
SCORES = Histogram(
    "prediction_score", "Predicted probability of >50K", buckets=[i / 10 for i in range(1, 10)]
)

_state: dict = {"model": None, "meta": {}, "loaded_from": None, "mtime": None}


def _model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", "artifacts/model.joblib"))


def _log_path() -> Optional[Path]:
    p = os.getenv("PREDICTION_LOG", "logs/predictions.jsonl")
    return Path(p) if p else None


def get_model():
    path = _model_path()
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Model not found at {path}. Run `make train` or `make export`.",
        )
    mtime = path.stat().st_mtime
    if _state["model"] is None or _state["loaded_from"] != str(path) or _state["mtime"] != mtime:
        _state["model"] = joblib.load(path)
        _state["meta"] = read_meta(path.parent / META_FILENAME)
        _state["loaded_from"] = str(path)
        _state["mtime"] = mtime
    return _state["model"], _state["meta"]


class Person(BaseModel):
    age: int = Field(..., ge=0, le=120, examples=[37])
    workclass: Optional[str] = Field(None, examples=["Private"])
    fnlwgt: Optional[int] = Field(
        None, ge=0, examples=[284582], description="ignored by the model, kept for schema compat"
    )
    education_num: int = Field(..., ge=1, le=16, examples=[13])
    marital_status: Optional[str] = Field(None, examples=["Married-civ-spouse"])
    occupation: Optional[str] = Field(None, examples=["Exec-managerial"])
    relationship: Optional[str] = Field(None, examples=["Husband"])
    race: Optional[str] = Field(None, examples=["White"])
    sex: Optional[str] = Field(None, examples=["Male"])
    capital_gain: int = Field(0, ge=0)
    capital_loss: int = Field(0, ge=0)
    hours_per_week: int = Field(40, ge=1, le=99)
    native_country: Optional[str] = Field(None, examples=["United-States"])


class Prediction(BaseModel):
    label: str
    score: float
    threshold: float
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    run_id: Optional[str] = None


def _append_log(record: dict) -> None:
    path = _log_path()
    if not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        pass  # never fail a prediction because of logging


@app.on_event("startup")
def preload_model() -> None:
    """Load the model eagerly so /health reflects reality and the first request is fast."""
    try:
        get_model()
    except HTTPException:
        pass  # no model yet: /predict answers 503 until `make train` / `make export`


@app.get("/metrics", include_in_schema=False)
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/health")
def health():
    try:
        _, meta = get_model()  # picks up a freshly exported model (hot reload)
    except HTTPException:
        meta = {}
    return {
        "status": "ok",
        "model_loaded": _state["model"] is not None,
        "model_path": str(_model_path()),
        "model_name": meta.get("model_name"),
        "model_version": meta.get("version"),
        "alias": meta.get("alias"),
        "run_id": meta.get("run_id"),
        "decision_threshold": meta.get("decision_threshold"),
        "data_sha256": meta.get("data_sha256"),
        "trained_at": meta.get("trained_at"),
    }


@app.post("/predict", response_model=Prediction)
def predict(person: Person) -> Prediction:
    t0 = time.perf_counter()
    model, meta = get_model()
    threshold = float(meta.get("decision_threshold", 0.5))
    features = person.model_dump()
    X = pd.DataFrame([features])
    for c in X.columns:
        if pd.api.types.is_integer_dtype(X[c]):
            X[c] = X[c].astype("float64")
    score = float(model.predict_proba(X)[0, 1])
    label = ">50K" if score >= threshold else "<=50K"
    version = str(meta.get("version")) if meta.get("version") is not None else "unknown"

    latency = time.perf_counter() - t0
    PREDICTIONS.labels(label=label, model_version=version).inc()
    LATENCY.observe(latency)
    SCORES.observe(score)
    _append_log(
        {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_version": version,
            "run_id": meta.get("run_id"),
            "features": features,
            "score": round(score, 6),
            "label": label,
            "latency_ms": round(latency * 1000, 3),
        }
    )
    return Prediction(
        label=label,
        score=round(score, 4),
        threshold=round(threshold, 4),
        model_name=meta.get("model_name"),
        model_version=version,
        run_id=meta.get("run_id"),
    )
