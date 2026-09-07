"""Minimal FastAPI microservice serving the trained pipeline.

uvicorn src.app:app --reload
curl -X POST localhost:8000/predict -H 'content-type: application/json' -d @examples/person.json
"""

from __future__ import annotations

import os
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
_model = None

app = FastAPI(title="Adult Income Classifier", version="1.0.0")


class Person(BaseModel):
    age: int = Field(..., ge=0, le=120, examples=[37])
    workclass: Optional[str] = Field(None, examples=["Private"])
    fnlwgt: int = Field(..., ge=0, examples=[284582])
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


def get_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=503, detail=f"Model not found at {MODEL_PATH}. Run `make train`."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None, "model_path": MODEL_PATH}


@app.post("/predict", response_model=Prediction)
def predict(person: Person) -> Prediction:
    model = get_model()
    X = pd.DataFrame([person.model_dump()])
    score = float(model.predict_proba(X)[0, 1])
    return Prediction(label=">50K" if score >= 0.5 else "<=50K", score=round(score, 4))
