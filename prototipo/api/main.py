"""API de inferencia PulseML v2: riesgo de incumplimiento de SLA por incidente.

Autenticación: encabezado X-API-Key comparado con la variable de entorno PULSEML_API_KEY.
"""
import json
import os
import secrets
from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from drift import drift_report

ART = Path(os.getenv("PULSEML_ARTIFACTS", Path(__file__).resolve().parents[1] / "model" / "artifacts"))
model = joblib.load(ART / "model.joblib") if (ART / "model.joblib").exists() else None
meta = json.loads((ART / "metadata.json").read_text(encoding="utf-8")) if (ART / "metadata.json").exists() else {}
reference = json.loads((ART / "reference.json").read_text(encoding="utf-8")) if (ART / "reference.json").exists() else {}

app = FastAPI(title="PulseML Inference API", version="2.0.0",
              description="Predice el riesgo de que un incidente de TI incumpla su SLA.")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_key(key: Optional[str] = Security(api_key_header)):
    expected = os.getenv("PULSEML_API_KEY")
    if not expected or not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status_code=401, detail="API key inválida o ausente")


class Incident(BaseModel):
    contact_type: str = Field(examples=["Phone"])
    location: Optional[str] = Field(None, examples=["Location 143"])
    category: Optional[str] = Field(None, examples=["Category 55"])
    subcategory: Optional[str] = Field(None, examples=["Subcategory 170"])
    u_symptom: Optional[str] = Field(None, examples=["Symptom 72"])
    impact: str = Field(examples=["2 - Medium"])
    urgency: str = Field(examples=["2 - Medium"])
    priority: str = Field(examples=["3 - Moderate"])
    assignment_group: Optional[str] = Field(None, examples=["Group 56"])
    opened_by: Optional[str] = Field(None, examples=["Opened by 8"])
    opened_hour: int = Field(ge=0, le=23, examples=[10])
    opened_weekday: int = Field(ge=0, le=6, examples=[1])
    knowledge: int = Field(ge=0, le=1, examples=[0])
    u_priority_confirmation: int = Field(ge=0, le=1, examples=[0])
    notify_email: int = Field(ge=0, le=1, examples=[0])


class Prediction(BaseModel):
    probabilidad_incumplimiento: float
    riesgo: str
    umbral: float
    modelo_version: str


def _frame(items: List[Incident]) -> pd.DataFrame:
    return pd.DataFrame([i.model_dump() for i in items])[meta["features"]]


def _loaded():
    if model is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")


@app.get("/health")
def health():
    return {"status": "ok" if model is not None else "degradado", "modelo_cargado": model is not None,
            "modelo_version": meta.get("version"), "metricas_validacion": meta.get("metricas")}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_key)])
def predict(item: Incident):
    _loaded()
    p = float(model.predict_proba(_frame([item]))[0, 1])
    thr = meta["umbral"]
    return Prediction(probabilidad_incumplimiento=round(p, 4), riesgo="alto" if p >= thr else "bajo",
                      umbral=thr, modelo_version=meta["version"])


@app.post("/predict/batch", response_model=List[Prediction], dependencies=[Depends(require_key)])
def predict_batch(items: List[Incident]):
    _loaded()
    ps = model.predict_proba(_frame(items))[:, 1]
    thr = meta["umbral"]
    return [Prediction(probabilidad_incumplimiento=round(float(p), 4), riesgo="alto" if p >= thr else "bajo",
                       umbral=thr, modelo_version=meta["version"]) for p in ps]


@app.post("/monitor/drift", dependencies=[Depends(require_key)])
def monitor_drift(items: List[Incident]):
    if len(items) < 100:
        raise HTTPException(status_code=422, detail="Se requieren al menos 100 incidentes para estimar drift")
    return drift_report(reference, _frame(items))
