"""
api/main.py
API de inferencia con FastAPI.
Carga el modelo entrenado y expone un endpoint /predict.
"""
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(
    title="MLOps TFG Inference API",
    description="API de inferencia para el ecosistema MLOps del TFG",
    version="1.0.0",
)

# --------------- Carga del modelo ---------------
MODEL_PATH = os.getenv("MODEL_PATH", "model/model.joblib")

try:
    model = joblib.load(MODEL_PATH)
except Exception:
    model = None


# --------------- Schemas ---------------
class PredictionRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class PredictionResponse(BaseModel):
    prediction: int
    label: str


# --------------- Endpoints ---------------
LABEL_MAP = {0: "setosa", 1: "versicolor", 2: "virginica"}


@app.get("/")
def health_check():
    if model is None:
        return {"status": "warning", "message": "Model not loaded"}
    return {"status": "ok", "message": "MLOps API is running 🚀"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        return {"prediction": -1, "label": "error: model not loaded"}

    features = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]
    pred = int(model.predict(features)[0])

    return PredictionResponse(prediction=pred, label=LABEL_MAP.get(pred, "unknown"))
