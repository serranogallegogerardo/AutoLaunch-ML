"""Genera ejemplos reales de solicitud y respuesta de la API para la documentación."""
import json
import os
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "api"))
sys.path.insert(0, str(HERE / "tests"))
os.environ["PULSEML_API_KEY"] = "clave-ejemplo"

from fastapi.testclient import TestClient  # noqa: E402

import main  # noqa: E402
from conftest import payload  # noqa: E402

os.environ["PULSEML_API_KEY"] = "clave-ejemplo"
c = TestClient(main.app)
H = {"X-API-Key": "clave-ejemplo"}
inc = pd.read_csv(HERE.parent / "results" / "incidentes_preparados.csv", parse_dates=["opened_at"])
req = {"contact_type": "Phone", "location": "Location 204", "category": "Category 26", "subcategory": "Subcategory 174",
       "u_symptom": "Symptom 491", "impact": "2 - Medium", "urgency": "2 - Medium", "priority": "3 - Moderate",
       "assignment_group": "Group 70", "opened_by": "Opened by 17", "opened_hour": 10, "opened_weekday": 1,
       "knowledge": 0, "u_priority_confirmation": 0, "notify_email": 0}
mayo = inc[inc["opened_at"] >= pd.Timestamp("2016-05-01")].sample(500, random_state=42)
out = {
    "health": c.get("/health").json(),
    "predict_request": req,
    "predict_response": c.post("/predict", json=req, headers=H).json(),
    "predict_sin_clave": {"status": c.post("/predict", json=req).status_code,
                          "body": c.post("/predict", json=req).json()},
    "predict_invalido": {"status": c.post("/predict", json=req | {"opened_hour": 30}, headers=H).status_code},
    "drift_mayo_500": c.post("/monitor/drift", json=[payload(r) for _, r in mayo.iterrows()], headers=H).json(),
}
(HERE.parent / "results" / "ejemplos_api.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
