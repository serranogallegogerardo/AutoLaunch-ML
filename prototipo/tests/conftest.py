import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))
os.environ["PULSEML_API_KEY"] = "clave-de-prueba"

FEATURES = ["contact_type", "location", "category", "subcategory", "u_symptom", "impact", "urgency",
            "priority", "assignment_group", "opened_by", "opened_hour", "opened_weekday", "knowledge",
            "u_priority_confirmation", "notify_email"]


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    import main
    return TestClient(main.app)


@pytest.fixture(scope="session")
def incidentes():
    return pd.read_csv(ROOT.parent / "results" / "incidentes_preparados.csv", parse_dates=["opened_at"])


def payload(row):
    cat = {c: (None if pd.isna(row[c]) else str(row[c])) for c in FEATURES[:10]}
    return cat | {c: int(row[c]) for c in FEATURES[10:]}


HEADERS = {"X-API-Key": "clave-de-prueba"}
