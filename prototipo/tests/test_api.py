import pandas as pd

from conftest import HEADERS, payload


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["modelo_cargado"] is True
    assert body["metricas_validacion"]["roc_auc"] >= 0.75


def test_predict_sin_clave_rechazado(client, incidentes):
    r = client.post("/predict", json=payload(incidentes.iloc[0]))
    assert r.status_code == 401


def test_predict_clave_incorrecta_rechazado(client, incidentes):
    r = client.post("/predict", json=payload(incidentes.iloc[0]), headers={"X-API-Key": "otra"})
    assert r.status_code == 401


def test_predict_valido(client, incidentes):
    r = client.post("/predict", json=payload(incidentes.iloc[0]), headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probabilidad_incumplimiento"] <= 1.0
    assert body["riesgo"] in {"alto", "bajo"}


def test_predict_invalido_422(client, incidentes):
    bad = payload(incidentes.iloc[0]) | {"opened_hour": 30}
    assert client.post("/predict", json=bad, headers=HEADERS).status_code == 422


def test_categoria_no_vista_no_falla(client, incidentes):
    item = payload(incidentes.iloc[0]) | {"category": "Category 999", "location": None}
    assert client.post("/predict", json=item, headers=HEADERS).status_code == 200


def test_batch_coincide_con_individual(client, incidentes):
    rows = [payload(incidentes.iloc[i]) for i in range(5)]
    batch = client.post("/predict/batch", json=rows, headers=HEADERS).json()
    single = [client.post("/predict", json=r, headers=HEADERS).json() for r in rows]
    assert [b["probabilidad_incumplimiento"] for b in batch] == [s["probabilidad_incumplimiento"] for s in single]


def test_drift_detecta_periodo_posterior(client, incidentes):
    mayo = incidentes[incidentes["opened_at"] >= pd.Timestamp("2016-05-01")].sample(500, random_state=42)
    r = client.post("/monitor/drift", json=[payload(x) for _, x in mayo.iterrows()], headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["alerta"] is True
    assert body["nivel"]["knowledge"] == "significativo"


def test_drift_estable_en_entrenamiento(client, incidentes):
    abril = incidentes[incidentes["opened_at"] < pd.Timestamp("2016-05-01")].sample(500, random_state=42)
    body = client.post("/monitor/drift", json=[payload(x) for _, x in abril.iterrows()], headers=HEADERS).json()
    assert body["nivel"]["priority"] == "estable"
    assert body["psi"]["knowledge"] < 0.1


def test_drift_lote_chico_422(client, incidentes):
    rows = [payload(incidentes.iloc[i]) for i in range(10)]
    assert client.post("/monitor/drift", json=rows, headers=HEADERS).status_code == 422
