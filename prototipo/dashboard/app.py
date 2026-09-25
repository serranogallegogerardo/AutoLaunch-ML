"""Dashboard PulseML v2: consulta de riesgo por incidente y monitoreo de drift.

Todos los valores mostrados provienen de la API o de los archivos de resultados
generados por los scripts de análisis; no hay métricas fijas en el código.
"""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
API_KEY = os.getenv("PULSEML_API_KEY", "")
RESULTS = Path(os.getenv("PULSEML_RESULTS", Path(__file__).resolve().parents[2] / "results"))
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title="PulseML | Riesgo de SLA", layout="wide")
st.title("PulseML: riesgo de incumplimiento de SLA")
st.caption("Prototipo académico para TechServe Solutions S.A. (organización modelada). "
           "Datos: registro público de incidentes de la UCI.")


def get_health():
    try:
        return requests.get(f"{API_URL}/health", timeout=5).json()
    except requests.RequestException:
        return None


health = get_health()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Estado de la API", "En línea" if health else "Sin conexión")
if health:
    m = health.get("metricas_validacion") or {}
    c2.metric("Versión del modelo", health.get("modelo_version") or "-")
    c3.metric("ROC-AUC (prueba temporal)", f"{m.get('roc_auc', 0):.3f}")
    c4.metric("PR-AUC (prueba temporal)", f"{m.get('pr_auc', 0):.3f}")

tab_inf, tab_mon = st.tabs(["Consulta de incidente", "Monitoreo de drift"])

with tab_inf:
    st.subheader("Datos del incidente al momento del registro")
    with st.form("incidente"):
        a, b, c = st.columns(3)
        item = {
            "contact_type": a.selectbox("Canal", ["Phone", "Self service", "Email", "IVR", "Direct opening"]),
            "impact": a.selectbox("Impacto", ["1 - High", "2 - Medium", "3 - Low"], index=1),
            "urgency": a.selectbox("Urgencia", ["1 - High", "2 - Medium", "3 - Low"], index=1),
            "priority": a.selectbox("Prioridad", ["1 - Critical", "2 - High", "3 - Moderate", "4 - Low"], index=2),
            "category": b.text_input("Categoría", "Category 26"),
            "subcategory": b.text_input("Subcategoría", "Subcategory 174"),
            "u_symptom": b.text_input("Síntoma", "Symptom 491"),
            "location": b.text_input("Ubicación", "Location 204"),
            "assignment_group": c.text_input("Grupo asignado", "Group 70"),
            "opened_by": c.text_input("Registrado por", "Opened by 17"),
            "opened_hour": c.slider("Hora de apertura", 0, 23, 10),
            "opened_weekday": c.slider("Día de la semana (0 = lunes)", 0, 6, 1),
            "knowledge": int(c.checkbox("Usó base de conocimiento")),
            "u_priority_confirmation": int(c.checkbox("Prioridad confirmada")),
            "notify_email": 0,
        }
        enviar = st.form_submit_button("Calcular riesgo")
    if enviar:
        item = {k: (v or None) if isinstance(v, str) else v for k, v in item.items()}
        try:
            r = requests.post(f"{API_URL}/predict", json=item, headers=HEADERS, timeout=10)
            if r.ok:
                d = r.json()
                st.metric("Probabilidad de incumplir el SLA", f"{100 * d['probabilidad_incumplimiento']:.1f} %")
                (st.error if d["riesgo"] == "alto" else st.success)(
                    f"Riesgo {d['riesgo']} (umbral {d['umbral']:.2f}, modelo {d['modelo_version']})")
            else:
                st.error(f"La API respondió {r.status_code}: {r.text}")
        except requests.RequestException as e:
            st.error(f"Error de conexión: {e}")

with tab_mon:
    st.subheader("Estabilidad de variables: entrenamiento frente a periodo posterior")
    path = RESULTS / "drift.csv"
    if path.exists():
        dd = pd.read_csv(path)
        st.bar_chart(dd.set_index("variable")["psi"].sort_values(ascending=False))
        st.dataframe(dd[["variable", "psi", "nivel", "ks", "ks_p"]], hide_index=True, width="stretch")
        st.caption("PSI < 0,10 estable; 0,10 a 0,25 moderado; ≥ 0,25 significativo (requiere revisar o reentrenar).")
        alertas = dd[dd["nivel"] == "significativo"]["variable"].tolist()
        if alertas:
            st.warning("Drift significativo en: " + ", ".join(alertas))
    else:
        st.info("Ejecute scripts/06_drift.py para generar results/drift.csv.")
    ev = RESULTS / "evaluacion.json"
    if ev.exists():
        mens = pd.DataFrame(json.loads(ev.read_text(encoding="utf-8"))["desempeno_mensual"])
        st.subheader("Desempeño mensual en el periodo de prueba")
        st.dataframe(mens, hide_index=True)
