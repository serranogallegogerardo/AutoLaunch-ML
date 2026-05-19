"""
dashboard/app.py
Dashboard de observabilidad MLOps con Streamlit.
Permite hacer inferencias contra la API y visualizar métricas de MLflow.
"""
import streamlit as st
import requests
import os

# --------------- Config ---------------
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="MLOps Dashboard - TFG", page_icon="📊", layout="wide")

st.title("📊 MLOps Dashboard Analítico — TFG")
st.caption("Interfaz de observabilidad sobre la API de inferencia y el tracking de modelos.")

# --------------- Sidebar: Inferencia ---------------
st.sidebar.header("🔮 Realizar Inferencia")
st.sidebar.write("Ajustá los parámetros del iris y hacé clic en **Predecir**.")

sl = st.sidebar.slider("Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1)
sw = st.sidebar.slider("Sepal Width (cm)", 2.0, 5.0, 3.5, 0.1)
pl = st.sidebar.slider("Petal Length (cm)", 1.0, 7.0, 1.4, 0.1)
pw = st.sidebar.slider("Petal Width (cm)", 0.1, 3.0, 0.2, 0.1)

if st.sidebar.button("🚀 Predecir"):
    payload = {
        "sepal_length": sl,
        "sepal_width": sw,
        "petal_length": pl,
        "petal_width": pw,
    }
    try:
        res = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        if res.status_code == 200:
            data = res.json()
            st.sidebar.success(
                f"Clase predicha: **{data['prediction']}** ({data['label']})"
            )
        else:
            st.sidebar.error(f"Error de la API: {res.status_code}")
    except requests.exceptions.ConnectionError:
        st.sidebar.error(
            f"No se pudo conectar a la API en {API_URL}. ¿Está corriendo?"
        )
    except Exception as e:
        st.sidebar.error(f"Error inesperado: {e}")

# --------------- Main area ---------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("🏥 Estado de la API")
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("✅ API activa y respondiendo")
        else:
            st.warning(f"⚠️ API respondió con código {health.status_code}")
    except Exception:
        st.error("❌ API no disponible")

with col2:
    st.subheader("📈 Métricas de MLflow")
    st.info(
        "Para ver el panel completo de MLflow, ejecutá en tu terminal:\n\n"
        "```\nmlflow ui --backend-store-uri sqlite:///mlruns.db\n```\n\n"
        "Luego abrí http://localhost:5000"
    )

st.divider()
st.subheader("🏗️ Arquitectura del Ecosistema")
st.markdown(
    """
    ```
    Developer  →  GitHub  →  GitHub Actions (CI/CD)
                                    │
                         ┌──────────┴──────────┐
                         │                     │
                    Tests (pytest)      Build Docker Image
                                              │
                                        Google Cloud Run
                                              │
                                    ┌─────────┴─────────┐
                                    │                   │
                               FastAPI             MLflow Tracking
                              (Inference)         (Params & Metrics)
                                    │
                              Streamlit Dashboard
                             (Observability UI)
    ```
    """
)
