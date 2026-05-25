import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px

# Configuración de página
st.set_page_config(
    page_title="PulseML | MLOps Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Estilos personalizados (Premium Dark Look)
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
</style>
""", unsafe_allow_html=True)

# URL de la API (desde variable de entorno o local)
API_URL = os.getenv("API_URL", "https://gerakp-pulseml-api.hf.space")

# --- Barra Lateral ---
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=50)
    st.title("PulseML Ops")
    st.markdown("---")
    st.subheader("🔮 Panel de Inferencia")
    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.1)
    sw = st.slider("Sepal Width (cm)", 2.0, 5.0, 3.5)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 1.4)
    pw = st.slider("Petal Width (cm)", 0.1, 3.0, 0.2)
    
    predict_btn = st.button("🚀 Predecir Ahora", use_container_width=True)
    st.markdown("---")
    st.info("Este modelo clasifica flores Iris basándose en 4 medidas.")

# --- Cabecera Principal ---
st.title("📊 PulseML: Ecosistema MLOps End-to-End")
st.markdown("### Observabilidad del Modelo & Inferencia en Tiempo Real")

# Health Check de la API
try:
    health_res = requests.get(f"{API_URL}/")
    is_online = health_res.status_code == 200
except:
    is_online = False

# Fila de Métricas Superiores
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Estado Inferencia", "Online 🟢" if is_online else "Offline 🔴")
with m2:
    st.metric("Modelo Activo", "RandomForest", "v1.2.0")
with m3:
    st.metric("Tracking Backend", "MLflow", "SQLite")
with m4:
    st.metric("Provider", "HF Spaces", "Docker")

st.markdown("---")

# --- Resultados y Explicabilidad ---
col_res, col_xai = st.columns([1, 1.5])

with col_res:
    st.subheader("📈 Resultado del Servicio")
    if predict_btn:
        with st.spinner("Consultando API..."):
            try:
                payload = {
                    "sepal_length": sl, "sepal_width": sw,
                    "petal_length": pl, "petal_width": pw
                }
                res = requests.post(f"{API_URL}/predict", json=payload)
                if res.status_code == 200:
                    data = res.json()
                    st.balloons()
                    st.success(f"### Predicción: {data['label'].capitalize()}")
                    st.code(f"Class ID: {data['prediction']}", language="bash")
                else:
                    st.error(f"Error {res.status_code}: {res.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")
    else:
        st.write("Ajusta los sliders a la izquierda y presiona 'Predecir'.")

with col_xai:
    st.subheader("🧠 Explainability (XAI)")
    # Simulamos importancia de features basada en el entrenamiento local
    importance_data = {
        "Atributo": ["Petal Width", "Petal Length", "Sepal Length", "Sepal Width"],
        "Importancia": [0.45, 0.42, 0.10, 0.03]
    }
    df_imp = pd.DataFrame(importance_data)
    fig = px.bar(df_imp, x="Importancia", y="Atributo", orientation='h',
                 color="Importancia", color_continuous_scale="Viridis",
                 title="Feature Importance (RandomForest)")
    fig.update_layout(height=300, showlegend=False, margin=dict(l=0,r=0,b=0,t=40))
    st.plotly_chart(fig, use_container_width=True)

# --- Arquitectura del Sistema ---
st.markdown("---")
with st.expander("🏗️ Ver Arquitectura Técnica del Ecosistema"):
    st.markdown("""
    ```mermaid
    graph LR
        subgraph LocalLab [Local Lab / PC]
            A[Entrenamiento<br/>RandomForest] --> B[MLflow<br/>Tracking]
            A --> C[Export Joblib]
        end
        
        subgraph Cloud [Hugging Face Spaces]
            D[PulseML API<br/>FastAPI]
            E[PulseML Dashboard<br/>Streamlit]
            E -- JSON/HTTPS --> D
        end
        
        C -- Upload --> D
    ```
    """)
    st.info("Esta arquitectura demuestra un ciclo de vida MLOps completo: Tracking en local -> Empaquetado -> Despliegue en Cloud (Inferencia y UI).")
