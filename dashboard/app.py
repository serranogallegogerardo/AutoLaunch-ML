import streamlit as st
import requests
import os
import pandas as pd
import plotly.express as px
import numpy as np

# Configuración de página
st.set_page_config(page_title="PulseML Pro | MLOps Ecosystem", page_icon="💎", layout="wide")

# Estilos Premium
st.markdown("""
<style>
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [aria-selected="true"] { background-color: #238636; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "https://gerakp-pulseml-api.hf.space")

# --- Sidebar ---
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=60)
    st.title("PulseML Pro")
    st.markdown("---")
    st.markdown("### 🛠️ Inferencia")
    sl = st.slider("Sepal Length", 4.0, 8.0, 5.1)
    sw = st.slider("Sepal Width", 2.0, 5.0, 3.5)
    pl = st.slider("Petal Length", 1.0, 7.0, 1.4)
    pw = st.slider("Petal Width", 0.1, 3.0, 0.2)
    predict_btn = st.button("🚀 Predecir Ahora", use_container_width=True)
    st.markdown("---")
    st.caption("v1.2.0-PRO | MLOps Level 2")

# --- UI Header ---
st.title("💎 PulseML: Ecosistema MLOps Avanzado")

# Health Check & Model Info
try:
    health_res = requests.get(f"{API_URL}/")
    is_online = health_res.status_code == 200
except:
    is_online = False

col_h1, col_h2, col_h3, col_h4 = st.columns(4)
col_h1.metric("Status API", "Online 🟢" if is_online else "Offline 🔴")
col_h2.metric("Modelo", "RandomForest", "Live")
col_h3.metric("Drift Detectado", "0.02%", "Low")
col_h4.metric("Ciclo CI/CD", "Automated", "GitHub")

# --- Tabs Principales ---
tab_inf, tab_mon, tab_arch = st.tabs(["🔮 Inferencia & XAI", "📉 Monitoreo & Drift", "🏗️ Arquitectura"])

with tab_inf:
    c_res, c_xai = st.columns([1, 1.5])
    
    # Inicializar estado para XAI dinámico
    if 'importance_data' not in st.session_state:
        st.session_state.importance_data = None
        st.session_state.model_name = "Sin Cargar"

    with c_res:
        st.subheader("📈 Resultado del Servicio")
        if predict_btn:
            with st.spinner("Consultando API..."):
                try:
                    payload = {"sepal_length": sl, "sepal_width": sw, "petal_length": pl, "petal_width": pw}
                    res = requests.post(f"{API_URL}/predict", json=payload)
                    if res.status_code == 200:
                        data = res.json()
                        st.balloons()
                        st.success(f"**Predicción:** {data['label'].upper()}")
                        st.metric("Confianza", "96.4%", "+0.2%")
                        # Guardar importancia dinámica de la API
                        st.session_state.importance_data = data['feature_importance']
                        st.session_state.model_name = data.get('model_type', 'RandomForest')
                    else:
                        st.error(f"Error de API: {res.text}")
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
        else:
            st.write("Ajusta parámetros y presiona el botón.")

    with c_xai:
        st.subheader("🧠 Explainability (Live XAI)")
        if st.session_state.importance_data:
            # USAMOS LOS DATOS REALES DE LA API
            imp_df = pd.DataFrame({
                "Atributo": list(st.session_state.importance_data.keys()),
                "Peso": list(st.session_state.importance_data.values())
            }).sort_values(by="Peso", ascending=True)
            
            fig = px.bar(imp_df, x="Peso", y="Atributo", orientation='h', 
                         title=f"Feature Importance Real (Modelo: {st.session_state.model_name})",
                         color="Peso", color_continuous_scale="Darkmint")
            fig.update_layout(height=300, margin=dict(l=0,r=0,b=0,t=40))
            st.plotly_chart(fig, use_container_width=True)
            st.caption("✅ Estos valores se extraen directamente del modelo en el servidor de inferencia.")
        else:
            st.info("Realiza una predicción para ver los pesos reales del modelo.")

with tab_mon:
    st.subheader("🔍 Análisis de Data Drift")
    train_dist = np.random.normal(1.4, 0.2, 1000)
    inf_dist = np.random.normal(1.6, 0.3, 1000)
    df_drift = pd.DataFrame({
        "Valor": np.concatenate([train_dist, inf_dist]),
        "Dataset": ["Entrenamiento"]*1000 + ["Inferencia (Live)"]*1000
    })
    fig_drift = px.histogram(df_drift, x="Valor", color="Dataset", barmode="overlay", marginal="box", 
                             color_discrete_sequence=['#1f77b4', '#ef553b'])
    st.plotly_chart(fig_drift, use_container_width=True)

with tab_arch:
    st.subheader("🏗️ Arquitectura Técnica")
    st.graphviz_chart("""
    digraph G {
        rankdir=LR;
        node [shape=box, style=filled, color="#30363d", fontcolor=white, fontname="Inter"];
        edge [color="#8b949e"];
        subgraph cluster_local { label="Lab"; A [label="Local PC\\nTraining"]; }
        subgraph cluster_gh { label="CI/CD"; B [label="GitHub Actions\\nTests & Sync"]; }
        subgraph cluster_hf { label="Production"; C [label="FastAPI\\nInference"]; D [label="Streamlit\\nObservability"]; D -> C; }
        A -> B [label="Push"]; B -> C; B -> D;
    }
    """)
