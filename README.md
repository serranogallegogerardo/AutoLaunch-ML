# 🚀 MLOps TFG — Ecosistema Automatizado

> **Prototipo de ecosistema MLOps automatizado con despliegue continuo y dashboard de observabilidad analítica mediante MLflow y Streamlit.**

**Autor:** Gerardo Juan Martín Serrano Gallego  
**Carrera:** Licenciatura en Ciencia de Datos — Universidad Siglo 21  
**Tipo:** Prototipado Tecnológico

---

## 📁 Estructura del Proyecto

```
mlops-tfg/
├── model/
│   ├── train.py              # Entrenamiento + MLflow tracking
│   └── model.joblib           # Modelo serializado
├── api/
│   ├── main.py               # FastAPI inference API
│   ├── requirements.txt
│   └── Dockerfile
├── dashboard/
│   ├── app.py                # Streamlit observability dashboard
│   └── requirements.txt
├── mlflow/
│   └── config.py             # Configuración centralizada MLflow
├── tests/
│   └── test_api.py           # Tests unitarios
├── .github/workflows/
│   └── ci-cd.yml             # Pipeline CI/CD con GitHub Actions
├── .gitignore
├── requirements.txt           # Dependencias globales
└── README.md
```

## ⚡ Quick Start

```bash
# 1. Crear entorno virtual
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Entrenar el modelo (se loguea en MLflow)
python model/train.py

# 4. Levantar la API
uvicorn api.main:app --reload

# 5. En otra terminal, levantar el dashboard
streamlit run dashboard/app.py

# 6. (Opcional) Ver panel de MLflow
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## 🛠️ Stack Tecnológico

| Tecnología | Uso | Licencia |
|-----------|-----|----------|
| Python 3.9 | Lenguaje principal | PSF |
| scikit-learn | Modelo de ML | BSD-3 |
| FastAPI | API de inferencia | MIT |
| Docker | Contenerización | Apache 2.0 |
| GitHub Actions | CI/CD | GitHub |
| MLflow | Tracking de experimentos | Apache 2.0 |
| Streamlit | Dashboard de observabilidad | Apache 2.0 |
| GCP Cloud Run | Deploy en la nube | Google |
