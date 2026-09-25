"""Prueba de latencia local: levanta la API con uvicorn y envía N solicitudes HTTP secuenciales.

Mide el tiempo de ida y vuelta en la misma máquina (sin red externa). No representa
la latencia del despliegue en Hugging Face Spaces.
"""
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "tests"))
from conftest import FEATURES, payload  # noqa: E402

N, PORT, KEY = 500, 8765, "clave-latencia"
env = dict(os.environ, PULSEML_API_KEY=KEY)
srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
                       cwd=HERE / "api", env=env)
try:
    base = f"http://127.0.0.1:{PORT}"
    for _ in range(60):
        try:
            if httpx.get(f"{base}/health").status_code == 200:
                break
        except httpx.HTTPError:
            time.sleep(0.5)
    df = pd.read_csv(HERE.parent / "results" / "incidentes_preparados.csv").sample(N, random_state=42)
    rows = [payload(r) for _, r in df.iterrows()]
    lat, errors = [], 0
    with httpx.Client(base_url=base, headers={"X-API-Key": KEY}) as c:
        for r in rows[:20]:
            c.post("/predict", json=r)
        for r in rows:
            t0 = time.perf_counter()
            resp = c.post("/predict", json=r)
            lat.append((time.perf_counter() - t0) * 1000)
            errors += resp.status_code != 200
    lat = np.array(lat)
    out = {"solicitudes": N, "errores": int(errors), "p50_ms": round(float(np.percentile(lat, 50)), 1),
           "p95_ms": round(float(np.percentile(lat, 95)), 1), "p99_ms": round(float(np.percentile(lat, 99)), 1),
           "media_ms": round(float(lat.mean()), 1), "max_ms": round(float(lat.max()), 1),
           "entorno": f"{platform.system()} {platform.release()}, Python {platform.python_version()}, "
                      f"{os.cpu_count()} CPU lógicas, uvicorn 1 worker, localhost",
           "fecha": time.strftime("%Y-%m-%d %H:%M")}
    (HERE.parent / "results" / "latencia.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=1))
finally:
    srv.terminate()
    srv.wait(timeout=10)
