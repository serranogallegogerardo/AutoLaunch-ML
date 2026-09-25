"""Latencia de /predict en el Space público, medida desde el cliente (incluye la red)."""
import json
import os
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = "https://gerakp-pulseml-api.hf.space"
N = 50
req = json.loads((HERE.parent / "results" / "ejemplos_api.json").read_text(encoding="utf-8"))["predict_request"]
body = json.dumps(req).encode()
headers = {"X-API-Key": os.environ["PULSEML_API_KEY"], "Content-Type": "application/json"}

urllib.request.urlopen(f"{BASE}/health", timeout=120).read()
lat, errores = [], 0
for _ in range(N):
    t0 = time.perf_counter()
    try:
        urllib.request.urlopen(urllib.request.Request(f"{BASE}/predict", data=body, headers=headers), timeout=60).read()
    except Exception:
        errores += 1
        continue
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
out = {"url": BASE, "solicitudes": N, "errores": errores, "p50_ms": round(float(np.percentile(lat, 50)), 1),
       "p95_ms": round(float(np.percentile(lat, 95)), 1), "max_ms": round(float(lat.max()), 1),
       "nota": "Medida desde un cliente en Argentina; incluye la latencia de red hasta Hugging Face.",
       "fecha": time.strftime("%Y-%m-%d %H:%M")}
(HERE.parent / "results" / "latencia_nube.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
