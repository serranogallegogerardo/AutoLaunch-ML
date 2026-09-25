"""Monitoreo: drift de datos (PSI, KS) y de predicción entre entrenamiento y periodo posterior."""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402
from plots import BLUE, plt, save  # noqa: E402

RES = P.ROOT / "results"
inc = pd.read_csv(RES / "incidentes_preparados.csv", parse_dates=["opened_at"])
train, test = P.temporal_split(inc)
model = joblib.load(RES / "modelo_seleccionado.joblib")["model"]


def nivel(v):
    return "estable" if v < 0.1 else "moderado" if v < 0.25 else "significativo"


rows = []
for c in P.FEATURES:
    if c in P.CAT_FEATURES:
        a = train[c].fillna("NA").astype(str).astype(object)
        b = test[c].fillna("NA").astype(str).astype(object)
        ks, pv = None, None
    else:
        a, b = train[c], test[c]
        r = ks_2samp(a, b); ks, pv = round(float(r.statistic), 4), float(r.pvalue)
    v = round(P.psi(a, b), 4)
    rows.append({"variable": c, "psi": v, "nivel": nivel(v), "ks": ks, "ks_p": pv})

s_tr = pd.Series(model.predict_proba(train[P.FEATURES])[:, 1])
s_te = pd.Series(model.predict_proba(test[P.FEATURES])[:, 1])
r = ks_2samp(s_tr, s_te)
v = round(P.psi(s_tr, s_te), 4)
rows.append({"variable": "score_predicho", "psi": v, "nivel": nivel(v),
             "ks": round(float(r.statistic), 4), "ks_p": float(r.pvalue)})
dd = pd.DataFrame(rows)
dd.to_csv(RES / "drift.csv", index=False)

d2 = dd.set_index("variable")["psi"].sort_values()
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.barh(d2.index, d2.values, color=["#C00000" if x >= 0.25 else "#ED7D31" if x >= 0.1 else BLUE for x in d2.values])
ax.axvline(0.1, color="grey", ls="--", lw=0.8); ax.axvline(0.25, color="grey", ls=":", lw=0.8)
ax.set_xlabel("PSI: entrenamiento (feb-abr 2016) frente a periodo posterior (may 2016 en adelante)")
ax.set_title("Estabilidad de variables y del score (PSI)"); save(fig, "fig_10_drift_psi.png")

out = {"umbral_psi": {"estable": "<0.10", "moderado": "0.10-0.25", "significativo": ">=0.25"},
       "variables": rows,
       "tasa_objetivo": {"train": round(float(train[P.TARGET].mean()), 4), "test": round(float(test[P.TARGET].mean()), 4)},
       "score_medio": {"train": round(float(s_tr.mean()), 4), "test": round(float(s_te.mean()), 4)}}
(RES / "drift.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=True))
