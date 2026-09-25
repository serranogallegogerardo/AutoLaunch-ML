"""Comprensión de datos: perfil del log crudo y de la tabla por incidente."""
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as P  # noqa: E402

RES = P.ROOT / "results"
RES.mkdir(exist_ok=True)

ev = P.load_events()
miss = ev.isna().mean().sort_values(ascending=False)
miss.rename("proporcion_faltante").to_csv(RES / "faltantes_eventos.csv")
pd.DataFrame([{"atributo": c, "tipo": str(ev[c].dtype), "cardinalidad": int(ev[c].nunique()),
               "faltantes_pct": round(100 * ev[c].isna().mean(), 1),
               "ejemplo": str(ev[c].dropna().iloc[0]) if ev[c].notna().any() else ""}
              for c in ev.columns]).to_csv(RES / "diccionario_crudo.csv", index=False)

inc = P.build_incidents(ev)
inc.to_csv(RES / "incidentes_preparados.csv", index=False)
rh = inc["resolution_hours"]
q1, q3 = rh.quantile([0.25, 0.75])
hi = q3 + 1.5 * (q3 - q1)
monthly = inc.set_index("opened_at").resample("MS")[P.TARGET].agg(["size", "mean"])
monthly.to_csv(RES / "volumen_mensual.csv")

out = {
    "crudo": {
        "filas_eventos": int(ev.shape[0]), "atributos": int(ev.shape[1]),
        "incidentes_unicos": int(ev["number"].nunique()),
        "filas_duplicadas_exactas": int(ev.duplicated().sum()),
        "tamano_mb": round(P.RAW.stat().st_size / 1e6, 1),
        "estados": {k: int(v) for k, v in ev["incident_state"].value_counts().items()},
        "atributos_con_faltantes": int((miss > 0).sum()),
        "atributos_mas_90pct_faltantes": miss[miss > 0.9].index.tolist(),
    },
    "incidentes": {
        "n": int(len(inc)), "tasa_incumplimiento_sla": round(float(inc[P.TARGET].mean()), 4),
        "desde": str(inc["opened_at"].min()), "hasta": str(inc["opened_at"].max()),
        "resolucion_horas_mediana": round(float(rh.median()), 1),
        "resolucion_horas_p90": round(float(rh.quantile(0.9)), 1),
        "resolucion_negativa": int((rh < 0).sum()), "sin_fecha_resolucion": int(rh.isna().sum()),
        "resolucion_outliers_iqr": int((rh > hi).sum()), "umbral_iqr_horas": round(float(hi), 1),
        "reasignaciones_media": round(float(inc["reassignment_count"].mean()), 2),
        "reaperturas_pct": round(float((inc["reopen_count"] > 0).mean()), 4),
        "eventos_por_incidente_mediana": float(inc["n_events"].median()),
        "faltantes_predictoras": {k: round(float(v), 4) for k, v in inc[P.FEATURES].isna().mean().items() if v > 0},
        "cardinalidad": {c: int(inc[c].nunique()) for c in P.CAT_FEATURES},
        "contact_type": {k: int(v) for k, v in inc["contact_type"].value_counts().items()},
    },
    "volumen_mensual": {k.strftime("%Y-%m"): [int(n), round(float(t), 3)]
                        for k, n, t in zip(monthly.index, monthly["size"], monthly["mean"])},
}
(RES / "perfil.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(out, indent=1, ensure_ascii=False))
