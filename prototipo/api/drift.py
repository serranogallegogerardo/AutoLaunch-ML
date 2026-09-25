"""Cálculo de drift (PSI) de un lote de incidentes frente al perfil de entrenamiento."""
import numpy as np
import pandas as pd

EPS = 1e-4


def _psi(ep: pd.Series, ap: pd.Series) -> float:
    idx = ep.index.union(ap.index)
    ep = ep.reindex(idx, fill_value=0) + EPS
    ap = ap.reindex(idx, fill_value=0) + EPS
    return float(((ap - ep) * np.log(ap / ep)).sum())


def psi_feature(ref: dict, values: pd.Series, bins: int = 10) -> float:
    if ref["tipo"] == "categorica":
        ep = pd.Series(ref["frecuencias"], dtype=float)
        ep["Otros"] = max(0.0, 1.0 - ep.sum())
        s = values.fillna("NA").astype(str)
        s = s.where(s.isin(ep.index), "Otros")
        return _psi(ep, s.value_counts(normalize=True))
    base = pd.Series(ref["valores"], dtype=float)
    edges = np.unique(np.quantile(base, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ep = pd.cut(base, edges).value_counts(normalize=True, sort=False)
    ap = pd.cut(values.astype(float), edges).value_counts(normalize=True, sort=False)
    return _psi(ep, ap)


def nivel(v: float) -> str:
    return "estable" if v < 0.1 else "moderado" if v < 0.25 else "significativo"


def drift_report(reference: dict, batch: pd.DataFrame) -> dict:
    out = {c: round(psi_feature(reference[c], batch[c]), 4) for c in reference if c in batch}
    return {"n": int(len(batch)), "psi": out, "nivel": {c: nivel(v) for c, v in out.items()},
            "alerta": any(v >= 0.25 for v in out.values())}
