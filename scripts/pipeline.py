"""Preparación de datos compartida entre el análisis y el prototipo.

Unidad de análisis: incidente. Predictoras: atributos disponibles al registrar el
ticket (primer evento del log). Objetivo: sla_breach = 1 si el incidente cerró sin
cumplir el SLA (made_sla del último evento = False).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "incident_event_log.csv"
SEED = 42
CUTOFF = pd.Timestamp("2016-05-01")

CAT_FEATURES = [
    "contact_type", "location", "category", "subcategory", "u_symptom",
    "impact", "urgency", "priority", "assignment_group", "opened_by",
]
NUM_FEATURES = ["opened_hour", "opened_weekday", "knowledge", "u_priority_confirmation", "notify_email"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "sla_breach"


def load_events(path=RAW):
    return pd.read_csv(path, na_values=["?"], low_memory=False)


def build_incidents(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev.sort_values(["number", "sys_mod_count"])
    first = ev.groupby("number").first()
    last = ev.groupby("number").last()
    inc = pd.DataFrame(index=first.index)
    for c in CAT_FEATURES:
        inc[c] = first[c]
    opened = pd.to_datetime(first["opened_at"], format="%d/%m/%Y %H:%M", errors="coerce")
    resolved = pd.to_datetime(last["resolved_at"], format="%d/%m/%Y %H:%M", errors="coerce")
    inc["opened_at"] = opened
    inc["opened_hour"] = opened.dt.hour
    inc["opened_weekday"] = opened.dt.weekday
    inc["knowledge"] = first["knowledge"].astype(int)
    inc["u_priority_confirmation"] = first["u_priority_confirmation"].astype(int)
    inc["notify_email"] = (first["notify"] == "Send Email").astype(int)
    inc["resolution_hours"] = (resolved - opened).dt.total_seconds() / 3600
    inc["reassignment_count"] = last["reassignment_count"]
    inc["reopen_count"] = last["reopen_count"]
    inc["n_events"] = ev.groupby("number").size()
    inc[TARGET] = (~last["made_sla"].astype(bool)).astype(int)
    return inc.reset_index()


def temporal_split(inc: pd.DataFrame):
    train = inc[inc["opened_at"] < CUTOFF].copy()
    test = inc[inc["opened_at"] >= CUTOFF].copy()
    return train, test


def make_preprocessor(kind="onehot"):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    if kind == "onehot":
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("ohe", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20, sparse_output=True)),
        ])
        num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:
        cat = Pipeline([
            ("imp", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-1, min_frequency=20)),
        ])
        num = Pipeline([("imp", SimpleImputer(strategy="median"))])
    return ColumnTransformer([("cat", cat, CAT_FEATURES), ("num", num, NUM_FEATURES)])


def psi(expected, actual, bins=10):
    """Population Stability Index sobre una variable numérica o categórica."""
    e, a = pd.Series(expected), pd.Series(actual)
    if not pd.api.types.is_numeric_dtype(e) or e.nunique() <= bins:
        e, a = e.astype(str), a.astype(str)
        cats = e.value_counts().index[:30]
        ep = e.where(e.isin(cats), "Otros").value_counts(normalize=True)
        ap = a.where(a.isin(cats), "Otros").value_counts(normalize=True)
    else:
        edges = np.unique(np.quantile(e.dropna(), np.linspace(0, 1, bins + 1)))
        edges[0], edges[-1] = -np.inf, np.inf
        ep = pd.cut(e, edges).value_counts(normalize=True, sort=False)
        ap = pd.cut(a, edges).value_counts(normalize=True, sort=False)
    idx = ep.index.union(ap.index)
    ep = ep.reindex(idx, fill_value=0) + 1e-4
    ap = ap.reindex(idx, fill_value=0) + 1e-4
    return float(((ap - ep) * np.log(ap / ep)).sum())
