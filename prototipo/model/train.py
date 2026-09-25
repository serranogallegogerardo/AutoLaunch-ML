"""Entrenamiento del modelo de riesgo de incumplimiento de SLA con registro en MLflow.

El modelo solo se publica en model/artifacts si supera la compuerta de calidad
(QUALITY_GATE). Si no la supera, el proceso termina con código 1 y el pipeline de
CI/CD se detiene.
"""
import json
import os
import sys
from pathlib import Path

import joblib
import mlflow
import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
import pipeline as P  # noqa: E402
from modelos import candidatos  # noqa: E402

ART = HERE / "artifacts"
ART.mkdir(exist_ok=True)
QUALITY_GATE = {"roc_auc_min": float(os.getenv("PULSEML_GATE_ROC_AUC", "0.75")), "pr_auc_min_sobre_azar": 2.0}
MODEL_NAME = "Gradient Boosting (HGB)"


def reference_profile(df):
    """Distribución de referencia de cada predictora para el monitoreo de drift."""
    ref = {}
    for c in P.FEATURES:
        s = df[c]
        if c in P.CAT_FEATURES or s.nunique() <= 10:
            ref[c] = {"tipo": "categorica",
                      "frecuencias": s.fillna("NA").astype(str).value_counts(normalize=True).head(30).round(6).to_dict()}
        else:
            ref[c] = {"tipo": "numerica", "valores": s.dropna().sample(min(2000, len(s)), random_state=P.SEED).tolist()}
    return ref


def main():
    inc = P.build_incidents(P.load_events())
    train, test = P.temporal_split(inc)
    Xtr, ytr, Xte, yte = train[P.FEATURES], train[P.TARGET], test[P.FEATURES], test[P.TARGET]
    model = candidatos()[MODEL_NAME]

    mlflow.set_tracking_uri(f"sqlite:///{(HERE.parent / 'mlruns.db').as_posix()}")
    mlflow.set_experiment("sla_breach_techserve")
    with mlflow.start_run(run_name="hgb_temporal_split") as run:
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        thr = 0.484  # máximo F1 en validación temporal interna (scripts/05_evaluacion.py)
        metrics = {"roc_auc": roc_auc_score(yte, p), "pr_auc": average_precision_score(yte, p),
                   "f1_umbral": f1_score(yte, (p >= thr).astype(int)), "tasa_test": float(yte.mean()),
                   "n_train": len(train), "n_test": len(test)}
        mlflow.log_params({"modelo": MODEL_NAME, "corte_temporal": str(P.CUTOFF.date()), "seed": P.SEED,
                           "umbral": thr, **{f"gate_{k}": v for k, v in QUALITY_GATE.items()}})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        gate_ok = (metrics["roc_auc"] >= QUALITY_GATE["roc_auc_min"] and
                   metrics["pr_auc"] >= QUALITY_GATE["pr_auc_min_sobre_azar"] * metrics["tasa_test"])
        mlflow.set_tag("quality_gate", "aprobado" if gate_ok else "rechazado")
        print(json.dumps({k: round(float(v), 4) for k, v in metrics.items()}, indent=1))
        if not gate_ok:
            print("Compuerta de calidad RECHAZADA: el modelo no se publica.")
            sys.exit(1)

        meta = {"modelo": MODEL_NAME, "version": run.info.run_id[:8], "umbral": thr,
                "features": P.FEATURES, "cat_features": P.CAT_FEATURES,
                "metricas": {k: round(float(v), 4) for k, v in metrics.items()}, "quality_gate": QUALITY_GATE}
        joblib.dump(model, ART / "model.joblib")
        (ART / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        (ART / "reference.json").write_text(json.dumps(reference_profile(Xtr), ensure_ascii=False), encoding="utf-8")
        mlflow.log_artifact(str(ART / "metadata.json"))
        print("Compuerta de calidad aprobada. Modelo publicado:", meta["version"])


if __name__ == "__main__":
    np.random.seed(P.SEED)
    main()
