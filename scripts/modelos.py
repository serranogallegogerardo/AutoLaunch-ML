"""Algoritmos candidatos con sus hiperparámetros."""
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import pipeline as P


def candidatos():
    return {
        "Regresión logística": Pipeline([
            ("prep", P.make_preprocessor("onehot")),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))]),
        "Random Forest": Pipeline([
            ("prep", P.make_preprocessor("onehot")),
            ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=3, n_jobs=-1,
                                           class_weight="balanced_subsample", random_state=P.SEED))]),
        "Gradient Boosting (HGB)": Pipeline([
            ("prep", P.make_preprocessor("ordinal")),
            ("clf", HistGradientBoostingClassifier(
                categorical_features=list(range(len(P.CAT_FEATURES))), learning_rate=0.06,
                max_iter=400, max_leaf_nodes=31, l2_regularization=1.0,
                class_weight="balanced", random_state=P.SEED))]),
    }
