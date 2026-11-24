"""Unsupervised anomaly detection on sensor features. An anomaly is NOT a confirmed failure."""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.svm import OneClassSVM

from ml.data.ai4i import FEATURES_CAT, FEATURES_NUM
from ml.features.engineering import build_preprocessor

COLS = FEATURES_NUM + FEATURES_CAT


def detectors(c=0.02, seed=42):
    return {
        "isolation_forest": IsolationForest(n_estimators=200, contamination=c, random_state=seed),
        "lof": LocalOutlierFactor(n_neighbors=35, contamination=c, novelty=True),
        "one_class_svm": OneClassSVM(nu=c, gamma="scale"),
    }


def fit(df_normal: pd.DataFrame, name="isolation_forest", contamination=0.02):
    return Pipeline([("prep", build_preprocessor()), ("det", detectors(contamination)[name])]).fit(df_normal[COLS])


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "anomaly_score": -model.decision_function(df[COLS]),
            "is_anomaly": model.predict(df[COLS]) == -1,
        },
        index=df.index,
    )
