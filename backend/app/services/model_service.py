"""Serves real predictions, anomaly scores and SHAP explanations from the registry's ACTIVE model."""

from pathlib import Path

import numpy as np
import pandas as pd
import shap
from fastapi import HTTPException

from backend.app.core import config
from backend.app.db import AuditLog, SessionLocal
from backend.app.services import registry
from ml.anomaly_detection.detector import score as anomaly_score
from ml.features.engineering import feature_names
from ml.training.train import INPUT_COLS, load_anomaly, load_model

NUMERIC = [c for c in INPUT_COLS if c != "type"]
BOUNDS = {
    "air_temp": (250, 400),
    "process_temp": (250, 420),
    "rpm": (1, 10000),
    "torque": (0, 500),
    "tool_wear": (0, 1000),
}


class ModelService:
    def __init__(self):
        self.pipe = None
        self.meta = None
        self.anom = None
        self._expl = None

    def load(self):
        with SessionLocal() as db:
            row = registry.active(db)
            if row is None and (config.MODEL_DIR / "manifest.json").exists():
                row = registry.bootstrap(db, config.MODEL_DIR)
                db.add(AuditLog(action="model_bootstrap", detail=row.version))
                db.commit()
            if row is None:
                raise FileNotFoundError("no active model in registry")
            d = Path(row.path)
        pipe, meta = load_model(d)
        self.anom = load_anomaly(d, meta)
        self.pipe, self.meta, self._expl = pipe, meta, None

    def ensure(self):
        if self.pipe is None:
            try:
                self.load()
            except (FileNotFoundError, RuntimeError, ValueError, KeyError) as e:
                raise HTTPException(503, f"Model unavailable: {e}") from None

    def risk(self, p: float) -> str:
        t = self.meta["threshold"]
        return "high" if p >= t else "medium" if p >= t / 2 else "low"

    def predict_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.ensure()
        p = self.pipe.predict_proba(df[INPUT_COLS])[:, 1]
        t = self.meta["threshold"]
        return pd.DataFrame(
            {
                "failure_probability": p,
                "predicted_failure": (p >= t).astype(int),
                "risk": [self.risk(x) for x in p],
            },
            index=df.index,
        )

    def anomaly_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.ensure()
        return anomaly_score(self.anom, df)

    def explain(self, row: pd.DataFrame, top: int = 5):
        self.ensure()
        clf = self.pipe.named_steps["clf"]
        if self._expl is None:
            self._expl = shap.TreeExplainer(clf) if hasattr(clf, "feature_importances_") else None
        if self._expl is None:
            raise HTTPException(501, "SHAP explanation implemented for tree models only")
        x = self.pipe.named_steps["prep"].transform(row[INPUT_COLS])
        sv = self._expl.shap_values(x)
        sv = sv[1] if isinstance(sv, list) else sv
        sv = np.asarray(sv).reshape(-1, len(feature_names()))[0]
        names = feature_names()
        return [
            {
                "feature": names[i],
                "shap_value": float(sv[i]),
                "direction": "raises risk" if sv[i] > 0 else "lowers risk",
            }
            for i in np.argsort(-np.abs(sv))[:top]
        ]


svc = ModelService()
