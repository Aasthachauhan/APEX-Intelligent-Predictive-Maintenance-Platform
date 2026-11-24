"""Train + compare failure models. Selection: validation PR-AUC (not accuracy). Threshold: max F2
on validation (recall-weighted). The test split is only used for the final report."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from ml.anomaly_detection.detector import fit as fit_anomaly
from ml.data.ai4i import FEATURES_CAT, FEATURES_NUM, TARGET, load_real, simulate
from ml.features.engineering import build_preprocessor

SEED = 42
INPUT_COLS = FEATURES_NUM + FEATURES_CAT


def candidates(pos_weight: float):
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "decision_tree": DecisionTreeClassifier(max_depth=8, class_weight="balanced", random_state=SEED),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=SEED,
        ),
        "xgboost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.08,
            scale_pos_weight=pos_weight,
            eval_metric="aucpr",
            random_state=SEED,
            n_jobs=4,
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            class_weight="balanced",
            random_state=SEED,
            verbose=-1,
            n_jobs=4,
        ),
    }


def metrics(y, p, thr: float) -> dict:
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred)),
        "f1": float(f1_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "specificity": float(tn / (tn + fp)),
        "fpr": float(fp / (fp + tn)),
        "fnr": float(fn / (fn + tp)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def best_threshold(y, p) -> float:
    pr, rc, th = precision_recall_curve(y, p)
    if len(th) == 0:
        return 0.5
    f2 = (5 * pr[:-1] * rc[:-1]) / np.clip(4 * pr[:-1] + rc[:-1], 1e-12, None)
    return float(th[int(np.argmax(f2))])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(df: pd.DataFrame, out_dir: Path, data_source: str, use_mlflow: bool = False) -> dict:
    if len(df) < 200 or df[TARGET].nunique() < 2 or int(df[TARGET].sum()) < 20:
        raise ValueError("Insufficient data: need >=200 rows and >=20 positive examples")
    X, y = df[INPUT_COLS], df[TARGET].astype(int)
    Xtv, Xte, ytv, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    Xtr, Xva, ytr, yva = train_test_split(Xtv, ytv, test_size=0.25, stratify=ytv, random_state=SEED)
    pw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    results: dict[str, dict] = {}
    fitted: dict[str, tuple] = {}
    for name, est in candidates(pw).items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", est)]).fit(Xtr, ytr)
        pva = pipe.predict_proba(Xva)[:, 1]
        thr = best_threshold(yva, pva)
        results[name] = {
            "val": metrics(yva, pva, thr),
            "train_seconds": round(time.time() - t0, 2),
        }
        fitted[name] = (pipe, thr)
    best = max(results, key=lambda k: results[k]["val"]["pr_auc"])
    pipe, thr = fitted[best]
    for name, (p, t) in fitted.items():
        results[name]["test"] = metrics(yte, p.predict_proba(Xte)[:, 1], t)
    out_dir.mkdir(parents=True, exist_ok=True)
    art = out_dir / "model.joblib"
    joblib.dump(pipe, art)
    anom = out_dir / "anomaly.joblib"
    joblib.dump(fit_anomaly(Xtr[ytr == 0]), anom)
    now = datetime.now(timezone.utc)
    manifest = {
        "model_name": best,
        "version": now.strftime("%Y%m%d%H%M%S"),
        "trained_at": now.isoformat(),
        "data_source": data_source,
        "n_rows": len(df),
        "positive_rate": float(y.mean()),
        "threshold": thr,
        "selection_rule": "max validation PR-AUC; threshold maximises validation F2",
        "input_columns": INPUT_COLS,
        "seed": SEED,
        "sha256": sha256(art),
        "anomaly_sha256": sha256(anom),
        "reference_sample": Xtr.sample(min(1000, len(Xtr)), random_state=SEED).to_dict("list"),
        "results": results,
        "selected_test_metrics": results[best]["test"],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=float), encoding="utf-8")
    if use_mlflow:
        import mlflow

        mlflow.set_tracking_uri(f"sqlite:///{(out_dir.parent / 'mlflow.db').resolve()}")
        mlflow.set_experiment("apex-failure")
        for n, r in results.items():
            with mlflow.start_run(run_name=n):
                mlflow.log_params({"data_source": data_source, "seed": SEED})
                mlflow.log_metrics({f"val_{k}": v for k, v in r["val"].items()} | {f"test_{k}": v for k, v in r["test"].items()})
    return manifest


def load_model(model_dir: Path):
    """Loads only after sha256 verification (joblib = pickle: trusted artifacts only)."""
    m = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    if sha256(model_dir / "model.joblib") != m["sha256"]:
        raise RuntimeError("Model artifact integrity check failed")
    return joblib.load(model_dir / "model.joblib"), m


def load_anomaly(model_dir: Path, manifest: dict):
    """sha256-verified load of the anomaly detector saved with the model."""
    path = model_dir / "anomaly.joblib"
    if sha256(path) != manifest["anomaly_sha256"]:
        raise RuntimeError("Anomaly artifact integrity check failed")
    return joblib.load(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--out", default="artifacts/active")
    ap.add_argument("--mlflow", action="store_true")
    a = ap.parse_args()
    df, src = (
        (load_real(a.csv), f"REAL:{a.csv}")
        if a.csv
        else (
            simulate(10000, SEED),
            "SIMULATED (AI4I-rule simulator, NOT the UCI file)",
        )
    )
    m = run(df, Path(a.out), src, a.mlflow)
    print(
        json.dumps(
            {
                "model": m["model_name"],
                "source": m["data_source"],
                "threshold": m["threshold"],
            },
            indent=1,
        )
    )
    for n, r in m["results"].items():
        t = r["test"]
        print(
            f"{n:20s} val_prauc={r['val']['pr_auc']:.3f} | test P={t['precision']:.3f} R={t['recall']:.3f} F1={t['f1']:.3f} ROC={t['roc_auc']:.3f} PR={t['pr_auc']:.3f} FN={t['fn']} FP={t['fp']}"
        )
