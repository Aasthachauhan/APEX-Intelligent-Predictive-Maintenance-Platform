"""Input-drift monitoring: PSI (+ KS p-value for numeric features). Thresholds follow the common
industry convention (PSI<0.1 stable, 0.1-0.25 moderate, >=0.25 significant); they are heuristics,
not statistical guarantees. Drift is a *proposal* to investigate/retrain - never an automatic action."""

import numpy as np
import pandas as pd
from scipy import stats

PSI_WARN, PSI_DRIFT, MIN_ROWS = 0.1, 0.25, 100
_EPS = 1e-4


def _psi_from_props(r: np.ndarray, n: np.ndarray) -> float:
    r, n = np.clip(r, _EPS, None), np.clip(n, _EPS, None)
    return float(np.sum((n - r) * np.log(n / r)))


def psi(ref, new, bins: int = 10) -> float:
    ref = np.asarray(ref, float)
    new = np.asarray(new, float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    n = np.histogram(new, edges)[0] / max(len(new), 1)
    return _psi_from_props(r, n)


def _status(v: float) -> str:
    return "drift" if v >= PSI_DRIFT else "warn" if v >= PSI_WARN else "ok"


def drift_report(ref: pd.DataFrame, new: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> dict:
    if len(new) < MIN_ROWS:
        raise ValueError(f"Need at least {MIN_ROWS} rows for a meaningful drift estimate, got {len(new)}")
    feats: dict[str, dict] = {}
    for c in num_cols:
        col = pd.to_numeric(new[c], errors="coerce")
        miss = float(col.isna().mean())
        v = col.dropna()
        if len(v) < MIN_ROWS:
            feats[c] = {
                "psi": None,
                "status": "drift",
                "missing_rate": miss,
                "note": "too many missing/non-numeric values",
            }
            continue
        p = psi(ref[c], v)
        feats[c] = {
            "psi": p,
            "ks_pvalue": float(stats.ks_2samp(ref[c], v).pvalue),
            "status": _status(p),
            "missing_rate": miss,
        }
    for c in cat_cols:
        cats = sorted(set(ref[c].astype(str)) | set(new[c].astype(str)))
        r = ref[c].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        n = new[c].astype(str).value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
        p = _psi_from_props(r, n)
        unseen = sorted(set(new[c].astype(str)) - set(ref[c].astype(str)))
        feats[c] = {
            "psi": p,
            "status": _status(p),
            "unseen_categories": unseen,
            "missing_rate": float(new[c].isna().mean()),
        }
    n_drift = sum(f["status"] == "drift" for f in feats.values())
    overall = "drift" if n_drift else "warn" if any(f["status"] == "warn" for f in feats.values()) else "ok"
    return {
        "overall": overall,
        "n_rows": len(new),
        "n_features_drifted": n_drift,
        "features": feats,
        "retrain_recommended": n_drift >= 2,
        "note": "Heuristic PSI thresholds; a recommendation to investigate, not proof of model degradation.",
    }
