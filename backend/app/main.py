import io
import threading
import time
import uuid
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app import security
from backend.app.core import config
from backend.app.db import (
    AuditLog,
    DriftReport,
    ModelVersion,
    Prediction,
    User,
    get_db,
    init_db,
)
from backend.app.services import registry
from backend.app.services.model_service import BOUNDS, NUMERIC, svc
from ml.data.ai4i import FEATURES_CAT, TARGET, from_frame
from ml.monitoring.drift import drift_report
from ml.training.train import run as train_run

app = FastAPI(title="APEX Predictive Maintenance API", version="0.2.0")
init_db()
if config.CORS_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_methods=["*"],
        allow_headers=["*"],
        allow_credentials=False,
    )
_hits: dict[str, deque] = defaultdict(deque)
_train_lock = threading.Lock()
SENSOR_COLS = [*NUMERIC, "type"]
TOKEN_TYPE = "bearer"  # noqa: S105  # nosec B105 - OAuth2 scheme name, not a credential


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    key = request.client.host if request.client else "unknown"
    q, t = _hits[key], time.time()
    while q and t - q[0] > 60:
        q.popleft()
    if len(q) >= config.RATE_LIMIT_PER_MIN:
        return JSONResponse({"detail": "Rate limit exceeded"}, 429)
    q.append(t)
    resp = await call_next(request)
    if not q:
        _hits.pop(key, None)
    return resp


@app.exception_handler(RequestValidationError)
async def validation_handler(_: Request, exc: RequestValidationError):
    # Do not echo `input` back: it can hold NaN/inf (not JSON-serialisable) or sensitive data.
    errs = [{"loc": e.get("loc"), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse({"detail": errs}, 422)


class Creds(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class Sensors(BaseModel):
    air_temp: float = Field(ge=BOUNDS["air_temp"][0], le=BOUNDS["air_temp"][1], description="K")
    process_temp: float = Field(ge=BOUNDS["process_temp"][0], le=BOUNDS["process_temp"][1], description="K")
    rpm: float = Field(ge=BOUNDS["rpm"][0], le=BOUNDS["rpm"][1])
    torque: float = Field(ge=BOUNDS["torque"][0], le=BOUNDS["torque"][1], description="Nm")
    tool_wear: float = Field(ge=BOUNDS["tool_wear"][0], le=BOUNDS["tool_wear"][1], description="min")
    type: str = Field(pattern="^[LMH]$")


class Refresh(BaseModel):
    refresh_token: str


class PromoteReq(BaseModel):
    force: bool = False
    reason: str = Field(default="", max_length=300)


def audit(db: Session, uid: int | None, action: str, detail: str = "") -> None:
    db.add(AuditLog(user_id=uid, action=action, detail=detail[:500]))
    db.commit()


def read_csv_upload(file: UploadFile, max_rows: int) -> pd.DataFrame:
    raw = file.file.read(config.MAX_UPLOAD_BYTES + 1)
    if len(raw) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(400, "Unreadable CSV") from None
    if len(df) > max_rows:
        raise HTTPException(413, f"Too many rows (max {max_rows})")
    df.columns = df.columns.astype(str).str.strip()
    return df


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------- auth
@app.post("/auth/register", status_code=201)
def register(c: Creds, db: Session = Depends(get_db)):
    email = c.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "Email already registered")
    role = "admin" if email in config.ADMIN_EMAILS else "user"  # allowlist, never "first user wins"
    u = User(email=email, password_hash=security.hash_pw(c.password), role=role)
    db.add(u)
    db.commit()
    audit(db, u.id, "register")
    return {"id": u.id, "email": u.email, "role": u.role}


@app.post("/auth/login")
def login(db: Session = Depends(get_db), form: OAuth2PasswordRequestForm = Depends()):
    u = db.scalar(select(User).where(User.email == form.username.lower()))
    if not security.verify_pw(form.password, u.password_hash if u else None) or not u:
        audit(db, None, "login_failed")
        raise HTTPException(401, "Invalid credentials")
    audit(db, u.id, "login")
    return {
        "access_token": security.make_token(u.id, "access"),
        "refresh_token": security.make_token(u.id, "refresh"),
        "token_type": TOKEN_TYPE,
    }


@app.post("/auth/refresh")
def refresh(r: Refresh, db: Session = Depends(get_db)):
    uid = security.decode(r.refresh_token, "refresh", db)
    return {
        "access_token": security.make_token(uid, "access"),
        "token_type": TOKEN_TYPE,
    }


@app.post("/auth/logout")
def logout(
    r: Refresh,
    db: Session = Depends(get_db),
    user: User = Depends(security.current_user),
):
    security.revoke(db, r.refresh_token)
    audit(db, user.id, "logout")
    return {
        "status": "refresh token revoked",
        "note": "Access tokens expire on their own (<=15 min).",
    }


# ---------------------------------------------------------------- model + prediction
@app.get("/model/info")
def model_info(_: User = Depends(security.current_user)):
    svc.ensure()
    m = svc.meta
    keys = (
        "model_name",
        "version",
        "trained_at",
        "data_source",
        "threshold",
        "selection_rule",
        "input_columns",
        "selected_test_metrics",
        "sha256",
    )
    return {k: m[k] for k in keys}


def _log(db: Session, user: User, df: pd.DataFrame, out: pd.DataFrame) -> list[str]:
    ids = [str(uuid.uuid4()) for _ in range(len(df))]
    rows = [
        {
            "id": i,
            "user_id": user.id,
            "model_version": svc.meta["version"],
            "inputs": {k: (v.item() if hasattr(v, "item") else v) for k, v in r.items()},
            "probability": float(p),
            "predicted_failure": int(f),
            "risk": k,
        }
        for i, r, p, f, k in zip(
            ids,
            df[SENSOR_COLS].to_dict("records"),
            out.failure_probability,
            out.predicted_failure,
            out.risk,
            strict=True,
        )
    ]
    db.bulk_insert_mappings(Prediction, rows)
    db.commit()
    return ids


@app.post("/predict")
def predict(
    s: Sensors,
    db: Session = Depends(get_db),
    user: User = Depends(security.current_user),
):
    df = pd.DataFrame([s.model_dump()])
    out = svc.predict_df(df)
    pid = _log(db, user, df, out)[0]
    o = out.iloc[0]
    return {
        "prediction_id": pid,
        "failure_risk_flag": int(o.predicted_failure),
        "failure_probability": float(o.failure_probability),
        "risk": o.risk,
        "health_index": round(1 - float(o.failure_probability), 4),
        "model_version": svc.meta["version"],
        "timestamp": pd.Timestamp.now("UTC").isoformat(),
        "note": "Model-estimated risk, not a confirmed failure. health_index = 1 - probability; it is an uncalibrated convenience score, not a physical measurement.",
    }


@app.post("/explain")
def explain(s: Sensors, _: User = Depends(security.current_user)):
    return {
        "top_contributions": svc.explain(pd.DataFrame([s.model_dump()])),
        "note": "SHAP values describe model behaviour, not physical causation.",
    }


@app.post("/anomaly")
def anomaly(s: Sensors, _: User = Depends(security.current_user)):
    r = svc.anomaly_df(pd.DataFrame([s.model_dump()])).iloc[0]
    return {
        "anomaly_score": float(r.anomaly_score),
        "is_anomaly": bool(r.is_anomaly),
        "note": "Unsupervised: unusual relative to normal training data. NOT a confirmed failure and distinct from /predict.",
    }


def valid_mask(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    num = df[NUMERIC].apply(pd.to_numeric, errors="coerce").astype(float)
    ok = pd.Series(np.isfinite(num.to_numpy()).all(axis=1), index=df.index)
    for c, (lo, hi) in BOUNDS.items():
        ok &= num[c].between(lo, hi)
    ok &= df["type"].isin(["L", "M", "H"])
    return num, ok


@app.post("/predict/batch")
def predict_batch(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(security.current_user),
):
    df = read_csv_upload(file, config.MAX_BATCH_ROWS)
    miss = [c for c in SENSOR_COLS if c not in df.columns]
    if miss:
        raise HTTPException(422, f"Missing columns: {miss}")
    num, ok = valid_mask(df)
    good = df[ok].copy()
    good[NUMERIC] = num[ok]
    bad = df[~ok]
    if len(good):
        out = svc.predict_df(good)
        _log(db, user, good, out)
        csv = good.join(out).to_csv(index=False)
    else:
        svc.ensure()
        csv = ""
    return {
        "n_ok": len(good),
        "n_failed": len(bad),
        "failed_rows": [int(i) for i in bad.index],
        "csv": csv,
        "failed_reason": f"non-finite, out of range {BOUNDS}, or type not in L/M/H",
    }


@app.get("/predictions")
def history(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(security.current_user),
):
    rows = db.scalars(
        select(Prediction).where(Prediction.user_id == user.id).order_by(Prediction.created_at.desc()).limit(max(1, min(limit, 500)))
    ).all()
    return [
        {
            "id": r.id,
            "probability": r.probability,
            "risk": r.risk,
            "model_version": r.model_version,
        }
        for r in rows
    ]


# ---------------------------------------------------------------- MLOps: drift
@app.post("/monitoring/drift")
def drift(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User = Depends(security.current_user),
):
    svc.ensure()
    df = read_csv_upload(file, config.MAX_BATCH_ROWS)
    miss = [c for c in SENSOR_COLS if c not in df.columns]
    if miss:
        raise HTTPException(422, f"Schema mismatch, missing columns: {miss}")
    ref = pd.DataFrame(svc.meta["reference_sample"])
    try:
        rep = drift_report(ref, df, NUMERIC, FEATURES_CAT)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    out = {"model_version": svc.meta["version"], **rep}
    db.add(
        DriftReport(
            user_id=user.id,
            model_version=svc.meta["version"],
            status=rep["overall"],
            report=out,
        )
    )
    db.commit()
    return out


@app.get("/monitoring/drift/history")
def drift_history(db: Session = Depends(get_db), user: User = Depends(security.current_user)):
    rows = db.scalars(select(DriftReport).where(DriftReport.user_id == user.id).order_by(DriftReport.created_at.desc()).limit(50)).all()
    return [
        {
            "id": r.id,
            "status": r.status,
            "model_version": r.model_version,
            "created_at": r.created_at.isoformat(),
        }
        for r in rows
    ]


# ---------------------------------------------------------------- MLOps: registry (admin only)
def _mv(m: ModelVersion) -> dict:
    return {
        "version": m.version,
        "status": m.status,
        "data_source": m.data_source,
        "metrics": m.metrics,
        "promoted_at": m.promoted_at.isoformat() if m.promoted_at else None,
    }


@app.get("/admin/models")
def list_models(db: Session = Depends(get_db), _: User = Depends(security.current_admin)):
    svc.ensure()  # triggers bootstrap if the registry is empty
    return [_mv(m) for m in db.scalars(select(ModelVersion).order_by(ModelVersion.created_at.desc()))]


@app.post("/admin/retrain")
def retrain(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    admin: User = Depends(security.current_admin),
):
    """Trains on uploaded data and registers a CANDIDATE. It never becomes active automatically."""
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(409, "A training job is already running")
    try:
        df = read_csv_upload(file, config.MAX_TRAIN_ROWS)
        try:
            data = from_frame(df)
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        if TARGET not in data or data[TARGET].isin([0, 1]).all() is False:
            raise HTTPException(422, "Target must be binary 0/1")
        tmp = config.REGISTRY_DIR / f"_tmp_{uuid.uuid4().hex}"
        try:
            manifest = train_run(data, tmp, f"UPLOAD:{(file.filename or 'data.csv')[:80]}")
        except ValueError as e:
            raise HTTPException(422, str(e)) from None
        final = config.REGISTRY_DIR / manifest["version"]
        tmp.rename(final)
        row = registry.register_candidate(db, final)
        audit(db, admin.id, "retrain", row.version)
        return {
            **_mv(row),
            "policy_failures": registry.policy_failures(row, registry.active(db)),
        }
    finally:
        _train_lock.release()


@app.post("/admin/models/{version}/promote")
def promote(
    version: str,
    req: PromoteReq,
    db: Session = Depends(get_db),
    admin: User = Depends(security.current_admin),
):
    row = registry.promote(db, version, req.force, req.reason)
    svc.load()
    audit(db, admin.id, "promote", f"{version} force={req.force} reason={req.reason}")
    return _mv(row)


@app.post("/admin/models/rollback")
def rollback(db: Session = Depends(get_db), admin: User = Depends(security.current_admin)):
    row = registry.rollback(db)
    svc.load()
    audit(db, admin.id, "rollback", row.version)
    return _mv(row)
