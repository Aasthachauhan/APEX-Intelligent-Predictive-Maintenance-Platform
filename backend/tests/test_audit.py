"""Regression tests for defects found in the audit + tests for registry / drift / anomaly / RBAC."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.app.main import _train_lock
from backend.app.services.model_service import svc
from ml.data.ai4i import simulate
from ml.monitoring.drift import drift_report, psi
from ml.training.train import INPUT_COLS

OK = dict(air_temp=300, process_temp=310, rpm=1500, torque=40, tool_wear=100, type="M")
COLS = INPUT_COLS


def csv_of(df):
    return df[COLS].to_csv(index=False)


# ---- BUG-1/2: non-finite / out-of-range input crashed with 500
def test_batch_rejects_inf_negative_zero_rows_instead_of_500(client, auth):
    csv = pd.DataFrame(
        [
            OK,
            {**OK, "rpm": "inf"},
            {**OK, "torque": -5},
            {**OK, "rpm": 0},
            {**OK, "air_temp": "nan"},
        ]
    ).to_csv(index=False)
    r = client.post("/predict/batch", files={"file": ("b.csv", csv)}, headers=auth[0])
    assert r.status_code == 200
    assert (r.json()["n_ok"], r.json()["failed_rows"]) == (1, [1, 2, 3, 4])


def test_nan_json_is_422_not_500(client, auth):
    body = '{"air_temp":NaN,"process_temp":310,"rpm":1500,"torque":40,"tool_wear":1,"type":"M"}'
    r = client.post(
        "/predict",
        content=body,
        headers={**auth[0], "content-type": "application/json"},
    )
    assert r.status_code == 422 and "input" not in r.text


def test_batch_edge_files(client, auth):
    hdr_only = ",".join(COLS) + "\n"
    r = client.post("/predict/batch", files={"file": ("b.csv", hdr_only)}, headers=auth[0])
    assert r.status_code == 200 and r.json()["n_ok"] == 0
    # Test CSV with whitespace in headers
    ws_csv = " air_temp , process_temp , rpm , torque , tool_wear , type \n300,310,1500,40,100,M\n"
    r_ws = client.post("/predict/batch", files={"file": ("b.csv", ws_csv)}, headers=auth[0])
    assert r_ws.status_code == 200 and r_ws.json()["n_ok"] == 1
    assert client.post("/predict/batch", files={"file": ("b.csv", "")}, headers=auth[0]).status_code == 400
    assert client.post(
        "/predict/batch",
        files={"file": ("b.bin", b"\xff\xfe\x00\x01garbage")},
        headers=auth[0],
    ).status_code in (400, 422)


# ---- SEC-1: default signing key
def test_production_refuses_missing_or_short_secret():
    env = {k: v for k, v in os.environ.items() if k not in ("APEX_SECRET_KEY", "APEX_ENV")}
    for extra in ({}, {"APEX_ENV": "production", "APEX_SECRET_KEY": "short"}):
        r = subprocess.run(
            [sys.executable, "-c", "import backend.app.core.config"],
            env={**env, "PYTHONPATH": ".", **extra},
            capture_output=True,
            text=True,
        )
        assert r.returncode != 0 and "APEX_SECRET_KEY" in r.stderr
    ok = subprocess.run(
        [sys.executable, "-c", "import backend.app.core.config"],
        env={**env, "PYTHONPATH": ".", "APEX_SECRET_KEY": "k" * 40},
        capture_output=True,
    )
    assert ok.returncode == 0


# ---- SEC-2: first user must NOT become admin; allowlist only
def test_admin_only_via_allowlist(client, admin, user2):
    me_role = client.post("/auth/register", json={"email": "third@x.com", "password": "password123"}).json()["role"]
    assert me_role == "user"
    assert (
        client.post("/auth/register", json={"email": "ADMIN@x.com", "password": "password123"}).status_code == 409
    )  # case-insensitive dup


# ---- SEC-3: logout revokes refresh token
def test_logout_revokes_refresh(client):
    client.post("/auth/register", json={"email": "lo@x.com", "password": "password123"})
    t = client.post("/auth/login", data={"username": "lo@x.com", "password": "password123"}).json()
    h = {"Authorization": f"Bearer {t['access_token']}"}
    assert client.post("/auth/refresh", json={"refresh_token": t["refresh_token"]}).status_code == 200
    assert client.post("/auth/logout", json={"refresh_token": t["refresh_token"]}, headers=h).status_code == 200
    assert client.post("/auth/refresh", json={"refresh_token": t["refresh_token"]}).status_code == 401
    assert client.post("/auth/logout", json={"refresh_token": t["access_token"]}, headers=h).status_code == 400
    assert client.post("/auth/logout", json={"refresh_token": "malformed.token.value"}, headers=h).status_code == 400
    assert client.post("/auth/logout", json={"refresh_token": t["refresh_token"]}).status_code == 401  # needs auth


# ---- RBAC
@pytest.mark.parametrize(
    ("method", "url"),
    [
        ("get", "/admin/models"),
        ("post", "/admin/models/rollback"),
        ("post", "/admin/models/x/promote"),
    ],
)
def test_admin_endpoints_forbidden_for_users_and_anon(client, user2, method, url):
    kw = {"json": {}} if method == "post" else {}
    assert getattr(client, method)(url, headers=user2[0], **kw).status_code == 403
    assert getattr(client, method)(url, **kw).status_code == 401


def test_model_unavailable_returns_503(client, auth, monkeypatch):
    import backend.app.services.model_service as ms

    pipe, meta, anom = svc.pipe, svc.meta, svc.anom
    svc.pipe = None
    monkeypatch.setattr(ms, "load_model", lambda d: (_ for _ in ()).throw(FileNotFoundError("gone")))
    try:
        r = client.post("/predict", json=OK, headers=auth[0])
        assert r.status_code == 503 and "Model unavailable" in r.json()["detail"]
    finally:
        svc.pipe, svc.meta, svc.anom = pipe, meta, anom


# ---- anomaly endpoint
def test_anomaly_flags_extreme_not_normal(client, auth):
    normal = client.post("/anomaly", json=OK, headers=auth[0]).json()
    weird = client.post(
        "/anomaly",
        json={**OK, "torque": 480, "rpm": 200, "air_temp": 395},
        headers=auth[0],
    ).json()
    assert weird["anomaly_score"] > normal["anomaly_score"] and weird["is_anomaly"] and not normal["is_anomaly"]
    assert "NOT a confirmed failure" in weird["note"]


# ---- leakage guard: failure-mode label columns can never be model inputs
def test_failure_mode_columns_not_model_inputs(trained):
    assert not {"TWF", "HDF", "PWF", "OSF", "RNF", "machine_failure", "UDI"} & set(trained["input_columns"])


# ---- drift (unit)
def test_psi_identical_near_zero_and_shift_large():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 5000)
    assert psi(a, rng.normal(0, 1, 5000)) < 0.05
    assert psi(a, rng.normal(2, 1, 5000)) > 0.25


def test_drift_report_unit():
    ref = simulate(3000, 1)
    same = drift_report(ref, simulate(3000, 2), ["rpm", "torque"], ["type"])
    shifted = drift_report(
        ref,
        simulate(3000, 2).assign(rpm=lambda d: d.rpm + 500, torque=lambda d: d.torque * 1.6),
        ["rpm", "torque"],
        ["type"],
    )
    assert same["overall"] == "ok" and not same["retrain_recommended"]
    assert shifted["overall"] == "drift" and shifted["retrain_recommended"]
    with pytest.raises(ValueError, match="at least"):
        drift_report(ref, ref.head(10), ["rpm"], ["type"])
    unseen = drift_report(ref, ref.head(500).assign(type="Z"), ["rpm"], ["type"])
    assert unseen["features"]["type"]["unseen_categories"] == ["Z"] and unseen["features"]["type"]["status"] == "drift"


# ---- drift (API)
def test_drift_api(client, auth):
    same = client.post(
        "/monitoring/drift",
        files={"file": ("d.csv", csv_of(simulate(1500, 99)))},
        headers=auth[0],
    ).json()
    assert same["overall"] in ("ok", "warn") and not same["retrain_recommended"]
    bad = simulate(1500, 99).assign(rpm=lambda d: d.rpm + 600, torque=lambda d: d.torque * 1.7)
    r = client.post("/monitoring/drift", files={"file": ("d.csv", csv_of(bad))}, headers=auth[0]).json()
    assert r["overall"] == "drift" and r["retrain_recommended"]
    assert client.post("/monitoring/drift", files={"file": ("d.csv", "a,b\n1,2")}, headers=auth[0]).status_code == 422
    assert (
        client.post(
            "/monitoring/drift",
            files={"file": ("d.csv", csv_of(simulate(1500, 1).head(20)))},
            headers=auth[0],
        ).status_code
        == 422
    )
    assert len(client.get("/monitoring/drift/history", headers=auth[0]).json()) >= 2


# ---- registry lifecycle: retrain -> candidate (not active) -> policy -> promote -> rollback
def test_registry_lifecycle(client, admin, auth):
    h = admin[0]
    models = client.get("/admin/models", headers=h).json()
    active0 = next(m for m in models if m["status"] == "active")
    # retrain from raw-ish upload (internal schema + target)
    data = simulate(5000, 7)
    r = client.post(
        "/admin/retrain",
        files={"file": ("train.csv", data[[*COLS, "machine_failure"]].to_csv(index=False))},
        headers=h,
    )
    assert r.status_code == 200, r.text
    cand = r.json()
    assert cand["status"] == "candidate"
    assert client.get("/model/info", headers=h).json()["version"] == active0["version"]  # NOT silently swapped
    # a policy-failing candidate is refused unless forced with a reason
    from backend.app.db import ModelVersion, SessionLocal

    with SessionLocal() as db:
        row = db.get(ModelVersion, cand["version"])
        row.metrics = {**row.metrics, "recall": 0.1}
        db.commit()
    p = client.post(f"/admin/models/{cand['version']}/promote", json={}, headers=h)
    assert p.status_code == 422 and "recall" in str(p.json())
    assert client.post(f"/admin/models/{cand['version']}/promote", json={"force": True}, headers=h).status_code == 422  # force needs reason
    with SessionLocal() as db:
        row = db.get(ModelVersion, cand["version"])
        row.metrics = {**row.metrics, "recall": 0.8}
        db.commit()
    ok = client.post(f"/admin/models/{cand['version']}/promote", json={}, headers=h)
    assert ok.status_code == 200 and ok.json()["status"] == "active"
    assert client.get("/model/info", headers=h).json()["version"] == cand["version"]
    assert client.post("/predict", json=OK, headers=auth[0]).json()["model_version"] == cand["version"]
    assert client.post(f"/admin/models/{cand['version']}/promote", json={}, headers=h).status_code == 409  # not a candidate anymore
    rb = client.post("/admin/models/rollback", headers=h)
    assert rb.status_code == 200 and rb.json()["version"] == active0["version"]
    assert client.get("/model/info", headers=h).json()["version"] == active0["version"]
    statuses = {m["version"]: m["status"] for m in client.get("/admin/models", headers=h).json()}
    assert statuses[cand["version"]] == "rolled_back" and statuses[active0["version"]] == "active"


def test_retrain_negative_cases(client, admin):
    h = admin[0]
    assert client.post("/admin/retrain", files={"file": ("t.csv", "a,b\n1,2")}, headers=h).status_code == 422
    tiny = simulate(100, 1)[[*COLS, "machine_failure"]].to_csv(index=False)
    assert client.post("/admin/retrain", files={"file": ("t.csv", tiny)}, headers=h).status_code == 422  # insufficient rows
    nofail = simulate(1000, 1)[[*COLS, "machine_failure"]].assign(machine_failure=0).to_csv(index=False)
    assert client.post("/admin/retrain", files={"file": ("t.csv", nofail)}, headers=h).status_code == 422
    assert client.post("/admin/retrain", files={"file": ("t.csv", "")}, headers=h).status_code == 400


def test_duplicate_training_request_rejected(client, admin):
    assert _train_lock.acquire(blocking=False)
    try:
        ok = simulate(1000, 1)[[*COLS, "machine_failure"]].to_csv(index=False)
        assert client.post("/admin/retrain", files={"file": ("t.csv", ok)}, headers=admin[0]).status_code == 409
    finally:
        _train_lock.release()


def test_tampered_candidate_cannot_be_promoted(client, admin):
    from backend.app.db import ModelVersion, SessionLocal

    r = client.post(
        "/admin/retrain",
        files={
            "file": (
                "t.csv",
                simulate(4000, 11)[[*COLS, "machine_failure"]].to_csv(index=False),
            )
        },
        headers=admin[0],
    ).json()
    with SessionLocal() as db:
        path = db.get(ModelVersion, r["version"]).path
    with open(f"{path}/model.joblib", "ab") as f:
        f.write(b"tamper")
    p = client.post(
        f"/admin/models/{r['version']}/promote",
        json={"force": True, "reason": "test"},
        headers=admin[0],
    )
    assert p.status_code == 422 and "integrity" in str(p.json())


def test_cli_register_and_promote(trained, tmp_path):
    import json
    import shutil

    new_dir = tmp_path / "cli_cand"
    shutil.copytree(Path(os.environ["APEX_MODEL_DIR"]), new_dir)
    m = json.loads((new_dir / "manifest.json").read_text(encoding="utf-8"))
    m["version"] = "99999999999999"
    (new_dir / "manifest.json").write_text(json.dumps(m), encoding="utf-8")
    r = subprocess.run(
        [sys.executable, "-m", "backend.app.cli", "register", str(new_dir), "--promote"],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )
    assert r.returncode == 0
    assert "active" in r.stdout
