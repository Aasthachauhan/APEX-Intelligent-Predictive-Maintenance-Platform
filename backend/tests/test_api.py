from pathlib import Path

import pandas as pd
import pytest

OK = dict(air_temp=300, process_temp=310, rpm=1500, torque=40, tool_wear=100, type="M")
RISKY = dict(air_temp=300, process_temp=308, rpm=1300, torque=65, tool_wear=230, type="L")


def test_unauthorized(client):
    assert client.post("/predict", json=OK).status_code == 401
    assert client.get("/model/info").status_code == 401


def test_register_validation_and_duplicate(client):
    assert client.post("/auth/register", json={"email": "bad", "password": "password123"}).status_code == 422
    assert client.post("/auth/register", json={"email": "a@x.com", "password": "password123"}).status_code in (201, 409)
    assert client.post("/auth/register", json={"email": "a@x.com", "password": "password123"}).status_code == 409


def test_bad_login(client):
    assert client.post("/auth/login", data={"username": "a@x.com", "password": "nope-nope"}).status_code == 401


def test_refresh_and_token_type(client, auth):
    h, t = auth
    assert client.post("/auth/refresh", json={"refresh_token": t["refresh_token"]}).status_code == 200
    assert client.post("/auth/refresh", json={"refresh_token": t["access_token"]}).status_code == 401
    assert client.get("/model/info", headers={"Authorization": f"Bearer {t['refresh_token']}"}).status_code == 401


def test_predict_is_model_output_and_risky_scores_higher(client, auth):
    h, _ = auth
    a = client.post("/predict", json=OK, headers=h).json()
    b = client.post("/predict", json=RISKY, headers=h).json()
    assert 0 <= a["failure_probability"] <= 1 and a["risk"] in {"low", "medium", "high"}
    assert b["failure_probability"] > a["failure_probability"]
    assert "not a confirmed failure" in a["note"] and a["prediction_id"] != b["prediction_id"]


@pytest.mark.parametrize(
    "bad",
    [
        {**OK, "rpm": -5},
        {**OK, "type": "Z"},
        {**OK, "torque": "abc"},
        {k: v for k, v in OK.items() if k != "torque"},
    ],
)
def test_invalid_inputs(client, auth, bad):
    assert client.post("/predict", json=bad, headers=auth[0]).status_code == 422


def test_matches_offline_model(client, auth, trained):
    from ml.training.train import load_model

    pipe, m = load_model(Path(__import__("os").environ["APEX_MODEL_DIR"]))
    direct = pipe.predict_proba(pd.DataFrame([RISKY]))[0, 1]
    assert client.post("/predict", json=RISKY, headers=auth[0]).json()["failure_probability"] == pytest.approx(direct)


def test_explain(client, auth):
    r = client.post("/explain", json=RISKY, headers=auth[0]).json()["top_contributions"]
    assert len(r) == 5 and all(abs(x["shap_value"]) > 0 for x in r)


def test_batch_valid_and_failed_rows(client, auth):
    csv = pd.DataFrame([OK, RISKY, {**OK, "type": "Q"}]).to_csv(index=False)
    r = client.post("/predict/batch", files={"file": ("b.csv", csv)}, headers=auth[0]).json()
    assert (r["n_ok"], r["n_failed"], r["failed_rows"]) == (2, 1, [2])
    assert "failure_probability" in r["csv"].splitlines()[0]
    assert client.post("/predict/batch", files={"file": ("b.csv", "a,b\n1,2")}, headers=auth[0]).status_code == 422


def test_history_scoped_to_user(client, auth):
    client.post("/auth/register", json={"email": "b@x.com", "password": "password123"})
    t = client.post("/auth/login", data={"username": "b@x.com", "password": "password123"}).json()
    other = client.get("/predictions", headers={"Authorization": f"Bearer {t['access_token']}"}).json()
    mine = client.get("/predictions", headers=auth[0]).json()
    assert other == [] and len(mine) > 0


def test_tampered_artifact_rejected(trained, tmp_path):
    import shutil

    from ml.training.train import load_model

    d = tmp_path / "m"
    shutil.copytree(Path(__import__("os").environ["APEX_MODEL_DIR"]), d)
    with open(d / "model.joblib", "ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="integrity"):
        load_model(d)
