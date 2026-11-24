import os
import tempfile
from pathlib import Path

_tmp = tempfile.mkdtemp()
os.environ["APEX_ENV"] = "dev"
os.environ["APEX_ADMIN_EMAILS"] = "admin@x.com"
os.environ["APEX_REGISTRY_DIR"] = f"{_tmp}/registry"
os.environ["DATABASE_URL"] = f"sqlite:///{_tmp}/t.db"
os.environ["APEX_MODEL_DIR"] = f"{_tmp}/model"
os.environ["APEX_RATE_LIMIT"] = "100000"
import pytest  # noqa: E402

from ml.data.ai4i import simulate  # noqa: E402
from ml.training.train import run  # noqa: E402


@pytest.fixture(scope="session")
def trained():
    return run(simulate(6000, 1), Path(_tmp) / "model", "SIMULATED-test")


@pytest.fixture(scope="session")
def client(trained):
    from fastapi.testclient import TestClient

    from backend.app.main import app

    return TestClient(app)


@pytest.fixture(scope="session")
def auth(client):
    client.post("/auth/register", json={"email": "a@x.com", "password": "password123"})
    t = client.post("/auth/login", data={"username": "a@x.com", "password": "password123"}).json()
    return {"Authorization": f"Bearer {t['access_token']}"}, t


def _hdr(client, email):
    client.post("/auth/register", json={"email": email, "password": "password123"})
    t = client.post("/auth/login", data={"username": email, "password": "password123"}).json()
    return {"Authorization": f"Bearer {t['access_token']}"}, t


@pytest.fixture(scope="session")
def admin(client):
    return _hdr(client, "admin@x.com")


@pytest.fixture(scope="session")
def user2(client):
    return _hdr(client, "plain@x.com")
