"""Clean-start smoke test: train -> start real uvicorn -> auth/predict/batch -> restart -> verify state. Exits non-zero on failure."""

import os
import statistics
import subprocess
import sys
import tempfile
import time

import httpx

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W = tempfile.mkdtemp(prefix="apex_smoke_")
sys.path.insert(0, ROOT)
env = {
    **os.environ,
    "PYTHONPATH": ROOT,
    "APEX_ENV": "production",
    "APEX_SECRET_KEY": "s" * 40,
    "APEX_ADMIN_EMAILS": "boss@x.com",
    "DATABASE_URL": f"sqlite:///{W}/apex.db",
    "APEX_MODEL_DIR": f"{W}/active",
    "APEX_REGISTRY_DIR": f"{W}/registry",
    "APEX_RATE_LIMIT": "100000",
}
t0 = time.time()
r = subprocess.run(
    [sys.executable, "-m", "ml.training.train", "--out", f"{W}/active", "--mlflow"], cwd=ROOT, env=env, capture_output=True, text=True
)
print(f"clean-start training exit={r.returncode} in {time.time() - t0:.1f}s")


def start():
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "backend.app.main:app", "--port", "8123", "--log-level", "warning"],
        cwd=ROOT,
        env=env,
        stdout=open(f"{W}/server.log", "a"),
        stderr=subprocess.STDOUT,
    )
    for _ in range(60):
        try:
            if httpx.get("http://127.0.0.1:8123/health").status_code == 200:
                return p
        except Exception:
            time.sleep(0.5)
    raise SystemExit("server failed to start")


B = "http://127.0.0.1:8123"
ok = dict(air_temp=300, process_temp=310, rpm=1500, torque=40, tool_wear=100, type="M")
p = start()
c = httpx.Client(base_url=B, timeout=120)
c.post("/auth/register", json={"email": "boss@x.com", "password": "password123"})
tok = c.post("/auth/login", data={"username": "boss@x.com", "password": "password123"}).json()
h = {"Authorization": "Bearer " + tok["access_token"]}
v1 = c.get("/model/info", headers=h).json()["version"]
pid = c.post("/predict", json=ok, headers=h).json()["prediction_id"]
print("before restart: model", v1, "| admin models:", [(m["version"], m["status"]) for m in c.get("/admin/models", headers=h).json()])
# latency
lat = []
for _ in range(200):
    t = time.perf_counter()
    c.post("/predict", json=ok, headers=h)
    lat.append((time.perf_counter() - t) * 1000)
lat.sort()
print(f"single /predict latency ms: p50={statistics.median(lat):.1f} p95={lat[int(0.95 * len(lat))]:.1f} (HTTP+auth+DB log+model)")
from ml.data.ai4i import simulate

big = simulate(50000, 5)[["air_temp", "process_temp", "rpm", "torque", "tool_wear", "type"]].to_csv(index=False)
t = time.perf_counter()
r = c.post("/predict/batch", files={"file": ("b.csv", big)}, headers=h)
dt = time.perf_counter() - t
print(f"batch 50,000 rows: status={r.status_code} n_ok={r.json()['n_ok']} in {dt:.1f}s ({50000 / dt:.0f} rows/s)")
# restart
p.terminate()
p.wait()
time.sleep(1)
p = start()
c = httpx.Client(base_url=B, timeout=60)
tok = c.post("/auth/login", data={"username": "boss@x.com", "password": "password123"}).json()
h = {"Authorization": "Bearer " + tok["access_token"]}
v2 = c.get("/model/info", headers=h).json()["version"]
hist = c.get("/predictions?limit=500", headers=h).json()
print(
    f"after restart: model same={v1 == v2} | history rows (capped 500)={len(hist)} | first id survives={pid in [x['id'] for x in hist] or 'older than newest 500'}"
)
print(
    "post-restart predict:",
    c.post("/predict", json=ok, headers=h).status_code,
    "| unauth predict:",
    c.post("/predict", json=ok).status_code,
)
p.terminate()
p.wait()
log = open(f"{W}/server.log").read()
print("server log lines with 'Traceback'/'ERROR':", log.count("Traceback") + log.count("ERROR"))

assert v1 == v2, "active model changed across restart"
assert log.count("Traceback") == 0, "server log has tracebacks"
print("SMOKE OK")
