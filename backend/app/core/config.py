import os
from pathlib import Path

ENV = os.getenv("APEX_ENV", "production")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./apex.db")  # Postgres: postgresql+psycopg://...
_key = os.getenv("APEX_SECRET_KEY", "")
if not _key:
    if ENV != "dev":
        raise RuntimeError("APEX_SECRET_KEY must be set (>=32 chars). Use APEX_ENV=dev for local development only.")
    _key = "dev-only-change-me-" + "x" * 16
if ENV != "dev" and len(_key) < 32:
    raise RuntimeError("APEX_SECRET_KEY must be at least 32 characters")
SECRET_KEY = _key
MODEL_DIR = Path(os.getenv("APEX_MODEL_DIR", "artifacts/active"))  # bootstrap source only
REGISTRY_DIR = Path(os.getenv("APEX_REGISTRY_DIR", "artifacts/registry"))
ADMIN_EMAILS = {e.strip().lower() for e in os.getenv("APEX_ADMIN_EMAILS", "").split(",") if e.strip()}
CORS_ORIGINS = [o.strip() for o in os.getenv("APEX_CORS_ORIGINS", "").split(",") if o.strip()]
ACCESS_MIN, REFRESH_DAYS = 15, 7
RATE_LIMIT_PER_MIN = int(os.getenv("APEX_RATE_LIMIT", "60"))
MAX_BATCH_ROWS = 50_000
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_TRAIN_ROWS = 200_000
MIN_PROMOTION_RECALL = 0.5  # promotion policy: candidate test recall floor
MAX_PRAUC_REGRESSION = 0.02  # promotion policy: allowed PR-AUC drop vs active
