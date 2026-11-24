"""Usage: python -m backend.app.cli register <model_dir> [--promote]"""

import argparse
from pathlib import Path

from backend.app.db import SessionLocal, init_db
from backend.app.services import registry

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["register"])
    ap.add_argument("model_dir")
    ap.add_argument("--promote", action="store_true")
    a = ap.parse_args()
    init_db()
    with SessionLocal() as db:
        row = registry.register_candidate(db, Path(a.model_dir))
        if a.promote:
            row = registry.promote(db, row.version)
        print(row.version, row.status)
