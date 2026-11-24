"""Model registry: candidate -> active -> archived, with an explicit promotion policy and rollback.
No model becomes active implicitly after training; promotion is an admin action that must pass the policy
(or be force-overridden with a recorded reason)."""

import json
from pathlib import Path

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.core import config
from backend.app.db import ModelVersion, now
from ml.training.train import load_model


def active(db: Session) -> ModelVersion | None:
    return db.scalar(select(ModelVersion).where(ModelVersion.status == "active"))


def _row(m: dict, d: Path, status: str) -> ModelVersion:
    return ModelVersion(
        version=m["version"],
        status=status,
        path=str(d),
        sha256=m["sha256"],
        data_source=m["data_source"],
        metrics=m["selected_test_metrics"] | {"model_name": m["model_name"]},
    )


def register_candidate(db: Session, model_dir: Path) -> ModelVersion:
    m = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    load_model(model_dir)  # integrity check before it enters the registry
    if db.get(ModelVersion, m["version"]):
        raise HTTPException(409, f"Version {m['version']} already registered")
    row = _row(m, model_dir, "candidate")
    db.add(row)
    db.commit()
    return row


def bootstrap(db: Session, model_dir: Path) -> ModelVersion:
    """Dev convenience: an existing trained dir becomes the first active model (audited by caller)."""
    m = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    row = db.get(ModelVersion, m["version"]) or _row(m, model_dir, "active")
    row.status, row.promoted_at = "active", now()
    db.merge(row)
    db.commit()
    return row


def policy_failures(cand: ModelVersion, act: ModelVersion | None) -> list[str]:
    out = []
    if cand.metrics["recall"] < config.MIN_PROMOTION_RECALL:
        out.append(f"candidate recall {cand.metrics['recall']:.3f} < required {config.MIN_PROMOTION_RECALL}")
    if act and cand.metrics["pr_auc"] < act.metrics["pr_auc"] - config.MAX_PRAUC_REGRESSION:
        out.append(
            f"candidate PR-AUC {cand.metrics['pr_auc']:.3f} regresses vs active {act.metrics['pr_auc']:.3f} by more than {config.MAX_PRAUC_REGRESSION}"
        )
    return out


def promote(db: Session, version: str, force: bool = False, reason: str = "") -> ModelVersion:
    cand = db.get(ModelVersion, version)
    if not cand:
        raise HTTPException(404, "Unknown model version")
    if cand.status != "candidate":
        raise HTTPException(
            409,
            f"Only candidates can be promoted (status={cand.status}); use rollback for archived versions",
        )
    cur = active(db)
    fails = policy_failures(cand, cur)
    if fails and not (force and reason.strip()):
        raise HTTPException(
            422,
            {
                "message": "Promotion policy failed (force requires a reason)",
                "failures": fails,
            },
        )
    try:
        load_model(Path(cand.path))
    except Exception as e:
        raise HTTPException(422, f"Artifact failed integrity check: {e}") from None
    if cur:
        cur.status = "archived"
    cand.status, cand.promoted_at = "active", now()
    db.commit()
    return cand


def rollback(db: Session) -> ModelVersion:
    cur = active(db)
    target = db.scalar(select(ModelVersion).where(ModelVersion.status == "archived").order_by(ModelVersion.promoted_at.desc()))
    if not cur or not target:
        raise HTTPException(409, "No previous version to roll back to")
    try:
        load_model(Path(target.path))
    except Exception as e:
        raise HTTPException(422, f"Rollback target failed integrity check: {e}") from None
    cur.status = "rolled_back"
    target.status, target.promoted_at = "active", now()
    db.commit()
    return target
