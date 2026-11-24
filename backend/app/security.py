import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.app.core import config
from backend.app.db import RevokedToken, User, get_db

oauth2 = OAuth2PasswordBearer(tokenUrl="/auth/login")
_DUMMY = bcrypt.hashpw(b"dummy", bcrypt.gensalt()).decode()


def hash_pw(p: str) -> str:
    return bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode()


def verify_pw(p: str, h: str | None) -> bool:
    """Constant-ish time: unknown users still pay for one bcrypt check (no user enumeration by timing)."""
    ok = bcrypt.checkpw(p.encode(), (h or _DUMMY).encode())
    return ok and h is not None


def make_token(sub: int, kind: str) -> str:
    ttl = timedelta(minutes=config.ACCESS_MIN) if kind == "access" else timedelta(days=config.REFRESH_DAYS)
    exp = datetime.now(timezone.utc) + ttl
    return jwt.encode(
        {"sub": str(sub), "typ": kind, "jti": str(uuid.uuid4()), "exp": exp},
        config.SECRET_KEY,
        "HS256",
    )


def decode(tok: str, kind: str, db: Session | None = None) -> int:
    try:
        d = jwt.decode(tok, config.SECRET_KEY, ["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired token") from None
    if d.get("typ") != kind:
        raise HTTPException(401, "Wrong token type")
    if db is not None:
        revoked = db.get(RevokedToken, str(d.get("jti")))
        if revoked is not None:
            raise HTTPException(401, "Token has been revoked")
    return int(d["sub"])


def revoke(db: Session, tok: str) -> None:
    try:
        d = jwt.decode(tok, config.SECRET_KEY, ["HS256"])
    except jwt.PyJWTError:
        raise HTTPException(400, "Invalid or expired token") from None
    if d.get("typ") != "refresh":
        raise HTTPException(400, "Only refresh tokens can be revoked")
    if not db.get(RevokedToken, d["jti"]):
        db.add(RevokedToken(jti=d["jti"], expires_at=datetime.fromtimestamp(d["exp"], timezone.utc)))
        db.commit()


def current_user(tok: str = Depends(oauth2), db: Session = Depends(get_db)) -> User:
    u = db.get(User, decode(tok, "access"))
    if not u:
        raise HTTPException(401, "User not found")
    return u


def current_admin(u: User = Depends(current_user)) -> User:
    if u.role != "admin":
        raise HTTPException(403, "Admin role required")
    return u
